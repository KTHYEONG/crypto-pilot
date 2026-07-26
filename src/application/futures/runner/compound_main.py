from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.application.futures.runner.compound_config import (
    CompoundRunArtifacts,
    CompoundRunConfig,
)
from src.application.futures.runner.compound_data import (
    build_multiscale_market_cube,
)
from src.application.futures.runner.compound_universe import (
    EmptyPITUniverseError,
    build_daily_pit_universe,
)
from src.application.futures.runner.data_lake_runtime import (
    build_data_lake_runtime,
    finalize_quarterly_signal_data,
    prepare_quarterly_bootstrap,
)
from src.application.futures.runner.models import RunnerResult
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    L2GateVerdict,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.engine import run_multiscale_compound_engine
from src.domain.futures.compound.holdout_store import SealedHoldoutStore
from src.domain.futures.data_lake.coverage_policy import (
    DataCoverageError,
    exclude_symbols_with_funding_gaps,
)
from src.domain.futures.data_lake.ingestion import StorageBudgetError
from src.domain.futures.data_lake.run_windows import (
    QuarterlyWindowConfig,
    resolve_completed_quarter_window,
)

_logger = logging.getLogger(__name__)

_holdout_store_instance: SealedHoldoutStore | None = None


def _get_holdout_store(config: CompoundRunConfig) -> SealedHoldoutStore:
    global _holdout_store_instance
    if _holdout_store_instance is None:
        holdout_path = Path("data/futures/lake/holdout_store.db")
        _holdout_store_instance = SealedHoldoutStore(holdout_path)
    return _holdout_store_instance


def _build_artifact_paths(config: CompoundRunConfig) -> CompoundRunArtifacts:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = Path(f"logs/futures/compound/{ts}")
    base.mkdir(parents=True, exist_ok=True)
    return CompoundRunArtifacts(
        result_path=str(base / "result.json"),
        target_weights_path=str(base / "target_weights.npy"),
        manifest_path=str(base / "manifest.json"),
    )


def _write_artifacts(
    paths: CompoundRunArtifacts,
    engine_result: CompoundEngineResult,
    config: CompoundRunConfig,
    market: MarketFeatureCube | None = None,
) -> None:
    result_data = {
        "reference_date": config.reference_date,
        "seed": config.seed,
        "base_timeframe": config.base_timeframe,
        "n_bars": int(engine_result.ledger.timestamps_ns.size),
        "n_symbols": int(engine_result.ledger.target_weights_2d.shape[1]) if engine_result.ledger.target_weights_2d.ndim > 1 else 0,
        "universe_symbols": list(market.symbols) if market is not None and hasattr(market, 'symbols') else [],
        "l2": {
            "verdict": engine_result.l2.verdict.value,
            "annualized_log_growth": engine_result.l2.annualized_log_growth,
            "cagr": engine_result.l2.cagr,
            "absolute_cagr": engine_result.l2.absolute_cagr,
            "excess_growth_lcb90": engine_result.l2.excess_growth_lcb90,
            "excess_growth_probability": engine_result.l2.excess_growth_probability,
            "stressed_excess_growth_lcb90": engine_result.l2.stressed_excess_growth_lcb90,
            "equity_multiple": engine_result.l2.equity_multiple,
            "sharpe": engine_result.l2.sharpe,
            "sharpe_probability": engine_result.l2.sharpe_probability,
            "deflated_sharpe_probability": engine_result.l2.deflated_sharpe_probability,
            "max_drawdown": engine_result.l2.max_drawdown,
            "daily_cvar95": engine_result.l2.daily_cvar95,
            "annual_volatility": engine_result.l2.annual_volatility,
            "annual_turnover": engine_result.l2.annual_turnover,
            "cost_drag_ratio": engine_result.l2.cost_drag_ratio,
            "capacity_utilisation_p95": engine_result.l2.capacity_utilisation_p95,
            "integrity_ok": engine_result.l2.integrity_ok,
            "reasons": list(engine_result.l2.reasons),
        },
        "l3": {
            "verdict": engine_result.l3.verdict.value,
            "posterior_growth_probability": engine_result.l3.posterior_growth_probability,
            "holdout_days": engine_result.l3.holdout_days,
            "max_drawdown": engine_result.l3.max_drawdown,
            "daily_cvar95": engine_result.l3.daily_cvar95,
            "reasons": list(engine_result.l3.reasons),
        },
        "dry_run": os.environ.get("L2_DRY_RUN", "0") == "1",
    }
    with open(paths.result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    np.save(paths.target_weights_path, engine_result.ledger.target_weights_2d)

    manifest = {
        "model_version": engine_result.handoff.model_version,
        "data_manifest_hash": engine_result.handoff.data_manifest_hash,
        "n_timestamps": int(engine_result.ledger.timestamps_ns.size),
        "integrity_ok": engine_result.ledger.integrity_ok,
        "l3_verdict": engine_result.l3.verdict.value,
    }
    with open(paths.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _logger.info("artifacts written: result=%s weights=%s manifest=%s",
                 paths.result_path, paths.target_weights_path, paths.manifest_path)


def run_multiscale_compound_main(config: CompoundRunConfig) -> RunnerResult:
    try:
        _logger.info("multiscale compound run: date=%s sync=%s",
                     config.reference_date, config.sync)

        engine_config = CompoundEngineConfig()
        runtime = build_data_lake_runtime(config)

        ref_date_str = config.reference_date or datetime.now(UTC).strftime("%Y-%m-%d")
        requested_date = datetime.strptime(ref_date_str, "%Y-%m-%d").date()

        window = resolve_completed_quarter_window(
            requested_date,
            QuarterlyWindowConfig(
                warmup_days=config.window.warmup_days,
                l1_days=config.window.l1_days,
                l2_days=config.window.l2_days,
                l3_days=config.window.l3_days,
            ),
        )

        bootstrap = prepare_quarterly_bootstrap(
            config=config, runtime=runtime, window=window,
        )

        snapshot = bootstrap.snapshot
        ref_dt = pd.Timestamp(ref_date_str, tz="UTC")
        ns_per_hour = 3_600_000_000_000
        n_bars = config.history_days * 24
        start_dt = ref_dt - pd.Timedelta(days=config.history_days)
        execution_calendar = pd.date_range(start=start_dt, periods=n_bars, freq="h", tz="UTC")

        universe = build_daily_pit_universe(
            snapshot=snapshot, execution_calendar=execution_calendar, config=config,
        )

        _logger.info(
            "universe built: %d symbols from snapshot %s",
            len(universe.symbols), snapshot.snapshot_id,
        )

        from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog
        alpha_catalog = build_multiscale_alpha_catalog()

        filtered_cube, excluded_symbols = exclude_symbols_with_funding_gaps(
            snapshot=snapshot,
            universe=universe.state_cube,
            start_time_ns=window.l1_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            max_gap_ns=86_400_000_000_000,
        )
        if excluded_symbols:
            _logger.warning(
                "excluded %d symbols with funding gaps: %s",
                len(excluded_symbols),
                ", ".join(excluded_symbols),
            )
            universe = replace(universe, state_cube=filtered_cube)

        prepared = finalize_quarterly_signal_data(
            config=config, runtime=runtime, bootstrap=bootstrap,
            universe=universe, catalog=alpha_catalog,
        )

        market = build_multiscale_market_cube(
            snapshot=snapshot, universe=universe, config=config,
            field_plan=prepared.field_plan,
        )

        holdout_store = _get_holdout_store(config)
        holdout_id = (
            f"quarterly-{window.cutoff_date.isoformat()}-"
            f"{snapshot.manifest_hash[:12]}"
        )

        holdout_manifest = SealedHoldoutManifest(
            holdout_id=holdout_id,
            start_time_ns=window.l3_start_ns,
            end_time_ns=window.cutoff_exclusive_ns,
            holdout_days=(window.cutoff_exclusive_ns - window.l3_start_ns) // (24 * ns_per_hour),
            model_version="quarterly-v1",
            data_manifest_hash=market.data_manifest_hash,
            universe_state_hash=snapshot.universe_state_hash,
        )

        try:
            holdout_store.create(holdout_manifest)
        except Exception:
            _logger.info("holdout %s already exists, reusing", holdout_id)

        _logger.info("running multiscale compound engine with quarterly window")
        engine_result = run_multiscale_compound_engine(
            market=market,
            universe=universe.state_cube,
            window=window,
            recipe_plan=prepared.recipe_plan,
            holdout_store=holdout_store,
            holdout_id=holdout_id,
            config=engine_config,
        )

        paths = _build_artifact_paths(config)
        _write_artifacts(paths, engine_result, config, market=market)

        if engine_result.l2.verdict != L2GateVerdict.PASS:
            _logger.info(
                "L2 verdict=%s: L3 holdout unconsumed",
                engine_result.l2.verdict.value,
            )

        if not engine_result.l2.integrity_ok:
            return RunnerResult(exit_code=1, reason="integrity_failure")

        if engine_result.l3.verdict.value in ("reject",):
            return RunnerResult(exit_code=0, reason=f"l3_{engine_result.l3.verdict.value}")

        return RunnerResult(exit_code=0, reason=f"ok:l3_{engine_result.l3.verdict.value}")

    except EmptyPITUniverseError as exc:
        _logger.error("no deployable universe: %s", exc)
        return RunnerResult(exit_code=1, reason=f"empty_universe:{exc}")
    except DataCoverageError as exc:
        _logger.error("data coverage error: %s", exc)
        return RunnerResult(exit_code=1, reason=f"data_coverage:{exc}")
    except StorageBudgetError as exc:
        _logger.error("storage budget exceeded: %s", exc)
        return RunnerResult(exit_code=1, reason=f"storage_budget:{exc}")
    except Exception as exc:
        _logger.exception("multiscale compound run failed")
        return RunnerResult(exit_code=1, reason=str(exc))
