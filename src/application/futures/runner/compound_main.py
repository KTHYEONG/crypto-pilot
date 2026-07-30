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
    restrict_pit_universe_to_symbols,
)
from src.application.futures.runner.data_lake_runtime import (
    build_data_lake_runtime,
    finalize_quarterly_signal_data,
    prepare_quarterly_bootstrap,
)
from src.application.futures.runner.models import RunnerResult
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CandidateTrialLedger,
    CompoundEngineResult,
    DeploymentCandidate,
    DeploymentVerdict,
    L2Evaluation,
    L2GateVerdict,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.deployment import publish_promoted_strategy
from src.domain.futures.compound.engine import run_multiscale_compound_engine
from src.domain.futures.compound.holdout_store import SealedHoldoutStore
from src.domain.futures.compound.provenance import compute_strategy_spec_hash
from src.domain.futures.data_lake.coverage_policy import (
    DataCoverageError,
    exclude_symbols_with_funding_gaps,
)
from src.domain.futures.data_lake.ingestion import (
    StorageBudgetError,
    build_ingestion_plan,
    restrict_to_complete_core_symbols,
)
from src.domain.futures.data_lake.run_windows import (
    QuarterlyWindowConfig,
    build_quarterly_execution_calendar,
    resolve_completed_quarter_window,
)

_logger = logging.getLogger(__name__)

_holdout_store_instance: SealedHoldoutStore | None = None
_trial_ledger_instance: CandidateTrialLedger | None = None


def build_compound_engine_config(
    run_config: CompoundRunConfig,
) -> CompoundEngineConfig:
    nav = run_config.portfolio_nav_usdt
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError(f"portfolio_nav_usdt must be finite positive, got {nav}")
    engine_config = CompoundEngineConfig()
    engine_config = replace(
        engine_config,
        allocator=replace(engine_config.allocator, portfolio_nav_usdt=nav),
        dense_sim=replace(engine_config.dense_sim, nav_usdt=nav),
    )
    return engine_config


def _get_holdout_store(config: CompoundRunConfig) -> SealedHoldoutStore:
    global _holdout_store_instance
    if _holdout_store_instance is None:
        holdout_path = Path("data/futures/lake/holdout_store.db")
        _holdout_store_instance = SealedHoldoutStore(holdout_path)
    return _holdout_store_instance


def _get_trial_ledger(config: CompoundRunConfig) -> CandidateTrialLedger:
    global _trial_ledger_instance
    if _trial_ledger_instance is None:
        ledger_path = Path("data/futures/lake/candidate_trials.db")
        _trial_ledger_instance = CandidateTrialLedger(ledger_path)
    return _trial_ledger_instance


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
            "max_name_weight_p95": engine_result.l2.max_name_weight_p95,
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


def write_l2_gate_inputs(
    run_dir: Path, evaluation: L2Evaluation,
    l1_prior_excess_1d: np.ndarray | None = None,
) -> Path | None:
    if evaluation.daily_strategy_returns_1d.size == 0:
        _logger.warning("[P0] l2_gate_inputs skipped: empty gate-input series (verdict=%s)", evaluation.verdict)
        return None
    out_path = run_dir / "l2_gate_inputs.npz"
    if l1_prior_excess_1d is not None:
        np.savez_compressed(
            str(out_path),
            daily_strategy_returns_1d=evaluation.daily_strategy_returns_1d,
            daily_benchmark_returns_1d=evaluation.daily_benchmark_returns_1d,
            daily_excess_returns_1d=evaluation.daily_excess_returns_1d,
            daily_fee_returns_1d=evaluation.daily_fee_returns_1d,
            daily_day_start_ns=evaluation.daily_day_start_ns,
            l1_prior_excess_1d=l1_prior_excess_1d,
        )
    else:
        np.savez_compressed(
            str(out_path),
            daily_strategy_returns_1d=evaluation.daily_strategy_returns_1d,
            daily_benchmark_returns_1d=evaluation.daily_benchmark_returns_1d,
            daily_excess_returns_1d=evaluation.daily_excess_returns_1d,
            daily_fee_returns_1d=evaluation.daily_fee_returns_1d,
            daily_day_start_ns=evaluation.daily_day_start_ns,
        )
    _logger.info("[P0] l2_gate_inputs written: %s (oos_days=%d)",
                 out_path, evaluation.oos_days)
    return out_path


def run_multiscale_compound_main(config: CompoundRunConfig) -> RunnerResult:
    try:
        _logger.info("multiscale compound run: date=%s sync=%s",
                     config.reference_date, config.sync)

        engine_config = build_compound_engine_config(config)
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
        if config.history_days < 730:
            start_dt = pd.Timestamp(ref_date_str, tz="UTC") - pd.Timedelta(days=config.history_days)
            execution_calendar = pd.date_range(
                start=start_dt, periods=config.history_days * 24, freq="h", tz="UTC",
            )
        else:
            execution_calendar = build_quarterly_execution_calendar(window)
        ns_per_hour = 3_600_000_000_000

        universe = build_daily_pit_universe(
            snapshot=snapshot, execution_calendar=execution_calendar, config=config,
        )
        acquisition_start = datetime.fromtimestamp(
            window.acquisition_start_ns / 1_000_000_000, tz=UTC,
        ).date()
        core_plan = restrict_to_complete_core_symbols(
            plan=build_ingestion_plan(
                config=config.data_lake,
                reference_date=window.cutoff_date,
                start_date=acquisition_start,
            ),
            catalog=runtime.catalog,
        )
        universe = restrict_pit_universe_to_symbols(
            universe=universe,
            allowed_symbols=core_plan.selected_symbols,
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
            field_plan=prepared.field_plan, window=window,
        )

        strategy_spec_hash = compute_strategy_spec_hash(config=engine_config)
        holdout_store = _get_holdout_store(config)
        trial_ledger = _get_trial_ledger(config)
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
            strategy_spec_hash=strategy_spec_hash,
            universe_state_hash=snapshot.universe_state_hash,
        )

        sealed = holdout_store.ensure_sealed(holdout_manifest)
        _logger.info("holdout %s sealed: spec_hash=%s", holdout_id, sealed.strategy_spec_hash)

        _logger.info("running multiscale compound engine with quarterly window")
        engine_result = run_multiscale_compound_engine(
            market=market,
            universe=universe.state_cube,
            window=window,
            recipe_plan=prepared.recipe_plan,
            holdout_store=holdout_store,
            holdout_id=holdout_id,
            config=engine_config,
            strategy_spec_hash=strategy_spec_hash,
            trial_ledger=trial_ledger,
        )

        paths = _build_artifact_paths(config)
        _write_artifacts(paths, engine_result, config, market=market)

        write_l2_gate_inputs(Path(paths.result_path).parent, engine_result.l2)

        if not engine_result.l2.integrity_ok:
            return RunnerResult(exit_code=1, reason="integrity_failure")

        if engine_result.l2.verdict == L2GateVerdict.NO_EVIDENCE:
            _logger.info("L2 verdict=NO_EVIDENCE: cash-only, exit 0")
            return RunnerResult(exit_code=0, reason="cash_no_evidence")

        deployment_path: Path | None = None
        candidate = getattr(engine_result, "deployment_candidate", None)
        if engine_result.l3.verdict == DeploymentVerdict.PROMOTE and isinstance(candidate, DeploymentCandidate):  # pragma: no cover
            destination = Path("data/futures/compound/deployments")  # pragma: no cover
            deployment_path = publish_promoted_strategy(  # pragma: no cover
                result=engine_result,
                candidate=candidate,
                config=engine_config,
                destination=destination,
            )
            if deployment_path is not None:  # pragma: no cover
                _logger.info("deployment published: %s", deployment_path)  # pragma: no cover
        else:
            _logger.info(
                "L2 verdict=%s: no promotion",
                engine_result.l2.verdict.value,
            )

        if engine_result.l3.verdict.value in ("reject",):
            return RunnerResult(exit_code=0, reason=f"l3_{engine_result.l3.verdict.value}")

        if deployment_path is not None:
            return RunnerResult(exit_code=0, reason=f"promoted:{deployment_path.name}")  # pragma: no cover

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
