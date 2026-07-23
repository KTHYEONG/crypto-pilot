from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

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
    prepare_data_snapshot,
)
from src.application.futures.runner.models import RunnerResult
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.engine import run_multiscale_compound_engine
from src.domain.futures.data_lake.ingestion import DataCoverageError, StorageBudgetError

_logger = logging.getLogger(__name__)


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
) -> None:
    result_data = {
        "reference_date": config.reference_date,
        "seed": config.seed,
        "base_timeframe": config.base_timeframe,
        "n_bars": int(engine_result.ledger.timestamps_ns.size),
        "n_symbols": int(engine_result.ledger.target_weights_2d.shape[1]) if engine_result.ledger.target_weights_2d.ndim > 1 else 0,
        "universe_symbols": list(engine_result.alpha_tape.symbols),
        "l2": {
            "annualized_log_growth": engine_result.l2.annualized_log_growth,
            "growth_ci90_lower": engine_result.l2.growth_ci90[0],
            "growth_ci90_upper": engine_result.l2.growth_ci90[1],
            "equity_multiple": engine_result.l2.equity_multiple,
            "max_drawdown": engine_result.l2.max_drawdown,
            "daily_cvar95": engine_result.l2.daily_cvar95,
            "annual_volatility": engine_result.l2.annual_volatility,
            "turnover": engine_result.l2.turnover,
            "safe": engine_result.l2.safe,
            "integrity_ok": engine_result.l2.integrity_ok,
        },
        "l3": {
            "verdict": engine_result.l3.verdict.value,
            "posterior_growth_probability": engine_result.l3.posterior_growth_probability,
            "holdout_days": engine_result.l3.holdout_days,
            "max_drawdown": engine_result.l3.max_drawdown,
            "daily_cvar95": engine_result.l3.daily_cvar95,
            "reasons": list(engine_result.l3.reasons),
        },
    }
    with open(paths.result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    np.save(paths.target_weights_path, engine_result.ledger.target_weights_2d)

    manifest = {
        "model_version": engine_result.alpha_tape.model_version,
        "data_manifest_hash": engine_result.alpha_tape.data_manifest_hash,
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
        snapshot = prepare_data_snapshot(config=config, runtime=runtime)

        universe = build_daily_pit_universe(snapshot=snapshot, config=config.universe)

        _logger.info(
            "universe built: %d symbols from snapshot %s",
            len(universe.symbols), snapshot.snapshot_id,
        )

        market = build_multiscale_market_cube(snapshot=snapshot, universe=universe, config=config)

        holdout_bars = 180 * 24
        holdout_start_bar = max(0, market.timestamps_ns.size - holdout_bars)
        holdout_start_ns = int(market.timestamps_ns[holdout_start_bar]) if market.timestamps_ns.size > 0 else 0

        holdout_manifest = SealedHoldoutManifest(
            holdout_id=f"multiscale-{config.reference_date or 'live'}",
            start_time_ns=holdout_start_ns,
            end_time_ns=int(market.timestamps_ns[-1]) if market.timestamps_ns.size > 0 else holdout_start_ns,
            holdout_days=180,
            model_version="multiscale-v1",
            data_manifest_hash=market.data_manifest_hash,
        )

        _logger.info("running multiscale compound engine")
        engine_result = run_multiscale_compound_engine(
            market=market,
            universe=universe,
            holdout_manifest=holdout_manifest,
            config=engine_config,
        )

        paths = _build_artifact_paths(config)
        _write_artifacts(paths, engine_result, config)

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
