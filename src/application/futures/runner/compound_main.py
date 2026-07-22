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
from src.application.futures.runner.compound_data import check_data_readiness, load_hourly_data
from src.application.futures.runner.compound_universe import (
    build_pit_universe_state,
    resolve_universe_symbols,
    sync_universe_ledger,
)
from src.application.futures.runner.models import RunnerResult
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.data_plane import build_compound_market_feature_cube
from src.domain.futures.compound.engine import run_compound_engine

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


def _run_engine_from_loaded_data(
    cube: MarketFeatureCube,
    holdout_manifest: SealedHoldoutManifest,
    config: CompoundEngineConfig,
) -> CompoundEngineResult:
    return run_compound_engine(
        cube=cube,
        holdout_manifest=holdout_manifest,
        config=config,
    )


def run_compound_main(config: CompoundRunConfig) -> RunnerResult:
    try:
        _logger.info("compound run starting: date=%s sync=%s seed=%d",
                     config.reference_date, config.sync, config.seed)

        ref_dt, _synced = sync_universe_ledger(config)

        symbols = resolve_universe_symbols(config, ref_dt)
        if not symbols:
            _logger.info("no cached symbols found; using empty universe")
            symbols = ("BTCUSDT", "ETHUSDT")

        _logger.info("loading hourly data for %d symbols", len(symbols))
        data_maps = load_hourly_data(config, symbols, ref_dt=ref_dt)

        if not check_data_readiness(data_maps):
            return RunnerResult(exit_code=1, reason="insufficient_data_readiness")

        valid_symbols = tuple(
            sym for sym in symbols
            if sym in data_maps and "1h" in data_maps[sym] and not data_maps[sym]["1h"].empty
        )
        if not valid_symbols:
            return RunnerResult(exit_code=1, reason="no_valid_symbol_data")

        state_cube = build_pit_universe_state(valid_symbols, ref_dt)

        engine_config = CompoundEngineConfig()
        cube = build_compound_market_feature_cube(
            data_maps=data_maps,
            symbols=valid_symbols,
            state_cube=state_cube,
            timeframe="1h",
            data_manifest_hash="cached-hourly-data",
            config=engine_config.data,
        )

        holdout_start_ns = (
            int(cube.timestamps_ns[-1])
            if cube.timestamps_ns.size > 0
            else int(np.datetime64(ref_dt).astype(np.int64))
        )
        holdout_days = 90
        holdout_start_bar = max(0, cube.timestamps_ns.size - holdout_days * 24)
        holdout_start_ns = int(cube.timestamps_ns[holdout_start_bar]) if holdout_start_bar < cube.timestamps_ns.size else holdout_start_ns

        holdout_manifest = SealedHoldoutManifest(
            holdout_id=f"compound-{config.reference_date or 'live'}",
            start_time_ns=holdout_start_ns,
            end_time_ns=int(cube.timestamps_ns[-1]) if cube.timestamps_ns.size > 0 else holdout_start_ns,
            holdout_days=holdout_days,
            model_version="compound-v1",
            data_manifest_hash=cube.data_manifest_hash,
        )

        _logger.info("running compound engine")
        engine_result = _run_engine_from_loaded_data(
            cube=cube,
            holdout_manifest=holdout_manifest,
            config=engine_config,
        )

        paths = _build_artifact_paths(config)
        _write_artifacts(paths, engine_result, config)

        equity_len = len(engine_result.ledger.equity_1d)
        ts_len = len(engine_result.ledger.timestamps_ns)
        tw_len = len(engine_result.ledger.target_weights_2d)
        if not (equity_len == ts_len == tw_len):
            _logger.error("length mismatch: equity=%d timestamps=%d target_weights=%d",
                          equity_len, ts_len, tw_len)
            return RunnerResult(exit_code=1, reason="ledger_length_mismatch")

        _logger.info("compound run successful: equity=%.4f l2_growth=%.6f l3_verdict=%s",
                     float(engine_result.ledger.equity_1d[-1]),
                     engine_result.l2.annualized_log_growth,
                     engine_result.l3.verdict.value)

        return RunnerResult(exit_code=0, reason=f"ok:l3_{engine_result.l3.verdict.value}")

    except Exception as exc:
        _logger.exception("compound run failed")
        return RunnerResult(exit_code=1, reason=str(exc))
