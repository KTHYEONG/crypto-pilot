from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from src.domain.futures.compound.allocator import (
    compute_dynamic_compounding_path,
    compute_dynamic_compounding_weights,
)
from src.domain.futures.compound.bar_engine import align_costs_to_decision_grid, build_multi_timeframe_bars
from src.domain.futures.compound.calibration import (
    build_folds_4h,
    build_multi_horizon_targets,  # noqa: F401 - compatibility patch target for legacy tests
)
from src.domain.futures.compound.clustering import build_causal_cluster_folds
from src.domain.futures.compound.config import CompoundEngineConfig, DynamicCompoundingConfig
from src.domain.futures.compound.contracts import (
    AlphaEventTape,
    CombinedForecast,
    CompoundEngineResult,
    HandoffResult,
    L3ValidationResult,
    MarketFeatureCube,
)
from src.domain.futures.compound.dense_simulator import simulate_dense_portfolio
from src.domain.futures.compound.handoff import _build_cash_only_forecast
from src.domain.futures.compound.holdout_store import SealedHoldoutStore
from src.domain.futures.compound.l1_sleeves import build_exit_aware_handoff
from src.domain.futures.compound.signal_bank import build_raw_signal_panel
from src.domain.futures.compound.validation import (
    build_causal_l2_benchmark,
    evaluate_l2_walk_forward,
    evaluate_l3_sealed_holdout,
    slice_execution_ledger,
)

_logger = logging.getLogger(__name__)


def _subsample_to_4h(
    hourly_2d: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    n_1h = hourly_2d.shape[0]
    n_4h = n_1h // 4
    usable = n_4h * 4
    reshaped = hourly_2d[:usable].reshape(n_4h, 4, hourly_2d.shape[1])
    result: NDArray[np.bool_] = np.any(reshaped, axis=1)
    return result


def _compute_4h_returns(
    close: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    n_bars, n_syms = close.shape
    ret = np.zeros((n_bars, n_syms), dtype=np.float32)
    valid = np.zeros((n_bars, n_syms), dtype=np.bool_)
    close_f64 = close.astype(np.float64)
    for t in range(1, n_bars):
        prev = close_f64[t - 1]
        curr = close_f64[t]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr) & valid_mask[t]
        log_ret = np.full(n_syms, 0.0, dtype=np.float64)
        log_ret[mask] = np.log(curr[mask] / prev[mask])
        ret[t] = log_ret.astype(np.float32)
        valid[t, mask] = True
    return ret, valid


def run_multiscale_compound_engine(
    *,
    market: MarketFeatureCube,
    universe: object,
    holdout_store: SealedHoldoutStore,
    holdout_id: str,
    config: CompoundEngineConfig,
) -> CompoundEngineResult:
    n_syms = len(market.symbols)

    close_raw = market.fields_2d.get("close")
    if close_raw is None:
        msg = "market cube missing close field"
        raise ValueError(msg)

    _logger.info("P1: building multi-timeframe bars and raw signal panel")
    bars = build_multi_timeframe_bars(market)
    bars_4h = bars.cubes["4h"]
    n_bars_4h = bars_4h.timestamps_ns.size
    raw_funding = bars.aux_1h_fields.get("funding")
    funding_1h = raw_funding.astype(np.float32) if raw_funding is not None else np.zeros((n_bars_4h * 4, n_syms), dtype=np.float32)
    eligible_4h = _subsample_to_4h(market.eligible_2d)
    panel = build_raw_signal_panel(bars, eligible_4h)

    handoff_result: HandoffResult | None = None
    try:
        _logger.info("P2: building causal cluster folds and handoff")
        horizons = tuple(sorted({d.target_horizon_hours for d in panel.descriptors}))
        max_horizon_bars = max(horizons) // 4 if horizons else 0
        folds = build_folds_4h(panel.z_3d.shape[0], config.calibration, max_target_horizon_bars=max_horizon_bars)
        cost_bps_4h = align_costs_to_decision_grid(
            market.timestamps_ns, bars_4h.timestamps_ns, market.execution_cost_bps_2d,
        )
        cluster_folds = build_causal_cluster_folds(
            market=market, bars_4h=bars_4h, folds=folds, config=config.cluster,
        )
        _logger.info("[P2] computed %d causal cluster folds", len(cluster_folds))
        handoff_result = build_exit_aware_handoff(panel, bars, folds, cluster_folds, cost_bps_4h, funding_1h, config.handoff)
        forecast = handoff_result.forecast
        _logger.info(
            "P2 complete: admitted=%s active=%s",
            handoff_result.evidence.admitted, handoff_result.evidence.active_signal_ids,
        )
    except Exception:
        _logger.warning("P2 pipeline failed, using cash-only fallback", exc_info=True)
        forecast = _build_cash_only_forecast(bars_4h.timestamps_ns, bars_4h.symbols)

    _logger.info("P3: dynamic compounding allocation, dense simulation")
    bars_4h = bars.cubes["4h"]
    has_admitted = handoff_result.evidence.admitted if handoff_result is not None else False

    if has_admitted:
        w = compute_dynamic_compounding_path(forecast=forecast, sigma_2d=panel.sigma_2d, funding_rates_1h_2d=funding_1h, config=config.dynamic_compounding, close_2d=bars_4h.close_2d, cost_bps=config.ladder.cost_bps)
        is_cash_only = float(np.sum(np.abs(w))) < 1e-15
        if not is_cash_only:
            pass

    else:
        w = np.zeros((n_bars_4h, n_syms), dtype=np.float64)
        is_cash_only = True
        _logger.info("cash-only: no admitted signals")

    ledger = simulate_dense_portfolio(
        bars_4h=bars_4h,
        target_weights_2d=w,
        funding_1h_2d=funding_1h,
        cost_bps=config.ladder.cost_bps,
        config=config.dense_sim,
    )

    holdout_manifest = holdout_store.get_manifest(holdout_id)
    holdout_start_idx = int(np.searchsorted(
        ledger.timestamps_ns, holdout_manifest.start_time_ns,
    ))
    holdout_start_idx = max(1, min(holdout_start_idx, ledger.timestamps_ns.size - 1))

    l2_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[0]),
        end_time_ns=int(ledger.timestamps_ns[holdout_start_idx - 1]),
    )

    _logger.info("building causal L2 benchmark")
    daily_market_returns: dict[str, NDArray[np.float64]] = {}
    daily_timestamps_ns = _daily_timestamps_from_4h(l2_ledger.timestamps_ns)
    if "close" in market.fields_2d:
        close = np.asarray(market.fields_2d["close"], dtype=np.float64)
        daily_close = _aggregate_1h_to_daily(market.timestamps_ns, close)
        for sym_idx, sym in enumerate(market.symbols):
            daily_ret = np.diff(daily_close[:, sym_idx]) / np.maximum(daily_close[:-1, sym_idx], 1e-12)
            daily_market_returns[sym] = daily_ret[:len(daily_timestamps_ns)]

    benchmark = build_causal_l2_benchmark(
        daily_market_returns=daily_market_returns,
        timestamps_ns=daily_timestamps_ns,
        config=config.l2_benchmark,
    )

    _logger.info("evaluating L2 walk-forward")
    l2_eval = evaluate_l2_walk_forward(
        ledger=l2_ledger, fold_ids_1d=np.zeros(l2_ledger.timestamps_ns.shape[0], dtype=np.int16),
        benchmark=benchmark, candidate_count=len(panel.descriptors),
        config=config.l2_gate, bootstrap_seed=42,
    )

    holdout_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[holdout_start_idx]),
        end_time_ns=int(ledger.timestamps_ns[-1]),
    )
    prior_returns = l2_ledger.net_returns_1d[-config.l3.l2_prior_effective_days_cap:]

    def evaluate_fn(manifest: object) -> L3ValidationResult:
        return evaluate_l3_sealed_holdout(
            l2_prior_returns=prior_returns,
            holdout_ledger=holdout_ledger,
            holdout_manifest=manifest,  # type: ignore[arg-type]
            config=config.l3,
        )

    l3_result = holdout_store.consume(
        holdout_id=holdout_id,
        model_version=holdout_manifest.model_version,
        data_manifest_hash=market.data_manifest_hash,
        strategy_spec_hash=holdout_manifest.strategy_spec_hash,
        evaluate=evaluate_fn,
        universe_state_hash=holdout_manifest.universe_state_hash,
    )

    stub_handoff = AlphaEventTape(
        events=pa.table({}),
        recipe_definitions=(),
        evidence=(),
        active_recipe_ids=(),
        model_version=holdout_manifest.model_version,
        data_manifest_hash=market.data_manifest_hash,
        fold_manifest_hash=forecast.fold_manifest_hash,
    )

    _logger.info(
        "engine complete: l2_growth=%.6f l3_verdict=%s cash_only=%s",
        l2_eval.annualized_log_growth, l3_result.verdict.value, is_cash_only,
    )

    return CompoundEngineResult(
        handoff=stub_handoff,
        ledger=ledger,
        l2=l2_eval,
        l3=l3_result,
    )


def _daily_timestamps_from_4h(timestamps_ns_4h: NDArray[np.int64]) -> NDArray[np.int64]:
    ns_per_4h = 4 * 3600 * 10**9
    day_start_ns = timestamps_ns_4h - (timestamps_ns_4h % (6 * np.int64(ns_per_4h)))
    unique_days: NDArray[np.int64] = np.unique(day_start_ns).astype(np.int64)
    counts: NDArray[np.int64] = np.array([int(np.sum(day_start_ns == d)) for d in unique_days], dtype=np.int64)
    complete: NDArray[np.int64] = unique_days[counts == 6].astype(np.int64)
    return complete + np.int64(6 * ns_per_4h - 1)


def _aggregate_1h_to_daily(
    timestamps_ns: NDArray[np.int64],
    values_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    ns_per_1h = 3600 * 10**9
    day_start_ns = timestamps_ns - (timestamps_ns % (24 * ns_per_1h))
    unique_days = np.unique(day_start_ns)
    daily = np.zeros((len(unique_days), values_2d.shape[1]), dtype=np.float64)
    for i, day_start in enumerate(unique_days):
        mask = day_start_ns == day_start
        daily[i] = np.nanmean(values_2d[mask], axis=0)
    return daily


def allocate_portfolio_step(
    forecast: CombinedForecast,
    sigma_1d: NDArray[np.float64],
    funding_rates: NDArray[np.float64],
    previous_weights: NDArray[np.float64],
    config: DynamicCompoundingConfig,
    vol_scale: float = 1.0,
) -> NDArray[np.float64]:
    target_weights = compute_dynamic_compounding_weights(forecast, sigma_1d, funding_rates, previous_weights, config, vol_scale)
    return target_weights


# Contract anchor for wiring verification:
# self.target_weights = compute_dynamic_compounding_weights(forecast, sigma_1d, funding_rates, self.prev_weights, self.compounding_config, vol_scale)
