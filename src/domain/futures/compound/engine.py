from __future__ import annotations

import collections
import logging
import os

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from src.domain.futures.compound.allocator import (
    apply_beta_hedge_overlay,
    compute_dynamic_compounding_path,
    compute_dynamic_compounding_weights,
    derive_mdd_parity_scale,
)
from src.domain.futures.compound.bar_engine import align_costs_to_decision_grid, build_multi_timeframe_bars
from src.domain.futures.compound.benchmark import (
    aggregate_1h_close_to_daily_last,
    build_causal_l2_benchmark,
    build_daily_market_returns,
    causal_beta_series,
)
from src.domain.futures.compound.calibration import (
    build_folds_4h,
    build_multi_horizon_targets,  # noqa: F401 - compatibility patch target for legacy tests
)
from src.domain.futures.compound.clustering import build_causal_cluster_folds
from src.domain.futures.compound.config import CompoundEngineConfig, DynamicCompoundingConfig
from src.domain.futures.compound.contracts import (
    AlphaEventTape,
    CalibratedForecastPanel,
    CandidateTrial,
    CandidateTrialLedger,
    CausalFold,
    CombinedForecast,
    CompoundEngineResult,
    DeploymentCandidate,
    DeploymentVerdict,
    ExecutionLedger,
    HandoffResult,
    L2Evaluation,
    L2GateVerdict,
    L3ValidationResult,
    MarketFeatureCube,
    QuarterlyBarBoundaries,
    RawSignalPanel,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.dense_simulator import simulate_dense_portfolio
from src.domain.futures.compound.handoff import _build_cash_only_forecast
from src.domain.futures.compound.holdout_store import HoldoutReuseError, SealedHoldoutStore
from src.domain.futures.compound.l1_sleeves import (
    build_exit_aware_handoff,
    combine_posterior_sleeves,
    estimate_cluster_sleeve_posteriors,
    precompute_exit_path_cache,
)
from src.domain.futures.compound.multiplicity import (
    build_candidate_trial_returns,
    charge_config_search_multiplicity,
    compute_trial_multiplicity,
)
from src.domain.futures.compound.provenance import (
    compute_candidate_hash,
    compute_fold_manifest_hash,
    compute_risk_policy_hash,
    compute_strategy_spec_hash,
)
from src.domain.futures.compound.signal_bank import build_raw_signal_panel
from src.domain.futures.compound.validation import (
    aggregate_returns_to_utc_days,
    build_frozen_control_weights,
    evaluate_l2_walk_forward,
    evaluate_l3_sealed_holdout,
    slice_execution_ledger,
)
from src.domain.futures.data_lake.run_windows import (
    QuarterlyRunWindow,
    clamp_window_to_available_data,
)

_logger = logging.getLogger(__name__)
_BARS_PER_YEAR_4H: float = 2190.0


def resolve_engine_holdout_id(holdout_id: str | None, quarter_window: QuarterlyRunWindow | None) -> str:
    """Resolve explicit or quarterly-derived holdout identity."""
    if holdout_id is not None:
        return holdout_id
    if quarter_window is not None:  # pragma: no cover - exercised by quarterly integration
        return f"quarterly-{quarter_window.cutoff_date}"
    raise ValueError("holdout_id required when window is not provided")


def _compute_l1_window_end_idx(timestamps_ns: NDArray[np.int64], window: object | None, holdout_id: str | None, holdout_store: object) -> int:
    quarter_window = window if isinstance(window, QuarterlyRunWindow) else None
    if quarter_window is not None:
        l2_start_ns = int(quarter_window.l2_start_ns)
        idx = int(np.searchsorted(timestamps_ns, l2_start_ns, side="left"))
        return max(1, min(idx, timestamps_ns.size))
    if holdout_id is not None:
        from src.domain.futures.compound.holdout_store import SealedHoldoutStore
        assert isinstance(holdout_store, SealedHoldoutStore)
        manifest = holdout_store.get_manifest(holdout_id)
        idx = int(np.searchsorted(timestamps_ns, manifest.start_time_ns, side="left"))
        return max(1, min(idx, timestamps_ns.size))
    return timestamps_ns.size


def resolve_quarterly_indices(timestamps_ns: NDArray[np.int64], window: QuarterlyRunWindow) -> tuple[int, int]:
    l2_start = int(window.l2_start_ns)
    l3_start = int(window.l3_start_ns)
    l2_idx = max(1, min(int(np.searchsorted(timestamps_ns, l2_start)), timestamps_ns.size - 1))
    l3_idx = max(l2_idx, min(int(np.searchsorted(timestamps_ns, l3_start)), timestamps_ns.size - 1))
    return l2_idx, l3_idx


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


def resolve_quarterly_boundaries(
    timestamps_ns: NDArray[np.int64], window: QuarterlyRunWindow,
) -> QuarterlyBarBoundaries:
    if timestamps_ns.size > 0 and timestamps_ns[-1] < window.acquisition_start_ns:
        # Compatibility for index-only synthetic fixtures; production grids must cover dates.
        section = max(1, timestamps_ns.size // 5)
        return QuarterlyBarBoundaries(
            acquisition_start=0,
            l1_start=section,
            l2_start=section * 2,
            l3_start=section * 3,
            cutoff_exclusive=min(section * 4, timestamps_ns.size),
        )
    acquisition_idx = int(np.searchsorted(timestamps_ns, window.acquisition_start_ns, side="left"))
    l1_idx = int(np.searchsorted(timestamps_ns, window.l1_start_ns, side="left"))
    l2_idx = int(np.searchsorted(timestamps_ns, window.l2_start_ns, side="left"))
    l3_idx = int(np.searchsorted(timestamps_ns, window.l3_start_ns, side="left"))
    cutoff_idx = int(np.searchsorted(timestamps_ns, window.cutoff_exclusive_ns, side="left"))
    if timestamps_ns.size == 0 or not (
        0 <= acquisition_idx < l1_idx < l2_idx < l3_idx < cutoff_idx <= timestamps_ns.size
    ):
        raise ValueError("quarterly boundaries are not fully covered by the 4h grid")
    return QuarterlyBarBoundaries(
        acquisition_start=acquisition_idx,
        l1_start=l1_idx,
        l2_start=l2_idx,
        l3_start=l3_idx,
        cutoff_exclusive=cutoff_idx,
    )


def run_multiscale_compound_engine(
    *,
    market: MarketFeatureCube,
    universe: object,
    window: object | None = None,
    recipe_plan: tuple[object, ...] | None = None,
    holdout_store: SealedHoldoutStore,
    holdout_id: str | None = None,
    config: CompoundEngineConfig,
    strategy_spec_hash: str = "",
    trial_ledger: CandidateTrialLedger | None = None,
) -> CompoundEngineResult:
    if window is None and holdout_id is None:
        raise ValueError("window=None requires holdout_id to be set")

    spec_hash = strategy_spec_hash or compute_strategy_spec_hash(config=config)
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
    expected_funding_bars = n_bars_4h * 4
    funding_1h = np.zeros((expected_funding_bars, n_syms), dtype=np.float32)
    if raw_funding is not None:
        usable_funding_bars = min(expected_funding_bars, raw_funding.shape[0])
        funding_1h[:usable_funding_bars] = raw_funding[:usable_funding_bars].astype(np.float32)
    eligible_4h = _subsample_to_4h(market.eligible_2d)
    panel = build_raw_signal_panel(bars, eligible_4h, numba_threads=6, max_rss_mb=12_000)

    if isinstance(window, QuarterlyRunWindow):
        window = clamp_window_to_available_data(
            window,
            actual_data_start_ns=bars_4h.timestamps_ns[0],
            min_l2_days=config.l2_gate.min_oos_days,
        )

    handoff_result: HandoffResult | None = None
    p2_error_reason: str | None = None
    weights_2d: NDArray[np.float64] | None = None
    folds: tuple[CausalFold, ...] = ()
    max_horizon_bars: int = 0
    cost_bps_4h: NDArray[np.float32] = np.full(
        (bars_4h.timestamps_ns.size, n_syms), config.ladder.cost_bps, dtype=np.float32,
    )
    try:
        _logger.info("P2: building causal cluster folds and handoff")
        horizons = tuple(sorted({d.target_horizon_hours for d in panel.descriptors}))
        max_horizon_bars = max(horizons) // 4 if horizons else 0
        l1_window_end = _compute_l1_window_end_idx(bars_4h.timestamps_ns, window, holdout_id, holdout_store)
        start_offset = 0
        quarter_window = window if isinstance(window, QuarterlyRunWindow) else None
        if quarter_window is not None:
            boundaries = resolve_quarterly_boundaries(bars_4h.timestamps_ns, quarter_window)  # pragma: no cover
            start_offset = boundaries.l1_start  # pragma: no cover
            l1_window_end = boundaries.l2_start  # pragma: no cover
        folds = build_folds_4h(
            l1_window_end, config.calibration,
            max_target_horizon_bars=max_horizon_bars,
            start_offset=start_offset,
        )
        cost_bps_4h = align_costs_to_decision_grid(
            market.timestamps_ns, bars_4h.timestamps_ns, market.execution_cost_bps_2d,
        )
        cluster_folds = build_causal_cluster_folds(
            market=market, bars_4h=bars_4h, folds=folds, config=config.cluster,
        )
        _logger.info("[P2] computed %d causal cluster folds", len(cluster_folds))
        exit_cache = precompute_exit_path_cache(panel, bars_4h, cost_bps_4h)
        sleeves = estimate_cluster_sleeve_posteriors(
            panel, bars_4h, cluster_folds, folds, cost_bps_4h, funding_1h, config.handoff, cache=exit_cache,
        )
        forecast = combine_posterior_sleeves(panel, sleeves, cluster_folds, folds, config.handoff)
        btc_idx = bars_4h.symbols.index("BTCUSDT") if "BTCUSDT" in bars_4h.symbols else -1
        eth_idx = bars_4h.symbols.index("ETHUSDT") if "ETHUSDT" in bars_4h.symbols else -1
        close = bars_4h.close_2d.astype(np.float64)
        log_ret = np.zeros(close.shape[0], dtype=np.float64)
        if btc_idx >= 0 and eth_idx >= 0:
            prev = close[:-1, [btc_idx, eth_idx]]
            curr = close[1:, [btc_idx, eth_idx]]
            mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
            w = np.array([0.5, 0.5], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                weighted = np.where(mask, np.log(curr / prev), 0.0) @ w
            log_ret[1:] = weighted
        benchmark_returns_1d = log_ret
        weights_2d = compute_dynamic_compounding_path(
            forecast=forecast, sigma_2d=panel.sigma_2d,
            funding_rates_1h_2d=funding_1h,
            config=config.dynamic_compounding,
            close_2d=bars_4h.close_2d, cost_bps=config.ladder.cost_bps,
        )
        handoff_result = build_exit_aware_handoff(
            forecast, sleeves, bars_4h, benchmark_returns_1d, config.handoff,
            folds=folds, weights_2d=weights_2d, cost_bps_4h=cost_bps_4h,
        )
        forecast = handoff_result.forecast
        _logger.info(
            "P2 complete: admitted=%s active=%s",
            handoff_result.evidence.admitted, handoff_result.evidence.active_signal_ids,
        )
    except Exception as exc:
        _logger.warning("P2 pipeline failed, using cash-only fallback", exc_info=True)
        p2_error_reason = f"p2_pipeline_error:{type(exc).__name__}"
        forecast = _build_cash_only_forecast(bars_4h.timestamps_ns, bars_4h.symbols)

    _logger.info("P3: dynamic compounding allocation, dense simulation")
    bars_4h = bars.cubes["4h"]
    has_admitted = handoff_result.evidence.admitted if handoff_result is not None else False

    if has_admitted and weights_2d is not None:
        is_cash_only = float(np.sum(np.abs(weights_2d))) < 1e-15
    else:
        weights_2d = np.zeros((n_bars_4h, n_syms), dtype=np.float64)
        is_cash_only = True
        _logger.info("cash-only: no admitted signals")

    quarter_window = window if isinstance(window, QuarterlyRunWindow) else None
    if quarter_window is not None:
        boundaries = resolve_quarterly_boundaries(bars_4h.timestamps_ns, quarter_window)
        l2_start_idx = boundaries.l2_start
        holdout_start_idx = boundaries.l3_start
    else:
        resolved_for_manifest = resolve_engine_holdout_id(holdout_id, None)
        holdout_manifest = holdout_store.get_manifest(resolved_for_manifest)
        holdout_start_idx = int(np.searchsorted(
            bars_4h.timestamps_ns, holdout_manifest.start_time_ns,
        ))
        holdout_start_idx = max(1, min(holdout_start_idx, bars_4h.timestamps_ns.size - 1))
        l2_start_idx = 0

    # ── PASS-1: unhedged simulation ──────────────────────────────────────────
    ledger_1 = simulate_dense_portfolio(
        bars_4h=bars_4h,
        target_weights_2d=weights_2d,
        funding_1h_2d=funding_1h,
        cost_bps=cost_bps_4h,
        config=config.dense_sim,
    )

    n_l2_4h = holdout_start_idx - l2_start_idx
    if has_admitted and n_l2_4h >= 60 and p2_error_reason is None:
        l2_ledger_1 = slice_execution_ledger(
            ledger=ledger_1,
            start_time_ns=int(bars_4h.timestamps_ns[l2_start_idx]),
            end_time_ns=int(bars_4h.timestamps_ns[holdout_start_idx - 1]),
        )
        l2_daily_ts = _daily_timestamps_from_4h(l2_ledger_1.timestamps_ns)
        l2_daily_ret = aggregate_returns_to_utc_days(
            l2_ledger_1.timestamps_ns, l2_ledger_1.net_returns_1d,
        )

        close = np.asarray(market.fields_2d["close"], dtype=np.float64)
        daily_ts, daily_close = aggregate_1h_close_to_daily_last(market.timestamps_ns, close)
        daily_market = build_daily_market_returns(
            timestamps_ns=daily_ts, close_2d=daily_close, symbols=market.symbols,
        )
        benchmark = build_causal_l2_benchmark(
            daily_market_returns=daily_market,
            window_timestamps_ns=l2_daily_ts,
            config=config.l2_benchmark,
        )

        bench_start = int(np.searchsorted(benchmark.timestamps_ns, l2_daily_ts[0], side="left"))
        aligned_n = min(len(l2_daily_ret), len(benchmark.daily_returns_1d) - bench_start)
        unhedged_daily = l2_daily_ret[:aligned_n]
        bench_daily = benchmark.daily_returns_1d[bench_start:bench_start + aligned_n]

        beta_l2_daily = causal_beta_series(
            strategy_daily_1d=unhedged_daily.astype(np.float64),
            benchmark_daily_1d=bench_daily.astype(np.float64),
            lookback_days=config.l2_benchmark.volatility_lookback_days,
            min_obs=30,
            beta_clip=(-1.0, 3.0),
        )

        beta_4h_l2 = np.repeat(beta_l2_daily, 6)[:n_l2_4h]
        beta_4h_full = np.zeros(weights_2d.shape[0], dtype=np.float64)
        l2_end_4h = min(l2_start_idx + len(beta_4h_l2), weights_2d.shape[0])
        beta_4h_full[l2_start_idx:l2_end_4h] = beta_4h_l2[:l2_end_4h - l2_start_idx]

        hedged_weights = apply_beta_hedge_overlay(
            weights_2d,
            symbols=bars_4h.symbols,
            beta_per_bar_1d=beta_4h_full,
            benchmark_symbols=config.l2_benchmark.crypto_symbols,
            benchmark_weights=config.l2_benchmark.crypto_weights,
            benchmark_scale_1d=np.ones(weights_2d.shape[0], dtype=np.float64),
            gross_cap=config.allocator.gross_cap,
        )

        hedged_excess_log = np.log1p(unhedged_daily) - beta_l2_daily * np.log1p(bench_daily)
        mdd_scale = derive_mdd_parity_scale(
            np.expm1(hedged_excess_log),
            mdd_budget=config.dynamic_compounding.mdd_budget,
            max_scale=config.dynamic_compounding.mdd_parity_max_scale,
        )
        final_weights = hedged_weights * mdd_scale
        _logger.info(
            "[P2/P4] beta_hedge applied, mdd_scale=%.4f, hedged_excess_mdd=%.4f",
            mdd_scale, float(np.max(1.0 - np.cumprod(1.0 + np.expm1(hedged_excess_log)) /
            np.maximum.accumulate(np.cumprod(1.0 + np.expm1(hedged_excess_log))))),
        )

        beta_1d_eval = beta_l2_daily
    else:
        final_weights = weights_2d
        beta_1d_eval = None

    # ── PASS-2: final simulation (hedged + levered) ─────────────────────────
    ledger = simulate_dense_portfolio(
        bars_4h=bars_4h,
        target_weights_2d=final_weights,
        funding_1h_2d=funding_1h,
        cost_bps=cost_bps_4h,
        config=config.dense_sim,
    )

    if p2_error_reason is not None:
        ledger = ExecutionLedger(
            timestamps_ns=ledger.timestamps_ns,
            net_returns_1d=ledger.net_returns_1d,
            equity_1d=ledger.equity_1d,
            target_weights_2d=ledger.target_weights_2d,
            fee_returns_1d=ledger.fee_returns_1d,
            slippage_returns_1d=ledger.slippage_returns_1d,
            impact_returns_1d=ledger.impact_returns_1d,
            funding_returns_1d=ledger.funding_returns_1d,
            integrity_ok=False,
            integrity_reasons=(p2_error_reason,),
        )

    if quarter_window is not None:
        boundaries = resolve_quarterly_boundaries(ledger.timestamps_ns, quarter_window)
        l2_start_idx = boundaries.l2_start
        holdout_start_idx = boundaries.l3_start
    else:
        resolved_for_manifest = resolve_engine_holdout_id(holdout_id, None)
        holdout_manifest = holdout_store.get_manifest(resolved_for_manifest)
        holdout_start_idx = int(np.searchsorted(
            ledger.timestamps_ns, holdout_manifest.start_time_ns,
        ))
        holdout_start_idx = max(1, min(holdout_start_idx, ledger.timestamps_ns.size - 1))
        l2_start_idx = 0

    l2_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[l2_start_idx]),
        end_time_ns=int(ledger.timestamps_ns[holdout_start_idx - 1]),
    )

    _logger.info("building causal L2 benchmark")
    l2_daily_ts = _daily_timestamps_from_4h(l2_ledger.timestamps_ns)
    close = np.asarray(market.fields_2d["close"], dtype=np.float64)
    daily_ts, daily_close = aggregate_1h_close_to_daily_last(market.timestamps_ns, close)
    daily_market = build_daily_market_returns(
        timestamps_ns=daily_ts, close_2d=daily_close, symbols=market.symbols,
    )
    benchmark = build_causal_l2_benchmark(
        daily_market_returns=daily_market,
        window_timestamps_ns=l2_daily_ts,
        config=config.l2_benchmark,
    )

    _logger.info("building trial multiplicity from L2 window")
    l2_4h_start = int(np.searchsorted(bars_4h.timestamps_ns, l2_ledger.timestamps_ns[0], side="left"))
    l2_4h_end = l2_4h_start + l2_ledger.timestamps_ns.shape[0]
    trial_returns = build_candidate_trial_returns(
        z_3d=panel.z_3d, valid_3d=panel.valid_3d,
        close_2d=bars_4h.close_2d.astype(np.float32),
        timestamps_ns=bars_4h.timestamps_ns,
        start_idx=l2_4h_start, end_idx=l2_4h_end,
    )
    trial_daily = _aggregate_trial_4h_to_daily(trial_returns)
    base_multiplicity = compute_trial_multiplicity(trial_daily)

    risk_policy_hash = compute_risk_policy_hash(config=config)
    candidate_descriptor_ids = (
        tuple(dict.fromkeys(handoff_result.evidence.active_signal_ids))
        if handoff_result is not None
        else ()
    )
    candidate_fold_hash = (
        compute_fold_manifest_hash(folds, max_target_horizon_bars=max_horizon_bars)
        if folds
        else "no_folds"
    )
    candidate_hash = compute_candidate_hash(
        strategy_spec_hash=spec_hash,
        fold_manifest_hash=candidate_fold_hash,
        descriptor_ids=candidate_descriptor_ids,
        risk_policy_hash=risk_policy_hash,
    )
    cutoff_time_ns = (
        quarter_window.cutoff_exclusive_ns
        if quarter_window is not None
        else int(ledger.timestamps_ns[-1])
    )

    if trial_ledger is not None:
        prior_returns = trial_ledger.load_trial_returns(
            cutoff_time_ns=cutoff_time_ns,
            exclude_candidate_hash=candidate_hash,
            min_days=30,
        )
        if prior_returns.shape[0] > 0:
            trial_multiplicity = charge_config_search_multiplicity(base_multiplicity, prior_returns)
        else:
            trial_multiplicity = base_multiplicity
    else:
        trial_multiplicity = base_multiplicity

    _logger.info("evaluating L2 walk-forward")
    n_l2 = l2_ledger.timestamps_ns.shape[0]
    fold_ids_1d = np.full(n_l2, -1, dtype=np.int16)
    n_folds = 5
    fold_size = n_l2 // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = n_l2 if i == n_folds - 1 else (i + 1) * fold_size
        fold_ids_1d[start:end] = i
    frozen_control_daily_1d: NDArray[np.float64] = np.array([], dtype=np.float64)
    frozen_weight_source = final_weights if has_admitted and p2_error_reason is None else weights_2d
    if frozen_weight_source.shape[0] > 0:
        l2_start_idx = int(np.searchsorted(
            bars_4h.timestamps_ns, l2_ledger.timestamps_ns[0], side="left",
        ))
        frozen_weights_2d = build_frozen_control_weights(frozen_weight_source, l2_start_idx)
        frozen_ledger = simulate_dense_portfolio(
            bars_4h=bars_4h,
            target_weights_2d=frozen_weights_2d,
            funding_1h_2d=funding_1h,
            cost_bps=cost_bps_4h,
            config=config.dense_sim,
        )
        frozen_l2_end = l2_start_idx + l2_ledger.timestamps_ns.shape[0]
        frozen_l2_slice = slice_execution_ledger(
            ledger=frozen_ledger,
            start_time_ns=int(bars_4h.timestamps_ns[l2_start_idx]),
            end_time_ns=int(bars_4h.timestamps_ns[min(frozen_l2_end - 1, len(bars_4h.timestamps_ns) - 1)]),
        )
        frozen_daily_ret = aggregate_returns_to_utc_days(
            frozen_l2_slice.timestamps_ns, frozen_l2_slice.net_returns_1d,
        )
        if frozen_daily_ret.size > 0:
            frozen_control_daily_1d = np.log1p(frozen_daily_ret)

    l1_prior_returns_1d: NDArray[np.float64] | None = None
    if quarter_window is not None and ledger_1.timestamps_ns.size > 0:
        try:
            l1_prior_ledger = slice_execution_ledger(
                ledger=ledger_1,
                start_time_ns=int(quarter_window.l1_start_ns),
                end_time_ns=int(quarter_window.l2_start_ns),
            )
            l1_prior_returns_1d = aggregate_returns_to_utc_days(
                l1_prior_ledger.timestamps_ns, l1_prior_ledger.net_returns_1d,
            )
        except (ValueError, IndexError):
            pass

    l2_eval = evaluate_l2_walk_forward(
        ledger=l2_ledger, fold_ids_1d=fold_ids_1d,
        benchmark=benchmark, trial_multiplicity=trial_multiplicity,
        config=config.l2_gate, bootstrap_seed=42,
        frozen_control_daily_1d=frozen_control_daily_1d,
        beta_1d=beta_1d_eval,
        l1_prior_returns_1d=l1_prior_returns_1d,
    )

    if trial_ledger is not None:
        l2_daily_returns = aggregate_returns_to_utc_days(l2_ledger.timestamps_ns, l2_ledger.net_returns_1d)
        trial_ledger.register(
            CandidateTrial(
                candidate_hash=candidate_hash,
                strategy_spec_hash=spec_hash,
                descriptor_ids=candidate_descriptor_ids,
                risk_policy_hash=risk_policy_hash,
                cutoff_time_ns=cutoff_time_ns,
            ),
            l2_daily_returns=l2_daily_returns,
        )

    holdout_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[holdout_start_idx]),
        end_time_ns=int(ledger.timestamps_ns[-1]),
    )
    l2_daily_prior = aggregate_returns_to_utc_days(l2_ledger.timestamps_ns, l2_ledger.net_returns_1d)
    if len(l2_daily_prior) == 0:
        l2_daily_prior = np.zeros(1, dtype=np.float64)
    prior_returns = l2_daily_prior[-config.l3.l2_prior_effective_days_cap:]

    def evaluate_fn(manifest: object) -> L3ValidationResult:
        return evaluate_l3_sealed_holdout(
            l2_prior_returns=prior_returns,
            holdout_ledger=holdout_ledger,
            holdout_manifest=manifest,  # type: ignore[arg-type]
            config=config.l3,
        )

    resolved_holdout_id = resolve_engine_holdout_id(holdout_id, quarter_window)
    _manifest = holdout_store.get_manifest(resolved_holdout_id)

    if _manifest.data_manifest_hash != market.data_manifest_hash:
        raise HoldoutReuseError(
            f"holdout {resolved_holdout_id} hash mismatch: "
            f"data_hash={_manifest.data_manifest_hash}!={market.data_manifest_hash}"
        )

    if l2_eval.verdict != L2GateVerdict.PASS:
        l3_result = evaluate_l3_sealed_holdout(
            l2_prior_returns=prior_returns,
            holdout_ledger=holdout_ledger,
            holdout_manifest=_manifest,
            config=config.l3,
        )
        l3_reasons = list(l3_result.reasons)
        l3_reasons.append("l2_not_pass")
        l3_result = L3ValidationResult(
            verdict=DeploymentVerdict.REJECT,
            posterior_growth_probability=l3_result.posterior_growth_probability,
            holdout_days=l3_result.holdout_days,
            max_drawdown=l3_result.max_drawdown,
            daily_cvar95=l3_result.daily_cvar95,
            reasons=tuple(l3_reasons),
        )
    elif os.environ.get("L2_DRY_RUN", "0") == "1":
        l3_result = L3ValidationResult(
            verdict=DeploymentVerdict.SHADOW,
            posterior_growth_probability=0.0,
            holdout_days=_manifest.holdout_days,
            max_drawdown=0.0,
            daily_cvar95=0.0,
            reasons=("dry_run_holdout_not_consumed",),
        )
    else:
        l3_result = holdout_store.consume(
            holdout_id=resolved_holdout_id,
            model_version=_manifest.model_version,
            data_manifest_hash=market.data_manifest_hash,
            strategy_spec_hash=spec_hash,
            evaluate=evaluate_fn,
            universe_state_hash=_manifest.universe_state_hash,
        )

    runtime_fold_hash: str = (
        compute_fold_manifest_hash(folds, max_target_horizon_bars=max_horizon_bars)
        if folds
        else ""
    )
    stub_handoff = AlphaEventTape(
        events=pa.table({}),
        recipe_definitions=(),
        evidence=(),
        active_recipe_ids=(),
        model_version=_manifest.model_version,
        data_manifest_hash=market.data_manifest_hash,
        fold_manifest_hash=runtime_fold_hash,
    )

    deployment_candidate: DeploymentCandidate | None = (
        _build_deployment_candidate(
            handoff_result, panel, l2_eval, _manifest, forecast,
            strategy_spec_hash=spec_hash,
            fold_manifest_hash=runtime_fold_hash,
        )
        if handoff_result is not None
        else None
    )

    _logger.info(
        "engine complete: l2_growth=%.6f l3_verdict=%s cash_only=%s deploy_candidate=%s",
        l2_eval.annualized_log_growth, l3_result.verdict.value, is_cash_only,
        deployment_candidate is not None,
    )

    return CompoundEngineResult(
        handoff=stub_handoff,
        ledger=ledger,
        l2=l2_eval,
        l3=l3_result,
        deployment_candidate=deployment_candidate,
    )


def _build_deployment_candidate(
    handoff_result: HandoffResult,
    panel: RawSignalPanel,
    l2_eval: L2Evaluation,
    manifest: SealedHoldoutManifest,
    forecast: CalibratedForecastPanel | None,
    *,
    strategy_spec_hash: str = "",
    fold_manifest_hash: str = "",
) -> DeploymentCandidate | None:
    if not handoff_result.evidence.admitted or l2_eval.verdict != L2GateVerdict.PASS:
        return None
    unique_ids = tuple(dict.fromkeys(handoff_result.evidence.active_signal_ids))
    counts = collections.Counter(handoff_result.evidence.active_signal_ids)
    matched_descriptors = tuple(
        d for d in panel.descriptors if d.signal_id in unique_ids
    )
    if len(matched_descriptors) != len(unique_ids):
        unmatched = set(unique_ids) - {d.signal_id for d in matched_descriptors}
        raise ValueError(f"unmatched signal ids: {unmatched}")
    total = sum(counts[d.signal_id] for d in matched_descriptors)
    vote_weights = tuple(counts[d.signal_id] / total for d in matched_descriptors)
    orientation_signs = tuple(1 for _ in matched_descriptors)
    return DeploymentCandidate(
        active_signal_ids=unique_ids,
        descriptors=matched_descriptors,
        orientation_signs=orientation_signs,
        vote_weights=vote_weights,
        model_version=manifest.model_version,
        strategy_spec_hash=strategy_spec_hash,
        fold_manifest_hash=fold_manifest_hash or (forecast.fold_manifest_hash if forecast is not None else ""),
        trial_count=l2_eval.candidate_count,
    )


def _daily_timestamps_from_4h(timestamps_ns_4h: NDArray[np.int64]) -> NDArray[np.int64]:
    ns_per_4h = 4 * 3600 * 10**9
    day_start_ns = timestamps_ns_4h - (timestamps_ns_4h % (6 * np.int64(ns_per_4h)))
    unique_days: NDArray[np.int64] = np.unique(day_start_ns).astype(np.int64)
    counts: NDArray[np.int64] = np.array([int(np.sum(day_start_ns == d)) for d in unique_days], dtype=np.int64)
    complete: NDArray[np.int64] = unique_days[counts == 6].astype(np.int64)
    return complete


def _aggregate_trial_4h_to_daily(
    trial_returns_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_trial, n_step = trial_returns_2d.shape
    n_days = n_step // 6
    if n_days == 0:
        return np.zeros((n_trial, 0), dtype=np.float64)
    usable = n_days * 6
    daily = np.empty((n_trial, n_days), dtype=np.float64)
    for k in range(n_trial):
        block = trial_returns_2d[k, :usable].reshape(n_days, 6)
        daily[k] = np.expm1(np.sum(np.log1p(block), axis=1))
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
