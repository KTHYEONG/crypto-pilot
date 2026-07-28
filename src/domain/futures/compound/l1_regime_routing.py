from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.allocator import compute_dynamic_compounding_path
from src.domain.futures.compound.bootstrap import (
    circular_stationary_bootstrap_growth,
    politis_white_block_length,
)
from src.domain.futures.compound.config import (
    DynamicCompoundingConfig,
    RegimeRouterConfig,
)
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CausalClusterFold,
    CausalFold,
    CausalRegimePanel,
    ExpertReturnTape,
    L1SleevePosterior,
    PrequentialExpertRoute,
    RawSignalPanel,
    RegimeExpertEvidence,
    RouteAttribution,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder
from src.domain.futures.compound.l1_sleeves import stress_execution_costs
from src.domain.futures.compound.provenance import compute_fold_manifest_hash

_LOGGER = logging.getLogger(__name__)
_BARS_PER_YEAR_4H: float = 2190.0


def build_causal_regime_panel(
    benchmark_returns_1d: NDArray[np.float64],
    decision_timestamps_ns: NDArray[np.int64],
    config: RegimeRouterConfig,
) -> CausalRegimePanel:
    n = len(benchmark_returns_1d)
    if n != len(decision_timestamps_ns):
        raise ValueError(
            f"benchmark_returns_1d length {n} != decision_timestamps_ns length {len(decision_timestamps_ns)}"
        )

    code_1d = np.zeros(n, dtype=np.int8)
    available_at_ns_1d = np.zeros(n, dtype=np.int64)
    prev_state: int = 0
    dwell: int = 0
    vol_history: list[float] = []

    for t in range(n):
        past = benchmark_returns_1d[:t]
        finite_past = past[np.isfinite(past)]
        n_finite = len(finite_past)

        if n_finite < 42 or t < config.regime_history_bars:
            state = 0
            dwell = 1
        else:
            window = past[-config.regime_history_bars:]
            finite_window = window[np.isfinite(window)]
            current_vol = float(np.std(finite_window, ddof=1))
            vol_history.append(current_vol)

            trend_window = past[-config.trend_lookback_bars:]
            finite_trend = trend_window[np.isfinite(trend_window)]
            if len(finite_trend) < 10:
                trend_t = 0.0
            else:
                m = float(np.mean(finite_trend))
                s = float(np.std(finite_trend, ddof=1))
                trend_t = abs(m) / max(s, 1e-15) * math.sqrt(len(finite_trend))

            stress_enter = False
            stress_exit = False
            if len(vol_history) >= config.regime_history_bars:
                dist = np.array(vol_history[:-1], dtype=np.float64)
                enter_thresh = float(np.percentile(dist, config.stress_enter_quantile * 100))
                exit_thresh = float(np.percentile(dist, config.stress_exit_quantile * 100))
                stress_enter = current_vol > enter_thresh
                stress_exit = current_vol <= exit_thresh

            trend_enter = trend_t >= config.trend_enter_tstat
            trend_exit = trend_t < config.trend_exit_tstat

            dwell_ok = dwell >= config.min_dwell_bars

            if stress_enter and dwell_ok:
                state = 3
            elif prev_state == 3:
                state = (2 if trend_enter else 1) if stress_exit and dwell_ok else 3
            elif trend_enter and dwell_ok:
                state = 2
            elif prev_state == 2:
                state = 1 if trend_exit and dwell_ok else 2
            else:
                state = 1

            if state == prev_state:
                dwell += 1
            else:
                dwell = 1

        code_1d[t] = state
        prev_state = state
        available_at_ns_1d[t] = decision_timestamps_ns[t]

    return CausalRegimePanel(
        decision_timestamps_ns=decision_timestamps_ns,
        code_1d=code_1d,
        available_at_ns_1d=available_at_ns_1d,
        names=("cold", "chop", "trend", "stress"),
    )


def _expert_member_mask(
    sleeves: tuple[L1SleevePosterior, ...],
    signal_id: str,
    fold_id: int,
    n_symbols: int,
) -> NDArray[np.bool_]:
    sig_sleeves = [
        s for s in sleeves
        if s.signal_id == signal_id and s.outer_fold_id == fold_id and s.admitted
    ]
    if not sig_sleeves:
        return np.zeros(n_symbols, dtype=np.bool_)
    mask = np.zeros(n_symbols, dtype=np.bool_)
    for s in sig_sleeves:
        mask |= s.member_mask_1d
    return mask


def build_fold_local_shadow_tape(
    panel: RawSignalPanel,
    sleeves: tuple[L1SleevePosterior, ...],
    folds: tuple[CausalFold, ...],
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    allocator_config: DynamicCompoundingConfig,
    regime_panel: CausalRegimePanel,
) -> ExpertReturnTape:
    t_total, n_symbols = panel.z_3d.shape[0], panel.z_3d.shape[1]

    close = bars_4h.close_2d.astype(np.float64)
    log_ret = np.zeros((t_total, n_symbols), dtype=np.float64)
    for t in range(1, t_total):
        prev = close[t - 1, :n_symbols]
        curr = close[t, :n_symbols]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        log_ret[t, mask] = np.log(curr[mask] / prev[mask])

    tape_decisions: list[NDArray[np.int64]] = []
    tape_executions: list[NDArray[np.int64]] = []
    tape_availables: list[NDArray[np.int64]] = []
    tape_signal_ids: list[NDArray[np.str_]] = []
    tape_fold_ids: list[NDArray[np.int16]] = []
    tape_regime_codes: list[NDArray[np.int8]] = []
    tape_gross: list[NDArray[np.float64]] = []
    tape_cost: list[NDArray[np.float64]] = []
    tape_funding: list[NDArray[np.float64]] = []
    tape_net: list[NDArray[np.float64]] = []

    for fold in folds:
        fold_id = fold.fold_id
        oos_start = fold.oos_start
        oos_end = fold.oos_end_exclusive
        if oos_end - oos_start < 2:
            continue

        signal_ids = sorted({
            s.signal_id for s in sleeves
            if s.outer_fold_id == fold_id and s.admitted
        })
        if not signal_ids:
            continue

        for signal_id in signal_ids:
            sig_sleeves = [
                s for s in sleeves
                if s.signal_id == signal_id and s.outer_fold_id == fold_id and s.admitted
            ]
            if not sig_sleeves:
                continue
            signs = {int(np.sign(s.fitted_beta)) for s in sig_sleeves}
            if 0 in signs or len(signs) != 1:
                continue
            orientation = signs.pop()

            member_mask = _expert_member_mask(sleeves, signal_id, fold_id, n_symbols)
            if not np.any(member_mask):
                continue

            signal_idx = next(
                i for i, d in enumerate(panel.descriptors) if d.signal_id == signal_id
            )
            mu_2d = np.where(
                member_mask.reshape(1, -1),
                orientation * panel.z_3d[:, :, signal_idx].astype(np.float64),
                0.0,
            ).astype(np.float32)

            mini_panel = CalibratedForecastPanel(
                decision_timestamps_ns=panel.decision_timestamps_ns,
                symbols=panel.symbols,
                mu_2d=mu_2d,
                se_2d=np.full((t_total, n_symbols), np.nan, dtype=np.float32),
                family_mu_3d=np.zeros((t_total, n_symbols, 1), dtype=np.float32),
                family_ids=("expert",),
                admitted_signal_ids=(signal_id,),
                fold_manifest_hash="",
            )
            weights = compute_dynamic_compounding_path(
                forecast=mini_panel,
                sigma_2d=panel.sigma_2d,
                funding_rates_1h_2d=funding_1h_2d,
                config=allocator_config,
                close_2d=bars_4h.close_2d,
                cost_bps=1e-8,
            )

            n_oos = oos_end - oos_start
            oos_gross = np.zeros(n_oos, dtype=np.float64)
            oos_cost = np.zeros(n_oos, dtype=np.float64)
            oos_funding = np.zeros(n_oos, dtype=np.float64)
            oos_net = np.zeros(n_oos, dtype=np.float64)

            prev_pos = np.zeros(n_symbols, dtype=np.float64)
            for k, t in enumerate(range(oos_start, oos_end)):
                pos = weights[t]
                if t + 1 < t_total:
                    oos_gross[k] = float(np.dot(pos, log_ret[t + 1]))
                turnover = np.abs(pos - prev_pos)
                oos_cost[k] = -float(np.dot(cost_bps_4h[t], turnover) * 1e-4)
                f_start = t * 4
                f_end = min(f_start + 4, funding_1h_2d.shape[0])
                avg_funding = np.mean(funding_1h_2d[f_start:f_end], axis=0) if f_end > f_start else np.zeros(n_symbols)
                oos_funding[k] = float(np.dot(pos, avg_funding))
                oos_net[k] = oos_gross[k] + oos_cost[k] + oos_funding[k]
                prev_pos = pos

            tape_decisions.append(panel.decision_timestamps_ns[oos_start:oos_end])
            tape_executions.append(bars_4h.timestamps_ns[oos_start:oos_end])
            next_ts = bars_4h.timestamps_ns[np.minimum(np.arange(oos_start, oos_end) + 1, t_total - 1)]
            last_ts = bars_4h.timestamps_ns[t_total - 1]
            avail = np.where(np.arange(n_oos) < t_total - 1 - oos_start, next_ts, last_ts)
            tape_availables.append(avail)
            tape_signal_ids.append(np.array([signal_id] * n_oos, dtype=str))
            tape_fold_ids.append(np.full(n_oos, fold_id, dtype=np.int16))
            tape_regime_codes.append(regime_panel.code_1d[oos_start:oos_end].astype(np.int8))
            tape_gross.append(oos_gross)
            tape_cost.append(oos_cost)
            tape_funding.append(oos_funding)
            tape_net.append(oos_net)

    if not tape_decisions:
        return ExpertReturnTape(
            decision_time_ns_1d=np.zeros(0, dtype=np.int64),
            execution_time_ns_1d=np.zeros(0, dtype=np.int64),
            available_time_ns_1d=np.zeros(0, dtype=np.int64),
            signal_id_1d=np.array([], dtype=str),
            outer_fold_id_1d=np.zeros(0, dtype=np.int16),
            regime_code_1d=np.zeros(0, dtype=np.int8),
            gross_return_1d=np.zeros(0, dtype=np.float64),
            execution_cost_return_1d=np.zeros(0, dtype=np.float64),
            funding_return_1d=np.zeros(0, dtype=np.float64),
            net_return_1d=np.zeros(0, dtype=np.float64),
        )

    return ExpertReturnTape(
        decision_time_ns_1d=np.concatenate(tape_decisions),
        execution_time_ns_1d=np.concatenate(tape_executions),
        available_time_ns_1d=np.concatenate(tape_availables),
        signal_id_1d=np.concatenate(tape_signal_ids),
        outer_fold_id_1d=np.concatenate(tape_fold_ids),
        regime_code_1d=np.concatenate(tape_regime_codes),
        gross_return_1d=np.concatenate(tape_gross),
        execution_cost_return_1d=np.concatenate(tape_cost),
        funding_return_1d=np.concatenate(tape_funding),
        net_return_1d=np.concatenate(tape_net),
    )


def _compute_unconditional_evidence(
    signal_net: NDArray[np.float64],
    signal_gross: NDArray[np.float64],
    signal_cost: NDArray[np.float64],
    signal_funding: NDArray[np.float64],
    config: RegimeRouterConfig,
) -> tuple[float, float, float, bool, list[str]]:
    finite = np.isfinite(signal_net)
    n_finite = int(np.sum(finite))
    reasons: list[str] = []

    if n_finite < config.min_effective_blocks:
        reasons.append("insufficient_samples")
        return 0.0, 0.0, 0.0, False, reasons

    try:
        block_length = politis_white_block_length(signal_net[finite])
    except ValueError:
        block_length = 5.0
    effective_blocks = int(np.floor(n_finite / max(block_length, 1.0)))
    if effective_blocks < config.min_effective_blocks:
        reasons.append("insufficient_effective_blocks")
        return 0.0, 0.0, 0.0, False, reasons

    lcb90, _, prob_positive = circular_stationary_bootstrap_growth(
        signal_net[finite], _BARS_PER_YEAR_4H,
        n_bootstrap=config.n_bootstrap,
        block_size=block_length,
        seed=42,
    )

    pass_all = True
    if lcb90 <= 0.0:
        reasons.append("growth_lcb90_not_positive")
        pass_all = False
    if prob_positive < config.min_posterior_probability:
        reasons.append("posterior_below_threshold")
        pass_all = False

    stressed = stress_execution_costs(signal_gross, signal_cost, signal_funding, 2.0)
    stressed_finite = stressed[np.isfinite(stressed)]
    if len(stressed_finite) > 0:
        growth_2x = float(np.mean(np.log1p(stressed_finite))) * _BARS_PER_YEAR_4H
    else:
        growth_2x = -1e6
    if growth_2x <= 0.0:
        reasons.append("growth_2x_cost_not_positive")
        pass_all = False

    return lcb90, prob_positive, growth_2x, pass_all, reasons


def _compute_temporal_evidence(
    signal_indices: NDArray[np.int64],
    signal_net: NDArray[np.float64],
) -> tuple[int, float, list[str]]:
    n_obs = len(signal_indices)
    reasons: list[str] = []

    if n_obs < 6:
        reasons.append("insufficient_temporal_samples")
        return 0, 0.0, reasons

    block_size_t = n_obs // 3
    block_growths: list[float] = []
    for b in range(3):
        b_start = b * block_size_t
        b_end = n_obs if b == 2 else (b + 1) * block_size_t
        b_net = signal_net[b_start:b_end]
        b_finite = b_net[np.isfinite(b_net)]
        if len(b_finite) < 2:
            block_growths.append(-1e6)
        else:
            g = float(np.mean(np.log1p(b_finite))) * _BARS_PER_YEAR_4H
            block_growths.append(g)
    positive_blocks = sum(1 for g in block_growths if g > 0.0)
    if positive_blocks < 2:
        reasons.append("insufficient_positive_temporal_blocks")
        return positive_blocks, 0.0, reasons

    flat = np.array(block_growths, dtype=np.float64)
    median_g = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median_g)))
    robust_growth = median_g - 1.4826 * mad
    if robust_growth <= 0.0:
        reasons.append("robust_temporal_growth_not_positive")
        return positive_blocks, robust_growth, reasons

    return positive_blocks, robust_growth, reasons


def _build_prequential_expert_route_impl(
    panel: RawSignalPanel,
    sleeves: tuple[L1SleevePosterior, ...],
    folds: tuple[CausalFold, ...],
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    regime_panel: CausalRegimePanel,
    config: RegimeRouterConfig,
    allocator_config: DynamicCompoundingConfig,
) -> PrequentialExpertRoute:
    tape = build_fold_local_shadow_tape(
        panel, sleeves, folds, bars_4h, cost_bps_4h, funding_1h_2d, allocator_config,
        regime_panel,
    )

    t_total, n_symbols = panel.z_3d.shape[0], panel.z_3d.shape[1]

    max_target_horizon_bars = max(
        (d.target_horizon_hours for d in panel.descriptors), default=0
    ) // 4
    fold_manifest_hash: str = ""
    if folds:
        try:
            fold_manifest_hash = compute_fold_manifest_hash(
                folds, max_target_horizon_bars=max_target_horizon_bars,
            )
        except ValueError:
            fold_manifest_hash = ""

    all_evidence: list[RegimeExpertEvidence] = []
    regime_codes_present = {int(c) for c in np.unique(regime_panel.code_1d)} - {0}
    n_hypotheses: int = 0

    mu_2d = np.zeros((t_total, n_symbols), dtype=np.float32)
    active_expert_count_1d = np.zeros(t_total, dtype=np.int16)

    fold_route_scales: dict[int, dict[str, float]] = {}

    unconditional_pass_count = 0
    temporal_pass_count = 0
    regime_pass_count = 0
    active_expert_count = 0
    reason_counts: dict[str, int] = {}

    recorder = L1AdmissionRecorder()

    for fold in folds:
        fold_id = fold.fold_id
        oos_start = fold.oos_start
        oos_end = fold.oos_end_exclusive
        if oos_end - oos_start < 2:
            fold_route_scales[fold_id] = {}
            continue

        if fold_id == 0:
            fold_route_scales[fold_id] = {}
            continue

        evidence_mask = tape.outer_fold_id_1d < fold_id
        if not np.any(evidence_mask):
            fold_route_scales[fold_id] = {}
            continue

        unique_signal_ids = np.unique(tape.signal_id_1d[evidence_mask])
        active_this_fold: dict[str, float] = {}

        for signal_id in unique_signal_ids:
            sig_mask = evidence_mask & (tape.signal_id_1d == signal_id)
            sig_indices = np.where(sig_mask)[0]
            if len(sig_indices) < 3:
                continue

            sig_gross = tape.gross_return_1d[sig_mask]
            sig_cost = tape.execution_cost_return_1d[sig_mask]
            sig_funding = tape.funding_return_1d[sig_mask]
            sig_net = tape.net_return_1d[sig_mask]

            lcb90, prob_positive, growth_2x, unconditional_pass, unconditional_reasons = (
                _compute_unconditional_evidence(
                    sig_net, sig_gross, sig_cost, sig_funding, config,
                )
            )
            n_hypotheses += 1

            if not unconditional_pass:
                for r in unconditional_reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                all_evidence.append(RegimeExpertEvidence(
                    signal_id=signal_id, outer_fold_id=fold_id,
                    regime_code=0, effective_blocks=0,
                    posterior_positive_probability=prob_positive,
                    growth_lcb90=lcb90, growth_2x_cost=growth_2x,
                    robust_inner_growth=0.0, positive_inner_folds=0,
                    scale=0.0, admitted=False,
                    reasons=tuple(unconditional_reasons),
                ))
                continue

            unconditional_pass_count += 1

            positive_blocks, robust_growth, temporal_reasons = _compute_temporal_evidence(
                sig_indices, sig_net,
            )
            if not temporal_reasons:
                temporal_pass_count += 1

            if temporal_reasons:
                for r in temporal_reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                all_evidence.append(RegimeExpertEvidence(
                    signal_id=signal_id, outer_fold_id=fold_id,
                    regime_code=0, effective_blocks=0,
                    posterior_positive_probability=prob_positive,
                    growth_lcb90=lcb90, growth_2x_cost=growth_2x,
                    robust_inner_growth=robust_growth,
                    positive_inner_folds=positive_blocks,
                    scale=0.0, admitted=False,
                    reasons=tuple(temporal_reasons),
                ))
                continue

            for regime_code in regime_codes_present:
                reg_mask = sig_mask & (tape.regime_code_1d == regime_code)
                reg_indices = np.where(reg_mask)[0]
                if len(reg_indices) < 3:
                    all_evidence.append(RegimeExpertEvidence(
                        signal_id=signal_id, outer_fold_id=fold_id,
                        regime_code=regime_code, effective_blocks=0,
                        posterior_positive_probability=prob_positive,
                        growth_lcb90=lcb90, growth_2x_cost=growth_2x,
                        robust_inner_growth=robust_growth,
                        positive_inner_folds=positive_blocks,
                        scale=0.0, admitted=False,
                        reasons=("insufficient_regime_samples",),
                    ))
                    continue

                reg_net = tape.net_return_1d[reg_mask]
                reg_finite = reg_net[np.isfinite(reg_net)]
                reg_blocks = len(reg_finite)

                regime_admitted = reg_blocks >= config.min_effective_blocks
                reg_scale = 1.0 if regime_admitted else 0.0

                if regime_admitted:
                    regime_pass_count += 1

                all_evidence.append(RegimeExpertEvidence(
                    signal_id=signal_id, outer_fold_id=fold_id,
                    regime_code=regime_code,
                    effective_blocks=reg_blocks,
                    posterior_positive_probability=prob_positive,
                    growth_lcb90=lcb90, growth_2x_cost=growth_2x,
                    robust_inner_growth=robust_growth,
                    positive_inner_folds=positive_blocks,
                    scale=reg_scale,
                    admitted=regime_admitted,
                    reasons=() if regime_admitted else ("insufficient_regime_blocks",),
                ))

                if regime_admitted:
                    existing = active_this_fold.get(signal_id, 0.0)
                    active_this_fold[signal_id] = max(existing, reg_scale)

            if recorder.enabled:
                recorder.record_regime_evidence(
                    signal_id=signal_id, outer_fold_id=fold_id,
                    regime_code=0, effective_blocks=0,
                    posterior_probability=prob_positive,
                    growth_lcb90=lcb90, growth_2x_cost=growth_2x,
                    robust_inner_growth=robust_growth,
                    positive_inner_folds=positive_blocks,
                    scale=max(active_this_fold.get(signal_id, 0.0), 0.0),
                    admitted=signal_id in active_this_fold,
                    reasons=(),
                )

        n_active = len(active_this_fold)
        active_expert_count += n_active

        max_cap = 0.50 if n_active <= 1 else 0.50 * n_active
        cap = min(1.0, max_cap)
        total_w = sum(active_this_fold.values())
        if total_w > cap and total_w > 0:
            active_this_fold = {
                k: v * cap / total_w for k, v in active_this_fold.items()
            }

        fold_route_scales[fold_id] = active_this_fold

        if n_active > 0:
            for signal_id, scale in active_this_fold.items():
                signal_idx = next(
                    (i for i, d in enumerate(panel.descriptors) if d.signal_id == signal_id),
                    None,
                )
                if signal_idx is not None:
                    member_mask = _expert_member_mask(sleeves, signal_id, fold_id, n_symbols)
                    if not np.any(member_mask):
                        member_mask = np.ones(n_symbols, dtype=np.bool_)
                    for t_idx in range(oos_start, oos_end):
                        if int(regime_panel.code_1d[t_idx]) != 0:
                            mu_2d[t_idx] += (
                                scale
                                * member_mask.astype(np.float32)
                                * panel.z_3d[t_idx, :, signal_idx].astype(np.float32)
                            )
            active_expert_count_1d[oos_start:oos_end] = n_active

    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        mu_2d=mu_2d,
        se_2d=np.full((t_total, n_symbols), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((t_total, n_symbols, max(len({d.family for d in panel.descriptors}), 1)), dtype=np.float32),
        family_ids=tuple(sorted({d.family for d in panel.descriptors})),
        admitted_signal_ids=tuple(sorted({
            s.signal_id for s in sleeves if s.admitted
        })),
        fold_manifest_hash=fold_manifest_hash,
    )

    attribution = RouteAttribution(
        candidate_experts=len(np.unique(tape.signal_id_1d)) if tape.signal_id_1d.shape[0] > 0 else 0,
        unconditional_pass=unconditional_pass_count,
        temporal_pass=temporal_pass_count,
        regime_pass=regime_pass_count,
        active_experts=active_expert_count,
        reason_counts=reason_counts,
    )

    return PrequentialExpertRoute(
        forecast=forecast,
        tape=tape,
        evidence=tuple(all_evidence),
        attribution=attribution,
        tested_hypotheses=n_hypotheses,
    )


def build_prequential_expert_route(
    panel: RawSignalPanel,
    sleeves: tuple[L1SleevePosterior, ...],
    cluster_folds: tuple[CausalClusterFold, ...],
    folds: tuple[CausalFold, ...],
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    regime_panel: CausalRegimePanel,
    config: RegimeRouterConfig,
    allocator_config: DynamicCompoundingConfig,
) -> PrequentialExpertRoute:
    return _build_prequential_expert_route_impl(
        panel, sleeves, folds, bars_4h, cost_bps_4h, funding_1h_2d,
        regime_panel, config, allocator_config,
    )


build_fold_local_regime_forecast = build_prequential_expert_route
