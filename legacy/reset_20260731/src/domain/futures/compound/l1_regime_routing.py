from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.allocator import (
    compute_dynamic_compounding_path,
    compute_funding_4h_2d,
)
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
    CausalFold,
    CausalRegimePanel,
    ExpertReturnTape,
    L1RoutingSleeve,
    L1SleevePosterior,
    PrequentialExpertRoute,
    RawSignalPanel,
    RegimeExpertEvidence,
    RouteAttribution,
    SignalFoldRecord,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder
from src.domain.futures.compound.l1_sleeves import stress_execution_costs
from src.domain.futures.compound.provenance import compute_fold_manifest_hash

_LOGGER = logging.getLogger(__name__)
_BARS_PER_YEAR_4H: float = 2190.0

_SleeveT = L1RoutingSleeve | L1SleevePosterior


@dataclass(slots=True, frozen=True)
class ExpertContribution:
    signal_id: str
    outer_fold_id: int
    orientation: int
    member_mask_1d: NDArray[np.bool_]
    signal_index: int


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
    sleeves: tuple[_SleeveT, ...],
    signal_id: str,
    fold_id: int,
    n_symbols: int,
) -> NDArray[np.bool_]:
    sig_sleeves = [
        s for s in sleeves
        if s.signal_id == signal_id and s.outer_fold_id == fold_id
    ]
    if not sig_sleeves:
        return np.zeros(n_symbols, dtype=np.bool_)
    mask = np.zeros(n_symbols, dtype=np.bool_)
    for s in sig_sleeves:
        mask |= s.member_mask_1d
    return mask


def collect_fold_expert_contributions(
    panel: RawSignalPanel, sleeves: tuple[_SleeveT, ...],
    fold_id: int, n_symbols: int,
) -> tuple[ExpertContribution, ...]:
    contributions: list[ExpertContribution] = []
    signal_ids = sorted({
        s.signal_id for s in sleeves
        if s.outer_fold_id == fold_id
    })
    for signal_id in signal_ids:
        sig_sleeves = [
            s for s in sleeves
            if s.signal_id == signal_id and s.outer_fold_id == fold_id
        ]
        if not sig_sleeves:
            continue
        descriptor = next(
            (d for d in panel.descriptors if d.signal_id == signal_id), None
        )
        if descriptor is None:
            continue
        orientation = descriptor.declared_orientation
        member_mask = _expert_member_mask(sleeves, signal_id, fold_id, n_symbols)
        if not np.any(member_mask):
            continue
        signal_idx = next(
            i for i, d in enumerate(panel.descriptors) if d.signal_id == signal_id
        )
        contributions.append(ExpertContribution(
            signal_id=signal_id, outer_fold_id=fold_id,
            orientation=orientation, member_mask_1d=member_mask,
            signal_index=signal_idx,
        ))
    return tuple(contributions)


def decompose_expert_gross_contribution(
    weights_2d: NDArray[np.float64],
    contribution_3d: NDArray[np.float64],
    log_ret_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_experts, t_total, n_symbols = contribution_3d.shape
    gross_e = np.zeros((n_experts, t_total), dtype=np.float64)
    abs_contrib = np.abs(contribution_3d)
    total_abs = np.sum(abs_contrib, axis=0).astype(np.float64)
    safe = total_abs > 1e-12
    share = np.zeros((n_experts, t_total, n_symbols), dtype=np.float64)
    for e in range(n_experts):
        share[e] = np.where(safe, abs_contrib[e] / np.maximum(total_abs, 1e-12), 0.0)

    for t in range(1, t_total):
        w_t = weights_2d[t]
        logret_t = log_ret_2d[t]
        book_t = float(np.dot(w_t, logret_t))
        if np.any(safe[t]):
            for e in range(n_experts):
                gross_e[e, t] = book_t * float(np.mean(share[e, t, safe[t]]))
    return gross_e


def blend_expert_contributions(
    z_3d: NDArray[np.float32],
    valid_3d: NDArray[np.bool_],
    signal_weights_1d: NDArray[np.float64],
) -> NDArray[np.float64]:
    if z_3d.ndim != 3 or valid_3d.ndim != 3 or signal_weights_1d.ndim != 1:
        raise ValueError(
            f"shape mismatch: z_3d {z_3d.shape}, valid_3d {valid_3d.shape}, "
            f"weights {signal_weights_1d.shape}"
        )
    if z_3d.shape != valid_3d.shape:
        raise ValueError(f"z_3d shape {z_3d.shape} != valid_3d shape {valid_3d.shape}")
    if z_3d.shape[2] != signal_weights_1d.shape[0]:
        raise ValueError(f"n_signals {z_3d.shape[2]} != weights {signal_weights_1d.shape[0]}")
    n_sigs = z_3d.shape[2]
    numer = np.zeros_like(z_3d[:, :, 0], dtype=np.float64)
    denom = np.zeros_like(z_3d[:, :, 0], dtype=np.float64)
    for k in range(n_sigs):
        w_k = signal_weights_1d[k]
        v_k = valid_3d[:, :, k]
        z_k = z_3d[:, :, k]
        z_safe = np.where(v_k, z_k, 0.0)
        numer += w_k * z_safe
        denom += np.abs(w_k) * v_k
    safe = denom > 0.0
    mu = np.zeros_like(numer)
    np.divide(numer, denom, out=mu, where=safe)
    return mu


def build_fold_expert_books(
    panel: RawSignalPanel, contributions: tuple[ExpertContribution, ...],
    bars_4h: TimeframeBarCube, funding_1h_2d: NDArray[np.float32],
    allocator_config: DynamicCompoundingConfig, cost_bps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t_total, n_symbols = panel.z_3d.shape[0], panel.z_3d.shape[1]
    n_experts = len(contributions)
    if n_experts == 0:
        return np.zeros((0, t_total, n_symbols), dtype=np.float64), np.zeros((t_total, n_symbols), dtype=np.float64)

    contribution_3d = np.zeros((n_experts, t_total, n_symbols), dtype=np.float64)
    for e, c in enumerate(contributions):
        contribution_3d[e] = (
            c.member_mask_1d.astype(np.float64) * c.orientation
            * panel.z_3d[:, :, c.signal_index].astype(np.float64)
        )

    sig_indices = [c.signal_index for c in contributions]
    mu_2d = blend_expert_contributions(
        panel.z_3d[:, :, sig_indices],
        panel.valid_3d[:, :, sig_indices],
        np.array([c.orientation * 1.0 for c in contributions], dtype=np.float64),
    )
    mini_panel = CalibratedForecastPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        mu_2d=mu_2d.astype(np.float32),
        se_2d=np.full((t_total, n_symbols), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((t_total, n_symbols, 1), dtype=np.float32),
        family_ids=("expert",),
        admitted_signal_ids=tuple(c.signal_id for c in contributions),
        fold_manifest_hash="",
    )
    book_weights_2d = compute_dynamic_compounding_path(
        forecast=mini_panel,
        sigma_2d=panel.sigma_2d,
        funding_rates_1h_2d=funding_1h_2d,
        config=allocator_config,
        close_2d=bars_4h.close_2d,
        cost_bps=cost_bps,
    )

    if not np.all(np.isfinite(book_weights_2d)):
        raise ValueError("non-finite allocator output in build_fold_expert_books")

    abs_contrib = np.abs(contribution_3d)
    total_abs = np.sum(abs_contrib, axis=0)
    safe = total_abs > 1e-12
    expert_weights_3d = np.zeros((n_experts, t_total, n_symbols), dtype=np.float64)
    for e in range(n_experts):
        share = np.where(safe, abs_contrib[e] / np.maximum(total_abs, 1e-12), 0.0)
        expert_weights_3d[e] = share * book_weights_2d

    return expert_weights_3d, book_weights_2d


def allocate_book_turnover_cost(
    book_weights_2d: NDArray[np.float64],
    expert_weights_3d: NDArray[np.float64],
    cost_bps_4h: NDArray[np.float32],
    oos_start: int,
    oos_end: int,
) -> NDArray[np.float64]:
    n_experts = expert_weights_3d.shape[0]
    n_oos = oos_end - oos_start
    if n_experts == 0:
        return np.zeros((0, n_oos), dtype=np.float64)

    d_2d = np.zeros((n_oos, expert_weights_3d.shape[2]), dtype=np.float64)
    book_cost_1d = np.zeros(n_oos, dtype=np.float64)
    for k, t in enumerate(range(oos_start, oos_end)):
        book_t = book_weights_2d[t]
        prev_book = book_weights_2d[t - 1] if t > oos_start else np.zeros_like(book_t)
        d = np.abs(book_t - prev_book)
        d_2d[k] = d
        book_cost_1d[k] = -float(np.dot(cost_bps_4h[t], d) * 1e-4)

    expert_cost = np.zeros((n_experts, n_oos), dtype=np.float64)
    abs_contrib = np.abs(expert_weights_3d)
    for k, t in enumerate(range(oos_start, oos_end)):
        d = d_2d[k]
        total_d = float(np.sum(d))
        sum_abs = np.sum(abs_contrib[:, t, :], axis=0)
        safe_sum = sum_abs > 1e-12
        share = np.where(safe_sum, abs_contrib[:, t, :] / np.maximum(sum_abs, 1e-12), 0.0)
        for e in range(n_experts):
            phi = float(np.dot(share[e], d)) / total_d if total_d > 1e-12 and np.any(safe_sum) else 1.0 / n_experts
            expert_cost[e, k] = phi * book_cost_1d[k]

    if np.any(book_cost_1d < -1e-12):
        for t_idx in range(n_oos):
            allocated_sum = float(np.sum(expert_cost[:, t_idx]))
            tol = 1e-12 * max(1.0, abs(book_cost_1d[t_idx]))
            if abs(allocated_sum - book_cost_1d[t_idx]) > tol:
                raise ValueError("book_cost_allocation_identity_violated")

    return expert_cost


def score_expert_returns(
    expert_weights_2d: NDArray[np.float64], asset_return_2d: NDArray[np.float64],
    expert_cost_1d: NDArray[np.float64], funding_4h_2d: NDArray[np.float64],
    oos_start: int, oos_end: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_oos = oos_end - oos_start
    if expert_cost_1d.shape[0] != n_oos:
        raise ValueError(
            f"expert_cost_1d length {expert_cost_1d.shape[0]} != n_oos {n_oos}"
        )
    gross = np.zeros(n_oos, dtype=np.float64)
    funding = np.zeros(n_oos, dtype=np.float64)
    net = np.zeros(n_oos, dtype=np.float64)

    for k, t in enumerate(range(oos_start, oos_end)):
        pos = expert_weights_2d[t]
        if t + 1 < expert_weights_2d.shape[0]:
            gross[k] = float(np.dot(pos, asset_return_2d[t + 1]))
        funding[k] = -float(np.dot(pos, funding_4h_2d[t]))
        net[k] = gross[k] + expert_cost_1d[k] + funding[k]

    if not np.all(np.isfinite(gross)) or not np.all(np.isfinite(expert_cost_1d)) or not np.all(np.isfinite(funding)) or not np.all(np.isfinite(net)):
        raise ValueError("non-finite return components")
    if np.any(net <= -1.0):
        raise ValueError("invalid_expert_return_domain: net return <= -1")

    return gross, expert_cost_1d, funding, net


def apply_walk_forward_carry(
    mu_2d: NDArray[np.float64], panel: RawSignalPanel,
    contributions: tuple[ExpertContribution, ...],
    route_scales: dict[str, float], regime_overlay: dict[int, float],
    regime_code_1d: NDArray[np.int8], deploy_start: int,
) -> int:
    if not route_scales or deploy_start >= mu_2d.shape[0]:
        return 0
    carried = int(mu_2d.shape[0]) - deploy_start
    for t_idx in range(deploy_start, mu_2d.shape[0]):
        code = int(regime_code_1d[t_idx])
        overlay = regime_overlay.get(code, 1.0)
        if code != 0:
            for c in contributions:
                scale = route_scales.get(c.signal_id, 0.0)
                if scale > 0:
                    mu_2d[t_idx] += (
                        scale * overlay
                        * c.member_mask_1d.astype(np.float64)
                        * c.orientation
                        * panel.z_3d[t_idx, :, c.signal_index].astype(np.float64)
                    )
    return carried


def concatenate_signal_evidence(
    history: dict[str, list[SignalFoldRecord]], signal_id: str,
) -> SignalFoldRecord:
    records = history.get(signal_id, [])
    if not records:
        return SignalFoldRecord(
            gross_1d=np.array([], dtype=np.float64),
            cost_1d=np.array([], dtype=np.float64),
            funding_1d=np.array([], dtype=np.float64),
            net_1d=np.array([], dtype=np.float64),
            regime_code_1d=np.array([], dtype=np.int8),
        )
    return SignalFoldRecord(
        gross_1d=np.concatenate([r.gross_1d for r in records]),
        cost_1d=np.concatenate([r.cost_1d for r in records]),
        funding_1d=np.concatenate([r.funding_1d for r in records]),
        net_1d=np.concatenate([r.net_1d for r in records]),
        regime_code_1d=np.concatenate([r.regime_code_1d for r in records]),
    )


def compute_regime_overlay(
    prior: SignalFoldRecord, regime_codes_present: set[int], config: RegimeRouterConfig,
) -> dict[int, float]:
    overlay: dict[int, float] = {}
    for regime_code in regime_codes_present:
        reg_mask = prior.regime_code_1d == regime_code
        reg_indices = np.where(reg_mask)[0]
        if len(reg_indices) < config.min_effective_blocks:
            overlay[regime_code] = 1.0
        else:
            reg_net = prior.net_1d[reg_mask]
            reg_mean_net = float(np.mean(reg_net)) if len(reg_net) > 0 else 0.0
            if reg_mean_net < 0.0:
                overlay[regime_code] = config.regime_overlay_floor
            else:
                overlay[regime_code] = 1.0
    return overlay


def _build_prequential_expert_route_impl(
    panel: RawSignalPanel,
    sleeves: tuple[_SleeveT, ...],
    folds: tuple[CausalFold, ...],
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    regime_panel: CausalRegimePanel,
    config: RegimeRouterConfig,
    allocator_config: DynamicCompoundingConfig,
    cost_bps: float,
) -> PrequentialExpertRoute:
    t_total, n_symbols = panel.z_3d.shape[0], panel.z_3d.shape[1]

    close = bars_4h.close_2d.astype(np.float64)
    asset_return = np.zeros((t_total, n_symbols), dtype=np.float64)
    for t in range(1, t_total):
        prev = close[t - 1]
        curr = close[t]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        asset_return[t, mask] = curr[mask] / prev[mask] - 1.0

    funding_4h_2d = compute_funding_4h_2d(funding_1h_2d, t_total)

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

    signal_evidence_history: dict[str, list[SignalFoldRecord]] = {}
    all_evidence: list[RegimeExpertEvidence] = []
    regime_codes_present = {int(c) for c in np.unique(regime_panel.code_1d)} - {0}
    n_hypotheses: int = 0

    mu_2d = np.zeros((t_total, n_symbols), dtype=np.float64)
    active_expert_count_1d = np.zeros(t_total, dtype=np.int16)

    fold_route_scales: dict[int, dict[str, float]] = {}

    regime_pass_count = 0
    active_expert_count = 0
    reason_counts: dict[str, int] = {}
    carry_contributions: tuple[ExpertContribution, ...] = ()
    final_fold_id: int = 0
    final_regime_overlay: dict[int, float] = {}

    recorder = L1AdmissionRecorder()

    for fold in folds:
        fold_id = fold.fold_id
        oos_start = fold.oos_start
        oos_end = fold.oos_end_exclusive
        if oos_end - oos_start < 2:
            fold_route_scales[fold_id] = {}
            continue

        contributions = collect_fold_expert_contributions(
            panel, sleeves, fold_id, n_symbols,
        )
        if not contributions:
            fold_route_scales[fold_id] = {}
            continue

        expert_weights_3d, book_weights_2d = build_fold_expert_books(
            panel, contributions, bars_4h, funding_1h_2d,
            allocator_config, cost_bps,
        )

        n_experts = len(contributions)
        active_this_fold: dict[str, float] = {}
        fold_regime_overlay: dict[int, float] = {}

        # P0: allocate book turnover cost to experts before scoring
        expert_cost_per_expert_2d = allocate_book_turnover_cost(
            book_weights_2d, expert_weights_3d, cost_bps_4h,
            oos_start, oos_end,
        )

        # P2: score every expert first (unchanged), deferring the admission
        # decision to a single ensemble-level composite test instead of a
        # per-expert 7-reason sequential cascade.
        expert_fold_data: list[tuple[str, SignalFoldRecord, SignalFoldRecord]] = []
        for e in range(n_experts):
            c = contributions[e]
            signal_id = c.signal_id
            expert_w = expert_weights_3d[e]
            expert_cost_1d = expert_cost_per_expert_2d[e]

            try:
                gross, expert_cost, funding_cost, net = score_expert_returns(
                    expert_w, asset_return, expert_cost_1d, funding_4h_2d,
                    oos_start, oos_end,
                )
            except ValueError:
                all_evidence.append(RegimeExpertEvidence(
                    signal_id=signal_id, outer_fold_id=fold_id,
                    regime_code=0, effective_blocks=0,
                    posterior_positive_probability=0.0,
                    growth_lcb90=0.0, growth_2x_cost=0.0,
                    robust_inner_growth=0.0, positive_inner_folds=0,
                    scale=0.0, admitted=False,
                    reasons=("invalid_expert_return_domain",),
                    n_evidence_bars=0,
                ))
                reason_counts["invalid_expert_return_domain"] = (
                    reason_counts.get("invalid_expert_return_domain", 0) + 1
                )
                continue

            # P0: prior from previous folds only
            record = SignalFoldRecord(
                gross_1d=gross, cost_1d=expert_cost,
                funding_1d=funding_cost, net_1d=net,
                regime_code_1d=regime_panel.code_1d[oos_start:oos_end].astype(np.int8),
            )
            prior = concatenate_signal_evidence(signal_evidence_history, signal_id)
            expert_fold_data.append((signal_id, record, prior))

        if expert_fold_data:
            ensemble_signal_ids = tuple(d[0] for d in expert_fold_data)
            ensemble_prior = aggregate_ensemble_evidence(
                signal_evidence_history, ensemble_signal_ids,
            )
            admitted, prob_positive, growth_2x_val, guard_reason = evaluate_ensemble_admission(
                ensemble_prior, config,
            )
            n_evidence_bars = ensemble_prior.net_1d.size

            if guard_reason is None:
                n_hypotheses += 1

            gate_reasons: list[str] = []
            if guard_reason is not None:
                gate_reasons.append(guard_reason)
            elif not admitted:
                if prob_positive < config.min_posterior_probability:
                    gate_reasons.append("growth_probability_below_threshold")
                if growth_2x_val <= 0.0:
                    gate_reasons.append("growth_2x_cost_not_positive")

            for reason in gate_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            for signal_id, record, prior in expert_fold_data:
                if admitted:
                    # P2: regime overlay (no longer gates admission)
                    regime_overlay = compute_regime_overlay(prior, regime_codes_present, config)
                    for code, overlay in regime_overlay.items():
                        fold_regime_overlay[code] = overlay
                        if overlay < 1.0:
                            regime_pass_count += 1
                        all_evidence.append(RegimeExpertEvidence(
                            signal_id=signal_id, outer_fold_id=fold_id,
                            regime_code=code,
                            effective_blocks=int(np.sum(prior.regime_code_1d == code)),
                            posterior_positive_probability=prob_positive,
                            growth_lcb90=0.0, growth_2x_cost=growth_2x_val,
                            robust_inner_growth=0.0,
                            positive_inner_folds=0,
                            scale=overlay,
                            admitted=True,
                            reasons=(),
                            regime_mean_net=float(np.mean(prior.net_1d[prior.regime_code_1d == code])) if np.any(prior.regime_code_1d == code) else 0.0,
                            n_evidence_bars=n_evidence_bars,
                        ))

                    active_this_fold[signal_id] = 1.0

                    if recorder.enabled:
                        recorder.record_regime_evidence(
                            signal_id=signal_id, outer_fold_id=fold_id,
                            regime_code=0, effective_blocks=0,
                            posterior_probability=prob_positive,
                            growth_lcb90=0.0, growth_2x_cost=growth_2x_val,
                            robust_inner_growth=0.0,
                            positive_inner_folds=0,
                            scale=1.0,
                            admitted=True,
                            reasons=(),
                            n_evidence_bars=n_evidence_bars,
                            regime_mean_net=0.0,
                        )
                else:
                    all_evidence.append(RegimeExpertEvidence(
                        signal_id=signal_id, outer_fold_id=fold_id,
                        regime_code=0, effective_blocks=0,
                        posterior_positive_probability=prob_positive,
                        growth_lcb90=0.0, growth_2x_cost=growth_2x_val,
                        robust_inner_growth=0.0, positive_inner_folds=0,
                        scale=0.0, admitted=False,
                        reasons=tuple(gate_reasons),
                        n_evidence_bars=n_evidence_bars,
                    ))

                # Always record fold evidence (regardless of gate result)
                signal_evidence_history.setdefault(signal_id, []).append(record)

        n_active = len(active_this_fold)
        active_expert_count += n_active

        max_cap = config.max_expert_weight if n_active <= 1 else config.max_expert_weight * n_active
        cap = min(1.0, max_cap)
        total_w = sum(active_this_fold.values())
        if total_w > cap and total_w > 0:
            active_this_fold = {
                k: v * cap / total_w for k, v in active_this_fold.items()
            }

        fold_route_scales[fold_id] = active_this_fold

        if n_active > 0:
            for c in contributions:
                scale = active_this_fold.get(c.signal_id, 0.0)
                if scale > 0:
                    for t_idx in range(oos_start, oos_end):
                        code = int(regime_panel.code_1d[t_idx])
                        overlay = fold_regime_overlay.get(code, 1.0)
                        if code != 0:
                            mu_2d[t_idx] += (
                                scale * overlay
                                * c.member_mask_1d.astype(np.float64)
                                * c.orientation
                                * panel.z_3d[t_idx, :, c.signal_index].astype(np.float64)
                            )
            active_expert_count_1d[oos_start:oos_end] = n_active
            carry_contributions = contributions
            final_fold_id = fold_id
            final_regime_overlay = fold_regime_overlay

    deploy_start = max(f.oos_end_exclusive for f in folds) if folds else 0

    if fold_route_scales.get(final_fold_id) and final_regime_overlay:
        apply_walk_forward_carry(
            mu_2d, panel, carry_contributions,
            fold_route_scales[final_fold_id], final_regime_overlay,
            regime_panel.code_1d, deploy_start,
        )

    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        mu_2d=mu_2d.astype(np.float32),
        se_2d=np.full((t_total, n_symbols), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((t_total, n_symbols, max(len({d.family for d in panel.descriptors}), 1)), dtype=np.float32),
        family_ids=tuple(sorted({d.family for d in panel.descriptors})),
        admitted_signal_ids=tuple(sorted({
            s.signal_id for s in sleeves
        })),
        fold_manifest_hash=fold_manifest_hash,
    )

    is_cash_only = bool(
        not any(
            v for fd in fold_route_scales.values()
            for v in fd.values()
        )
    )

    attribution = RouteAttribution(
        candidate_experts=n_hypotheses,
        unconditional_pass=0,
        temporal_pass=0,
        regime_pass=regime_pass_count,
        active_experts=active_expert_count,
        reason_counts=reason_counts,
    )

    return PrequentialExpertRoute(
        forecast=forecast,
        tape=ExpertReturnTape(
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
        ),
        evidence=tuple(all_evidence),
        attribution=attribution,
        tested_hypotheses=n_hypotheses,
        active_expert_count_1d=active_expert_count_1d,
        is_cash_only=is_cash_only,
        fold_route_scales=fold_route_scales,
    )


def build_prequential_expert_route(
    panel: RawSignalPanel,
    sleeves: tuple[_SleeveT, ...],
    folds: tuple[CausalFold, ...],
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    regime_panel: CausalRegimePanel,
    config: RegimeRouterConfig,
    allocator_config: DynamicCompoundingConfig,
    cost_bps: float = 8.0,
) -> PrequentialExpertRoute:
    return _build_prequential_expert_route_impl(
        panel, sleeves, folds, bars_4h, cost_bps_4h, funding_1h_2d,
        regime_panel, config, allocator_config, cost_bps,
    )


def aggregate_ensemble_evidence(
    history: dict[str, list[SignalFoldRecord]],
    signal_ids: tuple[str, ...],
) -> SignalFoldRecord:
    if not signal_ids:
        return SignalFoldRecord(
            gross_1d=np.array([], dtype=np.float64),
            cost_1d=np.array([], dtype=np.float64),
            funding_1d=np.array([], dtype=np.float64),
            net_1d=np.array([], dtype=np.float64),
            regime_code_1d=np.array([], dtype=np.int8),
        )
    records: list[SignalFoldRecord] = []
    for sid in signal_ids:
        rec_list = history.get(sid, [])
        if rec_list:
            records.append(concatenate_signal_evidence(history, sid))
    if not records:
        return SignalFoldRecord(
            gross_1d=np.array([], dtype=np.float64),
            cost_1d=np.array([], dtype=np.float64),
            funding_1d=np.array([], dtype=np.float64),
            net_1d=np.array([], dtype=np.float64),
            regime_code_1d=np.array([], dtype=np.int8),
        )
    ref_len = records[0].net_1d.shape[0]
    for r in records[1:]:
        if r.net_1d.shape[0] != ref_len:
            raise ValueError("ensemble_record_length_mismatch")
    summed_net = np.sum([r.net_1d for r in records], axis=0)
    summed_gross = np.sum([r.gross_1d for r in records], axis=0)
    summed_cost = np.sum([r.cost_1d for r in records], axis=0)
    summed_funding = np.sum([r.funding_1d for r in records], axis=0)
    return SignalFoldRecord(
        gross_1d=summed_gross,
        cost_1d=summed_cost,
        funding_1d=summed_funding,
        net_1d=summed_net,
        regime_code_1d=records[0].regime_code_1d.copy(),
    )


def evaluate_ensemble_admission(
    prior: SignalFoldRecord,
    config: RegimeRouterConfig,
) -> tuple[bool, float, float, str | None]:
    finite = prior.net_1d[np.isfinite(prior.net_1d)]
    n_evidence_bars = int(finite.shape[0])
    guard_reason: str | None = None

    if n_evidence_bars < config.min_evidence_bars:
        guard_reason = "insufficient_evidence"
        return False, 0.0, 0.0, guard_reason

    try:
        block_length = politis_white_block_length(finite)
    except ValueError:
        block_length = 5.0
    n_finite_blocks = int(np.floor(n_evidence_bars / max(block_length, 1.0)))

    if n_finite_blocks < config.min_effective_blocks:
        guard_reason = "insufficient_evidence"
        return False, 0.0, 0.0, guard_reason

    _, _, prob_positive = circular_stationary_bootstrap_growth(
        finite, _BARS_PER_YEAR_4H,
        n_bootstrap=config.n_bootstrap,
        block_size=block_length,
        seed=42,
    )

    growth_significant = prob_positive >= config.min_posterior_probability

    stressed = stress_execution_costs(prior.gross_1d, prior.cost_1d, prior.funding_1d, 2.0)
    stressed_finite = stressed[np.isfinite(stressed)]
    if len(stressed_finite) > 0:
        growth_2x_cost_val = float(np.mean(np.log1p(stressed_finite))) * _BARS_PER_YEAR_4H
    else:
        growth_2x_cost_val = -1e6

    cost_robust = growth_2x_cost_val > 0

    admitted = growth_significant and cost_robust
    return admitted, prob_positive, growth_2x_cost_val, guard_reason


build_fold_local_regime_forecast = build_prequential_expert_route
