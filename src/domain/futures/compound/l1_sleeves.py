"Exit-aware L1 evidence and posterior sleeve handoff."

from __future__ import annotations

import gc
import hashlib
import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CausalClusterFold,
    CausalFold,
    ExitPolicyKind,
    ExitPolicySpec,
    HandoffAdmissionEvidence,
    HandoffResult,
    L1SleevePosterior,
    MultiTimeframeBars,
    PrecomputedExitPaths,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.provenance import compute_fold_manifest_hash
from src.domain.futures.forecast.contracts import ExitPathRequest
from src.domain.futures.forecast.exit_path import label_exit_paths

_LOGGER = logging.getLogger(__name__)


def _atr(bars: TimeframeBarCube) -> NDArray[np.float64]:
    high = bars.high_2d.astype(np.float64)
    low = bars.low_2d.astype(np.float64)
    close = bars.close_2d.astype(np.float64)
    previous = np.vstack([close[:1], close[:-1]])
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    result = np.full_like(true_range, np.nan)
    alpha = 1.0 / 14.0
    state = np.full(close.shape[1], np.nan)
    for index in range(close.shape[0]):
        row = true_range[index]
        finite = np.isfinite(row)
        initial = finite & ~np.isfinite(state)
        state[initial] = row[initial]
        update = finite & np.isfinite(state)
        state[update] = alpha * row[update] + (1.0 - alpha) * state[update]
        result[index] = state
    return np.maximum(result, 1e-8)


def _policy_hash(descriptor: SignalDescriptor, stop: float | None, target: float | None, horizon: int) -> str:
    payload = f"{descriptor.signal_id}|{descriptor.candidate_version}|{stop}|{target}|{horizon}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def calibrate_exit_policy(
    descriptor: SignalDescriptor,
    oriented_score_2d: NDArray[np.float32],
    bars_4h: TimeframeBarCube,
    fit_slice: slice,
    inner_folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    config: HandoffConfig,
) -> ExitPolicySpec:
    "Calibrate an exit from fit-only excursions, with causal time fallback."
    del funding_1h_2d, config
    fit_end = fit_slice.stop if fit_slice.stop is not None else oriented_score_2d.shape[0]
    horizon = max(int(descriptor.target_horizon_hours // 4), 1)
    if len(inner_folds) < 4 or fit_end <= horizon + 1:
        return ExitPolicySpec(f"{descriptor.signal_id}:time", ExitPolicyKind.TIME, None, None, None, 0, horizon, -1, _policy_hash(descriptor, None, None, horizon))
    score = oriented_score_2d[: max(fit_end - horizon, 0)]
    events = np.argwhere(np.isfinite(score) & (np.abs(score) >= 1.0))
    if events.shape[0] < 200:
        return ExitPolicySpec(f"{descriptor.signal_id}:time", ExitPolicyKind.TIME, None, None, None, 0, horizon, -1, _policy_hash(descriptor, None, None, horizon))
    atr = _atr(bars_4h)
    decisions = events[:, 0].astype(np.int64)
    symbols = events[:, 1].astype(np.int64)
    request = ExitPathRequest(
        decision_idx=decisions, entry_idx=decisions + 1,
        side=np.where(oriented_score_2d[decisions, symbols] >= 0.0, 1, -1).astype(np.int8),
        horizon_bars=np.full(events.shape[0], horizon, dtype=np.int64),
        stop_atr_mult=np.ones(events.shape[0], dtype=np.float64),
        target_atr_mult=np.full(events.shape[0], 2.0, dtype=np.float64),
        min_hold_bars=np.ones(events.shape[0], dtype=np.int64), symbol_idx=symbols,
        open_2d=bars_4h.open_2d.astype(np.float64), high_2d=bars_4h.high_2d.astype(np.float64),
        low_2d=bars_4h.low_2d.astype(np.float64), close_2d=bars_4h.close_2d.astype(np.float64),
        atr_2d=atr, cost_bps_2d=cost_bps_4h.astype(np.float64),
        funding_2d=np.zeros_like(bars_4h.close_2d, dtype=np.float64),
        cost_floor_bps=np.full(events.shape[0], np.nan, dtype=np.float64),
        hurdle_bps=np.zeros(events.shape[0], dtype=np.float64), taker_round_trip_bps=0.0,
    )
    labels = label_exit_paths(request)
    winners = np.isfinite(labels.edge_bps) & (labels.edge_bps > 0.0)
    if int(winners.sum()) < 200:
        kind = ExitPolicyKind.TIME
        stop, target = None, None
    else:
        mae_r = np.maximum(-labels.mae_bps[winners] / 100.0, 0.0)
        mfe_r = np.maximum(labels.mfe_bps[winners] / 100.0, 0.0)
        stop = float(np.clip(np.quantile(mae_r, 0.80), 0.75, 2.50))
        target = float(np.clip(max(np.quantile(mfe_r, 0.50), 1.25 * stop), 1.00, 5.00))
        kind = ExitPolicyKind.ASYMMETRIC_ATR
    return ExitPolicySpec(f"{descriptor.signal_id}:{kind.value}", kind, stop, target, None, 0, horizon, inner_folds[-1].fold_id, _policy_hash(descriptor, stop, target, horizon))


def precompute_exit_paths(
    descriptor: SignalDescriptor,
    oriented_score_2d: NDArray[np.float32],
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
) -> PrecomputedExitPaths:
    """Label all causal paths once for one signal orientation."""
    horizon = max(int(descriptor.target_horizon_hours // 4), 1)
    n_bars = bars_4h.timestamps_ns.size
    score = oriented_score_2d[: max(n_bars - horizon, 0)]
    events = np.argwhere(np.isfinite(score) & (np.abs(score) >= 1.0))
    decisions = events[:, 0].astype(np.int64) if events.size > 0 else np.zeros(0, dtype=np.int64)
    symbols = events[:, 1].astype(np.int64) if events.size > 0 else np.zeros(0, dtype=np.int64)
    orientation_sign = int(np.sign(np.nanmean(oriented_score_2d)) or 1)

    if events.shape[0] == 0:
        return PrecomputedExitPaths(
            decision_idx=np.zeros(0, dtype=np.int64),
            edge_bps=np.zeros(0, dtype=np.float64),
            mae_bps=np.zeros(0, dtype=np.float64),
            mfe_bps=np.zeros(0, dtype=np.float64),
            horizon_bars=horizon,
            orientation_sign=orientation_sign,
        )

    atr = _atr(bars_4h)
    request = ExitPathRequest(
        decision_idx=decisions, entry_idx=decisions + 1,
        side=np.where(oriented_score_2d[decisions, symbols] >= 0.0, 1, -1).astype(np.int8),
        horizon_bars=np.full(events.shape[0], horizon, dtype=np.int64),
        stop_atr_mult=np.ones(events.shape[0], dtype=np.float64),
        target_atr_mult=np.full(events.shape[0], 2.0, dtype=np.float64),
        min_hold_bars=np.ones(events.shape[0], dtype=np.int64), symbol_idx=symbols,
        open_2d=bars_4h.open_2d.astype(np.float64), high_2d=bars_4h.high_2d.astype(np.float64),
        low_2d=bars_4h.low_2d.astype(np.float64), close_2d=bars_4h.close_2d.astype(np.float64),
        atr_2d=atr, cost_bps_2d=cost_bps_4h.astype(np.float64),
        funding_2d=np.zeros_like(bars_4h.close_2d, dtype=np.float64),
        cost_floor_bps=np.full(events.shape[0], np.nan, dtype=np.float64),
        hurdle_bps=np.zeros(events.shape[0], dtype=np.float64), taker_round_trip_bps=0.0,
    )
    labels = label_exit_paths(request)

    return PrecomputedExitPaths(
        decision_idx=decisions,
        edge_bps=labels.edge_bps.astype(np.float64),
        mae_bps=labels.mae_bps.astype(np.float64),
        mfe_bps=labels.mfe_bps.astype(np.float64),
        horizon_bars=horizon,
        orientation_sign=orientation_sign,
    )


def calibrate_exit_policy_from_paths(
    descriptor: SignalDescriptor,
    paths: PrecomputedExitPaths,
    *,
    fit_end_exclusive: int,
    calibration_fold_id: int,
) -> ExitPolicySpec:
    """Select a fit-only policy by causal event slicing."""
    horizon = paths.horizon_bars
    causal_boundary = fit_end_exclusive - horizon
    used = paths.decision_idx < causal_boundary
    n_used = int(np.sum(used))
    max_idx = int(np.max(paths.decision_idx[used])) if n_used > 0 else -1
    if max_idx >= 0:
        assert max_idx < causal_boundary, (
            f"max decision_idx {max_idx} >= fit_end_exclusive - horizon ({causal_boundary})"
        )

    if n_used < 200:
        return ExitPolicySpec(f"{descriptor.signal_id}:time", ExitPolicyKind.TIME, None, None, None, 0, horizon, calibration_fold_id, _policy_hash(descriptor, None, None, horizon))

    edges = paths.edge_bps[used]
    maes = paths.mae_bps[used]
    mfes = paths.mfe_bps[used]
    winners = np.isfinite(edges) & (edges > 0.0)
    n_winners = int(np.sum(winners))

    if n_winners < 200:
        return ExitPolicySpec(f"{descriptor.signal_id}:time", ExitPolicyKind.TIME, None, None, None, 0, horizon, calibration_fold_id, _policy_hash(descriptor, None, None, horizon))

    mae_r = np.maximum(-maes[winners] / 100.0, 0.0)
    mfe_r = np.maximum(mfes[winners] / 100.0, 0.0)
    stop = float(np.clip(np.quantile(mae_r, 0.80), 0.75, 2.50))
    target = float(np.clip(max(np.quantile(mfe_r, 0.50), 1.25 * stop), 1.00, 5.00))
    kind = ExitPolicyKind.ASYMMETRIC_ATR
    return ExitPolicySpec(f"{descriptor.signal_id}:{kind.value}", kind, stop, target, None, 0, horizon, calibration_fold_id, _policy_hash(descriptor, stop, target, horizon))


def _signal_evidence(
    feature: NDArray[np.float32],
    close: NDArray[np.float32],
    descriptor: SignalDescriptor,
    fit_end: int,
) -> tuple[float, float, float, int]:
    horizon = max(descriptor.target_horizon_hours // 4, 1)
    if fit_end <= horizon + 2:
        return 0.0, 1.0, 0.5, 0
    future = np.roll(close.astype(np.float64), -horizon, axis=0) / np.maximum(close, 1e-12) - 1.0
    future[-horizon:] = np.nan
    x = feature[: fit_end - horizon].astype(np.float64)
    y = future[: fit_end - horizon]
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[mask], y[mask]
    denom = float(np.dot(x_valid, x_valid)) + 1e-8
    beta = float(np.dot(x_valid, y_valid) / denom) if x_valid.size else 0.0
    residual = y_valid - beta * x_valid
    se = float(np.std(residual, ddof=1) / math.sqrt(denom)) if residual.size > 1 else 1.0
    probability = float(0.5 * (1.0 + math.erf(beta / max(se, 1e-12) / math.sqrt(2.0))))
    return beta, max(se, 1e-8), probability, int(x_valid.size)


def estimate_sleeve_posteriors(
    panel: RawSignalPanel,
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    config: HandoffConfig,
) -> tuple[L1SleevePosterior, ...]:
    "Estimate fit-only economic posterior for one winning exit per signal."
    if panel.z_3d.shape[:2] != bars_4h.close_2d.shape or cost_bps_4h.shape != bars_4h.close_2d.shape:
        raise ValueError("panel, bars, and cost shapes must agree")
    if not folds:
        return ()
    output: list[L1SleevePosterior] = []
    future_cache: dict[int, NDArray[np.float64]] = {}
    for index, descriptor in enumerate(panel.descriptors):
        horizon = max(descriptor.target_horizon_hours // 4, 1)
        future = future_cache.setdefault(horizon, np.roll(bars_4h.close_2d.astype(np.float64), -horizon, axis=0) / np.maximum(bars_4h.close_2d, 1e-12) - 1.0)
        future[-horizon:] = np.nan
        fold_returns: list[float] = []
        betas: list[float] = []
        ses: list[float] = []
        for fold in folds:
            beta, se, _, observations = _signal_evidence(panel.z_3d[:, :, index], bars_4h.close_2d, descriptor, fold.fit_end_exclusive)
            if observations == 0:
                continue
            betas.append(beta)
            ses.append(se)
            values = beta * panel.z_3d[fold.oos_start:fold.oos_end_exclusive, :, index].astype(np.float64) * future[fold.oos_start:fold.oos_end_exclusive]
            values = values[np.isfinite(values)]
            fold_returns.append(float(np.mean(values)) if values.size else 0.0)
        fit_end = folds[0].fit_end_exclusive
        beta = float(np.mean(betas)) if betas else 0.0
        oriented = (np.sign(beta) * panel.z_3d[:, :, index]).astype(np.float32)
        policy = calibrate_exit_policy(descriptor, oriented, bars_4h, slice(0, fit_end), folds, cost_bps_4h, funding_1h_2d, config)
        mean = float(np.mean(fold_returns)) if fold_returns else 0.0
        se = max(float(np.std(fold_returns, ddof=1) / math.sqrt(len(fold_returns))) if len(fold_returns) > 1 else 1.0, 1e-8)
        probability = float(0.5 * (1.0 + math.erf(mean / se / math.sqrt(2.0))))
        n_symbols = len(panel.symbols)
        member_mask = np.zeros(n_symbols, dtype=np.bool_)
        member_mask[:] = True
        import hashlib
        member_hash = hashlib.sha256(f"all:{descriptor.signal_id}".encode()).hexdigest()[:16]
        output.append(L1SleevePosterior(
            f"{descriptor.signal_id}:{policy.policy_id}", descriptor.signal_id, descriptor.family,
            -1, -1, member_mask, member_hash,
            policy, mean, se, probability, 1.0, tuple(fold_returns), len(fold_returns),
            probability >= 0.65 and sum(value > 0.0 for value in fold_returns) >= 4,
            () if probability >= 0.65 else ("posterior_below_floor",),
        ))
    return tuple(output)


def select_non_redundant_signals(
    panel: RawSignalPanel,
    signal_ids: tuple[str, ...],
    *,
    fit_end_exclusive: int,
    rho_threshold: float = 0.90,
    min_observations: int = 1_000,
) -> tuple[str, ...]:
    if fit_end_exclusive <= 0 or fit_end_exclusive > panel.z_3d.shape[0]:
        raise ValueError(
            f"fit_end_exclusive={fit_end_exclusive} out of range [1, {panel.z_3d.shape[0]}]"
        )
    if not signal_ids:
        return ()

    desc_map = {d.signal_id: d for d in panel.descriptors}
    idx_map: dict[str, int] = {}
    for i, d in enumerate(panel.descriptors):
        idx_map[d.signal_id] = i

    survivor = list(signal_ids)
    z_fit = panel.z_3d[:fit_end_exclusive]
    valid_fit = panel.valid_3d[:fit_end_exclusive]

    removed: set[str] = set()
    i = 0
    while i < len(survivor):
        if survivor[i] in removed:
            i += 1
            continue
        j = i + 1
        while j < len(survivor):
            if survivor[j] in removed or survivor[i] not in idx_map or survivor[j] not in idx_map:
                j += 1
                continue
            zi = z_fit[:, :, idx_map[survivor[i]]].ravel().astype(np.float64)
            zj = z_fit[:, :, idx_map[survivor[j]]].ravel().astype(np.float64)
            vi = valid_fit[:, :, idx_map[survivor[i]]].ravel()
            vj = valid_fit[:, :, idx_map[survivor[j]]].ravel()
            valid = vi & vj & np.isfinite(zi) & np.isfinite(zj)
            n_valid = int(valid.sum())
            if n_valid < min_observations:
                j += 1
                continue
            rho = float(np.corrcoef(zi[valid], zj[valid])[0, 1])
            if abs(rho) >= rho_threshold:
                hi_a = desc_map[survivor[i]].target_horizon_hours
                hi_b = desc_map[survivor[j]].target_horizon_hours
                if hi_a < hi_b:
                    removed.add(survivor[i])
                    break
                elif hi_b < hi_a:
                    removed.add(survivor[j])
                    j += 1
                else:
                    if survivor[i] > survivor[j]:
                        removed.add(survivor[i])
                        break
                    else:
                        removed.add(survivor[j])
                        j += 1
            else:
                j += 1
        if survivor[i] in removed:
            survivor = [s for s in survivor if s not in removed]
            i = 0
            continue
        i += 1

    return tuple(s for s in signal_ids if s not in removed)


def combine_posterior_sleeves(
    panel: RawSignalPanel,
    sleeves: tuple[L1SleevePosterior, ...],
    cluster_folds: tuple[CausalClusterFold, ...],
    folds: tuple[CausalFold, ...],
    config: HandoffConfig,
) -> CalibratedForecastPanel:
    t_total, n_symbols, _ = panel.z_3d.shape

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

    def empty() -> CalibratedForecastPanel:
        return CalibratedForecastPanel(
            panel.decision_timestamps_ns, panel.symbols,
            np.zeros((t_total, n_symbols), dtype=np.float32),
            np.full((t_total, n_symbols), np.nan, dtype=np.float32),
            np.zeros((t_total, n_symbols, 1), dtype=np.float32), (), (),
            fold_manifest_hash,
        )

    active = [s for s in sleeves if s.admitted]
    if not active:
        return empty()

    dedup_ids = select_non_redundant_signals(
        panel,
        tuple(sorted({s.signal_id for s in active})),
        fit_end_exclusive=folds[0].fit_end_exclusive,
        rho_threshold=config.dedup_rho_threshold,
        min_observations=config.min_dedup_observations,
    )

    if not dedup_ids:
        return empty()

    families = sorted({s.family for s in active if s.signal_id in dedup_ids})
    all_mu_3d: list[NDArray[np.float32]] = []
    mu_sum = np.zeros((t_total, n_symbols), dtype=np.float64)

    for sig_id in dedup_ids:
        sig_sleeves = [s for s in active if s.signal_id == sig_id]
        signal_idx = next(i for i, d in enumerate(panel.descriptors) if d.signal_id == sig_id)
        member_mask = np.zeros(n_symbols, dtype=bool)
        for s in sig_sleeves:
            member_mask |= s.member_mask_1d
        sig_mu = panel.z_3d[:, :, signal_idx].astype(np.float64)
        sig_mu = np.where(member_mask.reshape(1, -1), sig_mu, 0.0)
        mu_sum += sig_mu

    n_surviving = len(dedup_ids)
    mu = (mu_sum / n_surviving).astype(np.float32)

    for family in families:
        fam_sigs = [s for s in dedup_ids if any(
            a.family == family for a in active if a.signal_id == s
        )]
        fam_mu = np.zeros((t_total, n_symbols), dtype=np.float64)
        for sig_id in fam_sigs:
            sig_sleeves = [s for s in active if s.signal_id == sig_id]
            signal_idx = next(i for i, d in enumerate(panel.descriptors) if d.signal_id == sig_id)
            member_mask = np.zeros(n_symbols, dtype=bool)
            for s in sig_sleeves:
                member_mask |= s.member_mask_1d
            s_mu = panel.z_3d[:, :, signal_idx].astype(np.float64)
            s_mu = np.where(member_mask.reshape(1, -1), s_mu, 0.0)
            fam_mu += s_mu / n_surviving
        all_mu_3d.append(fam_mu.astype(np.float32))

    mu_3d = np.stack(all_mu_3d, axis=2) if all_mu_3d else np.zeros((t_total, n_symbols, 0), dtype=np.float32)
    return CalibratedForecastPanel(
        panel.decision_timestamps_ns, panel.symbols,
        mu, np.full((t_total, n_symbols), np.nan, dtype=np.float32),
        mu_3d, tuple(families), tuple(dedup_ids),
        fold_manifest_hash,
    )


def build_exit_aware_handoff(
    panel: RawSignalPanel,
    bars: MultiTimeframeBars,
    folds: tuple[CausalFold, ...],
    cluster_folds: tuple[CausalClusterFold, ...],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    config: HandoffConfig,
) -> HandoffResult:
    sleeves = estimate_cluster_sleeve_posteriors(panel, bars.cubes["4h"], cluster_folds, folds, cost_bps_4h, funding_1h_2d, config)
    forecast = combine_posterior_sleeves(panel, sleeves, cluster_folds, folds, config)

    admitted_sleeves = [s for s in sleeves if s.admitted]
    if not admitted_sleeves:
        no_evidence = HandoffAdmissionEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, (), False, ("no_admitted_sleeves",))
        _LOGGER.info("[L1] exit-aware handoff: no admitted sleeves, NO_EVIDENCE")
        return HandoffResult(forecast, no_evidence)

    fold_returns_list: list[float] = []
    for sleeve in admitted_sleeves:
        fold_returns_list.extend(sleeve.fold_net_returns)
    fold_returns = np.asarray(fold_returns_list, dtype=np.float64)
    growth = float(np.mean(fold_returns) * 2191.5) if fold_returns.size else 0.0
    positive = int(np.sum(fold_returns > 0.0)) if fold_returns.size else 0

    reasons: list[str] = []
    if growth <= 0.0:
        reasons.append("growth_lcb90_not_positive")
    if positive < config.min_positive_outer_folds:
        reasons.append("positive_folds_below_floor")

    admitted = not reasons
    evidence = HandoffAdmissionEvidence(
        growth, growth, growth, 0.0, 0.0, positive, float(len(admitted_sleeves)),
        tuple(s.signal_id for s in admitted_sleeves), admitted, tuple(reasons),
    )
    _LOGGER.info("[L1] exit-aware handoff admitted=%s sleeves=%d", admitted, len(admitted_sleeves))
    return HandoffResult(forecast, evidence)


def aggregate_cluster_group_returns(
    returns_2d: NDArray[np.float64],
    sigma_2d: NDArray[np.float64],
    winsorize_pct: float = 0.10,
) -> NDArray[np.float64]:
    if returns_2d.shape[0] == 0 or returns_2d.shape[1] == 0:
        return np.zeros(returns_2d.shape[0], dtype=np.float64)

    t = returns_2d.shape[0]
    result = np.zeros(t, dtype=np.float64)
    for i in range(t):
        row_r = returns_2d[i]
        row_s = sigma_2d[i]
        finite = np.isfinite(row_r) & np.isfinite(row_s) & (row_s > 0)
        n_finite = int(np.sum(finite))
        if n_finite == 0:
            result[i] = 0.0
            continue

        values = row_r[finite]
        if winsorize_pct > 0.0 and n_finite >= 3:
            lower = float(np.percentile(values, winsorize_pct * 100.0))
            upper = float(np.percentile(values, (1.0 - winsorize_pct) * 100.0))
            values = np.clip(values, lower, upper)

        weights = 1.0 / np.maximum(row_s[finite], 1e-12)
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            result[i] = float(np.sum(weights * values)) / w_sum
    return result


def _cluster_masked_beta(
    feature: NDArray[np.float32],
    close: NDArray[np.float32],
    descriptor: SignalDescriptor,
    fit_end: int,
    sym_indices: NDArray[np.int64],
) -> tuple[float, float, float, int]:
    horizon = max(descriptor.target_horizon_hours // 4, 1)
    if fit_end <= horizon + 2:
        return 0.0, 1.0, 0.5, 0
    future = np.roll(close.astype(np.float64), -horizon, axis=0) / np.maximum(close, 1e-12) - 1.0
    future[-horizon:] = np.nan
    x = feature[: fit_end - horizon, sym_indices].astype(np.float64)
    y = future[: fit_end - horizon, sym_indices]
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[mask], y[mask]
    denom = float(np.dot(x_valid, x_valid)) + 1e-8
    beta = float(np.dot(x_valid, y_valid) / denom) if x_valid.size else 0.0
    residual = y_valid - beta * x_valid
    se = float(np.std(residual, ddof=1) / math.sqrt(denom)) if residual.size > 1 else 1.0
    probability = float(0.5 * (1.0 + math.erf(beta / max(se, 1e-12) / math.sqrt(2.0))))
    return beta, max(se, 1e-8), probability, int(x_valid.size)


def estimate_cluster_sleeve_posteriors(
    panel: RawSignalPanel,
    bars_4h: TimeframeBarCube,
    cluster_folds: tuple[CausalClusterFold, ...],
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    config: HandoffConfig,
) -> tuple[L1SleevePosterior, ...]:
    if panel.z_3d.shape[:2] != bars_4h.close_2d.shape or cost_bps_4h.shape != bars_4h.close_2d.shape:
        raise ValueError("panel, bars, and cost shapes must agree")
    if not folds or not panel.descriptors:
        return ()
    n_symbols = len(panel.symbols)
    output: list[L1SleevePosterior] = []
    future_cache: dict[int, NDArray[np.float64]] = {}

    for descriptor in panel.descriptors:
        horizon = max(descriptor.target_horizon_hours // 4, 1)
        future = future_cache.setdefault(
            horizon,
            np.roll(bars_4h.close_2d.astype(np.float64), -horizon, axis=0)
            / np.maximum(bars_4h.close_2d, 1e-12)
            - 1.0,
        )
        future[-horizon:] = np.nan
        signal_idx = panel.descriptors.index(descriptor)
        signal_z = panel.z_3d[:, :, signal_idx]

        # Step 1: Cheap candidate-first posterior gate
        preliminary_viable: list[Any] = []
        for cf in cluster_folds:
            fold = next(f for f in folds if f.fold_id == cf.fold_id)
            cluster_panel = cf.panel
            unique_clusters = sorted(int(x) for x in np.unique(cluster_panel.cluster_labels))

            for cluster_id in unique_clusters:
                sym_mask = cluster_panel.cluster_labels == cluster_id
                sym_indices = np.where(sym_mask)[0]
                if len(sym_indices) < 2:
                    continue

                beta, se, probability, observations = _cluster_masked_beta(
                    signal_z, bars_4h.close_2d, descriptor,
                    fold.fit_end_exclusive, sym_indices,
                )
                if observations == 0:
                    continue

                oos_slice = slice(fold.oos_start, fold.oos_end_exclusive)
                cluster_z = signal_z[oos_slice][:, sym_indices].astype(np.float64)
                cluster_future = future[oos_slice][:, sym_indices]
                cluster_sigma = panel.sigma_2d[oos_slice][:, sym_indices].astype(np.float64)
                raw_values = beta * cluster_z * cluster_future
                aggregated = aggregate_cluster_group_returns(raw_values, cluster_sigma, 0.10)
                aggregated = aggregated[np.isfinite(aggregated)]
                fold_return = float(np.mean(aggregated)) if aggregated.size else 0.0

                admitted = probability >= 0.65
                preliminary_viable.append({
                    "cf": cf, "fold": fold, "cluster_id": cluster_id,
                    "sym_mask": sym_mask, "sym_indices": sym_indices,
                    "beta": beta, "se": se, "probability": probability,
                    "observations": observations, "fold_return": fold_return,
                    "admitted": admitted,
                })

        # Step 2: Precompute exit paths once per descriptor if any viable cluster
        oriented = (np.sign(np.mean([p["beta"] for p in preliminary_viable if p["admitted"]])) * signal_z
                    if any(p["admitted"] for p in preliminary_viable) else signal_z).astype(np.float32)
        paths: PrecomputedExitPaths | None = None
        if any(p["admitted"] for p in preliminary_viable):
            paths = precompute_exit_paths(descriptor, oriented, bars_4h, cost_bps_4h)

        # Step 3: Build sleeves with exit policy
        for p in preliminary_viable:
            sleeve_id = f"{descriptor.signal_id}:fold{p['cf'].fold_id}:cluster_{p['cluster_id']}"
            member_mask = np.zeros(n_symbols, dtype=np.bool_)
            member_mask[p["sym_indices"]] = True

            if p["admitted"] and paths is not None:
                exit_policy = calibrate_exit_policy_from_paths(
                    descriptor, paths,
                    fit_end_exclusive=p["fold"].fit_end_exclusive,
                    calibration_fold_id=p["cf"].fold_id,
                )
            else:
                exit_policy = ExitPolicySpec(
                    f"{descriptor.signal_id}:time", ExitPolicyKind.TIME,
                    None, None, None, 0,
                    max(int(descriptor.target_horizon_hours // 4), 1),
                    p["cf"].fold_id,
                    _policy_hash(descriptor, None, None, max(int(descriptor.target_horizon_hours // 4), 1)),
                )

            reasons: tuple[str, ...] = ()
            if not p["admitted"]:
                reasons = ("posterior_below_floor",)

            output.append(L1SleevePosterior(
                sleeve_id=sleeve_id,
                signal_id=descriptor.signal_id,
                family=descriptor.family,
                outer_fold_id=p["cf"].fold_id,
                cluster_id=p["cluster_id"],
                member_mask_1d=member_mask,
                member_hash=p["cf"].member_hash,
                exit_policy=exit_policy,
                mean_net_return=p["fold_return"],
                standard_error=max(p["se"], 1e-8),
                posterior_positive_probability=p["probability"],
                residual_novelty=1.0,
                fold_net_returns=(p["fold_return"],),
                effective_events=p["observations"],
                admitted=p["admitted"],
                reasons=reasons,
            ))

        # Step 4: Release descriptor-scoped path arrays
        paths = None
        gc.collect()

    return tuple(output)
