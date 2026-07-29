"Exit-aware L1 evidence and posterior sleeve handoff."

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CausalClusterFold,
    CausalFold,
    ExitPathCache,
    ExitPolicyKind,
    ExitPolicySpec,
    FamilyEdgeScreen,
    HandoffAdmissionEvidence,
    HandoffResult,
    L1RoutingSleeve,
    L1SleevePosterior,
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
    atr: NDArray[np.float64] | None = None,
) -> PrecomputedExitPaths:
    """Label all causal paths once for one signal orientation."""
    horizon = max(int(descriptor.target_horizon_hours // 4), 1)
    n_bars = bars_4h.timestamps_ns.size
    score = oriented_score_2d[: max(n_bars - horizon, 0)]
    events = np.argwhere(np.isfinite(score) & (np.abs(score) >= 1.0))
    decisions = events[:, 0].astype(np.int64) if events.size > 0 else np.zeros(0, dtype=np.int64)
    symbols = events[:, 1].astype(np.int64) if events.size > 0 else np.zeros(0, dtype=np.int64)
    mean_score = np.nanmean(oriented_score_2d)
    orientation_sign = 1 if not np.isfinite(mean_score) else int(np.sign(mean_score) or 1)

    if events.shape[0] == 0:
        return PrecomputedExitPaths(
            decision_idx=np.zeros(0, dtype=np.int64),
            edge_bps=np.zeros(0, dtype=np.float64),
            mae_bps=np.zeros(0, dtype=np.float64),
            mfe_bps=np.zeros(0, dtype=np.float64),
            horizon_bars=horizon,
            orientation_sign=orientation_sign,
        )

    if atr is None:
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


def precompute_exit_path_cache(
    panel: RawSignalPanel,
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
) -> ExitPathCache:
    atr = _atr(bars_4h)
    paths_by_signal: dict[str, PrecomputedExitPaths] = {}
    for signal_idx, descriptor in enumerate(panel.descriptors):
        signal_z = panel.z_3d[:, :, signal_idx]
        finite_mask = np.isfinite(signal_z)
        if not np.any(finite_mask):
            continue
        oriented = signal_z.astype(np.float32)
        paths = precompute_exit_paths(descriptor, oriented, bars_4h, cost_bps_4h, atr=atr)
        paths_by_signal[descriptor.signal_id] = paths
    return ExitPathCache(paths_by_signal=paths_by_signal, atr=atr)


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
            policy, 0.0, mean, se, probability, 1.0, tuple(fold_returns), len(fold_returns),
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


def _compute_sleeve_returns_2d(
    sleeves: tuple[L1SleevePosterior, ...],
    bars_4h: TimeframeBarCube,
) -> NDArray[np.float64]:
    n_bars = bars_4h.close_2d.shape[0]
    n_syms = bars_4h.close_2d.shape[1]
    close = bars_4h.close_2d.astype(np.float64)
    log_ret = np.zeros((n_bars, n_syms), dtype=np.float64)
    prev = close[:-1]
    curr = close[1:]
    mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret_vals = np.where(mask, np.log(curr / prev), 0.0)
    log_ret[1:] = log_ret_vals
    log_ret[~np.isfinite(log_ret)] = 0.0

    n_sleeves = len(sleeves)
    sleeve_raw = np.zeros((n_bars, n_sleeves), dtype=np.float64)
    for i, sleeve in enumerate(sleeves):
        sym_idx = np.where(sleeve.member_mask_1d)[0]
        if len(sym_idx) == 0:
            continue
        member = log_ret[:, sym_idx]
        finite = np.isfinite(member)
        count = np.sum(finite, axis=1)
        sleeve_raw[:, i] = np.where(count > 0, np.nansum(member, axis=1) / count, 0.0)
    return sleeve_raw


def compute_beta_neutral_composite_returns(
    sleeves: tuple[L1SleevePosterior, ...],
    bars_4h: TimeframeBarCube,
    benchmark_returns_1d: NDArray[np.float64],
) -> NDArray[np.float64]:
    if not sleeves:
        raise ValueError("sleeves must be non-empty")
    n_bars = bars_4h.close_2d.shape[0]
    if len(benchmark_returns_1d) != n_bars:
        raise ValueError(
            f"benchmark_returns_1d length {len(benchmark_returns_1d)} != bars_4h bars {n_bars}"
        )

    n_sleeves = len(sleeves)
    sleeve_raw = _compute_sleeve_returns_2d(sleeves, bars_4h)

    bm = benchmark_returns_1d
    beta_resid = np.zeros((n_bars, n_sleeves), dtype=np.float64)
    for i in range(n_sleeves):
        sr = sleeve_raw[:, i]
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        count = 0
        for t in range(n_bars):
            if t == 0:
                beta_resid[t, i] = sr[t]
                continue
            x_t = bm[t - 1]
            y_t = sr[t - 1]
            if np.isfinite(x_t) and np.isfinite(y_t):
                count += 1
                sum_x += x_t
                sum_y += y_t
                sum_xx += x_t * x_t
                sum_xy += x_t * y_t
            if count < 5:
                beta_resid[t, i] = sr[t]
            else:
                denom = count * sum_xx - sum_x * sum_x
                beta = (count * sum_xy - sum_x * sum_y) / denom if denom > 1e-12 else 0.0
                beta_resid[t, i] = sr[t] - beta * bm[t]

    vols = np.zeros(n_sleeves, dtype=np.float64)
    for i in range(n_sleeves):
        br = beta_resid[:, i]
        finite = np.isfinite(br)
        n_finite = int(np.sum(finite))
        vols[i] = float(np.std(br[finite], ddof=1)) if n_finite > 10 else np.inf

    inv_vol = 1.0 / np.maximum(vols, 1e-12)
    inv_vol = inv_vol / np.sum(inv_vol)

    composite = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_sleeves):
        br = beta_resid[:, i]
        finite = np.isfinite(br)
        composite[finite] += inv_vol[i] * br[finite]
    return composite


def compute_l1_oos_portfolio_returns(
    weights_2d: NDArray[np.float64],
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
) -> NDArray[np.float64]:
    close_2d = bars_4h.close_2d
    t_total, n_sym = close_2d.shape
    if weights_2d.shape != (t_total, n_sym) or cost_bps_4h.shape != (t_total, n_sym):
        raise ValueError("shape mismatch among weights_2d, cost_bps_4h, and bars_4h.close_2d")
    if not folds:
        return np.zeros(0, dtype=np.float64)

    oos_mask = np.zeros(t_total, dtype=np.bool_)
    for f in folds:
        oos_start = f.oos_start
        oos_end = min(f.oos_end_exclusive, t_total - 1)
        if oos_start < oos_end:
            oos_mask[oos_start:oos_end] = True
    oos_indices = np.where(oos_mask)[0]
    n_oos = len(oos_indices)
    if n_oos < 10:
        return np.zeros(0, dtype=np.float64)

    log_ret = np.zeros((t_total, n_sym), dtype=np.float64)
    prev = close_2d[:-1].astype(np.float64)
    curr = close_2d[1:].astype(np.float64)
    valid = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.where(valid, np.log(curr / prev), 0.0)
    log_ret[~np.isfinite(log_ret)] = 0.0
    cost = cost_bps_4h.astype(np.float64)

    r_p = np.zeros(n_oos, dtype=np.float64)
    prev_pos = np.zeros(n_sym, dtype=np.float64)
    for k, t in enumerate(oos_indices):
        pos = weights_2d[t]
        r_p[k] = float(np.dot(pos, log_ret[t + 1])) - float(np.dot(cost[t], np.abs(pos - prev_pos))) * 1e-4
        prev_pos = pos
    return r_p


def compute_fold_growths(
    weights_2d: NDArray[np.float64],
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
) -> tuple[float, ...]:
    close_2d = bars_4h.close_2d
    t_total, n_sym = close_2d.shape
    if weights_2d.shape != (t_total, n_sym) or cost_bps_4h.shape != (t_total, n_sym):
        raise ValueError("shape mismatch")

    log_ret = np.zeros((t_total, n_sym), dtype=np.float64)
    prev = close_2d[:-1].astype(np.float64)
    curr = close_2d[1:].astype(np.float64)
    valid = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.where(valid, np.log(curr / prev), 0.0)
    log_ret[~np.isfinite(log_ret)] = 0.0
    cost = cost_bps_4h.astype(np.float64)

    growths: list[float] = []
    for f in folds:
        oos_start = f.oos_start
        oos_end = min(f.oos_end_exclusive, t_total)
        if oos_end - oos_start < 2:
            continue
        fold_rets = np.zeros(oos_end - oos_start, dtype=np.float64)
        prev_pos = np.zeros(n_sym, dtype=np.float64)
        for k, t in enumerate(range(oos_start, oos_end)):
            if t < t_total - 1:
                pos = weights_2d[t]
                fold_rets[k] = float(np.dot(pos, log_ret[t + 1])) - float(np.dot(cost[t], np.abs(pos - prev_pos))) * 1e-4
                prev_pos = pos
        finite_rets = fold_rets[np.isfinite(fold_rets)]
        if len(finite_rets) < 2:
            continue
        log_growth = float(np.mean(np.log1p(finite_rets))) * 2191.5
        growths.append(log_growth)
    return tuple(growths)


def _compute_oos_returns_decomposed(
    weights_2d: NDArray[np.float64],
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    close_2d = bars_4h.close_2d
    t_total, n_sym = close_2d.shape
    if weights_2d.shape != (t_total, n_sym):
        raise ValueError("shape mismatch")
    if not folds:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    log_ret = np.zeros((t_total, n_sym), dtype=np.float64)
    prev = close_2d[:-1].astype(np.float64)
    curr = close_2d[1:].astype(np.float64)
    valid = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.where(valid, np.log(curr / prev), 0.0)
    log_ret[~np.isfinite(log_ret)] = 0.0
    cost = cost_bps_4h.astype(np.float64)

    oos_mask = np.zeros(t_total, dtype=np.bool_)
    for f in folds:
        oos_start = f.oos_start
        oos_end = min(f.oos_end_exclusive, t_total - 1)
        if oos_start < oos_end:
            oos_mask[oos_start:oos_end] = True
    oos_indices = np.where(oos_mask)[0]
    n_oos = len(oos_indices)
    if n_oos < 10:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    gross = np.zeros(n_oos, dtype=np.float64)
    cost_ret = np.zeros(n_oos, dtype=np.float64)
    net = np.zeros(n_oos, dtype=np.float64)
    prev_pos = np.zeros(n_sym, dtype=np.float64)
    for k, t in enumerate(oos_indices):
        pos = weights_2d[t]
        gross[k] = float(np.dot(pos, log_ret[t + 1]))
        turnover = np.abs(pos - prev_pos)
        cost_ret[k] = -float(np.dot(cost[t], turnover)) * 1e-4
        net[k] = gross[k] + cost_ret[k]
        prev_pos = pos
    return net, gross, cost_ret


def _compute_fold_growths_decomposed(
    weights_2d: NDArray[np.float64],
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    close_2d = bars_4h.close_2d
    t_total, n_sym = close_2d.shape
    if weights_2d.shape != (t_total, n_sym):
        raise ValueError("shape mismatch")

    log_ret = np.zeros((t_total, n_sym), dtype=np.float64)
    prev = close_2d[:-1].astype(np.float64)
    curr = close_2d[1:].astype(np.float64)
    valid = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.where(valid, np.log(curr / prev), 0.0)
    log_ret[~np.isfinite(log_ret)] = 0.0
    cost = cost_bps_4h.astype(np.float64)

    net_growths: list[float] = []
    gross_growths: list[float] = []
    cost_growths: list[float] = []
    for f in folds:
        oos_start = f.oos_start
        oos_end = min(f.oos_end_exclusive, t_total)
        if oos_end - oos_start < 2:
            continue
        fold_gross = np.zeros(oos_end - oos_start, dtype=np.float64)
        fold_cost = np.zeros(oos_end - oos_start, dtype=np.float64)
        fold_net = np.zeros(oos_end - oos_start, dtype=np.float64)
        prev_pos = np.zeros(n_sym, dtype=np.float64)
        for k, t in enumerate(range(oos_start, oos_end)):
            if t < t_total - 1:
                pos = weights_2d[t]
                fold_gross[k] = float(np.dot(pos, log_ret[t + 1]))
                turnover = np.abs(pos - prev_pos)
                fold_cost[k] = -float(np.dot(cost[t], turnover)) * 1e-4
                fold_net[k] = fold_gross[k] + fold_cost[k]
                prev_pos = pos
        finite = fold_net[np.isfinite(fold_net)]
        if len(finite) < 2:
            continue
        net_log = float(np.mean(np.log1p(finite))) * 2191.5
        net_growths.append(net_log)
        gross_log = float(np.mean(np.log1p(fold_gross[np.isfinite(fold_gross)]))) * 2191.5
        gross_growths.append(gross_log)
        cost_log = float(np.mean(np.log1p(-fold_cost[np.isfinite(fold_cost)]))) * 2191.5 if np.any(fold_cost < 0) else 0.0
        cost_growths.append(cost_log)
    return tuple(net_growths), tuple(gross_growths), tuple(cost_growths)


def stress_execution_costs(
    gross_return_1d: NDArray[np.float64],
    execution_cost_return_1d: NDArray[np.float64],
    funding_return_1d: NDArray[np.float64],
    multiplier: float,
) -> NDArray[np.float64]:
    if multiplier < 0.0:
        raise ValueError(f"multiplier must be >= 0, got {multiplier}")
    return gross_return_1d + multiplier * execution_cost_return_1d + funding_return_1d


def compute_compounding_stability(
    weights_2d: NDArray[np.float64],
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    config: HandoffConfig,
) -> HandoffAdmissionEvidence:
    from src.domain.futures.compound.bootstrap import (
        circular_stationary_bootstrap_growth,
        politis_white_block_length,
    )

    portfolio_returns, gross_returns, cost_returns = _compute_oos_returns_decomposed(
        weights_2d, bars_4h, folds, cost_bps_4h,
    )
    if len(portfolio_returns) == 0:
        return HandoffAdmissionEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, (), False, ("no_oos_returns",), 0.0, ())

    finite_returns = portfolio_returns[np.isfinite(portfolio_returns)]
    try:
        resolved_pw_block = politis_white_block_length(finite_returns)
    except ValueError:
        resolved_pw_block = 0.0

    ann_lcb90, _, _ = circular_stationary_bootstrap_growth(
        portfolio_returns, 2191.5,
        n_bootstrap=config.n_bootstrap,
        block_size=resolved_pw_block or None,
    )

    log_ret = np.log1p(np.where(np.isfinite(portfolio_returns), portfolio_returns, 0.0))
    ann_growth = float(np.mean(log_ret)) * 2191.5

    fold_growths, _, _ = _compute_fold_growths_decomposed(weights_2d, bars_4h, folds, cost_bps_4h)
    positive_outer_folds = sum(1 for g in fold_growths if g > 0.0)

    robust_fold_growth = 0.0
    if fold_growths:
        g_array = np.array(fold_growths, dtype=np.float64)
        median_g = float(np.median(g_array))
        mad = float(np.median(np.abs(g_array - median_g)))
        robust_fold_growth = median_g - 1.4826 * mad

    ann_vol = float(np.std(portfolio_returns, ddof=1)) * math.sqrt(2191.5) if len(portfolio_returns) > 1 else 0.0

    cum_returns = np.cumprod(1.0 + portfolio_returns)
    peak = np.maximum.accumulate(cum_returns)
    dd = 1.0 - cum_returns / peak
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    stressed = stress_execution_costs(gross_returns, cost_returns, np.zeros_like(gross_returns), config.cost_stress_multiplier)
    stressed_growth = float(np.mean(np.log1p(np.where(np.isfinite(stressed), stressed, 0.0)))) * 2191.5
    growth_2x_cost = stressed_growth

    growth_lcb90_check = ann_lcb90 > 0
    growth_2x_check = growth_2x_cost > 0
    positive_folds_check = positive_outer_folds >= max(1, len(folds) * 4 // 5)
    robust_check = robust_fold_growth > 0
    vol_check = ann_vol <= config.max_ann_vol if config.max_ann_vol > 0 else True
    dd_check = max_dd <= config.max_drawdown if config.max_drawdown > 0 else True

    reasons: list[str] = []
    if not growth_lcb90_check:
        reasons.append("growth_lcb90_not_positive")
    if not growth_2x_check:
        reasons.append("growth_2x_cost_not_positive")
    if not positive_folds_check:
        reasons.append("insufficient_positive_folds")
    if not robust_check:
        reasons.append("robust_fold_growth_not_positive")
    if not vol_check:
        reasons.append("annual_volatility_exceeded")
    if not dd_check:
        reasons.append("max_drawdown_exceeded")

    admitted = not reasons

    return HandoffAdmissionEvidence(
        ann_growth, ann_lcb90, growth_2x_cost,
        max_dd, ann_vol, positive_outer_folds,
        1.0, (), admitted, tuple(reasons),
        robust_fold_growth=robust_fold_growth,
        fold_growths=fold_growths,
    )


def build_exit_aware_handoff(
    forecast: CalibratedForecastPanel,
    sleeves: tuple[L1RoutingSleeve | L1SleevePosterior, ...],
    bars_4h: TimeframeBarCube,
    benchmark_returns_1d: NDArray[np.float64],
    config: HandoffConfig,
    *,
    folds: tuple[CausalFold, ...],
    weights_2d: NDArray[np.float64],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32] | None = None,
) -> HandoffResult:
    from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder

    recorder = L1AdmissionRecorder()

    admitted_sleeves = [s for s in sleeves if getattr(s, 'admitted', True)]
    if not admitted_sleeves:
        no_evidence = HandoffAdmissionEvidence(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, (), False, ("no_admitted_sleeves",), 0.0, ())
        recorder.record_gate(admitted_sleeves=0, distinct_series=0, oos_bars=0, ann_growth=0.0, ann_lcb90=0.0, pw_block=0.0, turnover=0.0, cost_drag=0.0, positive_folds=0, fold_growths=(), mean_abs_net=0.0, admitted=False)
        _LOGGER.info("[L1] exit-aware handoff: no admitted sleeves, NO_EVIDENCE")
        return HandoffResult(forecast, no_evidence)

    from src.domain.futures.compound.bootstrap import politis_white_block_length

    funding = funding_1h_2d if funding_1h_2d is not None else np.zeros((bars_4h.close_2d.shape[0] * 4, bars_4h.close_2d.shape[1]), dtype=np.float32)

    evidence = compute_compounding_stability(
        weights_2d, bars_4h, folds, cost_bps_4h, funding, config,
    )
    evidence = HandoffAdmissionEvidence(
        evidence.annualized_log_growth,
        evidence.growth_lcb90,
        evidence.growth_2x_cost,
        evidence.max_drawdown,
        evidence.annual_volatility,
        evidence.positive_outer_folds,
        1.0,
        tuple(s.signal_id for s in admitted_sleeves),
        evidence.admitted,
        evidence.reasons,
        robust_fold_growth=evidence.robust_fold_growth,
        fold_growths=evidence.fold_growths,
    )

    portfolio_returns = compute_l1_oos_portfolio_returns(weights_2d, bars_4h, folds, cost_bps_4h)
    pw_block = 0.0
    if len(portfolio_returns) > 0:
        finite_returns = portfolio_returns[np.isfinite(portfolio_returns)]
        try:
            pw_block = politis_white_block_length(finite_returns)
        except ValueError:
            pw_block = 0.0

    _LOGGER.info("[L1] exit-aware handoff admitted=%s sleeves=%d distinct_series=%d oos_bars=%d ann_growth=%.4f ann_lcb90=%.4f robust_g=%.4f positive_folds=%d/%d",
                 evidence.admitted, len(admitted_sleeves), 1, len(portfolio_returns),
                 evidence.annualized_log_growth, evidence.growth_lcb90,
                 evidence.robust_fold_growth,
                 evidence.positive_outer_folds, len(evidence.fold_growths))
    recorder.record_gate(admitted_sleeves=len(admitted_sleeves), distinct_series=1,
                         oos_bars=len(portfolio_returns), ann_growth=evidence.annualized_log_growth,
                         ann_lcb90=evidence.growth_lcb90,
                         pw_block=pw_block, turnover=0.0, cost_drag=0.0,
                         positive_folds=evidence.positive_outer_folds,
                         fold_growths=evidence.fold_growths, mean_abs_net=0.0,
                         admitted=evidence.admitted)
    return HandoffResult(forecast, evidence, tuple(admitted_sleeves))


@njit(cache=True)  # type: ignore[untyped-decorator]
def aggregate_cluster_group_returns(
    returns_2d: np.ndarray,
    sigma_2d: np.ndarray,
    winsorize_pct: float = 0.10,
) -> np.ndarray:
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


def compute_chunked_2d_tensor_bootstrap(
    returns_2d: NDArray[np.float64],
    periods_per_year: float,
    n_bootstrap: int = 1000,
    chunk_size: int = 250,
    seed: int = 42,
) -> NDArray[np.float64]:
    n_bars, n_sleeves = returns_2d.shape
    if n_sleeves == 0:
        return np.empty(0, dtype=np.float64)
    if n_bars < 10:
        return np.full(n_sleeves, 0.0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    all_indices = rng.integers(0, n_bars, size=(n_bootstrap, n_bars))
    lcb90 = np.empty(n_sleeves, dtype=np.float64)
    log_buffer = np.empty((n_bars, min(chunk_size, n_sleeves)), dtype=np.float64)

    for start in range(0, n_sleeves, chunk_size):
        end = min(start + chunk_size, n_sleeves)
        chunk = returns_2d[:, start:end]
        n_chunk = end - start
        growth = np.empty((n_bootstrap, n_chunk), dtype=np.float64)

        for i in range(n_bootstrap):
            idx = all_indices[i]
            np.take(chunk, idx, axis=0, out=log_buffer[:, :n_chunk])
            np.log1p(np.where(np.isfinite(log_buffer[:, :n_chunk]), log_buffer[:, :n_chunk], 0.0), out=log_buffer[:, :n_chunk])
            growth[i] = periods_per_year * np.mean(log_buffer[:, :n_chunk], axis=0)

        lcb90[start:end] = np.percentile(growth, 10, axis=0)

    return lcb90


def _cluster_masked_beta(
    feature: NDArray[np.float32],
    close: NDArray[np.float32],
    descriptor: SignalDescriptor,
    fit_end: int,
    sym_indices: NDArray[np.int64],
    hac_lag_cap: int = 120,
) -> tuple[float, float, float, int, float]:
    horizon = max(descriptor.target_horizon_hours // 4, 1)
    if fit_end <= horizon + 2:
        return 0.0, 1.0, 0.5, 0, 1.0
    future = np.roll(close.astype(np.float64), -horizon, axis=0) / np.maximum(close, 1e-12) - 1.0
    future[-horizon:] = np.nan
    x = feature[: fit_end - horizon, sym_indices].astype(np.float64)
    y = future[: fit_end - horizon, sym_indices]
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[mask], y[mask]
    denom = float(np.dot(x_valid, x_valid)) + 1e-8
    beta = float(np.dot(x_valid, y_valid) / denom) if x_valid.size else 0.0

    # Driscoll-Kraay HAC standard error
    n_time = x.shape[0]
    residual_2d = y - beta * x
    g = np.zeros(n_time, dtype=np.float64)
    for t in range(n_time):
        mask_t = mask[t]
        g[t] = float(np.sum(x[t, mask_t] * residual_2d[t, mask_t]))
    se_ols = float(np.std(residual_2d[mask], ddof=1) / math.sqrt(denom)) if np.sum(mask) > 1 else 1.0
    hac_lag = min(horizon - 1, hac_lag_cap)
    if n_time <= hac_lag + 1 or n_time < 3:
        se = 1.0
    else:
        s_0 = float(np.mean(g * g))
        s_hac = s_0
        for l in range(1, hac_lag + 1):
            w = 1.0 - l / (hac_lag + 1.0)
            gamma_l = float(np.mean(g[l:] * g[:-l]))
            s_hac += 2.0 * w * gamma_l
        s_hac = max(s_hac, 1e-15)
        se = math.sqrt(s_hac / max(denom, 1e-15))
        se = max(se, se_ols * 1e-4)

    probability = float(0.5 * (1.0 + math.erf(beta / max(se, 1e-12) / math.sqrt(2.0))))
    return beta, max(se, 1e-8), probability, int(x_valid.size), se_ols


def estimate_cluster_sleeve_posteriors(
    panel: RawSignalPanel,
    bars_4h: TimeframeBarCube,
    cluster_folds: tuple[CausalClusterFold, ...],
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    config: HandoffConfig,
    cache: ExitPathCache | None = None,
) -> tuple[L1SleevePosterior, ...]:
    from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder

    if panel.z_3d.shape[:2] != bars_4h.close_2d.shape or cost_bps_4h.shape != bars_4h.close_2d.shape:
        raise ValueError("panel, bars, and cost shapes must agree")
    if not folds or not panel.descriptors:
        return ()
    n_symbols = len(panel.symbols)
    output: list[L1SleevePosterior] = []
    future_cache: dict[int, NDArray[np.float64]] = {}
    recorder = L1AdmissionRecorder()

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

                beta, se, probability, observations, se_ols = _cluster_masked_beta(
                    signal_z, bars_4h.close_2d, descriptor,
                    fold.fit_end_exclusive, sym_indices,
                    hac_lag_cap=config.hac_lag_cap,
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

                is_pass = probability >= config.min_sleeve_posterior_probability
                oos_pass = False
                if is_pass and aggregated.size >= config.min_oos_effective_blocks:
                    from src.domain.futures.compound.bootstrap import (
                        circular_stationary_bootstrap_growth,
                        politis_white_block_length,
                    )
                    try:
                        pw_block = politis_white_block_length(aggregated)
                    except ValueError:
                        pw_block = 5.0
                    _, _, oos_prob = circular_stationary_bootstrap_growth(
                        aggregated, 2191.5, n_bootstrap=config.n_bootstrap,
                        block_size=pw_block, seed=42,
                    )
                    oos_pass = oos_prob >= config.min_oos_posterior_probability
                admitted = is_pass and oos_pass
                n_blocks = max(1, fold.fit_end_exclusive // horizon)
                recorder.record_sleeve(
                    signal_id=descriptor.signal_id, fold=cf.fold_id, cluster=cluster_id,
                    beta=beta, se_hac=se, se_ols_ratio=(se / se_ols) if se_ols > 1e-12 else 1.0,
                    prob=probability, n_obs=observations, n_blocks=n_blocks, admitted=admitted,
                )
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
            cached = cache.get(descriptor.signal_id) if cache is not None else None
            orient_sign = int(np.sign(np.nanmean(oriented)) or 1)
            if cached is not None and cached.orientation_sign == orient_sign:
                paths = cached
            else:
                paths = precompute_exit_paths(descriptor, oriented, bars_4h, cost_bps_4h, atr=cache.atr if cache else None)

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
                if p["probability"] < config.min_sleeve_posterior_probability:
                    reasons = ("posterior_below_floor",)
                else:
                    reasons = ("oos_confirmation_failed",)

            output.append(L1SleevePosterior(
                sleeve_id=sleeve_id,
                signal_id=descriptor.signal_id,
                family=descriptor.family,
                outer_fold_id=p["cf"].fold_id,
                cluster_id=p["cluster_id"],
                member_mask_1d=member_mask,
                member_hash=p["cf"].member_hash,
                exit_policy=exit_policy,
                fitted_beta=p["beta"],
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
        pass

    return tuple(output)


def build_family_routing_sleeves(
    panel: RawSignalPanel,
    family_screen: FamilyEdgeScreen,
    cluster_folds: tuple[CausalClusterFold, ...],
    folds: tuple[CausalFold, ...],
) -> tuple[L1RoutingSleeve, ...]:
    panel_signal_ids = {d.signal_id for d in panel.descriptors}
    for sid in family_screen.admitted_signal_ids:
        if sid not in panel_signal_ids:
            raise ValueError(f"admitted signal id {sid} not found in panel")

    desc_map = {d.signal_id: d for d in panel.descriptors}
    admitted_set = set(family_screen.admitted_signal_ids)
    output: list[L1RoutingSleeve] = []
    n_symbols = len(panel.symbols)
    fold_map = {f.fold_id: f for f in folds}

    for cf in cluster_folds:
        if cf.fold_id not in fold_map:
            raise ValueError(f"fold {cf.fold_id} from cluster_folds not found in folds")

        cluster_panel = cf.panel
        unique_clusters = sorted(int(x) for x in np.unique(cluster_panel.cluster_labels))

        for cluster_id in unique_clusters:
            sym_mask = cluster_panel.cluster_labels == cluster_id
            sym_indices = np.where(sym_mask)[0]
            if len(sym_indices) < 2:
                continue

            for signal_id in sorted(admitted_set):
                descriptor = desc_map[signal_id]
                member_mask = np.zeros(n_symbols, dtype=np.bool_)
                member_mask[sym_indices] = True
                sleeve_id = f"{signal_id}:fold{cf.fold_id}:cluster_{cluster_id}"
                output.append(L1RoutingSleeve(
                    sleeve_id=sleeve_id,
                    signal_id=signal_id,
                    family=descriptor.family,
                    outer_fold_id=cf.fold_id,
                    cluster_id=cluster_id,
                    member_mask_1d=member_mask,
                    member_hash=cf.member_hash,
                    declared_orientation=descriptor.declared_orientation,
                ))

    for sleeve in output:
        desc = desc_map[sleeve.signal_id]
        if sleeve.declared_orientation != desc.declared_orientation:
            raise ValueError(
                f"orientation mismatch for {sleeve.signal_id}: "
                f"sleeve={sleeve.declared_orientation} vs descriptor={desc.declared_orientation}"
            )

    _LOGGER.info("[ALGO] build_family_routing_sleeves: %d sleeves from %d admitted signals",
                 len(output), len(admitted_set))

    return tuple(output)

