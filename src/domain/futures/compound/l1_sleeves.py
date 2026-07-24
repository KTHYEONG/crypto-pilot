"Exit-aware L1 evidence and posterior sleeve handoff."

from __future__ import annotations

import hashlib
import logging
import math

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CausalFold,
    ExitPolicyKind,
    ExitPolicySpec,
    HandoffAdmissionEvidence,
    HandoffResult,
    L1SleevePosterior,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
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
        output.append(L1SleevePosterior(
            f"{descriptor.signal_id}:{policy.policy_id}", descriptor.signal_id, descriptor.family, policy,
            mean, se, probability, 1.0, tuple(fold_returns), len(fold_returns),
            probability >= 0.65 and sum(value > 0.0 for value in fold_returns) >= 4,
            () if probability >= 0.65 else ("posterior_below_floor",),
        ))
    return tuple(output)


def combine_posterior_sleeves(
    panel: RawSignalPanel,
    sleeves: tuple[L1SleevePosterior, ...],
    folds: tuple[CausalFold, ...],
    config: HandoffConfig,
) -> CalibratedForecastPanel:
    "Combine sleeves with posterior reliability and family balancing."
    del folds, config
    t_total, n_symbols, _ = panel.z_3d.shape
    active = [sleeve for sleeve in sleeves if sleeve.admitted]
    def empty() -> CalibratedForecastPanel:
        return CalibratedForecastPanel(
            panel.decision_timestamps_ns, panel.symbols,
            np.zeros((t_total, n_symbols), dtype=np.float32),
            np.full((t_total, n_symbols), np.nan, dtype=np.float32),
            np.zeros((t_total, n_symbols, 1), dtype=np.float32), (), (), "",
        )
    if not active:
        return empty()
    families = sorted({sleeve.family for sleeve in active})
    family_mu: list[NDArray[np.float32]] = []
    ids: list[str] = []
    for family in families:
        selected = [sleeve for sleeve in active if sleeve.family == family]
        quality = np.asarray([max(sleeve.posterior_positive_probability - 0.5, 0.0) ** 2 / (sleeve.standard_error ** 2 + 1e-6) * sleeve.residual_novelty for sleeve in selected], dtype=np.float64)
        if float(quality.sum()) <= 0.0:
            continue
        quality /= quality.sum()
        matrices = []
        for sleeve in selected:
            signal_index = next(i for i, descriptor in enumerate(panel.descriptors) if descriptor.signal_id == sleeve.signal_id)
            matrices.append(panel.z_3d[:, :, signal_index])
        family_mu.append(np.nansum(np.stack(matrices, axis=2) * quality.reshape(1, 1, -1), axis=2).astype(np.float32) / max(len(families), 1))
        ids.extend(sleeve.signal_id for sleeve in selected)
    if not family_mu:
        return empty()
    mu = np.sum(np.stack(family_mu, axis=2), axis=2)
    gross = np.sum(np.abs(mu), axis=1, keepdims=True)
    mu = np.clip(mu / np.where(gross > 1e-12, gross, 1.0), -0.10, 0.10).astype(np.float32)
    return CalibratedForecastPanel(panel.decision_timestamps_ns, panel.symbols, mu, np.full((t_total, n_symbols), np.nan, dtype=np.float32), np.stack(family_mu, axis=2), tuple(families), tuple(ids), "")


def build_exit_aware_handoff(
    panel: RawSignalPanel,
    bars: MultiTimeframeBars,
    folds: tuple[CausalFold, ...],
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    config: HandoffConfig,
) -> HandoffResult:
    "Build an L1 handoff whose admission is based on exit-aware sleeves."
    sleeves = estimate_sleeve_posteriors(panel, bars.cubes["4h"], folds, cost_bps_4h, funding_1h_2d, config)
    forecast = combine_posterior_sleeves(panel, sleeves, folds, config)
    fold_returns = np.asarray([value for sleeve in sleeves for value in sleeve.fold_net_returns], dtype=np.float64)
    growth = float(np.mean(fold_returns) * 2190.0) if fold_returns.size else 0.0
    positive = int(np.sum(fold_returns > 0.0)) if fold_returns.size else 0
    reasons: list[str] = []
    if growth <= 0.0:
        reasons.append("growth_lcb90_not_positive")
    if positive < config.min_positive_outer_folds:
        reasons.append("positive_folds_below_floor")
    admitted = not reasons
    evidence = HandoffAdmissionEvidence(growth, growth, growth, 0.0, 0.0, positive, float(len(sleeves)), tuple(sleeve.signal_id for sleeve in sleeves if sleeve.admitted), admitted, tuple(reasons))
    _LOGGER.info("[L1] exit-aware handoff admitted=%s sleeves=%d", admitted, len(sleeves))
    return HandoffResult(forecast, evidence)
