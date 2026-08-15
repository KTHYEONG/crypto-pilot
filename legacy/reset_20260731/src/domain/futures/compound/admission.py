from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import AdmissionConfig
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CalibrationTarget,
    CausalFold,
    RawSignalPanel,
    SignalAdmissionEvidence,
    SignalCalibration,
)

_logger = logging.getLogger(__name__)

# P3 wiring: calibs = calibrate_signals(panel, target, folds, config.calibration); evidence = evaluate_signal_admission(panel, target, calibs, folds, cost_bps, config.admission); candidates = select_composite_candidates(evidence, config.admission); forecast = combine_composite_forecast(panel, calibs, evidence, folds, config.admission); composite_evidence = evaluate_composite_admission(panel, target, forecast, folds, cost_bps, config.admission)

def _block_bootstrap_lcb(
    series: NDArray[np.float64], n_bootstrap: int, block_size: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n_t = series.shape[0]
    if n_t < 2:
        return 0.0, 0.0, 1.0
    means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        boot = np.empty(n_t, dtype=np.float64)
        pos = 0
        while pos < n_t:
            blk_start = rng.integers(0, max(n_t - block_size, 1))
            blk_end = min(blk_start + block_size, n_t)
            n_copied = min(blk_end - blk_start, n_t - pos)
            boot[pos:pos + n_copied] = series[blk_start:blk_start + n_copied]
            pos += n_copied
        means[b] = np.mean(boot)
    boot_mean = np.mean(means)
    boot_std = np.std(means, ddof=1)
    lcb90 = float(boot_mean - 1.645 * boot_std)
    p_value = float(np.mean(means <= 0.0))
    return lcb90, float(boot_mean), p_value


def _benjamini_hochberg(p_values: list[float], q_threshold: float) -> list[float]:
    n_k = len(p_values)
    if n_k == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    q_vals = np.full(n_k, 1.0, dtype=np.float64)
    for i in range(n_k):
        q_vals[i] = min(sorted_p[i] * n_k / max(i + 1, 1), 1.0)
    for i in range(n_k - 2, -1, -1):
        q_vals[i] = min(q_vals[i], q_vals[i + 1])
    result = np.full(n_k, 1.0, dtype=np.float64)
    result[sorted_idx] = q_vals
    return list(result)


def _annualize_factor(n_bars: int) -> float:
    bars_per_year = 2190.0
    return math.sqrt(bars_per_year / max(n_bars, 1))


def _compute_net_return_series(
    position_norm: NDArray[np.float64],
    y_2d: NDArray[np.float64],
    prev_position_norm: NDArray[np.float64] | None,
    cost_bps_2d: NDArray[np.float32] | None,
    default_cost_bps: float,
    cost_multiplier: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    oos_t = position_norm.shape[0]
    raw_return = np.sum(position_norm * y_2d, axis=1)
    turnover = np.full(oos_t, np.nan, dtype=np.float64)
    if prev_position_norm is not None:
        for t in range(oos_t):
            prev = prev_position_norm[t]
            curr = position_norm[t]
            turnover[t] = float(np.sum(np.abs(curr - prev)))
    else:
        for t in range(1, oos_t):
            prev = position_norm[t - 1]
            curr = position_norm[t]
            turnover[t] = float(np.sum(np.abs(curr - prev)))
        turnover[0] = 0.0

    if cost_bps_2d is not None:
        cost_arr = cost_bps_2d[:oos_t, :]
        cost = float(np.nanmean(cost_arr)) * 1e-4
    else:
        cost = default_cost_bps * 1e-4

    turnover_norm = np.where(np.isfinite(turnover), turnover, 0.0)
    net = raw_return - cost_multiplier * cost * turnover_norm
    return net, turnover_norm


def evaluate_signal_admission(
    panel: RawSignalPanel, targets: dict[int, CalibrationTarget],
    calibrations: tuple[SignalCalibration, ...],
    folds: tuple[CausalFold, ...], cost_bps_2d: NDArray[np.float32] | None,
    config: AdmissionConfig, rng_seed: int = 42,
) -> tuple[SignalAdmissionEvidence, ...]:
    rng = np.random.default_rng(rng_seed)
    n_cat = len(calibrations)
    n_t = panel.z_3d.shape[0]
    _ = panel.z_3d.shape[1]
    fold_of_time = np.full(n_t, -1, dtype=np.int32)
    for fi, fold in enumerate(folds):
        fold_of_time[fold.oos_start:fold.oos_end_exclusive] = fi

    evidence_list: list[SignalAdmissionEvidence] = []
    for k in range(n_cat):
        signal_id = calibrations[k].signal_id
        family = panel.descriptors[k].family
        desc = panel.descriptors[k]

        tgt_horizon = desc.target_horizon_hours
        if tgt_horizon not in targets:
            raise ValueError(f"missing target for horizon={tgt_horizon}h")
        target = targets[tgt_horizon]

        effective_block_size = max(config.block_size, desc.target_horizon_hours // 4)

        oos_net_returns: list[float] = []
        oos_net_2x_returns: list[float] = []
        fold_signs: list[int] = []

        for fi, fold in enumerate(folds):
            oos_slice = slice(fold.oos_start, fold.oos_end_exclusive)
            oos_t = fold.oos_end_exclusive - fold.oos_start
            if oos_t < 1:
                continue

            beta_k = calibrations[k].beta_by_fold[fi]
            z_k_raw = panel.z_3d[oos_slice, :, k]
            y_k_raw = target.y_2d[oos_slice, :]

            valid_ki = np.isfinite(z_k_raw) & np.isfinite(y_k_raw)
            z_k = np.where(valid_ki, z_k_raw, 0.0)
            y_k = np.where(valid_ki, y_k_raw, 0.0)

            position = beta_k * z_k
            gross = np.abs(position).sum(axis=1, keepdims=True)
            gross = np.where(gross > 0, gross, 1.0)
            position_norm = position / gross

            raw_return = np.sum(position_norm * y_k, axis=1)
            cost_2d = cost_bps_2d[oos_slice] if cost_bps_2d is not None else None

            net_1x, _ = _compute_net_return_series(
                position_norm, y_k, None, cost_2d, config.default_cost_bps, 1.0,
            )
            net_2x, _ = _compute_net_return_series(
                position_norm, y_k, None, cost_2d, config.default_cost_bps, 2.0,
            )

            oos_net_returns.extend(net_1x.tolist())
            oos_net_2x_returns.extend(net_2x.tolist())

            signal_mean_return = float(np.nanmean(raw_return))
            fold_signs.append(1 if signal_mean_return > 0 else 0)

        oos_series = np.array(oos_net_returns, dtype=np.float64)
        oos_series = oos_series[np.isfinite(oos_series)]
        oos_series_2x = np.array(oos_net_2x_returns, dtype=np.float64)
        oos_series_2x = oos_series_2x[np.isfinite(oos_series_2x)]

        lcb90, _, p_value = _block_bootstrap_lcb(
            oos_series, config.n_bootstrap, effective_block_size, rng,
        )
        ann_factor = _annualize_factor(len(oos_series))
        net_growth_lcb90 = lcb90 * ann_factor

        _, boot_mean_2x, _ = _block_bootstrap_lcb(
            oos_series_2x, config.n_bootstrap, effective_block_size, rng,
        )
        net_mean_2x = boot_mean_2x * ann_factor

        fold_sign_consistency = len([s for s in fold_signs if s > 0]) / max(len(fold_signs), 1)

        reasons: list[str] = []
        if net_growth_lcb90 <= 0:
            reasons.append(f"net_growth_lcb90={net_growth_lcb90:.6f}<=0")
        if net_mean_2x <= 0:
            reasons.append(f"net_mean_2x={net_mean_2x:.6f}<=0")
        if fold_sign_consistency < config.sign_consistency_min:
            reasons.append(f"sign_consistency={fold_sign_consistency:.3f}<{config.sign_consistency_min}")

        n_effective = len(oos_series) / max(effective_block_size, 1)
        effective_sample_note = ""
        if n_effective < 50:
            effective_sample_note = (
                f"low_effective_sample: n_effective={n_effective:.1f}<50 "
                f"(block_size={effective_block_size})"
            )

        admitted = len(reasons) == 0

        _logger.info(
            "[EVAL] signal=%s family=%s beta_mean=%.4f lcb90=%.6f net_mean_2x=%.6f "
            "sign_consistency=%.3f p=%.4f admitted=%s%s",
            signal_id, family, float(np.mean(calibrations[k].beta_by_fold)),
            net_growth_lcb90, net_mean_2x, fold_sign_consistency, p_value, admitted,
            f" note={effective_sample_note}" if effective_sample_note else "",
        )

        evidence_list.append(SignalAdmissionEvidence(
            signal_id=signal_id,
            family=family,
            oos_net_growth_lcb90=net_growth_lcb90,
            oos_net_mean_2x_cost=net_mean_2x,
            fold_sign_consistency=fold_sign_consistency,
            p_value=float(p_value),
            fdr_q_value=1.0,
            admitted=admitted,
            reasons=tuple(reasons),
            effective_sample_note=effective_sample_note,
        ))

    bh_pvals = [ev.p_value for ev in evidence_list]
    bh_q = _benjamini_hochberg(bh_pvals, config.fdr_q_threshold)
    final_evidence: list[SignalAdmissionEvidence] = []
    for k, ev in enumerate(evidence_list):
        q_val = bh_q[k]
        reasons = list(ev.reasons)
        if q_val > config.fdr_q_threshold and not any("fdr_q" in r for r in reasons):
            reasons.append(f"fdr_q={q_val:.4f}>{config.fdr_q_threshold}")
        final_admitted = len(reasons) == 0
        final_evidence.append(SignalAdmissionEvidence(
            signal_id=ev.signal_id,
            family=ev.family,
            oos_net_growth_lcb90=ev.oos_net_growth_lcb90,
            oos_net_mean_2x_cost=ev.oos_net_mean_2x_cost,
            fold_sign_consistency=ev.fold_sign_consistency,
            p_value=ev.p_value,
            fdr_q_value=float(q_val),
            admitted=final_admitted,
            reasons=tuple(reasons),
            effective_sample_note=ev.effective_sample_note,
        ))

    return tuple(final_evidence)


def select_composite_candidates(
    evidence: tuple[SignalAdmissionEvidence, ...],
    config: AdmissionConfig,
) -> tuple[int, ...]:
    candidates: list[int] = []
    for k, ev in enumerate(evidence):
        if (ev.fold_sign_consistency >= config.composite_sign_consistency_min
                and ev.p_value <= config.composite_p_value_max):
            candidates.append(k)
    return tuple(candidates)


def combine_composite_forecast(
    panel: RawSignalPanel, calibrations: tuple[SignalCalibration, ...],
    evidence: tuple[SignalAdmissionEvidence, ...],
    folds: tuple[CausalFold, ...],
    config: AdmissionConfig,
) -> CalibratedForecastPanel:
    n_t = panel.z_3d.shape[0]
    n_syms = panel.z_3d.shape[1]

    candidate_idx = select_composite_candidates(evidence, config)
    n_candidates = len(candidate_idx)

    fold_of_time = np.full(n_t, max(len(folds) - 1, 0), dtype=np.int32)
    for fi, fold in enumerate(folds):
        fold_of_time[fold.oos_start:fold.oos_end_exclusive] = fi

    if n_candidates == 0:
        _logger.info("[ALGO] composite: zero candidates, returning zero-forecast")
        mu_2d = np.zeros((n_t, n_syms), dtype=np.float32)
        se_2d = np.full((n_t, n_syms), np.nan, dtype=np.float32)
        family_mu_3d = np.zeros((n_t, n_syms, 1), dtype=np.float32)
        return CalibratedForecastPanel(
            decision_timestamps_ns=panel.decision_timestamps_ns,
            symbols=panel.symbols,
            mu_2d=mu_2d,
            se_2d=se_2d,
            family_mu_3d=family_mu_3d,
            family_ids=(),
            admitted_signal_ids=(),
            fold_manifest_hash=f"folds_{len(folds)}_{folds[0].purge_bars}_{folds[0].embargo_bars}" if folds else "no_folds",
        )

    mu_2d = np.zeros((n_t, n_syms), dtype=np.float32)
    se_2d_flat = np.full(n_t, np.nan, dtype=np.float32)
    for t in range(n_t):
        fi = fold_of_time[t]
        weighted_sum = np.zeros(n_syms, dtype=np.float64)
        total_precision = 0.0
        for kidx in candidate_idx:
            cal = calibrations[kidx]
            beta_se = cal.beta_se_by_fold[fi] if fi < len(cal.beta_se_by_fold) else 0.0
            if beta_se <= 0:
                continue
            precision = 1.0 / (beta_se * beta_se)
            beta_k = cal.beta_by_fold[fi] if fi < len(cal.beta_by_fold) else 0.0
            scale_k = math.sqrt(panel.descriptors[kidx].target_horizon_hours / 4.0)
            z_t = panel.z_3d[t, :, kidx].astype(np.float64)
            weighted_sum += precision * beta_k * z_t / scale_k
            total_precision += precision
        if total_precision > 0:
            mu_2d[t, :] = (weighted_sum / total_precision).astype(np.float32)
            se_2d_flat[t] = float(np.sqrt(1.0 / total_precision))
        else:
            se_2d_flat[t] = np.nan

    se_2d = np.tile(se_2d_flat[:, np.newaxis], (1, n_syms)).astype(np.float32)
    oos_start = folds[0].oos_start
    mu_2d[:oos_start] = 0.0
    se_2d[:oos_start] = np.nan

    family_mu_3d = np.zeros((n_t, n_syms, 1), dtype=np.float32)
    fold_manifest_hash = f"folds_{len(folds)}_{folds[0].purge_bars}_{folds[0].embargo_bars}" if folds else "no_folds"

    _logger.info(
        "[ALGO] composite: %d candidates combined (precision-weighted); mu_2d shape %s",
        n_candidates, mu_2d.shape,
    )

    return CalibratedForecastPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        mu_2d=mu_2d,
        se_2d=se_2d,
        family_mu_3d=family_mu_3d,
        family_ids=(),
        admitted_signal_ids=tuple(evidence[k].signal_id for k in candidate_idx),
        fold_manifest_hash=fold_manifest_hash,
    )


def evaluate_composite_admission(
    panel: RawSignalPanel, targets: dict[int, CalibrationTarget],
    forecast: CalibratedForecastPanel,
    folds: tuple[CausalFold, ...],
    cost_bps_2d: NDArray[np.float32] | None,
    config: AdmissionConfig,
    rng_seed: int = 42,
) -> SignalAdmissionEvidence:
    rng = np.random.default_rng(rng_seed)

    oos_net_returns: list[float] = []
    oos_net_2x_returns: list[float] = []
    fold_signs: list[int] = []

    for _fi, fold in enumerate(folds):
        oos_slice = slice(fold.oos_start, fold.oos_end_exclusive)
        oos_t = fold.oos_end_exclusive - fold.oos_start
        if oos_t < 1:
            continue

        y_2d_slice: NDArray[np.float64] = np.zeros((oos_t, panel.z_3d.shape[1]), dtype=np.float64)
        for target in targets.values():
            if oos_t <= target.y_2d.shape[0]:
                y_2d_slice = target.y_2d[oos_slice].astype(np.float64)
                break

        mu_t = forecast.mu_2d[oos_slice].astype(np.float64)
        valid_t = np.isfinite(mu_t) & np.isfinite(y_2d_slice)
        position_norm = np.where(valid_t, mu_t, 0.0)
        y_2d_slice = np.where(valid_t, y_2d_slice, 0.0)
        gross = np.abs(position_norm).sum(axis=1, keepdims=True)
        gross = np.where(gross > 0, gross, 1.0)
        position_norm = position_norm / gross

        raw_return = np.sum(position_norm * y_2d_slice, axis=1)
        cost_2d = cost_bps_2d[oos_slice] if cost_bps_2d is not None else None

        net_1x, _ = _compute_net_return_series(
            position_norm, y_2d_slice, None, cost_2d, config.default_cost_bps, 1.0,
        )
        net_2x, _ = _compute_net_return_series(
            position_norm, y_2d_slice, None, cost_2d, config.default_cost_bps, 2.0,
        )

        oos_net_returns.extend(net_1x.tolist())
        oos_net_2x_returns.extend(net_2x.tolist())

        signal_mean_return = float(np.nanmean(raw_return))
        fold_signs.append(1 if signal_mean_return > 0 else 0)

    oos_series = np.array(oos_net_returns, dtype=np.float64)
    oos_series = oos_series[np.isfinite(oos_series)]
    oos_series_2x = np.array(oos_net_2x_returns, dtype=np.float64)
    oos_series_2x = oos_series_2x[np.isfinite(oos_series_2x)]

    effective_block_size = config.block_size
    lcb90, _, p_value = _block_bootstrap_lcb(
        oos_series, config.n_bootstrap, effective_block_size, rng,
    )
    ann_factor = _annualize_factor(len(oos_series))
    net_growth_lcb90 = lcb90 * ann_factor

    _, boot_mean_2x, _ = _block_bootstrap_lcb(
        oos_series_2x, config.n_bootstrap, effective_block_size, rng,
    )
    net_mean_2x = boot_mean_2x * ann_factor

    fold_sign_consistency = len([s for s in fold_signs if s > 0]) / max(len(fold_signs), 1)

    reasons: list[str] = []
    if net_growth_lcb90 <= 0:
        reasons.append(f"net_growth_lcb90={net_growth_lcb90:.6f}<=0")
    if net_mean_2x <= 0:
        reasons.append(f"net_mean_2x={net_mean_2x:.6f}<=0")
    if fold_sign_consistency < config.sign_consistency_min:
        reasons.append(f"sign_consistency={fold_sign_consistency:.3f}<{config.sign_consistency_min}")

    n_effective = len(oos_series) / max(effective_block_size, 1)
    effective_sample_note = ""
    if n_effective < 50:
        effective_sample_note = (
            f"low_effective_sample: n_effective={n_effective:.1f}<50 "
            f"(block_size={effective_block_size})"
        )

    admitted = len(reasons) == 0

    p_val = float(p_value)
    fdr_q_val = p_val
    admitted_signal_ids = forecast.admitted_signal_ids
    family = "composite"

    _logger.info(
        "[EVAL] composite=%s lcb90=%.6f net_mean_2x=%.6f "
        "sign_consistency=%.3f p=%.4f admitted=%s fdr_q=%.4f (%d candidates)%s",
        family, net_growth_lcb90, net_mean_2x, fold_sign_consistency, p_val, admitted,
        fdr_q_val, len(admitted_signal_ids),
        f" note={effective_sample_note}" if effective_sample_note else "",
    )

    return SignalAdmissionEvidence(
        signal_id="composite",
        family=family,
        oos_net_growth_lcb90=net_growth_lcb90,
        oos_net_mean_2x_cost=net_mean_2x,
        fold_sign_consistency=fold_sign_consistency,
        p_value=p_val,
        fdr_q_value=fdr_q_val,
        admitted=admitted,
        reasons=tuple(reasons),
        effective_sample_note=effective_sample_note,
    )
