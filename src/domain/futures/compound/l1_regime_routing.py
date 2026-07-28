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
    L1SleevePosterior,
    RawSignalPanel,
    RegimeExpertEvidence,
    RegimeRoutedForecast,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder
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


def _check_beta_consistency(
    sleeves: tuple[L1SleevePosterior, ...],
    signal_id: str,
    fold_id: int,
) -> bool:
    sig_sleeves = [
        s for s in sleeves
        if s.signal_id == signal_id and s.outer_fold_id == fold_id and s.admitted
    ]
    if not sig_sleeves:
        return False
    signs = set()
    for s in sig_sleeves:
        beta_sign = int(np.sign(s.mean_net_return))
        if beta_sign == 0:
            return False
        signs.add(beta_sign)
    return len(signs) <= 1


def _compute_expert_shadow_weights(
    panel: RawSignalPanel,
    signal_id: str,
    member_mask: NDArray[np.bool_],
    bars_4h: TimeframeBarCube,
    sigma_2d: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
    allocator_config: DynamicCompoundingConfig,
) -> NDArray[np.float64]:
    signal_idx = next(
        i for i, d in enumerate(panel.descriptors) if d.signal_id == signal_id
    )
    mu_2d = np.where(
        member_mask.reshape(1, -1),
        panel.z_3d[:, :, signal_idx],
        0.0,
    ).astype(np.float32)
    n_bars, n_syms = mu_2d.shape
    mini_panel = CalibratedForecastPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        mu_2d=mu_2d,
        se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
        family_ids=("expert",),
        admitted_signal_ids=(signal_id,),
        fold_manifest_hash="",
    )
    weights = compute_dynamic_compounding_path(
        forecast=mini_panel,
        sigma_2d=sigma_2d,
        funding_rates_1h_2d=funding_1h_2d,
        config=allocator_config,
        close_2d=bars_4h.close_2d,
        cost_bps=1e-8,
    )
    return weights


def _compute_calibration_returns(
    weights_2d: NDArray[np.float64],
    fold: CausalFold,
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
    funding_1h_2d: NDArray[np.float32],
) -> NDArray[np.float64]:
    cal_start = fold.calibration_start
    cal_end = fold.calibration_end_exclusive
    if cal_end - cal_start < 2:
        return np.zeros(0, dtype=np.float64)
    n_syms = bars_4h.close_2d.shape[1]
    close = bars_4h.close_2d.astype(np.float64)
    n_cal = cal_end - cal_start
    returns = np.zeros(n_cal - 1, dtype=np.float64)
    prev_pos = np.zeros(n_syms, dtype=np.float64)
    for k, t in enumerate(range(cal_start, cal_end - 1)):
        pos = weights_2d[t]
        log_ret = np.zeros(n_syms, dtype=np.float64)
        prev = close[t]
        curr = close[t + 1]
        valid = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        log_ret[valid] = np.log(curr[valid] / prev[valid])
        cost_drag = float(np.dot(cost_bps_4h[t], np.abs(pos - prev_pos))) * 1e-4
        returns[k] = float(np.dot(pos, log_ret)) - cost_drag
        prev_pos = pos
    return returns


def _inner_fold_labels(
    n_cal_bars: int,
    n_inner_folds: int,
    horizon_bars: int,
    embargo_bars: int,
) -> NDArray[np.int16]:
    total_bars = n_cal_bars
    if total_bars <= 0:
        return np.zeros(0, dtype=np.int16)
    prohibited = embargo_bars + horizon_bars
    if total_bars <= prohibited:
        return np.zeros(total_bars, dtype=np.int16)
    usable = total_bars - prohibited
    fold_size = usable // n_inner_folds
    if fold_size <= 0:
        return np.zeros(total_bars, dtype=np.int16)
    labels = np.full(total_bars, -1, dtype=np.int16)
    for i in range(n_inner_folds):
        start = prohibited + i * fold_size
        end = total_bars if i == n_inner_folds - 1 else prohibited + (i + 1) * fold_size
        labels[start:end] = i
    return labels


def _estimate_regime_evidence(
    calibration_returns: NDArray[np.float64],
    regime_mask: NDArray[np.bool_],
    inner_fold_labels: NDArray[np.int16],
    config: RegimeRouterConfig,
) -> tuple[float, float, float, float, int, int, float, float]:
    regime_rets = calibration_returns[regime_mask]
    if len(regime_rets) == 0:
        return (0.0, 0.0, 0.5, 0.0, 0, 0, 0.0, 0.0)

    try:
        block_length = politis_white_block_length(regime_rets)
    except ValueError:
        block_length = 5.0
    effective_blocks = int(np.floor(len(regime_rets) / max(block_length, 1.0)))
    if effective_blocks < config.min_effective_blocks:
        return (0.0, 0.0, 0.5, 0.0, 0, effective_blocks, 0.0, 0.0)

    prior_strength = config.prior_effective_blocks
    n_prior = min(prior_strength, len(calibration_returns))
    prior_rets = calibration_returns[-n_prior:] if n_prior > 0 else regime_rets

    combined = np.concatenate([prior_rets, regime_rets])
    lcb90, ucb, prob_positive = circular_stationary_bootstrap_growth(
        combined, _BARS_PER_YEAR_4H,
        n_bootstrap=config.n_bootstrap,
        block_size=block_length,
        seed=42,
    )

    ret_2x = regime_rets * 2.0
    growth_2x = float(np.mean(np.log1p(ret_2x[np.isfinite(ret_2x)]))) * _BARS_PER_YEAR_4H if np.any(np.isfinite(ret_2x)) else -1e6

    inner_growths: list[float] = []
    unique_folds = np.unique(inner_fold_labels[inner_fold_labels >= 0])
    for f_id in unique_folds:
        fold_mask = regime_mask & (inner_fold_labels == f_id)
        fold_rets = calibration_returns[fold_mask]
        if len(fold_rets) < 2:
            continue
        g = float(np.mean(np.log1p(fold_rets[np.isfinite(fold_rets)]))) * _BARS_PER_YEAR_4H
        inner_growths.append(g)
    positive_inner = sum(1 for g in inner_growths if g > 0)

    flat = np.array(inner_growths, dtype=np.float64)
    if len(flat) > 0:
        median_g = float(np.median(flat))
        mad = float(np.median(np.abs(flat - median_g))) if len(flat) > 0 else 0.0
        robust_inner_growth = median_g - 1.4826 * mad
    else:
        robust_inner_growth = 0.0

    ann_vol = float(np.std(regime_rets, ddof=1)) * math.sqrt(_BARS_PER_YEAR_4H) if len(regime_rets) > 1 else 0.0

    return (lcb90, ucb, prob_positive, growth_2x, positive_inner, effective_blocks, robust_inner_growth, ann_vol)


def _regime_evidence_to_scale(
    lcb90: float,
    prob_positive: float,
    growth_2x: float,
    positive_inner: int,
    effective_blocks: int,
    robust_inner_growth: float,
    config: RegimeRouterConfig,
) -> tuple[float, bool, tuple[str, ...]]:
    reasons: list[str] = []
    if effective_blocks < config.min_effective_blocks:
        reasons.append("insufficient_regime_blocks")
    if lcb90 <= 0.0:
        reasons.append("growth_lcb90_not_positive")
    if prob_positive < config.min_posterior_probability:
        reasons.append("posterior_probability_below_threshold")
    if growth_2x <= 0.0:
        reasons.append("growth_2x_cost_not_positive")
    if positive_inner < config.min_positive_inner_folds:
        reasons.append("insufficient_positive_inner_folds")
    if robust_inner_growth <= 0.0:
        reasons.append("robust_inner_growth_not_positive")

    if effective_blocks < config.min_effective_blocks:
        return (0.0, False, tuple(reasons))

    admitted = bool(not reasons)
    scale = 0.0 if not admitted else min(1.0, max(0.0, (prob_positive - 0.5) / 0.4))

    return (scale, admitted, tuple(reasons))


def build_fold_local_regime_forecast(
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
) -> RegimeRoutedForecast:
    t_total, n_symbols, _ = panel.z_3d.shape
    recorder = L1AdmissionRecorder()

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

    active_expert_count_1d = np.zeros(t_total, dtype=np.int16)

    regime_expert_weights: dict[int, dict[str, float]] = {}
    for rc in regime_codes_present:
        regime_expert_weights[rc] = {}

    for fold in folds:
        fold_id = fold.fold_id

        distinct_signal_ids = sorted({
            s.signal_id for s in sleeves
            if s.outer_fold_id == fold_id and s.admitted
        })
        if not distinct_signal_ids:
            continue

        for signal_id in distinct_signal_ids:
            if not _check_beta_consistency(sleeves, signal_id, fold_id):
                all_evidence.append(RegimeExpertEvidence(
                    signal_id=signal_id,
                    outer_fold_id=fold_id,
                    regime_code=0,
                    effective_blocks=0,
                    posterior_positive_probability=0.0,
                    growth_lcb90=0.0,
                    growth_2x_cost=0.0,
                    robust_inner_growth=0.0,
                    positive_inner_folds=0,
                    scale=0.0,
                    admitted=False,
                    reasons=("beta_sign_instability",),
                ))
                continue

            member_mask = _expert_member_mask(sleeves, signal_id, fold_id, n_symbols)
            if not np.any(member_mask):
                continue

            signal_idx = next(
                i for i, d in enumerate(panel.descriptors) if d.signal_id == signal_id
            )
            horizon_bars = max(int(panel.descriptors[signal_idx].target_horizon_hours // 4), 1)

            expert_weights = _compute_expert_shadow_weights(
                panel, signal_id, member_mask,
                bars_4h, panel.sigma_2d,
                funding_1h_2d, allocator_config,
            )
            cal_returns = _compute_calibration_returns(
                expert_weights, fold,
                bars_4h, cost_bps_4h, funding_1h_2d,
            )
            if len(cal_returns) < 10:
                continue

            cal_code = regime_panel.code_1d[
                fold.calibration_start:fold.calibration_start + len(cal_returns)
            ]
            inner_labels = _inner_fold_labels(
                len(cal_returns), config.n_inner_folds,
                horizon_bars, 42,
            )

            for regime_code in regime_codes_present:
                if regime_code == 0:
                    scale = 0.0
                    all_evidence.append(RegimeExpertEvidence(
                        signal_id=signal_id,
                        outer_fold_id=fold_id,
                        regime_code=0,
                        effective_blocks=0,
                        posterior_positive_probability=0.0,
                        growth_lcb90=0.0,
                        growth_2x_cost=0.0,
                        robust_inner_growth=0.0,
                        positive_inner_folds=0,
                        scale=0.0,
                        admitted=False,
                        reasons=("cold_regime_no_admission",),
                    ))
                    continue

                regime_mask = cal_code == regime_code
                if int(np.sum(regime_mask)) < 3:
                    all_evidence.append(RegimeExpertEvidence(
                        signal_id=signal_id,
                        outer_fold_id=fold_id,
                        regime_code=regime_code,
                        effective_blocks=0,
                        posterior_positive_probability=0.0,
                        growth_lcb90=0.0,
                        growth_2x_cost=0.0,
                        robust_inner_growth=0.0,
                        positive_inner_folds=0,
                        scale=0.0,
                        admitted=False,
                        reasons=("insufficient_regime_samples",),
                    ))
                    continue

                n_hypotheses += 1

                lcb90, _ucb, prob, growth_2x, pos_inner, eff_blocks, robust_g, ann_vol = (
                    _estimate_regime_evidence(
                        cal_returns, regime_mask, inner_labels, config,
                    )
                )

                scale, admitted, reasons = _regime_evidence_to_scale(
                    lcb90, prob, growth_2x, pos_inner, eff_blocks, robust_g, config,
                )

                if scale > 0 and admitted:
                    regime_expert_weights[regime_code][signal_id] = scale

                all_evidence.append(RegimeExpertEvidence(
                    signal_id=signal_id,
                    outer_fold_id=fold_id,
                    regime_code=regime_code,
                    effective_blocks=eff_blocks,
                    posterior_positive_probability=prob,
                    growth_lcb90=lcb90,
                    growth_2x_cost=growth_2x,
                    robust_inner_growth=robust_g,
                    positive_inner_folds=pos_inner,
                    scale=scale,
                    admitted=admitted,
                    reasons=reasons,
                    annual_volatility=ann_vol,
                ))

                if L1AdmissionRecorder().enabled:
                    recorder.record_sleeve(
                        signal_id=signal_id, fold=fold_id, cluster=regime_code,
                        beta=0.0, se_hac=0.0, se_ols_ratio=0.0,
                        prob=prob, n_obs=eff_blocks, n_blocks=eff_blocks,
                        admitted=admitted,
                    )

    for regime_code in regime_codes_present:
        expert_scales = regime_expert_weights.get(regime_code, {})
        if not expert_scales:
            continue

        signal_ids_list = list(expert_scales.keys())
        sig_indices: list[int] = []
        sig_scales: list[float] = []
        for sid in signal_ids_list:
            idx = next(
                (i for i, d in enumerate(panel.descriptors) if d.signal_id == sid),
                None,
            )
            if idx is not None:
                sig_indices.append(idx)
                sig_scales.append(expert_scales[sid])

        if len(sig_indices) >= 2:
            fit_end = folds[0].fit_end_exclusive if folds else 1000
            z_fit = panel.z_3d[:fit_end]
            flat = z_fit[:, :, sig_indices].reshape(fit_end * n_symbols, len(sig_indices))
            corr_matrix = np.asarray(np.corrcoef(flat, rowvar=False), dtype=np.float64)
            survivor_mask = np.ones(len(sig_indices), dtype=bool)
            for i in range(len(sig_indices)):
                if not survivor_mask[i]:
                    continue
                for j in range(i + 1, len(sig_indices)):
                    if not survivor_mask[j]:
                        continue
                    if abs(float(corr_matrix[i, j])) >= config.max_expert_correlation:
                        ei: RegimeExpertEvidence | None = None
                        ej: RegimeExpertEvidence | None = None
                        for ev in reversed(all_evidence):
                            if ev.signal_id == signal_ids_list[i] and ev.regime_code == regime_code:
                                ei = ev
                            if ev.signal_id == signal_ids_list[j] and ev.regime_code == regime_code:
                                ej = ev
                        if ei is None or ej is None:
                            continue
                        if ej.robust_inner_growth > ei.robust_inner_growth:
                            survivor_mask[i] = False
                            break
                        elif ei.robust_inner_growth > ej.robust_inner_growth:
                            survivor_mask[j] = False
                        elif ej.effective_blocks > ei.effective_blocks:
                            survivor_mask[i] = False
                            break
                        elif ei.effective_blocks > ej.effective_blocks:
                            survivor_mask[j] = False
                        else:
                            if signal_ids_list[j] > signal_ids_list[i]:
                                survivor_mask[i] = False
                                break
                            else:
                                survivor_mask[j] = False

            sig_indices = [sig_indices[i] for i in range(len(sig_indices)) if survivor_mask[i]]
            sig_scales = [sig_scales[i] for i in range(len(sig_scales)) if survivor_mask[i]]

        total_raw = 0.0
        blend: list[tuple[str, float]] = []
        for sid, s in zip([signal_ids_list[i] for i in range(len(sig_indices))], sig_scales, strict=False):
            ev0 = next(
                (e for e in reversed(all_evidence) if e.signal_id == sid and e.regime_code == regime_code),
                None,
            )
            if ev0 is None:
                continue
            raw_score = s * max(ev0.growth_lcb90, 0.0) / max(ev0.annual_volatility, 0.01)
            total_raw += raw_score
            w = min(raw_score / max(total_raw, 1e-15), config.max_expert_weight) if total_raw > 0 else 0.0
            blend.append((sid, w))

        total_w = sum(w for _, w in blend)
        cap = min(1.0, config.max_expert_weight * len(blend))
        if total_w > cap and total_w > 0:
            blend = [(sid, w * cap / total_w) for sid, w in blend]

        regime_expert_weights[regime_code] = dict(blend)

    mu_2d = np.zeros((t_total, n_symbols), dtype=np.float32)
    for t in range(t_total):
        rc = int(regime_panel.code_1d[t])
        if rc == 0:
            active_this_bar = 0
        else:
            weights = regime_expert_weights.get(rc, {})
            active_this_bar = len(weights)
            if active_this_bar == 0:
                pass
            else:
                for sid, w in weights.items():
                    idx = next(
                        (i for i, d in enumerate(panel.descriptors) if d.signal_id == sid),
                        None,
                    )
                    if idx is not None:
                        member_mask = _expert_member_mask(sleeves, sid, -1, n_symbols)
                    else:
                        member_mask = np.ones(n_symbols, dtype=np.bool_)
                    mu_2d[t] += w * panel.z_3d[t, :, idx].astype(np.float32)
        active_expert_count_1d[t] = active_this_bar

    family_ids = tuple(sorted({d.family for d in panel.descriptors}))
    family_mu_3d = np.zeros((t_total, n_symbols, max(len(family_ids), 1)), dtype=np.float32)

    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        mu_2d=mu_2d,
        se_2d=np.full((t_total, n_symbols), np.nan, dtype=np.float32),
        family_mu_3d=family_mu_3d,
        family_ids=family_ids,
            admitted_signal_ids=tuple(sorted({
                s.signal_id for s in sleeves if s.admitted
            })),
        fold_manifest_hash=fold_manifest_hash,
    )

    return RegimeRoutedForecast(
        forecast=forecast,
        evidence=tuple(all_evidence),
        active_expert_count_1d=active_expert_count_1d,
        tested_hypotheses=n_hypotheses,
    )



