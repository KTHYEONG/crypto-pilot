from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.calibration import _pooled_ridge_beta
from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CalibrationTarget,
    CausalFold,
    CausalityError,
    HandoffAdmissionEvidence,
    HandoffResult,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)

_logger = logging.getLogger(__name__)


def apply_causal_holding_kernel(
    forecast_3d: NDArray[np.float32],
    horizon_bars_1d: NDArray[np.int16],
) -> NDArray[np.float32]:
    t_total, n_syms, n_signals = forecast_3d.shape
    result = np.zeros_like(forecast_3d)
    for k in range(n_signals):
        h = int(horizon_bars_1d[k])
        if h < 1:
            result[:, :, k] = forecast_3d[:, :, k]
            continue
        m = forecast_3d[:, :, k].astype(np.float64)
        cum = np.zeros((t_total + 1, n_syms), dtype=np.float64)
        np.cumsum(m, axis=0, out=cum[1:])
        kernel = np.zeros((t_total, n_syms), dtype=np.float64)
        kernel[:h] = cum[1:h + 1] / h
        if t_total > h:
            kernel[h:] = (cum[h + 1:] - cum[1:t_total - h + 1]) / h
        result[:, :, k] = kernel.astype(np.float32)
    return result


def _compute_4h_returns(
    bars_4h: TimeframeBarCube,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    close = bars_4h.close_2d.astype(np.float64)
    n_bars = close.shape[0]
    n_syms = close.shape[1]
    ret = np.zeros((n_bars, n_syms), dtype=np.float32)
    valid = np.zeros((n_bars, n_syms), dtype=np.bool_)
    for t in range(1, n_bars):
        prev = close[t - 1]
        curr = close[t]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        log_ret = np.full(n_syms, 0.0, dtype=np.float64)
        log_ret[mask] = np.log(curr[mask] / prev[mask])
        ret[t] = log_ret.astype(np.float32)
        valid[t, mask] = True
    return ret, valid


def _simulate_sleeve_pnl(
    weight_2d: NDArray[np.float32],
    returns_4h: NDArray[np.float32],
    cost_bps_4h: NDArray[np.float32],
    cost_multiplier: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_bars, n_syms = weight_2d.shape
    net_ret = np.zeros(n_bars, dtype=np.float64)
    equity = np.ones(n_bars, dtype=np.float64)
    prev_w = np.zeros(n_syms, dtype=np.float64)
    pending_cost = 0.0

    for t in range(n_bars):
        if t == 0:
            prev_w = weight_2d[0].astype(np.float64)
            continue
        bar_ret = float(np.nansum(prev_w * returns_4h[t].astype(np.float64)))
        fee = -pending_cost
        net_ret[t] = bar_ret + fee
        equity[t] = equity[t - 1] * max(1.0 + net_ret[t], 1e-12)
        turnover = float(np.sum(np.abs(weight_2d[t].astype(np.float64) - prev_w)))
        taker_bps = float(np.mean(cost_bps_4h[t])) if cost_bps_4h.ndim > 1 else float(cost_bps_4h[t])
        pending_cost = turnover * taker_bps * cost_multiplier * 1e-4
        prev_w = weight_2d[t].astype(np.float64)

    return net_ret, equity


def _bootstrap_lcb(
    returns: NDArray[np.float64],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 4:
        return 0.0
    ann_factor = 2190.0 / n
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_log = float(np.sum(np.log1p(r[idx]))) * ann_factor
        samples[i] = boot_log
    samples.sort()
    return float(samples[int(n_bootstrap * 0.10)])


def _compute_fold_betas(
    panel: RawSignalPanel,
    targets: dict[int, CalibrationTarget],
    fold: CausalFold,
) -> NDArray[np.float32]:
    n_signals = panel.z_3d.shape[2]
    betas = np.zeros(n_signals, dtype=np.float32)
    fit_slice = slice(fold.fit_start, fold.fit_end_exclusive)
    for k in range(n_signals):
        tgt_horizon = panel.descriptors[k].target_horizon_hours
        if tgt_horizon not in targets:
            raise ValueError(f"missing target for horizon={tgt_horizon}h")
        target = targets[tgt_horizon]
        z_k = panel.z_3d[fit_slice, :, k]
        y_fit = target.y_2d[fit_slice, :]
        v = panel.valid_3d[fit_slice, :, k] & target.valid_2d[fit_slice, :]
        b, _, n = _pooled_ridge_beta(z_k, y_fit, v, ridge_lambda=0.01)
        if n < 100:
            b = 0.0
        betas[k] = float(b)
    return betas


def _family_correlation_components(
    forecasts_3d: NDArray[np.float32],
    descriptors: tuple[SignalDescriptor, ...],
    family: str,
    max_corr: float,
) -> list[list[int]]:
    family_indices = [i for i, d in enumerate(descriptors) if d.family == family]
    if len(family_indices) <= 1:
        return [[i] for i in family_indices]

    t_total, n_syms, _ = forecasts_3d.shape
    n_fam = len(family_indices)
    flat = forecasts_3d[:, :, family_indices].reshape(t_total * n_syms, n_fam)
    mask = np.all(np.isfinite(flat), axis=1)
    if mask.sum() < 2:
        return [[i] for i in family_indices]

    corr_arr = np.corrcoef(flat[mask], rowvar=False)
    corr: NDArray[np.float64] = np.asarray(np.nan_to_num(corr_arr, nan=0.0), dtype=np.float64)

    n = n_fam
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] >= max_corr:
                adj[i].add(j)
                adj[j].add(i)

    visited = [False] * n
    components: list[list[int]] = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            comp: list[int] = []
            while stack:
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True
                    comp.append(family_indices[v])
                    stack.extend([x for x in adj[v] if not visited[x]])
            components.append(comp)
    return components


def _compute_component_return(
    panel: RawSignalPanel,
    betas: NDArray[np.float32],
    fold: CausalFold,
    component_indices: list[int],
    returns_4h: NDArray[np.float32],
    cost_bps_4h: NDArray[np.float32],
    config: HandoffConfig,
) -> float:
    t_total, n_syms, _ = panel.z_3d.shape
    hb = np.array([panel.descriptors[k].target_horizon_hours // 4 for k in component_indices], dtype=np.int16)
    m = np.zeros((t_total, n_syms, len(component_indices)), dtype=np.float32)
    for ji, k in enumerate(component_indices):
        m[:, :, ji] = betas[k] * panel.z_3d[:, :, k]
    kernel = apply_causal_holding_kernel(m, hb)
    avg = np.mean(kernel, axis=2).astype(np.float32)

    fit_slice = slice(fold.fit_start, fold.fit_end_exclusive)
    w = avg[fit_slice]
    w_std = np.std(w, axis=1, keepdims=True)
    w_std = np.where(w_std > 1e-12, w_std, 1.0)
    w = w / w_std

    net_ret, _ = _simulate_sleeve_pnl(
        w, returns_4h[fit_slice], cost_bps_4h[fit_slice],
        config.cost_stress_multiplier,
    )
    r = net_ret[1:]
    ann_factor = 2190.0 / max(len(r), 1)
    total_log = float(np.sum(np.log1p(np.where(np.isfinite(r), r, 0.0))))
    return total_log * ann_factor


def _build_cash_only_forecast(
    timestamps_ns: NDArray[np.int64],
    symbols: tuple[str, ...],
) -> CalibratedForecastPanel:
    n_bars = timestamps_ns.size
    n_syms = len(symbols)
    return CalibratedForecastPanel(
        decision_timestamps_ns=timestamps_ns,
        symbols=symbols,
        mu_2d=np.zeros((n_bars, n_syms), dtype=np.float32),
        se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
        family_ids=(),
        admitted_signal_ids=(),
        fold_manifest_hash="",
    )


def build_prequential_handoff(
    panel: RawSignalPanel,
    targets: dict[int, CalibrationTarget],
    folds: tuple[CausalFold, ...],
    bars_4h: TimeframeBarCube,
    cost_bps_4h: NDArray[np.float32],
    config: HandoffConfig,
    *,
    rng_seed: int = 42,
) -> HandoffResult:
    t_total, n_syms, _ = panel.z_3d.shape
    returns_4h, _ = _compute_4h_returns(bars_4h)

    if panel.decision_timestamps_ns.size != t_total:
        raise CausalityError("panel timestamps mismatch with z_3d shape")
    if cost_bps_4h.shape[0] != t_total or cost_bps_4h.shape[1] != n_syms:
        raise ValueError(f"cost_bps_4h shape {cost_bps_4h.shape} != ({t_total}, {n_syms})")

    n_folds = len(folds)
    if n_folds < 1:
        return HandoffResult(
            forecast=_build_cash_only_forecast(panel.decision_timestamps_ns, panel.symbols),
            evidence=HandoffAdmissionEvidence(
                annualized_log_growth=0.0,
                growth_lcb90=0.0,
                growth_2x_cost=0.0,
                max_drawdown=0.0,
                annual_volatility=0.0,
                positive_outer_folds=0,
                effective_breadth=0.0,
                active_signal_ids=(),
                admitted=False,
                reasons=("no_folds",),
            ),
        )

    rng = np.random.default_rng(rng_seed)

    all_selected_ids: set[str] = set()
    all_oos_nets: list[NDArray[np.float64]] = []
    all_oos_equities: list[NDArray[np.float64]] = []
    all_fold_positive = 0

    for fold in folds:
        _logger.debug(
            "handoff fold %d: fit=[%d:%d) oos=[%d:%d)",
            fold.fold_id, fold.fit_start, fold.fit_end_exclusive,
            fold.oos_start, fold.oos_end_exclusive,
        )

        if fold.fit_end_exclusive <= fold.fit_start or fold.oos_end_exclusive <= fold.oos_start:
            continue

        betas = _compute_fold_betas(panel, targets, fold)
        valid_beta = np.isfinite(betas) & (betas != 0.0)
        if not valid_beta.any():
            oos_len = fold.oos_end_exclusive - fold.oos_start
            all_oos_nets.append(np.zeros(oos_len, dtype=np.float64))
            all_oos_equities.append(np.ones(oos_len, dtype=np.float64))
            continue

        hb = np.array([d.target_horizon_hours // 4 for d in panel.descriptors], dtype=np.int16)
        n_signals = panel.z_3d.shape[2]
        m = np.zeros((t_total, n_syms, n_signals), dtype=np.float32)
        for k in range(n_signals):
            m[:, :, k] = betas[k] * panel.z_3d[:, :, k]
        a = apply_causal_holding_kernel(m, hb)

        families = sorted({d.family for d in panel.descriptors})
        family_reps: dict[str, dict[str, float | list[int]]] = {}

        for family in families:
            components = _family_correlation_components(a, panel.descriptors, family, config.max_pairwise_correlation)
            for comp in components:
                k0 = comp[0]
                if not np.isfinite(betas[k0]) or betas[k0] == 0.0:
                    continue
                ann_g = _compute_component_return(panel, betas, fold, comp, returns_4h, cost_bps_4h, config)
                if family not in family_reps:
                    family_reps[family] = {"growth": ann_g, "indices": comp}
                else:
                    prev_growth = family_reps[family]["growth"]
                    if isinstance(prev_growth, (int, float)) and ann_g > prev_growth:
                        family_reps[family] = {"growth": ann_g, "indices": comp}

        active_indices: list[int] = []
        for rep in family_reps.values():
            idx_val = rep["indices"]
            if isinstance(idx_val, list):
                active_indices.extend(idx_val)

        if not active_indices:
            oos_len = fold.oos_end_exclusive - fold.oos_start
            all_oos_nets.append(np.zeros(oos_len, dtype=np.float64))
            all_oos_equities.append(np.ones(oos_len, dtype=np.float64))
            continue

        for k in active_indices:
            all_selected_ids.add(panel.descriptors[k].signal_id)

        family_list = sorted({panel.descriptors[k].family for k in active_indices})
        n_families = max(len(family_list), 1)
        fit_slice = slice(fold.fit_start, fold.fit_end_exclusive)

        sleeve_weights: list[NDArray[np.float32]] = []
        for fam in family_list:
            fam_k = [k for k in active_indices if panel.descriptors[k].family == fam]
            fam_forecasts = a[:, :, fam_k]
            if len(fam_k) > 1:
                avg_forecast = np.mean(fam_forecasts, axis=2).astype(np.float32)
            else:
                avg_forecast = fam_forecasts[:, :, 0]

            fit_f = avg_forecast[fit_slice]
            fam_vol = np.std(fit_f, axis=0, keepdims=True)
            fam_vol = np.where(fam_vol > 1e-12, fam_vol, 1.0)
            scaled = avg_forecast / fam_vol
            sleeve_weights.append(scaled * (1.0 / n_families))

        if not sleeve_weights:
            oos_len = fold.oos_end_exclusive - fold.oos_start
            all_oos_nets.append(np.zeros(oos_len, dtype=np.float64))
            all_oos_equities.append(np.ones(oos_len, dtype=np.float64))
            continue

        combined_weight = np.sum(np.stack(sleeve_weights, axis=0), axis=0)

        oos_slice = slice(fold.oos_start, fold.oos_end_exclusive)
        w_oos = combined_weight[oos_slice]
        w_oos_std = np.std(w_oos, axis=1, keepdims=True)
        w_oos_std = np.where(w_oos_std > 1e-12, w_oos_std, 1.0)
        w_oos = w_oos / w_oos_std

        oos_ret, oos_eq = _simulate_sleeve_pnl(
            w_oos, returns_4h[oos_slice], cost_bps_4h[oos_slice],
            config.cost_stress_multiplier,
        )
        all_oos_nets.append(oos_ret)
        all_oos_equities.append(oos_eq)

        oos_vol = float(np.std(oos_ret[1:], ddof=1) * math.sqrt(2190.0)) if oos_ret[1:].size > 1 else 0.0
        peak = np.maximum.accumulate(oos_eq)
        dd = 1.0 - oos_eq / np.where(peak > 0, peak, 1.0)
        oos_mdd = float(np.max(dd))
        if oos_mdd <= config.max_drawdown and oos_vol <= config.max_ann_vol:
            all_fold_positive += 1

    if not all_oos_nets:
        return HandoffResult(
            forecast=_build_cash_only_forecast(panel.decision_timestamps_ns, panel.symbols),
            evidence=HandoffAdmissionEvidence(
                annualized_log_growth=0.0,
                growth_lcb90=0.0,
                growth_2x_cost=0.0,
                max_drawdown=0.0,
                annual_volatility=0.0,
                positive_outer_folds=0,
                effective_breadth=0.0,
                active_signal_ids=(),
                admitted=False,
                reasons=("no_oos_data",),
            ),
        )

    combined_oos_returns = np.concatenate(all_oos_nets)
    combined_oos_equity = np.concatenate(all_oos_equities)
    n_oos = len(combined_oos_returns)

    ann_factor = 2190.0 / max(n_oos, 1)
    log_rets = np.log1p(np.where(np.isfinite(combined_oos_returns), combined_oos_returns, 0.0))
    ann_growth = float(np.sum(log_rets)) * ann_factor

    lcb90 = _bootstrap_lcb(combined_oos_returns, config.n_bootstrap, rng)

    peak = np.maximum.accumulate(combined_oos_equity)
    dd = 1.0 - combined_oos_equity / np.where(peak > 0, peak, 1.0)
    mdd = float(np.max(dd))

    finite_ret = combined_oos_returns[np.isfinite(combined_oos_returns)]
    if np.any(finite_ret != 0.0) and len(finite_ret) > 1:
        vol = float(np.std(finite_ret, ddof=1) * math.sqrt(2190.0))
    else:
        vol = 0.0

    effective_breadth = float(max(len(all_selected_ids), 1))
    unique_ids = tuple(sorted(all_selected_ids))

    reasons: list[str] = []
    if lcb90 <= 0:
        reasons.append("growth_lcb90_not_positive")
    if mdd > config.max_drawdown:
        reasons.append(f"max_drawdown_{mdd:.4f}_exceeds_{config.max_drawdown}")
    if vol > config.max_ann_vol:
        reasons.append(f"ann_vol_{vol:.4f}_exceeds_{config.max_ann_vol}")
    if all_fold_positive < config.min_positive_outer_folds:
        reasons.append(f"positive_folds_{all_fold_positive}_below_{config.min_positive_outer_folds}")

    admitted = len(reasons) == 0

    if admitted:
        mu_2d = np.zeros((t_total, n_syms), dtype=np.float32)
        family_ids: tuple[str, ...] = ()
        if active_indices:
            family_ids = tuple(sorted({panel.descriptors[k].family for k in active_indices}))
            n_fam = max(len(family_ids), 1)
            sleeve_w: list[NDArray[np.float32]] = []
            for fam in family_ids:
                fam_k = [k for k in active_indices if panel.descriptors[k].family == fam]
                fam_forecasts = a[:, :, fam_k]
                mu_3d_k = np.mean(fam_forecasts, axis=2).astype(np.float32) if len(fam_k) > 1 else fam_forecasts[:, :, 0]
                fvol_arr = np.std(mu_3d_k, axis=0, keepdims=True)
                fvol_arr = np.where(fvol_arr > 1e-12, fvol_arr, 1.0).astype(np.float32)
                sleeve_w.append((mu_3d_k / fvol_arr) * (1.0 / n_fam))
            if sleeve_w:
                mu_2d = np.sum(np.stack(sleeve_w, axis=0), axis=0).astype(np.float32)

        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=panel.decision_timestamps_ns,
            symbols=panel.symbols,
            mu_2d=mu_2d,
            se_2d=np.full((t_total, n_syms), np.nan, dtype=np.float32),
            family_mu_3d=np.zeros((t_total, n_syms, max(len(family_ids), 1)), dtype=np.float32),
            family_ids=family_ids,
            admitted_signal_ids=unique_ids,
            fold_manifest_hash="",
        )
    else:
        forecast = _build_cash_only_forecast(panel.decision_timestamps_ns, panel.symbols)

    evidence = HandoffAdmissionEvidence(
        annualized_log_growth=ann_growth,
        growth_lcb90=lcb90,
        growth_2x_cost=lcb90,
        max_drawdown=mdd,
        annual_volatility=vol,
        positive_outer_folds=all_fold_positive,
        effective_breadth=effective_breadth,
        active_signal_ids=unique_ids,
        admitted=admitted,
        reasons=tuple(reasons),
    )

    _logger.info(
        "handoff: admitted=%s growth=%.6f lcb90=%.6f mdd=%.4f vol=%.4f pos_folds=%d/%d active=%s",
        admitted, ann_growth, lcb90, mdd, vol, all_fold_positive, n_folds, unique_ids,
    )

    return HandoffResult(forecast=forecast, evidence=evidence)
