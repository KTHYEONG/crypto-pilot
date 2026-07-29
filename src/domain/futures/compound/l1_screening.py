from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.stats import norm, rankdata

from src.domain.futures.compound.allocator import (
    compute_dynamic_compounding_path,
)
from src.domain.futures.compound.bootstrap import (
    circular_stationary_bootstrap_growth,
    politis_white_block_length,
)
from src.domain.futures.compound.config import (
    DynamicCompoundingConfig,
    HandoffConfig,
)
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CausalFold,
    FamilyEdgeRecord,
    FamilyEdgeScreen,
    RawSignalPanel,
    SignalEdgeRecord,
    SignalEdgeScreen,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder

_LOGGER = logging.getLogger(__name__)


def compute_cross_sectional_ic(
    z_2d: NDArray[np.float32],
    forward_return_2d: NDArray[np.float64],
    oos_slices: tuple[slice, ...],
    *,
    min_cross_section: int = 8,
) -> NDArray[np.float64]:
    if z_2d.ndim != 2 or forward_return_2d.ndim != 2:
        raise ValueError("z_2d and forward_return_2d must be 2-D")
    if z_2d.shape != forward_return_2d.shape:
        raise ValueError(f"shape mismatch: {z_2d.shape} vs {forward_return_2d.shape}")
    total_bars = z_2d.shape[0]
    ic_1d = np.full(total_bars, np.nan, dtype=np.float64)

    for oos_sl in oos_slices:
        for t in range(oos_sl.start or 0, oos_sl.stop or total_bars):
            z_t = z_2d[t]
            ret_t = forward_return_2d[t]
            valid = np.isfinite(z_t) & np.isfinite(ret_t)
            n_valid = int(np.sum(valid))
            if n_valid < min_cross_section:
                continue
            z_rank = rankdata(z_t[valid])
            ret_rank = rankdata(ret_t[valid])
            n = n_valid
            rho = (np.sum(z_rank * ret_rank) - n * ((n + 1) / 2) ** 2) / (
                n * (n ** 2 - 1) / 12
            )
            ic_1d[t] = np.clip(rho, -1.0, 1.0)
    return ic_1d


def newey_west_tstat(series_1d: NDArray[np.float64], max_lag: int) -> tuple[float, float]:
    n = series_1d.shape[0]
    if n < 30:
        return (0.0, 0.0)
    max_lag = min(max_lag, n // 4)
    mu = float(np.nanmean(series_1d))
    demeaned = series_1d - mu
    gamma0 = float(np.nansum(demeaned ** 2)) / n
    if gamma0 <= 0.0:
        return (0.0, 0.0)
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        gamma_lag = float(np.nansum(demeaned[:-lag] * demeaned[lag:])) / n
        weight = 1.0 - lag / (max_lag + 1)
        nw_var += 2.0 * weight * gamma_lag
    se = 0.0 if nw_var <= 0.0 else math.sqrt(nw_var / n)
    if se <= 0.0:
        return (0.0, 0.0)
    t_stat = mu / se
    return (t_stat, se)


def estimate_effective_independence(ic_matrix_2d: NDArray[np.float64]) -> float:
    if ic_matrix_2d.ndim != 2:
        raise ValueError("ic_matrix_2d must be 2-D")
    n_signals = ic_matrix_2d.shape[1]
    if n_signals < 2:
        return float(n_signals)
    # IC bars outside every OOS slice are NaN in every column identically
    # (compute_cross_sectional_ic seeds the full array with NaN and only fills
    # OOS bars). Requiring column-wise all-finite over those rows too would
    # always fail and collapse n_eff to 1.0, silently disabling the Sidak
    # correction. Drop rows with no data across any signal first.
    row_has_data = np.any(np.isfinite(ic_matrix_2d), axis=1)
    ic_matrix_2d = ic_matrix_2d[row_has_data]
    if ic_matrix_2d.shape[0] < 2:
        return 1.0
    valid_cols = np.all(np.isfinite(ic_matrix_2d), axis=0)
    n_valid = int(np.sum(valid_cols))
    if n_valid < 2:
        return 1.0
    sub = ic_matrix_2d[:, valid_cols]
    col_stds = np.std(sub, axis=0)
    const_cols = col_stds < 1e-12
    if np.all(const_cols):
        return 1.0
    sub = sub[:, ~const_cols]
    corr = np.corrcoef(sub.T)
    eigenvals = linalg.eigvalsh(corr)
    eigenvals = np.maximum(eigenvals, 0.0)
    total = float(np.sum(eigenvals))
    if total <= 0.0:
        return 1.0
    participation_ratio = float(total ** 2 / np.sum(eigenvals ** 2))
    return max(participation_ratio, 1.0)


def screen_family_edge(
    panel: RawSignalPanel,
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    config: HandoffConfig,
) -> FamilyEdgeScreen:
    oos_slices: list[slice] = []
    for fold in folds:
        oos_start = fold.oos_start
        oos_end = fold.oos_end_exclusive
        if oos_end - oos_start > 0:
            oos_slices.append(slice(oos_start, oos_end))

    if not oos_slices:
        return FamilyEdgeScreen(
            records=(),
            n_effective_independent=1.0,
            admitted_families=(),
            admitted_signal_ids=(),
        )

    families: dict[str, list[int]] = {}
    for i, desc in enumerate(panel.descriptors):
        families.setdefault(desc.family, []).append(i)

    records: list[FamilyEdgeRecord] = []

    n_bars = panel.z_3d.shape[0]
    n_syms = panel.z_3d.shape[1]
    log_ret = np.zeros((n_bars, n_syms), dtype=np.float64)
    close = bars_4h.close_2d.astype(np.float64)
    for t in range(1, n_bars):
        prev = close[t - 1]
        curr = close[t]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        log_ret[t, mask] = np.log(curr[mask] / prev[mask])

    close_px = bars_4h.close_2d.astype(np.float64)

    signal_ic_list: list[NDArray[np.float64]] = []
    for desc_idx, desc in enumerate(panel.descriptors):
        horizon_bars = desc.target_horizon_hours // 4
        z_2d = panel.z_3d[:, :, desc_idx].astype(np.float32)
        fwd = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
        for t in range(n_bars - horizon_bars):
            prev_px = close_px[t]
            fut_px = close_px[t + horizon_bars]
            mask = (prev_px > 0) & np.isfinite(prev_px) & (fut_px > 0) & np.isfinite(fut_px)
            fwd[t, mask] = np.log(fut_px[mask] / prev_px[mask])
        ic_1d = compute_cross_sectional_ic(
            z_2d, fwd, tuple(oos_slices), min_cross_section=8,
        )
        signal_ic_list.append(ic_1d)

    ic_matrix = np.column_stack(signal_ic_list) if signal_ic_list else np.zeros((n_bars, 0), dtype=np.float64)

    n_eff = estimate_effective_independence(ic_matrix)

    alpha = config.family_screen_alpha
    n_families = len(families)
    sidak_alpha = alpha / max(n_families, 1) if n_eff <= 0 else 1.0 - (1.0 - alpha) ** (1.0 / max(n_eff, 1.0))

    recorder = L1AdmissionRecorder()
    admitted_families_list: list[str] = []
    admitted_signal_ids_list: list[str] = []

    for family, indices in families.items():
        family_desc = panel.descriptors[indices[0]]
        declared_orientation = family_desc.declared_orientation
        n_sig = len(indices)
        ic_vals: list[float] = []
        n_bars_valid = 0
        for ic_1d in [signal_ic_list[i] for i in indices]:
            valid_ic = ic_1d[np.isfinite(ic_1d)]
            n_bars_valid = max(n_bars_valid, int(valid_ic.shape[0]))
            ic_vals.extend(valid_ic.tolist())

        if n_bars_valid < config.min_family_ic_samples:
            rec = FamilyEdgeRecord(
                family=family, n_signals=n_sig, n_ic_bars=n_bars_valid,
                mean_ic=0.0, t_newey_west=0.0, p_two_sided=1.0,
                sidak_alpha=float(sidak_alpha),
                declared_orientation=declared_orientation,
                admitted=False, reasons=("insufficient_ic_samples",),
            )
            records.append(rec)
            if recorder.enabled:
                recorder.record_family_screen(
                    family=family, n_signals=n_sig, n_ic_bars=n_bars_valid,
                    mean_ic=0.0, t_newey_west=0.0, sidak_alpha=float(sidak_alpha),
                    declared_orientation=declared_orientation, admitted=False,
                    reasons=("insufficient_ic_samples",),
                )
            continue

        ic_arr = np.array(ic_vals, dtype=np.float64)
        mean_ic = float(np.mean(ic_arr))
        max_h = max(desc.target_horizon_hours // 4 for desc in panel.descriptors)
        t_stat, _ = newey_west_tstat(ic_arr, max(1, max_h))

        p_two_sided = 2.0 * norm.sf(abs(t_stat))

        reasons_list: list[str] = []

        if p_two_sided >= sidak_alpha and abs(t_stat) <= 2.0:
            final_admitted = False
            reasons_list.append("not_significant_after_sidak")
        elif t_stat * declared_orientation < 0:
            final_admitted = False
            reasons_list.append("declared_orientation_contradicted")
        elif p_two_sided < sidak_alpha and abs(t_stat) > 2.0:
            final_admitted = True
        else:
            final_admitted = False
            reasons_list.append("not_significant_after_sidak")

        records.append(FamilyEdgeRecord(
            family=family,
            n_signals=n_sig,
            n_ic_bars=n_bars_valid,
            mean_ic=mean_ic,
            t_newey_west=t_stat,
            p_two_sided=p_two_sided,
            sidak_alpha=float(sidak_alpha),
            declared_orientation=declared_orientation,
            admitted=final_admitted,
            reasons=tuple(reasons_list),
        ))

        if recorder.enabled:
            recorder.record_family_screen(
                family=family, n_signals=n_sig, n_ic_bars=n_bars_valid,
                mean_ic=mean_ic, t_newey_west=t_stat,
                sidak_alpha=float(sidak_alpha),
                declared_orientation=declared_orientation,
                admitted=final_admitted,
                reasons=tuple(reasons_list),
            )

        if final_admitted:
            admitted_families_list.append(family)
            admitted_signal_ids_list.extend(
                panel.descriptors[i].signal_id for i in indices
            )

    return FamilyEdgeScreen(
        records=tuple(records),
        n_effective_independent=n_eff,
        admitted_families=tuple(admitted_families_list),
        admitted_signal_ids=tuple(admitted_signal_ids_list),
    )


_BARS_PER_YEAR_4H_SCREEN: float = 2190.0


def replay_signal_standalone_book(
    z_2d: NDArray[np.float32], panel: RawSignalPanel, bars_4h: TimeframeBarCube,
    funding_1h_2d: NDArray[np.float32], oos_slices: tuple[slice, ...],
    allocator_config: DynamicCompoundingConfig, cost_bps: float,
    *,
    declared_orientation: int = 1,
) -> tuple[NDArray[np.float64], float]:
    n_bars, n_syms = z_2d.shape
    mu_2d = (z_2d * declared_orientation).astype(np.float64)
    mini_panel = CalibratedForecastPanel(
        decision_timestamps_ns=panel.decision_timestamps_ns,
        symbols=panel.symbols,
        mu_2d=mu_2d.astype(np.float32),
        se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
        family_ids=("standalone",),
        admitted_signal_ids=("standalone",),
        fold_manifest_hash="",
    )
    weights_2d = compute_dynamic_compounding_path(
        forecast=mini_panel,
        sigma_2d=panel.sigma_2d,
        funding_rates_1h_2d=funding_1h_2d,
        config=allocator_config,
        close_2d=bars_4h.close_2d,
        cost_bps=cost_bps,
    )
    if not np.all(np.isfinite(weights_2d)):
        raise ValueError("non_finite_standalone_book")

    close_px = bars_4h.close_2d.astype(np.float64)
    asset_return_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
    for t in range(1, n_bars):
        prev = close_px[t - 1]
        curr = close_px[t]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        asset_return_2d[t, mask] = curr[mask] / prev[mask] - 1.0

    oos_net_list: list[float] = []
    total_turnover = 0.0
    n_oos_bars = 0
    for oos_sl in oos_slices:
        for t in range(oos_sl.start, oos_sl.stop):
            w = weights_2d[t]
            prev_w = weights_2d[t - 1] if t > 0 else np.zeros(n_syms, dtype=np.float64)
            ret_t = asset_return_2d[t + 1] if t + 1 < n_bars else np.zeros(n_syms, dtype=np.float64)
            port_ret = float(np.dot(w, ret_t))
            turnover = float(np.sum(np.abs(w - prev_w)))
            cost_t = cost_bps * 1e-4 * turnover
            oos_net_list.append(port_ret - cost_t)
            total_turnover += turnover
            n_oos_bars += 1

    net_1d = np.array(oos_net_list, dtype=np.float64)
    mean_turnover = total_turnover / max(n_oos_bars, 1)
    return net_1d, mean_turnover


def screen_signal_net_edge(
    net_1d: NDArray[np.float64], config: HandoffConfig,
) -> tuple[bool, float, float]:
    finite = net_1d[np.isfinite(net_1d)]
    n = int(finite.shape[0])
    if n < config.min_family_ic_samples:
        return False, 0.0, 0.0
    ann_net = float(np.mean(finite)) * _BARS_PER_YEAR_4H_SCREEN
    try:
        block_length = politis_white_block_length(finite)
    except ValueError:
        block_length = 5.0
    _, _, prob_positive = circular_stationary_bootstrap_growth(
        finite, _BARS_PER_YEAR_4H_SCREEN,
        n_bootstrap=config.n_bootstrap,
        block_size=block_length,
        seed=42,
    )
    passes = bool(prob_positive >= config.min_growth_posterior_probability and ann_net > 0)
    return passes, prob_positive, ann_net


def discover_effective_horizon(
    weights_2d: NDArray[np.float64],
    bars_4h: TimeframeBarCube,
    oos_slices: tuple[slice, ...],
    search_orientation: int,
    candidate_horizons_hours: tuple[int, ...] = (24, 48, 96, 144, 216, 432, 648),
    family_screen_alpha: float = 0.05,
) -> tuple[int, int, float]:
    """Returns (effective_horizon_hours, effective_orientation, t_stat).
    (0, 0, 0.0) when no candidate clears the Sidak-corrected threshold.

    [RULE-EH-3] Uses weights_2d (smoothed distribution book, not raw z).
    [RULE-EH-4] Sidak correction across candidate_horizons_hours.
    [LIMIT-03] All-zero weights_2d -> (0,0,0.0).
    """
    if not np.any(np.isfinite(weights_2d)):
        return (0, 0, 0.0)
    finite_weights = weights_2d[np.isfinite(weights_2d)]
    if finite_weights.shape[0] == 0 or float(np.max(np.abs(finite_weights))) < 1e-12:
        return (0, 0, 0.0)

    n_horizons = len(candidate_horizons_hours)
    sidak_alpha_eh = 1.0 - (1.0 - family_screen_alpha) ** (1.0 / max(n_horizons, 1))

    n_bars, n_syms = weights_2d.shape
    close_px = bars_4h.close_2d.astype(np.float64)

    best_horizon = 0
    best_t_stat = 0.0
    best_orientation = 0

    for horizon_hours in candidate_horizons_hours:
        horizon_bars = horizon_hours // 4
        if horizon_bars < 1 or horizon_bars >= n_bars:
            continue
        fwd = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
        for t in range(n_bars - horizon_bars):
            prev_px = close_px[t]
            fut_px = close_px[t + horizon_bars]
            mask = (prev_px > 0) & np.isfinite(prev_px) & (fut_px > 0) & np.isfinite(fut_px)
            fwd[t, mask] = np.log(fut_px[mask] / prev_px[mask])

        ic_1d = compute_cross_sectional_ic(
            weights_2d.astype(np.float32), fwd, oos_slices, min_cross_section=8,
        )
        valid_ic = ic_1d[np.isfinite(ic_1d)]
        if valid_ic.shape[0] < 8:
            continue

        t_stat, _ = newey_west_tstat(valid_ic, max(1, horizon_bars))
        p_val = 2.0 * norm.sf(abs(t_stat))

        if t_stat * search_orientation > 0 and p_val < sidak_alpha_eh and (best_horizon == 0 or horizon_hours < best_horizon):
            best_horizon = horizon_hours
            best_t_stat = t_stat
            best_orientation = search_orientation

    return (best_horizon, best_orientation, best_t_stat)


def screen_signal_edge(
    panel: RawSignalPanel,
    bars_4h: TimeframeBarCube,
    folds: tuple[CausalFold, ...],
    config: HandoffConfig,
    *,
    funding_1h_2d: NDArray[np.float32],
    allocator_config: DynamicCompoundingConfig,
) -> SignalEdgeScreen:
    oos_slices: list[slice] = []
    for fold in folds:
        oos_start = fold.oos_start
        oos_end = fold.oos_end_exclusive
        if oos_end - oos_start > 0:
            oos_slices.append(slice(oos_start, oos_end))

    if not oos_slices:
        return SignalEdgeScreen(
            records=(),
            n_effective_independent=1.0,
            admitted_signal_ids=(),
            admitted_families=(),
        )

    n_bars = panel.z_3d.shape[0]
    n_syms = panel.z_3d.shape[1]
    close_px = bars_4h.close_2d.astype(np.float64)

    signal_ic_list: list[NDArray[np.float64]] = []
    for desc_idx, desc in enumerate(panel.descriptors):
        horizon_bars = desc.target_horizon_hours // 4
        z_2d = panel.z_3d[:, :, desc_idx].astype(np.float32)
        fwd = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
        for t in range(n_bars - horizon_bars):
            prev_px = close_px[t]
            fut_px = close_px[t + horizon_bars]
            mask = (prev_px > 0) & np.isfinite(prev_px) & (fut_px > 0) & np.isfinite(fut_px)
            fwd[t, mask] = np.log(fut_px[mask] / prev_px[mask])
        ic_1d = compute_cross_sectional_ic(
            z_2d, fwd, tuple(oos_slices), min_cross_section=8,
        )
        signal_ic_list.append(ic_1d)

    ic_matrix = np.column_stack(signal_ic_list) if signal_ic_list else np.zeros((n_bars, 0), dtype=np.float64)

    n_eff = estimate_effective_independence(ic_matrix)

    alpha = config.family_screen_alpha
    n_signals = len(panel.descriptors)
    sidak_alpha = 1.0 - (1.0 - alpha) ** (1.0 / max(n_eff, 1.0)) if n_eff > 0 else alpha / max(n_signals, 1)

    recorder = L1AdmissionRecorder()
    records: list[SignalEdgeRecord] = []
    admitted_signal_ids_list: list[str] = []
    admitted_families_set: set[str] = set()

    for desc_idx, desc in enumerate(panel.descriptors):
        signal_id = desc.signal_id
        family = desc.family
        speed = desc.speed
        target_horizon_hours = desc.target_horizon_hours
        declared_orientation = desc.declared_orientation
        max_lag = target_horizon_hours // 4

        ic_1d = signal_ic_list[desc_idx]
        valid_ic = ic_1d[np.isfinite(ic_1d)]
        n_ic_bars = int(valid_ic.shape[0])

        if n_ic_bars < config.min_family_ic_samples:
            records.append(SignalEdgeRecord(
                signal_id=signal_id, family=family, speed=speed,
                target_horizon_hours=target_horizon_hours,
                n_ic_bars=n_ic_bars, mean_ic=0.0, t_newey_west=0.0,
                p_two_sided=1.0, sidak_alpha=float(sidak_alpha),
                declared_orientation=declared_orientation,
                admitted=False, reasons=("insufficient_ic_samples",),
            ))
            if recorder.enabled:
                recorder.record_family_screen(
                    family=signal_id, n_signals=1, n_ic_bars=n_ic_bars,
                    mean_ic=0.0, t_newey_west=0.0, sidak_alpha=float(sidak_alpha),
                    declared_orientation=declared_orientation, admitted=False,
                    reasons=("insufficient_ic_samples",),
                )
            continue

        mean_ic = float(np.mean(valid_ic))
        t_stat, _ = newey_west_tstat(valid_ic, max(1, max_lag))
        p_two_sided = 2.0 * norm.sf(abs(t_stat))

        reasons_list: list[str] = []

        if p_two_sided >= sidak_alpha:
            final_admitted = False
            reasons_list.append("not_significant_after_sidak")
        elif t_stat * declared_orientation < 0:
            final_admitted = False
            reasons_list.append("declared_orientation_contradicted")
        else:
            final_admitted = True

        # P1: 4th gate — net edge screen (C3 replay)
        eh_hours = 0
        eh_orientation = 0
        eh_t_stat_val = 0.0

        if final_admitted:
            try:
                z_2d_signal = panel.z_3d[:, :, desc_idx].astype(np.float32)
                net_1d, turnover = replay_signal_standalone_book(
                    z_2d_signal, panel, bars_4h, funding_1h_2d,
                    tuple(oos_slices), allocator_config, config.screen_cost_bps,
                    declared_orientation=declared_orientation,
                )
                net_passes, net_prob, net_ann = screen_signal_net_edge(net_1d, config)
                if not net_passes:
                    final_admitted = False
                    reasons_list.append("net_edge_not_significant_after_cost")
            except ValueError:
                final_admitted = False
                reasons_list.append("net_edge_replay_failed")
                net_prob = 0.0
                net_ann = 0.0
                turnover = 0.0
        else:
            net_prob = 0.0
            net_ann = 0.0
            turnover = 0.0

            # P5: effective horizon search for eligible IC-failing signals
            if reasons_list and reasons_list[0] in (
                "not_significant_after_sidak", "declared_orientation_contradicted",
            ):
                search_orientation = (
                    declared_orientation
                    if reasons_list[0] == "not_significant_after_sidak"
                    else -declared_orientation
                )
                try:
                    z_2d_signal = panel.z_3d[:, :, desc_idx].astype(np.float32)
                    mu_2d = z_2d_signal.astype(np.float64) * search_orientation
                    mini_panel = CalibratedForecastPanel(
                        decision_timestamps_ns=panel.decision_timestamps_ns,
                        symbols=panel.symbols,
                        mu_2d=mu_2d.astype(np.float32),
                        se_2d=np.full((n_bars, n_syms), np.nan, dtype=np.float32),
                        family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
                        family_ids=("standalone",),
                        admitted_signal_ids=("standalone",),
                        fold_manifest_hash="",
                    )
                    weights_2d = compute_dynamic_compounding_path(
                        forecast=mini_panel,
                        sigma_2d=panel.sigma_2d,
                        funding_rates_1h_2d=funding_1h_2d,
                        config=allocator_config,
                        close_2d=bars_4h.close_2d,
                        cost_bps=config.screen_cost_bps,
                    )
                    eh_hours, eh_orientation, eh_t_stat_val = discover_effective_horizon(
                        weights_2d, bars_4h, tuple(oos_slices),
                        search_orientation,
                    )
                    if eh_hours > 0:
                        net_1d_eh, turnover_eh = replay_signal_standalone_book(
                            z_2d_signal, panel, bars_4h, funding_1h_2d,
                            tuple(oos_slices), allocator_config,
                            config.screen_cost_bps,
                            declared_orientation=eh_orientation,
                        )
                        net_passes, net_prob, net_ann = screen_signal_net_edge(
                            net_1d_eh, config,
                        )
                        if net_passes:
                            final_admitted = True
                            reasons_list = []
                        else:
                            reasons_list.append(
                                "net_edge_not_significant_after_cost",
                            )
                        turnover = turnover_eh
                    else:
                        reasons_list.append("no_effective_horizon_found")
                except (ValueError, RuntimeError):
                    reasons_list.append("net_edge_replay_failed")
                    net_prob = 0.0
                    net_ann = 0.0
                    turnover = 0.0

        edge_per_turn = (net_ann / max(turnover * _BARS_PER_YEAR_4H_SCREEN, 1e-12)) * 1e4 if turnover > 0 else 0.0

        records.append(SignalEdgeRecord(
            signal_id=signal_id, family=family, speed=speed,
            target_horizon_hours=target_horizon_hours,
            n_ic_bars=n_ic_bars, mean_ic=mean_ic,
            t_newey_west=t_stat, p_two_sided=p_two_sided,
            sidak_alpha=float(sidak_alpha),
            declared_orientation=declared_orientation,
            admitted=final_admitted,
            reasons=tuple(reasons_list),
            intrinsic_turnover_per_bar=turnover,
            net_growth_ann=net_ann,
            net_growth_probability=net_prob,
            edge_per_turnover_bps=edge_per_turn,
            effective_horizon_hours=eh_hours,
            effective_orientation=eh_orientation,
            effective_horizon_t_stat=eh_t_stat_val,
        ))

        if recorder.enabled:
            recorder.record_family_screen(
                family=signal_id, n_signals=1, n_ic_bars=n_ic_bars,
                mean_ic=mean_ic, t_newey_west=t_stat,
                sidak_alpha=float(sidak_alpha),
                declared_orientation=declared_orientation,
                admitted=final_admitted,
                reasons=tuple(reasons_list),
                intrinsic_turnover_per_bar=turnover,
                net_growth_ann=net_ann,
                net_growth_probability=net_prob,
                edge_per_turnover_bps=edge_per_turn,
                effective_horizon_hours=eh_hours,
                effective_orientation=eh_orientation,
                effective_horizon_t_stat=eh_t_stat_val,
            )

        if final_admitted:
            admitted_signal_ids_list.append(signal_id)
            admitted_families_set.add(family)

    return SignalEdgeScreen(
        records=tuple(records),
        n_effective_independent=n_eff,
        admitted_signal_ids=tuple(admitted_signal_ids_list),
        admitted_families=tuple(sorted(admitted_families_set)),
    )
