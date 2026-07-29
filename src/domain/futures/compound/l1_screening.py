from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.stats import norm, rankdata

from src.domain.futures.compound.config import HandoffConfig
from src.domain.futures.compound.contracts import (
    CausalFold,
    FamilyEdgeRecord,
    FamilyEdgeScreen,
    RawSignalPanel,
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
