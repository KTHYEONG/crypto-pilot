from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L1LegConfig, L1RoutingConfig
from src.domain.futures.compound.contracts import (
    CausalFold,
    LegBook,
    RawSignalPanel,
    SignalConceptSpec,
    SymbolLegBook,
)
from src.domain.futures.compound.l1_concept_bank import (
    build_tranche_book,
    build_tranche_target,
    cap_per_name_weights,
)

_logger = logging.getLogger(__name__)


def compute_causal_scale(
    smoothed_2d: NDArray[np.float64],
    config: L1RoutingConfig,
) -> NDArray[np.float64]:
    if smoothed_2d.ndim != 2:
        raise ValueError(f"smoothed_2d must be 2-D, got {smoothed_2d.ndim}")
    n_t, n_s = smoothed_2d.shape
    result = np.zeros((n_t, n_s), dtype=np.float64)
    for t in range(config.normalization_warmup_bars, n_t):
        prior = smoothed_2d[:t]
        std = np.nanstd(prior, axis=0, ddof=1)
        std = np.where(std > 1e-12, std, 1.0)
        raw = smoothed_2d[t] / std
        result[t] = np.clip(raw, -config.signal_clip, config.signal_clip)
    return result


def build_per_symbol_leg_books(
    panel: RawSignalPanel,
    eligible_2d: NDArray[np.bool_],
    registry: tuple[SignalConceptSpec, ...],
    asset_return_2d: NDArray[np.float64],
    cost_bps: float,
    config: L1RoutingConfig,
) -> tuple[SymbolLegBook, ...]:
    if not registry:
        raise ValueError("registry must be non-empty")
    n_t, n_s = eligible_2d.shape
    family_z: dict[str, list[NDArray[np.float32]]] = {}
    for idx, d in enumerate(panel.descriptors):
        family_z.setdefault(d.family, []).append(panel.z_3d[:, :, idx])

    books: list[SymbolLegBook] = []
    for spec in registry:
        member_scores: list[NDArray[np.float32]] = []
        for family_name in spec.member_signal_ids:
            member_scores.extend(family_z.get(family_name, []))
        if not member_scores:
            v = np.zeros((n_t, n_s), dtype=np.float64)
        else:
            stacked = np.stack(member_scores, axis=-1).astype(np.float64)
            nan_cnt = np.sum(~np.isnan(stacked), axis=-1)
            nan_sum = np.nansum(stacked, axis=-1)
            grid_avg = np.divide(nan_sum, nan_cnt, out=np.zeros_like(nan_sum), where=nan_cnt > 0)
            for t in range(n_t):
                row = grid_avg[t]
                abs_sum = float(np.sum(np.abs(row)))
                if abs_sum > 1e-12:
                    grid_avg[t] = row / abs_sum
                else:
                    grid_avg[t] = 0.0
            v = grid_avg
        v_unit = build_tranche_target(v.astype(np.float32), eligible_2d, spec.mode)
        band_books = [build_tranche_book(v_unit, h) for h in spec.horizon_band_bars]
        band_avg = np.mean(band_books, axis=0)
        bk = compute_causal_scale(band_avg, config)
        bk = np.where(eligible_2d, bk, 0.0)

        per_symbol_net = np.zeros((n_t, n_s), dtype=np.float64)
        for t in range(1, n_t):
            cost_turnover = cost_bps * 1e-4 * np.abs(bk[t] - bk[t - 1])
            per_symbol_net[t] = bk[t - 1] * asset_return_2d[t] - cost_turnover

        books.append(SymbolLegBook(
            concept_id=spec.concept_id,
            book_2d=bk,
            per_symbol_net_2d=per_symbol_net,
        ))
    return tuple(books)


def rank_families_causal(
    legs: tuple[LegBook, ...],
    folds: tuple[CausalFold, ...],
    fold_index: int,
    cost_bps: float,
    config: L1RoutingConfig,
) -> tuple[str, ...]:
    if fold_index < 0:
        raise ValueError(f"fold_index must be >= 0, got {fold_index}")
    if not legs:
        raise ValueError("legs must be non-empty")

    prior_slices = [slice(f.oos_start, f.oos_end_exclusive) for f in folds[:fold_index]]
    if not prior_slices:
        return tuple(l.spec.concept_id for l in legs)

    k_ = len(legs)
    scores = np.zeros(k_, dtype=np.float64)
    for k, leg in enumerate(legs):
        net_rets: list[float] = []
        for sl in prior_slices:
            if sl.start >= leg.book_2d.shape[0]:
                continue
            end = min(sl.stop, leg.book_2d.shape[0])
            if end <= sl.start:
                continue
            gross_seg = leg.gross_return_1d[sl.start:end]
            turnover_seg = leg.turnover_1d[sl.start:end]
            cost_seg = cost_bps * 1e-4 * turnover_seg
            net_seg = gross_seg - cost_seg
            finite = np.isfinite(net_seg)
            if np.any(finite):
                net_rets.append(float(np.mean(net_seg[finite])))
        if net_rets:
            scores[k] = float(np.mean(net_rets))
        else:
            scores[k] = -np.inf

    order = np.argsort(-scores)
    top_k = min(config.family_top_k, k_)
    selected = order[:top_k]
    _logger.debug(
        "[ALGO] fold=%d families=%s scores=%s",
        fold_index,
        [legs[i].spec.concept_id for i in selected],
        scores[selected],
    )
    return tuple(legs[i].spec.concept_id for i in selected)


def select_symbols_causal(
    sym_leg: SymbolLegBook,
    folds: tuple[CausalFold, ...],
    fold_index: int,
    config: L1RoutingConfig,
) -> NDArray[np.bool_]:
    if fold_index < 0:
        raise ValueError(f"fold_index must be >= 0, got {fold_index}")

    prior_slices = [slice(f.oos_start, f.oos_end_exclusive) for f in folds[:fold_index]]
    if not prior_slices:
        n_s = sym_leg.per_symbol_net_2d.shape[1]
        return np.ones(n_s, dtype=np.bool_)

    n_s = sym_leg.per_symbol_net_2d.shape[1]
    scores = np.full(n_s, -np.inf, dtype=np.float64)
    for s in range(n_s):
        vals: list[float] = []
        for sl in prior_slices:
            if sl.start >= sym_leg.per_symbol_net_2d.shape[0]:
                continue
            end = min(sl.stop, sym_leg.per_symbol_net_2d.shape[0])
            if end <= sl.start:
                continue
            seg = sym_leg.per_symbol_net_2d[sl.start:end, s]
            finite = np.isfinite(seg)
            if np.any(finite):
                vals.append(float(np.mean(seg[finite])))
        if vals:
            scores[s] = float(np.mean(vals))

    finite_mask = np.isfinite(scores)
    n_finite = int(np.sum(finite_mask))
    if n_finite == 0:
        return np.zeros(n_s, dtype=np.bool_)

    n_select = min(config.symbol_top_n, n_finite)
    top_idx = np.argpartition(-scores, n_select - 1)[:n_select]
    mask = np.zeros(n_s, dtype=np.bool_)
    mask[top_idx] = True
    return mask


def accumulate_symbol_routed_book(
    legs: tuple[LegBook, ...],
    sym_legs: tuple[SymbolLegBook, ...],
    folds: tuple[CausalFold, ...],
    cost_bps: float,
    leg_config: L1LegConfig,
    routing: L1RoutingConfig,
    *,
    fallback_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    if not routing.enabled:
        return fallback_2d.copy()

    sym_map = {s.concept_id: s for s in sym_legs}

    n_t, n_s = fallback_2d.shape
    result = np.zeros((n_t, n_s), dtype=np.float64)

    # Track which rows are routed (not fallback or carry-forward)
    routed_rows = np.zeros(n_t, dtype=np.bool_)

    for i, fold in enumerate(folds):
        oos_slice = slice(fold.oos_start, min(fold.oos_end_exclusive, n_t))
        if oos_slice.start >= oos_slice.stop:
            continue

        if i < routing.min_rank_folds:
            result[oos_slice] = fallback_2d[oos_slice]
            continue

        chosen_families = rank_families_causal(legs, folds, i, cost_bps, routing)
        if not chosen_families:
            result[oos_slice] = fallback_2d[oos_slice]
            continue

        family_books: list[NDArray[np.float64]] = []
        for cid in chosen_families:
            sym_leg = sym_map.get(cid)
            if sym_leg is None:
                continue
            mask = select_symbols_causal(sym_leg, folds, i, routing)
            n_selected = int(np.sum(mask))
            if n_selected == 0:
                continue
            bk = sym_leg.book_2d.copy()
            bk[:, ~mask] = 0.0

            for t in range(oos_slice.start):
                row = bk[t]
                sel = mask & (np.abs(row) > 1e-12)
                n_sel = int(np.sum(sel))
                if n_sel > 0:
                    demean = float(np.mean(row[sel]))
                    bk[t, sel] -= demean
            for t in range(oos_slice.start, oos_slice.stop):
                row = bk[t]
                sel = mask & (np.abs(row) > 1e-12)
                n_sel = int(np.sum(sel))
                if n_sel > 0:
                    demean = float(np.mean(row[sel]))
                    bk[t, sel] -= demean
                    gross = float(np.sum(np.abs(bk[t])))
                    if gross > 1e-12:
                        bk[t] /= gross

            family_books.append(bk)

        if not family_books:
            result[oos_slice] = fallback_2d[oos_slice]
            continue

        stacked = np.stack(family_books, axis=-1)
        route = np.mean(stacked, axis=-1)
        result[oos_slice] = route[oos_slice]
        routed_rows[oos_slice] = True

    # cap only the routed rows
    routed_part = result[routed_rows]
    if len(routed_part) > 0:
        capped = cap_per_name_weights(routed_part, leg_config.max_name_weight)
        result[routed_rows] = capped

    # carry-forward: after last fold, freeze last computed row
    if folds:
        last_end = min(folds[-1].oos_end_exclusive, n_t)
        if last_end < n_t and last_end > 0:
            result[last_end:] = result[last_end - 1:last_end]

    _logger.debug(
        "[ALGO] routed_book: folds=%d families=%s n_symbols=%d",
        len(folds),
        [l.spec.concept_id for l in legs],
        n_s,
    )
    return result
