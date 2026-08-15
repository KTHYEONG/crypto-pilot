from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import (
    LegBook,
    RawSignalPanel,
    SignalConceptSpec,
    SignalDescriptor,
)

_logger = logging.getLogger(__name__)


_DEFAULT_REGISTRY: tuple[SignalConceptSpec, ...] = (
    SignalConceptSpec(
        concept_id="trend_ema", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("trend_ema",),
    ),
    SignalConceptSpec(
        concept_id="momentum_ts", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("momentum_ts",),
    ),
    SignalConceptSpec(
        concept_id="breakout_donchian", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("breakout_donchian",),
    ),
    SignalConceptSpec(
        concept_id="rsi", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("rsi",),
    ),
    SignalConceptSpec(
        concept_id="mfi", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("mfi",),
    ),
    SignalConceptSpec(
        concept_id="aroon_oscillator", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("aroon_oscillator",),
    ),
    SignalConceptSpec(
        concept_id="adx_directional", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("adx_directional",),
    ),
    SignalConceptSpec(
        concept_id="obv_trend", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("obv_trend",),
    ),
    SignalConceptSpec(
        concept_id="keltner_breakout", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("keltner_breakout",),
    ),
    SignalConceptSpec(
        concept_id="volume_zscore", mode="ts", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("volume_zscore",),
    ),
    SignalConceptSpec(
        concept_id="bollinger_bandwidth", mode="ts", horizon_band_bars=(6, 12, 24),
        declared_orientation=1, member_signal_ids=("bollinger_bandwidth",),
    ),
)


def build_concept_registry(
    descriptors: tuple[SignalDescriptor, ...],
    config: L1LegConfig,
) -> tuple[SignalConceptSpec, ...]:
    known_families = {d.family for d in descriptors}
    for spec in _DEFAULT_REGISTRY:
        for fam in spec.member_signal_ids:
            if fam not in known_families:
                raise ValueError(
                    f"concept {spec.concept_id!r} references member family {fam!r} "
                    f"which does not exist in descriptors"
                )
    return _DEFAULT_REGISTRY


def build_tranche_target(
    z_2d: NDArray[np.float32],
    eligible_2d: NDArray[np.bool_],
    mode: str,
    min_cross_section: int = 10,
) -> NDArray[np.float64]:
    if mode not in ("xs", "ts"):
        raise ValueError(f"mode must be 'xs' or 'ts', got {mode!r}")
    n_t, n_s = z_2d.shape
    v = np.zeros((n_t, n_s), dtype=np.float64)
    for t in range(n_t):
        eligible = eligible_2d[t]
        n_eligible = int(np.sum(eligible))
        if n_eligible < min_cross_section:
            continue
        z_t = z_2d[t].astype(np.float64)
        vals = z_t[eligible]
        vals = np.where(np.isfinite(vals), vals, 0.0)
        if mode == "xs":
            vals = vals - np.mean(vals)
        abs_sum = float(np.sum(np.abs(vals)))
        vals = vals / abs_sum if abs_sum > 1e-12 else np.zeros_like(vals)
        v[t, eligible] = vals
    return v


def build_tranche_book(
    v_2d: NDArray[np.float64],
    horizon_bars: int,
) -> NDArray[np.float64]:
    if horizon_bars <= 0:
        raise ValueError(f"horizon_bars must be > 0, got {horizon_bars}")
    n_t, n_s = v_2d.shape
    book = np.zeros((n_t, n_s), dtype=np.float64)
    if horizon_bars == 1:
        return v_2d.copy()
    cumsum = np.zeros((n_t + 1, n_s), dtype=np.float64)
    cumsum[1:] = np.cumsum(v_2d, axis=0)
    for t in range(n_t):
        start = max(0, t - horizon_bars + 1)
        n = t - start + 1
        book[t] = (cumsum[t + 1] - cumsum[start]) / n
    return book


def compute_lagged_gross_returns(
    book_2d: NDArray[np.float64], asset_return_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    if book_2d.shape != asset_return_2d.shape:
        raise ValueError(
            f"book_2d shape {book_2d.shape} != asset_return_2d shape {asset_return_2d.shape}",
        )
    n_t = book_2d.shape[0]
    g = np.zeros(n_t, dtype=np.float64)
    g[1:] = np.einsum("ts,ts->t", book_2d[:-1], asset_return_2d[1:])
    return g


def cap_per_name_weights(
    book_2d: NDArray[np.float64], max_name_weight: float, max_iter: int = 8,
) -> NDArray[np.float64]:
    if not (0.0 < max_name_weight <= 1.0):
        raise ValueError(f"max_name_weight must be in (0, 1], got {max_name_weight}")
    result = book_2d.copy()
    abs_w = np.abs(result)
    gross = np.sum(abs_w, axis=1, keepdims=True)
    gross = np.where(gross > 1e-12, gross, 1.0)
    abs_w = abs_w / gross
    result = result / gross
    for _ in range(max_iter):
        exceed = abs_w > max_name_weight
        if not np.any(exceed):
            break
        clamped = np.where(exceed, max_name_weight, abs_w)
        surplus = np.sum(np.where(exceed, abs_w - max_name_weight, 0.0), axis=1, keepdims=True)
        unsaturated = ~exceed
        unsaturated_sum = np.sum(abs_w * unsaturated, axis=1, keepdims=True)
        unsaturated_sum = np.where(unsaturated_sum > 1e-12, unsaturated_sum, 1.0)
        redistribution = surplus * (abs_w * unsaturated) / unsaturated_sum
        abs_w = clamped + redistribution
    sign = np.sign(result)
    result = sign * abs_w
    return result


def build_leg_books(
    panel: RawSignalPanel,
    eligible_2d: NDArray[np.bool_],
    close_2d: NDArray[np.float32],
    registry: tuple[SignalConceptSpec, ...],
    config: L1LegConfig,
) -> tuple[LegBook, ...]:
    # registry member_signal_ids are FAMILY names (e.g. "rsi"), not full
    # signal_id ("rsi:fast"); a family's full lookback ladder is grid-averaged
    # together, which is also where the RULE-04 lookback averaging happens.
    family_z: dict[str, list[NDArray[np.float32]]] = {}
    for idx, d in enumerate(panel.descriptors):
        family_z.setdefault(d.family, []).append(panel.z_3d[:, :, idx])

    n_t, n_s = eligible_2d.shape
    close_f64 = close_2d.astype(np.float64)
    ret_2d = np.zeros((n_t, n_s), dtype=np.float64)
    for t in range(1, n_t):
        prev = close_f64[t - 1]
        curr = close_f64[t]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_2d[t, mask] = np.log(curr[mask] / prev[mask])

    legs: list[LegBook] = []
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
                if abs_sum > 1e-12 and int(np.sum(eligible_2d[t])) >= config.min_cross_section:
                    grid_avg[t] = row / abs_sum
                else:
                    grid_avg[t] = 0.0
            v = grid_avg
        v_unit = build_tranche_target(v.astype(np.float32), eligible_2d, spec.mode, config.min_cross_section)
        # RULE-04: band-average over every horizon in the frozen band, not
        # just the fastest one -- this is the object the spec's prequential
        # result was measured on.
        band_books = [build_tranche_book(v_unit, h) for h in spec.horizon_band_bars]
        band_avg = np.mean(band_books, axis=0)
        book = np.zeros_like(band_avg)
        for t in range(n_t):
            abs_sum = float(np.sum(np.abs(band_avg[t])))
            book[t] = band_avg[t] / abs_sum if abs_sum > 1e-12 else 0.0
        book = cap_per_name_weights(book, config.max_name_weight)
        turnover = np.zeros(n_t, dtype=np.float64)
        for t in range(1, n_t):
            turnover[t] = float(np.sum(np.abs(book[t] - book[t - 1])))
        gross_return = compute_lagged_gross_returns(book, ret_2d)
        legs.append(LegBook(
            spec=spec,
            book_2d=book,
            gross_return_1d=gross_return,
            turnover_1d=turnover,
        ))
    return tuple(legs)
