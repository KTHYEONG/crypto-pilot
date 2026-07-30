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
        concept_id="trend_momentum", mode="xs", horizon_band_bars=(6, 12, 24),
        declared_orientation=1,
        member_signal_ids=(
            "trend_ema", "momentum_ts", "breakout_donchian",
            "rsi", "cci", "mfi", "aroon_oscillator",
            "adx_directional", "obv_trend", "keltner_breakout",
        ),
    ),
    SignalConceptSpec(
        concept_id="vol_regime", mode="ts", horizon_band_bars=(6, 12, 24),
        declared_orientation=1,
        member_signal_ids=("volume_zscore", "bollinger_bandwidth", "volatility_squeeze_keltner"),
    ),
)


def build_concept_registry(
    descriptors: tuple[SignalDescriptor, ...],
    config: L1LegConfig,
) -> tuple[SignalConceptSpec, ...]:
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
            grid_avg = np.nanmean(stacked, axis=-1)
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
        turnover = np.zeros(n_t, dtype=np.float64)
        for t in range(1, n_t):
            turnover[t] = float(np.sum(np.abs(book[t] - book[t - 1])))
        gross_return = np.zeros(n_t, dtype=np.float64)
        for t in range(1, n_t):
            gross_return[t] = float(np.dot(book[t], ret_2d[t]))
        legs.append(LegBook(
            spec=spec,
            book_2d=book,
            gross_return_1d=gross_return,
            turnover_1d=turnover,
        ))
    return tuple(legs)
