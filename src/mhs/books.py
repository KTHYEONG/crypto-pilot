"""Rank-weight books and the phase-tranche combination.

Fast and slow ranks/signals are constructed independently; the only Phase 1
combination is the preregistered 50/50 portfolio allocation (see
``PHASE_1_BOOK_BLEND_WEIGHTS``). A shared TrendScore or performance-selected
blend is signal pooling and is prohibited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_realized_vol_tilt(
    weights: pd.DataFrame, realized_vol: pd.DataFrame,
) -> pd.DataFrame:
    """Scale an existing weight book's magnitudes inversely to trailing realized vol.

    Cells with a non-finite or non-positive ``realized_vol`` fall back to a
    tilt of 1.0 (no scaling) rather than an undefined or infinite weight,
    mirroring ``_causal_family_inverse_vol_weights``'s never-NaN fallback
    (``src/research/technical_experts/cross_sectional.py``). This is an
    unnormalized intermediate -- callers renormalize afterward (e.g. via
    ``renormalize_within_mask``) to restore dollar-neutral/unit-gross.
    """
    if not weights.index.equals(realized_vol.index) or list(weights.columns) != list(realized_vol.columns):
        raise ValueError("weights and realized_vol must be identically indexed and columned")
    vol = realized_vol.to_numpy(dtype="float64")
    valid = np.isfinite(vol) & (vol > 0.0)
    tilt = np.where(valid, 1.0 / np.where(valid, vol, 1.0), 1.0)
    return weights * tilt


def rank_weight_book(
    signal: pd.DataFrame,
    eligible: pd.DataFrame,
    sign: int,
    min_symbols: int,
) -> pd.DataFrame:
    """Dollar-neutral, unit-gross rank weights across eligible symbols.

    Percentile-ranks each row, subtracts the row mean of the percentile ranks
    (exact dollar neutrality; rank(pct) has mean ``(n+1)/(2n)``, so subtracting
    the constant 0.5 would import a net long/short tilt), multiplies by
    ``sign``, then divides by the row gross so ``abs`` weights sum to 1.0.
    Rows with fewer than ``min_symbols`` eligible symbols (or zero gross)
    return all zeros.
    """
    if sign not in (-1, 1):
        raise ValueError(f"sign must be -1 or +1, got {sign}")
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    if not signal.index.equals(eligible.index) or list(signal.columns) != list(eligible.columns):
        raise ValueError("signal and eligible must be identically indexed and columned")

    masked = signal.where(eligible)
    ranks = masked.rank(axis=1, pct=True)
    centered = ranks.sub(ranks.mean(axis=1), axis=0).mul(sign)
    gross = centered.abs().sum(axis=1)
    out = centered.div(gross.where(gross > 0), axis=0)
    enough = eligible.sum(axis=1) >= min_symbols
    return out.where(enough, other=0.0).fillna(0.0)


def renormalize_within_mask(
    weights: pd.DataFrame, mask: pd.DataFrame, min_symbols: int,
) -> pd.DataFrame:
    """Re-center and re-normalize an existing weight book onto a boolean subset.

    Restores the dollar-neutral (row sum 0), unit-gross (row abs-sum 1)
    invariant within the surviving ``mask`` columns after they have been
    selected from a wider book (e.g. an execution roster). Columns outside
    ``mask`` are excluded from the row mean/gross (not treated as 0, which
    would bias the center); rows with fewer than ``min_symbols`` True mask
    cells return all zeros, mirroring ``rank_weight_book``.
    """
    if min_symbols < 2:
        raise ValueError(f"min_symbols must be >= 2, got {min_symbols}")
    if not weights.index.equals(mask.index) or list(weights.columns) != list(mask.columns):
        raise ValueError("weights and mask must be identically indexed and columned")
    masked = weights.where(mask)
    centered = masked.sub(masked.mean(axis=1), axis=0)
    gross = centered.abs().sum(axis=1)
    out = centered.div(gross.where(gross > 0), axis=0)
    enough = mask.sum(axis=1) >= min_symbols
    return out.where(enough, other=0.0).fillna(0.0)


def phase_tranche_book(weights: pd.DataFrame, tranche_count: int) -> pd.DataFrame:
    """Combine ``tranche_count`` staggered single-phase books.

    The combined target is the trailing mean of the last ``tranche_count``
    step-grid weight rows (``min_periods=tranche_count``), with leading rows
    filled with 0.0. ``tranche_count == 1`` is the identity single-phase book,
    permitted only for diagnostics.
    """
    if tranche_count < 1:
        raise ValueError(f"tranche_count must be >= 1, got {tranche_count}")
    return weights.rolling(tranche_count, min_periods=tranche_count).mean().fillna(0.0)
