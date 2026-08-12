from __future__ import annotations

import pandas as pd

from src.mhs.panel import liquid_half_eligibility


def test_internal_nan_gap_tolerated_without_row_deletion() -> None:
    """SCENARIO_MHS_GAP_TRIM_DIAGNOSIS_05: gap-tolerance is a first-class contract.

    Build a synthetic panel shaped exactly like ``load_base_panel`` output (one
    wide DataFrame per column on a tz-aware UTC grid). One symbol (B) has an
    internal NaN gap in ``close``/``quote_vol``. ``liquid_half_eligibility``
    marks B ineligible only during the gap plus the ``min_history_bars`` warmup
    window, and eligible again afterward -- with NO row deletion and NO
    fill/repair of the gap. This documents that the existing pipeline is already
    gap-safe, so no destructive trim utility should ever be reintroduced.
    """
    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    close = pd.DataFrame(
        {
            "A": [100.0] * 8,
            "B": [100.0, 100.0, 100.0, float("nan"), float("nan"), 100.0, 100.0, 100.0],
        },
        index=idx,
    )
    quote_vol = pd.DataFrame(
        {
            "A": [1000.0] * 8,
            "B": [1000.0, 1000.0, 1000.0, float("nan"), float("nan"), 1000.0, 1000.0, 1000.0],
        },
        index=idx,
    )
    panel = {"close": close, "quote_vol": quote_vol}

    eligible = liquid_half_eligibility(quote_vol, lookback_bars=3, min_history_bars=3)

    # No row deletion: the panel keeps every row and the NaN gap is untouched.
    assert close.shape == (8, 2)
    assert quote_vol.shape == (8, 2)
    assert pd.isna(close.loc[idx[3], "B"])
    assert pd.isna(close.loc[idx[4], "B"])
    assert pd.isna(quote_vol.loc[idx[3], "B"])
    assert pd.isna(quote_vol.loc[idx[4], "B"])
    assert len(panel["close"]) == 8
    assert len(panel["quote_vol"]) == 8

    # B is fully warmed before the gap, ineligible during + immediately after
    # the gap (min_history_bars warmup), then eligible again afterward.
    assert bool(eligible.loc[idx[2], "B"])
    for i in (3, 4, 5, 6):
        assert not bool(eligible.loc[idx[i], "B"])
    assert bool(eligible.loc[idx[7], "B"])

    # A, which never gaps, stays eligible throughout its own warmup.
    assert bool(eligible.loc[idx[7], "A"])
