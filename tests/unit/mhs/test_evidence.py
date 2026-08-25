"""Evidence-layer unit contract: selection-window overlap disclosure (I1)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.mhs.evidence import selection_overlap_fraction
from src.mhs.params import DEFAULT_SELECTION_WINDOW

_UTC = "UTC"


def test_selection_window_is_the_registered_defaults_span() -> None:
    """The disclosure denominator window matches the span the CLI defaults
    (growth_extreme, committee_kelly_sizing, breadth 60) were measured on."""
    registered = DEFAULT_SELECTION_WINDOW
    assert registered == (
        pd.Timestamp("2021-01-01", tz=_UTC),
        pd.Timestamp("2025-12-31", tz=_UTC),
    )


def test_full_containment_reports_exact_one() -> None:
    fraction = selection_overlap_fraction(
        pd.Timestamp("2021-01-01", tz=_UTC), pd.Timestamp("2025-12-31", tz=_UTC)
    )
    assert fraction == 1.0


def test_disjoint_and_zero_length_windows_report_zero() -> None:
    after = selection_overlap_fraction(
        pd.Timestamp("2026-01-01", tz=_UTC), pd.Timestamp("2026-12-31", tz=_UTC)
    )
    before = selection_overlap_fraction(
        pd.Timestamp("2019-01-01", tz=_UTC), pd.Timestamp("2020-12-31", tz=_UTC)
    )
    zero_length = selection_overlap_fraction(
        pd.Timestamp("2024-06-01", tz=_UTC), pd.Timestamp("2024-06-01", tz=_UTC)
    )
    assert after == 0.0
    assert before == 0.0
    assert zero_length == 0.0


def test_partial_overlap_is_fractional_and_clipped() -> None:
    start, end = DEFAULT_SELECTION_WINDOW
    half = selection_overlap_fraction(start, end + (end - start))
    assert half == pytest.approx(0.5)


def test_inverted_window_fails_closed() -> None:
    with pytest.raises(ValueError, match="report_end"):
        selection_overlap_fraction(
            pd.Timestamp("2026-01-01", tz=_UTC), pd.Timestamp("2025-01-01", tz=_UTC)
        )
