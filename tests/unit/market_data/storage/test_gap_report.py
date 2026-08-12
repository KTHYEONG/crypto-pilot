from __future__ import annotations

import pandas as pd
import pytest

from src.market_data.storage.gap_report import detect_internal_gaps


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


class TestDetectInternalGaps:
    """SCENARIO_MHS_GAP_TRIM_DIAGNOSIS_01: a single internal False run between a
    symbol's own first and last True is reported exactly once, with inclusive
    start/end timestamps and length equal to the count of False bars."""

    def test_single_internal_gap_one_false_bar(self) -> None:
        valid = pd.DataFrame({"A": [True, True, False, True, True]}, index=_index(5))
        gaps = detect_internal_gaps(valid)
        assert gaps == {"A": [(pd.Timestamp("2024-01-01 02:00", tz="UTC"), pd.Timestamp("2024-01-01 02:00", tz="UTC"), 1)]}

    def test_single_internal_gap_two_false_bars(self) -> None:
        valid = pd.DataFrame({"A": [True, True, False, False, True]}, index=_index(5))
        gaps = detect_internal_gaps(valid)
        assert gaps == {"A": [(pd.Timestamp("2024-01-01 02:00", tz="UTC"), pd.Timestamp("2024-01-01 03:00", tz="UTC"), 2)]}

    def test_symbol_without_internal_gap_is_absent(self) -> None:
        # SCENARIO_MHS_GAP_TRIM_DIAGNOSIS_02: leading/trailing False, a fully
        # valid span, and an all-False symbol are edge padding / no-life cases,
        # never "no internal gaps" entries -- they must be absent entirely.
        valid = pd.DataFrame(
            {
                "leading": [False, True, True, True, False],
                "full": [True, True, True, True, True],
                "empty": [False, False, False, False, False],
            },
            index=_index(5),
        )
        gaps = detect_internal_gaps(valid)
        assert "leading" not in gaps
        assert "full" not in gaps
        assert "empty" not in gaps

    def test_multisymbol_reports_only_gapped_and_chronological(self) -> None:
        # SCENARIO_MHS_GAP_TRIM_DIAGNOSIS_03: only symbols with an internal gap
        # appear as keys; a symbol with two separate gaps returns both in
        # chronological order; multi-bar runs keep their full length.
        valid = pd.DataFrame(
            {
                "A": [True, False, True, False, True, True, True],
                "B": [True, True, True, True, True, True, True],
                "C": [True, False, False, True, False, False, True],
            },
            index=_index(7),
        )
        gaps = detect_internal_gaps(valid)
        assert set(gaps) == {"A", "C"}
        assert gaps["A"] == [
            (pd.Timestamp("2024-01-01 01:00", tz="UTC"), pd.Timestamp("2024-01-01 01:00", tz="UTC"), 1),
            (pd.Timestamp("2024-01-01 03:00", tz="UTC"), pd.Timestamp("2024-01-01 03:00", tz="UTC"), 1),
        ]
        assert gaps["C"] == [
            (pd.Timestamp("2024-01-01 01:00", tz="UTC"), pd.Timestamp("2024-01-01 02:00", tz="UTC"), 2),
            (pd.Timestamp("2024-01-01 04:00", tz="UTC"), pd.Timestamp("2024-01-01 05:00", tz="UTC"), 2),
        ]

    def test_readonly_and_boolean_dtype_contract(self) -> None:
        # SCENARIO_MHS_GAP_TRIM_DIAGNOSIS_04: the input is never mutated and
        # non-boolean-dtype input fails closed with ValueError.
        valid = pd.DataFrame({"A": [True, False, True, True]}, index=_index(4))
        before = valid.copy()
        detect_internal_gaps(valid)
        assert valid.equals(before)
        assert valid.index.equals(before.index)
        assert list(valid.columns) == list(before.columns)

        with pytest.raises(ValueError, match="boolean"):
            detect_internal_gaps(pd.DataFrame({"A": [1, 0, 1, 1]}, index=_index(4)))
        with pytest.raises(ValueError, match="boolean"):
            detect_internal_gaps(pd.DataFrame({"A": [1.0, 0.0, 1.0, 1.0]}, index=_index(4)))
        with pytest.raises(ValueError, match="boolean"):
            detect_internal_gaps(pd.Series([True, False, True], index=_index(3)))

    def test_min_useful_bars_does_not_filter(self) -> None:
        # Every internal gap is reported regardless of size: filtering and
        # severity judgment are a caller/CLI concern, never this function's.
        valid = pd.DataFrame({"A": [True, False, True]}, index=_index(3))
        assert detect_internal_gaps(valid, min_useful_bars=720) == {
            "A": [(pd.Timestamp("2024-01-01 01:00", tz="UTC"), pd.Timestamp("2024-01-01 01:00", tz="UTC"), 1)],
        }
