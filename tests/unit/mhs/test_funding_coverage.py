"""Invariant guards for funding coverage compression."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution.contracts import FundingAlignment, align_funding_with_knowledge, funding_coverage_gaps


def _grid(periods: int = 8) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=periods, freq="3min", tz="UTC")


def _alignment(
    grid: pd.DatetimeIndex,
    known_map: dict[str, list[bool]],
    failures: dict[str, str] | None = None,
) -> FundingAlignment:
    rates = pd.DataFrame({s: [0.0] * len(grid) for s in known_map}, index=grid, dtype="float64")
    known = pd.DataFrame(
        {s: list(v) for s, v in known_map.items()}, index=grid,
    ).astype(bool)
    return FundingAlignment(rates=rates, known=known, source_failures=dict(failures or {}))


def test_known_zero_rate_emits_no_gap() -> None:
    grid = _grid()
    alignment = _alignment(grid, {"AUSDT": [True] * len(grid)})
    assert funding_coverage_gaps(alignment, grid) == ()


def test_interior_hole_compresses_to_single_maximal_interval() -> None:
    grid = _grid(8)
    alignment = _alignment(grid, {"AUSDT": [True, False, False, False, False, True, True, True]})
    gaps = funding_coverage_gaps(alignment, grid)
    assert len(gaps) == 1
    assert gaps[0].symbol == "AUSDT"
    assert gaps[0].start == grid[1]
    assert gaps[0].end == grid[4]
    assert gaps[0].reason == "OBSERVATION_GAP"


def test_leading_and_trailing_holes_stay_distinct() -> None:
    grid = _grid(8)
    alignment = _alignment(grid, {"AUSDT": [False, False, True, True, True, True, False, False]})
    gaps = funding_coverage_gaps(alignment, grid)
    assert len(gaps) == 2
    assert gaps[0].start == grid[0]
    assert gaps[0].end == grid[1]
    assert gaps[1].start == grid[6]
    assert gaps[1].end == grid[7]
    assert gaps[0].start < gaps[1].start


def test_source_failure_reason_is_explicit() -> None:
    grid = _grid(6)
    alignment = align_funding_with_knowledge(
        {}, grid, symbols=["AUSDT"], source_failures={"AUSDT": "LOAD_FAILED"},
    )
    gaps = funding_coverage_gaps(alignment, grid)
    assert len(gaps) == 1
    assert gaps[0].reason == "SOURCE_UNAVAILABLE"


def test_long_inter_observation_gap_is_explicit() -> None:
    grid = pd.date_range("2024-01-01", periods=200, freq="3min", tz="UTC")
    series = pd.Series(
        [0.0001, 0.0002],
        index=pd.DatetimeIndex([grid[0], grid[0] + pd.Timedelta(hours=9)]),
        dtype="float64",
    )
    alignment = align_funding_with_knowledge({"AUSDT": series}, grid, symbols=["AUSDT"])
    gaps = funding_coverage_gaps(alignment, grid)
    assert len(gaps) >= 1
    assert all(g.reason == "OBSERVATION_GAP" for g in gaps)
    assert any(g.start > grid[0] and g.end < grid[-1] for g in gaps)


def test_boundary_gap_at_maximum_is_not_fabricated() -> None:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = start + pd.Timedelta(hours=8, minutes=5)
    grid = pd.date_range(start, end, freq="5min", tz="UTC")
    series = pd.Series([0.0001, 0.0002], index=pd.DatetimeIndex([start, end]), dtype="float64")
    alignment = align_funding_with_knowledge({"AUSDT": series}, grid, symbols=["AUSDT"])
    assert funding_coverage_gaps(alignment, grid) == ()


def test_grid_mismatch_fails_closed() -> None:
    grid = _grid()
    alignment = _alignment(grid, {"AUSDT": [True] * len(grid)})
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, grid[1:])


def test_timezone_mismatch_fails_closed() -> None:
    from datetime import timedelta, timezone

    grid = _grid()
    alignment = _alignment(grid, {"AUSDT": [True] * len(grid)})
    naive = pd.DatetimeIndex(grid.tz_localize(None))
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, naive)
    eastern = grid.tz_convert(timezone(timedelta(hours=-5)))
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, eastern)


def test_column_mismatch_fails_closed() -> None:
    grid = _grid()
    rates = pd.DataFrame({"AUSDT": [0.0] * len(grid)}, index=grid, dtype="float64")
    known = pd.DataFrame(
        {"AUSDT": [True] * len(grid), "BUSDT": [True] * len(grid)}, index=grid,
    ).astype(bool)
    alignment = FundingAlignment(rates=rates, known=known, source_failures={})
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, grid)


def test_repeated_calls_are_deterministic() -> None:
    grid = _grid(10)
    alignment = _alignment(
        grid,
        {
            "BUSDT": [False, True, True, False, False, True, True, True, False, False],
            "AUSDT": [True, False, False, True, True, True, False, True, True, True],
        },
    )
    first = funding_coverage_gaps(alignment, grid)
    second = funding_coverage_gaps(alignment, grid)
    assert first == second
    assert [(g.start, g.end, g.symbol) for g in first] == sorted(
        [(g.start, g.end, g.symbol) for g in first]
    )


def test_malformed_grid_fails_closed() -> None:
    from typing import cast

    grid = _grid()
    alignment = _alignment(grid, {"AUSDT": [True] * len(grid)})
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, cast(pd.DatetimeIndex, ["not-an-index"]))  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, grid[:0])
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, pd.DatetimeIndex([grid[0], pd.NaT]))
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, pd.DatetimeIndex([grid[0], grid[0]]))
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, grid[::-1])


def test_known_index_mismatch_fails_closed() -> None:
    grid = _grid()
    rates = pd.DataFrame({"AUSDT": [0.0] * len(grid)}, index=grid, dtype="float64")
    shifted = grid + pd.Timedelta(minutes=3)
    known = pd.DataFrame({"AUSDT": [True] * len(grid)}, index=shifted).astype(bool)
    alignment = FundingAlignment(rates=rates, known=known, source_failures={})
    with pytest.raises(DataIntegrityError):
        funding_coverage_gaps(alignment, grid)
