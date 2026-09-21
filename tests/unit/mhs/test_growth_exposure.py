"""Invariant scenarios for the log-growth exposure solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mhs.growth_exposure import (
    GapSample,
    LogGrowthExposureSolution,
    roster_gap_sample,
    solve_log_growth_exposure,
    structurally_excluded_symbols,
)
from src.mhs.params import (
    FROZEN_EXPOSURE_GAP_THRESHOLD,
    FROZEN_EXPOSURE_GRID,
    FROZEN_EXPOSURE_MEAN_HAIRCUT,
    FROZEN_EXPOSURE_PLATEAU_TOLERANCE,
    FROZEN_EXPOSURE_SEED,
)

_GRID: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)


def _returns(n: int = 800, start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    t = np.arange(n, dtype="float64")
    values = 0.0012 + 0.02 * np.sin(t / 9.0) + 0.008 * np.sin(t / 3.7 + 1.0)
    return pd.Series(values, index=idx, dtype="float64")


def _no_gaps() -> GapSample:
    return GapSample(magnitudes=np.array([], dtype="float64"), events_per_year=0.0)


def _solve(
    unit_returns: pd.Series,
    gaps: GapSample,
    mean_haircut: float = 0.0,
    grid: tuple[float, ...] = _GRID,
    seed: int = 7,
) -> LogGrowthExposureSolution:
    return solve_log_growth_exposure(
        unit_returns,
        max_name_weight=0.05,
        gaps=gaps,
        mean_haircut=mean_haircut,
        grid=grid,
        plateau_tolerance=0.05,
        n_paths=128,
        horizon_years=2.0,
        mean_block_days=20,
        seed=seed,
    )


def test_registered_exposure_constants() -> None:
    """The solver's only registered inputs hold their probed values."""
    assert FROZEN_EXPOSURE_MEAN_HAIRCUT == 0.25
    assert FROZEN_EXPOSURE_GAP_THRESHOLD == 0.50
    assert tuple(round(1.0 + 0.25 * i, 2) for i in range(61)) == FROZEN_EXPOSURE_GRID
    assert FROZEN_EXPOSURE_PLATEAU_TOLERANCE == 0.05
    assert FROZEN_EXPOSURE_SEED == 20260921


def test_solve_is_deterministic_under_seed() -> None:
    """Identical inputs give bit-identical solutions."""
    returns = _returns()
    assert _solve(returns, _no_gaps()) == _solve(returns, _no_gaps())


def test_solve_higher_haircut_never_raises_chosen() -> None:
    """Removing more selection optimism never raises the chosen rung."""
    returns = _returns()
    assert _solve(returns, _no_gaps(), mean_haircut=0.4).chosen <= _solve(returns, _no_gaps(), mean_haircut=0.0).chosen


def test_solve_gaps_lower_growth_curve() -> None:
    """Gap losses can only drag each rung's stressed growth down."""
    returns = _returns()
    gaps = GapSample(magnitudes=np.array([0.6, 0.8, 1.0]), events_per_year=3.0)
    with_gaps = _solve(returns, gaps, mean_haircut=0.25)
    without_gaps = _solve(returns, _no_gaps(), mean_haircut=0.25)
    assert all(g <= w for g, w in zip(with_gaps.growth, without_gaps.growth, strict=True))


def test_solve_ruin_makes_rung_inadmissible() -> None:
    """A rung that loses everything on a gap day carries -inf growth and is never chosen."""
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    t = np.arange(500, dtype="float64")
    returns = pd.Series(0.0015 + 0.004 * np.sin(t / 5.0), index=idx, dtype="float64")
    gaps = GapSample(magnitudes=np.array([1.0]), events_per_year=2.0)
    solution = solve_log_growth_exposure(
        returns,
        max_name_weight=1.0 / 8.0,
        gaps=gaps,
        mean_haircut=0.0,
        grid=(1.0, 2.0, 4.0, 8.0),
        plateau_tolerance=0.05,
        n_paths=64,
        horizon_years=2.0,
        mean_block_days=20,
        seed=3,
    )
    assert solution.growth[-1] == float("-inf")
    assert solution.ruin_probability[-1] > 0.0
    assert solution.chosen <= 4.0


def test_solve_chosen_is_low_end_of_plateau() -> None:
    """The chosen rung is the smallest rung within tolerance of the best growth."""
    solution = _solve(_returns(), _no_gaps())
    best = max(g for g, r in zip(solution.growth, solution.ruin_probability, strict=True) if r == 0.0)
    plateau = [
        rung
        for rung, g, r in zip(solution.grid, solution.growth, solution.ruin_probability, strict=True)
        if r == 0.0 and g >= best - 0.05 * abs(best)
    ]
    assert len(plateau) >= 2
    assert solution.chosen == min(plateau)
    assert solution.chosen <= solution.argmax


def test_solve_negative_mean_fails_closed() -> None:
    """No admissible rung with positive growth raises instead of returning a losing exposure."""
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    returns = pd.Series(-0.002 + 0.001 * np.sin(np.arange(500, dtype="float64")), index=idx, dtype="float64")
    with pytest.raises(ValueError, match="positive"):
        _solve(returns, _no_gaps())


def test_solve_short_history_rejected() -> None:
    """Fewer than 365 finite rows cannot evidence a yearly-compounding choice."""
    with pytest.raises(ValueError, match="365"):
        _solve(_returns(n=200), _no_gaps())


def test_solve_rejects_invalid_inputs() -> None:
    """Every malformed solver input fails closed with a field-specific reason."""
    returns = _returns()
    gaps = _no_gaps()

    def _bad(match: str, **overrides: object) -> None:
        params: dict[str, object] = {
            "unit_returns": returns,
            "max_name_weight": 0.05,
            "gaps": gaps,
            "mean_haircut": 0.0,
            "grid": _GRID,
            "plateau_tolerance": 0.05,
            "n_paths": 128,
            "horizon_years": 2.0,
            "mean_block_days": 20,
            "seed": 7,
        }
        params.update(overrides)
        with pytest.raises(ValueError, match=match):
            solve_log_growth_exposure(**params)  # type: ignore[arg-type]

    _bad("Series", unit_returns=[0.01] * 500)
    _bad("max_name_weight", max_name_weight=True)
    _bad("max_name_weight", max_name_weight=float("nan"))
    _bad("gaps", gaps=GapSample(magnitudes=np.array([0.5]), events_per_year=-1.0))
    _bad("gaps", gaps=GapSample(magnitudes=np.array([], dtype="float64"), events_per_year=2.0))
    _bad("mean_haircut", mean_haircut=1.0)
    _bad("grid", grid=())
    _bad("grid", grid=(2.0, 1.0))
    _bad("grid", grid="bad")
    _bad("plateau_tolerance", plateau_tolerance=1.0)
    _bad("n_paths", n_paths=0)
    _bad("horizon_years", horizon_years=0.0)
    _bad("mean_block_days", mean_block_days=0)
    _bad("seed", seed=1.5)
    _bad("at least one day", horizon_years=0.001)


def _gap_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    closes = pd.DataFrame(
        {
            "A": [100.0, 180.0, 180.0, 126.0, 126.0],
            "B": [100.0, 100.0, 40.0, 40.0, 40.0],
        },
        index=idx,
        dtype="float64",
    )
    roster = pd.DataFrame(False, index=idx, columns=["A", "B"], dtype=bool)
    roster.loc[idx[0], "A"] = True
    roster.loc[idx[1], "B"] = True
    roster.loc[idx[2], "A"] = True
    return closes, roster


def test_gap_sample_counts_both_directions() -> None:
    """Rostered +80% and -60% moves count; a -30% move stays below threshold."""
    closes, roster = _gap_frame()
    sample = roster_gap_sample(closes, roster, threshold=0.5)
    assert sorted(sample.magnitudes.tolist()) == pytest.approx([0.6, 0.8])
    assert sample.events_per_year == pytest.approx(2.0 / (5.0 / 365.0))


def test_gap_sample_ignores_unrostered_moves() -> None:
    """A -90% move on a non-roster day carries no gap frequency."""
    idx = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    closes = pd.DataFrame({"A": [100.0, 10.0, 10.0]}, index=idx, dtype="float64")
    roster = pd.DataFrame(False, index=idx, columns=["A"], dtype=bool)
    sample = roster_gap_sample(closes, roster, threshold=0.5)
    assert sample.magnitudes.size == 0
    assert sample.events_per_year == 0.0


def test_gap_sample_rejects_invalid_inputs() -> None:
    """Misaligned frames, non-positive closes, and out-of-range thresholds fail closed."""
    closes, roster = _gap_frame()
    with pytest.raises(ValueError, match="threshold"):
        roster_gap_sample(closes, roster, threshold=1.5)
    with pytest.raises(ValueError, match="aligned"):
        roster_gap_sample(closes, roster.iloc[:3], threshold=0.5)
    bad = closes.copy()
    bad.iloc[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        roster_gap_sample(bad, roster, threshold=0.5)


def test_gap_sample_skips_unobserved_census_cells() -> None:
    """Pre-listing and post-delisting NaN closes are normal census state, never a move or an error."""
    idx = pd.date_range("2021-01-01", periods=4, freq="D", tz="UTC")
    closes = pd.DataFrame(
        {"A": [np.nan, 100.0, 20.0, np.nan], "B": [100.0, 100.0, 100.0, 100.0]}, index=idx, dtype="float64",
    )
    roster = pd.DataFrame(True, index=idx, columns=["A", "B"], dtype=bool)
    sample = roster_gap_sample(closes, roster, threshold=0.5)
    assert sample.magnitudes.tolist() == pytest.approx([0.8])
    assert sample.events_per_year == pytest.approx(1.0 / (4.0 / 365.0))


def _gap_registry_row(
    symbol: str, *, plane: str = "ohlcv_3m", reason: str = "DELISTED", resolved_at: str | None = None,
) -> dict:
    return {
        "symbol": symbol, "plane": plane, "start": "2021-01-01T00:00:00Z", "end": None,
        "reason": reason, "evidence": "test fixture", "verified_at": "2026-09-21T00:00:00Z",
        "resolved_at": resolved_at,
    }


def _write_gap_registry(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "gaps.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_structurally_excluded_symbols_selects_delisted_only(tmp_path: Path) -> None:
    """DELISTED is a permanent trading exclusion; SOURCE_ABSENT is merely missing data."""
    path = _write_gap_registry(
        tmp_path,
        [_gap_registry_row("LUNAUSDT"), _gap_registry_row("PUMPUSDT", reason="SOURCE_ABSENT")],
    )
    assert structurally_excluded_symbols(path=path) == frozenset({"LUNAUSDT"})


def test_structurally_excluded_symbols_ignores_resolved_records(tmp_path: Path) -> None:
    """A resolved (no-longer-active) DELISTED record no longer excludes its symbol."""
    path = _write_gap_registry(tmp_path, [_gap_registry_row("LUNAUSDT", resolved_at="2026-09-22T00:00:00Z")])
    assert structurally_excluded_symbols(path=path) == frozenset()


def test_structurally_excluded_symbols_respects_plane(tmp_path: Path) -> None:
    """A DELISTED record on a different plane does not exclude the symbol from this plane's query."""
    path = _write_gap_registry(tmp_path, [_gap_registry_row("LUNAUSDT", plane="funding")])
    assert structurally_excluded_symbols(plane="ohlcv_3m", path=path) == frozenset()
    assert structurally_excluded_symbols(plane="funding", path=path) == frozenset({"LUNAUSDT"})
