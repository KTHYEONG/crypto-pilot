"""FundingAttribution daily frame keeps UTC index, canonical order, and float64."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.execution.funding_attribution import FundingAttribution


def _attribution() -> tuple[FundingAttribution, pd.DatetimeIndex]:
    attribution = FundingAttribution(["AAA", "BBB", "CCC"])
    grid = pd.date_range("2021-06-01", periods=48, freq="h", tz="UTC")
    charged = np.zeros((48, 3), dtype="float64")
    charged[2:5, 0] = -0.001 * 10.0 * 100.0
    charged[2:5, 1] = 0.0005 * 10.0 * 100.0
    charged[26:30, 2] = -0.002 * 5.0 * 100.0
    attribution.add_chunk(grid, ["AAA", "BBB", "CCC"], charged, 0)
    return attribution, grid


def test_daily_frame_shape_and_totals() -> None:
    """Two days and three symbols stay UTC-indexed, ordered, and reconciled."""
    attribution, _ = _attribution()
    frame = attribution.daily_frame(["AAA", "BBB", "CCC"])
    assert list(frame.columns) == ["AAA", "BBB", "CCC"]
    assert str(frame.index.tz) == "UTC"
    assert len(frame) == 2
    assert all(dtype == np.dtype("float64") for dtype in frame.dtypes)
    totals = attribution.totals()
    assert set(totals) == {"AAA", "BBB", "CCC"}
    for sym in totals:
        assert totals[sym] == frame[sym].sum()
    assert totals["AAA"] < 0.0
    assert totals["CCC"] < 0.0


def test_chunks_accumulate_and_reindex_missing() -> None:
    """Later chunks add to totals; absent symbols reindex to zero."""
    attribution, grid = _attribution()
    extra = np.zeros((24, 2), dtype="float64")
    extra[:, 0] = -1.0
    before = dict(attribution.totals())
    attribution.add_chunk(grid[:24], ["AAA", "CCC"], extra, 0)
    after = attribution.totals()
    assert after["AAA"] == before["AAA"] + float(extra[:, 0].sum())
    assert after["CCC"] == before["CCC"] + float(extra[:, 1].sum())
    assert after["BBB"] == before["BBB"]
    frame = attribution.daily_frame(["CCC", "AAA"])
    assert list(frame.columns) == ["CCC", "AAA"]
    widened = attribution.daily_frame(["AAA", "BBB", "CCC", "DDD"])
    assert float(widened["DDD"].sum()) == 0.0


def test_streamed_columns_exclude_overlap_and_preserve_daily_charges() -> None:
    """Overlapping windows charge only kept bars, including a changing symbol roster."""
    columns = ["AAA", "BBB", "CCC"]
    attribution = FundingAttribution(columns)
    grid = pd.date_range("2021-06-01 23:00", periods=4, freq="h", tz="UTC")
    first = attribution.begin_chunk(grid[:2], ["BBB", "AAA"], 0)
    assert first.charges.shape == (2, 2)
    first.add_column(0, np.array([2.0, 3.0]))
    first.add_column(1, np.array([-1.0, -4.0]))
    attribution.commit_chunk(first)
    second = attribution.begin_chunk(grid, ["CCC", "AAA"], 2)
    assert second.charges.shape == (1, 2)
    second.add_column(0, np.array([999.0, 999.0, -5.0, 1.0]))
    second.add_column(1, np.array([999.0, 999.0, 2.0, -1.0]))
    attribution.commit_chunk(second)
    expected = pd.DataFrame(
        [[-1.0, 2.0, 0.0], [-3.0, 3.0, -4.0]],
        index=pd.date_range("2021-06-01", periods=2, tz="UTC"), columns=columns,
    )
    pd.testing.assert_frame_equal(attribution.daily_frame(columns), expected, check_freq=False)
    assert attribution.totals() == {"AAA": -4.0, "BBB": 5.0, "CCC": -4.0}
