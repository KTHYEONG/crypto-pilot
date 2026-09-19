"""Invariant guards for prefix-invariant observations and causal history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.availability import (
    ObservationAsOf,
    ObservationAvailability,
    observed_history_mask,
    select_available_observations,
)


def _hourly(n: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def _proxy_availability(
    values: pd.DataFrame, lag_hours: int = 1
) -> ObservationAvailability:
    completed = values.index + pd.Timedelta(hours=lag_hours)
    completed_ns = np.asarray(completed.as_unit("ns").asi8, dtype="int64")
    frame = pd.DataFrame(
        np.tile(completed_ns[:, None], (1, len(values.columns))).astype("datetime64[ns]"),
        index=values.index,
        columns=list(values.columns),
    )
    return ObservationAvailability(
        event_time=values.index,
        completed_at=completed,
        available_at=frame,
        provenance="archive_completion_proxy",
    )


def test_select_available_observations_hides_unpublished_bar() -> None:
    """Unpublished bar: 00:00 bar completing at 01:00 is unavailable at 00:59."""
    events = _hourly(3)
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=events)
    availability = _proxy_availability(values)
    out = select_available_observations(
        values, availability, pd.DatetimeIndex([events[0] + pd.Timedelta(minutes=59)])
    )
    assert bool(out["A"].isna().all())


def test_select_available_observations_usable_at_publication_equality() -> None:
    """Publication equality: the bar available at 01:00 is usable for a decision."""
    events = _hourly(3)
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=events)
    availability = _proxy_availability(values)
    out = select_available_observations(
        values, availability, pd.DatetimeIndex([events[1]])
    )
    assert float(out["A"].iloc[0]) == 1.0


def test_select_available_observations_uses_prior_on_delayed_publication() -> None:
    """Delayed publication: at 01:03 the prior published observation is used."""
    events = _hourly(3)
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=events)
    completed = events + pd.Timedelta(hours=1)
    published = pd.DataFrame(
        {
            "A": [
                completed[0],
                completed[1] + pd.Timedelta(minutes=7),
                completed[2],
            ]
        },
        index=events,
    )
    availability = ObservationAvailability(
        event_time=events,
        completed_at=completed,
        available_at=published,
        provenance="recorded",
    )
    out = select_available_observations(
        values, availability, pd.DatetimeIndex([events[1] + pd.Timedelta(minutes=3)])
    )
    assert float(out["A"].iloc[0]) == 1.0


def test_select_available_observations_rejects_ambiguous_revision() -> None:
    """Ambiguous revision: conflicting same-event values fail certification."""
    events = _hourly(2)
    dup_index = pd.DatetimeIndex([events[0], events[0]])
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=dup_index)
    availability = _proxy_availability(
        pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([events[0]]))
    )
    with pytest.raises(DataIntegrityError, match="unique"):
        select_available_observations(values, availability, pd.DatetimeIndex([events[0]]))


def test_observed_history_mask_keeps_earlier_ineligibility() -> None:
    """History accumulation: earlier eligibility stays false when later bars arrive."""
    events = _hourly(6)
    values = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]}, index=events)
    before = observed_history_mask(values.iloc[:3], min_history_bars=3)
    assert not bool(before["A"].any())
    after = observed_history_mask(values, min_history_bars=3)
    pd.testing.assert_frame_equal(after.iloc[:3], before)
    assert bool(after["A"].iloc[-1])


def test_select_available_observations_ignores_future_only_symbol() -> None:
    """Future-only symbol: a later column leaves earlier available values unchanged."""
    events = _hourly(4)
    base = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=events)
    extended = base.copy()
    extended["B"] = [np.nan, np.nan, 30.0, 40.0]
    decisions = pd.DatetimeIndex([events[1], events[2]])
    solo = select_available_observations(base, _proxy_availability(base), decisions)
    joint = select_available_observations(
        extended, _proxy_availability(extended), decisions
    )
    pd.testing.assert_series_equal(joint["A"], solo["A"])
    assert bool(joint["B"].isna().iloc[0])
    solo_mask = observed_history_mask(solo[["A"]], min_history_bars=1)
    joint_mask = observed_history_mask(joint[["A"]], min_history_bars=1)
    pd.testing.assert_frame_equal(joint_mask, solo_mask)


def test_observation_as_of_separates_values_and_knowledge() -> None:
    """Absent data stays missing instead of becoming a normal price."""
    events = _hourly(2)
    values = pd.DataFrame({"A": [1.0, np.nan]}, index=events)
    known = pd.DataFrame(
        np.array([[True], [False]]), index=events, columns=["A"]
    )
    snapshot = ObservationAsOf(
        values=values, known=known, availability=_proxy_availability(values)
    )
    assert bool(snapshot.values["A"].isna().iloc[1])
    assert not bool(snapshot.known["A"].iloc[1])


def test_select_available_observations_rejects_non_datetime_index() -> None:
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.RangeIndex(2))
    availability = _proxy_availability(
        pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([_hourly(1)[0]]))
    )
    with pytest.raises(DataIntegrityError, match="DatetimeIndex"):
        select_available_observations(
            values, availability, pd.DatetimeIndex([_hourly(1)[0]])
        )


def test_select_available_observations_rejects_naive_labels() -> None:
    events = pd.date_range("2021-01-01", periods=2, freq="1h")
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=events)
    availability = _proxy_availability(
        pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([events[0]]))
    )
    with pytest.raises(DataIntegrityError, match="UTC"):
        select_available_observations(values, availability, _hourly(1)[:1])


def test_select_available_observations_rejects_duplicate_columns() -> None:
    events = _hourly(2)
    values = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=events, columns=["A", "A"])
    availability = _proxy_availability(
        pd.DataFrame({"A": [1.0, 2.0]}, index=events)
    )
    with pytest.raises(DataIntegrityError, match="unique"):
        select_available_observations(values, availability, _hourly(1)[:1])


def test_select_available_observations_rejects_non_string_columns() -> None:
    events = _hourly(2)
    values = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=events, columns=[1, 2])
    completed = events + pd.Timedelta(hours=1)
    frame = pd.DataFrame(
        np.tile(
            np.asarray(completed.as_unit("ns").asi8, dtype="int64")[:, None], (1, 2)
        ).astype("datetime64[ns]"),
        index=events,
        columns=[1, 2],
    )
    availability = ObservationAvailability(
        event_time=events,
        completed_at=completed,
        available_at=frame,
        provenance="archive_completion_proxy",
    )
    with pytest.raises(DataIntegrityError, match="strings"):
        select_available_observations(values, availability, _hourly(1)[:1])


def test_select_available_observations_rejects_unknown_provenance() -> None:
    events = _hourly(2)
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=events)
    availability = _proxy_availability(values)
    forged = ObservationAvailability(
        event_time=availability.event_time,
        completed_at=availability.completed_at,
        available_at=availability.available_at,
        provenance="measured",  # type: ignore[arg-type]
    )
    with pytest.raises(DataIntegrityError, match="provenance"):
        select_available_observations(values, forged, _hourly(1)[:1])


def test_select_available_observations_rejects_misaligned_event_labels() -> None:
    events = _hourly(2)
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=events)
    shifted = ObservationAvailability(
        event_time=_hourly(2, start="2021-01-02"),
        completed_at=events + pd.Timedelta(hours=1),
        available_at=_proxy_availability(values).available_at,
        provenance="archive_completion_proxy",
    )
    with pytest.raises(DataIntegrityError, match="event labels"):
        select_available_observations(values, shifted, _hourly(1)[:1])


def test_select_available_observations_rejects_misaligned_completion() -> None:
    events = _hourly(3)
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=events)
    availability = _proxy_availability(values)
    forged = ObservationAvailability(
        event_time=availability.event_time,
        completed_at=availability.completed_at[:2],
        available_at=availability.available_at,
        provenance="archive_completion_proxy",
    )
    with pytest.raises(DataIntegrityError, match="completion"):
        select_available_observations(values, forged, _hourly(1)[:1])


def test_select_available_observations_rejects_misaligned_publication_frame() -> None:
    events = _hourly(2)
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=events)
    availability = _proxy_availability(values)
    extra = availability.available_at.copy()
    extra["B"] = extra["A"]
    forged = ObservationAvailability(
        event_time=availability.event_time,
        completed_at=availability.completed_at,
        available_at=extra,
        provenance="archive_completion_proxy",
    )
    with pytest.raises(DataIntegrityError, match="publication"):
        select_available_observations(values, forged, _hourly(1)[:1])


def test_select_available_observations_rejects_publication_before_completion() -> None:
    events = _hourly(2)
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=events)
    completed = events + pd.Timedelta(hours=1)
    early = pd.DataFrame(
        {"A": [events[0] + pd.Timedelta(minutes=30), completed[1]]}, index=events
    )
    availability = ObservationAvailability(
        event_time=events,
        completed_at=completed,
        available_at=early,
        provenance="recorded",
    )
    with pytest.raises(DataIntegrityError, match="precede"):
        select_available_observations(values, availability, _hourly(1)[:1])


def test_observed_history_mask_rejects_non_positive_requirement() -> None:
    events = _hourly(2)
    values = pd.DataFrame({"A": [1.0, 2.0]}, index=events)
    for bad in (0, -3, True, "3"):
        with pytest.raises(ValueError, match="positive integer"):
            observed_history_mask(values, min_history_bars=bad)  # type: ignore[arg-type]


def test_observed_history_mask_rejects_ambiguous_labels() -> None:
    events = _hourly(2)
    values = pd.DataFrame(
        {"A": [1.0, 2.0]}, index=pd.DatetimeIndex([events[0], events[0]])
    )
    with pytest.raises(DataIntegrityError, match="unique"):
        observed_history_mask(values, min_history_bars=1)


def _write_lake_member(path, symbol: str, closes: np.ndarray, grid: pd.DatetimeIndex) -> None:
    ms = ((grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(
        dtype="int64"
    )
    frame = pd.DataFrame(
        {
            "timestamp": ms[: len(closes)],
            "close": closes,
            "open": closes,
            "high": closes + 0.3,
            "low": closes - 0.3,
            "quote_vol": np.full(len(closes), 1e6),
            "taker_buy_quote": np.full(len(closes), 5e5),
            "volume": np.full(len(closes), 10.0),
        }
    )
    frame.to_parquet(path / "1h" / f"{symbol}.parquet", index=False)


def test_load_base_panel_causal_history_retains_short_lived_instrument(tmp_path) -> None:
    """Short instrument life: early observations survive without whole-window bars."""
    from src.mhs.panel import load_base_panel

    grid = _hourly(500)
    lake = tmp_path / "ohlcv"
    (lake / "1h").mkdir(parents=True)
    rng = np.random.default_rng(7)
    _write_lake_member(lake, "LONGAUSDT", 100.0 + np.cumsum(rng.normal(0, 0.5, 500)), grid)
    _write_lake_member(
        lake, "SHORTAUSDT", 50.0 + np.cumsum(rng.normal(0, 0.5, 100)), grid[:100]
    )
    start, end = grid[0], grid[-1]
    legacy = load_base_panel(
        str(lake), "1h", ("close", "open"), start, end, partition="all", min_bars=200
    )
    assert list(legacy["close"].columns) == ["LONGAUSDT"]
    causal = load_base_panel(
        str(lake),
        "1h",
        ("close", "open"),
        start,
        end,
        partition="all",
        min_bars=200,
        selection_mode="causal_history",
    )
    assert list(causal["close"].columns) == ["LONGAUSDT", "SHORTAUSDT"]
    pd.testing.assert_series_equal(causal["close"]["LONGAUSDT"], legacy["close"]["LONGAUSDT"])
    assert bool(causal["close"]["SHORTAUSDT"].iloc[:100].notna().all())
    assert bool(causal["close"]["SHORTAUSDT"].iloc[100:].isna().all())
    mask = observed_history_mask(causal["close"], min_history_bars=200)
    assert bool(mask["LONGAUSDT"].iloc[-1])
    assert not bool(mask["SHORTAUSDT"].iloc[-1])


def test_load_base_panel_window_extension_preserves_prefix(tmp_path) -> None:
    """Window extension: old planes and eligibility prefixes are unchanged."""
    from src.mhs.panel import load_base_panel

    grid = _hourly(300)
    lake = tmp_path / "ohlcv"
    (lake / "1h").mkdir(parents=True)
    rng = np.random.default_rng(11)
    for sym in ("EXTAAUSDT", "EXTABUSDT"):
        _write_lake_member(lake, sym, 100.0 + np.cumsum(rng.normal(0, 0.5, 300)), grid)
    start = grid[0]
    short = load_base_panel(
        str(lake), "1h", ("close", "quote_vol"), start, grid[199],
        partition="all", min_bars=10, selection_mode="causal_history",
    )
    long = load_base_panel(
        str(lake), "1h", ("close", "quote_vol"), start, grid[299],
        partition="all", min_bars=10, selection_mode="causal_history",
    )
    for plane in ("close", "quote_vol"):
        pd.testing.assert_frame_equal(long[plane].loc[: grid[199]], short[plane])
    short_mask = observed_history_mask(short["close"], min_history_bars=10)
    long_mask = observed_history_mask(long["close"], min_history_bars=10)
    pd.testing.assert_frame_equal(long_mask.loc[: grid[199]], short_mask)


def test_load_base_panel_unreadable_required_source_fails_closed(tmp_path) -> None:
    """Unreadable required source: input integrity fails instead of recomputing."""
    import pyarrow as pa

    from src.mhs.panel import load_base_panel

    grid = _hourly(50)
    lake = tmp_path / "ohlcv"
    (lake / "1h").mkdir(parents=True)
    rng = np.random.default_rng(13)
    _write_lake_member(lake, "GOODAUSDT", 100.0 + np.cumsum(rng.normal(0, 0.5, 50)), grid)
    _write_lake_member(lake, "BADAUSDT", 100.0 + np.cumsum(rng.normal(0, 0.5, 50)), grid)
    with open(lake / "1h" / "BADAUSDT.parquet", "wb") as handle:
        handle.write(b"not a parquet file")
    with pytest.raises((OSError, pa.ArrowException)):
        load_base_panel(
            str(lake), "1h", ("close",), grid[0], grid[-1],
            partition="all", min_bars=10, selection_mode="causal_history",
        )


def test_load_base_panel_rejects_unknown_selection_mode(tmp_path) -> None:
    from src.mhs.panel import load_base_panel

    grid = _hourly(10)
    lake = tmp_path / "ohlcv"
    (lake / "1h").mkdir(parents=True)
    with pytest.raises(ValueError, match="selection_mode"):
        load_base_panel(
            str(lake), "1h", ("close",), grid[0], grid[-1],
            partition="all", selection_mode="nope",  # type: ignore[arg-type]
        )
