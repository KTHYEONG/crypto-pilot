from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.frozen_research_universe import build_frozen_pit_roster

_SYMBOLS = ("AAA", "BBB", "CCC")


def _daily(
    symbols: tuple[str, ...] = _SYMBOLS,
    n: int = 100,
    start: str = "2021-01-01",
    price: float = 100.0,
    turnover: float = 5_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = pd.DataFrame(price, index=idx, columns=list(symbols), dtype="float64")
    qv = pd.DataFrame(turnover, index=idx, columns=list(symbols), dtype="float64")
    return close, qv


def test_roster_ignores_same_day_spike_until_next_day() -> None:
    """A turnover spike on day D cannot change roster D, only D+1."""
    close, qv = _daily()
    spike_day = qv.index[90]
    qv_spiked = qv.copy()
    qv_spiked.loc[spike_day, "CCC"] = 500_000_000.0
    base = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    spiked = build_frozen_pit_roster(close, qv_spiked, _SYMBOLS, breadth=20)
    pd.testing.assert_frame_equal(spiked.loc[[spike_day]], base.loc[[spike_day]])
    assert spiked.dtypes.eq(bool).all()
    assert not bool(spiked.isna().any().any())
    assert not bool(base.iloc[0].any())


def test_roster_requires_85_traded_days_in_trailing_90() -> None:
    """84 traded observations ineligible; 85 eligible at equal liquidity."""
    close, qv = _daily(n=92)
    qv.loc[qv.index[:6], "BBB"] = 0.0
    roster = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    day = qv.index[90]
    assert bool(roster.loc[day, "AAA"])
    assert not bool(roster.loc[day, "BBB"])


def test_roster_enforces_median_floor_at_one_million() -> None:
    """Median just below the floor excluded; exactly at floor admitted."""
    close, qv = _daily(n=95)
    qv["AAA"] = 999_999.0
    qv["BBB"] = 1_000_000.0
    roster = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    day = qv.index[94]
    assert not bool(roster.loc[day, "AAA"])
    assert bool(roster.loc[day, "BBB"])


def test_roster_requires_all_30_turnover_observations_for_liquidity_median() -> None:
    """A seasoned symbol with one missing recent turnover bar remains ineligible."""
    close, qv = _daily(n=100)
    qv.loc[qv.index[70], "AAA"] = float("nan")
    roster = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    decision_day = qv.index[91]
    assert not bool(roster.loc[decision_day, "AAA"])
    assert bool(roster.loc[decision_day, "BBB"])


def test_roster_treats_zero_volume_day_as_not_traded() -> None:
    """Positive close with zero turnover is not traded and not selectable."""
    close, qv = _daily(n=95)
    zombie = qv.index[93]
    qv.loc[zombie, "AAA"] = 0.0
    roster = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    assert not bool(roster.loc[zombie + pd.Timedelta(days=1), "AAA"])
    qv_all_zero = qv.copy()
    qv_all_zero.loc[:, "BBB"] = 0.0
    roster2 = build_frozen_pit_roster(close, qv_all_zero, _SYMBOLS, breadth=20)
    assert not bool(roster2.any(axis=0)["BBB"])


def test_roster_keeps_earlier_rows_after_later_delisting() -> None:
    """Appending a delisted tail leaves every earlier roster row unchanged."""
    close, qv = _daily(n=100)
    roster_full_tail = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    close_short, qv_short = _daily(n=95)
    roster_short = build_frozen_pit_roster(close_short, qv_short, _SYMBOLS, breadth=20)
    pd.testing.assert_frame_equal(roster_full_tail.loc[close_short.index], roster_short)


def test_roster_breaks_ties_by_column_order_and_caps_width() -> None:
    """Equal medians admit exactly the first 20 columns in stable order."""
    symbols = tuple(f"S{i:02d}" for i in range(25))
    close, qv = _daily(symbols=symbols, n=100)
    roster = build_frozen_pit_roster(close, qv, symbols, breadth=20)
    row = roster.iloc[99]
    assert int(row.sum()) == 20
    assert list(row[row].index) == list(symbols[:20])
    roster40 = build_frozen_pit_roster(close, qv, symbols, breadth=40)
    assert int(roster40.iloc[99].sum()) == 25


def test_roster_arbitrary_breadth_preserves_pit_lag() -> None:
    close, qv = _daily()
    spike_day = qv.index[90]
    qv_spiked = qv.copy()
    qv_spiked.loc[spike_day, "CCC"] = 500_000_000.0
    base = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=12)
    spiked = build_frozen_pit_roster(close, qv_spiked, _SYMBOLS, breadth=12)
    pd.testing.assert_frame_equal(spiked.loc[[spike_day]], base.loc[[spike_day]])
    assert bool((spiked.sum(axis=1) <= 12).all())


def test_roster_oversized_breadth_does_not_invent_members() -> None:
    close, qv = _daily()
    roster = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=12)
    day = qv.index[99]
    eligible = {"AAA", "BBB", "CCC"}
    assert set(roster.loc[day][roster.loc[day]].index).issubset(eligible)
    assert int(roster.loc[day].sum()) <= 3


def test_roster_rejects_invalid_breadth() -> None:
    close, qv = _daily()
    for bad in (0, -1, -20, True, False):
        with pytest.raises(ValueError, match="breadth"):
            build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=bad)  # type: ignore[arg-type]


def test_roster_rejects_invalid_census_alignment_and_values() -> None:
    close, qv = _daily()
    with pytest.raises(DataIntegrityError, match="census_symbols"):
        build_frozen_pit_roster(close, qv, (), breadth=20)
    with pytest.raises(DataIntegrityError, match="census_symbols"):
        build_frozen_pit_roster(close, qv, ("AAA", "AAA", "CCC"), breadth=20)
    with pytest.raises(DataIntegrityError, match="census_symbols"):
        build_frozen_pit_roster(close, qv, ("AAA", "", "CCC"), breadth=20)
    with pytest.raises(DataIntegrityError, match="identical index"):
        build_frozen_pit_roster(close, qv.rename(columns={"AAA": "ZZZ"}), _SYMBOLS, breadth=20)
    with pytest.raises(DataIntegrityError, match="identical index"):
        build_frozen_pit_roster(close.iloc[1:], qv, _SYMBOLS, breadth=20)
    bad_close = close.copy()
    bad_close.iloc[50, 0] = -3.0
    with pytest.raises(DataIntegrityError, match="positive finite"):
        build_frozen_pit_roster(bad_close, qv, _SYMBOLS, breadth=20)
    bad_qv = qv.copy()
    bad_qv.iloc[50, 1] = -1.0
    with pytest.raises(DataIntegrityError, match="positive finite"):
        build_frozen_pit_roster(close, bad_qv, _SYMBOLS, breadth=20)
    bad_inf = qv.copy()
    bad_inf.iloc[50, 1] = float("inf")
    with pytest.raises(DataIntegrityError, match="positive finite"):
        build_frozen_pit_roster(close, bad_inf, _SYMBOLS, breadth=20)


def test_roster_rejects_non_utc_midnight_or_gapped_index() -> None:
    close, qv = _daily()
    naive_close = close.copy()
    naive_close.index = naive_close.index.tz_localize(None)
    naive_qv = qv.copy()
    naive_qv.index = naive_qv.index.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="UTC-midnight"):
        build_frozen_pit_roster(naive_close, naive_qv, _SYMBOLS, breadth=20)
    eastern = close.copy()
    from datetime import timedelta, timezone

    eastern.index = eastern.index.tz_convert(timezone(timedelta(hours=-5)))
    eastern_qv = qv.copy()
    eastern_qv.index = eastern_qv.index.tz_convert(timezone(timedelta(hours=-5)))
    with pytest.raises(DataIntegrityError, match="UTC-midnight"):
        build_frozen_pit_roster(eastern, eastern_qv, _SYMBOLS, breadth=20)
    noon = close.copy()
    noon.index = noon.index + pd.Timedelta(hours=12)
    noon_qv = qv.copy()
    noon_qv.index = noon_qv.index + pd.Timedelta(hours=12)
    with pytest.raises(DataIntegrityError, match="UTC-midnight"):
        build_frozen_pit_roster(noon, noon_qv, _SYMBOLS, breadth=20)
    dup = pd.concat([close.iloc[[0]], close])
    dup_qv = pd.concat([qv.iloc[[0]], qv])
    with pytest.raises(DataIntegrityError, match="UTC-midnight"):
        build_frozen_pit_roster(dup, dup_qv, _SYMBOLS, breadth=20)
    gapped = close.drop(close.index[50])
    gapped_qv = qv.drop(qv.index[50])
    with pytest.raises(DataIntegrityError, match="UTC-midnight"):
        build_frozen_pit_roster(gapped, gapped_qv, _SYMBOLS, breadth=20)
    assert np.issubdtype(close.to_numpy().dtype, np.floating)


def _blocked(
    idx: pd.DatetimeIndex, symbols: tuple[str, ...], day: pd.Timestamp | None = None, sym: str | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame(False, index=idx, columns=list(symbols), dtype=bool)
    if day is not None and sym is not None:
        frame.loc[day, sym] = True
    return frame


def test_blocked_day_returns_seat_to_runner_up() -> None:
    close, qv = _daily()
    day = qv.index[95]
    blocked = _blocked(qv.index, _SYMBOLS, day, "AAA")
    roster = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20, blocked_decisions=blocked)
    assert not bool(roster.loc[day, "AAA"])
    assert bool(roster.loc[day, "BBB"])
    assert bool(roster.loc[day, "CCC"])
    plain = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    assert int(roster.loc[day].sum()) == int(plain.loc[day].sum()) - 1


def test_unblocked_day_keeps_normal_seat() -> None:
    close, qv = _daily()
    day = qv.index[95]
    other = qv.index[94]
    blocked = _blocked(qv.index, _SYMBOLS, day, "AAA")
    roster = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20, blocked_decisions=blocked)
    assert bool(roster.loc[other, "AAA"])


def test_blocked_decisions_rejects_misaligned_frame() -> None:
    close, qv = _daily()
    with pytest.raises(DataIntegrityError, match="blocked_decisions"):
        build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20, blocked_decisions="nope")  # type: ignore[arg-type]
    shifted = _blocked(qv.index[1:], _SYMBOLS)
    with pytest.raises(DataIntegrityError, match="blocked_decisions"):
        build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20, blocked_decisions=shifted)
    renamed = _blocked(qv.index, _SYMBOLS).rename(columns={"AAA": "ZZZ"})
    with pytest.raises(DataIntegrityError, match="blocked_decisions"):
        build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20, blocked_decisions=renamed)
    non_bool = _blocked(qv.index, _SYMBOLS).astype(object)
    with pytest.raises(DataIntegrityError, match="blocked_decisions"):
        build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20, blocked_decisions=non_bool)


def test_no_block_matches_absent_block_frame() -> None:
    close, qv = _daily()
    plain = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20)
    empty = build_frozen_pit_roster(close, qv, _SYMBOLS, breadth=20, blocked_decisions=_blocked(qv.index, _SYMBOLS))
    pd.testing.assert_frame_equal(empty, plain)
