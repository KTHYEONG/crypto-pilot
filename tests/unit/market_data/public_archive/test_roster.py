from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.roster import (
    LiquidRoster,
    ever_liquid_symbols,
    load_roster,
    merge_roster,
    save_roster,
)


def _frame(data: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(next(iter(data.values()))), freq="D", tz="UTC")
    return pd.DataFrame(data, index=index)


def test_ranking_uses_only_prior_days() -> None:
    frame = _frame({"A": [10.0, 10.0, 10.0, 1000.0], "B": [20.0, 20.0, 20.0, 20.0]})
    assert ever_liquid_symbols(frame, top_n=1, lookback_days=2) == ("B",)


def test_union_over_time() -> None:
    frame = _frame({
        "A": [100.0, 100.0, 1.0, 1.0, 1.0, 1.0],
        "B": [1.0, 1.0, 1.0, 1.0, 100.0, 100.0],
    })
    assert ever_liquid_symbols(frame, top_n=1, lookback_days=2) == ("A", "B")


def test_ties_broken_by_symbol() -> None:
    frame = _frame({"B": [5.0, 5.0, 5.0], "A": [5.0, 5.0, 5.0]})
    assert ever_liquid_symbols(frame, top_n=1, lookback_days=2) == ("A",)


def test_roster_is_grow_only() -> None:
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    existing = LiquidRoster(version=3, as_of=as_of, top_n=1, lookback_days=2, symbols=("A", "B"))
    merged = merge_roster(existing, ("B", "C"), as_of=as_of, top_n=1, lookback_days=2)
    assert merged.symbols == ("A", "B", "C")
    assert merged.version == 4
    again = merge_roster(merged, ("A",), as_of=as_of, top_n=1, lookback_days=2)
    assert again.symbols == ("A", "B", "C")
    assert again.version == 4


def test_malformed_roster_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "roster.json"
    assert load_roster(missing) is None
    missing.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_roster(missing)


def test_roster_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "_roster" / "aggtrades_liquid.json"
    roster = LiquidRoster(
        version=1,
        as_of=pd.Timestamp("2024-02-01 12:00", tz="UTC"),
        top_n=60,
        lookback_days=30,
        symbols=("A", "B"),
    )
    save_roster(path, roster)
    assert load_roster(path) == roster


def test_ever_liquid_guards() -> None:
    frame = _frame({"A": [1.0, 2.0, 3.0]})
    assert ever_liquid_symbols(frame, top_n=0, lookback_days=2) == ()
    assert ever_liquid_symbols(frame, top_n=1, lookback_days=0) == ()
    assert ever_liquid_symbols(pd.DataFrame(), top_n=1, lookback_days=2) == ()
    assert ever_liquid_symbols(frame, top_n=1, lookback_days=5) == ()


def test_load_roster_rejects_bad_payloads(tmp_path: Path) -> None:
    path = tmp_path / "roster.json"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_roster(path)
    path.write_text('{"version": 1}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_roster(path)
    path.write_text(
        '{"version": 1, "as_of": "2024-01-01", "top_n": 1, "lookback_days": 1, "symbols": ["B", "A"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError):
        load_roster(path)
    path.write_text(
        '{"version": "1", "as_of": "2024-01-01", "top_n": 1, "lookback_days": 1, "symbols": ["A"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError):
        load_roster(path)
    path.write_text(
        '{"version": 1, "as_of": "NaT", "top_n": 1, "lookback_days": 1, "symbols": ["A"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataIntegrityError):
        load_roster(path)


def test_merge_roster_fresh() -> None:
    merged = merge_roster(
        None, ("B", "A"),
        as_of=pd.Timestamp("2024-02-01", tz="UTC"), top_n=2, lookback_days=2,
    )
    assert merged.version == 1
    assert merged.symbols == ("A", "B")


def test_missing_volume_never_enters_roster() -> None:
    frame = _frame({"A": [1.0, 2.0, 3.0], "MISSING": [float("nan")] * 3})
    assert ever_liquid_symbols(frame, top_n=60, lookback_days=2) == ("A",)


def test_nan_allowed_in_complete_temporal_window() -> None:
    frame = _frame({"A": [10.0, float("nan"), 0.0], "B": [1.0, 1.0, 0.0]})
    assert ever_liquid_symbols(frame, top_n=1, lookback_days=2) == ("A",)


def test_invalid_utf8_roster_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "roster.json"
    path.write_bytes(b"\xff")
    with pytest.raises(DataIntegrityError):
        load_roster(path)


def test_lookback_is_calendar_days() -> None:
    frame = _frame({"A": [1.0, 2.0, 3.0]})
    frame.index = pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-20"], utc=True)
    assert ever_liquid_symbols(frame, top_n=1, lookback_days=2) == ()
