"""Invariant scenarios for the single source-gap interval registry."""

from __future__ import annotations

import json
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs import data_policy as data_policy_mod
from src.mhs.data_policy import SOURCE_GAP_EXCLUDED_SYMBOLS, source_gap_excluded_symbols
from src.mhs.source_gaps import (
    active_intervals,
    blocked_mask,
    blocked_symbols_between,
    clear_source_gap_registry_cache,
    load_source_gap_registry,
)

_START = "2022-01-01T00:00:00Z"
_MID = "2022-01-03T00:00:00Z"
_END = "2022-01-05T00:00:00Z"
_VERIFIED = "2022-02-01T00:00:00Z"
UTC_PLUS_9 = timezone(timedelta(hours=9))


@pytest.fixture(autouse=True)
def _fresh_registry_cache() -> Any:
    clear_source_gap_registry_cache()
    yield
    clear_source_gap_registry_cache()


def _row(
    symbol: str = "AAAUSDT",
    plane: str = "ohlcv_3m",
    start: str | None = _START,
    end: str | None = _MID,
    reason: str = "SOURCE_ABSENT",
    evidence: str = "Vision monthly klines and REST re-query both empty",
    verified_at: str = _VERIFIED,
    resolved_at: str | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "plane": plane,
        "start": start,
        "end": end,
        "reason": reason,
        "evidence": evidence,
        "verified_at": verified_at,
        "resolved_at": resolved_at,
    }


def _write_registry(tmp_path: Path, rows: list[Any]) -> Path:
    path = tmp_path / "gaps.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _grid(start: str, end: str, freq: str = "h") -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, tz="UTC")


def test_load_returns_normalized_order_and_repeatable(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        [
            _row(symbol="BBBUSDT", start="2022-04-01T00:00:00Z", end="2022-04-03T00:00:00Z"),
            _row(symbol="AAAUSDT", start=_START, end=_MID),
            _row(symbol="AAAUSDT", start="2022-04-01T00:00:00Z", end="2022-04-03T00:00:00Z"),
        ],
    )
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    first = load_source_gap_registry(path)
    assert [(iv.symbol, iv.plane, iv.start.isoformat()) for iv in first] == [
        ("AAAUSDT", "ohlcv_3m", "2022-01-01T00:00:00+00:00"),
        ("AAAUSDT", "ohlcv_3m", "2022-04-01T00:00:00+00:00"),
        ("BBBUSDT", "ohlcv_3m", "2022-04-01T00:00:00+00:00"),
    ]
    assert load_source_gap_registry(path) == first


def test_load_rejects_naive_timestamp(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(start="2022-01-01T00:00:00")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_reversed_interval(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(start=_MID, end=_START)])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_blank_evidence(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(evidence="   ")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_overlapping_active_intervals(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        [
            _row(start=_START, end=_END),
            _row(start=_MID, end="2022-01-07T00:00:00Z"),
        ],
    )
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_resolved_record_excluded_from_overlap_and_active(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        [
            _row(start=_START, end=_END, resolved_at="2022-03-01T00:00:00Z"),
            _row(start=_MID, end="2022-01-07T00:00:00Z"),
        ],
    )
    assert len(load_source_gap_registry(path)) == 2
    assert [iv.symbol for iv in active_intervals(path=path)] == ["AAAUSDT"]


def test_blocked_mask_applies_half_open_interval(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(start=_START, end=_MID)])
    grid = _grid("2021-12-31T22:00:00Z", "2022-01-03T02:00:00Z")
    frame = blocked_mask(["AAAUSDT"], grid, plane="ohlcv_3m", path=path)
    assert str(frame.dtypes.iloc[0]) == "bool"
    assert bool(frame.loc[pd.Timestamp(_START), "AAAUSDT"])
    assert not bool(frame.loc[pd.Timestamp(_MID), "AAAUSDT"])
    assert not bool(frame.loc[pd.Timestamp("2021-12-31T22:00:00Z"), "AAAUSDT"])


def test_blocked_mask_propagates_open_interval(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(start=_START, end=None)])
    grid = _grid("2021-12-31T22:00:00Z", "2022-01-03T02:00:00Z")
    frame = blocked_mask(["AAAUSDT"], grid, plane="ohlcv_3m", path=path)
    assert bool(frame.loc[grid >= pd.Timestamp(_START)].all().iloc[0])
    assert not bool(frame.loc[pd.Timestamp("2021-12-31T22:00:00Z"), "AAAUSDT"])


def test_blocked_mask_passes_unknown_symbol(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    grid = _grid("2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z")
    frame = blocked_mask(["ZZZUSDT"], grid, plane="ohlcv_3m", path=path)
    assert not bool(frame.to_numpy().any())


def test_blocked_mask_isolates_plane(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(plane="funding")])
    grid = _grid("2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z")
    frame = blocked_mask(["AAAUSDT"], grid, plane="ohlcv_3m", path=path)
    assert not bool(frame.to_numpy().any())


def test_blocked_mask_rejects_naive_index(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    grid = pd.date_range("2022-01-01", "2022-01-02", freq="h")
    with pytest.raises(DataIntegrityError):
        blocked_mask(["AAAUSDT"], grid, plane="ohlcv_3m", path=path)


def test_blocked_symbols_between_excludes_touching_window(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(start=_START, end=_MID)])
    assert blocked_symbols_between(
        pd.Timestamp(_MID), pd.Timestamp(_END), plane="ohlcv_3m", path=path
    ) == frozenset()
    assert blocked_symbols_between(
        pd.Timestamp("2022-01-02T00:00:00Z"), pd.Timestamp("2022-01-04T00:00:00Z"),
        plane="ohlcv_3m", path=path,
    ) == frozenset({"AAAUSDT"})


def test_source_gap_excluded_symbols_matches_active_view(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = source_gap_excluded_symbols()
    assert set(SOURCE_GAP_EXCLUDED_SYMBOLS) == expected
    assert len(SOURCE_GAP_EXCLUDED_SYMBOLS) == len(expected)
    assert "LUNAUSDT" in SOURCE_GAP_EXCLUDED_SYMBOLS
    monkeypatch.setattr(data_policy_mod, "active_intervals", lambda **kwargs: ())
    assert source_gap_excluded_symbols() == frozenset()


def test_packaged_registry_loads_probe_confirmed_intervals() -> None:
    intervals = load_source_gap_registry()
    assert load_source_gap_registry() is intervals
    by_symbol = [iv.symbol for iv in active_intervals()]
    for symbol in ("LUNAUSDT", "MANAUSDT", "NEARUSDT"):
        assert symbol in by_symbol
    luna = [iv for iv in intervals if iv.symbol == "LUNAUSDT"]
    assert len(luna) == 1
    assert luna[0].end is None
    assert luna[0].reason == "DELISTED"
    mana = [
        (iv.start.isoformat(), iv.end.isoformat() if iv.end else None)
        for iv in intervals
        if iv.symbol == "MANAUSDT" and "Binance Vision" in iv.evidence
    ]
    assert mana == [
        ("2022-02-26T00:00:00+00:00", "2022-03-01T00:00:00+00:00"),
        ("2022-04-01T00:00:00+00:00", "2022-04-03T00:00:00+00:00"),
    ]
    near = [(iv.start.isoformat(), iv.end.isoformat() if iv.end else None) for iv in intervals if iv.symbol == "NEARUSDT"]
    assert near == mana


def test_load_rejects_unreadable_registry(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(tmp_path / "missing.jsonl")


def test_load_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "gaps.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_non_object_record(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [["AAAUSDT"]])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_missing_field(tmp_path: Path) -> None:
    row = _row()
    del row["evidence"]
    path = _write_registry(tmp_path, [row])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_unknown_plane(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(plane="ohlcv_1s")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_unknown_reason(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(reason="NO_DATA")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_lowercase_symbol(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(symbol="aaausdt")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_non_utc_offset(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(start="2022-01-01T09:00:00+09:00")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_malformed_timestamp(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(start="not-a-time")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_empty_timestamp(tmp_path: Path) -> None:
    row = _row()
    row["start"] = 123
    path = _write_registry(tmp_path, [row])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_non_string_end(tmp_path: Path) -> None:
    row = _row()
    row["end"] = 5
    path = _write_registry(tmp_path, [row])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_non_string_resolved_at(tmp_path: Path) -> None:
    row = _row()
    row["resolved_at"] = 5
    path = _write_registry(tmp_path, [row])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_resolved_before_verified(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row(resolved_at="2022-01-15T00:00:00Z")])
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_load_rejects_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "gaps.jsonl"
    path.write_bytes(b"\xff\xfe invalid \x00 bytes\n")
    with pytest.raises(DataIntegrityError):
        load_source_gap_registry(path)


def test_blocked_mask_rejects_non_datetime_index(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    with pytest.raises(DataIntegrityError):
        blocked_mask(["AAAUSDT"], pd.RangeIndex(0, 3), plane="ohlcv_3m", path=path)  # type: ignore[arg-type]


def test_blocked_mask_rejects_non_utc_index(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    grid = pd.date_range("2022-01-01", "2022-01-02", freq="h", tz=UTC_PLUS_9)
    with pytest.raises(DataIntegrityError):
        blocked_mask(["AAAUSDT"], grid, plane="ohlcv_3m", path=path)


def test_blocked_mask_rejects_duplicate_symbols(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    grid = _grid("2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z")
    with pytest.raises(DataIntegrityError):
        blocked_mask(["AAAUSDT", "AAAUSDT"], grid, plane="ohlcv_3m", path=path)


def test_blocked_symbols_between_rejects_naive_window(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    with pytest.raises(DataIntegrityError):
        blocked_symbols_between(
            pd.Timestamp("2022-01-01"), pd.Timestamp(_END, tz="UTC"), plane="ohlcv_3m", path=path
        )


def test_blocked_symbols_between_rejects_non_utc_window(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    with pytest.raises(DataIntegrityError):
        blocked_symbols_between(
            pd.Timestamp("2022-01-01T09:00:00+09:00"), pd.Timestamp(_END, tz="UTC"),
            plane="ohlcv_3m", path=path,
        )


def test_blocked_symbols_between_rejects_empty_window(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    with pytest.raises(DataIntegrityError):
        blocked_symbols_between(
            pd.Timestamp(_MID), pd.Timestamp(_MID), plane="ohlcv_3m", path=path
        )


def test_blocked_symbols_between_rejects_non_timestamp_window(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    with pytest.raises(DataIntegrityError):
        blocked_symbols_between("2022-01-01", pd.Timestamp(_END, tz="UTC"), plane="ohlcv_3m", path=path)  # type: ignore[arg-type]


def test_loader_defaults_missing_extent_to_unscoped(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [_row()])
    (iv,) = load_source_gap_registry(path)
    assert iv.extent == "UNSCOPED"


def test_loader_rejects_unknown_extent(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [{**_row(), "extent": "EDGE"}])
    with pytest.raises(DataIntegrityError, match="unknown extent"):
        load_source_gap_registry(path)


@pytest.mark.parametrize(
    ("reason", "extent", "excluded"),
    [
        ("SOURCE_ABSENT", "LISTING_EDGE", False),
        ("SOURCE_ABSENT", "OPEN_EDGE", False),
        ("SOURCE_ABSENT", "INTERIOR", False),
        ("SOURCE_ABSENT", "UNSCOPED", True),
        ("DELISTED", "OPEN_EDGE", True),
    ],
)
def test_excluded_symbols_follow_reason_and_extent_not_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str, extent: str, excluded: bool
) -> None:
    """Evidence text that mentions an edge must not change the rule; only reason/extent fields do."""
    path = _write_registry(
        tmp_path, [{**_row(reason=reason, evidence="listing edge interior gap open-ended edge"), "extent": extent}]
    )
    intervals = load_source_gap_registry(path)
    monkeypatch.setattr(data_policy_mod, "active_intervals", lambda **kwargs: intervals)
    assert (source_gap_excluded_symbols() == frozenset({"AAAUSDT"})) is excluded
