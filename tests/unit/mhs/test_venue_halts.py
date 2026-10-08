"""Part 3 venue-halt registry invariants: validation, committed content, root binding."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.venue_halts import (
    EMPTY_VENUE_HALT_REGISTRY,
    VenueHaltInterval,
    assemble_venue_halt_registry,
    default_venue_halt_registry_path,
    load_venue_halt_registry,
    parse_venue_halt_registry,
    venue_halt_registry_for_root,
    venue_halt_registry_jsonl,
)

T0 = pd.Timestamp("2022-05-01 22:27", tz="UTC")


def _halt(
    start: pd.Timestamp = T0,
    end: pd.Timestamp | None = None,
    present: int = 12,
    zero: int = 12,
) -> VenueHaltInterval:
    return VenueHaltInterval(
        halt_id=start.strftime("%Y-%m-%dT%H:%MZ"),
        start=start,
        end=end if end is not None else start + pd.Timedelta(minutes=9),
        present_symbols=present,
        zero_symbols=zero,
        evidence="test halt",
        verified_at=pd.Timestamp("2026-07-01T00:00:00Z"),
    )


def _raw(**overrides) -> bytes:
    record = {
        "kind": "venue_halt",
        "halt_id": "2022-05-01T22:27Z",
        "start": "2022-05-01T22:27:00Z",
        "end": "2022-05-01T22:36:00Z",
        "present_symbols": 12,
        "zero_symbols": 12,
        "evidence": "test halt",
        "verified_at": "2026-07-01T00:00:00Z",
    }
    record.update(overrides)
    return (json.dumps(record) + "\n").encode()


def test_valid_file_round_trips_with_stable_digest() -> None:
    registry = assemble_venue_halt_registry([_halt(), _halt(T0 + pd.Timedelta(hours=1))])
    reparsed = parse_venue_halt_registry(venue_halt_registry_jsonl(registry), source="test")
    assert reparsed == registry
    assert reparsed.digest == registry.digest
    assert venue_halt_registry_jsonl(reparsed) == venue_halt_registry_jsonl(registry)


@pytest.mark.parametrize(
    "override",
    [
        {"start": "2022-05-01T22:36:00Z", "end": "2022-05-01T22:27:00Z"},
        {"start": "2022-05-01T22:27:30Z"},
        {"end": "2022-05-01T22:36:30Z"},
        {"present_symbols": 9},
        {"present_symbols": True},
        {"zero_symbols": 10},
        {"zero_symbols": 13},
        {"zero_symbols": "12"},
        {"start": "2022-05-01 22:27:00"},
        {"start": ""},
        {"start": "2022-05-01T22:27:00+02:00"},
        {"start": "not-a-timestamp"},
        {"halt_id": ""},
        {"halt_id": 7},
        {"kind": "settlement"},
        {"evidence": "  "},
        {"verified_at": "2026-07-01 00:00:00"},
        {"present_symbols": 12, "zero_symbols": 12, "extra": 1},
    ],
)
def test_halt_registry_validation_rejects(override) -> None:
    with pytest.raises(DataIntegrityError):
        parse_venue_halt_registry(_raw(**override), source="test")


def test_halt_registry_rejects_malformed_inputs() -> None:
    with pytest.raises(DataIntegrityError):
        parse_venue_halt_registry(b"{oops", source="test")
    with pytest.raises(DataIntegrityError):
        parse_venue_halt_registry(b"\xff\xfe", source="test")
    with pytest.raises(DataIntegrityError):
        parse_venue_halt_registry(b"[1, 2]\n", source="test")
    missing = {k: v for k, v in json.loads(_raw()).items() if k != "halt_id"}

    with pytest.raises(DataIntegrityError):
        parse_venue_halt_registry((json.dumps(missing) + "\n").encode(), source="test")
    assert parse_venue_halt_registry(b"", source="test") == EMPTY_VENUE_HALT_REGISTRY
    assert parse_venue_halt_registry(b"\n", source="test") == EMPTY_VENUE_HALT_REGISTRY
    assert venue_halt_registry_jsonl(EMPTY_VENUE_HALT_REGISTRY) == b""


def test_overlapping_halts_rejected() -> None:
    with pytest.raises(DataIntegrityError, match=r"[Oo]verlap"):
        assemble_venue_halt_registry([_halt(), _halt(T0 + pd.Timedelta(minutes=3))])


def test_committed_halt_registry_loads() -> None:
    registry = load_venue_halt_registry()
    assert [h.halt_id for h in registry.halts] == [
        "2021-03-02T01:03Z",
        "2022-05-01T22:27Z",
        "2022-05-28T16:42Z",
        "2023-09-12T08:36Z",
        "2023-11-14T11:39Z",
        "2024-10-28T16:21Z",
        "2024-10-28T20:00Z",
        "2025-01-29T01:24Z",
        "2025-01-29T02:42Z",
        "2025-08-29T06:18Z",
    ]
    expected = [
        ("2021-03-02T01:03Z", "2021-03-02T02:00Z", 19, 88, 88),
        ("2022-05-01T22:27Z", "2022-05-01T22:54Z", 9, 141, 141),
        ("2022-05-28T16:42Z", "2022-05-28T17:15Z", 11, 139, 139),
        ("2023-09-12T08:36Z", "2023-09-12T08:51Z", 5, 200, 200),
        ("2023-11-14T11:39Z", "2023-11-14T11:42Z", 1, 227, 225),
        ("2024-10-28T16:21Z", "2024-10-28T16:33Z", 4, 306, 306),
        ("2024-10-28T20:00Z", "2024-10-28T21:12Z", 24, 306, 306),
        ("2025-01-29T01:24Z", "2025-01-29T01:36Z", 4, 363, 362),
        ("2025-01-29T02:42Z", "2025-01-29T02:45Z", 1, 363, 362),
        ("2025-08-29T06:18Z", "2025-08-29T06:36Z", 6, 481, 461),
    ]
    for halt, (halt_id, end, bars, present, zero) in zip(registry.halts, expected, strict=True):
        assert halt.halt_id == halt_id
        assert halt.start.strftime("%Y-%m-%dT%H:%MZ") == halt_id
        assert halt.end == pd.Timestamp(end, tz="UTC")
        assert int((halt.end - halt.start).total_seconds() // 180) == bars
        assert halt.present_symbols == present
        assert halt.zero_symbols == zero
    assert default_venue_halt_registry_path().exists()


def test_root_binding(tmp_path) -> None:
    from src.common.paths import FUTURES_DATA_DIR

    canonical = venue_halt_registry_for_root(str(FUTURES_DATA_DIR / "ohlcv"))
    assert len(canonical.halts) == 10
    assert venue_halt_registry_for_root(tmp_path) == EMPTY_VENUE_HALT_REGISTRY


def test_overlapping_selects_window_halts() -> None:
    registry = load_venue_halt_registry()
    start = pd.Timestamp("2021-03-02T01:00Z")
    end = pd.Timestamp("2021-03-02T02:03Z")
    assert [h.halt_id for h in registry.overlapping(start, end)] == ["2021-03-02T01:03Z"]
    assert registry.overlapping(pd.Timestamp("2021-03-03T00:00Z"), pd.Timestamp("2021-03-04T00:00Z")) == ()


def test_explicit_path_and_cache(tmp_path) -> None:
    from src.mhs.venue_halts import clear_venue_halt_registry_cache

    target = tmp_path / "halts.jsonl"
    target.write_bytes(venue_halt_registry_jsonl(assemble_venue_halt_registry([_halt()])))
    assert len(load_venue_halt_registry(target).halts) == 1
    clear_venue_halt_registry_cache()
    assert load_venue_halt_registry().digest == load_venue_halt_registry().digest
    with pytest.raises(DataIntegrityError):
        load_venue_halt_registry(tmp_path / "absent.jsonl")
