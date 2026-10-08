"""Invariant guards for the point-in-time instrument settlement registry (spec 34 part 1)."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs import instrument_settlements as settlements_mod
from src.mhs.instrument_settlements import (
    EMPTY_SETTLEMENT_REGISTRY,
    clear_instrument_settlement_registry_cache,
    load_instrument_settlement_registry,
    parse_instrument_settlement_registry,
    record_digest,
    settlement_registry_for_root,
)

_LAST = "2022-01-01T00:00:00Z"
_LAST_MS = int(pd.Timestamp("2022-01-01T00:00:00Z").value // 1_000_000)


def _settlement(**overrides: object) -> dict:
    symbol = str(overrides.get("symbol", "AAAUSDT"))
    last_trade_at = str(overrides.get("last_trade_at", _LAST))
    delivery_at = str(overrides.get("delivery_at", last_trade_at))
    if "announced_at" in overrides:
        announced_at = str(overrides["announced_at"])
    else:
        try:
            stamp = pd.Timestamp(last_trade_at)
            if pd.isna(stamp):
                raise ValueError("unparseable last_trade_at")
            announced_at = (stamp - pd.Timedelta(days=7)).isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError):
            announced_at = "2021-12-25T00:00:00Z"
    payload: dict = {
        "kind": "settlement",
        "symbol": symbol,
        "event_id": f"{symbol}:0",
        "announced_at": announced_at,
        "announcement_source": "proxy_lead",
        "announcement_evidence": "",
        "last_trade_at": last_trade_at,
        "delivery_at": delivery_at,
        "settlement_price": 1.25,
        "price_source": "flat_1h_klines",
        "price_evidence": "flat_1h_klines delivery=2022-01-01T00:00:00+00:00 flat_bars=5 price=1.25",
        "fee_bps": 5.0,
        "evidence_digest": "sha256:" + "ab" * 32,
        "verified_at": "2026-07-01T00:00:00Z",
    }
    payload.update(overrides)
    if "event_id" not in overrides:
        try:
            delivery_stamp = pd.Timestamp(str(payload["delivery_at"]))
            if pd.isna(delivery_stamp):
                raise ValueError("unparseable delivery_at")
            stamp_ms = int(delivery_stamp.value // 1_000_000)
        except (ValueError, TypeError):
            stamp_ms = 0
        payload["event_id"] = f"{symbol}:{stamp_ms}"
    return payload


def _truncation(**overrides: object) -> dict:
    payload: dict = {
        "kind": "data_truncation",
        "symbol": "ZZZUSDT",
        "data_end": "2026-07-01T00:00:00Z",
        "evidence": "collection horizon, archive ends at the lake end",
        "verified_at": "2026-07-01T00:00:00Z",
    }
    payload.update(overrides)
    return payload


def _parse_many(*payloads: dict) -> settlements_mod.InstrumentSettlementRegistry:
    raw = ("\n".join(json.dumps(payload) for payload in payloads) + "\n").encode()
    return parse_instrument_settlement_registry(raw, source="test")


def test_valid_registry_round_trip() -> None:
    curated = _settlement(
        symbol="BBBUSDT",
        last_trade_at="2022-04-11T09:00:00Z",
        announcement_source="curated",
        announcement_evidence="Binance notice: delisting on 2022-04-11",
        announced_at="2022-04-05T12:34:56Z",
        price_source="twap30_proxy",
        price_evidence="twap30_proxy window=final half hour bars=10",
    )
    registry = _parse_many(_settlement(), curated, _truncation())
    first = registry.settlements_for("AAAUSDT")[0]
    assert first.symbol == "AAAUSDT"
    assert first.event_id == f"AAAUSDT:{_LAST_MS}"
    assert first.last_trade_at == pd.Timestamp(_LAST)
    assert first.delivery_at == pd.Timestamp(_LAST)
    assert first.announced_at == pd.Timestamp(_LAST) - pd.Timedelta(days=7)
    assert first.settlement_price == 1.25
    assert registry.settlements_for("BBBUSDT")[0].announcement_evidence.startswith("Binance notice")
    assert registry.truncation_for("ZZZUSDT") is not None
    assert registry.truncation_for("AAAUSDT") is None
    assert record_digest(first) == record_digest(first)


def test_digest_is_formatting_independent() -> None:
    first, second = _settlement(), _settlement(symbol="BBBUSDT")
    plain = ("\n".join(json.dumps(payload) for payload in (first, second)) + "\n").encode()
    shuffled = ("\n".join(
        json.dumps(payload, indent=2, sort_keys=True) for payload in (second, first)
    ) + "\n").encode()
    assert (
        parse_instrument_settlement_registry(plain, source="a").digest
        == parse_instrument_settlement_registry(shuffled, source="b").digest
    )
    bumped = dict(first, settlement_price=math.nextafter(1.25, 2.0))
    assert (
        parse_instrument_settlement_registry(plain, source="a").digest
        != _parse_many(bumped, second).digest
    )


def test_lifecycle_ordering_enforced() -> None:
    with pytest.raises(DataIntegrityError):
        _parse_many(_settlement(announced_at="2022-01-02T00:00:00Z"))
    with pytest.raises(DataIntegrityError):
        _parse_many(_settlement(delivery_at="2021-12-31T21:00:00Z"))


def test_proxy_lead_must_match_param() -> None:
    announced = (pd.Timestamp(_LAST) - pd.Timedelta(days=6)).isoformat().replace("+00:00", "Z")
    with pytest.raises(DataIntegrityError, match="regenerate the registry"):
        _parse_many(_settlement(announced_at=announced))


def test_off_grid_timestamp_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        _parse_many(_settlement(last_trade_at="2022-01-01T00:01:00Z"))


def test_submicrosecond_grid_violation_is_not_truncated() -> None:
    payload = _settlement()
    payload["last_trade_at"] = "2022-01-01T00:00:00.000000001Z"
    with pytest.raises(DataIntegrityError, match="off the 3m grid"):
        _parse_many(payload)


def test_registry_lookup_indexes_are_immutable() -> None:
    registry = _parse_many(_settlement())
    with pytest.raises(TypeError):
        registry._settlement_index["AAAUSDT"] = ()


@pytest.mark.parametrize("price", [0, -1, float("nan")])
def test_non_positive_or_non_finite_price_rejected(price: float) -> None:
    line = json.dumps(_settlement(settlement_price=price))
    with pytest.raises(DataIntegrityError):
        parse_instrument_settlement_registry(line.encode(), source="test")


def test_curated_without_evidence_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        _parse_many(
            _settlement(announcement_source="curated", announcement_evidence="   "),
        )


def test_duplicate_and_overlapping_lifecycles_rejected() -> None:
    with pytest.raises(DataIntegrityError, match="duplicate event_id"):
        _parse_many(_settlement(), _settlement())
    early = _settlement(
        symbol="BBBUSDT",
        last_trade_at="2022-04-11T09:00:00Z",
        delivery_at="2022-04-11T09:00:00Z",
    )
    late_announced = (pd.Timestamp("2022-04-11T09:00:00Z") - pd.Timedelta(days=7)).isoformat().replace("+00:00", "Z")
    late = _settlement(
        symbol="BBBUSDT",
        last_trade_at="2022-05-11T09:00:00Z",
        delivery_at="2022-05-11T09:00:00Z",
        announced_at=late_announced,
        announcement_source="curated",
        announcement_evidence="Binance notice: second delisting wave",
    )
    assert early["announced_at"] <= late["announced_at"] <= early["delivery_at"]
    with pytest.raises(DataIntegrityError, match="overlapping lifecycles"):
        _parse_many(early, late)


def test_unknown_or_extra_field_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        _parse_many(dict(_settlement(), foo="bar"))
    with pytest.raises(DataIntegrityError):
        _parse_many({"kind": "mystery", "symbol": "AAAUSDT"})


def test_root_binding(tmp_path) -> None:
    from src.common.paths import FUTURES_DATA_DIR

    canonical = settlement_registry_for_root(str(FUTURES_DATA_DIR / "ohlcv"))
    assert len(canonical.settlements) == 145
    assert settlement_registry_for_root(tmp_path) is EMPTY_SETTLEMENT_REGISTRY


def test_default_cache_observes_file_changes(tmp_path, monkeypatch) -> None:
    target = tmp_path / "instrument_settlements.jsonl"
    target.write_bytes((json.dumps(_settlement()) + "\n").encode())
    monkeypatch.setattr(
        settlements_mod, "default_instrument_settlement_registry_path", lambda: target,
    )
    clear_instrument_settlement_registry_cache()
    try:
        first = load_instrument_settlement_registry()
        assert load_instrument_settlement_registry() is first
        target.write_bytes((json.dumps(_settlement(symbol="BBBUSDT")) + "\n").encode())
        second = load_instrument_settlement_registry()
        assert second is not first
        assert second.settlements_for("BBBUSDT")
        assert load_instrument_settlement_registry() is second
    finally:
        clear_instrument_settlement_registry_cache()


def test_committed_registry_loads() -> None:
    clear_instrument_settlement_registry_cache()
    try:
        registry = load_instrument_settlement_registry()
    finally:
        clear_instrument_settlement_registry_cache()
    assert len(registry.settlements) == 145
    assert len(registry.truncations) == 0
    counts: dict[str, int] = {}
    for record in registry.settlements:
        counts[record.price_source] = counts.get(record.price_source, 0) + 1
        assert record.announcement_source == "proxy_lead"
    assert counts == {"flat_1h_klines": 137, "twap30_proxy": 8}


@pytest.mark.parametrize(
    "moment", ["", "not-a-time", "2022-01-01T00:00:00", "2022-01-01T02:00:00+02:00"],
)
def test_malformed_timestamps_rejected(moment: str) -> None:
    with pytest.raises(DataIntegrityError):
        _parse_many(_settlement(last_trade_at=moment))


def test_non_string_timestamp_rejected() -> None:
    payload = _settlement()
    payload["last_trade_at"] = 1640995200000
    with pytest.raises(DataIntegrityError):
        _parse_many(payload)


def test_scalar_fields_validated() -> None:
    bad_payloads = [
        _settlement(symbol="aaa"),
        _settlement(settlement_price="1.25"),
        _settlement(fee_bps="5.0"),
        _settlement(fee_bps=-1.0),
        _settlement(announcement_source="bogus"),
        _settlement(price_source="bogus"),
        _settlement(announcement_evidence=123),
        _settlement(price_evidence="   "),
        _settlement(evidence_digest=""),
        _settlement(event_id="AAAUSDT:1"),
        _truncation(evidence="  "),
    ]
    for payload in bad_payloads:
        with pytest.raises(DataIntegrityError):
            _parse_many(payload)
    incomplete = _truncation()
    del incomplete["evidence"]
    with pytest.raises(DataIntegrityError):
        _parse_many(incomplete)


def test_jsonl_envelope_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        parse_instrument_settlement_registry(b"{oops", source="test")
    with pytest.raises(DataIntegrityError):
        parse_instrument_settlement_registry(b"[1, 2]\n", source="test")
    with pytest.raises(DataIntegrityError):
        parse_instrument_settlement_registry(b"\xff\xfe\x00", source="test")


def test_truncation_conflicts_rejected() -> None:
    with pytest.raises(DataIntegrityError, match="more than one truncation"):
        _parse_many(_settlement(), _truncation(symbol="AAAUSDT"), _truncation(symbol="AAAUSDT"))
    inside = _truncation(symbol="AAAUSDT", data_end="2021-12-30T00:00:00Z")
    with pytest.raises(DataIntegrityError, match="inside"):
        _parse_many(_settlement(), inside)


def test_empty_registry_serializes_empty() -> None:
    from src.mhs.instrument_settlements import settlement_registry_jsonl

    assert settlement_registry_jsonl(EMPTY_SETTLEMENT_REGISTRY) == b""


def test_loader_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(DataIntegrityError, match="unreadable"):
        load_instrument_settlement_registry(tmp_path / "absent.jsonl")


def _gap_interval(symbol: str, reason: str, extent: str, start: str, end: str | None = None):
    from datetime import UTC

    from src.mhs.source_gaps import SourceGapInterval

    def _dt(value: str):
        return pd.Timestamp(value, tz="UTC").to_pydatetime().astimezone(UTC)

    verified = _dt("2026-01-01T00:00:00Z")
    return SourceGapInterval(
        symbol=symbol, plane="ohlcv_3m", start=_dt(start),
        end=_dt(end) if end is not None else None, reason=reason,
        evidence="test fixture", verified_at=verified, resolved_at=None,
        extent=extent,  # type: ignore[arg-type]
    )


def test_superseded_delisted_by_any_record() -> None:
    from src.mhs.instrument_settlements import source_gap_superseded_by_settlement

    registry = _parse_many(_settlement(symbol="LUNAUSDT", last_trade_at="2022-05-12T15:33:00Z"))
    assert source_gap_superseded_by_settlement(
        _gap_interval("LUNAUSDT", "DELISTED", "UNSCOPED", "2021-01-01T00:00:00Z"), registry
    ) is True
    assert source_gap_superseded_by_settlement(
        _gap_interval("LUNAUSDT", "DELISTED", "UNSCOPED", "2021-01-01T00:00:00Z"),
        EMPTY_SETTLEMENT_REGISTRY,
    ) is False


def test_superseded_post_trade_open_edge() -> None:
    from src.mhs.instrument_settlements import source_gap_superseded_by_settlement

    registry = _parse_many(_settlement(symbol="AAAUSDT", last_trade_at="2022-01-01T00:00:00Z"))
    assert source_gap_superseded_by_settlement(
        _gap_interval("AAAUSDT", "SOURCE_ABSENT", "OPEN_EDGE", "2022-01-01T00:00:00Z"), registry
    ) is True
    assert source_gap_superseded_by_settlement(
        _gap_interval("AAAUSDT", "SOURCE_ABSENT", "OPEN_EDGE", "2021-12-31T00:00:00Z"), registry
    ) is False


def test_pre_delivery_unscoped_not_superseded() -> None:
    from src.mhs.instrument_settlements import source_gap_superseded_by_settlement

    registry = _parse_many(_settlement(symbol="BTCSTUSDT", last_trade_at="2021-03-12T02:00:00Z"))
    assert source_gap_superseded_by_settlement(
        _gap_interval("BTCSTUSDT", "SOURCE_ABSENT", "UNSCOPED", "2021-01-01T00:00:00Z", "2021-03-04T07:00:00Z"),
        registry,
    ) is False


def test_interior_never_superseded() -> None:
    from src.mhs.instrument_settlements import source_gap_superseded_by_settlement

    registry = _parse_many(_settlement(symbol="AAAUSDT", last_trade_at="2022-01-01T00:00:00Z"))
    assert source_gap_superseded_by_settlement(
        _gap_interval("AAAUSDT", "SOURCE_ABSENT", "INTERIOR", "2022-01-02T00:00:00Z", "2022-01-03T00:00:00Z"),
        registry,
    ) is False
