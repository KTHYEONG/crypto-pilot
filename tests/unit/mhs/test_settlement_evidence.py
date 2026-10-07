"""Invariant guards for lake settlement evidence and the registry audit (spec 34 part 1)."""

from __future__ import annotations

import pandas as pd
import pytest

import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.mhs.instrument_settlements import (
    EMPTY_SETTLEMENT_REGISTRY,
    InstrumentSettlementRecord,
    assemble_instrument_settlement_registry,
)
from src.mhs.settlement_evidence import (
    assert_settlement_registry_complete,
    audit_settlement_registry,
    derive_settlement_price,
    measure_symbol_tail,
)

T0 = pd.Timestamp("2025-01-01T00:00:00Z")
STEP = pd.Timedelta(minutes=3)


def _stamps(start: pd.Timestamp, count: int, step: pd.Timedelta = STEP) -> list[int]:
    return [int((start + step * index).value // 1_000_000) for index in range(count)]


def _write_3m(path, stamps: list[int], closes: list[float], quote_vols: list[float]) -> None:
    pd.DataFrame({
        "timestamp": stamps,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "quote_vol": quote_vols,
    }).to_parquet(path, index=False)


def _write_1h(path, stamps: list[int], closes: list[float], volumes: list[float]) -> None:
    pd.DataFrame({
        "timestamp": stamps,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": volumes, "quote_vol": [c * v for c, v in zip(closes, volumes, strict=True)],
    }).to_parquet(path, index=False)


def _record(symbol: str, last_trade: pd.Timestamp, **overrides) -> InstrumentSettlementRecord:
    delivery = overrides.pop("delivery", last_trade)
    announced = overrides.pop("announced", last_trade - pd.Timedelta(days=7))
    fields = {
        "symbol": symbol,
        "event_id": f"{symbol}:{int(delivery.value // 1_000_000)}",
        "announced_at": announced,
        "announcement_source": "proxy_lead",
        "announcement_evidence": "",
        "last_trade_at": last_trade,
        "delivery_at": delivery,
        "settlement_price": 1.25,
        "price_source": "flat_1h_klines",
        "price_evidence": "flat_1h_klines test evidence",
        "fee_bps": 5.0,
        "evidence_digest": "sha256:" + "00" * 32,
        "verified_at": pd.Timestamp("2026-07-01T00:00:00Z"),
    }
    fields.update(overrides)
    return InstrumentSettlementRecord(**fields)  # type: ignore[arg-type]


def test_tail_profile_on_flat_tail(tmp_path) -> None:
    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 25)
    _write_3m(path, stamps, [1.0 + 0.01 * i for i in range(20)] + [2.0] * 5,
              [10.0] * 20 + [0.0] * 5)
    profile = measure_symbol_tail(path)
    assert profile.last_liquid_bar == T0 + STEP * 19
    assert profile.trailing_flat_bars == 5
    assert profile.last_bar == T0 + STEP * 24
    assert profile.first_bar == T0


def test_interior_flat_run_is_not_a_tail(tmp_path) -> None:
    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 6)
    _write_3m(path, stamps, [1.0, 1.0, 1.0, 1.0, 1.0, 1.1],
              [10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    profile = measure_symbol_tail(path)
    assert profile.trailing_flat_bars == 0
    assert profile.last_liquid_bar == T0 + STEP * 5


def test_tail_scan_reads_suffix_only(tmp_path, monkeypatch) -> None:
    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 100)
    frame = pd.DataFrame({
        "timestamp": stamps,
        "open": [1.0] * 100, "high": [1.0] * 100, "low": [1.0] * 100,
        "close": [1.0] * 100, "quote_vol": [10.0] * 100,
    })
    frame.to_parquet(path, index=False, row_group_size=10)
    assert pq.ParquetFile(path).metadata.num_row_groups == 10
    calls: list[int] = []
    real = pq.ParquetFile.read_row_group

    def _spy(self, index: int, *args, **kwargs):
        calls.append(index)
        return real(self, index, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _spy)
    profile = measure_symbol_tail(path)
    assert profile.last_liquid_bar == T0 + STEP * 99
    assert calls == [9]


def test_flat_1h_evidence_priced(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    liquid_3m = _stamps(T0, 20)
    _write_3m(root / "3m" / "AAAUSDT.parquet", liquid_3m,
              [1.0 + 0.01 * i for i in range(20)], [10.0] * 20)
    last_trade = T0 + STEP * 20
    assert last_trade == pd.Timestamp("2025-01-01T01:00:00Z")
    (root / "1h").mkdir(parents=True)
    hour = pd.Timedelta(hours=1)
    liquid_1h = [int((T0 + hour * 0).value // 1_000_000)]
    flat_1h = [int((T0 + hour * i).value // 1_000_000) for i in (1, 2, 3)]
    _write_1h(root / "1h" / "AAAUSDT.parquet", liquid_1h + flat_1h,
              [1.20] + [1.25] * 3, [5.0] + [0.0] * 3)
    first = derive_settlement_price(root, "AAAUSDT", last_trade)
    second = derive_settlement_price(root, "AAAUSDT", last_trade)
    assert first is not None
    assert first.price_source == "flat_1h_klines"
    assert first.settlement_price == 1.25
    assert second is not None
    assert second.evidence_digest == first.evidence_digest


def test_twap_proxy_priced(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    stamps = _stamps(T0, 10)
    closes = [float(100 + i) for i in range(10)]
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps, closes, [10.0] * 10)
    last_trade = T0 + STEP * 10
    evidence = derive_settlement_price(root, "AAAUSDT", last_trade)
    assert evidence is not None
    assert evidence.price_source == "twap30_proxy"
    assert evidence.settlement_price == pytest.approx(sum(closes) / len(closes))


def test_insufficient_twap_bars_yields_none(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    stamps = _stamps(T0, 10)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps,
              [float(100 + i) for i in range(10)], [10.0] * 4 + [0.0] * 6)
    last_trade = T0 + STEP * 10
    assert derive_settlement_price(root, "AAAUSDT", last_trade) is None


def test_twap_excludes_flat_bars_and_accumulates_float32_closes_in_float64(tmp_path) -> None:
    import hashlib
    import json

    import numpy as np

    (tmp_path / "3m").mkdir()
    stamps = _stamps(T0, 10)
    closes = np.array([0.1, 900.0, 0.2, 800.0, 0.3, 700.0, 0.4, 600.0, 0.5, 500.0], dtype="float32")
    volumes = [10.0, 0.0] * 5
    _write_3m(tmp_path / "3m" / "AAAUSDT.parquet", stamps, closes, volumes)
    evidence = derive_settlement_price(tmp_path, "AAAUSDT", T0 + STEP * 10)
    assert evidence is not None
    assert evidence.settlement_price == float(closes[::2].astype("float64").mean())
    bars = [[stamps[i], *([float(closes[i])] * 4), volumes[i]] for i in range(0, 10, 2)]
    payload = {"symbol": "AAAUSDT", "price_source": "twap30_proxy", "bars": bars}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert evidence.evidence_digest == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_ended_symbol_without_record_is_missing(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    audit_end = pd.Timestamp("2025-02-01T00:00:00Z")
    stamps = _stamps(T0, 20)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps,
              [1.0] * 20, [10.0] * 20)
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=audit_end, registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert [symbol for symbol, _reason in report.missing] == ["AAAUSDT"]
    with pytest.raises(DataIntegrityError, match="AAAUSDT"):
        assert_settlement_registry_complete(
            root, ["AAAUSDT"], audit_end=audit_end, registry=EMPTY_SETTLEMENT_REGISTRY,
        )


def test_flat_tail_without_record_is_missing(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    audit_end = pd.Timestamp("2025-01-02T00:00:00Z")
    liquid = _stamps(T0, 20)
    flat_start = T0 + STEP * 20
    flat = _stamps(flat_start, 100)
    _write_3m(root / "3m" / "AAAUSDT.parquet", liquid + flat,
              [1.0] * 20 + [2.0] * 100, [10.0] * 20 + [0.0] * 100)
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=audit_end, registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert not report.complete
    assert [symbol for symbol, _reason in report.missing] == ["AAAUSDT"]


def test_matching_record_completes_audit(tmp_path) -> None:
    from src.application.ops.settlement_registry import build_settlement_registry

    root = tmp_path
    (root / "3m").mkdir(parents=True)
    audit_end = pd.Timestamp("2025-01-02T00:00:00Z")
    liquid = _stamps(T0, 20)
    flat = _stamps(T0 + STEP * 20, 100)
    _write_3m(root / "3m" / "AAAUSDT.parquet", liquid + flat,
              [2.0] * 20 + [2.0] * 100, [10.0] * 20 + [0.0] * 100)
    (root / "1h").mkdir(parents=True)
    hour = pd.Timedelta(hours=1)
    stamps_1h = [int((T0 + hour * i).value // 1_000_000) for i in range(5)]
    _write_1h(root / "1h" / "AAAUSDT.parquet", stamps_1h, [2.0] * 5,
              [5.0] + [0.0] * 4)
    build = build_settlement_registry(
        root, horizon=audit_end, existing=EMPTY_SETTLEMENT_REGISTRY,
        verified_at=audit_end, symbols=["AAAUSDT"],
    )
    assert not build.unresolved
    assert build.registry.settlements_for("AAAUSDT")[0].price_source == "flat_1h_klines"
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=audit_end, registry=build.registry,
    )
    assert report.complete


def test_truncation_explains_collection_end_only(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    audit_end = pd.Timestamp("2025-02-01T00:00:00Z")
    stamps = _stamps(T0, 20)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps, [1.0] * 20, [10.0] * 20)
    last_bar = T0 + STEP * 19
    truncation = assemble_instrument_settlement_registry(
        [],
        [settlements_truncation("AAAUSDT", last_bar + STEP)],
    )
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=audit_end, registry=truncation,
    )
    assert report.complete
    flat = _stamps(T0 + STEP * 20, 100)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps + flat,
              [1.0] * 20 + [2.0] * 100, [10.0] * 20 + [0.0] * 100)
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=audit_end, registry=truncation,
    )
    assert [symbol for symbol, _reason in report.missing] == ["AAAUSDT"]


def settlements_truncation(symbol: str, data_end: pd.Timestamp):
    from src.mhs.instrument_settlements import DataTruncationRecord

    return DataTruncationRecord(
        symbol=symbol, data_end=data_end,
        evidence="test collection horizon",
        verified_at=pd.Timestamp("2026-07-01T00:00:00Z"),
    )


def test_stale_records_detected(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    stamps = _stamps(T0, 30)
    closes = [1.0 + 0.001 * i for i in range(30)]
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps, closes, [10.0] * 30)
    last_label = T0 + STEP * 29
    early = _record("AAAUSDT", last_label, price_source="curated",
                    price_evidence="notice", settlement_price=closes[-1])
    outside = _record("AAAUSDT", last_label + STEP, price_source="curated",
                      price_evidence="notice", settlement_price=1000.0)
    spanned = _record("AAAUSDT", last_label - STEP, delivery=last_label,
                      price_source="curated", price_evidence="notice",
                      settlement_price=closes[-2])
    registry = assemble_instrument_settlement_registry([early, outside, spanned], [])
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=pd.Timestamp("2025-02-01T00:00:00Z"), registry=registry,
    )
    assert {identity for identity, _reason in report.stale} == {
        early.event_id, outside.event_id, spanned.event_id,
    }


def test_horizon_bounds_the_audit(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    stamps = _stamps(T0, 20)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps, [1.0] * 20, [10.0] * 20)
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=T0 + STEP * 5, registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert report.complete


def test_symbols_without_archive_are_skipped(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    report = audit_settlement_registry(
        root, ["GHOSTUSDT"], audit_end=pd.Timestamp("2025-02-01T00:00:00Z"),
        registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert report.complete
    assert_settlement_registry_complete(
        root, ["GHOSTUSDT"], audit_end=pd.Timestamp("2025-02-01T00:00:00Z"),
        registry=EMPTY_SETTLEMENT_REGISTRY,
    )


def test_all_findings_reported(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    audit_end = pd.Timestamp("2025-02-01T00:00:00Z")
    for symbol in ("AAAUSDT", "BBBUSDT"):
        stamps = _stamps(T0, 20)
        _write_3m(root / "3m" / f"{symbol}.parquet", stamps, [1.0] * 20, [10.0] * 20)
    covering = _stamps(T0, 14880)
    _write_3m(root / "3m" / "CCCUSDT.parquet", covering,
              [1.0 + 0.0001 * i for i in range(14880)], [10.0] * 14880)
    stale_record = _record("CCCUSDT", T0 + STEP * 10, price_source="curated",
                           price_evidence="notice", settlement_price=1.0)
    registry = assemble_instrument_settlement_registry([stale_record], [])
    with pytest.raises(DataIntegrityError) as excinfo:
        assert_settlement_registry_complete(
            root, ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
            audit_end=audit_end, registry=registry,
        )
    message = str(excinfo.value)
    assert "AAAUSDT" in message
    assert "BBBUSDT" in message
    assert stale_record.event_id in message


def test_tail_reader_fails_closed(tmp_path) -> None:
    from src.mhs.settlement_evidence import _coerce_bars

    with pytest.raises(DataIntegrityError, match="unreadable"):
        measure_symbol_tail(tmp_path / "absent.parquet")
    junk = tmp_path / "junk.parquet"
    junk.write_bytes(b"not a parquet")
    with pytest.raises(DataIntegrityError, match="unreadable"):
        measure_symbol_tail(junk)
    narrow = tmp_path / "narrow.parquet"
    pd.DataFrame({"timestamp": [1], "open": [1.0]}).to_parquet(narrow, index=False)
    with pytest.raises(DataIntegrityError, match="required columns"):
        measure_symbol_tail(narrow)
    cols = ["timestamp", "open", "high", "low", "close", "quote_vol"]
    empty = tmp_path / "empty.parquet"
    pd.DataFrame({column: [] for column in cols}).to_parquet(empty, index=False)
    with pytest.raises(DataIntegrityError, match="empty"):
        measure_symbol_tail(empty)
    with pytest.raises(DataIntegrityError, match="no placeable bars"):
        _coerce_bars(pd.DataFrame({column: [] for column in cols}), "empty.parquet")
    frame = pd.DataFrame({
        "timestamp": ["oops"], "open": [1.0], "high": [1.0],
        "low": [1.0], "close": [1.0], "quote_vol": [0.0],
    })
    with pytest.raises(DataIntegrityError, match="non-finite bars"):
        _coerce_bars(frame, "junk.parquet")


def test_liquid_bar_with_nonfinite_close_cannot_be_priced(tmp_path) -> None:
    (tmp_path / "3m").mkdir()
    closes = [1.0] * 10
    closes[4] = float("nan")
    _write_3m(tmp_path / "3m" / "AAAUSDT.parquet", _stamps(T0, 10), closes, [10.0] * 10)
    with pytest.raises(DataIntegrityError, match="non-finite"):
        derive_settlement_price(tmp_path, "AAAUSDT", T0 + STEP * 10)


def test_first_bar_without_statistics(tmp_path) -> None:
    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 12)
    pd.DataFrame({
        "timestamp": stamps,
        "open": [1.0] * 12, "high": [1.0] * 12, "low": [1.0] * 12,
        "close": [1.0] * 12, "quote_vol": [10.0] * 12,
    }).to_parquet(path, index=False, write_statistics=False)
    profile = measure_symbol_tail(path)
    assert profile.first_bar == T0
    assert profile.last_bar == T0 + STEP * 11
    assert profile.last_liquid_bar == T0 + STEP * 11
    assert profile.trailing_flat_bars == 0


def test_statistics_failure_falls_back_to_first_group(tmp_path) -> None:
    from types import SimpleNamespace

    from src.mhs.settlement_evidence import _suffix_first_ms

    class _BadMetadata:
        def row_group(self, _index: int) -> None:
            raise RuntimeError("metadata unavailable")

    stub = SimpleNamespace(
        schema=SimpleNamespace(
            names=["timestamp", "open", "high", "low", "close", "quote_vol"],
        ),
        metadata=_BadMetadata(),
    )
    assert _suffix_first_ms(stub) is None

    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 12)
    _write_3m(path, stamps, [1.0] * 12, [10.0] * 12)
    profile = measure_symbol_tail(path)
    assert profile.first_bar == T0
    assert profile.last_liquid_bar == T0 + STEP * 11


def test_suffix_scan_failure_modes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 25)
    frame = pd.DataFrame({
        "timestamp": stamps,
        "open": [1.0] * 25, "high": [1.0] * 25, "low": [1.0] * 25,
        "close": [1.0] * 25, "quote_vol": [10.0] * 25,
    })
    frame.to_parquet(path, index=False, row_group_size=10)
    real = pq.ParquetFile.read_row_group

    def _passthrough(self, index: int, *args: object, **kwargs: object):
        if index == 2:
            raise DataIntegrityError("boom")
        return real(self, index, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _passthrough)
    with pytest.raises(DataIntegrityError, match="boom"):
        measure_symbol_tail(path)

    def _unexpected(self, index: int, *args: object, **kwargs: object):
        raise RuntimeError("disk hiccup")

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _unexpected)
    with pytest.raises(DataIntegrityError, match="unreadable"):
        measure_symbol_tail(path)


def test_first_group_fallback_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 25)
    pd.DataFrame({
        "timestamp": stamps,
        "open": [1.0] * 25, "high": [1.0] * 25, "low": [1.0] * 25,
        "close": [1.0] * 25, "quote_vol": [10.0] * 25,
    }).to_parquet(path, index=False, row_group_size=10, write_statistics=False)
    real = pq.ParquetFile.read_row_group

    def _fail_first(self, index: int, *args: object, **kwargs: object):
        if index == 0:
            raise RuntimeError("first group unreadable")
        return real(self, index, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _fail_first)
    with pytest.raises(DataIntegrityError, match="unreadable"):
        measure_symbol_tail(path)


def test_resolve_failure_fails_closed(tmp_path, monkeypatch) -> None:
    from pathlib import Path as _Path

    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    _write_3m(path, _stamps(T0, 5), [1.0] * 5, [10.0] * 5)
    monkeypatch.setattr(
        _Path, "resolve",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("resolve failed")),
    )
    with pytest.raises(DataIntegrityError, match="unreadable"):
        measure_symbol_tail(path)


def test_read_frame_failures(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    junk = root / "3m" / "AAAUSDT.parquet"
    junk.write_bytes(b"not a parquet")
    with pytest.raises(DataIntegrityError, match="unreadable"):
        derive_settlement_price(root, "AAAUSDT", T0 + STEP)
    narrow = root / "3m" / "BBBUSDT.parquet"
    pd.DataFrame({"timestamp": [1], "open": [1.0]}).to_parquet(narrow, index=False)
    with pytest.raises(DataIntegrityError, match="unreadable"):
        derive_settlement_price(root, "BBBUSDT", T0 + STEP)
    cols = ["timestamp", "open", "high", "low", "close", "quote_vol"]
    empty = root / "3m" / "CCCUSDT.parquet"
    pd.DataFrame({column: [] for column in cols}).to_parquet(empty, index=False)
    with pytest.raises(DataIntegrityError, match="empty"):
        derive_settlement_price(root, "CCCUSDT", T0 + STEP)


def test_derive_rejects_bad_moments_and_missing_archives(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    with pytest.raises(DataIntegrityError):
        derive_settlement_price(root, "AAAUSDT", None)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        derive_settlement_price(root, "AAAUSDT", pd.Timestamp("2025-01-01T00:00:00"))
    with pytest.raises(DataIntegrityError, match="unreadable"):
        derive_settlement_price(root, "GHOSTUSDT", T0 + STEP)


def test_flat_without_liquid_3m_bar_rejected(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    (root / "1h").mkdir(parents=True)
    stamps = _stamps(T0, 10)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps, [2.0] * 10, [0.0] * 10)
    hour = pd.Timedelta(hours=1)
    stamps_1h = [int((T0 + hour * i).value // 1_000_000) for i in range(4)]
    _write_1h(root / "1h" / "AAAUSDT.parquet", stamps_1h, [2.0] * 4, [0.0] * 4)
    with pytest.raises(DataIntegrityError, match="no liquid bar"):
        derive_settlement_price(root, "AAAUSDT", T0 + STEP * 10)


def test_envelope_without_liquid_window_is_false(tmp_path) -> None:
    from src.mhs.settlement_evidence import settlement_price_within_envelope

    root = tmp_path
    (root / "3m").mkdir(parents=True)
    _write_3m(root / "3m" / "AAAUSDT.parquet", _stamps(T0, 10), [2.0] * 10, [0.0] * 10)
    record = _record("AAAUSDT", T0 + STEP * 10, settlement_price=2.0)
    assert settlement_price_within_envelope(root, record) is False


def test_archive_starting_after_horizon_skipped(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    _write_3m(root / "3m" / "AAAUSDT.parquet", _stamps(T0, 10), [1.0] * 10, [10.0] * 10)
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=T0 - STEP, registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert report.complete


def test_future_record_reconciliation_skipped(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    stamps = _stamps(T0, 20000)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps, [1.0] * 20000, [10.0] * 20000)
    audit_end = T0 + STEP * 10000
    future = _record("AAAUSDT", T0 + STEP * 15000, settlement_price=999.0)
    registry = assemble_instrument_settlement_registry([future], [])
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=audit_end, registry=registry,
    )
    assert report.complete


def test_absent_last_trade_bar_is_stale(tmp_path) -> None:
    root = tmp_path
    (root / "3m").mkdir(parents=True)
    stamps = _stamps(T0, 20)
    _write_3m(root / "3m" / "AAAUSDT.parquet", stamps, [1.0] * 20, [10.0] * 20)
    record = _record("AAAUSDT", T0, price_source="curated",
                     price_evidence="notice", settlement_price=1.0)
    registry = assemble_instrument_settlement_registry([record], [])
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=pd.Timestamp("2025-02-01T00:00:00Z"), registry=registry,
    )
    assert [identity for identity, _reason in report.stale] == [record.event_id]


def test_evidence_mismatch_is_stale(tmp_path) -> None:
    import dataclasses

    from src.application.ops.settlement_registry import build_settlement_registry

    root = tmp_path
    (root / "3m").mkdir(parents=True)
    audit_end = pd.Timestamp("2025-01-02T00:00:00Z")
    liquid = _stamps(T0, 20)
    flat = _stamps(T0 + STEP * 20, 100)
    _write_3m(root / "3m" / "AAAUSDT.parquet", liquid + flat,
              [2.0] * 120, [10.0] * 20 + [0.0] * 100)
    (root / "1h").mkdir(parents=True)
    hour = pd.Timedelta(hours=1)
    stamps_1h = [int((T0 + hour * i).value // 1_000_000) for i in range(5)]
    _write_1h(root / "1h" / "AAAUSDT.parquet", stamps_1h, [2.0] * 5, [5.0] + [0.0] * 4)
    build = build_settlement_registry(
        root, horizon=audit_end, existing=EMPTY_SETTLEMENT_REGISTRY,
        verified_at=audit_end, symbols=["AAAUSDT"],
    )
    assert not build.unresolved
    genuine = build.registry.settlements_for("AAAUSDT")[0]
    tampered = dataclasses.replace(genuine, evidence_digest="sha256:" + "ff" * 32)
    registry = assemble_instrument_settlement_registry([tampered], [])
    report = audit_settlement_registry(
        root, ["AAAUSDT"], audit_end=audit_end, registry=registry,
    )
    assert [identity for identity, _reason in report.stale] == [tampered.event_id]


def test_never_liquid_tail(tmp_path) -> None:
    path = tmp_path / "3m" / "AAAUSDT.parquet"
    path.parent.mkdir(parents=True)
    stamps = _stamps(T0, 10)
    _write_3m(path, stamps, [2.0] * 10, [0.0] * 10)
    profile = measure_symbol_tail(path)
    assert profile.last_liquid_bar is None
    assert profile.trailing_flat_bars == 10
    assert profile.last_bar == T0 + STEP * 9
