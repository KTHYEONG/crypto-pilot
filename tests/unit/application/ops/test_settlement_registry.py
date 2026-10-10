"""Invariant guards for the operator settlement-registry generator (spec 34 part 1)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.application.ops.settlement_registry import (
    build_settlement_registry,
    write_settlement_registry,
)
from src.common.errors import DataIntegrityError
from src.core.instrument_settlements import (
    EMPTY_SETTLEMENT_REGISTRY,
    assemble_instrument_settlement_registry,
    parse_instrument_settlement_registry,
)
from src.core.settlement_evidence import audit_settlement_registry

T0 = pd.Timestamp("2025-01-01T00:00:00Z")
STEP = pd.Timedelta(minutes=3)
HORIZON = pd.Timestamp("2025-01-02T00:00:00Z")


def _stamps(start: pd.Timestamp, count: int) -> list[int]:
    return [int((start + STEP * index).value // 1_000_000) for index in range(count)]


def _write_3m(path, stamps: list[int], closes: list[float], quote_vols: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": stamps,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "quote_vol": quote_vols,
    }).to_parquet(path, index=False)


def _write_1h(path, stamps: list[int], closes: list[float], volumes: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": stamps,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": volumes, "quote_vol": [c * v for c, v in zip(closes, volumes, strict=True)],
    }).to_parquet(path, index=False)


def _lake(root) -> None:
    liquid = _stamps(T0, 20)
    flat = _stamps(T0 + STEP * 20, 100)
    _write_3m(root / "3m" / "FLATUSDT.parquet", liquid + flat,
              [2.0] * 120, [10.0] * 20 + [0.0] * 100)
    hour = pd.Timedelta(hours=1)
    stamps_1h = [int((T0 + hour * i).value // 1_000_000) for i in range(5)]
    _write_1h(root / "1h" / "FLATUSDT.parquet", stamps_1h, [2.0] * 5, [5.0] + [0.0] * 4)
    abrupt = _stamps(T0, 10)
    _write_3m(root / "3m" / "ABRUPTUSDT.parquet", abrupt,
              [float(50 + i) for i in range(10)], [10.0] * 10)
    live = _stamps(T0, 9600)
    _write_3m(root / "3m" / "LIVEUSDT.parquet", live,
              [3.0] * 9600, [10.0] * 9600)


def _build(root, **kwargs):
    params = {
        "horizon": HORIZON, "existing": EMPTY_SETTLEMENT_REGISTRY,
        "verified_at": HORIZON, "symbols": None,
    }
    params.update(kwargs)
    return build_settlement_registry(root, **params)


def test_builds_both_price_classes() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _lake(root)
        build = _build(root)
    assert not build.unresolved
    assert not build.changed
    by_symbol = {record.symbol: record for record in build.registry.settlements}
    assert set(by_symbol) == {"FLATUSDT", "ABRUPTUSDT"}
    assert by_symbol["FLATUSDT"].price_source == "flat_1h_klines"
    assert by_symbol["ABRUPTUSDT"].price_source == "twap30_proxy"
    for record in build.registry.settlements:
        assert record.announcement_source == "proxy_lead"
        assert record.announced_at == record.last_trade_at - pd.Timedelta(days=7)
        assert record.delivery_at == record.last_trade_at


def test_curated_fields_preserved(tmp_path) -> None:
    import dataclasses

    _lake(tmp_path)
    first = _build(tmp_path)
    assert not first.unresolved
    victim = first.registry.settlements_for("ABRUPTUSDT")[0]
    curated = dataclasses.replace(
        victim,
        announcement_source="curated",
        announced_at=victim.last_trade_at - pd.Timedelta(days=2),
        announcement_evidence="Binance notice: real announcement",
        verified_at=HORIZON,
    )
    existing = assemble_instrument_settlement_registry(
        [r for r in first.registry.settlements if r.symbol != "ABRUPTUSDT"] + [curated], [],
    )
    second = _build(tmp_path, existing=existing)
    kept = second.registry.settlements_for("ABRUPTUSDT")[0]
    assert kept.announced_at == curated.announced_at
    assert kept.announcement_source == "curated"
    assert kept.announcement_evidence == "Binance notice: real announcement"


def test_unresolved_blocks_write(tmp_path) -> None:
    stamps = _stamps(T0, 10)
    _write_3m(tmp_path / "3m" / "THINUSDT.parquet", stamps,
              [float(7 + i) for i in range(10)], [10.0] * 2 + [0.0] * 8)
    build = _build(tmp_path, symbols=["THINUSDT"])
    assert [symbol for symbol, _reason in build.unresolved] == ["THINUSDT"]
    target = tmp_path / "registry.jsonl"
    with pytest.raises(DataIntegrityError):
        write_settlement_registry(build, target)
    assert not target.exists()


def test_deterministic_bytes(tmp_path) -> None:
    _lake(tmp_path)
    first = _build(tmp_path)
    assert not first.unresolved
    out = tmp_path / "first.jsonl"
    assert write_settlement_registry(first, out) == 2
    raw_first = out.read_bytes()
    reparsed = parse_instrument_settlement_registry(raw_first, source=str(out))
    assert reparsed.digest == first.registry.digest
    second = _build(tmp_path, existing=reparsed)
    assert not second.changed
    out2 = tmp_path / "second.jsonl"
    write_settlement_registry(second, out2)
    assert out2.read_bytes() == raw_first


def test_generated_registry_passes_audit(tmp_path) -> None:
    _lake(tmp_path)
    build = _build(tmp_path)
    out = tmp_path / "registry.jsonl"
    write_settlement_registry(build, out)
    committed = parse_instrument_settlement_registry(out.read_bytes(), source=str(out))
    report = audit_settlement_registry(
        tmp_path, ["FLATUSDT", "ABRUPTUSDT", "LIVEUSDT"],
        audit_end=HORIZON, registry=committed,
    )
    assert report.complete
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["kind"] == "settlement"


def test_build_rejects_naive_moments(tmp_path) -> None:
    _lake(tmp_path)
    with pytest.raises(DataIntegrityError):
        _build(tmp_path, horizon=pd.Timestamp("2025-01-02T00:00:00"))
    with pytest.raises(DataIntegrityError):
        _build(tmp_path, verified_at=pd.Timestamp("2025-01-02T00:00:00"))
    with pytest.raises(DataIntegrityError):
        _build(tmp_path, horizon=None)  # type: ignore[arg-type]


def test_post_horizon_symbol_needs_no_record(tmp_path) -> None:
    stamps = [int((HORIZON + STEP * i).value // 1_000_000) for i in range(10)]
    _write_3m(tmp_path / "3m" / "LATEUSDT.parquet", stamps, [2.0] * 10, [10.0] * 10)
    build = _build(tmp_path, symbols=["LATEUSDT"])
    assert build.registry.settlements == ()
    assert build.unresolved == ()


def test_curated_price_preserved(tmp_path) -> None:
    import dataclasses

    _lake(tmp_path)
    first = _build(tmp_path)
    victim = first.registry.settlements_for("ABRUPTUSDT")[0]
    curated = dataclasses.replace(
        victim,
        price_source="curated",
        settlement_price=123.456,
        price_evidence="Binance settlement notice price",
        evidence_digest="sha256:" + "cd" * 32,
    )
    existing = assemble_instrument_settlement_registry(
        [r for r in first.registry.settlements if r.symbol != "ABRUPTUSDT"] + [curated], [],
    )
    second = _build(tmp_path, existing=existing)
    kept = second.registry.settlements_for("ABRUPTUSDT")[0]
    assert kept.settlement_price == 123.456
    assert kept.price_source == "curated"
    assert kept.price_evidence == "Binance settlement notice price"


def test_curated_carry_revalidates(tmp_path) -> None:
    import dataclasses

    _lake(tmp_path)
    first = _build(tmp_path)
    victim = first.registry.settlements_for("ABRUPTUSDT")[0]
    for mutate in (
        {"announcement_source": "curated", "announcement_evidence": "  "},
        {"announcement_source": "curated",
         "announcement_evidence": "notice",
         "announced_at": victim.last_trade_at + pd.Timedelta(hours=1)},
        {"price_source": "curated", "price_evidence": "  "},
    ):
        bad = dataclasses.replace(victim, **mutate)  # type: ignore[arg-type]
        existing = assemble_instrument_settlement_registry(
            [r for r in first.registry.settlements if r.symbol != "ABRUPTUSDT"] + [bad], [],
        )
        with pytest.raises(DataIntegrityError):
            _build(tmp_path, existing=existing)


def test_changed_record_reported(tmp_path) -> None:
    import dataclasses

    _lake(tmp_path)
    first = _build(tmp_path)
    victim = first.registry.settlements_for("ABRUPTUSDT")[0]
    drifted = dataclasses.replace(victim, settlement_price=victim.settlement_price + 1.0)
    existing = assemble_instrument_settlement_registry(
        [r for r in first.registry.settlements if r.symbol != "ABRUPTUSDT"] + [drifted], [],
    )
    second = _build(tmp_path, existing=existing)
    assert second.changed == (drifted.event_id,)


def test_never_liquid_lifecycle_unresolved(tmp_path) -> None:
    stamps = [int((T0 + STEP * i).value // 1_000_000) for i in range(10)]
    _write_3m(tmp_path / "3m" / "GHOSTUSDT.parquet", stamps, [2.0] * 10, [0.0] * 10)
    build = _build(tmp_path, symbols=["GHOSTUSDT"])
    assert [symbol for symbol, _reason in build.unresolved] == ["GHOSTUSDT"]


def test_missing_minute_archive_rejected(tmp_path) -> None:
    with pytest.raises(DataIntegrityError, match="build unreadable"):
        _build(tmp_path)
    (tmp_path / "3m").mkdir()
    with pytest.raises(DataIntegrityError, match="unreadable"):
        _build(tmp_path, symbols=["GHOSTUSDT"])


def test_unreadable_lake_root_rejected(tmp_path, monkeypatch) -> None:
    from pathlib import Path as _Path

    _lake(tmp_path)
    monkeypatch.setattr(
        _Path, "glob",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("glob failed")),
    )
    with pytest.raises(DataIntegrityError, match="unreadable"):
        _build(tmp_path)


def test_write_creates_parent_dirs(tmp_path) -> None:
    _lake(tmp_path)
    build = _build(tmp_path)
    target = tmp_path / "nested" / "dir" / "registry.jsonl"
    assert write_settlement_registry(build, target) == 2
    assert target.exists()


def test_write_failure_cleans_tmp(tmp_path) -> None:
    _lake(tmp_path)
    build = _build(tmp_path)
    target = tmp_path / "adir"
    target.mkdir()
    with pytest.raises(IsADirectoryError, match="Is a directory"):
        write_settlement_registry(build, target)
    assert list(tmp_path.glob(".settlements-*.tmp")) == []


def test_curated_price_survives_loss_of_proxy_evidence(tmp_path) -> None:
    import dataclasses

    _lake(tmp_path)
    initial = _build(tmp_path)
    record = initial.registry.settlements_for("ABRUPTUSDT")[0]
    curated = dataclasses.replace(record, price_source="curated", price_evidence="venue notice")
    existing = assemble_instrument_settlement_registry([curated], [])
    _write_3m(tmp_path / "3m" / "ABRUPTUSDT.parquet", _stamps(T0 + STEP * 8, 2),
              [58.0, 59.0], [10.0, 10.0])
    build = _build(tmp_path, existing=existing, symbols=["ABRUPTUSDT"])
    assert build.unresolved == ()
    assert build.registry.settlements_for("ABRUPTUSDT")[0] == curated


def test_subset_build_preserves_other_symbols_and_collection_declarations(tmp_path) -> None:
    from src.core.instrument_settlements import DataTruncationRecord

    _lake(tmp_path)
    initial = _build(tmp_path)
    flat = initial.registry.settlements_for("FLATUSDT")[0]
    truncation = DataTruncationRecord(
        symbol="ABRUPTUSDT", data_end=T0 + STEP * 10,
        evidence="collection stopped", verified_at=HORIZON,
    )
    existing = assemble_instrument_settlement_registry([flat], [truncation])
    build = _build(tmp_path, existing=existing, symbols=["ABRUPTUSDT"])
    assert build.registry.settlements == (flat,)
    assert build.registry.truncations == (truncation,)


def _halt_lake(root, n_live: int, n_zombie: int = 0) -> None:
    bars = _stamps(T0, 10)
    vols = [10.0] * 3 + [0.0] * 4 + [10.0] * 3
    for i in range(n_live):
        _write_3m(root / "3m" / f"LIVE{i:02d}USDT.parquet", bars,
                  [3.0] * 10, vols)
    for i in range(n_zombie):
        _write_3m(root / "3m" / f"ZOMB{i:02d}USDT.parquet", bars,
                  [3.0] * 10, vols)


def _zombie_registry(symbols: list[str]) -> object:
    from src.core.instrument_settlements import InstrumentSettlementRecord

    last_trade = T0 + STEP * 3
    records = [
        InstrumentSettlementRecord(
            symbol=symbol,
            event_id=f"{symbol}:{int(last_trade.value // 1_000_000)}",
            announced_at=last_trade - pd.Timedelta(days=7),
            announcement_source="proxy_lead",
            announcement_evidence="",
            last_trade_at=last_trade,
            delivery_at=last_trade,
            settlement_price=3.0,
            price_source="curated",
            price_evidence="test",
            fee_bps=5.0,
            evidence_digest="sha256:test",
            verified_at=HORIZON,
        )
        for symbol in symbols
    ]
    return assemble_instrument_settlement_registry(records, [])


def _build_halts(root, **kwargs):
    from src.application.ops.settlement_registry import build_venue_halt_registry

    params = {
        "horizon": T0 + STEP * 10, "settlements": EMPTY_SETTLEMENT_REGISTRY,
        "verified_at": HORIZON,
    }
    params.update(kwargs)
    return build_venue_halt_registry(root, **params)


def test_cross_section_detection(tmp_path) -> None:
    _halt_lake(tmp_path, 12)
    registry = _build_halts(tmp_path)
    assert len(registry.halts) == 1
    halt = registry.halts[0]
    assert halt.start == T0 + STEP * 3
    assert halt.end == T0 + STEP * 7
    assert (halt.present_symbols, halt.zero_symbols) == (12, 12)


def test_single_symbol_flat_tail_is_not_a_halt(tmp_path) -> None:
    bars = _stamps(T0, 100)
    _write_3m(tmp_path / "3m" / "FLATUSDT.parquet", bars,
              [3.0] * 100, [10.0] * 3 + [0.0] * 97)
    registry = _build_halts(tmp_path)
    assert registry.halts == ()


def test_zombie_tails_excluded(tmp_path) -> None:
    zombies = [f"ZOMB{i:02d}USDT" for i in range(3)]
    _halt_lake(tmp_path, 9, 3)
    registry = _build_halts(tmp_path, settlements=_zombie_registry(zombies))
    assert registry.halts == ()
    _halt_lake(tmp_path, 10, 3)
    registry = _build_halts(tmp_path, settlements=_zombie_registry(zombies))
    assert len(registry.halts) == 1
    assert (registry.halts[0].present_symbols, registry.halts[0].zero_symbols) == (10, 10)


def test_halt_generator_writes_canonical_bytes(tmp_path) -> None:
    from src.application.ops.settlement_registry import write_venue_halt_registry

    _halt_lake(tmp_path, 12)
    registry = _build_halts(tmp_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert write_venue_halt_registry(registry, first) == 1
    assert write_venue_halt_registry(registry, second) == 1
    assert first.read_bytes() == second.read_bytes()


def test_halt_grid_expands_for_earlier_archives(tmp_path) -> None:
    _halt_lake(tmp_path, 12)
    _write_3m(tmp_path / "3m" / "ZZEARLYUSDT.parquet", _stamps(T0 - STEP * 2, 2), [3.0] * 2, [10.0] * 2)
    registry = _build_halts(tmp_path)
    assert registry.halts[0].start == T0 + STEP * 3
    assert registry.halts[0].present_symbols == 12


@pytest.mark.parametrize(("stamps", "volumes"), [
    ([int(T0.value // 1_000_000)] * 12, [0.0] * 12),
    ([float("nan")], [0.0]),
    ([int(T0.value // 1_000_000) + 1], [0.0]),
    ([int(T0.value // 1_000_000)], [float("nan")]),
    ([int(T0.value // 1_000_000)], [-1.0]),
])
def test_halt_census_rejects_invalid_symbol_bars(tmp_path, stamps, volumes) -> None:
    _write_3m(tmp_path / "3m" / "BADUSDT.parquet", stamps, [3.0] * len(stamps), volumes)
    with pytest.raises(DataIntegrityError):
        _build_halts(tmp_path)


def test_positive_volume_after_last_trade_never_counts_as_live(tmp_path) -> None:
    _halt_lake(tmp_path, 10, 3)
    zombies = [f"ZOMB{i:02d}USDT" for i in range(3)]
    for symbol in zombies:
        _write_3m(tmp_path / "3m" / f"{symbol}.parquet", _stamps(T0, 10), [3.0] * 10, [10.0] * 10)
    registry = _build_halts(tmp_path, settlements=_zombie_registry(zombies))
    assert len(registry.halts) == 1
    assert registry.halts[0].present_symbols == 10


def test_halt_generator_rejects_naive_horizon(tmp_path) -> None:
    from src.application.ops.settlement_registry import build_venue_halt_registry

    _halt_lake(tmp_path, 12)
    with pytest.raises(DataIntegrityError):
        build_venue_halt_registry(
            tmp_path, horizon=pd.Timestamp("2025-01-02"),
            settlements=EMPTY_SETTLEMENT_REGISTRY, verified_at=HORIZON,
        )


def test_halt_generator_rejects_unreadable_lake(tmp_path) -> None:
    from src.application.ops.settlement_registry import build_venue_halt_registry

    with pytest.raises(DataIntegrityError):
        _build_halts(tmp_path / "absent")
    empty = tmp_path / "empty"
    (empty / "3m").mkdir(parents=True)
    with pytest.raises(DataIntegrityError):
        _build_halts(empty)
    with pytest.raises(DataIntegrityError):
        build_venue_halt_registry(
            tmp_path / "file.parquet", horizon=T0 + STEP * 10,
            settlements=EMPTY_SETTLEMENT_REGISTRY, verified_at=HORIZON,
        )


def test_halt_generator_rejects_unreadable_archive(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from src.application.ops.settlement_registry import build_venue_halt_registry

    _halt_lake(tmp_path, 12)
    corrupt = tmp_path / "3m" / "CORRUPTUSDT.parquet"
    corrupt.write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError, match="unreadable"):
        _build_halts(tmp_path)
    corrupt.unlink()
    real_glob = Path.glob

    def _boom(self, pattern):
        if pattern == "*.parquet":
            raise OSError("boom")
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", _boom)
    with pytest.raises(DataIntegrityError, match="unreadable"):
        build_venue_halt_registry(
            tmp_path, horizon=T0 + STEP * 10,
            settlements=EMPTY_SETTLEMENT_REGISTRY, verified_at=HORIZON,
        )


def test_halt_generator_ignores_post_horizon_bars(tmp_path) -> None:
    bars = _stamps(T0 + STEP * 10, 10)
    _write_3m(tmp_path / "3m" / "LATEUSDT.parquet", bars,
              [3.0] * 10, [0.0] * 10)
    registry = _build_halts(tmp_path)
    assert registry.halts == ()


def test_halt_generator_heartbeat_over_many_archives(tmp_path, caplog) -> None:
    import logging

    for i in range(101):
        _write_3m(tmp_path / "3m" / f"S{i:03d}USDT.parquet", _stamps(T0, 10),
                  [3.0] * 10, [10.0] * 10)
    with caplog.at_level(logging.INFO, logger="src.application.ops.settlement_registry"):
        registry = _build_halts(tmp_path)
    assert registry.halts == ()
    assert "stage=build_venue_halts scanned=100" in caplog.text


def test_halt_generator_groups_disjoint_runs(tmp_path) -> None:
    bars = _stamps(T0, 20)
    vols = [10.0] * 3 + [0.0] * 4 + [10.0] * 6 + [0.0] * 4 + [10.0] * 3
    for i in range(12):
        _write_3m(tmp_path / "3m" / f"LIVE{i:02d}USDT.parquet", bars,
                  [3.0] * 20, vols)
    registry = _build_halts(tmp_path, horizon=T0 + STEP * 20)
    assert [h.halt_id for h in registry.halts] == [
        (T0 + STEP * 3).strftime("%Y-%m-%dT%H:%MZ"),
        (T0 + STEP * 13).strftime("%Y-%m-%dT%H:%MZ"),
    ]


def test_halt_writer_creates_parents_and_cleans_failed_replace(tmp_path, monkeypatch) -> None:
    import os

    from src.application.ops.settlement_registry import write_venue_halt_registry

    _halt_lake(tmp_path, 12)
    registry = _build_halts(tmp_path)
    nested = tmp_path / "nested" / "dir" / "halts.jsonl"
    assert write_venue_halt_registry(registry, nested) == 1
    assert nested.exists()

    def _boom(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="boom"):
        write_venue_halt_registry(registry, tmp_path / "other.jsonl")
    assert not (tmp_path / "other.jsonl").exists()


def _patch_notices(monkeypatch, notices) -> None:
    import src.application.ops.settlement_registry as reg_mod

    monkeypatch.setattr(reg_mod, "load_delisting_notices", lambda: tuple(notices))


def _notice(code: str, release_at: pd.Timestamp, symbols: tuple[str, ...]) -> object:
    from src.core.delisting_announcements import DelistingNotice

    return DelistingNotice(
        code=code, title=f"Binance will delist {code}", release_at=release_at,
        kind="delist", symbols=symbols,
    )


_ABRUPT_LAST_TRADE = T0 + STEP * 10


def test_evidence_beats_proxy(monkeypatch) -> None:
    import tempfile
    from pathlib import Path

    release = _ABRUPT_LAST_TRADE - pd.Timedelta(days=5)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _lake(root)
        _patch_notices(monkeypatch, [_notice("C7", release, ("ABRUPTUSDT",))])
        build = _build(root)
    assert not build.unresolved
    record = build.registry.settlements_for("ABRUPTUSDT")[0]
    assert record.announcement_source == "binance_cms"
    assert record.announced_at == release
    assert record.announcement_evidence.startswith("binance-cms:C7|")
    assert build.announcements.binance_cms == 1


def test_curated_override_wins_over_evidence(monkeypatch, tmp_path) -> None:
    import dataclasses

    _patch_notices(monkeypatch, ())
    _lake(tmp_path)
    first = _build(tmp_path)
    victim = first.registry.settlements_for("ABRUPTUSDT")[0]
    curated = dataclasses.replace(
        victim,
        announcement_source="curated",
        announced_at=victim.last_trade_at - pd.Timedelta(days=2),
        announcement_evidence="Binance notice: real announcement",
    )
    existing = assemble_instrument_settlement_registry(
        [r for r in first.registry.settlements if r.symbol != "ABRUPTUSDT"] + [curated], [],
    )
    release = _ABRUPT_LAST_TRADE - pd.Timedelta(days=5)
    _patch_notices(monkeypatch, [_notice("C7", release, ("ABRUPTUSDT",))])
    second = _build(tmp_path, existing=existing)
    kept = second.registry.settlements_for("ABRUPTUSDT")[0]
    assert kept.announcement_source == "curated"
    assert kept.announced_at == curated.announced_at
    assert kept.announcement_evidence == "Binance notice: real announcement"


def test_no_notice_falls_back_to_proxy(monkeypatch, tmp_path) -> None:
    _patch_notices(monkeypatch, ())
    _lake(tmp_path)
    build = _build(tmp_path)
    assert not build.unresolved
    for record in build.registry.settlements:
        assert record.announcement_source == "proxy_lead"
        assert record.announced_at == record.last_trade_at - pd.Timedelta(days=7)
    assert build.announcements.proxy_lead == 2
    assert build.announcements.unmatched == ("ABRUPTUSDT", "FLATUSDT")


def test_post_last_trade_notice_is_reported(monkeypatch, tmp_path) -> None:
    release = _ABRUPT_LAST_TRADE + pd.Timedelta(hours=1)
    _patch_notices(monkeypatch, [_notice("C9", release, ("ABRUPTUSDT",))])
    _lake(tmp_path)
    build = _build(tmp_path)
    record = build.registry.settlements_for("ABRUPTUSDT")[0]
    assert record.announcement_source == "proxy_lead"
    flagged = [symbol for symbol, _reason in build.announcements.after_last_trade]
    assert "ABRUPTUSDT" in flagged
    assert "ABRUPTUSDT" in build.announcements.unmatched


def test_announcement_build_is_idempotent(monkeypatch, tmp_path) -> None:
    from src.core.instrument_settlements import settlement_registry_jsonl

    release = _ABRUPT_LAST_TRADE - pd.Timedelta(days=5)
    _patch_notices(monkeypatch, [_notice("C7", release, ("ABRUPTUSDT",))])
    _lake(tmp_path)
    first = _build(tmp_path)
    second = _build(tmp_path, existing=first.registry)
    assert second.changed == ()
    assert settlement_registry_jsonl(second.registry) == settlement_registry_jsonl(first.registry)


def test_price_fields_untouched_by_announcements(monkeypatch, tmp_path) -> None:
    _patch_notices(monkeypatch, ())
    _lake(tmp_path)
    before = _build(tmp_path)
    release = _ABRUPT_LAST_TRADE - pd.Timedelta(days=5)
    _patch_notices(monkeypatch, [_notice("C7", release, ("ABRUPTUSDT",))])
    after = _build(tmp_path, existing=before.registry)
    event_ms = int(_ABRUPT_LAST_TRADE.value // 1_000_000)
    assert after.changed == (f"ABRUPTUSDT:{event_ms}",)
    for symbol in ("ABRUPTUSDT", "FLATUSDT"):
        old = before.registry.settlements_for(symbol)[0]
        new = after.registry.settlements_for(symbol)[0]
        assert new.last_trade_at == old.last_trade_at
        assert new.delivery_at == old.delivery_at
        assert new.event_id == old.event_id
        assert new.settlement_price == old.settlement_price
        assert new.price_source == old.price_source
        assert new.price_evidence == old.price_evidence
        assert new.evidence_digest == old.evidence_digest
        assert new.fee_bps == old.fee_bps


def test_build_preserves_original_release_milliseconds(monkeypatch, tmp_path) -> None:
    from src.core.delisting_announcements import parse_delisting_evidence
    from src.core.instrument_settlements import settlement_registry_jsonl

    release_ms = int((_ABRUPT_LAST_TRADE - pd.Timedelta(days=5)).value // 1_000_000) + 501
    raw = json.dumps({
        "code": "C1", "title": "Delist ABRUPTUSDT", "release_ms": release_ms,
        "kind": "delist", "symbols": ["ABRUPTUSDT"], "collected_at": HORIZON.isoformat(),
    }).encode()
    notices = parse_delisting_evidence(raw, source="test")
    _patch_notices(monkeypatch, notices)
    _lake(tmp_path)
    first = _build(tmp_path)
    record = first.registry.settlements_for("ABRUPTUSDT")[0]
    assert record.announcement_evidence == f"binance-cms:C1|{release_ms}|Delist ABRUPTUSDT"
    assert record.announced_at == pd.Timestamp((release_ms + 999) // 1000, unit="s", tz="UTC")
    second = _build(tmp_path, existing=first.registry, verified_at=HORIZON + pd.Timedelta(days=1))
    assert second.changed == ()
    assert settlement_registry_jsonl(first.registry) == settlement_registry_jsonl(second.registry)
