"""Invariant scenarios for the source-gap verification CLI (spec 3)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.services.source_gap_audit import (
    SourceGapAuditReport,
    audit_source_gap_registry,
    measure_source_gaps,
    write_audited_registry,
)
from src.mhs.source_gaps import SourceGapInterval, load_source_gap_registry

_VERIFIED = "2022-02-01T00:00:00Z"


def _ms(idx: pd.DatetimeIndex) -> list[int]:
    return [int(t.value // 10**6) for t in idx]


def _write_ohlcv(root: Path, symbol: str, timeframe: str, idx: pd.DatetimeIndex) -> Path:
    directory = root / "ohlcv" / timeframe
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "timestamp": _ms(idx),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    path = directory / f"{symbol}.parquet"
    frame.to_parquet(path, index=False)
    return path


def _write_funding(root: Path, symbol: str, idx: pd.DatetimeIndex) -> Path:
    directory = root / "funding"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "timestamp": _ms(idx),
        "funding_rate": 0.0001,
        "datetime": idx,
    })
    path = directory / f"{symbol}.parquet"
    frame.to_parquet(path, index=False)
    return path


def _row(
    symbol: str = "AAAUSDT",
    plane: str = "ohlcv_3m",
    start: str | None = "2022-02-26T00:00:00Z",
    end: str | None = "2022-03-01T00:00:00Z",
    reason: str = "SOURCE_ABSENT",
    evidence: str = "Vision monthly klines and REST re-query both empty",
    verified_at: str = _VERIFIED,
    resolved_at: str | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol, "plane": plane, "start": start, "end": end,
        "reason": reason, "evidence": evidence,
        "verified_at": verified_at, "resolved_at": resolved_at,
    }


def _write_registry(path: Path, rows: list[Any]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _grid(start: str, end: str, freq: str = "3min") -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, tz="UTC")[:-1]


# --- Spec scenario 1: 내부 공백 측정 정확도 ------------------------------------


def test_measure_interior_gap_matches_removed_half_open_bounds(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    grid = _grid("2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z")
    removed_start = pd.Timestamp("2022-01-01T06:00:00Z")
    resumed_start = pd.Timestamp("2022-01-01T06:30:00Z")
    kept = grid[(grid < removed_start) | (grid >= resumed_start)]
    _write_ohlcv(root, "AAAUSDT", "3m", kept)
    gaps = measure_source_gaps(
        ["AAAUSDT"], plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-02T00:00:00Z"),
        data_root=root,
    )
    assert len(gaps) == 1
    assert gaps[0].start.isoformat() == removed_start.isoformat()
    assert gaps[0].end is not None
    assert gaps[0].end.isoformat() == resumed_start.isoformat()
    assert gaps[0].reason == "SOURCE_ABSENT"
    assert gaps[0].evidence.strip() != ""


# --- Spec scenario 2: 양끝 경계는 개방 구간 ------------------------------------


def test_measure_window_edges_stay_open_and_never_delisted(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = pd.Timestamp("2022-01-06T00:00:00Z")
    # 늦게 시작하는 심볼: 앞쪽이 리스팅 경계다.
    _write_ohlcv(root, "LATEUSDT", "3m", _grid("2022-01-05T00:00:00Z", "2022-01-06T00:00:00Z"))
    # 일찍 끝나는 심볼: 뒤쪽이 개방 구간이다.
    _write_ohlcv(root, "EARLYUSDT", "3m", _grid("2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"))
    # 창 전체가 자료보다 앞서는 심볼.
    _write_ohlcv(root, "FUTUREUSDT", "3m", _grid("2022-02-01T00:00:00Z", "2022-02-02T00:00:00Z"))
    # 창 전체가 자료보다 늦은 심볼.
    _write_ohlcv(root, "PASTUSDT", "3m", _grid("2021-12-01T00:00:00Z", "2021-12-02T00:00:00Z"))
    gaps = measure_source_gaps(
        ["LATEUSDT", "EARLYUSDT", "FUTUREUSDT", "PASTUSDT"],
        plane="ohlcv_3m", start=start, end=end, data_root=root,
    )
    assert gaps, "edge symbols must yield evidence"
    assert all(g.reason == "SOURCE_ABSENT" for g in gaps)
    assert all(g.reason != "DELISTED" for g in gaps)
    by_symbol = {g.symbol: g for g in gaps if g.end is not None}
    assert by_symbol["LATEUSDT"].start == start.to_pydatetime()
    trailing = [g for g in gaps if g.symbol == "EARLYUSDT" and g.end is None]
    assert len(trailing) == 1
    assert trailing[0].start > start.to_pydatetime()
    past = [g for g in gaps if g.symbol == "PASTUSDT"]
    assert len(past) == 1
    assert past[0].end is None


# --- Spec scenario 3: 완전 복구는 resolved -------------------------------------


def test_audit_fully_recovered_gap_is_resolved(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    root = tmp_path / "lake"
    _write_ohlcv(root, "AAAUSDT", "3m", _grid("2022-02-20T00:00:00Z", "2022-03-05T00:00:00Z"))
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-02-20T00:00:00Z"), end=pd.Timestamp("2022-03-05T00:00:00Z"),
        symbols=["AAAUSDT"], registry_path=registry, data_root=root,
    )
    assert len(report.resolved) == 1
    assert report.resolved[0].symbol == "AAAUSDT"
    assert report.narrowed == ()
    assert report.unchanged == ()


# --- Spec scenario 4: 부분 복구는 narrowed -------------------------------------


def test_audit_partially_recovered_gap_is_narrowed(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    root = tmp_path / "lake"
    recovered_until = pd.Timestamp("2022-02-27T12:00:00Z")
    full = _grid("2022-02-20T00:00:00Z", "2022-03-05T00:00:00Z")
    kept = full[
        (full < pd.Timestamp("2022-02-26T00:00:00Z"))
        | ((full >= pd.Timestamp("2022-02-26T00:00:00Z")) & (full < recovered_until))
        | (full >= pd.Timestamp("2022-03-01T00:00:00Z"))
    ]
    _write_ohlcv(root, "AAAUSDT", "3m", kept)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-02-20T00:00:00Z"), end=pd.Timestamp("2022-03-05T00:00:00Z"),
        symbols=["AAAUSDT"], registry_path=registry, data_root=root,
    )
    assert len(report.narrowed) == 1
    assert report.narrowed[0].start.isoformat() == recovered_until.isoformat()
    assert report.narrowed[0].end is not None
    assert report.narrowed[0].end.isoformat() == pd.Timestamp("2022-03-01T00:00:00Z").isoformat()
    assert report.resolved == (), "원 레코드는 resolved로 분류되지 않는다"


def test_audit_middle_recovery_yields_two_narrowed_pieces(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    root = tmp_path / "lake"
    full = _grid("2022-02-20T00:00:00Z", "2022-03-05T00:00:00Z")
    # 커밋 구간의 한가운데만 복구되면 잔여 양쪽이 각각 narrowed가 된다.
    kept = full[
        (full < pd.Timestamp("2022-02-26T00:00:00Z"))
        | ((full >= pd.Timestamp("2022-02-27T00:00:00Z")) & (full < pd.Timestamp("2022-02-28T00:00:00Z")))
        | (full >= pd.Timestamp("2022-03-01T00:00:00Z"))
    ]
    _write_ohlcv(root, "AAAUSDT", "3m", kept)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-02-20T00:00:00Z"), end=pd.Timestamp("2022-03-05T00:00:00Z"),
        symbols=["AAAUSDT"], registry_path=registry, data_root=root,
    )
    assert len(report.narrowed) == 2
    assert report.resolved == ()


# --- Spec scenario 5: 개방 구간 좁히기 ------------------------------------------


def test_audit_open_legacy_record_narrows_to_measured_day(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "reg.jsonl",
        [_row(symbol="CCCUUSDT", start="2020-01-01T00:00:00Z", end=None, evidence="legacy open")],
    )
    root = tmp_path / "lake"
    grid = _grid("2022-06-01T00:00:00Z", "2022-07-01T00:00:00Z")
    kept = grid[
        (grid < pd.Timestamp("2022-06-10T00:00:00Z"))
        | (grid >= pd.Timestamp("2022-06-11T00:00:00Z"))
    ]
    _write_ohlcv(root, "CCCUUSDT", "3m", kept)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-06-01T00:00:00Z"), end=pd.Timestamp("2022-07-01T00:00:00Z"),
        symbols=["CCCUUSDT"], registry_path=registry, data_root=root,
    )
    assert len(report.narrowed) == 1
    assert report.narrowed[0].start.isoformat() == "2022-06-10T00:00:00+00:00"
    assert report.narrowed[0].end is not None
    assert report.narrowed[0].end.isoformat() == "2022-06-11T00:00:00+00:00"


def test_audit_open_record_matching_trailing_edge_is_unchanged(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "reg.jsonl",
        [_row(symbol="CCCUUSDT", start="2020-01-01T00:00:00Z", end=None, evidence="legacy open")],
    )
    root = tmp_path / "lake"
    # 측정창보다 먼저 끊긴 자료: trailing 개방 구간이 레거시 개방 구간과 겹친다.
    _write_ohlcv(root, "CCCUUSDT", "3m", _grid("2022-06-01T00:00:00Z", "2022-06-05T00:00:00Z"))
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-06-01T00:00:00Z"), end=pd.Timestamp("2022-07-01T00:00:00Z"),
        symbols=["CCCUUSDT"], registry_path=registry, data_root=root,
    )
    assert report.resolved == ()
    assert len(report.narrowed) == 1
    assert report.narrowed[0].start.isoformat() == "2022-06-05T00:00:00+00:00"
    assert report.narrowed[0].end is None


# --- Spec scenario 6: 미등록 결손은 discovered ----------------------------------


def test_audit_unregistered_gap_is_discovered(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [])
    root = tmp_path / "lake"
    grid = _grid("2022-01-01T00:00:00Z", "2022-01-03T00:00:00Z")
    kept = grid[
        ~((grid >= pd.Timestamp("2022-01-02T00:00:00Z")) & (grid < pd.Timestamp("2022-01-02T01:00:00Z")))
    ]
    _write_ohlcv(root, "DDDUSDT", "3m", kept)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-03T00:00:00Z"),
        symbols=["DDDUSDT"], registry_path=registry, data_root=root,
    )
    assert len(report.discovered) == 1
    assert report.discovered[0].symbol == "DDDUSDT"
    assert report.discovered[0].start.isoformat() == "2022-01-02T00:00:00+00:00"


def test_audit_confirmed_gap_is_unchanged(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    root = tmp_path / "lake"
    grid = _grid("2022-02-20T00:00:00Z", "2022-03-05T00:00:00Z")
    kept = grid[
        (grid < pd.Timestamp("2022-02-26T00:00:00Z")) | (grid >= pd.Timestamp("2022-03-01T00:00:00Z"))
    ]
    _write_ohlcv(root, "AAAUSDT", "3m", kept)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-02-20T00:00:00Z"), end=pd.Timestamp("2022-03-05T00:00:00Z"),
        symbols=["AAAUSDT"], registry_path=registry, data_root=root,
    )
    assert len(report.unchanged) == 1
    assert report.resolved == ()
    assert report.narrowed == ()


# --- Spec scenario 7: 쓰기는 이력 보존 ------------------------------------------


def test_write_preserves_history_and_passes_loader_invariants(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "reg.jsonl",
        [
            _row(symbol="AAAUSDT", start="2022-01-01T00:00:00Z", end="2022-01-03T00:00:00Z", evidence="e1"),
            _row(symbol="BBBUSDT", start="2022-01-01T00:00:00Z", end="2022-01-10T00:00:00Z", evidence="e2"),
            _row(
                symbol="CCCUUSDT", start="2022-01-01T00:00:00Z", end="2022-01-02T00:00:00Z",
                evidence="e3", resolved_at="2022-03-01T00:00:00Z",
            ),
            _row(symbol="DDDUSDT", start="2022-01-01T00:00:00Z", end="2022-01-02T00:00:00Z", evidence="e4"),
            _row(symbol="EEEUSDT", start="2021-01-01T00:00:00Z", end="2021-01-02T00:00:00Z", evidence="e5"),
        ],
    )
    existing = load_source_gap_registry(registry)
    narrowed_replacement = SourceGapInterval(
        symbol="BBBUSDT", plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-05T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2022-01-10T00:00:00Z").to_pydatetime(),
        reason="SOURCE_ABSENT", evidence="remeasured residual",
        verified_at=pd.Timestamp("2022-03-01T00:00:00Z").to_pydatetime(), resolved_at=None,
    )
    unchanged_record = next(iv for iv in existing if iv.symbol == "DDDUSDT")
    report = SourceGapAuditReport(
        resolved=(next(iv for iv in existing if iv.symbol == "AAAUSDT"),),
        narrowed=(narrowed_replacement,),
        unchanged=(unchanged_record,),
        discovered=(),
    )
    verified_at = pd.Timestamp("2022-04-01T00:00:00Z")
    count = write_audited_registry(report, registry_path=registry, verified_at=verified_at)
    reloaded = load_source_gap_registry(registry)
    # 삭제 없이 보존된다: 기존 5건 + narrowed 신규 1건.
    assert count == 6
    assert len(reloaded) == 6
    by_symbol = {iv.symbol: iv for iv in reloaded if iv.end is not None and iv.start.isoformat() == "2022-01-01T00:00:00+00:00"}
    assert by_symbol["AAAUSDT"].resolved_at == verified_at.to_pydatetime()
    assert by_symbol["AAAUSDT"].verified_at == verified_at.to_pydatetime()
    assert by_symbol["BBBUSDT"].resolved_at == verified_at.to_pydatetime()
    assert by_symbol["DDDUSDT"].resolved_at is None
    assert by_symbol["DDDUSDT"].verified_at == verified_at.to_pydatetime()
    assert any(iv.symbol == "BBBUSDT" and iv.start.isoformat() == "2022-01-05T00:00:00+00:00" for iv in reloaded)
    # 이미 해소된 레코드는 그대로 유지된다.
    ccc = [iv for iv in reloaded if iv.symbol == "CCCUUSDT"]
    assert len(ccc) == 1
    assert ccc[0].resolved_at is not None
    # 감사 범위를 벗어난 활성 레코드는 손대지 않는다.
    eee = [iv for iv in reloaded if iv.symbol == "EEEUSDT"]
    assert len(eee) == 1
    assert eee[0].resolved_at is None
    assert eee[0].verified_at.isoformat() == "2022-02-01T00:00:00+00:00"


def test_write_rejects_violating_result_and_preserves_original(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "reg.jsonl",
        [_row(symbol="AAAUSDT", start="2022-01-01T00:00:00Z", end="2022-01-10T00:00:00Z", evidence="e1")],
    )
    before = registry.read_bytes()
    clashing = SourceGapInterval(
        symbol="AAAUSDT", plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-05T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2022-01-06T00:00:00Z").to_pydatetime(),
        reason="SOURCE_ABSENT", evidence="clash",
        verified_at=pd.Timestamp("2022-03-01T00:00:00Z").to_pydatetime(), resolved_at=None,
    )
    # narrowed와 겹치면서 resolved에도 unchanged에도 없는 활성 레코드가 남으면
    # 로더의 겹침 불변식을 위반하므로 원본을 보존하고 예외를 던진다.
    report = SourceGapAuditReport(
        resolved=(), narrowed=(clashing,), unchanged=(), discovered=(clashing,),
    )
    with pytest.raises(DataIntegrityError):
        write_audited_registry(
            report, registry_path=registry, verified_at=pd.Timestamp("2022-04-01T00:00:00Z"),
        )
    assert registry.read_bytes() == before


def test_write_rejects_bad_inputs(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    good_new = SourceGapInterval(
        symbol="AAAUSDT", plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-05T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2022-01-06T00:00:00Z").to_pydatetime(),
        reason="SOURCE_ABSENT", evidence="ok",
        verified_at=pd.Timestamp("2022-03-01T00:00:00Z").to_pydatetime(), resolved_at=None,
    )
    with pytest.raises(DataIntegrityError):
        write_audited_registry(
            SourceGapAuditReport(discovered=(good_new,)),
            registry_path=registry, verified_at=pd.Timestamp("2022-04-01"),
        )
    with pytest.raises(DataIntegrityError):
        write_audited_registry(
            SourceGapAuditReport(discovered=(good_new,)),
            registry_path=registry, verified_at=pd.Timestamp("2022-04-01T09:00:00+09:00"),
        )
    blank = SourceGapInterval(
        symbol="AAAUSDT", plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-05T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2022-01-06T00:00:00Z").to_pydatetime(),
        reason="SOURCE_ABSENT", evidence="   ",
        verified_at=pd.Timestamp("2022-03-01T00:00:00Z").to_pydatetime(), resolved_at=None,
    )
    with pytest.raises(DataIntegrityError):
        write_audited_registry(
            SourceGapAuditReport(discovered=(blank,)),
            registry_path=registry, verified_at=pd.Timestamp("2022-04-01T00:00:00Z"),
        )
    stale = SourceGapInterval(
        symbol="AAAUSDT", plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-05T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2022-01-06T00:00:00Z").to_pydatetime(),
        reason="SOURCE_ABSENT", evidence="ok",
        verified_at=pd.Timestamp("2022-03-01T00:00:00Z").to_pydatetime(),
        resolved_at=pd.Timestamp("2022-03-02T00:00:00Z").to_pydatetime(),
    )
    with pytest.raises(DataIntegrityError):
        write_audited_registry(
            SourceGapAuditReport(discovered=(stale,)),
            registry_path=registry, verified_at=pd.Timestamp("2022-04-01T00:00:00Z"),
        )
    with pytest.raises(DataIntegrityError):
        write_audited_registry(
            SourceGapAuditReport(), registry_path=tmp_path / "missing.jsonl",
            verified_at=pd.Timestamp("2022-04-01T00:00:00Z"),
        )


def test_write_commit_validation_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.market_data.services.source_gap_audit as audit_mod
    import src.mhs.source_gaps as gaps_mod

    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    before = registry.read_bytes()
    original_load = gaps_mod.load_source_gap_registry

    def _flaky_load(path: Path | None = None) -> Any:
        if path is not None and Path(path).name.startswith(".source_gaps_"):
            raise DataIntegrityError("simulated commit validation failure")
        return original_load(path)

    monkeypatch.setattr(audit_mod, "load_source_gap_registry", _flaky_load)
    with pytest.raises(DataIntegrityError, match="simulated"):
        write_audited_registry(
            SourceGapAuditReport(), registry_path=registry,
            verified_at=pd.Timestamp("2022-04-01T00:00:00Z"),
        )
    assert registry.read_bytes() == before
    assert list(tmp_path.glob(".source_gaps_*")) == []


def test_write_unwritable_target_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile as tempfile_mod

    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    before = registry.read_bytes()

    def _disk_full(**kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(tempfile_mod, "mkstemp", _disk_full)
    with pytest.raises(DataIntegrityError, match="unwritable"):
        write_audited_registry(
            SourceGapAuditReport(), registry_path=registry,
            verified_at=pd.Timestamp("2022-04-01T00:00:00Z"),
        )
    assert registry.read_bytes() == before


def test_write_replace_failure_cleans_up_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path as _Path

    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    before = registry.read_bytes()

    def _denied_replace(self: Path, target: Path) -> Any:
        raise OSError("replace denied")

    monkeypatch.setattr(_Path, "replace", _denied_replace)
    with pytest.raises(DataIntegrityError, match="unwritable"):
        write_audited_registry(
            SourceGapAuditReport(), registry_path=registry,
            verified_at=pd.Timestamp("2022-04-01T00:00:00Z"),
        )
    assert registry.read_bytes() == before
    assert list(tmp_path.glob(".source_gaps_*")) == []


# --- Spec scenario 8: 읽기 전용 기본값 ------------------------------------------


def test_cli_verify_source_gaps_defaults_to_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    from src.cli.commands.data import add_data_commands
    import src.market_data.services.source_gap_audit as audit_mod

    parser = argparse.ArgumentParser(prog="cli")
    sub = parser.add_subparsers(dest="command", required=True)
    data_parser = sub.add_parser("data")
    add_data_commands(data_parser)
    args = parser.parse_args([
        "data", "verify-source-gaps",
        "--start", "2022-02-20T00:00:00Z", "--end", "2022-03-05T00:00:00Z",
    ])
    assert args.plane == "ohlcv_3m"
    assert args.write is False
    assert args.handler.__name__ == "_verify_source_gaps"

    stamp = datetime(2022, 6, 10, tzinfo=UTC)
    noisy = SourceGapAuditReport(
        resolved=(
            SourceGapInterval(
                symbol="AAAUSDT", plane="ohlcv_3m", start=stamp, end=None,
                reason="SOURCE_ABSENT", evidence="e",
                verified_at=stamp, resolved_at=None,
            ),
        ),
        narrowed=(), unchanged=(), discovered=(),
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(audit_mod, "audit_source_gap_registry", lambda **kwargs: noisy)
    monkeypatch.setattr(
        audit_mod, "write_audited_registry",
        lambda report, **kwargs: calls.setdefault("write", (report, kwargs)) or 0,
    )
    args.handler(args)
    assert "write" not in calls, "--write 없이 실행하면 파일을 변경하지 않는다"

    args.write = True
    args.handler(args)
    assert calls["write"][0] is noisy


def test_cli_verify_source_gaps_read_only_keeps_registry_bytes() -> None:
    from argparse import Namespace

    from src.cli.commands.data import _verify_source_gaps
    from src.mhs.source_gaps import _default_registry_path

    target = _default_registry_path()
    before = target.read_bytes()
    args = Namespace(
        plane="ohlcv_3m", start="2022-02-20T00:00:00Z", end="2022-03-05T00:00:00Z",
        symbol=["MANAUSDT"], write=False,
    )
    _verify_source_gaps(args)
    assert target.read_bytes() == before


# --- 측정 입력 검증 --------------------------------------------------------------


def test_measure_rejects_non_timestamp_and_non_utc_window(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    good_end = pd.Timestamp("2022-01-02T00:00:00Z")
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["AAAUSDT"], plane="ohlcv_3m",
            start="2022-01-01",  # type: ignore[arg-type]
            end=good_end, data_root=root,
        )
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["AAAUSDT"], plane="ohlcv_3m",
            start=pd.Timestamp("2022-01-01T09:00:00+09:00"), end=good_end, data_root=root,
        )


def test_measure_rejects_bad_inputs(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    _write_ohlcv(root, "AAAUSDT", "3m", _grid("2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"))
    good_start = pd.Timestamp("2022-01-01T00:00:00Z")
    good_end = pd.Timestamp("2022-01-02T00:00:00Z")
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(["AAAUSDT"], plane="ohlcv_3m", start=good_end, end=good_start, data_root=root)
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["AAAUSDT"], plane="ohlcv_3m",
            start=pd.Timestamp("2022-01-01"), end=good_end, data_root=root,
        )
    with pytest.raises(DataIntegrityError):
        measure_source_gaps([], plane="ohlcv_3m", start=good_start, end=good_end, data_root=root)
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["AAAUSDT", "AAAUSDT"], plane="ohlcv_3m",
            start=good_start, end=good_end, data_root=root,
        )
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["aaausdt"], plane="ohlcv_3m", start=good_start, end=good_end, data_root=root,
        )
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["AAAUSDT"], plane="ohlcv_1s",  # type: ignore[arg-type]
            start=good_start, end=good_end, data_root=root,
        )


def test_measure_missing_or_empty_symbol_yields_single_open_record(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    (root / "ohlcv" / "3m").mkdir(parents=True)
    gaps = measure_source_gaps(
        ["GHOSTUSDT"], plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-02T00:00:00Z"),
        data_root=root,
    )
    assert len(gaps) == 1
    assert gaps[0].end is None
    assert gaps[0].reason == "SOURCE_ABSENT"
    empty_path = root / "ohlcv" / "3m" / "EMPTYUSDT.parquet"
    pd.DataFrame({"timestamp": pd.Series(dtype="int64")}).to_parquet(empty_path, index=False)
    gaps = measure_source_gaps(
        ["EMPTYUSDT"], plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-02T00:00:00Z"),
        data_root=root,
    )
    assert len(gaps) == 1
    assert gaps[0].end is None


def test_measure_rejects_unreadable_parquet(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    directory = root / "ohlcv" / "3m"
    directory.mkdir(parents=True)
    (directory / "BROKENUSDT.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["BROKENUSDT"], plane="ohlcv_3m",
            start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-02T00:00:00Z"),
            data_root=root,
        )


def test_measure_parquet_time_column_variants(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    directory = root / "ohlcv" / "3m"
    directory.mkdir(parents=True)
    idx = _grid("2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z")
    # datetime 컬럼만 있는 파일도 읽힌다.
    pd.DataFrame({"datetime": idx, "funding_rate": 0.0}).to_parquet(
        directory / "DTUSDT.parquet", index=False,
    )
    gaps = measure_source_gaps(
        ["DTUSDT"], plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-02T00:00:00Z"),
        data_root=root,
    )
    assert gaps == ()
    # 전부 NaN인 timestamp는 빈 파일과 동일하게 취급된다.
    pd.DataFrame({"timestamp": [float("nan")] * 3}).to_parquet(
        directory / "NANUSDT.parquet", index=False,
    )
    gaps = measure_source_gaps(
        ["NANUSDT"], plane="ohlcv_3m",
        start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-02T00:00:00Z"),
        data_root=root,
    )
    assert len(gaps) == 1
    assert gaps[0].end is None
    # 시간 컬럼이 없으면 fail-closed다.
    pd.DataFrame({"open": [1.0, 2.0]}).to_parquet(directory / "NOTIMEUSDT.parquet", index=False)
    with pytest.raises(DataIntegrityError):
        measure_source_gaps(
            ["NOTIMEUSDT"], plane="ohlcv_3m",
            start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-02T00:00:00Z"),
            data_root=root,
        )


def test_measure_funding_plane_uses_native_eight_hour_step(tmp_path: Path) -> None:
    root = tmp_path / "lake"
    grid = pd.date_range("2022-01-01T00:00:00Z", "2022-01-05T00:00:00Z", freq="8h", tz="UTC")
    removed = pd.Timestamp("2022-01-02T00:00:00Z")
    resumed = pd.Timestamp("2022-01-02T08:00:00Z")
    kept = grid[(grid != removed)]
    _write_funding(root, "AAAUSDT", kept)
    gaps = measure_source_gaps(
        ["AAAUSDT"], plane="funding",
        start=pd.Timestamp("2022-01-01T00:00:00Z"), end=pd.Timestamp("2022-01-05T00:00:00Z"),
        data_root=root,
    )
    assert len(gaps) == 1
    assert gaps[0].start.isoformat() == removed.isoformat()
    assert gaps[0].end is not None
    assert gaps[0].end.isoformat() == resumed.isoformat()


# --- 감사 범위 검증 ----------------------------------------------------------------


def test_audit_tolerates_unlistable_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path as _Path

    registry = _write_registry(tmp_path / "reg.jsonl", [_row(symbol="AAAUSDT")])

    def _denied(self: Path, pattern: str) -> Any:
        raise OSError("denied")

    monkeypatch.setattr(_Path, "glob", _denied)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-02-20T00:00:00Z"), end=pd.Timestamp("2022-03-05T00:00:00Z"),
        symbols=None, registry_path=registry, data_root=tmp_path / "lake",
    )
    # 레이크 목록이 막혀도 레지스트리 이름 기준으로는 감사된다.
    assert len(report.unchanged) == 1


def test_audit_closed_superset_commit_confirmed_by_window_slice(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "reg.jsonl",
        [_row(symbol="AAAUSDT", start="2022-01-01T00:00:00Z", end="2022-12-31T00:00:00Z", evidence="year")],
    )
    root = tmp_path / "lake"
    before = _grid("2022-05-01T00:00:00Z", "2022-06-01T00:00:00Z")
    after = _grid("2022-07-01T00:00:00Z", "2022-08-01T00:00:00Z")
    full = before.append(after)
    _write_ohlcv(root, "AAAUSDT", "3m", full)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-06-01T00:00:00Z"), end=pd.Timestamp("2022-07-01T00:00:00Z"),
        symbols=["AAAUSDT"], registry_path=registry, data_root=root,
    )
    # 커밋이 창을 포함해도 감사 구간이 통째로 비어 있으면 재확인된다.
    assert len(report.unchanged) == 1
    assert report.resolved == ()
    assert report.narrowed == ()


def test_audit_none_symbols_reconciles_registry_plus_lake(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [_row(symbol="AAAUSDT")])
    root = tmp_path / "lake"
    _write_ohlcv(root, "AAAUSDT", "3m", _grid("2022-02-20T00:00:00Z", "2022-03-05T00:00:00Z"))
    grid = _grid("2022-02-20T00:00:00Z", "2022-03-05T00:00:00Z")
    kept = grid[
        ~((grid >= pd.Timestamp("2022-02-27T00:00:00Z")) & (grid < pd.Timestamp("2022-02-28T00:00:00Z")))
    ]
    _write_ohlcv(root, "LAKEONLYUSDT", "3m", kept)
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-02-20T00:00:00Z"), end=pd.Timestamp("2022-03-05T00:00:00Z"),
        symbols=None, registry_path=registry, data_root=root,
    )
    assert any(iv.symbol == "AAAUSDT" for iv in report.resolved)
    assert any(iv.symbol == "LAKEONLYUSDT" for iv in report.discovered)


def test_audit_ignores_commits_outside_window_and_empty_scope(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "reg.jsonl", [_row()])
    root = tmp_path / "lake"
    _write_ohlcv(root, "AAAUSDT", "3m", _grid("2022-02-20T00:00:00Z", "2022-03-05T00:00:00Z"))
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-05-01T00:00:00Z"), end=pd.Timestamp("2022-06-01T00:00:00Z"),
        symbols=["AAAUSDT"], registry_path=registry, data_root=root,
    )
    # 윈도우 밖 커밋은 건드리지 않는다. 측정된 trailing 개방 구간은
    # 레지스트리 커버가 없으므로 discovered로 보고된다.
    assert report.resolved == ()
    assert report.narrowed == ()
    assert report.unchanged == ()
    assert len(report.discovered) == 1
    assert report.discovered[0].end is None
    empty_registry = _write_registry(tmp_path / "empty.jsonl", [])
    empty_root = tmp_path / "empty-lake"
    report = audit_source_gap_registry(
        plane="ohlcv_3m",
        start=pd.Timestamp("2022-05-01T00:00:00Z"), end=pd.Timestamp("2022-06-01T00:00:00Z"),
        symbols=None, registry_path=empty_registry, data_root=empty_root,
    )
    assert report == SourceGapAuditReport()


def test_clip_and_merge_helpers_handle_open_spans() -> None:
    from src.market_data.services.source_gap_audit import (
        _clip_to_commit,
        _merge_clipped,
    )

    t0 = datetime(2022, 1, 1, tzinfo=UTC)
    t1 = datetime(2022, 1, 2, tzinfo=UTC)
    t2 = datetime(2022, 1, 3, tzinfo=UTC)
    t3 = datetime(2022, 1, 4, tzinfo=UTC)

    def _iv(symbol: str, s: datetime, e: datetime | None) -> SourceGapInterval:
        return SourceGapInterval(
            symbol=symbol, plane="ohlcv_3m", start=s, end=e,
            reason="SOURCE_ABSENT", evidence="e",
            verified_at=t3, resolved_at=None,
        )

    assert _clip_to_commit(_iv("A", t2, t3), _iv("A", t0, t1), t0, t3) is None
    assert _merge_clipped([]) == []
    # 개방 구간이 뒤따르는 조각을 흡수한다.
    assert _merge_clipped([(t0, None), (t1, t2)]) == [(t0, None)]
    # 맞닿은 구간은 병합된다.
    assert _merge_clipped([(t0, t1), (t1, t2)]) == [(t0, t2)]
    assert _merge_clipped([(t2, None), (t0, t1)]) == [(t0, t1), (t2, None)]
