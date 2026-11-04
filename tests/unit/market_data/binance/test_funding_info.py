from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance import funding_info as fi

H = 3_600_000
NOW = pd.Timestamp("2026-11-03T23:00:30", tz="UTC")


def _row(symbol: str, hours: object) -> dict[str, object]:
    return {"symbol": symbol, "fundingIntervalHours": hours}


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_parse_valid() -> None:
    snap = fi.parse_funding_info([_row("QUSDT", 4), _row("BTCUSDT", 8), _row("XUSDT", 1)], captured_at=NOW)
    assert snap.interval_for("QUSDT") == 4 * H
    assert snap.interval_for("BTCUSDT") == 8 * H
    assert snap.interval_for("XUSDT") == H
    assert snap.interval_for("MISSINGUSDT") is None
    assert snap.captured_at == NOW


def test_parse_rejects_bad_rows() -> None:
    payload = [_row("AUSDT", 3), _row("BUSDT", True), {"fundingIntervalHours": 4}, _row("", 4), "junk", _row("OKUSDT", 4)]
    snap = fi.parse_funding_info(payload, captured_at=NOW)
    assert dict(snap.interval_ms) == {"OKUSDT": 4 * H}
    with pytest.raises(DataIntegrityError):
        fi.parse_funding_info([_row("AUSDT", 3), _row("BUSDT", False)], captured_at=NOW)


@pytest.mark.parametrize("payload", [[], {}, "x", None])
def test_parse_rejects_empty_payload(payload: object) -> None:
    with pytest.raises(DataIntegrityError):
        fi.parse_funding_info(payload, captured_at=NOW)


def test_parse_rejects_naive_clock() -> None:
    with pytest.raises(DataIntegrityError):
        fi.parse_funding_info([_row("AUSDT", 4)], captured_at=pd.Timestamp("2026-11-03T23:00:00"))


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError("u", 429, "limit", None, io.BytesIO(b"")),  # type: ignore[arg-type]
        urllib.error.URLError("down"),
        TimeoutError("slow"),
    ],
)
def test_fetch_maps_transport_failure_to_runtime_error(failure: Exception) -> None:
    calls: list[str] = []

    def opener(url: str, timeout: float) -> _Response:
        calls.append(url)
        raise failure

    with pytest.raises(RuntimeError):
        fi.fetch_funding_info(opener=opener)
    assert calls == [fi.FUNDING_INFO_URL]


def test_fetch_bad_json_is_runtime_error_and_valid_body_parses() -> None:
    with pytest.raises(RuntimeError):
        fi.fetch_funding_info(opener=lambda url, timeout: _Response(b"<html>"))
    body = json.dumps([_row("QUSDT", 4)]).encode()
    snap = fi.fetch_funding_info(opener=lambda url, timeout: _Response(body))
    assert snap.interval_for("QUSDT") == 4 * H
    assert snap.captured_at.tzinfo is not None


def test_snapshot_roundtrip_and_newest_wins(tmp_path: Path) -> None:
    root = tmp_path / "fi"
    older = fi.parse_funding_info([_row("QUSDT", 1)], captured_at=NOW - pd.Timedelta(hours=3))
    newer = fi.parse_funding_info([_row("QUSDT", 4), _row("BTCUSDT", 8)], captured_at=NOW)
    first = fi.write_funding_interval_snapshot(older, root)
    second = fi.write_funding_interval_snapshot(newer, root)
    loaded = fi.latest_funding_interval_snapshot_or_none(root)
    assert loaded is not None
    assert dict(loaded.interval_ms) == dict(newer.interval_ms)
    assert loaded.captured_at == newer.captured_at
    assert first.exists()
    assert second.exists()
    assert first != second


def test_latest_is_none_without_snapshots(tmp_path: Path) -> None:
    assert fi.latest_funding_interval_snapshot_or_none(tmp_path / "absent") is None
    (tmp_path / "empty").mkdir()
    assert fi.latest_funding_interval_snapshot_or_none(tmp_path / "empty") is None


@pytest.mark.parametrize("content", ["{", '{"captured_at": "2026-11-03T23:00:00+00:00"}', '{"captured_at": "x", "interval_hours": {"A": 4}}', '{"captured_at": "2026-11-03T23:00:00+00:00", "interval_hours": {"A": 3}}'])
def test_latest_corrupt_raises(tmp_path: Path, content: str) -> None:
    root = tmp_path / "fi"
    root.mkdir()
    (root / "20261103T230030Z.json").write_text(content, encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        fi.latest_funding_interval_snapshot_or_none(root)
