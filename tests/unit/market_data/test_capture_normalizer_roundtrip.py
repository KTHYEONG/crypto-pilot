"""End-to-end raw capture records derive through the normalizer."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.capture.config import CaptureConfig
from src.capture.journal import SegmentWriter
from src.capture.rest import GridSampler, RateGate, RestStatus
from src.capture.ws import ForceOrderCapture, WsStatus
from src.market_data.streams.dedupe_window import DedupeWindow
from src.market_data.streams.normalizer import NormalizerCheckpoint, NormalizerConfig, normalize_once

GRID_NS = int(datetime(2026, 9, 26, 10, 0, tzinfo=UTC).timestamp() * 1_000_000_000)
BOOK_BODY = json.dumps(
    [{"symbol": "BTCUSDT", "bidPrice": "70000.5", "bidQty": "1.2", "askPrice": "70001.0", "askQty": "0.8", "time": GRID_NS // 1_000_000}]
)
FORCE_ORDER_FRAME = json.dumps(
    {
        "e": "forceOrder",
        "E": GRID_NS // 1_000_000 + 1000,
        "o": {
            "s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "GTC", "q": "1.25",
            "p": "70000.0", "ap": "70000.0", "X": "FILLED", "l": "1.25", "z": "1.25",
            "T": GRID_NS // 1_000_000 + 1000,
        },
    }
)


def _normalize(root: Path, liquidations: Path):
    return normalize_once(
        root,
        liquidations,
        NormalizerConfig(),
        checkpoint=NormalizerCheckpoint(files={}, coverage={}),
        now=pd.Timestamp(GRID_NS + 300_000_000_000, unit="ns", tz="UTC"),
        dedupe=DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0),
    )


def _rows(root: Path, dataset: str | None) -> pd.DataFrame:
    files = sorted(root.rglob("*.parquet"))
    selected = [path for path in files if dataset is None or dataset in path.parts]
    return pd.concat([pd.read_parquet(path) for path in selected], ignore_index=True) if selected else pd.DataFrame()


async def _capture_rest(root: Path, status_code: int, body: str) -> RestStatus:
    async def fetch(_url: str) -> tuple[int, str, dict[str, str]]:
        return status_code, body, {}

    status = RestStatus()
    sampler = GridSampler(
        stream="book_ticker",
        url="https://example.invalid/book",
        interval_s=60,
        config=CaptureConfig(),
        writer=SegmentWriter(root, "book_ticker", "blue"),
        fetch=fetch,
        gate=RateGate(60.0),
        status=status,
        clock_ns=lambda: GRID_NS + 1_000_000_000,
        sleep=lambda _delay: asyncio.sleep(0),
        shutdown=lambda: False,
    )
    await sampler._sample_slot(GRID_NS)
    return status


def test_captured_rest_record_derives_book_ticker_rows(tmp_path: Path) -> None:
    status = asyncio.run(_capture_rest(tmp_path, 200, BOOK_BODY))
    _, report = _normalize(tmp_path, tmp_path / "liq")
    rows = _rows(tmp_path, "book_ticker")
    assert status.first_ok_at_ns == GRID_NS + 1_000_000_000
    assert report.rows_written["book_ticker"] == 1
    assert rows["symbol"].tolist() == ["BTCUSDT"]
    assert rows["captured_at"].iloc[0] == pd.Timestamp(GRID_NS, unit="ns", tz="UTC")


def test_captured_force_order_frame_derives_liquidation(tmp_path: Path) -> None:
    state = {"stopped": False}

    class Connection:
        async def read(self, _wait_s: float) -> tuple[str, Any]:
            state["stopped"] = True
            return "text", FORCE_ORDER_FRAME

        async def ping(self) -> None:
            return None

        async def close(self) -> None:
            return None

    async def connect(_url: str) -> Connection:
        return Connection()

    capture = ForceOrderCapture(
        config=CaptureConfig(),
        writer=SegmentWriter(tmp_path, "force_order", "blue"),
        status=WsStatus(),
        connect=connect,
        clock_ns=lambda: GRID_NS + 1_000_000_000,
        sleep=lambda _delay: asyncio.sleep(0),
        shutdown=lambda: state["stopped"],
    )
    async def run() -> None:
        await capture.run()

    asyncio.run(run())
    _, report = _normalize(tmp_path, tmp_path / "liq")
    rows = _rows(tmp_path / "liq", None)
    assert report.rows_written["force_order"] == 1
    assert len(rows) == 1
    assert (rows.iloc[0]["symbol"], rows.iloc[0]["side"]) == ("BTCUSDT", "SELL")
    assert rows.iloc[0]["price"] == pytest.approx(70000.0)
    assert rows.iloc[0]["orig_qty"] == pytest.approx(1.25)


def test_non_200_rest_record_neither_readies_nor_derives(tmp_path: Path) -> None:
    status = asyncio.run(_capture_rest(tmp_path, 204, ""))
    _, report = _normalize(tmp_path, tmp_path / "liq")
    assert status.first_ok_at_ns is None
    assert report.rows_written.get("book_ticker", 0) == 0
    assert _rows(tmp_path, "book_ticker").empty
