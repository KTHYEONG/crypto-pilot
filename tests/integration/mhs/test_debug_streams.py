"""Integration tests for MHS debug streams (P4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.mhs.telemetry import MHS_DEBUG_STREAMS, StageTelemetry, Tag

START = pd.Timestamp("2021-01-01", tz="UTC")
N_HOURS = 2000


def _write_mhs_market(root: Path, symbols: list[str], n_hours: int = N_HOURS) -> pd.Timestamp:
    hourly = pd.date_range(START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    hour_dir = root / "1h"
    minute_dir = root / "1m"
    five_dir = root / "5m"
    funding_dir = root / "funding"
    mark_dir = root / "markPriceKlines" / "1h"
    for d in (hour_dir, minute_dir, five_dir, funding_dir, mark_dir):
        d.mkdir(parents=True, exist_ok=True)

    n = len(hourly)
    minute_idx = pd.date_range(START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    for i, sym in enumerate(symbols):
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, n)))
        pd.DataFrame({"timestamp": epoch, "open": prices, "high": prices * 1.001, "low": prices * 0.999, "close": prices, "quote_vol": [1000.0] * n}).to_parquet(hour_dir / f"{sym}.parquet")

        minute_noise = rng.normal(0.0, 0.0003, len(minute_idx))
        hourly_level = np.repeat(prices, 60)[:len(minute_idx)]
        minute_prices = hourly_level * np.exp(minute_noise)
        pd.DataFrame({"timestamp": minute_epoch, "open": minute_prices, "high": minute_prices * 1.0005, "low": minute_prices * 0.9995, "close": minute_prices, "quote_vol": [1000.0] * len(minute_idx)}).to_parquet(minute_dir / f"{sym}.parquet")

        five_frame = pd.DataFrame({"open": minute_prices, "high": minute_prices * 1.0005, "low": minute_prices * 0.9995, "close": minute_prices}, index=minute_idx).resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        pd.DataFrame({"timestamp": (five_frame.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"), "open": five_frame["open"].to_numpy(), "high": five_frame["high"].to_numpy(), "low": five_frame["low"].to_numpy(), "close": five_frame["close"].to_numpy(), "quote_vol": np.full(len(five_frame), 5000.0)}).to_parquet(five_dir / f"{sym}.parquet")

        pd.DataFrame({"timestamp": epoch, "funding_rate": [0.00005] * n, "datetime": hourly}).to_parquet(funding_dir / f"{sym}.parquet")

        mark_hourly = pd.Series(minute_prices, index=minute_idx).resample("1h").last().reindex(hourly).to_numpy()
        pd.DataFrame({"timestamp": epoch, "open": mark_hourly, "high": mark_hourly, "low": mark_hourly, "close": mark_hourly, "datetime": hourly}).to_parquet(mark_dir / f"{sym}.parquet")
    return end


def test_debug_streams_constant():
    """MHS_DEBUG_STREAMS contains exactly the 7 required stream names."""
    assert len(MHS_DEBUG_STREAMS) == 7
    assert "panel" in MHS_DEBUG_STREAMS
    assert "folds" in MHS_DEBUG_STREAMS


def test_stream_writes_jsonl(tmp_path):
    """SCENARIO_ANALYSIS_ARCHITECTURE_11: stream() with sidecars enabled writes valid JSONL."""
    telemetry = StageTelemetry(log_run=False, debug_streams=True, streams_root=tmp_path)
    rows = [{"stage": "panel", "grid_bars": 100}, {"stage": "panel", "n_symbols": 8}]
    telemetry.stream("panel", rows)

    path = tmp_path / "panel.jsonl"
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert "stage" in row


def test_streams_off_creates_no_files(tmp_path):
    """With sidecars disabled, no files are created."""
    telemetry = StageTelemetry(log_run=False, debug_streams=False, streams_root=tmp_path)
    telemetry.stream("panel", [{"stage": "panel"}])
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_tag_scanning():
    """SCENARIO_ANALYSIS_ARCHITECTURE_12: Every tag in src/mhs/ belongs to {SYS, DATA, ALGO, EVAL}."""
    import re
    from pathlib import Path

    src_mhs = Path(__file__).resolve().parent.parent.parent.parent / "src" / "mhs"
    pattern = re.compile(r"\[([A-Z]+)\]")
    valid_tags = {Tag.SYS, Tag.DATA, Tag.ALGO, Tag.EVAL}
    violations = []
    for py_file in src_mhs.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001, S112
            continue
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for match in pattern.finditer(line):
                tag = match.group(1)
                # Skip docstrings and string literals containing [TAG]
                if tag not in valid_tags and '"""' not in line and "'''" not in line:
                    violations.append((py_file, i, tag))
    if violations:
        msg = "\n".join(f"  {p}:{line}: [{t}]" for p, line, t in violations[:10])
        pytest.fail(f"Invalid tags found:\n{msg}")
