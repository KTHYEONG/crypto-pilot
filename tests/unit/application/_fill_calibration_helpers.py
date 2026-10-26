"""Shared fixtures for fill calibration report tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.fill_calibration import FillCalibrationRequest
from src.engine.account_ledger import AccountMarkPanels


def _ts(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw, tz="UTC")


def _request(tmp_path: Path, **overrides) -> FillCalibrationRequest:
    base = {
        "evaluation_start": _ts("2023-01-02T00:00:00Z"),
        "evaluation_end": _ts("2023-01-04T00:00:00Z"),
        "archive_root": tmp_path / "archive",
        "k_bars": (1,),
        "bootstrap_samples": 100,
        "seed": 7,
    }
    base.update(overrides)
    return FillCalibrationRequest(**base)


def _toy_parts(symbols=("AAAUSDT", "BBBUSDT"), closes=(100.0, 200.0)):
    entries = pd.DatetimeIndex([_ts("2023-01-02T00:00:00Z"), _ts("2023-01-03T00:00:00Z")])
    anchors = pd.DatetimeIndex([_ts("2023-01-02T23:00:00Z"), _ts("2023-01-03T23:00:00Z")])
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    weights = pd.DataFrame([[1.0, 0.0], [0.5, 0.5]], index=entries, columns=list(symbols))
    close = pd.DataFrame(
        {s: np.full(len(grid), c, dtype="float64") for s, c in zip(symbols, closes, strict=True)}, index=grid
    )
    high = close * 1.001
    low = close * 0.999
    marks = AccountMarkPanels(close=close, high=high, low=low)
    funding = pd.DataFrame(0.0, index=entries, columns=list(symbols))
    adv = pd.DataFrame(1e9, index=entries, columns=list(symbols))
    sigma = pd.DataFrame(0.01, index=entries, columns=list(symbols))
    return (weights, marks, funding, adv, sigma, anchors)


def _install_backtest(monkeypatch: pytest.MonkeyPatch, parts) -> None:
    import src.application.fill_calibration as cal

    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)


def _install_tape(
    monkeypatch: pytest.MonkeyPatch, archive_root: Path, frames: dict[tuple[str, str], pd.DataFrame], coverage: dict
) -> None:
    from src.market_data.public_archive import store

    monkeypatch.setattr(store, "load_coverage", lambda root, dataset: coverage)
    for (symbol, month), frame in frames.items():
        path = store.partition_path(archive_root, "aggTrades", symbol, month, kind="slice")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)


def _slice_frame(day: date, rows: list[tuple[int, int, float, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transact_time_ms": [r[0] for r in rows],
            "agg_trade_id": [r[1] for r in rows],
            "price": [r[2] for r in rows],
            "quantity": [r[3] for r in rows],
            "is_buyer_maker": [r[4] for r in rows],
        }
    )


def _coverage_row(symbol: str, day: date, status: str):
    from src.market_data.public_archive.store import CoverageRow

    return CoverageRow(
        dataset="aggTrades",
        symbol=symbol,
        day=day,
        status=status,
        raw_sha256=None,
        rows=10,
        recorded_at=_ts("2023-01-05T00:00:00Z"),
        detail=None,
    )


def _day_ms(day: date, hour: float) -> int:
    return int((pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour)).value // 10**6)


def _args(**overrides):
    args = type("A", (), {})()
    args.start = "2023-01-01"
    args.end = "2023-02-01"
    args.archive_root = None
    args.paper_run_dir = None
    args.live_capture_root = None
    args.k_bars = None
    args.touch_lookback_s = None
    args.queue_ahead_fraction = None
    args.bootstrap_samples = None
    args.seed = None
    args.output_dir = None
    args.data_root = None
    args.total_tree_pss_bytes = None
    args.replay_tree_pss_bytes = None
    args.min_available_bytes = None
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _paper_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trades=None):
    _install_backtest(monkeypatch, _toy_parts())
    day = date(2023, 1, 2)
    stamp = _ts("2023-01-02T23:01:00Z")
    frame = _slice_frame(day, trades if trades is not None else [(int(stamp.value // 10**6) + 20_000, 1, 99.0, 100.0, True)])
    cov = {(s, d): _coverage_row(s, d, "ok") for s in ("AAAUSDT", "BBBUSDT") for d in (day, date(2023, 1, 3))}
    _install_tape(monkeypatch, tmp_path / "archive", {(s, "202301"): frame for s in ("AAAUSDT", "BBBUSDT")}, cov)
    live = tmp_path / "capture"
    live.mkdir()
    return stamp, live


def _paper_run(tmp_path: Path, lines: list[str], mode: str = "paper") -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    synthetic = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            synthetic.append(line)
            continue
        if isinstance(record, dict):
            if record.get("event") == "order_posted":
                record.setdefault("side", "BUY")
            elif record.get("event") == "fill":
                record.setdefault("qty", "5")
        synthetic.append(json.dumps(record) + "\n")
    (run_dir / "audit" / "a.jsonl").write_text("".join(synthetic), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"mode": mode}), encoding="utf-8")
    return run_dir
