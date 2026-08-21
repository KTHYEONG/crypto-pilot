"""One-time script to capture the golden report for MHS identity testing.

Run with: uv run python tools/capture_golden.py

This generates tests/fixtures/golden/mhs_report_golden.json which is
the bit-exact reference for all subsequent phases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_mhs_market(
    root: Path,
    symbols: list[str],
    n_hours: int = 2000,
) -> pd.Timestamp:
    """Replicate the synthetic market from test_mhs_horizon_diagnostic.py."""
    START = pd.Timestamp("2021-01-01", tz="UTC")
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
    n_min = len(minute_idx)

    for i, sym in enumerate(symbols):
        sym_hourly = hourly
        sym_epoch = epoch
        sym_n = n
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, sym_n)))
        pd.DataFrame(
            {
                "timestamp": sym_epoch,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "quote_vol": [1000.0] * sym_n,
            },
        ).to_parquet(hour_dir / f"{sym}.parquet")

        sym_minute = minute_idx
        sym_minute_epoch = minute_epoch
        sym_n_min = n_min
        minute_noise = rng.normal(0.0, 0.0003, sym_n_min)
        hourly_level = np.repeat(prices, 60)[:sym_n_min]
        minute_prices = hourly_level * np.exp(minute_noise)
        pd.DataFrame(
            {
                "timestamp": sym_minute_epoch,
                "open": minute_prices,
                "high": minute_prices * 1.0005,
                "low": minute_prices * 0.9995,
                "close": minute_prices,
                "quote_vol": [1000.0] * sym_n_min,
            },
        ).to_parquet(minute_dir / f"{sym}.parquet")

        five_frame = pd.DataFrame(
            {
                "open": minute_prices,
                "high": minute_prices * 1.0005,
                "low": minute_prices * 0.9995,
                "close": minute_prices,
            },
            index=sym_minute,
        ).resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"},
        ).dropna()
        pd.DataFrame(
            {
                "timestamp": (five_frame.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
                "open": five_frame["open"].to_numpy(),
                "high": five_frame["high"].to_numpy(),
                "low": five_frame["low"].to_numpy(),
                "close": five_frame["close"].to_numpy(),
                "quote_vol": np.full(len(five_frame), 5000.0),
            },
        ).to_parquet(five_dir / f"{sym}.parquet")

        pd.DataFrame(
            {
                "timestamp": sym_epoch,
                "funding_rate": [0.00005] * sym_n,
                "datetime": sym_hourly,
            },
        ).to_parquet(funding_dir / f"{sym}.parquet")

        mark_hourly = (
            pd.Series(minute_prices, index=sym_minute)
            .resample("1h")
            .last()
            .reindex(sym_hourly)
            .to_numpy()
        )
        pd.DataFrame(
            {
                "timestamp": sym_epoch,
                "open": mark_hourly,
                "high": mark_hourly,
                "low": mark_hourly,
                "close": mark_hourly,
                "datetime": sym_hourly,
            },
        ).to_parquet(mark_dir / f"{sym}.parquet")
    return end


def main() -> None:
    import src.application.research.mhs.marks as marks
    import src.application.research.mhs.statistics as statistics
    import src.market_data.services.futures_collection as fc
    from src.application.research.mhs.contracts import MhsDiagnosticRequest
    from src.application.research.mhs.evaluation import run_mhs_horizon_diagnostic
    from src.research.universe.pit_universe import symbol_partition

    START = pd.Timestamp("2021-01-01", tz="UTC")
    DEV_SYMBOLS = [
        sym for sym in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(sym) == "dev"
    ][:8]

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        end = _write_mhs_market(root, DEV_SYMBOLS)

        # Monkey-patch paths
        orig_funding = marks.funding_path
        orig_mark = fc._mark_price_path
        orig_reps = statistics._BOOTSTRAP_REPLICATES
        orig_block = statistics._BOOTSTRAP_MEAN_BLOCK
        orig_seed = statistics._BOOTSTRAP_SEED

        marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        statistics._BOOTSTRAP_REPLICATES = 20
        statistics._BOOTSTRAP_MEAN_BLOCK = 24
        statistics._BOOTSTRAP_SEED = 20260807

        try:
            report = run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START), end=str(end), data_root=str(root),
                    execution_timeframe="1m", log_run=False,
                )
            )
        finally:
            marks.funding_path = orig_funding
            fc._mark_price_path = orig_mark
            statistics._BOOTSTRAP_REPLICATES = orig_reps
            statistics._BOOTSTRAP_MEAN_BLOCK = orig_block
            statistics._BOOTSTRAP_SEED = orig_seed

        golden_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden" / "mhs_report_golden.json"
        golden_path.parent.mkdir(parents=True, exist_ok=True)

        payload = report.to_payload()
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Golden captured: {golden_path}")
        print(f"  status: {report.status}")
        print(f"  books: {sorted(report.books.keys())}")
        print(f"  blend: {report.blend is not None}")
        print(f"  folds: {len(report.folds)}")


if __name__ == "__main__":
    main()
