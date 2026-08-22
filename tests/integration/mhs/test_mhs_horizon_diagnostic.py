

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.application.research.mhs import evaluation as ev
from src.research.universe.pit_universe import symbol_partition

START = pd.Timestamp("2021-01-01", tz="UTC")
N_HOURS = 2000
DEV_SYMBOLS = [
    sym for sym in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
    if symbol_partition(sym) == "dev"
][:8]


def _write_mhs_market(
    root: Path,
    symbols: list[str],
    late_listings: dict[str, pd.Timestamp] | None = None,
    n_hours: int = N_HOURS,
) -> pd.Timestamp:
    late_listings = late_listings or {}
    hourly = pd.date_range(START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    hour_dir = root / "1h"
    minute_dir = root / "1m"
    five_dir = root / "5m"
    funding_dir = root / "funding"
    mark_dir = root / "markPriceKlines" / "1h"
    hour_dir.mkdir(parents=True, exist_ok=True)
    minute_dir.mkdir(parents=True, exist_ok=True)
    five_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)
    mark_dir.mkdir(parents=True, exist_ok=True)

    n = len(hourly)
    minute_idx = pd.date_range(START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    n_min = len(minute_idx)

    for i, sym in enumerate(symbols):
        sym_start = late_listings.get(sym, START)
        sym_hourly = hourly[hourly >= sym_start]
        sym_epoch = (sym_hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        sym_n = len(sym_hourly)
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

        sym_minute = minute_idx[minute_idx >= sym_start]
        sym_minute_epoch = (sym_minute - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        sym_n_min = len(sym_minute)
        # Minute fills track the same underlying instrument as the 1h close
        # within a tight intra-hour noise band, mirroring real exchange data
        # where the 1h close and mark price are the same market -- an
        # unrelated independent random walk diverges unboundedly from
        # `prices` over a multi-month window and spuriously trips the
        # fill_mark_parity_mask gate (I1) since mark is derived from these
        # same minute fills.
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

        # 5-minute execution grid for the default ``execution_timeframe="5m"``
        # runs (SCENARIO_MHS_ANNUALIZATION_04), derived from the same minute
        # path so the 1m/5m ledgers share one underlying price process.
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





























LATE_SYMBOL = "MHSAUSDT"
LATE_START = pd.Timestamp("2021-02-01", tz="UTC")












FOLD_WINDOW_FOLD = ev.AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)






_SUBPROCESS_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import pandas as pd
import src.market_data.services.futures_collection as fc
import src.application.research.mhs.marks as marks
from src.application.research.mhs.evaluation import persist_mhs_horizon_diagnostic_report

root = Path(sys.argv[2])
out = Path(sys.argv[3])
marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
fc._mark_price_path = lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
ev._BOOTSTRAP_REPLICATES = 20
ev._BOOTSTRAP_MEAN_BLOCK = 24
ev._BOOTSTRAP_SEED = 20260807
report = run_mhs_horizon_diagnostic(
    MhsDiagnosticRequest(
        start="2021-01-01", end=str(ev.pd.Timestamp(sys.argv[4])),
        data_root=str(root), mark_mode="cache_required",
        execution_timeframe="1m", log_run=False,
        max_rss_bytes=int(sys.argv[5]),
    ),
)
persist_mhs_horizon_diagnostic_report(report, out)
payload = json.loads(out.read_text())
sys.stdout.write(json.dumps({
    "status": report.status,
    "persisted": out.exists(),
    "go_eligible": report.research_go.eligible,
    "reasons": list(report.research_go.reason_codes),
    "stage_count": len(report.resource_measurements),
}))
"""






# SCENARIO_MHS_EVIDENCE_WEIGHTING_FOLD_LEVEL_CAGR_CHANGES is a manual
# integration test run via CLI, not automated pytest, due to ~3-minute
# full-panel replay cost.  The acceptance criterion requires at least one
# fold's primary_geometric_cagr to differ from the frozen equal-weight
# baseline [0.1183, 0.0815, 1.0502] by more than 1e-3.

