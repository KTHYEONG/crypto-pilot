"""
CLI: FRAMA vs SuperTrend direction agreement + mean EvR (POC; no production signal change).

Usage:
  python scripts/run_frama_evr_ab.py --csv path/to/ohlcv.csv
  python scripts/run_frama_evr_ab.py   # synthetic demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.spot_strategy.frama_evr_poc import run_frama_evr_ab_summary

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("frama_evr_ab")


def _demo_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    r = rng.normal(0.0, 0.02, size=n).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(r))
    noise = rng.normal(0.0, 0.5, size=n)
    high = close + np.abs(noise)
    low = close - np.abs(noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(1e3, 1e5, size=n)
    dt = pd.date_range("2020-01-01", periods=n, freq="4h")
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None, help="OHLCV CSV with open,high,low,close,volume")
    p.add_argument("--frama-period", type=int, default=16)
    p.add_argument("--st-period", type=int, default=10)
    p.add_argument("--st-mult", type=float, default=3.0)
    args = p.parse_args()

    if args.csv:
        path = Path(args.csv)
        df = pd.read_csv(path)
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise SystemExit(f"Missing column: {col}")
    else:
        df = _demo_df()
        _logger.info("Using synthetic demo data (pass --csv for real OHLCV).")

    s = run_frama_evr_ab_summary(
        df,
        frama_period=args.frama_period,
        supertrend_period=args.st_period,
        supertrend_mult=args.st_mult,
    )
    _logger.info("bars=%d", s.n_bars)
    _logger.info("frama_bull_share=%.4f st_bull_share=%.4f", s.frama_bull_share, s.st_bull_share)
    _logger.info("direction_agreement=%.4f mean_evr=%.6f", s.direction_agreement, s.mean_evr)


if __name__ == "__main__":
    main()
