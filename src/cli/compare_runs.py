from __future__ import annotations

import argparse

import pandas as pd

from src.results_log import RUNS_LOG_PATH, load_runs

_SUMMARY_COLS = [
    "ts", "git_sha", "git_dirty", "symbol", "end",
    "metrics.trade_count", "metrics.cagr", "metrics.mdd",
    "metrics.sharpe", "metrics.profit_factor", "metrics.win_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recorded backtest runs")
    parser.add_argument("--last", type=int, default=10, help="Show only the most recent N runs")
    parser.add_argument("--sort-by", default="ts", help="Column to sort by, e.g. metrics.sharpe")
    parser.add_argument("--full", action="store_true", help="Show every column instead of the summary set")
    args = parser.parse_args()

    df = load_runs()
    if df.empty:
        print(f"No runs recorded yet at {RUNS_LOG_PATH}")
        return

    if not args.full:
        cols = [c for c in _SUMMARY_COLS if c in df.columns]
        df = df[cols]

    df = df.sort_values(args.sort_by).tail(args.last)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
