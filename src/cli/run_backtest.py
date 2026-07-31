from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.application.baseline_evaluation import run_baseline_evaluation
from src.market_data.storage.loaders import load_funding_rates
from src.research.contracts import BaselineEvaluationRequest
from src.research.evaluation.policy import HOLDOUT_CUTOFF

_logger = logging.getLogger("BacktestRunner")

__all__ = ["HOLDOUT_CUTOFF"]


def _load_funding_rates(path: str) -> pd.Series:
    """Backward-compatible alias delegating to the shared data loader."""
    return load_funding_rates(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v1 Donchian backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--min-taker-buy-ratio", type=float, default=None)
    parser.add_argument("--funding-path", default=None)
    parser.add_argument("--unseal-holdout", action="store_true", default=False)
    parser.add_argument(
        "--no-log-run", action="store_true", default=False,
        help="Skip appending this run to docs/results/runs.jsonl",
    )
    args = parser.parse_args()

    request = BaselineEvaluationRequest(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        min_taker_buy_ratio=args.min_taker_buy_ratio,
        funding_path=args.funding_path,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_baseline_evaluation(request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
