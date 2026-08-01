from __future__ import annotations

import argparse
import logging

from src.application.sleeve_blend_evaluation import run_sleeve_blend_evaluation
from src.research.contracts import SleeveBlendEvaluationRequest

_logger = logging.getLogger("SleeveBlendBacktestRunner")

_DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-sleeve Donchian blend backtest")
    parser.add_argument("--symbols", nargs="+", default=list(_DEFAULT_SYMBOLS))
    parser.add_argument("--mdd-budget-fraction", type=float, default=0.85)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--unseal-holdout", action="store_true", default=False)
    parser.add_argument("--no-log-run", action="store_true", default=False)
    args = parser.parse_args()

    request = SleeveBlendEvaluationRequest(
        symbols=tuple(args.symbols),
        mdd_budget_fraction=args.mdd_budget_fraction,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    run_sleeve_blend_evaluation(request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
