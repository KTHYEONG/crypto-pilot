from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import CostModel, StrategySpec, ohlcv_path
from src.data.loader import load_ohlcv_4h
from src.engine import run_backtest
from src.metrics import compute_metrics
from src.results_log import record_run

_logger = logging.getLogger("BacktestRunner")

# End of the observation window (spec section 3.2). Note the 23:59:59 boundary:
# load_ohlcv_4h filters "index <= end", and a bare "2025-12-31" parses to
# 00:00:00, which would drop the last 5 bars of that day.
HOLDOUT_CUTOFF = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v1 Donchian backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--unseal-holdout", action="store_true", default=False)
    parser.add_argument(
        "--no-log-run", action="store_true", default=False,
        help="Skip appending this run to docs/results/runs.jsonl",
    )
    args = parser.parse_args()

    end: str | pd.Timestamp | None
    if args.unseal_holdout:
        end = args.end
        _logger.info("[EVAL] holdout unsealed: --end=%s", end or "(latest available)")
    elif args.end is None:
        # Default: never silently run past the sealed observation window.
        end = HOLDOUT_CUTOFF
    else:
        end_ts = pd.Timestamp(args.end, tz="UTC")
        if end_ts > HOLDOUT_CUTOFF:
            raise RuntimeError(
                f"Holdout sealed: --end {args.end} > {HOLDOUT_CUTOFF}. "
                "Pass --unseal-holdout to override."
            )
        end = args.end

    spec = StrategySpec(symbol=args.symbol)
    costs = CostModel()
    path = ohlcv_path(args.symbol, "1h")

    df = load_ohlcv_4h(path, start=args.start, end=end)
    result = run_backtest(df, spec, costs, initial_equity=args.initial_equity)
    metrics = compute_metrics(result.equity, result.trades)

    _logger.info(
        "[EVAL] strategy(risk=%.3f,lev<=%.1f)  cagr=%.4f mdd=%.4f sharpe=%.3f sortino=%.3f calmar=%.3f",
        spec.risk_per_trade, spec.max_leverage,
        metrics.cagr, metrics.mdd, metrics.sharpe, metrics.sortino, metrics.calmar,
    )
    _logger.info(
        "[EVAL] trades=%d win=%.3f pf=%.3f reason_mix=%s",
        metrics.trade_count, metrics.win_rate, metrics.profit_factor,
        result.trades["reason"].value_counts().to_dict() if len(result.trades) > 0 else {},
    )
    _logger.info("[EVAL] exposure=%.3f", metrics.exposure)
    _logger.info("[EVAL] trades_per_year=%s", metrics.trades_per_year)

    if not args.no_log_run:
        rec = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=args.start, end=str(end) if end is not None else None,
            initial_equity=args.initial_equity,
        )
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", rec["git_sha"], rec["git_dirty"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
