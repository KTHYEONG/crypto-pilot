from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.core.config import ohlcv_path
from src.core.types import CostModel, StrategySpec
from src.data.loader import load_ohlcv_4h
from src.engine.backtest import run_backtest
from src.engine.results_log import record_run
from src.validation.candidate_promotion import compose_promotion_verdict
from src.validation.metrics import compute_metrics
from src.validation.reliability_gate import (
    compute_equity_reliability_gate,
    compute_fold_distribution,
    compute_stress_test_gate,
    split_holdout_segment,
)

_logger = logging.getLogger("BacktestRunner")


def _load_funding_rates(path: str) -> pd.Series:
    """Load a published-funding parquet into a monotonic rate Series."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"funding path does not exist: {path}")
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    else:
        raise RuntimeError("funding parquet must contain a 'datetime' or 'timestamp' column")
    if "funding_rate" not in df.columns:
        raise RuntimeError("funding parquet must contain a 'funding_rate' column")
    rates = pd.to_numeric(df["funding_rate"], errors="coerce")
    series = pd.Series(rates.to_numpy(dtype="float64"), index=pd.DatetimeIndex(ts))
    return series[series.index.notna()].sort_index()

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
    parser.add_argument("--min-taker-buy-ratio", type=float, default=None)
    parser.add_argument("--funding-path", default=None)
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

    spec = StrategySpec(symbol=args.symbol, min_taker_buy_ratio=args.min_taker_buy_ratio)
    costs = CostModel()
    path = ohlcv_path(args.symbol, "1h")

    df = load_ohlcv_4h(path, start=args.start, end=end)

    funding_rates = None
    if args.funding_path is not None:
        funding_rates = _load_funding_rates(args.funding_path)
        _logger.info(
            "[EVAL] funding loaded: path=%s rows=%d", args.funding_path, len(funding_rates),
        )
        if len(df) > 0:
            bar_period = df.index[1] - df.index[0]
            window_end = df.index[-1] + bar_period
            funding_rates = funding_rates[
                (funding_rates.index >= df.index[0]) & (funding_rates.index < window_end)
            ]
            _logger.info(
                "[EVAL] funding aligned to bar window: rows=%d", len(funding_rates),
            )
    if spec.min_taker_buy_ratio is not None:
        _logger.info(
            "[EVAL] candidate identity: taker_flow_confirmation "
            "min_taker_buy_ratio=%.3f funding_path=%s",
            spec.min_taker_buy_ratio, args.funding_path or "(none)",
        )

    result = run_backtest(
        df, spec, costs, initial_equity=args.initial_equity, funding_rates=funding_rates,
    )
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

    observation_gate = compute_equity_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)
    stress_gate = compute_stress_test_gate(df, spec, costs, funding_rates=funding_rates)

    holdout_gate = None
    if args.unseal_holdout and result.equity.index[-1] > HOLDOUT_CUTOFF:
        segment = split_holdout_segment(result, HOLDOUT_CUTOFF)
        observation_gate = compute_equity_reliability_gate(
            segment.observation_equity, len(segment.observation_trades),
        )
        holdout_gate = compute_equity_reliability_gate(
            segment.holdout_equity, len(segment.holdout_trades),
        )

    _logger.info(
        "[EVAL] reliability observation=%s lcb90=%.4f block=%d trades=%d",
        observation_gate.verdict, observation_gate.lcb90_cagr,
        observation_gate.block_size_used, observation_gate.trade_count,
    )
    _logger.info(
        "[EVAL] reliability fold max_period_contribution=%.4f gate_pass=%s n_folds=%d",
        fold_distribution.max_period_contribution, fold_distribution.gate_pass,
        fold_distribution.n_folds,
    )
    _logger.info(
        "[EVAL] reliability stress_test=%s stressed_cagr=%.4f",
        stress_gate.verdict, stress_gate.point_cagr,
    )
    if holdout_gate is not None:
        _logger.info(
            "[EVAL] reliability holdout=%s trades=%d holdout_mdd=%.4f holdout_cagr_sign=%.4f",
            holdout_gate.verdict, holdout_gate.trade_count,
            segment.holdout_mdd, segment.holdout_cagr_sign,
        )

    promotion = compose_promotion_verdict(observation_gate, fold_distribution, stress_gate, holdout_gate)
    _logger.info(
        "[EVAL] promotion status=%s observation=%s fold_gate=%s stress=%s holdout=%s",
        promotion.status, promotion.observation_verdict, promotion.fold_gate_pass,
        promotion.stress_verdict, promotion.holdout_verdict,
    )

    if not args.no_log_run:
        rec = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=args.start, end=str(end) if end is not None else None,
            initial_equity=args.initial_equity,
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
            promotion=promotion,
        )
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", rec["git_sha"], rec["git_dirty"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
