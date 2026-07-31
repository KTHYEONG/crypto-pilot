from __future__ import annotations

import argparse
import dataclasses
import logging

import pandas as pd

from src.cli.run_backtest import HOLDOUT_CUTOFF, _load_funding_rates
from src.core.config import funding_path, ohlcv_path
from src.core.types import CostModel, PortfolioSpec, StrategySpec
from src.data.loader import DataIntegrityError, load_ohlcv_4h
from src.data.portfolio_universe import select_liquid_universe
from src.engine.portfolio_backtest import run_portfolio_backtest
from src.engine.results_log import record_portfolio_run
from src.validation.candidate_promotion import compose_promotion_verdict
from src.validation.metrics import compute_metrics
from src.validation.reliability_gate import (
    ReliabilityGateConfig,
    compute_fold_distribution,
    compute_portfolio_reliability_gate,
    split_holdout_segment,
)

_logger = logging.getLogger("PortfolioBacktestRunner")

# Pre-declared data-complete USDT-perpetual majors. The daily liquidity
# selection still chooses the top five by trailing quote volume among these.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
)

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0


def _load_symbol_frame(symbol: str, start: str | None, end: str | pd.Timestamp | None) -> pd.DataFrame | None:
    path = ohlcv_path(symbol, "1h")
    try:
        return load_ohlcv_4h(path, start=start, end=end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.warning("[EVAL] excluding %s: %s", symbol, exc)
        return None


def _load_symbol_funding(symbol: str, frame: pd.DataFrame) -> pd.Series | None:
    path = funding_path(symbol)
    if not path.exists():
        _logger.warning("[EVAL] excluding funding for %s: file missing", symbol)
        return None
    rates = _load_funding_rates(str(path))
    if len(rates) == 0:
        _logger.warning("[EVAL] excluding funding for %s: empty series", symbol)
        return None
    window_end = frame.index[-1] + (frame.index[1] - frame.index[0])
    return rates[(rates.index >= frame.index[0]) & (rates.index < window_end)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal liquidity portfolio v2 backtest")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--unseal-holdout", action="store_true", default=False)
    parser.add_argument("--no-log-run", action="store_true", default=False)
    args = parser.parse_args()

    if args.unseal_holdout:
        end: str | pd.Timestamp | None = args.end
        _logger.info("[EVAL] holdout unsealed: --end=%s", end or "(latest available)")
    elif args.end is None:
        end = HOLDOUT_CUTOFF
    else:
        end_ts = pd.Timestamp(args.end, tz="UTC")
        if end_ts > HOLDOUT_CUTOFF:
            raise RuntimeError(
                f"Holdout sealed: --end {args.end} > {HOLDOUT_CUTOFF}. "
                "Pass --unseal-holdout to override."
            )
        end = args.end

    strategy_spec = StrategySpec()
    portfolio_spec = PortfolioSpec()
    costs = CostModel()

    frames: dict[str, pd.DataFrame] = {}
    funding_rates: dict[str, pd.Series] = {}
    for symbol in args.symbols:
        frame = _load_symbol_frame(symbol, args.start, end)
        if frame is None:
            continue
        funding = _load_symbol_funding(symbol, frame)
        if funding is None:
            continue
        frames[symbol] = frame
        funding_rates[symbol] = funding

    if len(frames) < portfolio_spec.universe_size:
        raise RuntimeError(
            f"only {len(frames)} of {portfolio_spec.universe_size} required data-complete "
            f"symbols could be loaded: {sorted(frames)}"
        )
    _logger.info(
        "[EVAL] portfolio candidates=%d symbols=%s", len(frames), sorted(frames),
    )

    # The first bar cannot have a completed trailing liquidity window.  Use the
    # first timestamp at which every loaded symbol has the declared lookback;
    # the engine itself performs the same causal selection at each UTC day
    # boundary, so this log now describes a real eligible rebalance rather than
    # an intentionally empty warm-up selection.
    first_available = max(frame.index[0] for frame in frames.values())
    rebalance_time = pd.Timestamp(
        first_available + pd.Timedelta(days=portfolio_spec.liquidity_lookback_days),
    ).tz_convert("UTC")
    initial_universe = select_liquid_universe(frames, as_of=rebalance_time, spec=portfolio_spec)
    _logger.info(
        "[EVAL] portfolio universe as_of=%s selected=%s",
        rebalance_time, list(initial_universe),
    )

    result = run_portfolio_backtest(
        frames, funding_rates, strategy_spec, portfolio_spec, costs,
        initial_equity=args.initial_equity, signal_delay_bars=0,
    )
    metrics = compute_metrics(result.equity, result.trades)
    _logger.info(
        "[EVAL] portfolio strategy cagr=%.4f mdd=%.4f sharpe=%.3f trades=%d",
        metrics.cagr, metrics.mdd, metrics.sharpe, metrics.trade_count,
    )

    observation_gate = compute_portfolio_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)

    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_result = run_portfolio_backtest(
        frames, funding_rates, strategy_spec, portfolio_spec, stressed_costs,
        initial_equity=args.initial_equity, signal_delay_bars=1,
    )
    stress_metrics = compute_metrics(stress_result.equity, stress_result.trades)
    stress_gate = compute_portfolio_reliability_gate(
        stress_result.equity,
        len(stress_result.trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )

    holdout_gate = None
    if args.unseal_holdout and result.equity.index[-1] > HOLDOUT_CUTOFF:
        segment = split_holdout_segment(result, HOLDOUT_CUTOFF)
        observation_gate = compute_portfolio_reliability_gate(
            segment.observation_equity, len(segment.observation_trades),
        )
        holdout_gate = compute_portfolio_reliability_gate(
            segment.holdout_equity, len(segment.holdout_trades),
        )
        _logger.info(
            "[EVAL] reliability holdout=%s trades=%d holdout_mdd=%.4f holdout_cagr_sign=%.4f",
            holdout_gate.verdict, holdout_gate.trade_count,
            segment.holdout_mdd, segment.holdout_cagr_sign,
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
        stress_gate.verdict, stress_metrics.cagr,
    )

    promotion = compose_promotion_verdict(observation_gate, fold_distribution, stress_gate, holdout_gate)
    _logger.info(
        "[EVAL] promotion status=%s observation=%s fold_gate=%s stress=%s holdout=%s",
        promotion.status, promotion.observation_verdict, promotion.fold_gate_pass,
        promotion.stress_verdict, promotion.holdout_verdict,
    )

    if not args.no_log_run:
        rec = record_portfolio_run(
            symbols=tuple(sorted(frames)),
            portfolio_spec=portfolio_spec,
            costs=costs,
            result=result,
            metrics=metrics,
            start=args.start,
            end=str(end) if end is not None else None,
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
