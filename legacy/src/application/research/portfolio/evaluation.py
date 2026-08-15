from __future__ import annotations

import dataclasses
import logging

import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_4h
from src.research.contracts import (
    CostModel,
    EvaluationReport,
    PortfolioEvaluationRequest,
    PortfolioSpec,
    StrategySpec,
)
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    split_holdout_segment,
)
from src.research.portfolio.backtest import run_portfolio_backtest
from src.research.portfolio.defaults import STRESS_FEE_MULT, STRESS_SLIPPAGE_MULT
from src.research.portfolio.universe import select_liquid_universe
from src.research.provenance.results import record_portfolio_run

_logger = logging.getLogger("PortfolioBacktestRunner")


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
    rates = load_funding_rates(str(path))
    if len(rates) == 0:
        _logger.warning("[EVAL] excluding funding for %s: empty series", symbol)
        return None
    window_end = frame.index[-1] + (frame.index[1] - frame.index[0])
    return rates[(rates.index >= frame.index[0]) & (rates.index < window_end)]


def run_portfolio_evaluation(request: PortfolioEvaluationRequest) -> EvaluationReport:
    """Execute one sealed causal-liquidity portfolio evaluation.

    Applies the shared holdout policy, preserves daily liquidity selection,
    funding alignment, position limits, and the 2.5% aggregate initial-risk
    invariant, and appends a JSONL result only when ``request.log_run`` is true.
    """
    end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    if request.unseal_holdout:
        _logger.info("[EVAL] holdout unsealed: --end=%s", end or "(latest available)")

    strategy_spec = StrategySpec()
    portfolio_spec = PortfolioSpec()
    costs = CostModel()

    frames: dict[str, pd.DataFrame] = {}
    funding_rates: dict[str, pd.Series] = {}
    for symbol in request.symbols:
        frame = _load_symbol_frame(symbol, request.start, end)
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
        initial_equity=request.initial_equity, signal_delay_bars=0,
    )
    metrics = compute_metrics(result.equity, result.trades)
    _logger.info(
        "[EVAL] portfolio strategy cagr=%.4f mdd=%.4f sharpe=%.3f trades=%d",
        metrics.cagr, metrics.mdd, metrics.sharpe, metrics.trade_count,
    )

    observation_gate = compute_equity_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)

    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * STRESS_SLIPPAGE_MULT,
    )
    stress_result = run_portfolio_backtest(
        frames, funding_rates, strategy_spec, portfolio_spec, stressed_costs,
        initial_equity=request.initial_equity, signal_delay_bars=1,
    )
    stress_metrics = compute_metrics(stress_result.equity, stress_result.trades)
    stress_gate = compute_equity_reliability_gate(
        stress_result.equity,
        len(stress_result.trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )

    holdout_gate = None
    if request.unseal_holdout and result.equity.index[-1] > HOLDOUT_CUTOFF:
        segment = split_holdout_segment(result, HOLDOUT_CUTOFF)
        observation_gate = compute_equity_reliability_gate(
            segment.observation_equity, len(segment.observation_trades),
        )
        holdout_gate = compute_equity_reliability_gate(
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

    record = None
    if request.log_run:
        record = record_portfolio_run(
            symbols=tuple(sorted(frames)),
            portfolio_spec=portfolio_spec,
            costs=costs,
            result=result,
            metrics=metrics,
            start=request.start,
            end=str(end) if end is not None else None,
            initial_equity=request.initial_equity,
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
            promotion=promotion,
        )
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", record["git_sha"], record["git_dirty"])

    return EvaluationReport(
        status="PASS",
        result=result,
        metrics=metrics,
        observation=observation_gate,
        fold_distribution=fold_distribution,
        stress=stress_gate,
        holdout=holdout_gate,
        promotion=promotion,
        record=record,
    )
