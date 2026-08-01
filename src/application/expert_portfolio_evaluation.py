from __future__ import annotations

import dataclasses
import logging

import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_4h
from src.research.baseline.backtest import BacktestResult, run_backtest, run_directional_backtest
from src.research.contracts import CostModel, EvaluationReport, StrategySpec
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    split_holdout_segment,
)
from src.research.expert_portfolio.backtest import (
    ExpertPortfolioBacktestResult,
    run_expert_portfolio,
)
from src.research.expert_portfolio.contracts import (
    ExpertDefinition,
    ExpertPortfolioEvaluationRequest,
    ExpertPortfolioSpec,
)
from src.research.expert_portfolio.registry import load_expert_library
from src.research.provenance.results import record_expert_portfolio_run

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0

_EMPTY_TRADE_COLUMNS = (
    "expert_id",
    "symbol",
    "entry_bar",
    "exit_bar",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "qty",
    "reason",
    "pnl",
    "return_pct",
    "funding_pnl",
)

_logger = logging.getLogger("ExpertPortfolioBacktestRunner")


def _run_component(
    definition: ExpertDefinition,
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    signal_delay_bars: int,
) -> BacktestResult:
    """Execute one expert's causal return series via its registered runner."""
    if definition.runner == "run_backtest":
        if len(definition.symbols) != 1:
            raise ValueError(
                f"runner run_backtest requires exactly one symbol for {definition.expert_id}"
            )
        symbol = definition.symbols[0]
        df = load_ohlcv_4h(ohlcv_path(symbol, "1h"), start=start, end=end)
        return run_backtest(
            df, StrategySpec(symbol=symbol), costs, signal_delay_bars=signal_delay_bars,
        )
    if definition.runner == "run_directional_backtest":
        if len(definition.symbols) != 1:
            raise ValueError(
                f"runner run_directional_backtest requires exactly one symbol for "
                f"{definition.expert_id}"
            )
        symbol = definition.symbols[0]
        df = load_ohlcv_4h(ohlcv_path(symbol, "1h"), start=start, end=end)
        funding = load_funding_rates(funding_path(symbol))
        return run_directional_backtest(
            df, StrategySpec(symbol=symbol), costs, funding,
            signal_delay_bars=signal_delay_bars,
        )
    raise ValueError(
        f"runner '{definition.runner}' for expert {definition.expert_id} is not registered"
    )


def _concat_component_trades(
    results: dict[str, BacktestResult],
    common: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Concatenate component trades tagged with their expert, plus wall-clock times.

    ``entry_time``/``exit_time`` are resolved from each component's own equity
    index so holdout attribution never depends on a relative ``exit_bar`` whose
    meaning differs across components.
    """
    frames: list[pd.DataFrame] = []
    for expert_id, res in results.items():
        trades = res.trades.copy()
        if len(trades) > 0:
            trades["expert_id"] = expert_id
            trades["entry_time"] = res.equity.index[
                trades["entry_bar"].astype(int).to_numpy()
            ]
            trades["exit_time"] = res.equity.index[
                trades["exit_bar"].astype(int).to_numpy()
            ]
        frames.append(trades)
    if not frames:
        return pd.DataFrame(columns=list(_EMPTY_TRADE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def build_component_panel(
    spec: ExpertPortfolioSpec,
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    *,
    signal_delay_bars: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every expert's registered runner and return (panel, component trades).

    The panel is the completed component simple-return series on their common
    index; the trades frame is the concatenated component evidence used for the
    closed-trade sample-size guard and holdout attribution.
    """
    results: dict[str, BacktestResult] = {}
    for definition in spec.experts:
        results[definition.expert_id] = _run_component(
            definition, start, end, costs, signal_delay_bars,
        )
    common = sorted(set.intersection(*(set(res.equity.index) for res in results.values())))
    if len(common) < 2:
        raise DataIntegrityError(
            f"expert components share fewer than 2 common bars, got {len(common)}"
        )
    common = pd.DatetimeIndex(common)
    panel = pd.DataFrame(
        {expert_id: results[expert_id].equity.loc[common].pct_change() for expert_id in results},
        index=common,
    )
    trades = _concat_component_trades(results, common)
    return panel, trades


def _assemble_result(
    base: ExpertPortfolioBacktestResult,
    component_trades: pd.DataFrame,
) -> BacktestResult:
    """The master marked ledger with the concatenated component-trade evidence.

    The master ledger itself holds capital and opens no trades, but the gates
    use the component closed-trade count as a sample-size guard and the fold and
    holdout gates require a non-empty trade set, so the report carries the
    component trades exactly like the sleeve-blend path does.
    """
    return BacktestResult(
        equity=base.backtest_result.equity,
        trades=component_trades,
        signals=pd.DataFrame(),
    )


def run_expert_portfolio_evaluation(
    request: ExpertPortfolioEvaluationRequest,
) -> EvaluationReport:
    """Execute one sealed pre-registered expert portfolio evaluation.

    An unregistered library raises ``ValueError``. The base run computes the
    causal LCB targets; the stress run reuses the base target weights verbatim
    under stressed costs with the existing one-bar delay and never recomputes
    targets. Observation, fold, stress, and holdout reuse the existing canonical
    functions unchanged and promotion is composed only by
    ``compose_promotion_verdict``, so no allocation result can bypass a failing
    gate.
    """
    end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    if request.unseal_holdout:
        _logger.info("[EVAL] holdout unsealed: --end=%s", end or "(latest available)")

    library = load_expert_library(request.library_id)
    costs = CostModel()

    component_returns, component_trades = build_component_panel(
        library, request.start, end, costs,
    )
    base = run_expert_portfolio(
        component_returns, library, costs, initial_equity=request.initial_equity,
    )
    result = _assemble_result(base, component_trades)
    metrics = compute_metrics(result.equity, result.trades)
    _logger.info(
        "[EVAL] expert_portfolio library=%s cagr=%.4f mdd=%.4f trades=%d alloc_cost=%.6f",
        request.library_id, metrics.cagr, metrics.mdd, metrics.trade_count,
        float(base.allocation_cost.sum()),
    )

    observation_gate = compute_equity_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)

    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_panel, stress_trades = build_component_panel(
        library, request.start, end, stressed_costs, signal_delay_bars=1,
    )
    stress_base = run_expert_portfolio(
        stress_panel, library, stressed_costs,
        initial_equity=request.initial_equity,
        fixed_weights=base.target_weights,
        signal_delay_bars=1,
    )
    stress_result = _assemble_result(stress_base, stress_trades)
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
        record = record_expert_portfolio_run(
            library_fingerprint=library.fingerprint(),
            allocation_cost_total=float(base.allocation_cost.sum()),
            result=result,
            metrics=metrics,
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
