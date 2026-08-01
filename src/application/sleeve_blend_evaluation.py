from __future__ import annotations

import dataclasses
import logging

from src.research.contracts import (
    CostModel,
    EvaluationReport,
    SleeveBlendEvaluationRequest,
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
from src.research.provenance.results import record_sleeve_blend_run
from src.research.sleeve_blend.backtest import (
    run_directional_sleeve_portfolio_fixed_weights,
    run_directional_sleeve_portfolio_with_weights,
    run_fixed_sleeve_portfolio_calibrated,
    run_fixed_sleeve_portfolio_with_leverage,
)

_FUNDING_SIGNED_DIRECTIONAL_V1 = "funding_signed_directional_v1"

_logger = logging.getLogger("SleeveBlendBacktestRunner")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0


def run_sleeve_blend_evaluation(request: SleeveBlendEvaluationRequest) -> EvaluationReport:
    """Execute one sealed sleeve blend evaluation.

    The default ``candidate_kind`` runs the fixed equal-weight long-only blend:
    equal-weight-blend the equity curves, calibrate one MDD-budget leverage
    scalar from the base-cost run, and re-run under stressed costs with that
    *same* frozen scalar. ``funding_signed_directional_v1`` instead runs the
    funding-gated long/short sleeve unlevered (leverage 1.0) with causal
    inverse-vol risk weights; its stress re-run reuses the *base* weight series
    verbatim and never re-calibrates around stressed costs. Both connect to the
    existing reliability gates and fail-closed promotion.
    """
    end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    if request.unseal_holdout:
        _logger.info("[EVAL] holdout unsealed: --end=%s", end or "(latest available)")

    costs = CostModel()
    if request.candidate_kind == _FUNDING_SIGNED_DIRECTIONAL_V1:
        result, weights = run_directional_sleeve_portfolio_with_weights(
            request.symbols,
            request.start,
            end,
            costs,
            initial_equity=request.initial_equity,
        )
        leverage = 1.0
    else:
        result, leverage = run_fixed_sleeve_portfolio_calibrated(
            request.symbols,
            request.start,
            end,
            costs,
            request.mdd_budget_fraction,
            initial_equity=request.initial_equity,
            signal_delay_bars=0,
        )
    metrics = compute_metrics(result.equity, result.trades)
    _logger.info(
        "[EVAL] sleeve_blend candidate=%s lev=%.4f cagr=%.4f mdd=%.4f sharpe=%.3f trades=%d",
        request.candidate_kind, leverage, metrics.cagr, metrics.mdd,
        metrics.sharpe, metrics.trade_count,
    )

    observation_gate = compute_equity_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)

    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    if request.candidate_kind == _FUNDING_SIGNED_DIRECTIONAL_V1:
        stress_result = run_directional_sleeve_portfolio_fixed_weights(
            request.symbols,
            request.start,
            end,
            stressed_costs,
            weights,
            initial_equity=request.initial_equity,
            signal_delay_bars=1,
        )
    else:
        stress_result = run_fixed_sleeve_portfolio_with_leverage(
            request.symbols,
            request.start,
            end,
            stressed_costs,
            leverage,
            initial_equity=request.initial_equity,
            signal_delay_bars=1,
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
        mdd_budget_fraction: float | None = request.mdd_budget_fraction
        if request.candidate_kind == _FUNDING_SIGNED_DIRECTIONAL_V1:
            mdd_budget_fraction = None
        record = record_sleeve_blend_run(
            symbols=tuple(request.symbols),
            candidate_kind=request.candidate_kind,
            mdd_budget_fraction=mdd_budget_fraction,
            leverage=leverage,
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
