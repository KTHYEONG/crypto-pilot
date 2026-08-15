from __future__ import annotations

import logging

import pandas as pd

from src.common.config import ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_4h
from src.research.baseline.backtest import run_backtest
from src.research.contracts import (
    BaselineEvaluationRequest,
    CostModel,
    EvaluationReport,
    StrategySpec,
)
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (
    compute_equity_reliability_gate,
    compute_fold_distribution,
    compute_stress_test_gate,
    split_holdout_segment,
)
from src.research.provenance.results import record_run

_logger = logging.getLogger("BacktestRunner")


def _run_baseline_engine(
    request: BaselineEvaluationRequest,
    funding_rates: pd.Series | None,
    *,
    funding_path_label: str | None,
) -> EvaluationReport:
    """Shared baseline evaluation body driven by an explicitly supplied funding stream.

    ``funding_rates`` may be ``None`` only for the clearly named analytical
    zero-funding diagnostic; the promotion path always supplies an aligned
    stream, so a futures evaluation never silently assumes zero funding.
    """
    end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    if request.unseal_holdout:
        _logger.info("[EVAL] holdout unsealed: --end=%s", end or "(latest available)")

    spec = StrategySpec(symbol=request.symbol, min_taker_buy_ratio=request.min_taker_buy_ratio)
    costs = CostModel()
    path = ohlcv_path(request.symbol, "1h")

    df = load_ohlcv_4h(path, start=request.start, end=end)

    if funding_rates is not None:
        _logger.info(
            "[EVAL] funding loaded: path=%s rows=%d",
            funding_path_label or "(aligned)", len(funding_rates),
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
            spec.min_taker_buy_ratio, funding_path_label or "(none)",
        )

    result = run_backtest(
        df, spec, costs, initial_equity=request.initial_equity, funding_rates=funding_rates,
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
    if request.unseal_holdout and result.equity.index[-1] > HOLDOUT_CUTOFF:
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

    record = None
    if request.log_run:
        record = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=request.start, end=str(end) if end is not None else None,
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


def run_baseline_analysis_without_funding(
    request: BaselineEvaluationRequest,
) -> EvaluationReport:
    """Analytical zero-funding baseline for legacy diagnostics only.

    Clearly named as non-promotable: funding completeness is mandatory for
    futures promotion evidence. This path reproduces the frozen v1 result with
    ``funding_rates=None`` and is retained only for legacy tests; it must never
    be used for promotion evidence.
    """
    return _run_baseline_engine(request, None, funding_path_label=None)


def run_baseline_evaluation(request: BaselineEvaluationRequest) -> EvaluationReport:
    """Execute one sealed single-symbol Donchian promotion evaluation.

    A futures promotion evaluation requires an aligned funding stream: a missing
    ``funding_path`` raises ``DataIntegrityError`` (fail closed) instead of
    silently assuming zero funding. The clearly named
    :func:`run_baseline_analysis_without_funding` is the only zero-funding path
    and exists solely for legacy diagnostics.
    """
    if request.funding_path is None:
        raise DataIntegrityError(
            "futures baseline promotion evaluation requires an aligned funding stream; "
            "supply funding_path or use run_baseline_analysis_without_funding only "
            "for the analytical zero-funding legacy diagnostic"
        )
    funding_rates = load_funding_rates(request.funding_path)
    return _run_baseline_engine(
        request, funding_rates, funding_path_label=request.funding_path,
    )
