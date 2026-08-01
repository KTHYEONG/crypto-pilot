from __future__ import annotations

import dataclasses
import hashlib
import json
import logging

import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_4h
from src.research.baseline.backtest import BacktestResult
from src.research.contracts import (
    CostModel,
    EvaluationReport,
    TechnicalExpertEvaluationRequest,
)
from src.research.evaluation.metrics import Metrics, compute_metrics
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.evaluation.promotion import (
    CandidateIdentity,
    PromotionResult,
    compose_promotion_verdict,
)
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    split_holdout_segment,
)
from src.research.provenance.code_manifest import TECHNICAL_CODE_UNITS, compute_code_hash
from src.research.provenance.results import record_technical_expert_run
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.catalog import resolve_technical_candidate
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.provenance import technical_data_hashes

_logger = logging.getLogger("TechnicalExpertBacktestRunner")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0

_BASE_DELAY_BARS = 1
_STRESS_DELAY_BARS = 2

_INITIAL_EQUITY = 10_000.0


def _data_hashes(symbol: str) -> dict[str, str]:
    """Shared data fingerprint helper kept as the module-local entry point.

    ``technical_data_hashes`` is the single provenance source shared with the
    library admission evaluator so both fingerprint identical bytes.
    """
    return technical_data_hashes(symbol)


def _candidate_id(
    *,
    return_source: str,
    symbol: str,
    observation_end: str,
    costs: CostModel,
    data_hashes: dict[str, str],
    code_hash: str,
) -> str:
    payload = {
        "return_source": return_source,
        "symbol": symbol,
        "observation_end": observation_end,
        "costs": dataclasses.asdict(costs),
        "data_hashes": data_hashes,
        "code_hash": code_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _candidate_identity(
    *,
    candidate: TechnicalCandidate,
    symbol: str,
    observation_end: str,
    costs: CostModel,
    data_hashes: dict[str, str],
    code_hash: str,
    data_start: str,
    data_end: str,
) -> CandidateIdentity:
    return CandidateIdentity(
        hypothesis_id=candidate.return_source,
        code_hash=code_hash,
        parameters={
            "data_hashes": data_hashes,
            "costs": dataclasses.asdict(costs),
            "signal_delay_bars": _BASE_DELAY_BARS,
            "return_source": candidate.return_source,
            "family": candidate.family,
            "side": candidate.side,
            "candidate_config": dict(candidate.config),
        },
        data_start=data_start,
        data_end=data_end,
        return_source=candidate.return_source,
    )


def _load_technical_market_data(
    symbol: str,
    start: str | None,
    end: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and fail-closed validate the causal OHLCV/funding inputs for one symbol."""
    perp_p = ohlcv_path(symbol, "1h")
    fund_p = funding_path(symbol)
    for path, name in [(perp_p, "perp_ohlcv"), (fund_p, "funding")]:
        if not path.exists():
            raise DataIntegrityError(f"{name} data missing for {symbol}: {path}")

    bars = load_ohlcv_4h(perp_p, start=start, end=end)
    if len(bars) < 2:
        raise DataIntegrityError(f"bars data has fewer than 2 bars for {symbol}")
    period = bars.index[1] - bars.index[0]
    window_end = bars.index[-1] + period
    funding = load_funding_rates(fund_p)
    funding = funding[(funding.index >= bars.index[0]) & (funding.index < window_end)]
    if len(funding) == 0:
        raise DataIntegrityError(f"no settled funding events in window for {symbol}")
    return bars, funding


def _run_evaluation(
    frame: pd.DataFrame,
    funding: pd.Series,
    symbol: str,
    candidate: TechnicalCandidate,
    costs: CostModel,
    *,
    unseal_holdout: bool,
    identity: CandidateIdentity,
    log_run: bool,
) -> tuple[EvaluationReport, dict[str, object]]:
    result = run_technical_expert_backtest(
        frame, candidate, costs, funding,
        initial_equity=_INITIAL_EQUITY, signal_delay_bars=_BASE_DELAY_BARS,
    )
    metrics = compute_metrics(result.equity, result.trades)
    _logger.info(
        "[EVAL] technical_expert candidate=%s cagr=%.4f mdd=%.4f sharpe=%.3f trades=%d",
        candidate.return_source, metrics.cagr, metrics.mdd, metrics.sharpe,
        metrics.trade_count,
    )

    observation_gate = compute_equity_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)

    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_result = run_technical_expert_backtest(
        frame, candidate, stressed_costs, funding,
        initial_equity=_INITIAL_EQUITY, signal_delay_bars=_STRESS_DELAY_BARS,
    )
    stress_metrics = compute_metrics(stress_result.equity, stress_result.trades)
    stress_gate = compute_equity_reliability_gate(
        stress_result.equity,
        len(stress_result.trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )

    holdout_gate = None
    if unseal_holdout and result.equity.index[-1] > HOLDOUT_CUTOFF:
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
    promotion = dataclasses.replace(promotion, candidate=identity)
    _logger.info(
        "[EVAL] promotion status=%s observation=%s fold_gate=%s stress=%s holdout=%s",
        promotion.status, promotion.observation_verdict, promotion.fold_gate_pass,
        promotion.stress_verdict, promotion.holdout_verdict,
    )
    metrics_snapshot: dict[str, object] = {
        "cagr": round(metrics.cagr, 6),
        "mdd": round(metrics.mdd, 6),
        "sharpe": round(metrics.sharpe, 6),
        "trade_count": metrics.trade_count,
        "observation_lcb90": round(observation_gate.lcb90_cagr, 6),
        "fold_max_contribution": round(fold_distribution.max_period_contribution, 6),
        "stress_verdict": stress_gate.verdict,
    }
    rec: dict[str, object] | None = None
    if log_run:
        rec = record_technical_expert_run(
            symbol=symbol,
            candidate_id=candidate.candidate_id,
            return_source=candidate.return_source,
            signal_delay_bars=_BASE_DELAY_BARS,
            costs=costs,
            result=result,
            metrics=metrics,
            start=str(frame.index[0]),
            end=str(frame.index[-1]),
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
            promotion=promotion,
            candidate=promotion.candidate,
        )
    report = EvaluationReport(
        status="PASS",
        result=result,
        metrics=metrics,
        observation=observation_gate,
        fold_distribution=fold_distribution,
        stress=stress_gate,
        holdout=holdout_gate,
        promotion=promotion,
        record=rec,
    )
    return report, metrics_snapshot


def run_technical_expert_evaluation(
    request: TechnicalExpertEvaluationRequest,
    *,
    log_run: bool | None = None,
) -> EvaluationReport:
    """Execute one sealed technical-expert candidate screen.

    Returns an ``EvaluationReport`` with ``status == "PENDING"`` for any
    fail-closed early return (sealed-end violation or missing/integrity-invalid
    data). A malformed or unknown candidate id is rejected with ``ValueError``
    before any data execution. A full evaluation runs the frozen candidate under
    base costs and under stressed costs with one extra decision bar, composes
    the unchanged promotion verdict, writes exactly one provenance record when
    logging is requested, and appends anti-pattern evidence on rejection. It
    never mutates the expert catalog.
    """
    try:
        end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    except RuntimeError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    candidate = resolve_technical_candidate(request.candidate_id)
    try:
        frame, funding = _load_technical_market_data(request.symbol, request.start, end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    try:
        hashes = _data_hashes(request.symbol)
    except DataIntegrityError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    if end is None:
        period = frame.index[1] - frame.index[0]
        end = frame.index[-1] + period

    costs = CostModel()
    current_code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
    observation_end = str(end)
    candidate_id = _candidate_id(
        return_source=candidate.return_source,
        symbol=request.symbol,
        observation_end=observation_end,
        costs=costs,
        data_hashes=hashes,
        code_hash=current_code_hash,
    )
    identity = _candidate_identity(
        candidate=candidate,
        symbol=request.symbol,
        observation_end=observation_end,
        costs=costs,
        data_hashes=hashes,
        code_hash=current_code_hash,
        data_start=str(frame.index[0]),
        data_end=str(frame.index[-1]),
    )
    _logger.info(
        "[EVAL] candidate identity id=%s return_source=%s symbol=%s window_start=%s window_end=%s",
        candidate_id, candidate.return_source, request.symbol,
        frame.index[0], frame.index[-1],
    )

    should_log = request.log_run if log_run is None else log_run
    try:
        report, _ = _run_evaluation(
            frame,
            funding,
            request.symbol,
            candidate,
            costs,
            unseal_holdout=request.unseal_holdout,
            identity=identity,
            log_run=should_log,
        )
    except DataIntegrityError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    rec = report.record
    if rec is not None:
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", rec["git_sha"], rec["git_dirty"])
    return report


def _pending_report(request: TechnicalExpertEvaluationRequest) -> EvaluationReport:
    empty_equity = pd.Series(dtype="float64", name="equity")
    empty_trades = pd.DataFrame(columns=[
        "entry_bar", "exit_bar", "entry_price", "exit_price", "qty", "reason",
        "pnl", "return_pct", "funding_pnl", "side",
    ])
    empty_signals = pd.DataFrame(columns=["target"])
    pending = ReliabilityGateResult(
        lcb90_cagr=0.0, lcb95_cagr=0.0, p_negative=0.0,
        point_cagr=0.0, t_stat=0.0, trade_count=0,
        block_size_used=1, verdict="PENDING",
    )
    fold = FoldDistributionResult(
        n_folds=0, median_fold_cagr=0.0, worst_fold_cagr=0.0,
        median_fold_calmar=0.0, max_period_contribution=0.0, gate_pass=True,
    )
    empty_metrics = Metrics(
        cagr=0.0, mdd=0.0, sharpe=0.0, sortino=0.0, calmar=0.0,
        profit_factor=0.0, expectancy=0.0, win_rate=0.0, payoff_ratio=0.0,
        trade_count=0, exposure=0.0, turnover=0.0, trades_per_year={},
    )
    promotion = PromotionResult(
        status="REJECTED",
        observation_verdict="PENDING",
        fold_gate_pass=True,
        stress_verdict="PENDING",
        holdout_verdict=None,
    )
    return EvaluationReport(
        status="PENDING",
        result=BacktestResult(equity=empty_equity, trades=empty_trades, signals=empty_signals),
        metrics=empty_metrics,
        observation=pending,
        fold_distribution=fold,
        stress=pending,
        holdout=None,
        promotion=promotion,
    )
