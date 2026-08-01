from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.common.config import funding_path, metrics_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import BacktestResult
from src.research.contracts import (
    CostModel,
    EvaluationReport,
    OIDeleveragingEvaluationRequest,
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
from src.research.oi_deleveraging.backtest import run_open_interest_deleveraging_screen
from src.research.oi_deleveraging.contracts import OIDeleveragingMarketData
from src.research.oi_deleveraging.market_data import load_oi_deleveraging_market_data
from src.research.provenance.code_manifest import compute_code_hash
from src.research.provenance.results import record_oi_deleveraging_run

_logger = logging.getLogger("OIDeleveragingBacktestRunner")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0

_HYPOTHESIS_ID = "open_interest_deleveraging_v1"
_RETURN_SOURCE = "open_interest_deleveraging_v1"

_SOURCE_FILES = ("perp_ohlcv", "funding", "metrics")

_BASE_DELAY_BARS = 1
_STRESS_DELAY_BARS = 2


def _source_paths(symbol: str) -> dict[str, str]:
    return {
        "perp_ohlcv": str(ohlcv_path(symbol, "1h")),
        "funding": str(funding_path(symbol)),
        "metrics": str(metrics_path(symbol)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_hashes(symbol: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in _source_paths(symbol).items():
        p = Path(path)
        if not p.exists():
            raise DataIntegrityError(f"{name} data missing for {symbol}: {p}")
        hashes[name] = _file_sha256(p)
    return hashes


def _combined_data_hash(data_hashes: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(data_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _candidate_id(
    *,
    symbol: str,
    observation_end: str,
    costs: CostModel,
    data_hashes: dict[str, str],
    code_hash: str,
) -> str:
    payload = {
        "hypothesis_id": _HYPOTHESIS_ID,
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
    symbol: str,
    observation_end: str,
    costs: CostModel,
    data_hashes: dict[str, str],
    code_hash: str,
    data_start: str,
    data_end: str,
) -> CandidateIdentity:
    return CandidateIdentity(
        hypothesis_id=_HYPOTHESIS_ID,
        code_hash=code_hash,
        parameters={
            "data_hashes": data_hashes,
            "costs": dataclasses.asdict(costs),
            "signal_delay_bars": _BASE_DELAY_BARS,
            "return_source": _RETURN_SOURCE,
        },
        data_start=str(data_start),
        data_end=str(data_end),
        return_source=_RETURN_SOURCE,
    )


def _run_evaluation(
    market_data: OIDeleveragingMarketData,
    costs: CostModel,
    *,
    unseal_holdout: bool,
    candidate: CandidateIdentity,
    log_run: bool,
) -> tuple[EvaluationReport, dict[str, object]]:
    result = run_open_interest_deleveraging_screen(
        market_data, costs, signal_delay_bars=_BASE_DELAY_BARS,
    )
    metrics = compute_metrics(result.equity, result.trades)
    _logger.info(
        "[EVAL] oi_deleveraging cagr=%.4f mdd=%.4f sharpe=%.3f trades=%d",
        metrics.cagr, metrics.mdd, metrics.sharpe, metrics.trade_count,
    )

    observation_gate = compute_equity_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)

    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_result = run_open_interest_deleveraging_screen(
        market_data, stressed_costs, signal_delay_bars=_STRESS_DELAY_BARS,
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
    promotion = dataclasses.replace(promotion, candidate=candidate)
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
        rec = record_oi_deleveraging_run(
            symbol=market_data.symbol,
            signal_delay_bars=_BASE_DELAY_BARS,
            costs=costs,
            result=result,
            metrics=metrics,
            start=str(market_data.bars.index[0]),
            end=str(market_data.bars.index[-1]),
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


def run_oi_deleveraging_evaluation(
    request: OIDeleveragingEvaluationRequest,
    *,
    log_run: bool = True,
) -> EvaluationReport:
    """Execute one sealed open-interest deleveraging research evaluation.

    Returns an ``EvaluationReport`` with ``status == "PENDING"`` for any
    fail-closed early return (sealed-end violation or missing/integrity-invalid
    data). A full evaluation runs the fixed-sign screen under base costs and
    under stressed costs with one extra decision bar, composes the unchanged
    promotion verdict, writes exactly one provenance record when logging is
    requested, and appends anti-pattern evidence on rejection. It never
    registers a catalog blueprint.
    """
    try:
        end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    except RuntimeError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    try:
        market_data = load_oi_deleveraging_market_data(request.symbol, request.start, end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    try:
        hashes = _data_hashes(request.symbol)
    except DataIntegrityError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    if end is None:
        period = market_data.bars.index[1] - market_data.bars.index[0]
        end = market_data.bars.index[-1] + period

    costs = CostModel()
    current_code_hash = compute_code_hash()
    observation_end = str(end)
    candidate_id = _candidate_id(
        symbol=request.symbol,
        observation_end=observation_end,
        costs=costs,
        data_hashes=hashes,
        code_hash=current_code_hash,
    )
    identity = _candidate_identity(
        symbol=request.symbol,
        observation_end=observation_end,
        costs=costs,
        data_hashes=hashes,
        code_hash=current_code_hash,
        data_start=str(market_data.bars.index[0]),
        data_end=str(market_data.bars.index[-1]),
    )
    _logger.info(
        "[EVAL] candidate identity id=%s symbol=%s window_start=%s window_end=%s",
        candidate_id, request.symbol,
        market_data.bars.index[0], market_data.bars.index[-1],
    )

    should_log = log_run and request.log_run
    report, _ = _run_evaluation(
        market_data,
        costs,
        unseal_holdout=request.unseal_holdout,
        candidate=identity,
        log_run=should_log,
    )
    rec = report.record
    if rec is not None:
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", rec["git_sha"], rec["git_dirty"])
    return report


def _pending_report(request: OIDeleveragingEvaluationRequest) -> EvaluationReport:
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
