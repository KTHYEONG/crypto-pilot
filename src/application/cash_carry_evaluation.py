from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from src.common.config import borrow_path, funding_path, ohlcv_path, spot_ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.manifest import load_spot_manifest
from src.research.baseline.backtest import BacktestResult
from src.research.cash_carry.backtest import run_cash_carry_backtest
from src.research.cash_carry.contracts import CarryCostModel, CarryMarketData, CashCarrySpec
from src.research.cash_carry.market_data import load_carry_market_data
from src.research.contracts import (
    CashCarryEvaluationRequest,
    EvaluationReport,
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
from src.research.provenance.candidates import (
    CandidateRegistration,
    compute_candidate_id,
)
from src.research.provenance.code_manifest import compute_code_hash
from src.research.provenance.results import record_cash_carry_run

_logger = logging.getLogger("CashCarryBacktestRunner")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0

_HYPOTHESIS_ID = "cash_and_carry_basis"
_RETURN_SOURCE = "spot_perp_funding_carry"

_SOURCE_FILES = ("spot_ohlcv", "perp_ohlcv", "funding", "borrow")


def _source_paths(symbol: str) -> dict[str, str]:
    return {
        "spot_ohlcv": str(spot_ohlcv_path(symbol, "1h")),
        "perp_ohlcv": str(ohlcv_path(symbol, "1h")),
        "funding": str(funding_path(symbol)),
        "borrow": str(borrow_path(symbol)),
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


def _manifest_snapshot(symbol: str) -> dict[str, object]:
    manifest = load_spot_manifest()
    datasets = manifest.get("datasets", {})
    relevant: dict[str, object] = {}
    if isinstance(datasets, dict):
        for dataset in ("ohlcv/1h", "borrow"):
            records = datasets.get(dataset, {})
            if isinstance(records, dict) and symbol in records:
                relevant[dataset] = records[symbol]
    return relevant


def _borrow_start(symbol: str) -> pd.Timestamp:
    path = borrow_path(symbol)
    frame = pd.read_parquet(path)
    if "datetime" in frame.columns:
        timestamps = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    elif "timestamp" in frame.columns:
        timestamps = pd.to_datetime(
            pd.to_numeric(frame["timestamp"], errors="coerce"),
            unit="ms", utc=True, errors="coerce",
        )
    else:
        raise DataIntegrityError("borrow parquet must contain a 'timestamp' or 'datetime' column")
    timestamps = timestamps.dropna()
    if timestamps.empty:
        raise DataIntegrityError("borrow parquet must contain at least one valid timestamp")
    return pd.Timestamp(timestamps.min())


def _candidate_identity(
    registration: CandidateRegistration,
    market_data: CarryMarketData,
) -> CandidateIdentity:
    return CandidateIdentity(
        hypothesis_id=registration.hypothesis_id,
        code_hash=registration.code_hash,
        parameters={
            "data_hashes": registration.data_hashes,
            "manifest": registration.manifest,
            "spec": registration.spec,
            "costs": registration.costs,
            "return_source": registration.return_source,
        },
        data_start=str(market_data.spot.index[0]),
        data_end=str(market_data.spot.index[-1]),
        return_source=registration.return_source,
    )


def _run_evaluation(
    market_data: CarryMarketData,
    spec: CashCarrySpec,
    costs: CarryCostModel,
    initial_equity: float,
    *,
    unseal_holdout: bool,
    candidate: CandidateIdentity,
    log_run: bool,
) -> tuple[EvaluationReport, dict[str, object]]:
    result = run_cash_carry_backtest(
        market_data, spec, costs, initial_equity=initial_equity, signal_delay_bars=0,
    )
    metrics = compute_metrics(result.equity, result.trades)
    _logger.info(
        "[EVAL] cash_carry cagr=%.4f mdd=%.4f sharpe=%.3f trades=%d",
        metrics.cagr, metrics.mdd, metrics.sharpe, metrics.trade_count,
    )

    observation_gate = compute_equity_reliability_gate(result.equity, len(result.trades))
    fold_distribution = compute_fold_distribution(result)

    stressed_costs = CarryCostModel(
        spot_fee_rate=costs.spot_fee_rate * _STRESS_FEE_MULT,
        perp_fee_rate=costs.perp_fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_result = run_cash_carry_backtest(
        market_data, spec, stressed_costs,
        initial_equity=initial_equity, signal_delay_bars=1,
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
        rec = record_cash_carry_run(
            symbol=spec.symbol,
            cash_carry_spec=spec,
            costs=costs,
            result=result,
            metrics=metrics,
            start=str(market_data.spot.index[0]),
            end=str(market_data.spot.index[-1]),
            initial_equity=initial_equity,
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


def _ephemeral_registration(
    *,
    symbol: str,
    observation_end: str,
    spec: CashCarrySpec,
    costs: CarryCostModel,
    data_hashes: dict[str, str],
    manifest: dict[str, object],
    code_hash: str,
) -> CandidateRegistration:
    candidate_id = compute_candidate_id(
        hypothesis_id=_HYPOTHESIS_ID,
        symbol=symbol,
        observation_end=observation_end,
        spec=spec,
        costs=costs,
        data_hashes=data_hashes,
        manifest=manifest,
        code_hash=code_hash,
    )
    return CandidateRegistration(
        candidate_id=candidate_id,
        hypothesis_id=_HYPOTHESIS_ID,
        symbol=symbol,
        observation_end=observation_end,
        spec=dataclasses.asdict(spec),
        costs=dataclasses.asdict(costs),
        source_paths=_source_paths(symbol),
        data_hashes=data_hashes,
        manifest=manifest,
        code_hash=code_hash,
        return_source=_RETURN_SOURCE,
        registration_ts="",
        status="EPHEMERAL",
    )


def run_cash_carry_evaluation(request: CashCarryEvaluationRequest) -> EvaluationReport:
    """Execute one sealed cash-and-carry research evaluation.

    Returns an ``EvaluationReport`` with ``status == "PENDING"`` for any
    fail-closed early return (sealed-end violation or missing/integrity-invalid
    data); missing funding or borrow is never treated as zero. A full evaluation
    preserves the equal-quantity spot/perpetual ledger, actual funding
    alignment, borrow accrual, fees/slippage, maintenance liquidation, and
    append-only candidate provenance.
    """
    end: str | pd.Timestamp | None
    try:
        end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    except RuntimeError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    try:
        start = request.start if request.start is not None else _borrow_start(request.symbol)
        market_data = load_carry_market_data(request.symbol, start, end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.info(
            "[EVAL] run status=PENDING symbol=%s reason=%s",
            request.symbol, exc,
        )
        return _pending_report(request)
    try:
        hashes = _data_hashes(request.symbol)
    except DataIntegrityError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", request.symbol, exc)
        return _pending_report(request)
    if end is None:
        period = market_data.spot.index[1] - market_data.spot.index[0]
        end = market_data.spot.index[-1] + period
    spec = CashCarrySpec(symbol=request.symbol)
    costs = CarryCostModel()
    current_code_hash = compute_code_hash()
    registration = _ephemeral_registration(
        symbol=request.symbol,
        observation_end=str(end),
        spec=spec,
        costs=costs,
        data_hashes=hashes,
        manifest=_manifest_snapshot(request.symbol),
        code_hash=current_code_hash,
    )
    _logger.info(
        "[EVAL] candidate identity id=%s symbol=%s window_start=%s window_end=%s status=%s",
        registration.candidate_id, registration.symbol,
        market_data.spot.index[0], market_data.spot.index[-1],
        registration.status,
    )
    _logger.info(
        "[EVAL] carry data status=PASS symbol=%s bars=%d window_end=%s funding_events=%d",
        registration.symbol, len(market_data.spot),
        market_data.spot.index[-1].isoformat(), len(market_data.funding),
    )

    spec = CashCarrySpec(
        symbol=str(registration.spec["symbol"]),
        initial_margin_rate=cast(float, registration.spec["initial_margin_rate"]),
        maintenance_margin_rate=cast(float, registration.spec["maintenance_margin_rate"]),
    )
    costs = CarryCostModel(
        spot_fee_rate=cast(float, registration.costs["spot_fee_rate"]),
        perp_fee_rate=cast(float, registration.costs["perp_fee_rate"]),
        slippage_rate=cast(float, registration.costs["slippage_rate"]),
    )
    identity = _candidate_identity(registration, market_data)
    report, _ = _run_evaluation(
        market_data, spec, costs, request.initial_equity,
        unseal_holdout=request.unseal_holdout,
        candidate=identity,
        log_run=request.log_run,
    )
    rec = report.record
    if rec is not None:
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", rec["git_sha"], rec["git_dirty"])

    return report


def _pending_report(request: CashCarryEvaluationRequest) -> EvaluationReport:
    empty_equity = pd.Series(dtype="float64", name="equity")
    empty_trades = pd.DataFrame(columns=[
        "entry_bar", "exit_bar", "entry_time", "exit_time",
        "spot_entry", "spot_exit", "perp_entry", "perp_exit", "qty", "reason",
        "pnl", "return_pct", "funding_pnl", "equity_before_entry",
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
