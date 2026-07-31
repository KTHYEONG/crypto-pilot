from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import cast

import pandas as pd

from src.cli.run_backtest import HOLDOUT_CUTOFF
from src.core.config import borrow_path, funding_path, ohlcv_path, spot_ohlcv_path
from src.core.types import CarryCostModel, CashCarrySpec
from src.data.carry_data import CarryMarketData, load_carry_market_data
from src.data.loader import DataIntegrityError
from src.data.spot_collector import load_spot_manifest
from src.engine.cash_carry_backtest import run_cash_carry_backtest
from src.engine.results_log import record_cash_carry_run
from src.validation.candidate_promotion import (
    CandidateIdentity,
    PromotionResult,
    compose_promotion_verdict,
)
from src.validation.candidate_registry import (
    CandidateRegistration,
    compute_candidate_id,
)
from src.validation.metrics import compute_metrics
from src.validation.reliability_gate import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    split_holdout_segment,
)
from src.validation.research_memory import record_rejected_candidate

_logger = logging.getLogger("CashCarryBacktestRunner")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0

_HYPOTHESIS_ID = "cash_and_carry_basis"
_RETURN_SOURCE = "spot_perp_funding_carry"

_CARRY_MODULE_PATHS = (
    Path("src/cli/run_cash_carry_backtest.py"),
    Path("src/data/carry_data.py"),
    Path("src/data/loader.py"),
    Path("src/strategy/cash_carry.py"),
    Path("src/engine/cash_carry_backtest.py"),
)

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


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in _CARRY_MODULE_PATHS:
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()


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


def _resolve_end(end: str | None, unseal_holdout: bool) -> pd.Timestamp | None:
    if unseal_holdout:
        if end is None:
            return None
        return pd.Timestamp(end, tz="UTC")
    if end is None:
        return HOLDOUT_CUTOFF
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts > HOLDOUT_CUTOFF:
        raise RuntimeError(
            f"Holdout sealed: --end {end} > {HOLDOUT_CUTOFF}. "
            "Pass --unseal-holdout to override."
        )
    return end_ts


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
) -> tuple[dict[str, object] | None, dict[str, object], PromotionResult]:
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
    return rec, metrics_snapshot, promotion


def _record_anti_pattern(
    candidate_id: str,
    data_hash: str,
    code_hash: str,
    promotion: PromotionResult,
    metrics_snapshot: dict[str, object],
    run_log_reference: str,
) -> None:
    failed_gates: list[str] = []
    if promotion.observation_verdict != "PASS":
        failed_gates.append("observation")
    if not promotion.fold_gate_pass:
        failed_gates.append("fold")
    if promotion.stress_verdict != "PASS":
        failed_gates.append("stress")
    if promotion.holdout_verdict is not None and promotion.holdout_verdict != "PASS":
        failed_gates.append("holdout")
    record_rejected_candidate(
        candidate_id=candidate_id,
        data_hash=data_hash,
        code_hash=code_hash,
        hypothesis_id=_HYPOTHESIS_ID,
        failed_gates=failed_gates,
        reason=f"promotion status={promotion.status}",
        metrics=metrics_snapshot,
        run_log_reference=run_log_reference,
    )


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


def _run(args: argparse.Namespace) -> None:
    try:
        end = _resolve_end(args.end, args.unseal_holdout)
    except RuntimeError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", args.symbol, exc)
        return
    try:
        start = args.start if args.start is not None else _borrow_start(args.symbol)
        market_data = load_carry_market_data(args.symbol, start, end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.info(
            "[EVAL] run status=PENDING symbol=%s reason=%s",
            args.symbol, exc,
        )
        return
    try:
        hashes = _data_hashes(args.symbol)
    except DataIntegrityError as exc:
        _logger.info("[EVAL] run status=PENDING symbol=%s reason=%s", args.symbol, exc)
        return
    if end is None:
        period = market_data.spot.index[1] - market_data.spot.index[0]
        end = market_data.spot.index[-1] + period
    spec = CashCarrySpec(symbol=args.symbol)
    costs = CarryCostModel()
    current_code_hash = _code_hash()
    registration = _ephemeral_registration(
        symbol=args.symbol,
        observation_end=str(end),
        spec=spec,
        costs=costs,
        data_hashes=hashes,
        manifest=_manifest_snapshot(args.symbol),
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
    rec, metrics_snapshot, promotion = _run_evaluation(
        market_data, spec, costs, args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        candidate=identity,
        log_run=not args.no_log_run,
    )
    if rec is not None:
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", rec["git_sha"], rec["git_dirty"])
    if promotion.status == "REJECTED":
        run_ref = str(rec["ts"]) if rec is not None else "(not logged)"
        _record_anti_pattern(
            candidate_id=registration.candidate_id,
            data_hash=_combined_data_hash(hashes),
            code_hash=current_code_hash,
            promotion=promotion,
            metrics_snapshot=metrics_snapshot,
            run_log_reference=run_ref,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sealed cash-and-carry research runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a sealed cash-and-carry evaluation")
    run_p.add_argument("--symbol", default="BTCUSDT")
    run_p.add_argument("--start", default=None)
    run_p.add_argument("--end", default=None)
    run_p.add_argument("--initial-equity", type=float, default=10_000.0)
    run_p.add_argument("--unseal-holdout", action="store_true", default=False)
    run_p.add_argument("--no-log-run", action="store_true", default=False)
    run_p.set_defaults(func=_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    configured_level = getattr(
        logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO,
    )
    logging.basicConfig(level=configured_level)
    main()
