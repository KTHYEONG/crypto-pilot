from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.cli.run_backtest import HOLDOUT_CUTOFF, _load_funding_rates
from src.core.config import borrow_path, funding_path, ohlcv_path, spot_ohlcv_path
from src.core.types import CarryCostModel, CashCarrySpec
from src.data.carry_data import CarryMarketData, validate_carry_market_data
from src.data.loader import DataIntegrityError, load_ohlcv_4h
from src.engine.cash_carry_backtest import run_cash_carry_backtest
from src.engine.results_log import record_cash_carry_run
from src.validation.candidate_promotion import CandidateIdentity, compose_promotion_verdict
from src.validation.metrics import compute_metrics
from src.validation.reliability_gate import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    split_holdout_segment,
)

_logger = logging.getLogger("CashCarryBacktestRunner")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0

_CARRY_MODULE_PATHS = (
    Path("src/data/carry_data.py"),
    Path("src/strategy/cash_carry.py"),
    Path("src/engine/cash_carry_backtest.py"),
)


def _load_borrow_rates(path: str) -> pd.Series:
    """Load a per-bar quote-cash borrow parquet into a monotonic rate Series."""
    p = Path(path)
    if not p.exists():
        raise DataIntegrityError(f"borrow path does not exist: {path}")
    df = pd.read_parquet(p)
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    else:
        raise DataIntegrityError("borrow parquet must contain a 'datetime' or 'timestamp' column")
    if "borrow_rate" not in df.columns:
        raise DataIntegrityError("borrow parquet must contain a 'borrow_rate' column")
    rates = pd.to_numeric(df["borrow_rate"], errors="coerce")
    series = pd.Series(rates.to_numpy(dtype="float64"), index=pd.DatetimeIndex(ts))
    return series[series.index.notna()].sort_index()


def load_carry_market_data(
    symbol: str,
    start: str | None,
    end: str | pd.Timestamp | None,
) -> CarryMarketData:
    """Load and fail-closed validate the four cash-and-carry inputs for one symbol."""
    spot_p = spot_ohlcv_path(symbol, "1h")
    perp_p = ohlcv_path(symbol, "1h")
    fund_p = funding_path(symbol)
    borrow_p = borrow_path(symbol)
    for path, name in [(spot_p, "spot"), (perp_p, "perp"), (fund_p, "funding"), (borrow_p, "borrow")]:
        if not path.exists():
            raise DataIntegrityError(f"{name} data missing for {symbol}: {path}")

    spot = load_ohlcv_4h(spot_p, start=start, end=end)
    perp = load_ohlcv_4h(perp_p, start=start, end=end)
    period = spot.index[1] - spot.index[0] if len(spot) >= 2 else pd.Timedelta(hours=4)
    window_end = spot.index[-1] + period
    funding = _load_funding_rates(str(fund_p))
    funding = funding[(funding.index >= spot.index[0]) & (funding.index < window_end)]
    borrow = _load_borrow_rates(str(borrow_p))
    borrow = borrow[(borrow.index >= spot.index[0]) & (borrow.index < window_end)]
    borrow = borrow.reindex(spot.index)

    market_data = CarryMarketData(symbol=symbol, spot=spot, perp=perp, funding=funding, borrow=borrow)
    validate_carry_market_data(market_data)
    return market_data


def _data_hash(market_data: CarryMarketData) -> str:
    funding = pd.to_numeric(market_data.funding, errors="coerce").to_numpy()
    borrow = pd.to_numeric(market_data.borrow, errors="coerce").to_numpy()
    canonical = (
        market_data.symbol,
        str(market_data.spot.index[0]),
        str(market_data.spot.index[-1]),
        len(market_data.spot.index),
        tuple(round(float(x), 8) for x in funding),
        tuple(round(float(x), 8) for x in borrow),
    )
    return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in _CARRY_MODULE_PATHS:
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()


def _candidate_identity(
    market_data: CarryMarketData,
    spec: CashCarrySpec,
    costs: CarryCostModel,
) -> CandidateIdentity:
    return CandidateIdentity(
        hypothesis_id="cash_and_carry_basis",
        code_hash=_code_hash(),
        parameters={
            "data_hash": _data_hash(market_data),
            "spot_source": "spot/ohlcv/1h",
            "borrow_source": "spot/borrow",
            "margin_model": {
                "initial_margin_rate": spec.initial_margin_rate,
                "maintenance_margin_rate": spec.maintenance_margin_rate,
            },
            "costs": {
                "spot_fee_rate": costs.spot_fee_rate,
                "perp_fee_rate": costs.perp_fee_rate,
                "slippage_rate": costs.slippage_rate,
            },
        },
        data_start=str(market_data.spot.index[0]),
        data_end=str(market_data.spot.index[-1]),
        return_source="spot_perp_funding_carry",
    )


def _run_evaluation(
    market_data: CarryMarketData,
    spec: CashCarrySpec,
    costs: CarryCostModel,
    initial_equity: float,
    *,
    log_run: bool,
) -> dict[str, object] | None:
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
    if result.equity.index[-1] > HOLDOUT_CUTOFF:
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
    promotion = dataclasses.replace(promotion, candidate=_candidate_identity(market_data, spec, costs))
    _logger.info(
        "[EVAL] promotion status=%s observation=%s fold_gate=%s stress=%s holdout=%s",
        promotion.status, promotion.observation_verdict, promotion.fold_gate_pass,
        promotion.stress_verdict, promotion.holdout_verdict,
    )
    if not log_run:
        return None
    return record_cash_carry_run(
        symbol=spec.symbol,
        cash_carry_spec=spec,
        costs=costs,
        result=result,
        metrics=metrics,
        start=None,
        end=str(market_data.spot.index[-1]),
        initial_equity=initial_equity,
        observation_gate=observation_gate,
        fold_distribution=fold_distribution,
        stress_gate=stress_gate,
        holdout_gate=holdout_gate,
        promotion=promotion,
        candidate=promotion.candidate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sealed cash-and-carry research backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--unseal-holdout", action="store_true", default=False)
    parser.add_argument("--no-log-run", action="store_true", default=False)
    args = parser.parse_args()

    end: str | pd.Timestamp | None
    if args.unseal_holdout:
        end = args.end
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

    try:
        market_data = load_carry_market_data(args.symbol, args.start, end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.info(
            "[EVAL] carry data status=PENDING symbol=%s reason=%s",
            args.symbol, exc,
        )
        return
    _logger.info(
        "[EVAL] carry data status=PASS symbol=%s bars=%d window_end=%s funding_events=%d",
        args.symbol, len(market_data.spot), market_data.spot.index[-1].isoformat(),
        len(market_data.funding),
    )

    spec = CashCarrySpec(symbol=args.symbol)
    costs = CarryCostModel()
    rec = _run_evaluation(
        market_data, spec, costs, args.initial_equity, log_run=not args.no_log_run,
    )
    if rec is not None:
        _logger.info("[EVAL] run logged: git_sha=%s dirty=%s", rec["git_sha"], rec["git_dirty"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
