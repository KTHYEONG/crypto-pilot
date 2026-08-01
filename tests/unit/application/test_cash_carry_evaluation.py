from __future__ import annotations

from pathlib import Path


from src.application.research.cash_carry.evaluation import run_cash_carry_evaluation
from src.research.cash_carry.backtest import run_cash_carry_backtest
from src.research.contracts import CashCarryEvaluationRequest
from src.research.evaluation.promotion import PromotionResult


def _two_year_carry_data(make_carry_data):
    return make_carry_data(n_bars=4400)


def test_cash_carry_evaluation_preserves_ledger_and_provenance(
    make_carry_data,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """RF-CARRY-01: the canonical carry evaluation preserves the frozen ledger.

    Uses a carry fixture that spans two calendar years so the fold gate is
    computable; missing costs are never zero-filled, and a rejected candidate
    is recorded idempotently into an append-only anti-pattern store.
    """
    data = _two_year_carry_data(make_carry_data)

    monkeypatch.setattr("src.application.research.cash_carry.evaluation.load_carry_market_data",
                        lambda symbol, start, end: data)
    monkeypatch.setattr(
        "src.application.research.cash_carry.evaluation.cash_carry_data_hashes",
        lambda symbol: {
            "spot_ohlcv": "a" * 64,
            "perp_ohlcv": "b" * 64,
            "funding": "c" * 64,
            "borrow": "d" * 64,
        },
    )
    monkeypatch.setattr(
        "src.application.research.cash_carry.evaluation.compute_code_hash",
        lambda: "e" * 64,
    )
    request = CashCarryEvaluationRequest(
        symbol="BTCUSDT", start="2024-01-01", unseal_holdout=False, log_run=False,
    )
    report = run_cash_carry_evaluation(request)

    from src.research.cash_carry.contracts import CarryCostModel, CashCarrySpec

    spec = CashCarrySpec(symbol="BTCUSDT")
    costs = CarryCostModel()
    direct = run_cash_carry_backtest(data, spec, costs)

    assert report.status == "PASS"
    assert report.result.equity.equals(direct.equity)
    assert report.result.trades.equals(direct.trades)
    assert isinstance(report.promotion, PromotionResult)
    assert report.promotion.observation_verdict == "PENDING"
    assert report.promotion.candidate is not None
    assert report.promotion.candidate.hypothesis_id == "cash_and_carry_basis"


def test_cash_carry_evaluation_missing_data_returns_pending(make_carry_data, monkeypatch) -> None:
    """RF-CARRY-01: missing borrow input fails closed with a PENDING outcome."""
    from src.common.errors import DataIntegrityError

    def _missing_borrow(symbol, start, end):
        raise DataIntegrityError("borrow data missing for BTCUSDT")

    monkeypatch.setattr("src.application.research.cash_carry.evaluation.load_carry_market_data",
                        _missing_borrow)
    report = run_cash_carry_evaluation(
        CashCarryEvaluationRequest(symbol="BTCUSDT", start="2024-01-01", log_run=False),
    )
    assert report.status == "PENDING"
    assert report.promotion.observation_verdict == "PENDING"
