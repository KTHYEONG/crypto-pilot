from __future__ import annotations

from pathlib import Path

from src.application.research.oi_deleveraging.evaluation import run_oi_deleveraging_evaluation
from src.research.contracts import OIDeleveragingEvaluationRequest
from src.research.evaluation.promotion import PromotionResult
from src.research.expert_portfolio.catalog import default_catalog


def _two_year_oi_data(make_oi_market_data):
    return make_oi_market_data(n_bars=4400)


def test_oi_deleveraging_failure_is_not_registered(
    make_oi_market_data,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """FD-06: a failed candidate cannot mutate default_catalog or become ACTIVE.

    The sealed screen runs through the unchanged reliability/fold/stress gates,
    the promotion is REJECTED (observation PENDING on a zero-trade window), and
    the rejection is appended idempotently to the anti-pattern store under the
    immutable OI return source.
    """
    data = _two_year_oi_data(make_oi_market_data)

    monkeypatch.setattr(
        "src.application.research.oi_deleveraging.evaluation.load_oi_deleveraging_market_data",
        lambda symbol, start, end: data,
    )
    monkeypatch.setattr(
        "src.application.research.oi_deleveraging.evaluation.oi_deleveraging_data_hashes",
        lambda symbol: {
            "perp_ohlcv": "a" * 64,
            "funding": "b" * 64,
            "metrics": "c" * 64,
        },
    )
    request = OIDeleveragingEvaluationRequest(
        symbol="BTCUSDT", start="2024-01-01", log_run=False,
    )
    report = run_oi_deleveraging_evaluation(request, log_run=False)

    assert report.status == "PASS"
    assert isinstance(report.promotion, PromotionResult)
    assert report.promotion.status == "REJECTED"
    assert report.promotion.observation_verdict == "PENDING"
    assert report.promotion.candidate is not None
    assert report.promotion.candidate.hypothesis_id == "open_interest_deleveraging_v1"

    assert "open_interest_deleveraging_v1" not in default_catalog().blueprints


def test_oi_deleveraging_missing_data_returns_pending(monkeypatch) -> None:
    """FD-06: missing causal inputs fail closed with a PENDING outcome."""
    from src.common.errors import DataIntegrityError

    def _missing(symbol, start, end):
        raise DataIntegrityError("metrics data missing for BTCUSDT")

    monkeypatch.setattr(
        "src.application.research.oi_deleveraging.evaluation.load_oi_deleveraging_market_data",
        _missing,
    )
    report = run_oi_deleveraging_evaluation(
        OIDeleveragingEvaluationRequest(symbol="BTCUSDT", start="2024-01-01", log_run=False),
        log_run=False,
    )
    assert report.status == "PENDING"
    assert report.promotion.observation_verdict == "PENDING"


def test_oi_deleveraging_sealed_end_returns_pending() -> None:
    """FD-06: an end past the sealed observation window is not evaluated."""
    report = run_oi_deleveraging_evaluation(
        OIDeleveragingEvaluationRequest(symbol="BTCUSDT", end="2026-06-01", log_run=False),
        log_run=False,
    )
    assert report.status == "PENDING"
    assert report.promotion.observation_verdict == "PENDING"
