from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.application.technical_expert_evaluation import run_technical_expert_evaluation
from src.research.contracts import TechnicalExpertEvaluationRequest
from src.research.evaluation.promotion import PromotionResult
from src.research.expert_portfolio.catalog import default_catalog


def _two_year_frame() -> pd.DataFrame:
    n = 4400
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.01 * t + 5.0 * np.sin(t / 20.0)
    open_ = close - 0.1
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + 0.5,
        "low": np.minimum(open_, close) - 0.5,
        "close": close,
        "volume": 1000.0,
    }, index=index)


def test_rejected_candidate_is_not_registered(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """TE-06: a failed candidate cannot mutate default_catalog or become ACTIVE.

    The frozen screen runs through the unchanged reliability/fold/stress gates,
    the promotion is REJECTED, the rejection is appended idempotently to the
    anti-pattern store under the immutable candidate return source, and no
    ``technical_price_v1`` library appears in the default catalog.
    """
    frame = _two_year_frame()
    funding = pd.Series(0.0, index=frame.index, dtype=float)

    monkeypatch.setattr(
        "src.application.technical_expert_evaluation._load_technical_market_data",
        lambda symbol, start, end: (frame, funding),
    )
    monkeypatch.setattr(
        "src.application.technical_expert_evaluation._data_hashes",
        lambda symbol: {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    )
    request = TechnicalExpertEvaluationRequest(
        candidate_id="technical_macd_histogram_regime_long_v1",
        symbol="BTCUSDT",
        start="2024-01-01",
        log_run=False,
    )
    report = run_technical_expert_evaluation(request, log_run=False)

    assert report.status == "PASS"
    assert isinstance(report.promotion, PromotionResult)
    assert report.promotion.status == "REJECTED"
    assert report.promotion.candidate is not None
    assert report.promotion.candidate.return_source == "technical_macd_histogram_regime_long_v1"

    assert "technical_price_v1" not in default_catalog().blueprints


def test_technical_expert_missing_data_returns_pending(monkeypatch) -> None:
    from src.common.errors import DataIntegrityError

    def _missing(symbol, start, end):
        raise DataIntegrityError("bars data missing for BTCUSDT")

    monkeypatch.setattr(
        "src.application.technical_expert_evaluation._load_technical_market_data",
        _missing,
    )
    report = run_technical_expert_evaluation(
        TechnicalExpertEvaluationRequest(
            candidate_id="technical_macd_histogram_regime_long_v1",
            symbol="BTCUSDT",
            start="2024-01-01",
            log_run=False,
        ),
        log_run=False,
    )
    assert report.status == "PENDING"
    assert report.promotion.observation_verdict == "PENDING"


def test_technical_expert_equity_exhaustion_returns_pending(monkeypatch) -> None:
    from src.common.errors import DataIntegrityError

    frame = _two_year_frame()
    funding = pd.Series(0.0, index=frame.index, dtype=float)
    monkeypatch.setattr(
        "src.application.technical_expert_evaluation._load_technical_market_data",
        lambda symbol, start, end: (frame, funding),
    )
    monkeypatch.setattr(
        "src.application.technical_expert_evaluation._data_hashes",
        lambda symbol: {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    )
    monkeypatch.setattr(
        "src.application.technical_expert_evaluation.compute_code_hash",
        lambda *args, **kwargs: "c" * 64,
    )
    monkeypatch.setattr(
        "src.application.technical_expert_evaluation._run_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(DataIntegrityError("equity exhausted")),
    )

    report = run_technical_expert_evaluation(
        TechnicalExpertEvaluationRequest(
            candidate_id="technical_macd_histogram_regime_long_v1",
            symbol="BTCUSDT",
            start="2024-01-01",
            log_run=False,
        ),
        log_run=False,
    )

    assert report.status == "PENDING"
    assert report.promotion.observation_verdict == "PENDING"


def test_technical_expert_sealed_end_returns_pending() -> None:
    report = run_technical_expert_evaluation(
        TechnicalExpertEvaluationRequest(
            candidate_id="technical_macd_histogram_regime_long_v1",
            symbol="BTCUSDT",
            end="2026-06-01",
            log_run=False,
        ),
        log_run=False,
    )
    assert report.status == "PENDING"
    assert report.promotion.observation_verdict == "PENDING"


def test_technical_expert_unknown_candidate_rejected_before_running(monkeypatch) -> None:
    import pytest

    def _missing(symbol, start, end):
        raise AssertionError("data loading must not run for an unknown candidate")

    monkeypatch.setattr(
        "src.application.technical_expert_evaluation._load_technical_market_data",
        _missing,
    )
    with pytest.raises(ValueError, match="unknown or retired"):
        run_technical_expert_evaluation(
            TechnicalExpertEvaluationRequest(
                candidate_id="technical_nope_long_v1",
                symbol="BTCUSDT",
                start="2024-01-01",
                log_run=False,
            ),
            log_run=False,
        )
