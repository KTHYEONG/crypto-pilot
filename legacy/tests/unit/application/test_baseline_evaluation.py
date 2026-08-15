from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.baseline.evaluation import (
    run_baseline_analysis_without_funding,
    run_baseline_evaluation,
)
from src.research.baseline.backtest import run_backtest
from src.research.contracts import BaselineEvaluationRequest, CostModel, StrategySpec
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (
    compute_equity_reliability_gate,
    compute_fold_distribution,
    compute_stress_test_gate,
)


def _two_year_breakout_frame() -> pd.DataFrame:
    """Two well-separated breakout cycles spanning two calendar years."""
    n = 4400
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l_ = np.full(n, 99.0)
    c = np.full(n, 100.0)

    def cycle(start: int, jump: float) -> None:
        c[start] = 100.0 + jump
        h[start] = 100.0 + jump + 1.0
        l_[start] = 100.0 + jump - 1.0
        o[start + 1 : start + 8] = 100.0 + jump
        h[start + 1 : start + 8] = 100.0 + jump + 1.0
        l_[start + 1 : start + 8] = 100.0 + jump - 1.0
        c[start + 1 : start + 8] = 100.0 + jump
        o[start + 8 : start + 20] = 100.0 + jump - 2.0
        h[start + 8 : start + 20] = 100.0 + jump - 1.4
        l_[start + 8 : start + 20] = 100.0 + jump - 2.6
        c[start + 8 : start + 20] = 100.0 + jump - 2.0

    cycle(800, 6.0)
    cycle(2200, 9.0)
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=idx)


def test_baseline_evaluation_preserves_frozen_result(monkeypatch) -> None:
    """RF-BASE-01: the canonical application path equals the direct baseline path."""
    df = _two_year_breakout_frame()
    monkeypatch.setattr("src.application.research.baseline.evaluation.ohlcv_path", lambda *a: df.index[0])
    monkeypatch.setattr(
        "src.application.research.baseline.evaluation.load_ohlcv_4h",
        lambda path, start=None, end=None: df,
    )

    request = BaselineEvaluationRequest(
        symbol="BTCUSDT", start="2024-01-01", unseal_holdout=False, log_run=False,
    )
    report = run_baseline_analysis_without_funding(request)

    spec = StrategySpec(symbol="BTCUSDT")
    costs = CostModel()
    result = run_backtest(df, spec, costs)
    metrics = compute_metrics(result.equity, result.trades)
    observation = compute_equity_reliability_gate(result.equity, len(result.trades))
    folds = compute_fold_distribution(result)
    stress = compute_stress_test_gate(df, spec, costs)
    promotion = compose_promotion_verdict(observation, folds, stress, None)

    assert report.status == "PASS"
    assert report.result.equity.equals(result.equity)
    assert report.result.trades.equals(result.trades)
    assert report.result.signals.equals(result.signals)
    assert report.metrics == metrics
    assert report.promotion.status == promotion.status
    assert report.promotion.observation_verdict == promotion.observation_verdict


def test_baseline_evaluation_rejects_end_past_sealed_cutoff() -> None:
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        run_baseline_analysis_without_funding(
            BaselineEvaluationRequest(symbol="BTCUSDT", end="2026-01-01", log_run=False),
        )


def test_baseline_promotion_rejects_missing_funding() -> None:
    """A futures baseline promotion evaluation with no funding stream fails closed."""
    from src.common.errors import DataIntegrityError

    with pytest.raises(DataIntegrityError, match="funding"):
        run_baseline_evaluation(
            BaselineEvaluationRequest(symbol="BTCUSDT", log_run=False),
        )


def test_baseline_evaluation_unseals_holdout(monkeypatch) -> None:
    df = _two_year_breakout_frame()
    monkeypatch.setattr("src.application.research.baseline.evaluation.ohlcv_path", lambda *a: df.index[0])
    monkeypatch.setattr(
        "src.application.research.baseline.evaluation.load_ohlcv_4h",
        lambda path, start=None, end=None: df,
    )
    report = run_baseline_analysis_without_funding(
        BaselineEvaluationRequest(
            symbol="BTCUSDT", end="2026-01-01", unseal_holdout=True, log_run=False,
        ),
    )
    assert report.status == "PASS"
