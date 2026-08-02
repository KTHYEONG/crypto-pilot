from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.blend.evaluation import run_sleeve_blend_evaluation
from src.research.baseline.backtest import BacktestResult
from src.research.contracts import SleeveBlendEvaluationRequest

_APPLICATION_MODULE = "src.application.research.blend.evaluation"
_BACKTEST_MODULE = "src.research.sleeve_blend.fixed"


def _breakout_frame(signal_bar: int, crash_bar: int, n: int = 4400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l_ = np.full(n, 99.0)
    c = np.full(n, 100.0)
    c[signal_bar] = 106.0
    h[signal_bar] = 107.0
    l_[signal_bar] = 105.0
    o[signal_bar + 1 : crash_bar] = 106.0
    h[signal_bar + 1 : crash_bar] = 107.0
    l_[signal_bar + 1 : crash_bar] = 105.0
    c[signal_bar + 1 : crash_bar] = 106.0
    o[crash_bar:] = 90.0
    h[crash_bar:] = 91.0
    l_[crash_bar:] = 89.0
    c[crash_bar:] = 90.0
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=idx)


def _synthetic_result() -> BacktestResult:
    idx = pd.date_range("2024-01-01", periods=4400, freq="4h", tz="UTC")
    equity = pd.Series(10_000.0 * np.linspace(1.0, 1.4, len(idx)), index=idx, name="equity")
    trades = pd.DataFrame({
        "symbol": ["A"] * 40,
        "entry_bar": np.arange(40),
        "exit_bar": np.arange(40) + 1,
        "entry_time": idx[:40],
        "exit_time": idx[1:41],
        "entry_price": [100.0] * 40,
        "exit_price": [101.0] * 40,
        "qty": [10.0] * 40,
        "reason": ["channel"] * 40,
        "pnl": [10.0] * 40,
        "return_pct": [0.01] * 40,
        "funding_pnl": [0.0] * 40,
    })
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


def test_stress_reuses_base_leverage_not_recalibrated(monkeypatch) -> None:
    base_lev = 2.5
    stress_leverage = []

    def fake_calibrated(symbols, start, end, costs, mdd_budget_fraction, initial_equity=..., signal_delay_bars=0):
        return _synthetic_result(), base_lev

    def fake_with_leverage(symbols, start, end, costs, lev, initial_equity=..., signal_delay_bars=0):
        stress_leverage.append(lev)
        return _synthetic_result()

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.run_fixed_sleeve_portfolio_calibrated", fake_calibrated)
    monkeypatch.setattr(f"{_APPLICATION_MODULE}.run_fixed_sleeve_portfolio_with_leverage", fake_with_leverage)

    report = run_sleeve_blend_evaluation(
        SleeveBlendEvaluationRequest(symbols=("A", "B"), log_run=False),
    )
    assert stress_leverage == [base_lev]
    assert report.status == "PASS"


def test_directional_stress_reuses_base_weights_not_recalibrated(monkeypatch) -> None:
    # SC-SGV2-07: the directional candidate's stress run must receive the base
    # causal weight series verbatim (never re-calibrated around stressed costs).
    base_weights = pd.DataFrame(
        {"A:long": [0.25], "B:long": [0.75]},
        index=pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC"),
    )
    stress_weights: list[pd.DataFrame] = []

    def fake_with_weights(symbols, start, end, costs, initial_equity=...):
        return _synthetic_result(), base_weights

    def fake_fixed_weights(symbols, start, end, costs, weights, initial_equity=..., signal_delay_bars=...):
        stress_weights.append(weights)
        return _synthetic_result()

    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.run_directional_sleeve_portfolio_with_weights",
        fake_with_weights,
    )
    monkeypatch.setattr(
        f"{_APPLICATION_MODULE}.run_directional_sleeve_portfolio_fixed_weights",
        fake_fixed_weights,
    )

    report = run_sleeve_blend_evaluation(
        SleeveBlendEvaluationRequest(
            candidate_kind="funding_signed_directional_v1", log_run=False,
        ),
    )
    assert len(stress_weights) == 1
    pd.testing.assert_frame_equal(stress_weights[0], base_weights)
    assert report.status == "PASS"


def test_invalid_candidate_kind_rejected() -> None:
    with pytest.raises(ValueError, match="candidate_kind"):
        SleeveBlendEvaluationRequest(candidate_kind="unknown_kind")  # type: ignore[arg-type]


def test_sleeve_blend_evaluation_composes_promotion_like_baseline(monkeypatch) -> None:
    frames = {
        "BTCUSDT": _breakout_frame(signal_bar=260, crash_bar=275),
        "ETHUSDT": _breakout_frame(signal_bar=60, crash_bar=90),
    }
    monkeypatch.setattr(
        f"{_BACKTEST_MODULE}.ohlcv_path", lambda symbol, timeframe: Path(f"{symbol}.parquet"),
    )
    monkeypatch.setattr(
        f"{_BACKTEST_MODULE}.load_ohlcv_4h",
        lambda path, start=None, end=None: frames[Path(str(path)).stem],
    )

    report = run_sleeve_blend_evaluation(
        SleeveBlendEvaluationRequest(symbols=("BTCUSDT", "ETHUSDT"), log_run=False),
    )
    assert report.status == "PASS"
    assert report.promotion.status in {"REJECTED", "OBSERVATION_PASS", "HOLDOUT_PASS"}
    assert report.promotion.observation_verdict in {"PASS", "FAIL", "PENDING"}
