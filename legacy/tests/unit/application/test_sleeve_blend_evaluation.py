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


def test_tournament_request_requires_qualification_interval() -> None:
    with pytest.raises(ValueError, match="qualification_interval must not be empty"):
        SleeveBlendEvaluationRequest(
            candidate_kind="core5_causal_tournament_v1",
            discovery_end="2024-12-31 23:59:59+00:00",
            qualification_interval="",
        )


def test_tournament_request_requires_discovery_end() -> None:
    with pytest.raises(ValueError, match="discovery_end is required"):
        SleeveBlendEvaluationRequest(candidate_kind="core5_causal_tournament_v1")


def test_tournament_request_rejects_naive_discovery_end() -> None:
    with pytest.raises(ValueError, match="discovery_end must be tz-aware UTC"):
        SleeveBlendEvaluationRequest(
            candidate_kind="core5_causal_tournament_v1",
            discovery_end="2024-12-31 23:59:59",
        )


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


def _good_result() -> BacktestResult:
    """Steady multi-year rise with a recoverable dip: passes every reliability gate."""
    idx = pd.date_range("2022-01-01", periods=3 * 2190, freq="4h", tz="UTC")
    growth = np.linspace(1.0, 2.0, len(idx))
    noise = 1.0 + 0.003 * np.sin(np.arange(len(idx)) / 18.0)
    eqv = 10_000.0 * growth * noise
    dip = np.ones(len(idx))
    dip[1500:1650] = np.linspace(1.0, 0.82, 150)
    dip[1650:1750] = np.linspace(0.82, 1.0, 100)
    equity = pd.Series(eqv * dip, index=idx, name="equity")
    trades = pd.DataFrame({
        "symbol": ["A"] * 40,
        "entry_bar": np.arange(40),
        "exit_bar": np.arange(40) + 1,
        "entry_time": idx[:40],
        "exit_time": idx[1:41],
        "entry_price": [100.0] * 40,
        "exit_price": [104.0] * 40,
        "qty": [10.0] * 40,
        "reason": ["channel"] * 40,
        "pnl": [40.0] * 40,
        "return_pct": [0.04] * 40,
        "funding_pnl": [0.0] * 40,
    })
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame())


def _flat_result() -> BacktestResult:
    """Constant cash ledger: never admits a candidate."""
    idx = pd.date_range("2022-01-01", periods=3 * 2190, freq="4h", tz="UTC")
    equity = pd.Series(np.full(len(idx), 10_000.0), index=idx, name="equity")
    return BacktestResult(equity=equity, trades=pd.DataFrame(), signals=pd.DataFrame())


def _fake_tournament_report(request, *, base, stress, selected):
    from src.research.sleeve_blend.contracts import PortfolioBlendTournamentReport

    return PortfolioBlendTournamentReport(
        request=request,
        universe=request.universe,
        candidates=(),
        selected_return_sources=selected,
        blend_weights=tuple(1.0 / len(selected) for _ in selected),
        leverage_schedule=pd.Series(
            np.full(len(base.equity), 1.5), index=base.equity.index, name="leverage",
        ),
        schedule_hash="fake-hash",
        base_result=base,
        stress_result=stress,
        qualification_start=request.discovery_end,
        qualification_end=None,
    )


def test_pbgt_03_tournament_stress_reuses_identical_base_schedule(monkeypatch) -> None:
    """PBGT-03: stress consumes the tournament's own stress ledger (identical
    universe, members, weights, and leverage schedule) and never recalibrates."""
    import dataclasses

    from src.research.evaluation.reliability import (
        ReliabilityGateConfig,
        compute_equity_reliability_gate,
    )

    calls: list[object] = []
    good = _good_result()
    stress_good = _good_result()
    stress_good.equity = stress_good.equity ** 1.05

    def fake_tournament(request):
        calls.append(request)
        return _fake_tournament_report(
            request, base=good, stress=stress_good, selected=("donchian_long_only_v1",),
        )

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.run_strategy_tournament", fake_tournament)
    report = run_sleeve_blend_evaluation(SleeveBlendEvaluationRequest(
        candidate_kind="core5_causal_tournament_v1",
        discovery_end="2024-12-31 23:59:59+00:00",
        log_run=False,
    ))
    assert len(calls) == 1
    assert report.status == "PASS"
    expected_obs = compute_equity_reliability_gate(good.equity, len(good.trades))
    expected_stress = compute_equity_reliability_gate(
        stress_good.equity, len(stress_good.trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )
    assert report.observation.verdict == expected_obs.verdict
    assert report.stress.lcb90_cagr == pytest.approx(expected_stress.lcb90_cagr)


def test_pbgt_06_tournament_promotion_requires_full_discovery_pass(monkeypatch) -> None:
    """PBGT-06: only observation PASS, fold PASS, stress PASS, and an admitted
    fixed universe yield OBSERVATION_PASS; otherwise promotion is REJECTED."""
    good = _good_result()
    flat = _flat_result()

    def pass_fake(request):
        return _fake_tournament_report(
            request, base=good, stress=good, selected=("donchian_long_only_v1",),
        )

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.run_strategy_tournament", pass_fake)
    report = run_sleeve_blend_evaluation(SleeveBlendEvaluationRequest(
        candidate_kind="core5_causal_tournament_v1",
        discovery_end="2024-12-31 23:59:59+00:00",
        log_run=False,
    ))
    assert report.promotion.status == "OBSERVATION_PASS"

    def reject_fake(request):
        return _fake_tournament_report(request, base=flat, stress=flat, selected=())

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.run_strategy_tournament", reject_fake)
    report2 = run_sleeve_blend_evaluation(SleeveBlendEvaluationRequest(
        candidate_kind="core5_causal_tournament_v1",
        discovery_end="2024-12-31 23:59:59+00:00",
        log_run=False,
    ))
    assert report2.promotion.status == "REJECTED"


def test_tournament_evaluation_persists_auditable_fields(monkeypatch) -> None:
    """The tournament evaluation persists universe id, selection, schedule hash,
    and rejected reasons through record_sleeve_blend_run."""
    good = _good_result()
    captured: list[dict[str, object]] = []

    def fake_tournament(request):
        return _fake_tournament_report(
            request, base=good, stress=good, selected=("donchian_long_only_v1",),
        )

    def fake_record(**kwargs):
        captured.append(kwargs)
        return {"git_sha": "abc", "git_dirty": False}

    monkeypatch.setattr(f"{_APPLICATION_MODULE}.run_strategy_tournament", fake_tournament)
    monkeypatch.setattr(f"{_APPLICATION_MODULE}.record_sleeve_blend_run", fake_record)

    run_sleeve_blend_evaluation(SleeveBlendEvaluationRequest(
        candidate_kind="core5_causal_tournament_v1",
        discovery_end="2024-12-31 23:59:59+00:00",
        log_run=True,
    ))
    assert len(captured) == 1
    rec = captured[0]
    assert rec["candidate_kind"] == "core5_causal_tournament_v1"
    assert rec["universe_id"] == "core5_v1"
    assert rec["candidate_return_sources"] == ["donchian_long_only_v1"]
    assert rec["selection_window"] == "..2024-12-31 23:59:59+00:00"
    assert rec["leverage_schedule_hash"] == "fake-hash"
    assert rec["leverage"] is None
    assert rec["mdd_budget_fraction"] is None
    assert rec["rejected_candidate_reasons"] == {}
