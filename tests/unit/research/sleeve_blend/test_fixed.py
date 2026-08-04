from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import BacktestResult, run_backtest
from src.research.contracts import CostModel, StrategySpec
from src.research.evaluation.reliability import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
)
from src.research.sleeve_blend.common import _common_index
from src.research.sleeve_blend.fixed import (
    run_fixed_sleeve_portfolio,
    run_fixed_sleeve_portfolio_calibrated,
    run_fixed_sleeve_portfolio_with_leverage,
)

_BACKTEST_MODULE = "src.research.sleeve_blend.fixed"


def test_common_index_preserves_shared_equity_window() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
    result = BacktestResult(
        equity=pd.Series([10_000.0, 10_001.0], index=idx),
        trades=pd.DataFrame(),
        signals=pd.DataFrame(),
    )

    assert _common_index({"A": result, "B": result}).equals(idx)


def _breakout_frame(signal_bar: int, crash_bar: int, n: int = 4400) -> pd.DataFrame:
    """Flat base that breaks out at ``signal_bar``, holds, then crashes at ``crash_bar``.

    Offset ``signal_bar``/``crash_bar`` across symbols produces imperfectly
    correlated returns for blend testing.
    """
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


def _ramp_frame(n: int = 4400) -> pd.DataFrame:
    """Strictly rising price that never triggers a stop or channel exit.

    Produces a monotone non-decreasing equity curve with MDD == 0.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    opens = 100.0 + 0.01 * np.arange(n, dtype=np.float64)
    return pd.DataFrame({
        "open": opens,
        "high": opens + 0.5,
        "low": opens - 0.5,
        "close": opens + 0.25,
        "volume": 1000.0,
    }, index=idx)


def _trend_drop_frame(drop_bar: int, n: int = 4400) -> pd.DataFrame:
    """Profitable trend with a permanent marked drawdown at ``drop_bar``.

    One breakout long rides a sustained rise (equity peaks well above entry),
    then the price drops permanently below the prior lows at ``drop_bar``, which
    triggers a profitable channel exit while marking a real drawdown. The
    position never stops out, so the sleeve is net profitable with MDD < 0.
    Offset ``drop_bar`` across sleeves yields imperfectly correlated returns.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l_ = np.full(n, 99.0)
    c = np.full(n, 100.0)
    c[260] = 110.0
    h[260] = 111.0
    l_[260] = 109.0
    o[261:301] = 110.0
    h[261:301] = 111.0
    l_[261:301] = 109.0
    c[261:301] = 110.0
    o[301:361] = 118.0
    h[301:361] = 119.0
    l_[301:361] = 117.0
    c[301:361] = 118.0
    o[361:421] = 125.0
    h[361:421] = 126.0
    l_[361:421] = 124.0
    c[361:421] = 125.0
    o[421:481] = 130.0
    h[421:481] = 131.0
    l_[421:481] = 129.0
    c[421:481] = 130.0
    o[481:] = 130.0
    h[481:] = 131.0
    l_[481:] = 129.0
    c[481:] = 130.0
    o[drop_bar:] = 115.0
    h[drop_bar:] = 116.0
    l_[drop_bar:] = 114.0
    c[drop_bar:] = 115.0
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=idx)


def _install_frames(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> None:
    monkeypatch.setattr(
        f"{_BACKTEST_MODULE}.ohlcv_path", lambda symbol, timeframe: Path(f"{symbol}.parquet"),
    )
    monkeypatch.setattr(
        f"{_BACKTEST_MODULE}.load_ohlcv_4h",
        lambda path, start=None, end=None: frames[Path(str(path)).stem],
    )


def _mdd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def test_blend_is_lower_mdd_than_worst_sleeve(monkeypatch) -> None:
    frames = {
        "A": _breakout_frame(signal_bar=260, crash_bar=275),
        "B": _breakout_frame(signal_bar=60, crash_bar=90),
    }
    _install_frames(monkeypatch, frames)
    costs = CostModel()

    sleeve_mdds = {
        symbol: _mdd(run_backtest(df, StrategySpec(symbol=symbol), costs).equity)
        for symbol, df in frames.items()
    }
    avg_mdd = float(np.mean(list(sleeve_mdds.values())))

    blend = run_fixed_sleeve_portfolio_with_leverage(("A", "B"), None, None, costs, lev=1.0)
    blend_mdd = _mdd(blend.equity)

    assert blend_mdd >= avg_mdd


def test_leverage_scales_cagr_mdd_lcb90_but_not_fold_or_tstat(monkeypatch) -> None:
    frames = {
        "A": _trend_drop_frame(drop_bar=481),
        "B": _trend_drop_frame(drop_bar=1201),
    }
    _install_frames(monkeypatch, frames)
    costs = CostModel()

    low = run_fixed_sleeve_portfolio_with_leverage(("A", "B"), None, None, costs, lev=1.0)
    high = run_fixed_sleeve_portfolio_with_leverage(("A", "B"), None, None, costs, lev=2.0)

    low_gate = compute_equity_reliability_gate(low.equity, len(low.trades))
    high_gate = compute_equity_reliability_gate(high.equity, len(high.trades))
    low_fold = compute_fold_distribution(low)
    high_fold = compute_fold_distribution(high)

    assert high_gate.t_stat == pytest.approx(low_gate.t_stat, rel=1e-9)
    assert high_fold.max_period_contribution == pytest.approx(
        low_fold.max_period_contribution, rel=1e-2,
    )
    assert _mdd(high.equity) == pytest.approx(2.0 * _mdd(low.equity), rel=1e-2)
    assert high_gate.point_cagr == pytest.approx(2.0 * low_gate.point_cagr, rel=5e-2)
    assert abs(high_gate.lcb90_cagr) > abs(low_gate.lcb90_cagr)


def test_zero_or_positive_unlevered_mdd_raises(monkeypatch) -> None:
    frames = {"A": _ramp_frame(), "B": _ramp_frame()}
    _install_frames(monkeypatch, frames)
    costs = CostModel()
    with pytest.raises(DataIntegrityError, match="unlevered blended MDD"):
        run_fixed_sleeve_portfolio(("A", "B"), None, None, costs, mdd_budget_fraction=0.85)


def test_calibrated_leverage_matches_target_mdd_budget(monkeypatch) -> None:
    frames = {
        "A": _breakout_frame(signal_bar=260, crash_bar=275),
        "B": _breakout_frame(signal_bar=60, crash_bar=90),
    }
    _install_frames(monkeypatch, frames)
    costs = CostModel()
    _result, lev = run_fixed_sleeve_portfolio_calibrated(
        ("A", "B"), None, None, costs, mdd_budget_fraction=0.85,
    )
    target_mdd = ReliabilityGateConfig().mdd_floor * 0.85
    assert lev == pytest.approx(target_mdd / _mdd(_unlevered_blend(frames, costs)), rel=1e-9)


def test_pbgt_02_causal_schedule_prefix_is_immune_to_later_returns(monkeypatch) -> None:
    """PBGT-02: changing returns after a rebalance cannot change the leverage
    scheduled at or before that rebalance; insufficient history is zero."""
    from src.research.sleeve_blend.contracts import CausalLeverageSpec
    from src.research.sleeve_blend.fixed import build_causal_leverage_schedule

    idx = pd.date_range("2022-01-01", periods=2200, freq="4h", tz="UTC")
    unit = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    unit.iloc[1000:1050] = unit.iloc[1000:1050] * 0.7
    spec = CausalLeverageSpec(lookback_days=120, risk_budget_fraction=0.85, max_gross_leverage=3.0)

    schedule = build_causal_leverage_schedule(unit, spec)
    altered = unit.copy()
    altered.iloc[1500:] = altered.iloc[1500:] * 1.5
    schedule_altered = build_causal_leverage_schedule(altered, spec)

    assert schedule_altered.iloc[:1500].equals(schedule.iloc[:1500])
    lookback_bars = round(pd.Timedelta(days=120) / pd.Timedelta(hours=4))
    assert schedule.iloc[:lookback_bars].eq(0.0).all()
    assert schedule.iloc[lookback_bars] > 0.0


def test_fractional_kelly_schedule_causal_capped_and_prefix_immune() -> None:
    """FK-01: the fractional-Kelly/MDD schedule is causal, non-negative, capped
    at the gross limit, never exceeds the MDD cap, and ignores later marks."""
    from src.research.sleeve_blend.contracts import (
        CausalFractionalKellySpec,
        CausalLeverageSpec,
    )
    from src.research.sleeve_blend.fixed import (
        build_causal_fractional_kelly_schedule,
        build_causal_leverage_schedule,
    )

    idx = pd.date_range("2022-01-01", periods=4400, freq="4h", tz="UTC")
    unit = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    unit.iloc[1000:1050] = unit.iloc[1000:1050] * 0.7
    spec = CausalLeverageSpec()
    kelly = CausalFractionalKellySpec()

    schedule = build_causal_fractional_kelly_schedule(unit, spec, kelly)
    mdd_cap = build_causal_leverage_schedule(unit, spec)

    lookback_bars = round(pd.Timedelta(days=365) / pd.Timedelta(hours=4))
    assert schedule.iloc[:lookback_bars].eq(0.0).all()
    assert (schedule >= 0).all()
    assert (schedule <= spec.max_gross_leverage).all()
    assert (schedule.to_numpy() <= mdd_cap.to_numpy() + 1e-12).all()

    altered = unit.copy()
    altered.iloc[3500:] = altered.iloc[3500:] * 1.5
    rebuilt = build_causal_fractional_kelly_schedule(altered, spec, kelly)
    assert rebuilt.iloc[:3500].equals(schedule.iloc[:3500])


def test_fractional_kelly_zero_exposure_before_complete_lookback() -> None:
    """FK-02: without a complete lookback or with non-positive Kelly moments
    the policy exposure is zero, and malformed series raise ValueError."""
    from src.research.sleeve_blend.contracts import (
        CausalFractionalKellySpec,
        CausalLeverageSpec,
    )
    from src.research.sleeve_blend.fixed import build_causal_fractional_kelly_schedule

    idx = pd.date_range("2022-01-01", periods=400, freq="4h", tz="UTC")
    spec = CausalLeverageSpec()
    kelly = CausalFractionalKellySpec()

    declining = pd.Series(np.linspace(100.0, 80.0, len(idx)), index=idx)
    schedule = build_causal_fractional_kelly_schedule(declining, spec, kelly)
    assert schedule.eq(0.0).all()

    with pytest.raises(ValueError, match="unit_equity"):
        build_causal_fractional_kelly_schedule(
            pd.Series(np.full(200, -1.0), index=idx[:200]), spec, kelly,
        )

    with pytest.raises(ValueError, match="lookback_days"):
        build_causal_fractional_kelly_schedule(
            pd.Series(np.linspace(100.0, 110.0, 400), index=idx),
            CausalLeverageSpec(lookback_days=120), kelly,
        )


def test_run_fixed_sleeve_portfolio_with_schedule_applies_frozen_schedule(monkeypatch) -> None:
    """The frozen-schedule execution reuses the pre-built schedule verbatim."""
    from src.research.sleeve_blend.contracts import CausalLeverageSpec
    from src.research.sleeve_blend.fixed import (
        build_causal_leverage_schedule,
        run_fixed_sleeve_portfolio_with_schedule,
    )

    frames = {
        "A": _trend_drop_frame(drop_bar=481),
        "B": _trend_drop_frame(drop_bar=1201),
    }
    _install_frames(monkeypatch, frames)
    costs = CostModel()
    from src.research.sleeve_blend.fixed import apply_leverage_schedule

    unit = run_fixed_sleeve_portfolio_with_leverage(
        ("A", "B"), None, None, costs, lev=1.0,
    ).equity
    schedule = build_causal_leverage_schedule(
        unit, CausalLeverageSpec(lookback_days=60),
    )
    scheduled = run_fixed_sleeve_portfolio_with_schedule(
        ("A", "B"), None, None, costs, schedule,
    )
    assert (scheduled.equity > 0).all()
    lookback_bars = round(pd.Timedelta(days=60) / pd.Timedelta(hours=4))
    assert scheduled.equity.iloc[:lookback_bars].nunique() == 1
    pd.testing.assert_series_equal(
        scheduled.equity,
        apply_leverage_schedule(unit, schedule, initial_equity=10_000.0),
    )


def _unlevered_blend(frames: dict[str, pd.DataFrame], costs: CostModel) -> pd.Series:
    blend = run_fixed_sleeve_portfolio_with_leverage(
        tuple(frames), None, None, costs, lev=1.0,
    )
    return blend.equity
