from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.strategy.config import RegimeConfig
from src.domain.futures.strategy.market_regime import (
    compute_market_state_risk_off_1d,
    compute_xs_downside_breadth_1d,
)

# ── Helpers ──

def _constant_universe(t_len: int, n_sym: int, value: float = 100.0) -> NDArray[np.float64]:
    return np.full((t_len, n_sym), value, dtype=np.float64)


def _breadth_region_universe(
    t_len: int,
    n_sym: int,
    mom_window: int,
    n_neg: int,
    bar_start: int,
    bar_end: int,
    decay: float = 0.99,
) -> NDArray[np.float64]:
    """[T,N] close where ``n_neg`` symbols continuously decline during
    [bar_start, bar_end], producing sustained neg_frac = n_neg/N."""
    close = np.full((t_len, n_sym), 100.0, dtype=np.float64)
    for t in range(bar_start, min(bar_end + 1, t_len)):
        close[t, :n_neg] = close[t - 1, :n_neg] * decay
    return close


def _make_btc_crash_series(n_stable: int = 30) -> NDArray[np.float64]:
    """BTC: rise (t=0..14), crash (t=15..29), stabilize (n_stable bars)."""
    rise = np.linspace(100.0, 110.0, 15, dtype=np.float64)
    crash = np.linspace(110.0, 78.0, 15, dtype=np.float64)
    stable = np.full(n_stable, 78.0, dtype=np.float64)
    return np.concatenate([rise, crash, stable])


# ════════════════════════════════════════════════════════════
# Scenario Group 1 — compute_xs_downside_breadth_1d
# ════════════════════════════════════════════════════════════

class TestBreadth:
    """S1: compute_xs_downside_breadth_1d"""

    def test_breadth_neg_fraction_computes_valid_ratio(self) -> None:
        mom_window = 5
        T, N = 30, 5
        close = np.full((T, N), 100.0, dtype=np.float64)
        close[10, :3] = 95.0
        close[10, 3:] = 100.0
        breadth = compute_xs_downside_breadth_1d(close, mom_window=mom_window)
        assert breadth.shape == (T,)
        assert breadth[10] == pytest.approx(0.6, rel=1e-9)
        assert breadth[:5].sum() == 0.0

    def test_breadth_warmup_returns_zero(self) -> None:
        mom_window = 10
        T, N = 25, 4
        close = _constant_universe(T, N)
        close[15, :2] = 50.0
        breadth = compute_xs_downside_breadth_1d(close, mom_window=mom_window)
        assert breadth[:mom_window].sum() == 0.0

    def test_breadth_excludes_invalid_symbols_from_denominator(self) -> None:
        mom_window = 5
        T, N = 20, 5
        close = np.full((T, N), 100.0, dtype=np.float64)
        close[:, 0] = np.nan
        close[:, 1] = -1.0
        close[10, 2] = 95.0
        breadth = compute_xs_downside_breadth_1d(close, mom_window=mom_window)
        valid_N = 3
        assert breadth[10] == pytest.approx(1.0 / valid_N, rel=1e-9)

    def test_breadth_all_invalid_returns_zero(self) -> None:
        T, N = 15, 3
        close = np.full((T, N), np.nan, dtype=np.float64)
        breadth = compute_xs_downside_breadth_1d(close, mom_window=5)
        assert np.all(breadth == 0.0)

    def test_breadth_no_lookahead(self) -> None:
        mom_window = 5
        T, N = 25, 5
        close = np.full((T, N), 100.0, dtype=np.float64)
        close[10, :3] = 95.0
        ref = compute_xs_downside_breadth_1d(close, mom_window=mom_window)
        close[11:, :] = 200.0
        post = compute_xs_downside_breadth_1d(close, mom_window=mom_window)
        assert np.array_equal(ref[:11], post[:11])


# ════════════════════════════════════════════════════════════
# Scenario Group 2 — compute_market_state_risk_off_1d
# ════════════════════════════════════════════════════════════

class TestPanel:
    """S2: compute_market_state_risk_off_1d"""

    def test_panel_and_gate_suppresses_btc_only_false_positive(self) -> None:
        N = 5
        btc = _make_btc_crash_series(n_stable=30)
        T = btc.shape[0]
        universe = _breadth_region_universe(T, N, mom_window=5, n_neg=0,
                                            bar_start=0, bar_end=T - 1)
        risk_off = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.40,
            persistence_bars=1,
        )
        assert risk_off.shape == (T,)
        assert risk_off.dtype == np.bool_
        assert not risk_off.any()

    def test_panel_fires_on_confirmed_breadth_break(self) -> None:
        N = 5
        btc = _make_btc_crash_series(n_stable=30)
        T = btc.shape[0]
        universe = _breadth_region_universe(T, N, mom_window=5, n_neg=4,
                                            bar_start=12, bar_end=T - 1)
        risk_off = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.40,
            persistence_bars=1,
        )
        assert risk_off.any()

    def test_panel_exit_requires_breadth_below_exit_threshold(self) -> None:
        T, N = 80, 5
        btc = _make_btc_crash_series(n_stable=50)
        universe = np.full((T, N), 100.0, dtype=np.float64)
        for t in range(15, 50):
            universe[t, :] = universe[t - 1, :] * 0.99
        for t in range(50, T):
            universe[t, :2] = universe[t - 1, :2] * 0.99
            universe[t, 2:] = universe[t - 1, 2:] * 1.005
        risk_off = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.20,
            persistence_bars=1,
        )
        n_active = int(risk_off.sum())
        assert n_active > 5

    def test_panel_recovery_cooldown_delays_release(self) -> None:
        N = 5
        btc = _make_btc_crash_series(n_stable=50)
        T = btc.shape[0]
        universe = _breadth_region_universe(T, N, mom_window=5, n_neg=4,
                                            bar_start=12, bar_end=T - 1)
        risk_off_no_cooldown = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.40,
            persistence_bars=1,
            recovery_cooldown_bars=0,
        )
        risk_off_cooldown = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.40,
            persistence_bars=1,
            recovery_cooldown_bars=3,
        )
        assert risk_off_cooldown.sum() >= risk_off_no_cooldown.sum()

    def test_panel_persistence_requires_consecutive_bars(self) -> None:
        N = 5
        btc = _make_btc_crash_series(n_stable=30)
        T = btc.shape[0]
        universe = _breadth_region_universe(T, N, mom_window=5, n_neg=4,
                                            bar_start=15, bar_end=16)
        risk_off = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.40,
            persistence_bars=3,
        )
        assert not risk_off.any()

    def test_panel_output_is_one_bar_shifted(self) -> None:
        N = 5
        btc = _make_btc_crash_series(n_stable=30)
        T = btc.shape[0]
        universe = _breadth_region_universe(T, N, mom_window=5, n_neg=4,
                                            bar_start=12, bar_end=T - 1)
        risk_off = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.40,
            persistence_bars=1,
        )
        assert not risk_off[0]

    def test_panel_empty_input_returns_empty(self) -> None:
        btc = np.empty(0, dtype=np.float64)
        universe = np.empty((0, 5), dtype=np.float64)
        result = compute_market_state_risk_off_1d(
            btc, universe,
            dd_window=10, dd_threshold=0.06,
            mom_fast=5, mom_slow=20,
            breadth_mom_window=5,
            breadth_neg_frac_enter=0.60,
            breadth_neg_frac_exit=0.40,
        )
        assert result.shape == (0,)
        assert result.dtype == np.bool_


# ════════════════════════════════════════════════════════════
# Scenario Group 3 — Config 검증
# ════════════════════════════════════════════════════════════

class TestPanelConfig:
    """S3: RegimeConfig new field validation"""

    def test_regime_config_rejects_non_asymmetric_breadth_thresholds(self) -> None:
        with pytest.raises(ValueError, match="hysteresis"):
            RegimeConfig(breadth_neg_frac_enter=0.30, breadth_neg_frac_exit=0.50)

    def test_regime_config_rejects_unknown_reversal_mode(self) -> None:
        with pytest.raises(ValueError, match="reversal_mode"):
            RegimeConfig(reversal_mode="invalid")

    def test_regime_config_rejects_short_breadth_window(self) -> None:
        with pytest.raises(ValueError, match="breadth_mom_window"):
            RegimeConfig(breadth_mom_window=1)

    def test_regime_config_defaults_are_valid(self) -> None:
        cfg = RegimeConfig()
        assert cfg.reversal_mode == "btc"
        assert cfg.breadth_mom_window == 24
        assert cfg.breadth_neg_frac_enter == 0.60
        assert cfg.breadth_neg_frac_exit == 0.45
        assert cfg.reversal_recovery_cooldown_bars == 0
