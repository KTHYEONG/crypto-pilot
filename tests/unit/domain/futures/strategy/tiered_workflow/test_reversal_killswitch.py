from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.strategy.config import RegimeConfig
from src.domain.futures.strategy.tiered_workflow.awf_sim import Layer2FoldAttribution


def test_killswitch_hard_floor_overrides_caps() -> None:
    attr = Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=100,
        n_rebal=10,
        realized_total=0.05,
        realized_price=0.05,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.05,
        mean_gross_exp=0.0,
        mean_net_exp=0.0,
        sleeves_active_mean=0.0,
        friction_pass_ratio=0.0,
        throttle_mult_mean=0.0,
        dropped_below_cost=0,
        netting_events=0,
        risk_off_bars=10,
        risk_off_realized_price=-0.30,
        risk_on_realized_price=0.35,
    )
    assert attr.risk_off_bars == 10
    assert attr.risk_off_realized_price == -0.30
    assert attr.risk_on_realized_price == 0.35


def test_killswitch_disabled_is_noop() -> None:
    attr = Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=100,
        n_rebal=10,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.0,
        mean_net_exp=0.0,
        sleeves_active_mean=0.0,
        friction_pass_ratio=0.0,
        throttle_mult_mean=0.0,
        dropped_below_cost=0,
        netting_events=0,
    )
    assert attr.risk_off_bars == 0
    assert attr.risk_off_realized_price == 0.0
    assert attr.risk_on_realized_price == 0.0


def test_attribution_risk_off_decomposition() -> None:
    attr = Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=100,
        n_rebal=10,
        realized_total=-0.05,
        realized_price=-0.05,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=-0.05,
        mean_gross_exp=0.0,
        mean_net_exp=0.0,
        sleeves_active_mean=0.0,
        friction_pass_ratio=0.0,
        throttle_mult_mean=0.0,
        dropped_below_cost=0,
        netting_events=0,
        risk_off_bars=2,
        risk_off_realized_price=-70.0,
        risk_on_realized_price=45.0,
    )
    assert attr.risk_off_bars == 2
    assert attr.risk_off_realized_price == -70.0
    assert attr.risk_on_realized_price == 45.0


def test_reversal_config_validation() -> None:
    with pytest.raises(ValueError, match="reversal_mom_fast must be < reversal_mom_slow"):
        RegimeConfig(reversal_mom_fast=20, reversal_mom_slow=20)
    with pytest.raises(ValueError, match="reversal_mom_fast must be < reversal_mom_slow"):
        RegimeConfig(reversal_mom_fast=30, reversal_mom_slow=20)
    with pytest.raises(ValueError, match="reversal_dd_window must be >= 2"):
        RegimeConfig(reversal_dd_window=1)
    with pytest.raises(ValueError, match="reversal_dd_threshold must satisfy 0 < value < 1"):
        RegimeConfig(reversal_dd_threshold=1.5)
    with pytest.raises(ValueError, match="reversal_risk_off_floor must be in \\[0, crisis_gross_floor\\)"):
        RegimeConfig(reversal_risk_off_floor=0.20, crisis_gross_floor=0.15)


def test_reversal_config_validates_persistence_bars() -> None:
    with pytest.raises(ValueError, match="reversal_persistence_bars must be >= 1"):
        RegimeConfig(reversal_persistence_bars=0)
    with pytest.raises(ValueError, match="reversal_persistence_bars must be >= 1"):
        RegimeConfig(reversal_persistence_bars=-1)
    assert RegimeConfig().reversal_dd_threshold == pytest.approx(0.12)
    assert RegimeConfig().reversal_persistence_bars == 3


# ── L2 Reversal Recovery Cooldown (Scenario 4: btc-mode env wiring) ──


def test_reversal_config_validates_recovery_cooldown_bars() -> None:
    with pytest.raises(ValueError, match="reversal_recovery_cooldown_bars must be >= 0"):
        RegimeConfig(reversal_recovery_cooldown_bars=-1)
    assert RegimeConfig().reversal_recovery_cooldown_bars == 0


def _close_whipsaw() -> NDArray[np.float64]:
    rise = np.linspace(100.0, 110.0, 30, dtype=np.float64)
    drop = np.linspace(110.0, 85.0, 15, dtype=np.float64)
    bounce = np.linspace(85.0, 92.0, 5, dtype=np.float64)
    redrop = np.linspace(92.0, 78.0, 10, dtype=np.float64)
    return np.concatenate([rise, drop, bounce, redrop]).astype(np.float64)


def test_btc_mode_reversal_kill_applies_recovery_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4: env→config→compute chain — cooldown produces different risk-off output."""
    from src.domain.futures.strategy.market_regime import compute_reversal_risk_off_1d
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _reversal_config_from_env

    close = _close_whipsaw()
    monkeypatch.setenv("L2_REVERSAL_KILL", "1")
    monkeypatch.setenv("L2_REVERSAL_MODE", "btc")

    monkeypatch.setenv("L2_REVERSAL_RECOVERY_COOLDOWN", "0")
    cfg_0 = _reversal_config_from_env()
    out_0 = compute_reversal_risk_off_1d(
        close,
        dd_window=cfg_0.reversal_dd_window,
        dd_threshold=cfg_0.reversal_dd_threshold,
        mom_fast=cfg_0.reversal_mom_fast,
        mom_slow=cfg_0.reversal_mom_slow,
        persistence_bars=cfg_0.reversal_persistence_bars,
        recovery_cooldown_bars=cfg_0.reversal_recovery_cooldown_bars,
    )

    monkeypatch.setenv("L2_REVERSAL_RECOVERY_COOLDOWN", "8")
    cfg_8 = _reversal_config_from_env()
    out_8 = compute_reversal_risk_off_1d(
        close,
        dd_window=cfg_8.reversal_dd_window,
        dd_threshold=cfg_8.reversal_dd_threshold,
        mom_fast=cfg_8.reversal_mom_fast,
        mom_slow=cfg_8.reversal_mom_slow,
        persistence_bars=cfg_8.reversal_persistence_bars,
        recovery_cooldown_bars=cfg_8.reversal_recovery_cooldown_bars,
    )

    assert cfg_0.reversal_recovery_cooldown_bars == 0
    assert cfg_8.reversal_recovery_cooldown_bars == 8
    assert out_8.sum() >= out_0.sum(), "cooldown must extend risk-off bars"
