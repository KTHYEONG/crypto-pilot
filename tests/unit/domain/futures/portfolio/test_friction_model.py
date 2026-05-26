"""Tests for transaction friction and market impact calculations."""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.friction_model import (
    FrictionConfig,
    compute_coarse_precharge_bps,
    compute_impact_bps,
    resolve_cost_snapshot,
)


def test_compute_coarse_precharge_bps_math() -> None:
    """Test mathematical accuracy of compute_coarse_precharge_bps."""
    cfg = FrictionConfig(
        taker_fee_bps=4.0,
        maker_share=0.5,
        maker_rebate_bps=-2.0,
        latency_buffer_bps=0.5,
        tick_cost_bps=0.0,
    )

    # spread=5bps, impact=3bps, funding=1bps
    # fee_bps = 4.0 + 0.5 * (-2.0 - 4.0) = 1.0 bps
    # total = 1.0 (fee) + 5.0 (spread) + 3.0 (impact) + 0.0 (tick)
    # + 0.5 (latency) + 1.0 (funding) = 10.5 bps
    total = compute_coarse_precharge_bps(
        spread_bps=5.0,
        impact_bps=3.0,
        funding_proxy_bps=1.0,
        cfg=cfg,
    )

    assert abs(total - 10.5) < 1e-9


def test_compute_impact_bps_sqrt_proportionality() -> None:
    """Verify that impact scale behaves proportionally to sqrt of size."""
    # order size: 10,000 USDT, ADV: 1,000,000 USDT, sigma: 0.02 (2% daily vol)
    impact_small = compute_impact_bps(
        sigma_1d=0.02, order_notional=10000.0, adv_30d=1000000.0, k=0.5
    )

    # 4x order size -> 40,000 USDT. Impact should double (sqrt(4) = 2)
    impact_large = compute_impact_bps(
        sigma_1d=0.02, order_notional=40000.0, adv_30d=1000000.0, k=0.5
    )

    assert abs(impact_large - 2.0 * impact_small) < 1e-9


def test_compute_impact_bps_zero_adv_safeguard() -> None:
    """Test that zero ADV or zero order notional safely returns 0.0."""
    res_zero_adv = compute_impact_bps(sigma_1d=0.02, order_notional=10000.0, adv_30d=0.0)
    assert res_zero_adv == 0.0

    res_zero_order = compute_impact_bps(sigma_1d=0.02, order_notional=0.0, adv_30d=1000000.0)
    assert res_zero_order == 0.0


def test_maker_share_impact_on_fee() -> None:
    """Verify how maker vs taker share shifts the transaction costs."""
    # Pure Taker (maker_share = 0.0) -> fee should be taker_fee_bps (4.0)
    cfg_taker = FrictionConfig(maker_share=0.0, taker_fee_bps=4.0, maker_rebate_bps=-2.0)
    total_taker = compute_coarse_precharge_bps(
        spread_bps=0.0, impact_bps=0.0, funding_proxy_bps=0.0, cfg=cfg_taker
    )

    # Pure Maker (maker_share = 1.0) -> fee should be maker_rebate_bps (-2.0)
    cfg_maker = FrictionConfig(maker_share=1.0, taker_fee_bps=4.0, maker_rebate_bps=-2.0)
    total_maker = compute_coarse_precharge_bps(
        spread_bps=0.0, impact_bps=0.0, funding_proxy_bps=0.0, cfg=cfg_maker
    )

    # Difference in pre-charge should equal the fee gap.
    # 4.0 - (-2.0) = 6.0 bps
    # Plus standard latency buffer (0.5) remains same on both
    assert abs((total_taker - cfg_taker.latency_buffer_bps) - 4.0) < 1e-9
    assert abs((total_maker - cfg_maker.latency_buffer_bps) - (-2.0)) < 1e-9


def test_cost_snapshot_bps_fraction_conversion() -> None:
    """CostSnapshot bps/fraction conversion should be exact."""
    per_symbol = np.array([[10.0, 25.0]], dtype=np.float64)
    snapshot = resolve_cost_snapshot(execution_cost_bps_2d=per_symbol, shape=(1, 2))
    assert snapshot.execution_cost_bps_source == "per_symbol"
    assert np.allclose(snapshot.execution_cost_fraction_2d, per_symbol / 10000.0)


def test_cost_snapshot_fallback_source_and_shape() -> None:
    """Missing per-symbol costs should fallback to global round-trip cost."""
    snapshot = resolve_cost_snapshot(execution_cost_bps_2d=None, shape=(2, 3))
    assert snapshot.execution_cost_bps_source == "fallback_global"
    assert snapshot.execution_cost_bps_2d.shape == (2, 3)
    assert np.allclose(
        snapshot.execution_cost_fraction_2d,
        snapshot.execution_cost_bps_2d / 10000.0,
    )
