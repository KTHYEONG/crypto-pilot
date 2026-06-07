from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import RegimeConfig
from src.domain.futures.strategy.market_regime import (
    compute_market_regime_context,
    compute_risk_overlay,
    evaluate_regime_quality,
)


def _make_aligned() -> AlignedMarketData:
    t = 240
    calm_up = np.linspace(100.0, 135.0, 120, dtype=np.float64)
    break_down = np.linspace(135.0, 112.0, 40, dtype=np.float64)
    noisy_tail = 112.0 + np.cumsum(
        np.where(np.arange(80) % 2 == 0, 6.5, -7.5).astype(np.float64)
    )
    btc_close = np.concatenate([calm_up, break_down, noisy_tail])
    eth_close = btc_close * 0.97 + 2.0
    close = np.column_stack([btc_close, eth_close]).astype(np.float64)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, 2), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, 2), dtype=np.float64),
        active_mask=np.ones((t, 2), dtype=bool),
        warm_mask=np.ones((t, 2), dtype=bool),
        entry_block_mask=np.zeros((t, 2), dtype=bool),
        kill_mask=np.zeros((t, 2), dtype=bool),
        execution_cost_bps_2d=np.zeros((t, 2), dtype=np.float64),
    )


def test_compute_market_regime_context_when_overlay_based_returns_expected_shapes() -> None:
    # Arrange
    aligned = _make_aligned()

    # Act
    regime = compute_market_regime_context(aligned=aligned)
    overlay = compute_risk_overlay(aligned=aligned)

    # Assert
    assert regime.code_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.trend_score_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.vol_z_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.dispersion_z_1d.shape == (aligned.close_2d.shape[0],)
    assert regime.names().shape == (aligned.close_2d.shape[0],)
    assert set(np.unique(regime.names())).issubset(set(regime.name_by_code))
    assert overlay.overlay_mult_1d.shape == (aligned.close_2d.shape[0],)
    assert overlay.crisis_active_1d.shape == (aligned.close_2d.shape[0],)


def test_compute_risk_overlay_when_future_prices_perturbed_prefix_is_unchanged() -> None:
    # Arrange
    aligned = _make_aligned()
    perturbed_close = aligned.close_2d.copy()
    pivot = 140
    perturbed_close[pivot + 1 :, 0] *= 1.30
    perturbed_close[pivot + 1 :, 1] *= 0.85
    perturbed = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=perturbed_close,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=aligned.active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )

    # Act
    base_overlay = compute_risk_overlay(aligned=aligned)
    perturbed_overlay = compute_risk_overlay(aligned=perturbed)

    # Assert
    assert np.allclose(
        base_overlay.overlay_mult_1d[: pivot + 1],
        perturbed_overlay.overlay_mult_1d[: pivot + 1],
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    )


def test_compute_risk_overlay_when_volatility_increases_derisks_position_size() -> None:
    # Arrange
    aligned = _make_aligned()

    # Act
    overlay = compute_risk_overlay(aligned=aligned)

    # Assert
    calm_scale = float(np.median(overlay.vol_scale_1d[40:100]))
    stressed_scale = float(np.median(overlay.vol_scale_1d[-40:]))
    assert stressed_scale < calm_scale
    assert stressed_scale <= 1.0


def test_compute_risk_overlay_when_regime_break_occurs_flags_cusum_crisis() -> None:
    # Arrange
    aligned = _make_aligned()
    cfg = RegimeConfig(crisis_target_arl_bars=40, crisis_gross_floor=0.20)

    # Act
    overlay = compute_risk_overlay(aligned=aligned, cfg=cfg)

    # Assert
    assert bool(np.any(overlay.crisis_active_1d[120:200]))
    assert np.allclose(
        overlay.overlay_mult_1d[overlay.crisis_active_1d],
        cfg.crisis_gross_floor,
        atol=1e-12,
        rtol=0.0,
    )


def test_evaluate_regime_quality_when_overlay_is_helpful_passes_quality_gate() -> None:
    # Arrange
    aligned = _make_aligned()
    cfg = RegimeConfig(crisis_target_arl_bars=40, regime_min_n_eff=60)
    overlay = compute_risk_overlay(aligned=aligned, cfg=cfg)
    cal_eval_mask = np.ones(aligned.close_2d.shape[0], dtype=bool)
    base_edge = np.where(
        overlay.crisis_active_1d,
        -0.0030,
        0.0015 + 0.0020 * (overlay.overlay_mult_1d - np.mean(overlay.overlay_mult_1d)),
    )
    # Act
    report = evaluate_regime_quality(
        aligned=aligned,
        cfg=cfg,
        base_edge_1d=base_edge,
        cal_eval_mask=cal_eval_mask,
    )

    # Assert
    assert report.leakage_ok is True
    assert report.persistence_dwell >= 6.0
    assert report.overlay_lift_bps > 0.0
    assert report.overlay_lift_tstat >= cfg.regime_overlay_min_lift_tstat
    assert report.crisis_precision_ok is True
    assert report.passed is True
