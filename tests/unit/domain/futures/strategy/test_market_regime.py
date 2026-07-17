from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import RegimeConfig
from src.domain.futures.strategy.market_regime import (
    _persistence_targeted_band,
    _schmitt_directional_state,
    apply_regime_cap_release_cooldown,
    compute_market_regime_context,
    compute_risk_overlay,
    compute_trend_efficiency_1d,
    evaluate_regime_quality,
)
from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation


def _make_aligned() -> AlignedMarketData:
    t = 240
    calm_up = np.linspace(100.0, 135.0, 120, dtype=np.float64)
    break_down = np.linspace(135.0, 112.0, 40, dtype=np.float64)
    noisy_tail = 112.0 + np.cumsum(np.where(np.arange(80) % 2 == 0, 6.5, -7.5).astype(np.float64))
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


def test_btc_index_raises_when_no_btc_symbol_present() -> None:
    from src.domain.futures.strategy.market_regime import _btc_index

    symbols = ("ETHUSDT", "SOLUSDT")
    with pytest.raises(ValueError, match="BTC"):
        _btc_index(symbols)


def test_btc_index_returns_index_when_btc_present() -> None:
    from src.domain.futures.strategy.market_regime import _btc_index

    assert _btc_index(("ETHUSDT", "BTCUSDT", "SOLUSDT")) == 1
    assert _btc_index(("BTCUSDT",)) == 0


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


# ---------------------------------------------------------------------------
# P1: Schmitt hysteresis + persistence-targeted band tests (S8-S12)
# ---------------------------------------------------------------------------


def test_schmitt_hysteresis_reduces_flips_when_snr_oscillates_near_zero() -> None:
    # Arrange
    t = 100
    snr = np.tile([0.02, -0.02], t // 2).astype(np.float64)
    stateless_bull = (snr >= 0.0).astype(np.int8)
    stateless_flips = int(np.sum(stateless_bull[1:] != stateless_bull[:-1]))

    # Act
    hysteresis_state = _schmitt_directional_state(snr, enter_theta=0.35, exit_theta=0.15)
    hysteresis_flips = int(np.sum(hysteresis_state[1:] != hysteresis_state[:-1]))

    # Assert
    assert hysteresis_flips < stateless_flips
    assert hysteresis_flips <= 2  # near-zero 진동이면 초기 최대 1회 전환만


def test_schmitt_bull_state_maintained_above_exit_when_snr_stays_positive() -> None:
    # Arrange
    snr = np.array([0.4, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)

    # Act
    state = _schmitt_directional_state(snr, enter_theta=0.35, exit_theta=0.15)

    # Assert — t=0 진입, t=1..4 BULL 유지 (0.2 > -exit_theta=-0.15)
    assert all(s == 1 for s in state)


def test_schmitt_bull_exits_to_neutral_when_snr_crosses_negative_exit_theta() -> None:
    # Arrange
    snr = np.array([0.4, -0.2, -0.2], dtype=np.float64)

    # Act
    state = _schmitt_directional_state(snr, enter_theta=0.35, exit_theta=0.15)

    # Assert
    assert state[0] == 1  # BULL 진입 (snr=0.4 >= enter_theta=0.35)
    assert state[1] == 0  # NEUTRAL (snr=-0.2 <= -exit_theta=-0.15)
    assert state[2] == 0  # NEUTRAL 유지


def test_schmitt_bear_exits_to_neutral_when_snr_crosses_positive_exit_theta() -> None:
    # Arrange
    snr = np.array([-0.4, 0.2, 0.2], dtype=np.float64)

    # Act
    state = _schmitt_directional_state(snr, enter_theta=0.35, exit_theta=0.15)

    # Assert
    assert state[0] == 2  # BEAR 진입 (snr=-0.4 <= -enter_theta=-0.35)
    assert state[1] == 0  # NEUTRAL (snr=0.2 >= +exit_theta=0.15)
    assert state[2] == 0  # NEUTRAL 유지


def test_schmitt_nan_snr_preserves_current_state() -> None:
    # Arrange — NaN 구간에서 이전 상태 유지
    snr = np.array([0.4, np.nan, np.nan, -0.2], dtype=np.float64)

    # Act
    state = _schmitt_directional_state(snr, enter_theta=0.35, exit_theta=0.15)

    # Assert
    assert state[0] == 1  # BULL 진입
    assert state[1] == 1  # NaN → 상태 유지
    assert state[2] == 1  # NaN → 상태 유지
    assert state[3] == 0  # snr=-0.2 <= -0.15 → NEUTRAL


def test_persistence_targeted_band_shape_and_default_before_min_n() -> None:
    # Arrange
    t = 120
    snr_abs = np.abs(np.random.default_rng(42).standard_normal(t))

    # Act
    band = _persistence_targeted_band(snr_abs, target_dwell=6.0, min_n_eff=60)

    # Assert
    assert band.shape == (t,)
    # min_n_eff 이전 구간은 기본값 0.5
    assert np.all(band[:60] == 0.5)
    # min_n_eff 이후 구간은 양수 (quantile 결과)
    assert np.all(band[60:] > 0.0)


def test_persistence_targeted_band_causal_no_lookahead() -> None:
    # Arrange — 후반부를 크게 변경해도 전반부 band 불변
    rng = np.random.default_rng(7)
    snr_abs = np.abs(rng.standard_normal(200))
    pivot = 100

    perturbed = snr_abs.copy()
    perturbed[pivot + 1 :] *= 10.0  # 후반부 magnitude 10배

    # Act
    band_base = _persistence_targeted_band(snr_abs, target_dwell=8.0, min_n_eff=30)
    band_perturbed = _persistence_targeted_band(perturbed, target_dwell=8.0, min_n_eff=30)

    # Assert — pivot 이전 구간은 동일 (causal)
    assert np.allclose(band_base[: pivot + 1], band_perturbed[: pivot + 1], atol=1e-12)


def test_compute_market_regime_context_p1_code_range_valid() -> None:
    # Arrange
    aligned = _make_aligned()

    # Act
    regime = compute_market_regime_context(aligned=aligned)

    # Assert — P1 통합 후에도 code가 0-5 범위
    assert np.all((regime.code_1d >= 0) & (regime.code_1d <= 5))


def test_compute_market_regime_context_p1_leakage_unchanged() -> None:
    """P1 도입 후에도 causal(no-lookahead) 보장 확인."""
    # Arrange
    aligned = _make_aligned()
    pivot = 120
    perturbed_close = aligned.close_2d.copy()
    perturbed_close[pivot + 1 :, :] *= 1.15
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

    # Assert — overlay는 causal이므로 pivot 이전 동일해야 함
    assert np.allclose(
        base_overlay.overlay_mult_1d[: pivot + 1],
        perturbed_overlay.overlay_mult_1d[: pivot + 1],
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
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


class TestRegimeCompression:
    """6-state → 3-state regime compression."""

    def test_compress_all_states(self) -> None:
        from src.domain.futures.strategy.market_regime import compress_regime_codes

        codes = np.array([0, 1, 2, 3, 4, 5], dtype=np.int8)
        result = compress_regime_codes(codes)
        expected = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_compress_empty(self) -> None:
        from src.domain.futures.strategy.market_regime import compress_regime_codes

        codes = np.array([], dtype=np.int8)
        result = compress_regime_codes(codes)
        assert result.size == 0

    def test_compress_single_state(self) -> None:
        from src.domain.futures.strategy.market_regime import compress_regime_codes

        for src, dst in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]:
            codes = np.array([src], dtype=np.int8)
            result = compress_regime_codes(codes)
            assert result[0] == dst, f"code {src} → {result[0]} (expected {dst})"

    def test_trend_efficiency_1d_is_in_regime_context(self) -> None:
        from tests.unit.domain.futures.strategy.test_market_regime import _make_aligned

        aligned = _make_aligned()
        regime = compute_market_regime_context(aligned=aligned)
        assert hasattr(regime, "trend_efficiency_1d")
        assert regime.trend_efficiency_1d.shape == (aligned.close_2d.shape[0],)

    def test_trend_efficiency_1d_causal_no_lookahead(self) -> None:
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0], dtype=np.float64)
        er = compute_trend_efficiency_1d(close, window=4)
        assert np.isnan(er[:4]).all()
        assert np.isfinite(er[4:]).all()


# ---------------------------------------------------------------------------
# L2 Crisis Regime Cap Release Cooldown
# ---------------------------------------------------------------------------

def test_apply_regime_cap_release_cooldown_noop_when_zero() -> None:
    code = np.asarray([0, 1, 2, 0, 1], dtype=np.int8)
    out = apply_regime_cap_release_cooldown(code, cooldown_bars=0)
    assert np.array_equal(out, code)


def test_apply_regime_cap_release_cooldown_never_delays_bear_crisis_entry() -> None:
    code = np.asarray([0, 0, 0, 2, 2], dtype=np.int8)
    out = apply_regime_cap_release_cooldown(code, cooldown_bars=5)
    assert out[3] == 2


def test_apply_regime_cap_release_cooldown_delays_bull_return_after_crisis() -> None:
    code = np.asarray([2, 2, 0, 0, 0, 0], dtype=np.int8)
    out = apply_regime_cap_release_cooldown(code, cooldown_bars=3)
    expected = np.asarray([2, 2, 1, 1, 1, 0], dtype=np.int8)
    assert np.array_equal(out, expected)


def test_apply_regime_cap_release_cooldown_substitutes_bear_not_crisis() -> None:
    code = np.asarray([2, 0, 0], dtype=np.int8)
    out = apply_regime_cap_release_cooldown(code, cooldown_bars=3)
    # cooldown-substituted bars become bear(1), never crisis(2)
    assert out[1] == 1
    assert (out[out == 2]).size == 1  # only original crisis at bar 0 remains 2


def test_apply_regime_cap_release_cooldown_negative_cooldown_raises() -> None:
    code = np.asarray([0, 1, 2], dtype=np.int8)
    with pytest.raises(ValueError, match="cooldown_bars"):
        apply_regime_cap_release_cooldown(code, cooldown_bars=-1)


def test_run_awf_simulation_wires_regime_cap_release_cooldown_before_cap_call() -> None:
    import inspect

    source = inspect.getsource(_run_awf_simulation)
    assert "apply_regime_cap_release_cooldown(" in source
    assert "_regime_code_1d_for_cap[t]" in source
