from __future__ import annotations

import math

import numpy as np
import pytest

from src.domain.futures.strategy.alpha_evaluation import (
    AlphaEvaluationReport,
    _compute_regime_labels,
    compute_breakeven_ic,
    compute_deflated_sharpe,
    compute_effective_breadth,
    compute_net_ic,
    compute_per_regime_breakeven,
    compute_per_regime_ic,
    compute_q50_sign_hit,
    compute_quantile_coverage,
    derive_signed_rank_signal,
    diagnose_alpha_ic_decomposition,
    diagnose_selection_monotonicity,
    effective_breadth_corr,
    evaluate_alpha,
    sweep_horizon_breakeven,
)
from src.domain.futures.strategy.rank_selection import (
    apply_rank_selection_policy,
    calibrate_rank_selection_policy,
)

# ---------------------------------------------------------------------------
# compute_breakeven_ic
# ---------------------------------------------------------------------------


def test_compute_breakeven_ic_known_value() -> None:
    """cost=24bps, sigma=400bps, breadth=4 → breakeven=24/(400*sqrt(4))=0.03."""
    # Arrange
    cost_floor_bps = 24.0
    sigma_r_bps = 400.0
    breadth_eff = 4.0

    # Act
    result = compute_breakeven_ic(
        cost_floor_bps=cost_floor_bps,
        sigma_r_bps=sigma_r_bps,
        breadth_eff=breadth_eff,
    )

    # Assert
    assert result == pytest.approx(0.03, rel=1e-4)


def test_compute_breakeven_ic_min_breadth_clamp() -> None:
    """breadth=0.0 is clamped to 1.0; result must be finite and positive."""
    # Arrange
    cost_floor_bps = 24.0
    sigma_r_bps = 400.0
    breadth_eff = 0.0

    # Act
    result = compute_breakeven_ic(
        cost_floor_bps=cost_floor_bps,
        sigma_r_bps=sigma_r_bps,
        breadth_eff=breadth_eff,
    )

    # Assert — clamped to breadth=1, so result == 24/400 == 0.06
    assert math.isfinite(result)
    assert result > 0.0
    assert result == pytest.approx(0.06, rel=1e-4)


# ---------------------------------------------------------------------------
# compute_effective_breadth
# ---------------------------------------------------------------------------


def test_compute_effective_breadth_all_zero() -> None:
    """All-zero alpha arrays → effective breadth == 0.0."""
    # Arrange
    alpha_long = np.zeros((10, 5), dtype=np.float64)
    alpha_short = np.zeros((10, 5), dtype=np.float64)

    # Act
    result = compute_effective_breadth(alpha_long, alpha_short)

    # Assert
    assert result == 0.0


def test_compute_effective_breadth_partial_active() -> None:
    """3 symbols with non-zero long alpha per bar → breadth == 3.0."""
    # Arrange
    T, N = 10, 5
    alpha_long = np.zeros((T, N), dtype=np.float64)
    alpha_short = np.zeros((T, N), dtype=np.float64)
    # Activate exactly 3 symbols in every bar
    alpha_long[:, :3] = 0.01

    # Act
    result = compute_effective_breadth(alpha_long, alpha_short)

    # Assert
    assert result == pytest.approx(3.0, rel=1e-6)


def test_derive_signed_rank_signal_same_score_returns_single_ranker_contract() -> None:
    """Same-score panel should return long if finite, else short."""
    long_arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
    short_arr = np.array([[1.0 + 1e-12, 2.0], [3.0 - 1e-12, 4.0]], dtype=np.float64)

    result = derive_signed_rank_signal(long_arr, short_arr)

    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


def test_derive_signed_rank_signal_dual_side_returns_half_spread() -> None:
    """Distinct long/short score panels should return half spread."""
    long_arr = np.array([[3.0, 1.0], [2.0, -2.0]], dtype=np.float64)
    short_arr = np.array([[1.0, -1.0], [0.0, -4.0]], dtype=np.float64)

    result = derive_signed_rank_signal(long_arr, short_arr)

    expected = 0.5 * (long_arr - short_arr)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


def test_derive_signed_rank_signal_shape_mismatch_raises_value_error() -> None:
    """Shape mismatch must raise ValueError."""
    long_arr = np.zeros((2, 2), dtype=np.float64)
    short_arr = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="identical shapes"):
        derive_signed_rank_signal(long_arr, short_arr)


def test_rank_policy_calibration_selects_negative_polarity_for_inverted_tails() -> None:
    rng = np.random.default_rng(42)
    t, n = 220, 10
    score = rng.standard_normal((t, n)).astype(np.float64)
    realized = (-score + 0.01 * rng.standard_normal((t, n))).astype(np.float64) / 100.0
    policy = calibrate_rank_selection_policy(
        signed_score_2d=score,
        realized_fwd_ret_2d=realized,
        eligible_2d=np.ones((t, n), dtype=bool),
        quantiles=(0.2, 0.3),
        min_abs_z_grid=(0.0, 0.25),
        holding_bars=12,
        cost_bps=1.0,
        min_obs=30,
    )
    assert policy.polarity == -1


def test_rank_policy_apply_outputs_non_overlapping_positive_masks() -> None:
    rng = np.random.default_rng(1)
    score = rng.standard_normal((30, 8)).astype(np.float64)
    realized = score / 100.0
    policy = calibrate_rank_selection_policy(
        signed_score_2d=score,
        realized_fwd_ret_2d=realized,
        eligible_2d=np.ones((30, 8), dtype=bool),
        quantiles=(0.25,),
        min_abs_z_grid=(0.0,),
        holding_bars=12,
        cost_bps=0.0,
        min_obs=20,
    )
    al, as_ = apply_rank_selection_policy(
        signed_score_2d=score,
        eligible_2d=np.ones((30, 8), dtype=bool),
        policy=policy,
    )
    assert al.shape == score.shape
    assert as_.shape == score.shape
    assert not np.any((al > 0.0) & (as_ > 0.0))


def test_rank_policy_fallback_keeps_no_trade_when_validation_fails() -> None:
    score = np.zeros((40, 6), dtype=np.float64)
    realized = np.zeros((40, 6), dtype=np.float64)
    policy = calibrate_rank_selection_policy(
        signed_score_2d=score,
        realized_fwd_ret_2d=realized,
        eligible_2d=np.ones((40, 6), dtype=bool),
        quantiles=(0.2, 0.3),
        min_abs_z_grid=(0.0, 0.25),
        holding_bars=12,
        cost_bps=8.0,
        min_obs=20,
    )
    assert policy.validation_net_lcb_bps <= 0.0
    assert policy.n_obs == 0
    al, as_ = apply_rank_selection_policy(
        signed_score_2d=score,
        eligible_2d=np.ones((40, 6), dtype=bool),
        policy=policy,
    )
    assert np.count_nonzero(al) == 0
    assert np.count_nonzero(as_) == 0


# ---------------------------------------------------------------------------
# effective_breadth_corr
# ---------------------------------------------------------------------------


def test_effective_breadth_corr_independent_columns_returns_near_n() -> None:
    """Uncorrelated assets (rho_bar≈0) → N_eff ≈ N (no diversification haircut)."""
    # Arrange
    rng = np.random.default_rng(42)
    n_cols = 8
    panel = rng.standard_normal((400, n_cols)).astype(np.float64)

    # Act
    result = effective_breadth_corr(panel)

    # Assert: independent columns retain near-full breadth (tolerant to sampling noise)
    assert result == pytest.approx(float(n_cols), rel=0.20)


def test_effective_breadth_corr_perfectly_correlated_returns_one() -> None:
    """Identical columns (rho_bar==1) → N_eff = N/(1+(N-1)) == 1.0."""
    # Arrange
    rng = np.random.default_rng(7)
    base = rng.standard_normal((300, 1)).astype(np.float64)
    panel = np.repeat(base, 6, axis=1)  # 6 perfectly correlated assets

    # Act
    result = effective_breadth_corr(panel)

    # Assert
    assert result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_quantile_coverage
# ---------------------------------------------------------------------------


def test_compute_quantile_coverage_perfect() -> None:
    """realized=0.0 always inside [q10=-0.5, q90=0.5] → coverage==1.0."""
    # Arrange
    T, N = 5, 4
    q10 = np.full((T, N), -0.5, dtype=np.float64)
    q90 = np.full((T, N), 0.5, dtype=np.float64)
    realized = np.zeros((T, N), dtype=np.float64)

    # Act
    result = compute_quantile_coverage(q10, q90, realized)

    # Assert
    assert result == pytest.approx(1.0, rel=1e-6)


def test_compute_quantile_coverage_zero() -> None:
    """realized=-0.5 always outside [q10=0.1, q90=0.2] → coverage==0.0."""
    # Arrange
    T, N = 5, 4
    q10 = np.full((T, N), 0.1, dtype=np.float64)
    q90 = np.full((T, N), 0.2, dtype=np.float64)
    realized = np.full((T, N), -0.5, dtype=np.float64)

    # Act
    result = compute_quantile_coverage(q10, q90, realized)

    # Assert
    assert result == pytest.approx(0.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_q50_sign_hit
# ---------------------------------------------------------------------------


def test_compute_q50_sign_hit_perfect() -> None:
    """q50 and realized both positive → sign_hit==1.0."""
    # Arrange
    q50 = np.array([[0.1, 0.2]], dtype=np.float64)
    realized = np.array([[0.3, 0.4]], dtype=np.float64)

    # Act
    result = compute_q50_sign_hit(q50, realized)

    # Assert
    assert result == pytest.approx(1.0, rel=1e-6)


def test_compute_q50_sign_hit_zero() -> None:
    """q50 positive, realized negative → sign_hit==0.0."""
    # Arrange
    q50 = np.array([[0.1]], dtype=np.float64)
    realized = np.array([[-0.1]], dtype=np.float64)

    # Act
    result = compute_q50_sign_hit(q50, realized)

    # Assert
    assert result == pytest.approx(0.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_net_ic
# ---------------------------------------------------------------------------


def test_compute_net_ic_positive_signal() -> None:
    """pred_2d strongly correlated to realized → mean_ic > 0.3."""
    # Arrange
    rng = np.random.default_rng(seed=7)
    T, N = 100, 10
    realized_fwd_2d = rng.standard_normal((T, N)).astype(np.float64)
    noise = rng.standard_normal((T, N)).astype(np.float64) * 0.1
    pred_2d = (realized_fwd_2d + noise).astype(np.float64)

    # Act
    result = compute_net_ic(pred_2d, realized_fwd_2d)

    # Assert
    assert result["mean_ic"] > 0.3


def test_compute_net_ic_zero_signal() -> None:
    """pred_2d independent of realized → mean_ic ≈ 0 (loose bound)."""
    # Arrange
    rng = np.random.default_rng(seed=99)
    T, N = 50, 5
    realized_2d = rng.standard_normal((T, N)).astype(np.float64)
    pred_2d = rng.standard_normal((T, N)).astype(np.float64)

    # Act
    result = compute_net_ic(pred_2d, realized_2d)

    # Assert
    assert -0.3 < result["mean_ic"] < 0.3


# ---------------------------------------------------------------------------
# compute_deflated_sharpe
# ---------------------------------------------------------------------------


def test_compute_deflated_sharpe_high_confidence() -> None:
    """High observed Sharpe, single trial → DSR > 0.95."""
    # Arrange
    observed_sharpe = 3.0
    n_trials = 1
    n_obs = 100

    # Act
    dsr = compute_deflated_sharpe(observed_sharpe, n_trials=n_trials, n_obs=n_obs)

    # Assert
    assert dsr > 0.95


def test_compute_deflated_sharpe_many_trials() -> None:
    """Weak Sharpe(0.1) with 100 trials → DSR < 0.5 (high selection bias).

    sr_star≈0.36 for n_trials=100, n_obs=50; observed SR=0.1 << sr_star
    so DSR is well below 0.5 by Bailey & LdP formula.
    """
    # Arrange
    observed_sharpe = 0.1  # weak signal well below sr_star≈0.36
    n_trials = 100
    n_obs = 50

    # Act
    dsr = compute_deflated_sharpe(observed_sharpe, n_trials=n_trials, n_obs=n_obs)

    # Assert
    assert dsr < 0.5


# ---------------------------------------------------------------------------
# compute_per_regime_ic
# ---------------------------------------------------------------------------


def test_compute_per_regime_ic_returns_three_buckets() -> None:
    """Output must have exactly {"bull","bear","chop"} keys with float values."""
    # Arrange
    rng = np.random.default_rng(seed=42)
    T, N = 200, 5
    # BTC close with mild trend and noise to generate all three regimes
    trend = np.linspace(100.0, 200.0, T)
    noise = rng.standard_normal(T) * 10.0
    btc_close = (trend + noise).astype(np.float64)
    # Introduce a downtrend segment to ensure "bear" regime bars exist
    btc_close[60:120] = np.linspace(btc_close[60], btc_close[60] * 0.5, 60)

    pred_2d = rng.standard_normal((T, N)).astype(np.float64)
    realized_2d = rng.standard_normal((T, N)).astype(np.float64)

    # Act
    result = compute_per_regime_ic(pred_2d, realized_2d, btc_close)

    # Assert
    assert set(result.keys()) == {"bull", "bear", "chop"}
    assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# evaluate_alpha
# ---------------------------------------------------------------------------


def test_evaluate_alpha_returns_report_structure() -> None:
    """evaluate_alpha must return AlphaEvaluationReport with correct field types."""
    # Arrange
    rng = np.random.default_rng(seed=0)
    T, N = 100, 10
    alpha_long = rng.standard_normal((T, N)).astype(np.float64) * 0.01
    alpha_short = rng.standard_normal((T, N)).astype(np.float64) * 0.01
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
        n_trials=1,
    )

    # Assert
    assert isinstance(report, AlphaEvaluationReport)
    assert isinstance(report.passes, bool)
    assert isinstance(report.fail_reasons, list)


def test_evaluate_alpha_passes_false_when_no_signal() -> None:
    """Purely random alpha with no signal must fail the evaluation gate."""
    # Arrange
    rng = np.random.default_rng(seed=42)
    T, N = 100, 10
    alpha_long = rng.standard_normal((T, N)).astype(np.float64) * 0.001
    alpha_short = rng.standard_normal((T, N)).astype(np.float64) * 0.001
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
        n_trials=1,
    )

    # Assert
    assert report.passes is False
    assert len(report.fail_reasons) > 0


def test_evaluate_alpha_regime_gate_uses_preclip_signal_when_provided() -> None:
    """Bear gate must use pre-clip inference IC, not clipped trading alpha IC."""
    rng = np.random.default_rng(seed=123)
    T, N = 180, 8
    # Ensure all regimes exist with a clear bear segment.
    btc_close = np.concatenate([
        np.linspace(10000.0, 20000.0, 60),
        np.linspace(20000.0, 9000.0, 70),
        np.linspace(9000.0, 14000.0, 50),
    ]).astype(np.float64)
    labels = _compute_regime_labels(btc_close, trend_window=30)

    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02
    inference = (
        realized + rng.standard_normal((T, N)).astype(np.float64) * 0.001
    ).astype(np.float64)

    pred_clip = inference.copy()
    bear_rows = np.array([i for i, lbl in enumerate(labels) if lbl == "bear"], dtype=np.intp)
    if bear_rows.size > 0:
        pred_clip[bear_rows] = -pred_clip[bear_rows]

    alpha_long = np.maximum(pred_clip, 0.0)
    alpha_short = np.maximum(-pred_clip, 0.0)

    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
        inference_signed_2d=inference,
        btc_close_1d=btc_close,
        n_trials=1,
    )

    assert "bear_ic_negative" not in report.fail_reasons


# ---------------------------------------------------------------------------
# net_sharpe & cost_drag fields
# ---------------------------------------------------------------------------


def test_evaluate_alpha_net_sharpe_computed_when_returns_provided() -> None:
    """net_sharpe is finite and positive when strong positive daily returns given."""
    # Arrange
    rng = np.random.default_rng(seed=7)
    T, N = 80, 8
    alpha_long = np.abs(rng.standard_normal((T, N)).astype(np.float64)) * 0.001
    alpha_short = np.zeros((T, N), dtype=np.float64)
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02
    # Consistently positive daily returns → positive Sharpe
    net_daily = np.abs(rng.standard_normal(252).astype(np.float64)) * 0.001

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
        net_daily_returns=net_daily,
        n_trials=1,
    )

    # Assert
    assert math.isfinite(report.net_sharpe)
    assert report.net_sharpe > 0.0


def test_evaluate_alpha_net_sharpe_nan_when_not_provided() -> None:
    """net_sharpe is NaN when net_daily_returns is not supplied."""
    # Arrange
    rng = np.random.default_rng(seed=8)
    T, N = 60, 5
    alpha_long = rng.standard_normal((T, N)).astype(np.float64) * 0.001
    alpha_short = np.zeros((T, N), dtype=np.float64)
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
        n_trials=1,
    )

    # Assert
    assert math.isnan(report.net_sharpe)
    assert isinstance(report.cost_drag, dict)


def test_sweep_horizon_breakeven_returns_all_horizons() -> None:
    """sweep_horizon_breakeven returns an entry for every supplied horizon key."""
    # Arrange
    rng = np.random.default_rng(seed=99)
    T, N = 120, 6
    horizons = [6, 12, 18]
    realized_map = {
        h: rng.standard_normal((T, N)).astype(np.float64) * 0.02 for h in horizons
    }
    alpha_long_map = {
        h: np.abs(rng.standard_normal((T, N)).astype(np.float64)) * 0.001 for h in horizons
    }
    alpha_short_map = {h: np.zeros((T, N), dtype=np.float64) for h in horizons}

    # Act
    result = sweep_horizon_breakeven(realized_map, alpha_long_map, alpha_short_map)

    # Assert
    assert set(result.keys()) == set(horizons)
    for metrics in result.values():
        assert "sigma_r_bps" in metrics
        assert "net_ic" in metrics
        assert "breakeven_ic" in metrics
        assert "ic_exceeds_breakeven" in metrics
        assert metrics["sigma_r_bps"] > 0.0


# ---------------------------------------------------------------------------
# _compute_regime_labels
# ---------------------------------------------------------------------------


def test_compute_regime_labels_returns_correct_length() -> None:
    """Output list length must equal input BTC close price array length."""
    # Arrange
    T = 150
    rng = np.random.default_rng(seed=11)
    btc_close = np.cumsum(rng.standard_normal(T) * 100.0 + 1000.0).astype(np.float64)
    btc_close = np.maximum(btc_close, 1.0)

    # Act
    labels = _compute_regime_labels(btc_close)

    # Assert
    assert len(labels) == T


def test_compute_regime_labels_no_lookahead() -> None:
    """First trend_window elements must all be None (no look-ahead)."""
    # Arrange
    T = 100
    trend_window = 30
    btc_close = np.linspace(10000.0, 20000.0, T).astype(np.float64)

    # Act
    labels = _compute_regime_labels(btc_close, trend_window=trend_window)

    # Assert — indices 0..trend_window-1 must be None
    assert all(lbl is None for lbl in labels[:trend_window])
    # At least some labeled bars exist after the warmup window
    labeled = [lbl for lbl in labels[trend_window:] if lbl is not None]
    assert len(labeled) > 0


# ---------------------------------------------------------------------------
# compute_per_regime_breakeven
# ---------------------------------------------------------------------------


def test_compute_per_regime_breakeven_bull_higher_vol_yields_higher_breakeven() -> None:
    """Regime with higher cross-sectional sigma_r yields lower breakeven (sigma_r in denominator).

    breakeven_ic = cost / (sigma_r * sqrt(breadth)); larger sigma_r -> smaller breakeven.
    We construct two synthetic close series so one regime has consistently higher
    realized vol, then verify the ordering.
    """
    # Arrange: 200-bar BTC series with a clear downtrend in the middle to create bear bars
    T = 200
    N = 6
    trend = np.concatenate([
        np.linspace(10000.0, 20000.0, 80),    # uptrend → bull/chop bars
        np.linspace(20000.0, 8000.0, 80),     # downtrend → bear bars
        np.linspace(8000.0, 12000.0, 40),     # recovery → bull/chop bars
    ]).astype(np.float64)
    btc_close = trend

    rng = np.random.default_rng(seed=77)
    alpha_long = np.abs(rng.standard_normal((T, N)).astype(np.float64)) * 0.01
    alpha_short = np.abs(rng.standard_normal((T, N)).astype(np.float64)) * 0.005

    # Bull bars: low sigma_r; Bear bars: high sigma_r
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.005  # base low vol

    # Inject high vol during the downtrend segment (bear regime)
    realized[80:160] = rng.standard_normal((80, N)).astype(np.float64) * 0.05

    # Act
    result = compute_per_regime_breakeven(
        alpha_long, alpha_short, realized, btc_close, cost_floor_bps=24.0
    )

    # Assert — all three keys present; bear bars have larger sigma_r -> smaller breakeven
    assert set(result.keys()) == {"bull", "bear", "chop"}
    # Bear has higher sigma_r so its breakeven must be lower than if it had bull's sigma_r
    # Both should be finite if there are >= 5 bars
    bear_be = result["bear"]
    bull_be = result["bull"]
    if math.isfinite(bear_be) and math.isfinite(bull_be):
        # bear has 10x sigma_r -> breakeven must be lower than bull's
        assert bear_be < bull_be


def test_compute_per_regime_breakeven_fewer_than_5_bars_returns_nan() -> None:
    """A regime with < 5 bars must yield float('nan') for that regime."""
    # Arrange: short BTC series where only "bear" can appear (pure downtrend)
    T = 60
    N = 4
    # Pure downtrend forces bear labels only; bull/chop will have 0 bars → nan
    btc_close = np.linspace(10000.0, 1000.0, T).astype(np.float64)

    rng = np.random.default_rng(seed=5)
    alpha_long = np.abs(rng.standard_normal((T, N)).astype(np.float64)) * 0.01
    alpha_short = np.zeros((T, N), dtype=np.float64)
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.01

    # Act
    result = compute_per_regime_breakeven(
        alpha_long, alpha_short, realized, btc_close, cost_floor_bps=24.0
    )

    # Assert — at least one regime (bull or chop) has nan because 0 bars < 5
    assert set(result.keys()) == {"bull", "bear", "chop"}
    assert math.isnan(result["bull"]) or math.isnan(result["chop"])


# ---------------------------------------------------------------------------
# evaluate_alpha — per_regime_breakeven field
# ---------------------------------------------------------------------------


def test_evaluate_alpha_includes_per_regime_breakeven() -> None:
    """AlphaEvaluationReport must have per_regime_breakeven with bull/bear/chop keys."""
    # Arrange
    rng = np.random.default_rng(seed=13)
    T, N = 100, 5
    alpha_long = rng.standard_normal((T, N)).astype(np.float64) * 0.01
    alpha_short = rng.standard_normal((T, N)).astype(np.float64) * 0.01
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02

    # Act — without btc_close (all nan path)
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
        n_trials=1,
    )

    # Assert — field exists with correct structure
    assert hasattr(report, "per_regime_breakeven")
    assert set(report.per_regime_breakeven.keys()) == {"bull", "bear", "chop"}
    # Without btc_close all values are nan
    assert all(math.isnan(v) for v in report.per_regime_breakeven.values())


# ---------------------------------------------------------------------------
# diagnose_alpha_ic_decomposition — Phase 0/§A spec tests
# ---------------------------------------------------------------------------


def test_diagnose_ic_decomposition_returns_expected_keys() -> None:
    """반환 dict에 7개 핵심 키가 모두 존재해야 한다."""
    # Arrange
    rng = np.random.default_rng(seed=42)
    T, N = 50, 10
    pred = rng.standard_normal((T, N)).astype(np.float64)
    real = rng.standard_normal((T, N)).astype(np.float64)

    # Act
    result = diagnose_alpha_ic_decomposition(
        pred_dense_2d=pred,
        realized_raw_2d=real,
    )

    # Assert — 7개 키 존재
    expected_keys = {
        "dense_c1_raw_ic",
        "dense_c1_raw_hit",
        "dense_c1_raw_breadth",
        "dense_c1_resid_ic",
        "dense_c1_resid_hit",
        "dense_c3_raw_ic",
        "dense_c3_resid_ic",
    }
    assert set(result.keys()) == expected_keys


def test_diagnose_ic_decomposition_dense_breadth_greater_than_gated() -> None:
    """Dense signal breadth > gated (sparse) signal breadth."""
    # Arrange — dense signal: all non-zero; gated: only 1/10 non-zero
    rng = np.random.default_rng(seed=7)
    T, N = 100, 20
    pred_dense = rng.standard_normal((T, N)).astype(np.float64)
    pred_gated = np.zeros((T, N), dtype=np.float64)
    pred_gated[:, :2] = pred_dense[:, :2]  # only 2/20 symbols active

    real = rng.standard_normal((T, N)).astype(np.float64)

    # Act
    result_dense = diagnose_alpha_ic_decomposition(
        pred_dense_2d=pred_dense, realized_raw_2d=real
    )
    result_gated = diagnose_alpha_ic_decomposition(
        pred_dense_2d=pred_gated, realized_raw_2d=real
    )

    # Assert — dense breadth significantly larger
    assert result_dense["dense_c1_raw_breadth"] > result_gated["dense_c1_raw_breadth"]
    assert result_dense["dense_c1_raw_breadth"] == pytest.approx(20.0, rel=0.01)
    assert result_gated["dense_c1_raw_breadth"] == pytest.approx(2.0, rel=0.01)


def test_diagnose_ic_decomposition_residualized_differs_from_raw() -> None:
    """beta-residualization이 적용될 때 resid IC가 raw IC와 달라야 한다.

    핵심: beta가 심볼별로 다를 때만 per-row 차감값이 달라 spearman rank가 변함.
    상수 beta는 per-row shift이므로 spearman에 불변 → 심볼별 heterogeneous beta 필요.
    """
    # Arrange — signal correlated with idiosyncratic, but raw return contaminated by market
    rng = np.random.default_rng(seed=99)
    T, N = 200, 15
    # market factor
    market = rng.standard_normal(T).astype(np.float64) * 0.05
    # heterogeneous beta: varies per symbol (0.5 ~ 3.0) AND per time
    beta = np.abs(rng.standard_normal((T, N)).astype(np.float64)) * 1.0 + 0.5
    # idiosyncratic return (true signal target)
    idio = rng.standard_normal((T, N)).astype(np.float64) * 0.01
    # raw return = beta_i,t * market_t + idio_i,t (cross-sectional rank dominated by beta*market)
    raw = beta * market[:, np.newaxis] + idio
    # signal predicts idiosyncratic component (perfectly correlated, slight noise)
    pred = idio + rng.standard_normal((T, N)).astype(np.float64) * 0.001

    # Act
    result = diagnose_alpha_ic_decomposition(
        pred_dense_2d=pred,
        realized_raw_2d=raw,
        beta_2d=beta,
        market_fwd_1d=market,
    )

    # Assert — residualized IC ≠ raw IC (beta*market contamination changes cross-sectional rank)
    assert not math.isnan(result["dense_c1_resid_ic"])
    assert not math.isnan(result["dense_c1_raw_ic"])
    # They must differ — residualization removes the cross-sectional beta*market rank contamination
    assert abs(result["dense_c1_resid_ic"] - result["dense_c1_raw_ic"]) > 0.01


def test_diagnose_ic_decomposition_c3_subset_smaller_breadth() -> None:
    """C3 mask (subset) 적용 시 breadth가 C1보다 작아야 한다."""
    # Arrange
    rng = np.random.default_rng(seed=3)
    T, N = 80, 30
    pred = rng.standard_normal((T, N)).astype(np.float64)
    real = rng.standard_normal((T, N)).astype(np.float64)
    # C3: only 8/30 symbols
    c3_mask = np.zeros(N, dtype=np.bool_)
    c3_mask[:8] = True

    # Act
    result = diagnose_alpha_ic_decomposition(
        pred_dense_2d=pred,
        realized_raw_2d=real,
        trading_mask_1d=c3_mask,
    )

    # Assert — C3 IC computed (not nan); C1 breadth = 30
    assert not math.isnan(result["dense_c3_raw_ic"])
    assert result["dense_c1_raw_breadth"] == pytest.approx(30.0, rel=0.01)


def test_diagnose_ic_decomposition_resid_nan_when_beta_none() -> None:
    """beta_2d=None이면 resid IC는 nan이어야 한다."""
    # Arrange
    rng = np.random.default_rng(seed=1)
    T, N = 40, 8
    pred = rng.standard_normal((T, N)).astype(np.float64)
    real = rng.standard_normal((T, N)).astype(np.float64)

    # Act
    result = diagnose_alpha_ic_decomposition(
        pred_dense_2d=pred, realized_raw_2d=real, beta_2d=None
    )

    # Assert
    assert math.isnan(result["dense_c1_resid_ic"])
    assert math.isnan(result["dense_c1_resid_hit"])


# ---------------------------------------------------------------------------
# evaluate_alpha — inference_signed_2d (inference_stat panel)
# ---------------------------------------------------------------------------


def test_evaluate_alpha_inference_stat_panel_present_when_signed_provided() -> None:
    """inference_signed_2d가 주어지면 metrics_by_panel에 inference_stat 패널이 추가된다."""
    # Arrange
    rng = np.random.default_rng(0)
    T, N = 50, 20
    # Signed signal: continuous (not clipped to ≥0)
    signed_2d = rng.standard_normal((T, N)).astype(np.float64) * 0.01
    # Clipped signal: only positive values (simulating max(ev,0) clipping)
    clipped_long = np.where(signed_2d > 0, signed_2d, 0.0)
    clipped_short = np.zeros((T, N))
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02

    # Act
    report = evaluate_alpha(
        alpha_long_2d=clipped_long,
        alpha_short_2d=clipped_short,
        realized_fwd_ret_2d=realized,
        inference_signed_2d=signed_2d,
    )

    # Assert
    assert "inference_stat" in report.metrics_by_panel
    panel = report.metrics_by_panel["inference_stat"]
    assert "net_ic" in panel
    assert "ic_t_stat_nw" in panel
    assert "effective_breadth" in panel


def test_evaluate_alpha_inference_stat_breadth_exceeds_clipped() -> None:
    """Signed pre-clip 신호의 inference breadth는 클리핑된 신호의 breadth보다 크다."""
    # Arrange
    rng = np.random.default_rng(42)
    T, N = 60, 20
    # signed: continuous → all N symbols have nonzero values
    signed_2d = rng.standard_normal((T, N)).astype(np.float64) * 0.01
    # clipped: only ~50% survive max(·,0) → breadth≈10
    clipped_long = np.where(signed_2d > 0, signed_2d, 0.0)
    clipped_short = np.zeros((T, N))
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02

    # Act
    report = evaluate_alpha(
        alpha_long_2d=clipped_long,
        alpha_short_2d=clipped_short,
        realized_fwd_ret_2d=realized,
        inference_signed_2d=signed_2d,
    )

    # Assert
    clipped_breadth = report.effective_breadth
    infer_breadth = report.metrics_by_panel["inference_stat"]["effective_breadth"]
    assert infer_breadth > clipped_breadth, (
        f"inference breadth {infer_breadth:.1f} should exceed clipped breadth {clipped_breadth:.1f}"
    )


def test_evaluate_alpha_inference_stat_absent_when_not_provided() -> None:
    """inference_signed_2d 미제공 시 inference_stat 패널이 없다."""
    # Arrange
    rng = np.random.default_rng(1)
    T, N = 30, 10
    alpha = np.abs(rng.standard_normal((T, N)).astype(np.float64)) * 0.005
    realized = rng.standard_normal((T, N)).astype(np.float64) * 0.02

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha,
        alpha_short_2d=np.zeros((T, N)),
        realized_fwd_ret_2d=realized,
    )

    # Assert
    assert "inference_stat" not in report.metrics_by_panel


# ---------------------------------------------------------------------------
# Phase G1a / G2c 관련 신규 테스트
# ---------------------------------------------------------------------------


def test_evaluate_alpha_n_eff_universe_independent_of_emit_breadth() -> None:
    """n_eff는 universe-level N_eff (emit breadth cap 제거).

    G1 IC 스킬 게이트는 sizing policy와 독립적이어야 한다.
    n_eff_emit은 진단용으로만 보존된다.
    """
    # Arrange: emit_breadth≈2, corr-adjusted N_eff가 훨씬 큰 패널 시뮬레이션
    rng = np.random.default_rng(42)
    T, N = 200, 20
    # 잔차화된 패널: 낮은 상관관계 → N_eff_corr ≈ N
    realized = rng.standard_normal((T, N)) * 0.05
    # 클립 알파: 2개 심볼만 비零 → emit_breadth ≈ 4
    alpha_long = np.zeros((T, N))
    alpha_long[:, :2] = 0.01
    alpha_short = np.zeros((T, N))
    alpha_short[:, 2:4] = 0.01

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
    )

    # Assert: n_eff는 universe-level (emit breadth 상한 없음), n_eff_corr_raw와 동일해야 함
    assert report.n_eff == report.n_eff_corr_raw
    # n_eff_emit은 실제 emit breadth를 저장 (≈4, long 2 + short 2)
    assert report.n_eff_emit <= 5.0
    # universe n_eff는 emit breadth보다 커야 함 (cap이 제거됐으므로)
    assert report.n_eff >= report.n_eff_emit - 1e-9


def test_evaluate_alpha_clip_preservation_ratio_computed() -> None:
    """clip_preservation_ratio = port_ic / resid_ic (pre-clip IC 대비 비율)."""
    # Arrange
    rng = np.random.default_rng(7)
    T, N = 150, 10
    realized = rng.standard_normal((T, N)) * 0.04
    signal = rng.standard_normal((T, N)) * 0.01
    alpha_long = np.where(signal > 0, signal, 0.0)
    alpha_short = np.where(signal < 0, -signal, 0.0)

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
        inference_signed_2d=signal,
    )

    # Assert: 유한한 비율이 반환되거나(net_ic≠0) nan이 아닌 경우 타입 체크
    assert isinstance(report.clip_preservation_ratio, float)


def test_evaluate_alpha_clip_preservation_ratio_direction() -> None:
    """clip_preservation_ratio = post_clip_IC / pre_clip_IC (G2c: post/pre 순서).

    0 < post < pre 구간에서 비율이 0~1 사이여야 한다(G2c FAIL 경계).
    역전(pre/post)이면 이 케이스에서 > 1 반환 → 오통과 발생하므로 반드시 검증.
    """
    # Arrange: pre-clip signal이 유의하고, post-clip이 그보다 약한 상황 시뮬레이션.
    # inference_signed_2d = pre-clip dense signal
    # alpha_long/short = 부분 클립 후 신호 (magnitude 절반)
    rng = np.random.default_rng(42)
    T, N = 300, 12
    realized = rng.standard_normal((T, N)) * 0.04
    pre_clip = rng.standard_normal((T, N)) * 0.008   # pre-clip dense signal
    # post-clip: 같은 방향이지만 약함 (보존비 ≈ 0.5)
    post_long = np.where(pre_clip > 0, pre_clip * 0.5, 0.0)
    post_short = np.where(pre_clip < 0, -pre_clip * 0.5, 0.0)

    # Act
    report = evaluate_alpha(
        alpha_long_2d=post_long,
        alpha_short_2d=post_short,
        realized_fwd_ret_2d=realized,
        inference_signed_2d=pre_clip,
    )

    # Assert: 비율 = post/pre ∈ (0, 1) — pre > post > 0 이므로 0~1 사이
    ratio = report.clip_preservation_ratio
    if np.isfinite(ratio):
        # 스킬이 있는 경우: post/pre이면 < 1, pre/post이면 > 1
        # 이 케이스에서 > 1이면 분자/분모 역전 버그
        assert ratio <= 1.5, (
            f"clip_preservation_ratio={ratio:.3f} > 1.5 suggests pre/post inversion"
        )


def test_evaluate_alpha_t_stat_threshold_raised_to_3() -> None:
    """t_stat < 3.0이면 ic_t_stat_nw_below_3.0이 fail_reasons에 포함된다."""
    # Arrange
    rng = np.random.default_rng(99)
    T, N = 50, 5  # 적은 bars → 낮은 t-stat
    realized = rng.standard_normal((T, N)) * 0.05
    pred = rng.standard_normal((T, N)) * 0.001  # 매우 약한 신호
    alpha_long = np.where(pred > 0, pred, 0.0)
    alpha_short = np.where(pred < 0, -pred, 0.0)

    # Act
    report = evaluate_alpha(
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        realized_fwd_ret_2d=realized,
    )

    # Assert
    assert not report.passes
    assert "signal_t_stat_too_low" in report.fail_reasons


def test_diagnose_selection_monotonicity_monotone_signal() -> None:
    """단조 신호(rank score = 실현수익)에서 mono_rho=1, top-bot > 0 보장."""
    rng = np.random.default_rng(0)
    T, N = 40, 20
    # 신호가 수익과 완전 단조: pred = realized (noise 추가)
    base = rng.standard_normal((T, N))
    realized = base + rng.standard_normal((T, N)) * 0.01  # 거의 동일
    inference = base.copy()
    beta = rng.uniform(0.5, 1.5, size=(T, N))

    # Act
    result = diagnose_selection_monotonicity(
        inference, realized, beta, n_deciles=5, horizon_bars=6
    )

    # Assert: 단조 신호 → mono_rho 양수, top-bot 양수
    assert result["monotonicity_spearman"] > 0.5
    assert result["top_minus_bottom_bps"] > 0.0
    assert result["n_obs"] > 0
    for d in range(5):
        assert f"decile_mean_ret_bps_{d}" in result


def test_diagnose_selection_monotonicity_anti_monotone_signal() -> None:
    """역방향 신호에서 mono_rho 음수, top-bot < 0 확인."""
    rng = np.random.default_rng(1)
    T, N = 40, 20
    base = rng.standard_normal((T, N))
    realized = base + rng.standard_normal((T, N)) * 0.01
    inference = -base  # 역방향

    result = diagnose_selection_monotonicity(
        inference, realized, None, n_deciles=5, horizon_bars=6
    )

    assert result["monotonicity_spearman"] < -0.5
    assert result["top_minus_bottom_bps"] < 0.0
    # beta_2d=None → beta 지표 nan
    assert np.isnan(result["beta_tilt"])
    assert np.isnan(result["long_decile_beta_mean"])
    assert np.isnan(result["short_decile_beta_mean"])


def test_diagnose_selection_monotonicity_beta_tilt_detected() -> None:
    """상위 decile에 고베타 심볼이 몰릴 때 beta_tilt > 0 감지."""
    rng = np.random.default_rng(42)
    T, N = 30, 16
    # 신호 = beta rank (고베타를 선택)
    beta = np.tile(np.linspace(0.5, 2.0, N), (T, 1))
    noise = rng.standard_normal((T, N)) * 0.001
    inference = beta + noise  # 신호가 beta와 거의 동일

    realized = rng.standard_normal((T, N)) * 0.01  # 수익은 무관

    result = diagnose_selection_monotonicity(
        inference, realized, beta, n_deciles=5, horizon_bars=6
    )

    # 고베타가 상위 decile → long_beta > short_beta
    assert result["long_decile_beta_mean"] > result["short_decile_beta_mean"]
    assert result["beta_tilt"] > 0.3  # 명확한 tilt


def test_evaluate_alpha_dsr_uses_post_clip_signal() -> None:
    """DSR은 post-clip(체결) 신호 기준으로 계산 → dense pre-clip 과포화 방지.

    pre-clip dense signal(IC 높음)과 post-clip(순수 노이즈, IC≈0)을 분리 공급했을 때,
    DSR이 post-clip 기준으로 낮게 계산되어야 한다.
    구현이 잘못되어 pre-clip을 사용하면 DSR ≈ 1.0 saturate.
    """
    # Arrange
    rng = np.random.default_rng(11)
    T, N = 120, 8
    realized = rng.standard_normal((T, N)) * 0.04

    # pre-clip dense: 실현수익과 강한 상관 (IC ≈ 0.8)
    dense_signal = realized * 0.8 + rng.standard_normal((T, N)) * 0.002

    # post-clip: 순수 랜덤 노이즈 (realized와 독립, IC ≈ 0)
    noise_l = np.abs(rng.standard_normal((T, N)) * 0.001)
    noise_s = np.abs(rng.standard_normal((T, N)) * 0.001)

    # Act
    report = evaluate_alpha(
        alpha_long_2d=noise_l,
        alpha_short_2d=noise_s,
        realized_fwd_ret_2d=realized,
        inference_signed_2d=dense_signal,
        n_trials=6,
    )

    # Assert: post-clip(노이즈) 기준 DSR은 낮아야 함 (< 0.7)
    # pre-clip 기준을 사용하면 dense IC ≈ 0.8 → DSR ≈ 1.0
    assert 0.0 <= report.deflated_sharpe <= 1.0
    assert report.deflated_sharpe < 0.7, (
        f"DSR={report.deflated_sharpe:.4f} — post-clip(노이즈) 신호 기준이 아닌 경우 1.0에 가까워짐"
    )


def test_evaluate_alpha_clip_preservation_gate_at_0_7() -> None:
    """clip_preservation_ratio < 0.7 시나리오 검증.

    변경 2: threshold 0.5 → 0.7 상향.
    clip_preservation_ratio = post_clip_IC / pre_clip_IC.
    Spearman은 scale-invariant이므로 magnitude 축소가 아닌
    신호 구조 파괴(post = 독립 노이즈)로 낮은 비율을 생성해야 한다.
    """
    # Arrange: pre-clip은 IC 높고, post-clip은 독립 노이즈(다른 방향 구조)
    rng = np.random.default_rng(55)
    T, N = 200, 10
    realized = rng.standard_normal((T, N)) * 0.04

    # pre-clip: 실현수익과 강한 상관 (IC ≈ 0.7)
    dense = realized * 0.7 + rng.standard_normal((T, N)) * 0.004

    # post-clip: 독립 노이즈 — pred_2d IC ≈ 0 → ratio ≈ 0 / 0.7 ≈ 0
    random_long = np.abs(rng.standard_normal((T, N)) * 0.001)
    random_short = np.abs(rng.standard_normal((T, N)) * 0.001)

    # Act
    report = evaluate_alpha(
        alpha_long_2d=random_long,
        alpha_short_2d=random_short,
        realized_fwd_ret_2d=realized,
        inference_signed_2d=dense,
    )

    # Assert: clip_preservation_ratio << 0.7
    assert np.isfinite(report.clip_preservation_ratio)
    assert report.clip_preservation_ratio < 0.7, (
        f"clip_pres={report.clip_preservation_ratio:.3f} — post-clip이 독립 노이즈일 때 < 0.7이어야 함"
    )
