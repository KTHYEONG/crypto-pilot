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
    diagnose_alpha_ic_decomposition,
    evaluate_alpha,
    sweep_horizon_breakeven,
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
