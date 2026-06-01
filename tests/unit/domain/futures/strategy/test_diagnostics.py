from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.diagnostics import (
    feature_cs_ic_audit,
    ic_summary,
    passes_ic_gate,
    rolling_ic,
    top_bottom_spread_bps,
)


def test_rolling_ic_basic() -> None:
    # 5 steps, 6 assets
    sig = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1, 1],  # constant (correlation 0/undefined)
            [1, 2, 3, 4, np.nan, np.nan],  # only 4 valid symbols (below 5)
            [1, 3, 2, 5, 4, 6],
        ],
        dtype=np.float64,
    )
    # Forward returns
    fwd_ret = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Perfectly positive
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Perfectly negative
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.15, 0.25, 0.2, 0.4, 0.35, 0.5],  # Strongly positive
        ],
        dtype=np.float64,
    )

    ic = rolling_ic(sig, fwd_ret, method="spearman")

    assert len(ic) == 5
    assert np.allclose(ic[0], 1.0)
    assert np.allclose(ic[1], -1.0)
    assert ic[2] == 0.0  # Constant row should yield 0 correlation
    assert np.isnan(ic[3])  # Below 5 valid symbols should yield NaN
    assert ic[4] > 0.8  # Strong positive


def test_ic_summary_stats() -> None:
    ic_series = np.array([0.05, 0.08, -0.02, np.nan, 0.12, 0.03, 0.04])
    summary = ic_summary(ic_series)

    # Valid values: [0.05, 0.08, -0.02, 0.12, 0.03, 0.04] -> count = 6
    assert summary["n_obs"] == 6.0
    assert summary["mean_ic"] == np.mean([0.05, 0.08, -0.02, 0.12, 0.03, 0.04])
    assert summary["hit_ratio"] == 5.0 / 6.0


def test_passes_ic_gate_logic() -> None:
    summary_pass = {
        "mean_ic": 0.03,
        "t_stat": 2.5,
        "hit_ratio": 0.55,
    }
    summary_fail_ic = {
        "mean_ic": 0.01,
        "t_stat": 2.5,
        "hit_ratio": 0.55,
    }
    summary_fail_t = {
        "mean_ic": 0.03,
        "t_stat": 1.8,
        "hit_ratio": 0.55,
    }
    summary_fail_hit = {
        "mean_ic": 0.03,
        "t_stat": 2.5,
        "hit_ratio": 0.48,
    }

    assert passes_ic_gate(summary_pass, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.5) is True
    assert (
        passes_ic_gate(summary_fail_ic, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.5)
        is False
    )
    assert (
        passes_ic_gate(summary_fail_t, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.5) is False
    )
    assert (
        passes_ic_gate(summary_fail_hit, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.5)
        is False
    )


def test_feature_cs_ic_audit_detects_injected_signal() -> None:
    rng = np.random.default_rng(7)
    t_len, n_len, f_len = 40, 20, 3
    target = rng.normal(0.0, 1.0, size=(t_len, n_len))
    features = rng.normal(0.0, 1.0, size=(t_len, n_len, f_len))
    features[:, :, 0] = target + rng.normal(0.0, 0.05, size=(t_len, n_len))
    names = ("sig_feature", "noise_a", "noise_b")

    rows = feature_cs_ic_audit(
        features,
        names,
        target,
        breakeven_ic=0.02,
        horizon_bars=12,
        top_k=3,
    )

    assert len(rows) == 3
    assert rows[0]["name"] == "sig_feature"
    assert float(rows[0]["mean_ic"]) > 0.8
    assert "t_stat_nw" in rows[0]
    assert "gap" in rows[0]


def test_top_bottom_spread_bps_flat_cost_semantics() -> None:
    # Arrange: perfect long-short signal, known gross spread
    T, N = 50, 10
    score = np.tile(np.arange(N, dtype=np.float64), (T, 1))  # [0..9] each bar
    realized = np.zeros((T, N), dtype=np.float64)
    realized[:, -1] = 0.01   # top symbol: +1% per bar
    realized[:, 0] = -0.01   # bottom symbol: -1% per bar
    eligible = np.ones((T, N), dtype=bool)

    # Act
    result = top_bottom_spread_bps(score, realized, eligible, quantile=0.10, cost_bps=24.0)

    # Assert: gross ≈ 200bps (1%-(-1%)=2%), net = gross - 24 (FLAT, not turnover-weighted)
    assert result["gross_spread_bps"] == pytest.approx(200.0, abs=1.0)
    assert result["net_spread_bps"] == pytest.approx(200.0 - 24.0, abs=1.0)
    # turnover_proxy is selection-fraction, not portfolio-weight turnover
    assert 0.0 < result["turnover_proxy"] <= 1.0


def test_top_bottom_spread_bps_insufficient_rows_returns_zeros() -> None:
    score = np.ones((2, 4), dtype=np.float64)
    realized = np.ones((2, 4), dtype=np.float64) * 0.001
    eligible = np.ones((2, 4), dtype=bool)

    result = top_bottom_spread_bps(score, realized, eligible, quantile=0.35, cost_bps=24.0)

    assert result["n_obs"] == 0.0
    assert result["net_spread_lcb_bps"] == pytest.approx(-24.0)
