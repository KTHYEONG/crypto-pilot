from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.diagnostics import ic_summary, passes_ic_gate, rolling_ic


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
    assert passes_ic_gate(summary_fail_ic, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.5) is False
    assert passes_ic_gate(summary_fail_t, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.5) is False
    assert passes_ic_gate(summary_fail_hit, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.5) is False
