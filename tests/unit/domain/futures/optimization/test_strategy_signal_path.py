from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.optimization.optimizer import (
    _compose_strategy_scores_inplace,
    _run_portfolio_numba_block,
)


def _base_aligned(alpha_long: np.ndarray, alpha_short: np.ndarray) -> dict[str, np.ndarray]:
    n_bars, n_syms = alpha_long.shape
    z = np.zeros((n_bars, n_syms), dtype=np.float64)
    return {
        "alpha_long": alpha_long,
        "alpha_short": alpha_short,
        "hmm_prob_bull_calm": z.copy(),
        "hmm_prob_bull_vol_up": z.copy(),
        "hmm_prob_bear_trend": z.copy(),
        "hmm_prob_chop": z.copy(),
        "hmm_prob_crisis": z.copy(),
    }


def test_strategy_xs_generation_does_not_require_alpha_long_00() -> None:
    aligned = _base_aligned(
        alpha_long=np.full((16, 2), 0.02, dtype=np.float64),
        alpha_short=np.full((16, 2), 0.02, dtype=np.float64),
    )
    params = {"BETA_ALPHA": 3.0, "EV_HURDLE_BPS": 0.0}
    _compose_strategy_scores_inplace(aligned, params)
    assert "xs_score_long" in aligned
    assert "xs_score_short" in aligned


def test_strategy_xs_generated_non_zero_from_non_zero_alpha() -> None:
    aligned = _base_aligned(
        alpha_long=np.full((20, 3), 0.02, dtype=np.float64),
        alpha_short=np.full((20, 3), 0.02, dtype=np.float64),
    )
    params = {"BETA_ALPHA": 3.0, "EV_HURDLE_BPS": 0.0}
    _compose_strategy_scores_inplace(aligned, params)
    xs_l = np.asarray(aligned["xs_score_long"], dtype=np.float64)
    xs_s = np.asarray(aligned["xs_score_short"], dtype=np.float64)
    assert np.count_nonzero(np.abs(xs_l) > 1e-12) > 0
    assert np.count_nonzero(np.abs(xs_s) > 1e-12) > 0


def test_strategy_path_diag_xs_nz_counts_short_side_union() -> None:
    aligned = _base_aligned(
        alpha_long=np.zeros((12, 2), dtype=np.float64),
        alpha_short=np.full((12, 2), 0.02, dtype=np.float64),
    )
    params = {"BETA_ALPHA": 3.0, "EV_HURDLE_BPS": 0.0}
    _compose_strategy_scores_inplace(aligned, params)
    path_diag = aligned.get("_strategy_signal_path_diag")
    assert isinstance(path_diag, dict)
    assert float(path_diag.get("alpha_nz", 0.0)) > 0.0
    assert float(path_diag.get("xs_nz", 0.0)) > 0.0


def test_strategy_mode_fail_fast_when_alpha_prerequisite_missing() -> None:
    aligned = {
        "alpha_long": np.full((8, 1), 0.01, dtype=np.float64),
    }
    with pytest.raises(RuntimeError, match="requires aligned alpha_long/alpha_short"):
        _run_portfolio_numba_block({"STRATEGY_MODE": True}, aligned)


def test_strategy_compose_uses_static_execution_cost_when_present() -> None:
    alpha = np.full((6, 2), 0.03, dtype=np.float64)
    aligned = _base_aligned(alpha_long=alpha, alpha_short=alpha)
    aligned["execution_cost_bps_2d"] = np.array(
        [[10.0, 40.0]] * 6,
        dtype=np.float64,
    )
    _compose_strategy_scores_inplace(aligned, {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 0.0})
    xs_l = np.asarray(aligned["xs_score_long"], dtype=np.float64)
    assert np.all(xs_l[:, 0] > xs_l[:, 1])
    meta = aligned.get("_strategy_cost_snapshot_meta")
    assert isinstance(meta, dict)
    assert meta.get("execution_cost_bps_source") == "universe_static"


def test_strategy_compose_fallback_cost_source_metadata() -> None:
    alpha = np.full((6, 2), 0.03, dtype=np.float64)
    aligned = _base_aligned(alpha_long=alpha, alpha_short=alpha)
    _compose_strategy_scores_inplace(aligned, {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 0.0})
    meta = aligned.get("_strategy_cost_snapshot_meta")
    assert isinstance(meta, dict)
    assert meta.get("execution_cost_bps_source") == "fallback_global"


def test_strategy_compose_trial_toggle_uses_dynamic_cost_even_when_default_static() -> None:
    alpha = np.full((6, 2), 0.03, dtype=np.float64)
    aligned = _base_aligned(alpha_long=alpha, alpha_short=alpha)
    aligned["execution_cost_bps_2d_static"] = np.full((6, 2), 10.0, dtype=np.float64)
    aligned["execution_cost_fraction_2d_static"] = np.full(
        (6, 2), 10.0 / 10000.0, dtype=np.float64
    )
    aligned["execution_cost_bps_2d_dynamic"] = np.full((6, 2), 40.0, dtype=np.float64)
    aligned["execution_cost_fraction_2d_dynamic"] = np.full(
        (6, 2), 40.0 / 10000.0, dtype=np.float64
    )
    aligned["_cost_forecast_source_static"] = "universe_static"
    aligned["_cost_forecast_source_dynamic"] = "parametric_dynamic"
    # precompute default(static) 상태를 가정
    aligned["execution_cost_bps_2d"] = aligned["execution_cost_bps_2d_static"]

    _compose_strategy_scores_inplace(
        aligned,
        {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 0.0, "COST_FORECAST_DYNAMIC": True},
    )
    meta = aligned.get("_strategy_cost_snapshot_meta")
    assert isinstance(meta, dict)
    assert meta.get("execution_cost_bps_source") == "parametric_dynamic"
    assert float(aligned.get("_cost_forecast_dynamic", 0.0)) == 1.0
    xs_l = np.asarray(aligned["xs_score_long"], dtype=np.float64)
    # dynamic cost(40bps) 사용 시 static(10bps) 대비 순 alpha 감소
    assert np.all(xs_l < (0.03 - 10.0 / 10000.0))


def test_strategy_compose_cost_gate_amortize_uses_rebalance_bars() -> None:
    alpha = np.full((8, 2), 0.01, dtype=np.float64)
    aligned_base = _base_aligned(alpha_long=alpha, alpha_short=alpha)
    aligned_base["execution_cost_bps_2d"] = np.full((8, 2), 20.0, dtype=np.float64)

    aligned_no = {
        k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in aligned_base.items()
    }
    aligned_amort = {
        k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in aligned_base.items()
    }

    _compose_strategy_scores_inplace(
        aligned_no,
        {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 0.0, "REBALANCE_BARS": 5, "COST_GATE_AMORTIZE": False},
    )
    _compose_strategy_scores_inplace(
        aligned_amort,
        {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 0.0, "REBALANCE_BARS": 5, "COST_GATE_AMORTIZE": True},
    )

    mu_no = np.asarray(aligned_no["mu_long_2d"], dtype=np.float64)
    mu_amort = np.asarray(aligned_amort["mu_long_2d"], dtype=np.float64)
    assert np.all(mu_amort > mu_no)
    np.testing.assert_allclose(mu_amort - mu_no, (20.0 / 10000.0) * (1.0 - 1.0 / 5.0), rtol=1e-9)
