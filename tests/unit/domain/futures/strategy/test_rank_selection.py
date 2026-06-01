from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.rank_selection import (
    RankSelectionPolicy,
    _estimate_policy_metrics,
    apply_rank_selection_policy,
    build_signed_rank_weights,
    calibrate_rank_portfolio_policy,
    policy_from_dict,
    policy_to_dict,
)


def test_soft_cs_emits_broader_weights_than_tail_on_same_scores() -> None:
    rng = np.random.default_rng(11)
    t, n = 40, 16
    score = rng.standard_normal((t, n)).astype(np.float64)
    eligible = np.ones((t, n), dtype=bool)
    realized = {12: (score + 0.01 * rng.standard_normal((t, n))) / 100.0}

    soft = calibrate_rank_portfolio_policy(
        signed_score_2d=score,
        realized_fwd_ret_by_horizon=realized,
        eligible_2d=eligible,
        execution_cost_bps_2d=None,
        beta_2d=None,
        quantiles=(0.2,),
        min_abs_z_grid=(0.0,),
        holding_bars_candidates=(12,),
        selection_modes=("soft_cs",),
        cost_bps_fallback=0.0,
        min_obs=20,
        target_breadth_min=4,
    )
    tail = calibrate_rank_portfolio_policy(
        signed_score_2d=score,
        realized_fwd_ret_by_horizon=realized,
        eligible_2d=eligible,
        execution_cost_bps_2d=None,
        beta_2d=None,
        quantiles=(0.2,),
        min_abs_z_grid=(0.0,),
        holding_bars_candidates=(12,),
        selection_modes=("tail",),
        cost_bps_fallback=0.0,
        min_obs=20,
        target_breadth_min=4,
    )

    al_soft, as_soft = apply_rank_selection_policy(
        signed_score_2d=score,
        eligible_2d=eligible,
        policy=soft,
    )
    al_tail, as_tail = apply_rank_selection_policy(
        signed_score_2d=score,
        eligible_2d=eligible,
        policy=tail,
    )
    breadth_soft = float(np.mean(np.sum((al_soft + as_soft) > 0.0, axis=1)))
    breadth_tail = float(np.mean(np.sum((al_tail + as_tail) > 0.0, axis=1)))
    assert breadth_soft >= breadth_tail


def test_build_signed_rank_weights_projects_caps_without_beta_matrix() -> None:
    score = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    eligible = np.ones_like(score, dtype=bool)
    policy = RankSelectionPolicy(
        polarity=1,
        quantile=0.25,
        min_abs_z=0.0,
        weighting="equal",
        weight_k=3.0,
        holding_bars=12,
        validation_net_lcb_bps=1.0,
        validation_gross_bps=1.0,
        validation_ir_t=1.0,
        validation_monotonicity=1.0,
        n_obs=10,
        selection_mode="soft_cs",
    )

    weights = build_signed_rank_weights(
        signed_score_2d=score,
        eligible_2d=eligible,
        policy=policy,
        beta_2d=None,
        gross_target=1.0,
        max_abs_net_exposure=0.0,
        max_abs_beta_exposure=0.0,
    )

    assert np.isfinite(weights).all()
    assert float(np.sum(weights)) == pytest.approx(0.0, abs=1e-9)


def test_calibrate_rank_portfolio_policy_counts_validation_rows_only() -> None:
    score = np.full((7, 4), np.nan, dtype=np.float64)
    score[:5] = np.array(
        [
            [-3.0, -1.0, 1.0, 3.0],
            [-2.0, -1.0, 1.0, 2.0],
            [-1.0, -0.5, 0.5, 1.0],
            [-2.5, -1.5, 1.5, 2.5],
            [-1.5, -0.75, 0.75, 1.5],
        ],
        dtype=np.float64,
    )
    realized = np.full((7, 4), np.nan, dtype=np.float64)
    realized[:5] = score[:5] / 100.0
    eligible = np.ones((7, 4), dtype=bool)
    weights = np.zeros_like(score, dtype=np.float64)
    metrics = _estimate_policy_metrics(
        weights=weights,
        realized=realized,
        eligible=eligible,
        execution_cost_bps_2d=None,
        cost_bps_fallback=0.0,
        beta_2d=None,
        score=score,
    )

    assert metrics["n_obs"] == 5.0
    assert metrics["validation_gross_bps"] == pytest.approx(0.0, abs=1e-12)


def test_policy_from_dict_keeps_backward_compat_for_missing_new_keys() -> None:
    payload = {
        "polarity": 1,
        "quantile": 0.35,
        "min_abs_z": 0.0,
        "weighting": "tanh",
        "weight_k": 3.0,
        "holding_bars": 12,
        "validation_net_lcb_bps": 0.5,
        "validation_gross_bps": 0.8,
        "validation_ir_t": 1.1,
        "validation_monotonicity": 0.2,
        "n_obs": 100,
    }
    pol = policy_from_dict(payload)
    assert pol.selection_mode == "tail"
    assert np.isnan(pol.validation_turnover)


def test_policy_to_dict_round_trip_includes_new_keys() -> None:
    policy = RankSelectionPolicy(
        polarity=1,
        quantile=0.2,
        min_abs_z=0.0,
        weighting="tanh",
        weight_k=3.0,
        holding_bars=12,
        validation_net_lcb_bps=1.0,
        validation_gross_bps=2.0,
        validation_ir_t=3.0,
        validation_monotonicity=0.3,
        n_obs=120,
        selection_mode="soft_cs",
        validation_turnover=0.4,
        validation_cost_bps=0.2,
        validation_breadth=10.0,
        validation_abs_net_exposure=0.01,
        validation_abs_beta_exposure=0.02,
    )
    out = policy_to_dict(policy)
    assert out["selection_mode"] == "soft_cs"
    assert "validation_turnover" in out


def test_ema_2d_functional() -> None:
    from src.domain.futures.strategy.rank_selection import _ema_2d
    arr = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
        [3.0, np.nan],
    ], dtype=np.float64)
    # span = 3 -> alpha = 0.5
    smoothed = _ema_2d(arr, span=3)
    
    assert np.isnan(smoothed[0, 1])
    assert smoothed[0, 0] == 1.0
    # t=1: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    assert smoothed[1, 0] == 1.5
    assert smoothed[1, 1] == 3.0
    # t=2: 0.5 * 3.0 + 0.5 * 1.5 = 2.25
    assert smoothed[2, 0] == 2.25
    assert smoothed[2, 1] == 3.0

