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


def test_strategy_compose_uses_per_symbol_execution_cost_when_present() -> None:
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
    assert meta.get("execution_cost_bps_source") == "per_symbol"


def test_strategy_compose_fallback_cost_source_metadata() -> None:
    alpha = np.full((6, 2), 0.03, dtype=np.float64)
    aligned = _base_aligned(alpha_long=alpha, alpha_short=alpha)
    _compose_strategy_scores_inplace(aligned, {"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 0.0})
    meta = aligned.get("_strategy_cost_snapshot_meta")
    assert isinstance(meta, dict)
    assert meta.get("execution_cost_bps_source") == "fallback_global"
