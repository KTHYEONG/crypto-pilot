from __future__ import annotations

from typing import cast

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState, create_trial

from src.domain.futures.optimization.common import (
    _trial_diag_sampled,
    _weight_stage_diag,
)
from src.domain.futures.optimization.objectives import _build_strategy_compose_diag
from src.domain.futures.portfolio.friction_model import resolve_cost_snapshot
from src.domain.futures.strategy.diagnostics import (
    passes_directional_viability_gate,
    passes_signal_preservation_gate,
)


def test_build_strategy_compose_diag_has_expected_core_keys() -> None:
    alpha_l = np.full((8, 2), 0.01, dtype=np.float64)
    alpha_s = np.full((8, 2), 0.01, dtype=np.float64)
    xs_l = np.full((8, 2), 0.02, dtype=np.float64)
    xs_s = np.full((8, 2), 0.02, dtype=np.float64)
    cost_snap = resolve_cost_snapshot(execution_cost_bps_2d=None, shape=(8, 2))
    diag = _build_strategy_compose_diag(
        alpha_long=alpha_l,
        alpha_short=alpha_s,
        xs_long=xs_l,
        xs_short=xs_s,
        params={"BETA_ALPHA": 3.0, "EV_HURDLE_BPS": 2.0},
        cost_snapshot=cost_snap,
    )
    assert diag["alpha_long_nz_ratio"] > 0.0
    assert diag["xs_long_nz_ratio"] > 0.0
    assert diag["effective_threshold_bps"] > diag["ev_hurdle_bps"]
    assert "mu_pre_hurdle_p95_long" in diag
    assert "mu_pre_hurdle_p95_short" in diag
    assert "xs_long_preservation_ratio" in diag
    assert "xs_short_preservation_ratio" in diag
    assert diag["xs_long_preservation_ratio"] > 0.0
    assert diag["xs_short_preservation_ratio"] > 0.0


def test_passes_directional_viability_gate_accepts_quality_report_keys() -> None:
    report = {
        "alpha_long_non_zero_ratio": 0.08,
        "alpha_short_non_zero_ratio": 0.15,
    }
    assert (
        passes_directional_viability_gate(
            report,
            min_long_non_zero_ratio=0.05,
            min_short_non_zero_ratio=0.10,
        )
        is True
    )


def test_passes_signal_preservation_gate_uses_preservation_ratios() -> None:
    report = {"xs_long_preservation_ratio": 0.02, "xs_short_preservation_ratio": 0.25}
    assert (
        passes_signal_preservation_gate(
            report,
            min_long_preservation_ratio=0.01,
            min_short_preservation_ratio=0.20,
        )
        is True
    )


def test_weight_stage_diag_exposes_cap_hit_proxy() -> None:
    tw = np.array(
        [
            [0.2, 0.0],
            [0.1, 0.2],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    diag = _weight_stage_diag(tw, per_symbol_cap=0.2)
    assert float(diag["tw_row_nz_ratio"]) > 0.0
    assert float(diag["gross_mean"]) > 0.0
    assert isinstance(diag["cap_hit_proxy_ratio"], float)
    assert float(diag["cap_hit_proxy_ratio"]) >= 0.0


def test_trial_diag_sampled_policy() -> None:
    tr = create_trial(
        params={"x": 0.1},
        distributions={"x": FloatDistribution(0.0, 1.0)},
        value=0.0,
        state=TrialState.COMPLETE,
    )
    assert _trial_diag_sampled(None, n_trades=0) is True
    assert _trial_diag_sampled(None, n_trades=1) is False
    assert _trial_diag_sampled(cast(optuna.Trial, tr), n_trades=1) is True
