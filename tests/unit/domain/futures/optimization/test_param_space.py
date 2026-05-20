from __future__ import annotations

import optuna
from optuna.distributions import FloatDistribution

from src.domain.futures.optimization.optimizer import MLPhaseDContext, _suggest_ml_joint_nsga2
from src.domain.futures.optimization.phase_param_space import (
    V43_CORE_PARAM_KEYS,
    V43_FIXED_DEFAULTS,
    V43_RISK_PARAM_KEYS,
    V43_SIGNAL_PARAM_KEYS,
    suggest_joint_params,
)


def _ask_trial() -> tuple[optuna.study.Study, optuna.trial.Trial]:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    return study, trial


def test_v43_joint_suggest_only_core_params_and_log_distributions() -> None:
    study, trial = _ask_trial()
    params = suggest_joint_params(trial)
    study.tell(trial, 0.0)

    assert set(params.keys()) == set(V43_CORE_PARAM_KEYS)
    assert set(trial.params.keys()) == set(V43_CORE_PARAM_KEYS)

    removed = {
        "BETA_REGIME_BULL",
        "BETA_REGIME_CRISIS",
        "SLIPPAGE_BPS_BUFFER_MULT",
        "CRISIS_OVERRIDE_THRESHOLD",
        "CRISIS_GAMMA",
        "BETA_ALPHA",
        "TIME_BARRIER_H",
    }
    assert removed.isdisjoint(set(trial.params.keys()))

    for key in (
        "EV_HURDLE_BPS",
        "PORTFOLIO_KAPPA",
        "TARGET_ANN_VOL",
        "MAX_EXPOSURE_PER_COIN",
    ):
        dist = trial.distributions[key]
        assert isinstance(dist, FloatDistribution)
        assert dist.log


def test_optimizer_main_path_uses_v43_space_and_fixed_defaults() -> None:
    study, trial = _ask_trial()
    ctx = MLPhaseDContext(data_maps={}, symbols=[], tf="4h")
    merged = _suggest_ml_joint_nsga2(trial, ctx)
    study.tell(trial, 0.0)

    assert set(trial.params.keys()) == set(V43_CORE_PARAM_KEYS)
    assert "BETA_REGIME_BULL" not in trial.params
    assert "SLIPPAGE_BPS_BUFFER_MULT" not in trial.params

    for key, value in V43_FIXED_DEFAULTS.items():
        assert merged.get(key) == value


def test_optimizer_phase_a1_suggests_signal_only() -> None:
    study, trial = _ask_trial()
    ctx = MLPhaseDContext(data_maps={}, symbols=[], tf="4h", coordinate_phase="phase_a1")
    _suggest_ml_joint_nsga2(trial, ctx)
    study.tell(trial, 0.0)
    assert set(trial.params.keys()) == set(V43_SIGNAL_PARAM_KEYS)
    assert set(V43_RISK_PARAM_KEYS).isdisjoint(set(trial.params.keys()))


def test_optimizer_phase_a2_suggests_risk_only_with_frozen_signal() -> None:
    study, trial = _ask_trial()
    frozen_signal = {
        "BETA_REGIME_BEAR": 0.3,
        "BETA_REGIME_CHOP": 0.2,
        "K_LONG": 4,
        "K_SHORT": 2,
        "REBALANCE_BARS": 8,
        "EV_HURDLE_BPS": 12.0,
    }
    ctx = MLPhaseDContext(
        data_maps={},
        symbols=[],
        tf="4h",
        coordinate_phase="phase_a2",
        coordinate_frozen_params=frozen_signal,
    )
    merged = _suggest_ml_joint_nsga2(trial, ctx)
    study.tell(trial, 0.0)
    assert set(trial.params.keys()) == set(V43_RISK_PARAM_KEYS)
    for key, value in frozen_signal.items():
        assert merged[key] == value


def test_optimizer_phase_b_applies_fixed_and_shrunk_ranges() -> None:
    study, trial = _ask_trial()
    ctx = MLPhaseDContext(
        data_maps={},
        symbols=[],
        tf="4h",
        coordinate_phase="phase_b",
        coordinate_frozen_params={"K_LONG": 5},
        phase_ranges={"TARGET_ANN_VOL": (0.12, 0.16)},
    )
    _suggest_ml_joint_nsga2(trial, ctx)
    study.tell(trial, 0.0)
    assert "K_LONG" not in trial.params
    dist = trial.distributions["TARGET_ANN_VOL"]
    assert isinstance(dist, FloatDistribution)
    assert abs(float(dist.low) - 0.12) < 1e-12
    assert abs(float(dist.high) - 0.16) < 1e-12
