from __future__ import annotations

import pytest

from src.application.futures.optimization.config import (
    build_run_config_from_args as build_optimization,
)
from src.application.futures.run_contracts import (
    ExecutionPolicy,
    FuturesRunConfig,
    RunPolicyError,
)
from src.application.futures.run_policy import build_effective_run_config
from src.application.futures.runner.config import (
    build_run_config_from_args as build_runner,
)
from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig


def test_execution_policy_defaults() -> None:
    ep = ExecutionPolicy()
    assert ep.heavy_process_workers == 1
    assert ep.ltf_io_workers == 1
    assert ep.max_rss_mb == 12_000


def test_run_policy_validation_l0_requires_gate() -> None:
    with pytest.raises(RunPolicyError, match=r"requires l0_runtime.mode='gate'"):
        FuturesRunConfig(
            timeframe="4h", date=None, trials=1, phase="l0", sync="skip",
            refresh_universe=False, sync_metrics=False,
            l0_runtime=AlphaFoundryRuntimeConfig(mode="off"),
        )


def test_run_policy_l2_accepts_off() -> None:
    cfg = FuturesRunConfig(
        timeframe="4h", date=None, trials=1, phase="l2", sync="skip",
        refresh_universe=False, sync_metrics=False,
        l0_runtime=AlphaFoundryRuntimeConfig(mode="off"),
    )
    assert cfg.l0_runtime.mode == "off"


def test_config_facades_share_effective_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("L0_LTF_EXEC_1M_MAX_WORKERS", "2")
    args = {"phase": "l1", "timeframe": "4h", "trials": 1, "sync": "skip",
            "refresh_universe": False, "sync_metrics": False}

    runner_config = build_runner(args)
    optimization_config = build_optimization(args)

    assert runner_config == optimization_config
    assert runner_config.l0_runtime.mode == "gate"
    assert runner_config.execution_policy.heavy_process_workers == 1
    assert runner_config.policy_fingerprint


def test_run_policy_rejects_invalid_env_override() -> None:
    with pytest.raises(RunPolicyError, match="invalid value for L0_LTF_EXEC_1M_MAX_WORKERS"):
        build_effective_run_config(
            {"phase": "l1", "timeframe": "4h", "trials": 1, "sync": "skip",
             "refresh_universe": False, "sync_metrics": False},
            environ={"L0_LTF_EXEC_1M_MAX_WORKERS": "abc"},
        )


def test_execution_policy_rejects_invalid_ltf_workers() -> None:
    with pytest.raises(RunPolicyError, match="L0_LTF_EXEC_1M_MAX_WORKERS must be 1 or 2"):
        build_effective_run_config(
            {"phase": "l1", "timeframe": "4h", "trials": 1, "sync": "skip",
             "refresh_universe": False, "sync_metrics": False},
            environ={"L0_LTF_EXEC_1M_MAX_WORKERS": "5"},
        )


def test_run_policy_when_cli_omits_removed_alpha_foundry_flag_keeps_l1_gate() -> None:
    config = build_effective_run_config(
        {
            "phase": "l1",
            "timeframe": "4h",
            "trials": 1,
            "sync": "skip",
            "alpha_foundry": None,
        },
        environ={},
    )

    assert config.l0_runtime.mode == "gate"
