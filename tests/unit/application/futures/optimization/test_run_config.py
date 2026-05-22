from __future__ import annotations

import pytest

from src.application.futures.optimization.config import build_run_config_from_args


def test_build_run_config_accepts_quick_backtest_mode() -> None:
    cfg = build_run_config_from_args(
        {
            "mode": "quick-backtest",
            "tf": "4h",
            "trials": 1,
        }
    )
    assert cfg.mode == "quick-backtest"
    assert cfg.strategy is None


def test_build_run_config_rejects_legacy_full_mode() -> None:
    with pytest.raises(ValueError, match="legacy mode"):
        build_run_config_from_args(
            {
                "mode": "full",
                "tf": "4h",
                "trials": 1,
            }
        )


def test_build_run_config_rejects_legacy_flags() -> None:
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "mode": "quick-backtest",
                "alpha_only": True,
            }
        )


def test_build_run_config_requires_strategy_in_strategy_mode() -> None:
    with pytest.raises(ValueError, match="requires strategy"):
        build_run_config_from_args(
            {
                "mode": "strategy",
                "tf": "4h",
                "trials": 1,
                "strategy": None,
            }
        )


def test_build_run_config_rejects_strategy_in_quick_backtest() -> None:
    with pytest.raises(ValueError, match="cannot set strategy"):
        build_run_config_from_args(
            {
                "mode": "quick-backtest",
                "strategy": "momentum_v0",
            }
        )
