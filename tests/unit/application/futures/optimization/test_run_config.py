from __future__ import annotations

import pytest

from src.application.futures.optimization.config import build_run_config_from_args


def test_build_run_config_defaults_to_trials_100() -> None:
    cfg = build_run_config_from_args({"tf": "4h"})
    assert cfg.mode == "strategy"
    assert cfg.trials == 100


def test_build_run_config_accepts_quick_backtest_mode() -> None:
    cfg = build_run_config_from_args(
        {
            "mode": "quick-backtest",
            "tf": "4h",
            "trials": 1,
        }
    )
    assert cfg.mode == "quick-backtest"


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


def test_build_run_config_rejects_strategy_smoke_mode() -> None:
    """strategy-smoke는 legacy mode로 분류되어 거부되어야 한다."""
    with pytest.raises(ValueError, match="legacy mode"):
        build_run_config_from_args(
            {
                "mode": "strategy-smoke",
                "tf": "4h",
                "trials": 1,
            }
        )


def test_build_run_config_rejects_skip_universe_flag() -> None:
    """skip_universe는 legacy flag로 분류되어 거부되어야 한다."""
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "mode": "strategy",
                "tf": "4h",
                "trials": 1,
                "skip_universe": True,
            }
        )


def test_build_run_config_rejects_skip_data_sync_flag() -> None:
    """skip_data_sync는 legacy flag로 분류되어 거부되어야 한다."""
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "mode": "strategy",
                "tf": "4h",
                "trials": 1,
                "skip_data_sync": True,
            }
        )

