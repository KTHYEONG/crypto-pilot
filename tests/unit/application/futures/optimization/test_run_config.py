from __future__ import annotations

import pytest

from src.application.futures.optimization.config import build_run_config_from_args


def test_build_run_config_defaults_to_trials_100() -> None:
    cfg = build_run_config_from_args({"timeframe": "4h"})
    assert cfg.phase == "strategy"
    assert cfg.trials == 100


def test_build_run_config_accepts_strategy_phase() -> None:
    cfg = build_run_config_from_args({"phase": "strategy", "timeframe": "4h", "trials": 1})
    assert cfg.phase == "strategy"


def test_build_run_config_rejects_legacy_quick_backtest_phase() -> None:
    with pytest.raises(ValueError, match="legacy phase"):
        build_run_config_from_args(
            {
                "phase": "quick-backtest",
                "timeframe": "4h",
                "trials": 1,
            }
        )


def test_build_run_config_rejects_legacy_flags() -> None:
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "phase": "strategy",
                "alpha_only": True,
            }
        )


def test_build_run_config_rejects_legacy_mode_key() -> None:
    with pytest.raises(ValueError, match="legacy argument key"):
        build_run_config_from_args({"mode": "strategy", "trials": 1})


def test_build_run_config_rejects_legacy_tf_key() -> None:
    with pytest.raises(ValueError, match="legacy argument key"):
        build_run_config_from_args({"tf": "4h", "trials": 1})


def test_build_run_config_rejects_strategy_smoke_phase() -> None:
    """strategy-smoke는 legacy phase로 분류되어 거부되어야 한다."""
    with pytest.raises(ValueError, match="legacy phase"):
        build_run_config_from_args(
            {
                "phase": "strategy-smoke",
                "timeframe": "4h",
                "trials": 1,
            }
        )


def test_build_run_config_rejects_skip_universe_flag() -> None:
    """skip_universe는 legacy flag로 분류되어 거부되어야 한다."""
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "phase": "strategy",
                "timeframe": "4h",
                "trials": 1,
                "skip_universe": True,
            }
        )


def test_build_run_config_rejects_skip_data_sync_flag() -> None:
    """skip_data_sync는 legacy flag로 분류되어 거부되어야 한다."""
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "phase": "strategy",
                "timeframe": "4h",
                "trials": 1,
                "skip_data_sync": True,
            }
        )
