from __future__ import annotations

import pytest

from src.application.futures.optimization.config import build_run_config_from_args


def test_build_run_config_defaults_to_trials_100() -> None:
    cfg = build_run_config_from_args({"timeframe": "4h"})
    assert cfg.phase == "l3"
    assert cfg.trials == 100


def test_build_run_config_accepts_l3_phase() -> None:
    cfg = build_run_config_from_args({"phase": "l3", "timeframe": "4h", "trials": 1})
    assert cfg.phase == "l3"


def test_build_run_config_accepts_l2_phase() -> None:
    cfg = build_run_config_from_args({"phase": "l2", "timeframe": "4h", "trials": 1})
    assert cfg.phase == "l2"


def test_build_run_config_accepts_l1_phase() -> None:
    cfg = build_run_config_from_args({"phase": "l1", "timeframe": "4h", "trials": 1})
    assert cfg.phase == "l1"


def test_build_run_config_rejects_full_phase() -> None:
    """full alias는 제거됨; l3을 사용해야 한다."""
    with pytest.raises(ValueError, match="invalid active phase"):
        build_run_config_from_args({"phase": "full", "timeframe": "4h", "trials": 1})


def test_build_run_config_rejects_signal_phase() -> None:
    """signal alias는 제거됨; l1을 사용해야 한다."""
    with pytest.raises(ValueError, match="invalid active phase"):
        build_run_config_from_args({"phase": "signal", "timeframe": "4h", "trials": 1})


def test_build_run_config_rejects_alo_phase() -> None:
    """alo alias는 제거됨; l2를 사용해야 한다."""
    with pytest.raises(ValueError, match="invalid active phase"):
        build_run_config_from_args({"phase": "alo", "timeframe": "4h", "trials": 1})


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
                "phase": "l3",
                "alpha_only": True,
            }
        )


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
                "phase": "l3",
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
                "phase": "l3",
                "timeframe": "4h",
                "trials": 1,
                "skip_data_sync": True,
            }
        )
