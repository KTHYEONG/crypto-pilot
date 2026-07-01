"""Tests for active CLI contract and mocked fast-path verification.

These tests replace the removed --skip-universe, --skip-data-sync,
--bypass-champion-guard, --symbols, and legacy phase flags.
Fast execution paths are achieved through mock/patch fixtures.

Time complexity: O(1) per test — all IO is mocked.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from src.application.futures.optimization.config import (
    FuturesRunConfig,
    build_run_config_from_args,
)
from src.execution import opt_main_futures

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_run_config() -> FuturesRunConfig:
    """Minimal strategy run config without bypass fields.

    Replaces tests that previously required --skip-universe + --symbols.
    """
    return build_run_config_from_args(
        {
            "phase": "l3",
            "timeframe": "4h",
            "trials": 1,
            "sync": "auto",
        }
    )


@pytest.fixture
def mocked_pipeline_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Mock all IO-heavy pipeline stages for fast unit-level testing.

    Replaces the need for --skip-data-sync and --skip-universe flags.
    Returns a dict of call-tracking lists for assertion.
    """
    calls: dict[str, list[Any]] = {
        "sync": [],
        "universe": [],
        "data": [],
        "strategy": [],
    }
    from datetime import datetime

    window = opt_main_futures.QuarterlyWindow(
        fetch_start="2025-01-01",
        is_start="2025-04-01",
        oos_start="2026-01-01",
        end_date="2026-04-01",
        fetch_start_date=datetime.strptime("2025-01-01", "%Y-%m-%d").date(),
        is_start_date=datetime.strptime("2025-04-01", "%Y-%m-%d").date(),
        oos_start_date=datetime.strptime("2026-01-01", "%Y-%m-%d").date(),
        end_date_value=datetime.strptime("2026-04-01", "%Y-%m-%d").date(),
    )
    data_stage = opt_main_futures.DataStageResult(
        data_maps={"BTCUSDT": {}},
        oos_data_maps={"BTCUSDT": {}},
        valid_symbols=["BTCUSDT"],
    )

    from unittest.mock import Mock

    mock_snapshot = Mock()
    mock_meta = Mock()
    mock_meta.symbol = "BTCUSDT"
    mock_snapshot.selected = [mock_meta]

    monkeypatch.setattr(opt_main_futures, "_resolve_quarterly_window", lambda _: window)
    monkeypatch.setattr(
        opt_main_futures,
        "_ensure_universe_ledger_sync",
        lambda *a: calls["sync"].append(a),
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_ensure_cached_symbol_data_for_targets",
        lambda *a, **kw: None,
    )
    def _mock_universe(*a: Any, **kw: Any) -> tuple[Any, ...]:
        calls["universe"].append(a)
        calls["universe"].append(tuple(sorted(kw)))
        return (["BTCUSDT"], {}, (), (), mock_snapshot, {}, {})

    def _mock_data(*a: Any, **kw: Any) -> opt_main_futures.DataStageResult:
        calls["data"].append(a)
        calls["data"].append(tuple(sorted(kw)))
        return data_stage

    def _mock_strategy(*a: Any, **kw: Any) -> None:
        calls["strategy"].append(a)

    monkeypatch.setattr(
        opt_main_futures,
        "_run_universe_stage",
        _mock_universe,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_run_data_stage",
        _mock_data,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_run_regime_evaluation_stage",
        lambda *a: None,
    )
    monkeypatch.setattr(
        opt_main_futures,
        "_run_strategy_stage",
        _mock_strategy,
    )
    return calls


# ---------------------------------------------------------------------------
# Config contract tests — bypass fields rejected
# ---------------------------------------------------------------------------


def test_run_config_rejects_skip_universe() -> None:
    """skip_universe는 legacy flag로 분류되어 ValueError를 발생시켜야 한다."""
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "phase": "l3",
                "trials": 1,
                "skip_universe": True,
            }
        )


def test_run_config_rejects_skip_data_sync() -> None:
    """skip_data_sync는 legacy flag로 분류되어 ValueError를 발생시켜야 한다."""
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "phase": "l3",
                "trials": 1,
                "skip_data_sync": True,
            }
        )


def test_run_config_rejects_bypass_champion_guard() -> None:
    """bypass_champion_guard는 legacy flag로 분류되어 ValueError를 발생시켜야 한다."""
    with pytest.raises(ValueError, match="legacy flag"):
        build_run_config_from_args(
            {
                "phase": "l3",
                "trials": 1,
                "bypass_champion_guard": True,
            }
        )


def test_run_config_rejects_strategy_smoke() -> None:
    """strategy-smoke phase는 legacy phase로 분류되어 ValueError를 발생시켜야 한다."""
    with pytest.raises(ValueError, match="legacy phase"):
        build_run_config_from_args(
            {
                "phase": "strategy-smoke",
                "trials": 1,
            }
        )


def test_run_config_has_no_skip_universe_field(minimal_run_config: FuturesRunConfig) -> None:
    """FuturesRunConfig는 skip_universe 필드를 더 이상 보유하지 않아야 한다."""
    assert not hasattr(minimal_run_config, "skip_universe")


def test_run_config_has_no_skip_data_sync_field(minimal_run_config: FuturesRunConfig) -> None:
    """FuturesRunConfig는 skip_data_sync 필드를 더 이상 보유하지 않아야 한다."""
    assert not hasattr(minimal_run_config, "skip_data_sync")


def test_run_config_has_no_symbols_field(minimal_run_config: FuturesRunConfig) -> None:
    """FuturesRunConfig는 symbols 필드를 더 이상 보유하지 않아야 한다."""
    assert not hasattr(minimal_run_config, "symbols")


# ---------------------------------------------------------------------------
# CLI contract tests — bypass CLI flags rejected as unrecognized
# ---------------------------------------------------------------------------


def test_cli_rejects_symbols_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--symbols CLI 인자는 argparse 오류(exit 2)를 반환해야 한다."""
    monkeypatch.setattr(
        sys, "argv", ["opt_main_futures.py", "--phase", "l3", "--symbols", "BTCUSDT"]
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_cli_rejects_skip_universe_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip-universe CLI 인자는 argparse 오류(exit 2)를 반환해야 한다."""
    monkeypatch.setattr(
        sys, "argv", ["opt_main_futures.py", "--phase", "l3", "--skip-universe"]
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_cli_rejects_skip_data_sync_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip-data-sync CLI 인자는 argparse 오류(exit 2)를 반환해야 한다."""
    monkeypatch.setattr(
        sys, "argv", ["opt_main_futures.py", "--phase", "l3", "--skip-data-sync"]
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_cli_rejects_bypass_champion_guard_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--bypass-champion-guard CLI 인자는 argparse 오류(exit 2)를 반환해야 한다."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--phase", "l3", "--bypass-champion-guard"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_cli_rejects_alpha_only_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--alpha-only CLI 인자는 argparse 오류(exit 2)를 반환해야 한다."""
    monkeypatch.setattr(
        sys, "argv", ["opt_main_futures.py", "--phase", "l3", "--alpha-only"]
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_cli_rejects_quick_backtest_alias_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--quick-backtest alias 플래그는 argparse 오류(exit 2)를 반환해야 한다."""
    monkeypatch.setattr(
        sys, "argv", ["opt_main_futures.py", "--quick-backtest"]
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_cli_rejects_strategy_smoke_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """strategy-smoke phase는 argparse choices 검증으로 거부되어야 한다(exit 2)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--phase", "strategy-smoke"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Pipeline stage mock tests — replace skip_universe/skip_data_sync paths
# ---------------------------------------------------------------------------


def test_pipeline_always_calls_universe_stage(
    mocked_pipeline_stages: dict[str, list[Any]],
    minimal_run_config: FuturesRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Universe stage는 bypass 없이 항상 호출되어야 한다."""
    monkeypatch.setattr(
        opt_main_futures,
        "_run_optimization_stage",
        lambda *a, **kw: opt_main_futures.RunnerResult(exit_code=0, reason="ok"),
    )
    opt_main_futures.run_pipeline(minimal_run_config)
    assert mocked_pipeline_stages["universe"]


def test_pipeline_always_calls_sync_stage(
    mocked_pipeline_stages: dict[str, list[Any]],
    minimal_run_config: FuturesRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync stage는 bypass 없이 항상 호출되어야 한다."""
    monkeypatch.setattr(
        opt_main_futures,
        "_run_optimization_stage",
        lambda *a, **kw: opt_main_futures.RunnerResult(exit_code=0, reason="ok"),
    )
    opt_main_futures.run_pipeline(minimal_run_config)
    assert mocked_pipeline_stages["sync"]
