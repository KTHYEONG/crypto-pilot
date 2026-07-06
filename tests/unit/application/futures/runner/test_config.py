from __future__ import annotations

from typing import Any

import pytest

from src.application.futures.runner.config import (
    FuturesRunConfig,
    build_alpha_foundry_runtime_config,
    build_run_config_from_args,
    parse_active_phase,
    validate_run_config,
)
from src.application.futures.runner.models import MarketDataBundle, RunWindow


def make_run_config(phase: str = "l3") -> FuturesRunConfig:
    return FuturesRunConfig(
        timeframe="4h",
        date="2026-05-01",
        trials=3,
        phase=phase,
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,  # type: ignore[arg-type]
    )


def make_window() -> RunWindow:
    from datetime import date

    return RunWindow(
        fetch_start="2024-01-01",
        is_start="2024-04-01",
        oos_start="2025-01-01",
        end_date="2026-05-01",
        fetch_start_date=date(2024, 1, 1),
        is_start_date=date(2024, 4, 1),
        oos_start_date=date(2025, 1, 1),
        end_date_value=date(2026, 5, 1),
    )


def make_data_bundle() -> MarketDataBundle:
    return MarketDataBundle(
        data_maps={"BTCUSDT": {"4h": object()}},
        oos_data_maps={"BTCUSDT": {"4h": object()}},
        valid_symbols=("BTCUSDT",),
    )


class TestBuildRunConfig:
    @pytest.mark.parametrize(
        ("args_dict", "expected_match"),
        [
            ({"phase": "l3", "timeframe": "4h", "trials": 0, "sync": "skip"}, "trials must be >= 1"),
            ({"phase": "quick-backtest", "timeframe": "4h", "trials": 1, "sync": "skip"}, "removed phase"),
        ],
    )
    def test_build_run_config_rejects_invalid_or_removed_inputs(
        self,
        args_dict: dict[str, Any],
        expected_match: str,
    ) -> None:
        with pytest.raises(ValueError, match=expected_match):
            build_run_config_from_args(args_dict)

    def test_default_sync_mode_is_auto(self) -> None:
        config = build_run_config_from_args({"phase": "l3", "timeframe": "4h", "trials": 1})
        assert config.sync == "auto"

    @pytest.mark.parametrize("bad_sync", ["fast", "full", "elite_fast", "force", ""])
    def test_sync_legacy_or_invalid_values_rejected(self, bad_sync: str) -> None:
        with pytest.raises(ValueError, match="invalid sync mode"):
            build_run_config_from_args({"phase": "l3", "trials": 1, "sync": bad_sync})

    def test_sync_skip_is_accepted(self) -> None:
        config = build_run_config_from_args({"phase": "l3", "trials": 1, "sync": "skip"})
        assert config.sync == "skip"

    def test_parse_active_phase_direct(self) -> None:
        result = parse_active_phase("l1")
        assert result == "l1"

    def test_parse_active_phase_removed_raises(self) -> None:
        with pytest.raises(ValueError, match="removed phase"):
            parse_active_phase("strategy-smoke")

    def test_parse_active_phase_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown phase"):
            parse_active_phase("l4")

    def test_validate_run_config_direct(self) -> None:
        config = make_run_config()
        result = validate_run_config(config)
        assert result.trials == 3

    def test_validate_run_config_trials_zero_raises(self) -> None:
        config = FuturesRunConfig(
            timeframe="4h",
            date=None,
            trials=0,
            phase="l3",
            sync="skip",
            refresh_universe=False,
            sync_metrics=False,
            seed=42,
        )
        with pytest.raises(ValueError, match="trials must be >= 1"):
            validate_run_config(config)

    def test_build_alpha_foundry_runtime_config_direct(self) -> None:
        config = build_alpha_foundry_runtime_config({"alpha_foundry": "audit"})
        assert config.mode == "audit"

    def test_build_alpha_foundry_runtime_config_default(self) -> None:
        config = build_alpha_foundry_runtime_config({})
        assert config.mode == "off"

    def test_unknown_phase_via_build(self) -> None:
        with pytest.raises(ValueError, match="unknown phase"):
            build_run_config_from_args({"phase": "l4", "trials": 1, "sync": "skip"})

    def test_alpha_foundry_audit_mode_accepted(self) -> None:
        config = build_run_config_from_args(
            {
                "phase": "l3",
                "timeframe": "4h",
                "trials": 1,
                "sync": "skip",
                "alpha_foundry": "audit",
            }
        )
        assert config.alpha_foundry.mode == "audit"

    def test_alpha_foundry_gate_mode_accepted(self) -> None:
        config = build_run_config_from_args(
            {
                "phase": "l3",
                "timeframe": "4h",
                "trials": 1,
                "sync": "skip",
                "alpha_foundry": "gate",
            }
        )
        assert config.alpha_foundry.mode == "gate"

    def test_removed_arg_rejected(self) -> None:
        with pytest.raises(ValueError, match="removed argument"):
            build_run_config_from_args(
                {
                    "phase": "l3",
                    "timeframe": "4h",
                    "trials": 1,
                    "sync": "skip",
                    "alpha_only": True,
                }
            )
