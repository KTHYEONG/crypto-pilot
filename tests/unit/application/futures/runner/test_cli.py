from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from src.application.futures.runner.cli import run_from_cli
from src.application.futures.runner.config import (
    FuturesRunConfig,
    build_alpha_foundry_runtime_config,
)
from src.application.futures.runner.models import MarketDataBundle, RunnerResult, RunWindow


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


class TestRunFromCli:
    def test_run_from_cli_with_active_args_calls_pipeline(self, mocker: MockerFixture) -> None:
        mock_pipeline = mocker.patch("src.application.futures.runner.cli.run_pipeline")
        mock_pipeline.return_value = RunnerResult(0, "ok")

        result = run_from_cli(["--phase", "l3", "--timeframe", "4h", "--trials", "3", "--sync", "skip"])

        assert result == 0
        mock_pipeline.assert_called_once()
        config = mock_pipeline.call_args[0][0]
        assert config.phase == "l3"
        assert config.timeframe == "4h"
        assert config.trials == 3
        assert config.sync == "skip"

    @pytest.mark.parametrize(
        "argv",
        [
            ["--alpha-only"],
            ["--skip-universe"],
            ["--skip-data-sync"],
            ["--symbols", "BTCUSDT"],
            ["--phase", "strategy-smoke"],
        ],
    )
    def test_run_from_cli_with_removed_options_returns_usage_error(
        self,
        mocker: MockerFixture,
        argv: list[str],
    ) -> None:
        mock_pipeline = mocker.patch("src.application.futures.runner.cli.run_pipeline")

        result = run_from_cli(argv)

        assert result == 2
        mock_pipeline.assert_not_called()

    def test_alpha_foundry_audit_mode_passed_to_config(self, mocker: MockerFixture) -> None:
        mock_pipeline = mocker.patch("src.application.futures.runner.cli.run_pipeline")
        mock_pipeline.return_value = RunnerResult(0, "ok")

        result = run_from_cli(
            [
                "--phase",
                "l3",
                "--timeframe",
                "4h",
                "--trials",
                "3",
                "--sync",
                "skip",
                "--alpha-foundry",
                "audit",
            ]
        )

        assert result == 0
        mock_pipeline.assert_called_once()
        config = mock_pipeline.call_args[0][0]
        assert config.alpha_foundry.mode == "audit"

    def test_alpha_foundry_gate_mode_passed_to_config(self, mocker: MockerFixture) -> None:
        mock_pipeline = mocker.patch("src.application.futures.runner.cli.run_pipeline")
        mock_pipeline.return_value = RunnerResult(0, "ok")

        result = run_from_cli(
            [
                "--phase",
                "l3",
                "--timeframe",
                "4h",
                "--trials",
                "3",
                "--sync",
                "skip",
                "--alpha-foundry",
                "gate",
            ]
        )

        assert result == 0
        mock_pipeline.assert_called_once()
        config = mock_pipeline.call_args[0][0]
        assert config.alpha_foundry.mode == "gate"

    def test_alpha_foundry_default_is_off(self, mocker: MockerFixture) -> None:
        mock_pipeline = mocker.patch("src.application.futures.runner.cli.run_pipeline")
        mock_pipeline.return_value = RunnerResult(0, "ok")

        result = run_from_cli(["--phase", "l3", "--timeframe", "4h", "--trials", "3", "--sync", "skip"])

        assert result == 0
        config = mock_pipeline.call_args[0][0]
        assert config.alpha_foundry.mode == "off"

    def test_legacy_args_preserved_with_alpha_foundry(self, mocker: MockerFixture) -> None:
        mock_pipeline = mocker.patch("src.application.futures.runner.cli.run_pipeline")
        mock_pipeline.return_value = RunnerResult(0, "ok")

        result = run_from_cli(
            [
                "--phase",
                "l1",
                "--timeframe",
                "1h",
                "--trials",
                "10",
                "--sync",
                "auto",
                "--alpha-foundry",
                "audit",
                "--seed",
                "99",
            ]
        )

        assert result == 0
        config = mock_pipeline.call_args[0][0]
        assert config.phase == "l1"
        assert config.timeframe == "1h"
        assert config.trials == 10
        assert config.sync == "auto"
        assert config.seed == 99
        assert config.alpha_foundry.mode == "audit"


class TestBuildAlphaFoundryRuntimeConfig:
    def test_invalid_mode_raises_value_error(self) -> None:
        from argparse import Namespace

        args = Namespace(
            alpha_foundry="invalid",
            timeframe="4h",
            phase="l3",
            trials=42,
            sync="auto",
            refresh_universe=False,
            sync_metrics=False,
            seed=42,
            date=None,
        )
        with pytest.raises(ValueError, match="invalid alpha_foundry mode"):
            build_alpha_foundry_runtime_config(args)

    def test_valid_mode_returns_alpha_foundry_runtime_config(self) -> None:
        from argparse import Namespace

        args = Namespace(
            alpha_foundry="audit",
            timeframe="4h",
            phase="l3",
            trials=42,
            sync="auto",
            refresh_universe=False,
            sync_metrics=False,
            seed=42,
            date=None,
        )
        config = build_alpha_foundry_runtime_config(args)
        assert config.mode == "audit"
