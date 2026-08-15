from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from src.application.futures.runner.cli import run_from_cli
from src.application.futures.runner.models import RunnerResult


class TestRunFromCli:
    def test_run_from_cli_accepts_only_allowed_flags(self, mocker: MockerFixture) -> None:
        mocker.patch("src.application.futures.runner.compound_main.run_multiscale_compound_main",
                      return_value=RunnerResult(0, "ok"))
        result = run_from_cli(["--date", "2026-07-08", "--sync", "local", "--seed", "42"])
        assert result == 0

    def test_run_from_cli_without_any_args_returns_ok(self, mocker: MockerFixture) -> None:
        mocker.patch("src.application.futures.runner.compound_main.run_multiscale_compound_main",
                      return_value=RunnerResult(0, "ok"))
        result = run_from_cli([])
        assert result == 0

    @pytest.mark.parametrize(
        "argv",
        [
            ["--trials", "10"],
            ["--timeframe", "4h"],
            ["--mode", "active"],
            ["--alpha-only"],
            ["--quick-backtest"],
            ["--sync-metrics"],
        ],
    )
    def test_removed_flags_return_error(self, argv: list[str], mocker: MockerFixture) -> None:
        result = run_from_cli(argv)
        assert result == 2

    def test_invalid_phase_returns_error(self) -> None:
        with pytest.raises(SystemExit):
            run_from_cli(["--phase", "l3"])

    def test_invalid_sync_value_returns_error(self, mocker: MockerFixture) -> None:
        with pytest.raises(SystemExit):
            run_from_cli(["--sync", "invalid"])

    def test_unknown_flag_returns_error(self, mocker: MockerFixture) -> None:
        result = run_from_cli(["--unknown-flag"])
        assert result == 2

    def test_seed_flag_preserved(self, mocker: MockerFixture) -> None:
        mock_main = mocker.patch("src.application.futures.runner.compound_main.run_multiscale_compound_main",
                                  return_value=RunnerResult(0, "ok"))
        run_from_cli(["--seed", "99", "--sync", "local"])
        config = mock_main.call_args[0][0]
        assert config.seed == 99
        assert config.sync.value == "local"


def test_main_function_runs(mocker: MockerFixture) -> None:
    import sys

    from src.application.futures.runner.cli import main

    mocker.patch("src.application.futures.runner.compound_main.run_multiscale_compound_main",
                  return_value=RunnerResult(0, "ok"))
    mocker.patch.object(sys, "argv", ["prog", "--sync", "local"])
    result = main()
    assert result == 0
