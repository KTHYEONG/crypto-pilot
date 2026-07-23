from __future__ import annotations

import numpy as np
import pytest

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.compound_main import run_multiscale_compound_main
from src.domain.futures.compound.contracts import (
    AlphaForecastTape,
    DeploymentVerdict,
    ExecutionLedger,
    L2Evaluation,
    L3ValidationResult,
    MarketFeatureCube,
)
from src.domain.futures.compound.config import CompoundEngineConfig


def _make_mock_engine_result(mocker) -> object:
    mock_tape = mocker.Mock(spec=AlphaForecastTape)
    mock_tape.model_version = "multiscale-v1"
    mock_tape.data_manifest_hash = "h1"
    mock_tape.symbols = ("BTCUSDT", "ETHUSDT")

    mock_ledger = mocker.Mock(spec=ExecutionLedger)
    mock_ledger.integrity_ok = True
    mock_ledger.timestamps_ns = np.array([0, 1, 2], dtype=np.int64)
    mock_ledger.target_weights_2d = np.zeros((3, 2), dtype=np.float32)
    mock_ledger.net_returns_1d = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    mock_ledger.fee_returns_1d = np.zeros(3, dtype=np.float64)
    mock_ledger.slippage_returns_1d = np.zeros(3, dtype=np.float64)
    mock_ledger.impact_returns_1d = np.zeros(3, dtype=np.float64)
    mock_ledger.funding_returns_1d = np.zeros(3, dtype=np.float64)
    mock_ledger.equity_1d = np.array([1.0, 1.01, 1.02], dtype=np.float64)

    mock_l2 = mocker.Mock(spec=L2Evaluation)
    mock_l2.annualized_log_growth = 0.05
    mock_l2.growth_ci90 = (0.01, 0.09)
    mock_l2.equity_multiple = 1.1
    mock_l2.max_drawdown = 0.02
    mock_l2.daily_cvar95 = -0.01
    mock_l2.annual_volatility = 0.15
    mock_l2.turnover = 0.5
    mock_l2.safe = True
    mock_l2.integrity_ok = True

    mock_l3 = mocker.Mock(spec=L3ValidationResult)
    mock_l3.verdict = DeploymentVerdict.PROMOTE
    mock_l3.posterior_growth_probability = 0.75
    mock_l3.holdout_days = 180
    mock_l3.max_drawdown = 0.02
    mock_l3.daily_cvar95 = -0.01
    mock_l3.reasons = ()

    result = mocker.Mock()
    result.alpha_tape = mock_tape
    result.ledger = mock_ledger
    result.l2 = mock_l2
    result.l3 = mock_l3
    result.symbols = ("BTCUSDT", "ETHUSDT")
    return result


class TestRunMultiscaleCompoundMain:
    def test_main_is_callable(self) -> None:
        assert callable(run_multiscale_compound_main)

    def test_happy_path_returns_zero(self, mocker) -> None:
        mocker.patch(
            "src.application.futures.runner.compound_main.build_data_lake_runtime",
        )
        mocker.patch(
            "src.application.futures.runner.compound_main.prepare_data_snapshot",
            return_value=mocker.Mock(snapshot_id="s1", manifest_hash="h1",
                                      reference_time_ms=1_000_000, partitions=(), total_bytes=0),
        )
        mocker.patch(
            "src.application.futures.runner.compound_main.build_daily_pit_universe",
            return_value=mocker.Mock(symbols=("BTCUSDT", "ETHUSDT"), decision_dates=()),
        )
        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1, 2], dtype=np.int64)
        mock_cube.data_manifest_hash = "h1"
        mocker.patch(
            "src.application.futures.runner.compound_main.build_multiscale_market_cube",
            return_value=mock_cube,
        )
        mock_result = _make_mock_engine_result(mocker)
        mocker.patch(
            "src.application.futures.runner.compound_main.run_multiscale_compound_engine",
            return_value=mock_result,
        )

        config = CompoundRunConfig(
            reference_date="2026-07-08", sync="skip", refresh_universe=False,
        )
        result = run_multiscale_compound_main(config)
        assert result.exit_code == 0

    def test_integrity_failure_returns_one(self, mocker) -> None:
        mocker.patch(
            "src.application.futures.runner.compound_main.build_data_lake_runtime",
        )
        mocker.patch(
            "src.application.futures.runner.compound_main.prepare_data_snapshot",
            return_value=mocker.Mock(snapshot_id="s1", manifest_hash="h1",
                                      reference_time_ms=1_000_000, partitions=(), total_bytes=0),
        )
        mocker.patch(
            "src.application.futures.runner.compound_main.build_daily_pit_universe",
            return_value=mocker.Mock(symbols=("BTCUSDT",), decision_dates=()),
        )
        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1], dtype=np.int64)
        mock_cube.data_manifest_hash = "h1"
        mocker.patch(
            "src.application.futures.runner.compound_main.build_multiscale_market_cube",
            return_value=mock_cube,
        )
        mock_result = _make_mock_engine_result(mocker)
        mock_result.l2.integrity_ok = False
        mocker.patch(
            "src.application.futures.runner.compound_main.run_multiscale_compound_engine",
            return_value=mock_result,
        )

        config = CompoundRunConfig(
            reference_date="2026-07-08", sync="skip", refresh_universe=False,
        )
        result = run_multiscale_compound_main(config)
        assert result.exit_code == 1
        assert "integrity" in result.reason

    def test_reject_verdict_returns_zero(self, mocker) -> None:
        mocker.patch(
            "src.application.futures.runner.compound_main.build_data_lake_runtime",
        )
        mocker.patch(
            "src.application.futures.runner.compound_main.prepare_data_snapshot",
            return_value=mocker.Mock(snapshot_id="s1", manifest_hash="h1",
                                      reference_time_ms=1_000_000, partitions=(), total_bytes=0),
        )
        mocker.patch(
            "src.application.futures.runner.compound_main.build_daily_pit_universe",
            return_value=mocker.Mock(symbols=("BTCUSDT",), decision_dates=()),
        )
        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1], dtype=np.int64)
        mock_cube.data_manifest_hash = "h1"
        mocker.patch(
            "src.application.futures.runner.compound_main.build_multiscale_market_cube",
            return_value=mock_cube,
        )
        mock_result = _make_mock_engine_result(mocker)
        mock_result.l3.verdict = DeploymentVerdict.REJECT
        mocker.patch(
            "src.application.futures.runner.compound_main.run_multiscale_compound_engine",
            return_value=mock_result,
        )

        config = CompoundRunConfig(
            reference_date="2026-07-08", sync="skip", refresh_universe=False,
        )
        result = run_multiscale_compound_main(config)
        assert result.exit_code == 0
        assert "reject" in result.reason


def _setup_mocks(mocker) -> None:
    mocker.patch(
        "src.application.futures.runner.compound_main.build_data_lake_runtime",
    )
    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_data_snapshot",
        return_value=mocker.Mock(snapshot_id="s1", manifest_hash="h1",
                                  reference_time_ms=1_000_000, partitions=(), total_bytes=0),
    )


def test_empty_universe_error_returns_one(mocker) -> None:
    _setup_mocks(mocker)
    from src.application.futures.runner.compound_universe import (
        EmptyPITUniverseError,
    )

    mocker.patch(
        "src.application.futures.runner.compound_main.build_daily_pit_universe",
        side_effect=EmptyPITUniverseError("no symbols"),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync="skip", refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
    assert "empty_universe" in result.reason


def test_data_coverage_error_returns_one(mocker) -> None:
    _setup_mocks(mocker)
    from src.domain.futures.data_lake.ingestion import DataCoverageError

    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_data_snapshot",
        side_effect=DataCoverageError("incomplete cache"),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync="skip", refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
    assert "data_coverage" in result.reason


def test_storage_budget_error_returns_one(mocker) -> None:
    _setup_mocks(mocker)
    from src.domain.futures.data_lake.ingestion import StorageBudgetError

    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_data_snapshot",
        side_effect=StorageBudgetError("disk full"),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync="skip", refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
    assert "storage_budget" in result.reason


def test_generic_exception_returns_one(mocker) -> None:
    _setup_mocks(mocker)
    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_data_snapshot",
        side_effect=RuntimeError("unexpected"),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync="skip", refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
