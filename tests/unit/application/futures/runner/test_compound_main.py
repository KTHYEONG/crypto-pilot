from __future__ import annotations

import numpy as np
from datetime import date

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.compound_main import run_multiscale_compound_main
from src.domain.futures.data_lake.contracts import SyncMode
from src.domain.futures.data_lake.run_windows import QuarterlyWindowConfig, resolve_completed_quarter_window
from src.domain.futures.compound.contracts import (
    AlphaForecastTape,
    DeploymentVerdict,
    ExecutionLedger,
    L2Evaluation,
    L2GateVerdict,
    L3ValidationResult,
    MarketFeatureCube,
)


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
    mock_l2.verdict = L2GateVerdict.PASS
    mock_l2.annualized_log_growth = 0.05
    mock_l2.cagr = 0.051
    mock_l2.absolute_cagr = 0.052
    mock_l2.excess_growth_lcb90 = 0.01
    mock_l2.excess_growth_probability = 0.95
    mock_l2.stressed_excess_growth_lcb90 = 0.005
    mock_l2.equity_multiple = 1.1
    mock_l2.sharpe = 1.0
    mock_l2.sharpe_probability = 0.95
    mock_l2.deflated_sharpe_probability = 0.95
    mock_l2.max_drawdown = 0.02
    mock_l2.daily_cvar95 = -0.01
    mock_l2.annual_volatility = 0.15
    mock_l2.annual_turnover = 0.5
    mock_l2.cost_drag_ratio = 0.1
    mock_l2.capacity_utilisation_p95 = 0.05
    mock_l2.integrity_ok = True
    mock_l2.reasons = ()

    mock_l3 = mocker.Mock(spec=L3ValidationResult)
    mock_l3.verdict = DeploymentVerdict.PROMOTE
    mock_l3.posterior_growth_probability = 0.75
    mock_l3.holdout_days = 180
    mock_l3.max_drawdown = 0.02
    mock_l3.daily_cvar95 = -0.01
    mock_l3.reasons = ()

    result = mocker.Mock()
    result.handoff = mock_tape
    result.alpha_tape = mock_tape
    result.ledger = mock_ledger
    result.l2 = mock_l2
    result.l3 = mock_l3
    result.symbols = ("BTCUSDT", "ETHUSDT")
    return result


class TestRunMultiscaleCompoundMain:
    def test_main_is_callable(self) -> None:
        assert callable(run_multiscale_compound_main)

    def _lake_universe_mock(self, mocker, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> object:
        n_bars = 24
        n_syms = len(symbols)
        state_cube = mocker.Mock()
        state_cube.eligible = np.ones((n_bars, n_syms), dtype=np.bool_)
        state_cube.entry_block = np.zeros((n_bars, n_syms), dtype=np.bool_)
        state_cube.exit_required = np.zeros((n_bars, n_syms), dtype=np.bool_)
        state_cube.capacity_usdt = np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64)
        state_cube.risk_scale = np.ones((n_bars, n_syms), dtype=np.float64)
        state_cube.cost_bps = np.full((n_bars, n_syms), 12.0, dtype=np.float64)
        lake = mocker.Mock()
        lake.symbols = symbols
        lake.state_cube = state_cube
        lake.state_hash = "test-hash"
        return lake

    def test_happy_path_returns_zero(self, mocker) -> None:
        _setup_mocks(mocker)
        mocker.patch(
            "src.application.futures.runner.compound_main.build_daily_pit_universe",
            return_value=self._lake_universe_mock(mocker),
        )
        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1, 2], dtype=np.int64)
        mock_cube.data_manifest_hash = "h1"
        mock_cube.symbols = ("BTCUSDT", "ETHUSDT")
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
            reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
        )
        result = run_multiscale_compound_main(config)
        assert result.exit_code == 0

    def test_main_builds_universe_on_market_history_calendar(self, mocker) -> None:
        _setup_mocks(mocker)
        lake_universe = self._lake_universe_mock(mocker)
        captured: dict[str, object] = {}

        def _build_universe(**kwargs):
            captured.update(kwargs)
            return lake_universe

        mocker.patch(
            "src.application.futures.runner.compound_main.build_daily_pit_universe",
            side_effect=_build_universe,
        )
        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1, 2], dtype=np.int64)
        mock_cube.data_manifest_hash = "h1"
        mock_cube.symbols = ("BTCUSDT", "ETHUSDT")
        mocker.patch(
            "src.application.futures.runner.compound_main.build_multiscale_market_cube",
            return_value=mock_cube,
        )
        mocker.patch(
            "src.application.futures.runner.compound_main.run_multiscale_compound_engine",
            return_value=_make_mock_engine_result(mocker),
        )

        config = CompoundRunConfig(
            reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
            history_days=2,
        )
        result = run_multiscale_compound_main(config)

        calendar = captured["execution_calendar"]
        assert result.exit_code == 0
        assert len(calendar) == 48
        assert calendar[0].isoformat() == "2026-07-06T00:00:00+00:00"

    def test_integrity_failure_returns_one(self, mocker) -> None:
        _setup_mocks(mocker)
        mocker.patch(
            "src.application.futures.runner.compound_main.build_daily_pit_universe",
            return_value=self._lake_universe_mock(mocker, symbols=("BTCUSDT",)),
        )
        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1], dtype=np.int64)
        mock_cube.data_manifest_hash = "h1"
        mock_cube.symbols = ("BTCUSDT",)
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
            reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
        )
        result = run_multiscale_compound_main(config)
        assert result.exit_code == 1
        assert "integrity" in result.reason

    def test_reject_verdict_returns_zero(self, mocker) -> None:
        _setup_mocks(mocker)
        mocker.patch(
            "src.application.futures.runner.compound_main.build_daily_pit_universe",
            return_value=self._lake_universe_mock(mocker, symbols=("BTCUSDT",)),
        )
        mock_cube = mocker.Mock(spec=MarketFeatureCube)
        mock_cube.timestamps_ns = np.array([0, 1], dtype=np.int64)
        mock_cube.data_manifest_hash = "h1"
        mock_cube.symbols = ("BTCUSDT",)
        mocker.patch(
            "src.application.futures.runner.compound_main.build_multiscale_market_cube",
            return_value=mock_cube,
        )
        mock_result = _make_mock_engine_result(mocker)
        mock_result.l2.verdict = L2GateVerdict.FAIL
        mock_result.l3.verdict = DeploymentVerdict.REJECT
        mocker.patch(
            "src.application.futures.runner.compound_main.run_multiscale_compound_engine",
            return_value=mock_result,
        )

        config = CompoundRunConfig(
            reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
        )
        result = run_multiscale_compound_main(config)
        assert result.exit_code == 0
        assert "reject" in result.reason


def _setup_mocks(mocker) -> None:
    mocker.patch(
        "src.application.futures.runner.compound_main.build_data_lake_runtime",
    )
    snapshot = mocker.Mock(snapshot_id="s1", manifest_hash="h1",
                           reference_time_ms=1_000_000, partitions=(), total_bytes=0,
                           universe_state_hash="u1")
    window = resolve_completed_quarter_window(date(2026, 7, 8), QuarterlyWindowConfig())
    bootstrap = mocker.Mock(window=window, snapshot=snapshot)
    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_quarterly_bootstrap",
        return_value=bootstrap,
    )
    mocker.patch(
        "src.application.futures.runner.compound_main.finalize_quarterly_signal_data",
        return_value=mocker.Mock(field_plan=("open",), recipe_plan=()),
    )
    mocker.patch(
        "src.application.futures.runner.compound_main.exclude_symbols_with_funding_gaps",
        side_effect=lambda *, universe, **_: (universe, ()),
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
        reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
    assert "empty_universe" in result.reason


def test_data_coverage_error_returns_one(mocker) -> None:
    _setup_mocks(mocker)
    from src.domain.futures.data_lake.ingestion import DataCoverageError

    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_quarterly_bootstrap",
        side_effect=DataCoverageError("incomplete cache"),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
    assert "data_coverage" in result.reason


def test_storage_budget_error_returns_one(mocker) -> None:
    _setup_mocks(mocker)
    from src.domain.futures.data_lake.ingestion import StorageBudgetError

    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_quarterly_bootstrap",
        side_effect=StorageBudgetError("disk full"),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
    assert "storage_budget" in result.reason


def test_cash_only_engine_returns_normally(mocker) -> None:
    _setup_mocks(mocker)
    mocker.patch(
        "src.application.futures.runner.compound_main.build_daily_pit_universe",
        return_value=TestRunMultiscaleCompoundMain()._lake_universe_mock(mocker),
    )
    mock_cube = mocker.Mock(spec=MarketFeatureCube)
    mock_cube.timestamps_ns = np.array([0, 1], dtype=np.int64)
    mock_cube.data_manifest_hash = "h1"
    mock_cube.symbols = ("BTCUSDT", "ETHUSDT")
    mocker.patch(
        "src.application.futures.runner.compound_main.build_multiscale_market_cube",
        return_value=mock_cube,
    )

    from src.domain.futures.compound.contracts import DeploymentVerdict

    class FakeL2:
        verdict = L2GateVerdict.PASS
        annualized_log_growth = 0.0
        cagr = 0.0
        absolute_cagr = 0.0
        excess_growth_lcb90 = 0.0
        excess_growth_probability = 0.0
        stressed_excess_growth_lcb90 = 0.0
        equity_multiple = 1.0
        sharpe = 0.0
        sharpe_probability = 0.0
        deflated_sharpe_probability = 0.0
        max_drawdown = 0.0
        daily_cvar95 = 0.0
        annual_volatility = 0.0
        annual_turnover = 0.0
        cost_drag_ratio = 0.0
        capacity_utilisation_p95 = 0.0
        integrity_ok = True
        reasons = ()

    class FakeL3:
        verdict = DeploymentVerdict.PROMOTE
        posterior_growth_probability = 0.0
        holdout_days = 0
        max_drawdown = 0.0
        daily_cvar95 = 0.0
        reasons = ()

    class FakeHandoff:
        model_version = "v1"
        data_manifest_hash = "h1"

    class FakeLedger:
        timestamps_ns = np.array([0, 1], dtype=np.int64)
        target_weights_2d = np.zeros((2, 2), dtype=np.float32)
        integrity_ok = True

    mocker.patch(
        "src.application.futures.runner.compound_main.run_multiscale_compound_engine",
        return_value=mocker.Mock(
            ledger=FakeLedger(),
            l2=FakeL2(),
            l3=FakeL3(),
            handoff=FakeHandoff(),
            spec=["ledger", "l2", "l3", "handoff"],
        ),
    )

    result = run_multiscale_compound_main(
        CompoundRunConfig(reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False)
    )

    assert result.exit_code == 0


def test_generic_exception_returns_one(mocker) -> None:
    _setup_mocks(mocker)
    mocker.patch(
        "src.application.futures.runner.compound_main.prepare_quarterly_bootstrap",
        side_effect=RuntimeError("unexpected"),
    )
    config = CompoundRunConfig(
        reference_date="2026-07-08", sync=SyncMode.LOCAL, refresh_universe=False,
    )
    result = run_multiscale_compound_main(config)
    assert result.exit_code == 1
