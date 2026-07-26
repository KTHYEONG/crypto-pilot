from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    DeploymentVerdict,
    L2CategoryResult,
    L2Evaluation,
    L2GateVerdict,
    L3ValidationResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
    SignalDescriptor,
)
from src.domain.futures.compound.engine import run_multiscale_compound_engine
from src.domain.futures.compound.holdout_store import HoldoutReuseError, SealedHoldoutStore

_NS_PER_HOUR = 3_600_000_000_000


def _make_cube(
    n_bars: int,
    n_syms: int = 5,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"),
) -> MarketFeatureCube:
    close = np.column_stack(tuple(
        np.linspace(100, 110 + i, n_bars) for i in range(n_syms)
    )).astype(np.float64)
    arr_f32 = close.astype(np.float32)
    return MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * _NS_PER_HOUR,
        symbols=symbols,
        fields_2d={
            "open": arr_f32 * 0.9995,
            "high": arr_f32 * 1.005,
            "low": arr_f32 * 0.995,
            "close": arr_f32,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 50_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            "mark": arr_f32.copy(),
            "index": arr_f32.copy(),
            "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 25_000_000,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )


@pytest.fixture
def small_cube() -> MarketFeatureCube:
    return _make_cube(1024)


class TestRunMultiscaleCompoundEngine:
    def test_returns_compound_engine_result(self, tmp_path, small_cube: MarketFeatureCube) -> None:
        n_syms = len(small_cube.symbols)
        universe = type("Universe", (), {
            "symbols": small_cube.symbols, "snapshots": (),
        })()
        store = SealedHoldoutStore(tmp_path / "engine_test.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="engine-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        result = run_multiscale_compound_engine(
            market=small_cube,
            universe=universe,
            holdout_store=store,
            holdout_id="engine-test",
            config=config,
        )
        assert isinstance(result, CompoundEngineResult)
        assert result.ledger is not None
        assert result.l2 is not None
        assert result.l3 is not None

    def test_engine_idempotent_on_repeat(self, tmp_path, small_cube: MarketFeatureCube) -> None:
        universe = type("Universe", (), {
            "symbols": small_cube.symbols, "snapshots": (),
        })()
        store = SealedHoldoutStore(tmp_path / "engine_idem.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="idem-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        r1 = run_multiscale_compound_engine(
            market=small_cube, universe=universe,
            holdout_store=store, holdout_id="idem-test", config=config,
        )
        r2 = run_multiscale_compound_engine(
            market=small_cube, universe=universe,
            holdout_store=store, holdout_id="idem-test", config=config,
        )
        assert r1.l3 == r2.l3

    def test_stale_holdout_manifest_hash_mismatch_raises(self, tmp_path, small_cube: MarketFeatureCube) -> None:
        universe = type("Universe", (), {
            "symbols": small_cube.symbols, "snapshots": (),
        })()
        store = SealedHoldoutStore(tmp_path / "engine_hash_mismatch.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="hash-mismatch-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="stale_hash_from_prior_run",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        with pytest.raises(HoldoutReuseError, match="hash mismatch"):
            run_multiscale_compound_engine(
                market=small_cube, universe=universe,
                holdout_store=store, holdout_id="hash-mismatch-test", config=config,
            )

    def test_admitted_signals_path_produces_weights(self, tmp_path, mocker, small_cube: MarketFeatureCube) -> None:
        from src.domain.futures.compound.contracts import (
            CalibratedForecastPanel,
            HandoffResult,
            HandoffAdmissionEvidence,
            RawSignalPanel,
        )

        mock_panel = mocker.Mock(spec=RawSignalPanel)
        mock_panel.z_3d = np.zeros((256, 5, 3))
        mock_panel.sigma_2d = np.full((256, 5), 0.01, dtype=np.float32)
        mock_panel.descriptors = (mocker.Mock(spec=SignalDescriptor, target_horizon_hours=4),)
        mocker.patch(
            "src.domain.futures.compound.engine.build_raw_signal_panel",
            return_value=mock_panel,
        )
        from src.domain.futures.compound.contracts import CausalFold
        mock_folds = (CausalFold(0, 0, 50, 48, 50, 52, 102, 2, 42),)
        mocker.patch("src.domain.futures.compound.engine.build_folds_4h", return_value=mock_folds)
        mocker.patch("src.domain.futures.compound.engine.build_multi_horizon_targets", return_value={4: mocker.Mock()})
        mocker.patch("src.domain.futures.compound.engine.align_costs_to_decision_grid", return_value=np.full((256, 5), 8.0, dtype=np.float32))

        forecast_panel = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(256, dtype=np.int64),
            symbols=small_cube.symbols,
            mu_2d=np.ones((256, len(small_cube.symbols)), dtype=np.float32) * 0.001,
            se_2d=np.full((256, len(small_cube.symbols)), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((256, len(small_cube.symbols), 1), dtype=np.float32),
            family_ids=(),
            admitted_signal_ids=("sig1",),
            fold_manifest_hash="test",
        )
        evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.1, growth_lcb90=0.05, growth_2x_cost=0.05,
            max_drawdown=0.1, annual_volatility=0.15, positive_outer_folds=5,
            effective_breadth=1.0, active_signal_ids=("sig1",),
            admitted=True, reasons=(),
        )
        handoff_result = HandoffResult(forecast=forecast_panel, evidence=evidence)
        mocker.patch(
            "src.domain.futures.compound.engine.build_exit_aware_handoff",
            return_value=handoff_result,
        )

        n_syms = len(small_cube.symbols)
        universe = type("Universe", (), {"symbols": small_cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "admitted_test.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="admitted-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        result = run_multiscale_compound_engine(
            market=small_cube, universe=universe,
            holdout_store=store, holdout_id="admitted-test", config=config,
        )
        assert isinstance(result, CompoundEngineResult)
        assert not np.allclose(result.ledger.target_weights_2d, 0.0)

    def test_engine_passes_sigma_to_path(self, tmp_path, mocker, small_cube: MarketFeatureCube) -> None:
        from src.domain.futures.compound.contracts import (
            CalibratedForecastPanel,
            HandoffResult,
            HandoffAdmissionEvidence,
            RawSignalPanel,
        )

        distinct_sigma = np.full((256, 5), 0.037, dtype=np.float32)
        mock_panel = mocker.Mock(spec=RawSignalPanel)
        mock_panel.z_3d = np.zeros((256, 5, 3))
        mock_panel.sigma_2d = distinct_sigma
        mock_panel.descriptors = (mocker.Mock(spec=SignalDescriptor, target_horizon_hours=4),)
        mocker.patch(
            "src.domain.futures.compound.engine.build_raw_signal_panel",
            return_value=mock_panel,
        )
        from src.domain.futures.compound.contracts import CausalFold
        mock_folds = (CausalFold(0, 0, 50, 48, 50, 52, 102, 2, 42),)
        mocker.patch("src.domain.futures.compound.engine.build_folds_4h", return_value=mock_folds)
        mocker.patch("src.domain.futures.compound.engine.build_multi_horizon_targets", return_value={4: mocker.Mock()})
        mocker.patch("src.domain.futures.compound.engine.align_costs_to_decision_grid", return_value=np.full((256, 5), 8.0, dtype=np.float32))

        forecast_panel = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(256, dtype=np.int64),
            symbols=small_cube.symbols,
            mu_2d=np.ones((256, len(small_cube.symbols)), dtype=np.float32) * 0.001,
            se_2d=np.full((256, len(small_cube.symbols)), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((256, len(small_cube.symbols), 1), dtype=np.float32),
            family_ids=(),
            admitted_signal_ids=("sig1",),
            fold_manifest_hash="test",
        )
        evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.1, growth_lcb90=0.05, growth_2x_cost=0.05,
            max_drawdown=0.1, annual_volatility=0.15, positive_outer_folds=5,
            effective_breadth=1.0, active_signal_ids=("sig1",),
            admitted=True, reasons=(),
        )
        handoff_result = HandoffResult(forecast=forecast_panel, evidence=evidence)
        mocker.patch(
            "src.domain.futures.compound.engine.build_exit_aware_handoff",
            return_value=handoff_result,
        )
        path_spy = mocker.patch(
            "src.domain.futures.compound.engine.compute_dynamic_compounding_path",
            return_value=np.zeros((256, len(small_cube.symbols)), dtype=np.float64),
        )

        universe = type("Universe", (), {"symbols": small_cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "sigma_wiring_test.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="sigma-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)
        run_multiscale_compound_engine(
            market=small_cube, universe=universe,
            holdout_store=store, holdout_id="sigma-test", config=CompoundEngineConfig(),
        )

        assert path_spy.call_count == 1
        np.testing.assert_array_equal(path_spy.call_args.kwargs["sigma_2d"], distinct_sigma)

    def test_missing_close_field_raises(self, tmp_path) -> None:
        n_bars, n_syms = 10, 2
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64),
            symbols=("A", "B"),
            fields_2d={},
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.zeros((n_bars, n_syms), dtype=np.float64),
            execution_cost_bps_2d=np.zeros((n_bars, n_syms), dtype=np.float32),
            data_manifest_hash="h",
        )
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "missing_close.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="mc", start_time_ns=0, end_time_ns=1,
            holdout_days=30, model_version="v1", data_manifest_hash="h",
            strategy_spec_hash="s",
        )
        store.create(manifest)
        with pytest.raises(ValueError, match="missing close"):
            run_multiscale_compound_engine(
                market=cube, universe=universe,
                holdout_store=store, holdout_id="mc",
                config=CompoundEngineConfig(),
            )

    def test_allocate_portfolio_step_wiring(self) -> None:
        from src.domain.futures.compound.engine import allocate_portfolio_step
        from src.domain.futures.compound.config import DynamicCompoundingConfig
        from src.domain.futures.compound.contracts import CombinedForecast

        forecast = CombinedForecast(
            mu_robust_1d=np.array([0.005, -0.003], dtype=np.float64),
            variance_1d=np.array([1e-4, 1e-4], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        sigma_1d = np.array([0.01, 0.01], dtype=np.float64)
        funding = np.array([0.0001, -0.0001], dtype=np.float64)
        prev = np.zeros(2, dtype=np.float64)
        config = DynamicCompoundingConfig()
        result = allocate_portfolio_step(
            forecast=forecast, sigma_1d=sigma_1d,
            funding_rates=funding, previous_weights=prev,
            config=config,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] > 0
        assert result[1] < 0
        assert np.all(np.isfinite(result))

    def test_cash_only_no_admitted_signals(self, tmp_path) -> None:
        n_bars, n_syms = 500, 3
        close = np.ones((n_bars, n_syms), dtype=np.float32) * 100.0
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * _NS_PER_HOUR,
            symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            fields_2d={
                "close": close,
                "open": close * 0.9995,
                "high": close * 1.005,
                "low": close * 0.995,
                "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 1_000_000,
                "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
                "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
                "mark": close.copy(),
                "index": close.copy(),
                "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 500_000,
            },
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 8.0, dtype=np.float32),
            data_manifest_hash="cash_test",
        )
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "cash_engine.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="cash-test",
            start_time_ns=int(cube.timestamps_ns[-30]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=30,
            model_version="v1",
            data_manifest_hash="cash_test",
            strategy_spec_hash="spec_cash",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        result = run_multiscale_compound_engine(
            market=cube, universe=universe,
            holdout_store=store, holdout_id="cash-test", config=config,
        )
        assert isinstance(result, CompoundEngineResult)
        assert np.allclose(result.ledger.target_weights_2d, 0.0, atol=1e-10)

    def test_p2_exception_reaches_l2_and_l3_rejects(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.domain.futures.compound.contracts import (
            CompoundEngineResult, L2GateVerdict, DeploymentVerdict,
        )
        n = 500
        cube = _make_cube(n)
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()

        def _raise_p2(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated P2 failure")

        import src.domain.futures.compound.engine as eng
        monkeypatch.setattr(eng, "build_exit_aware_handoff", _raise_p2)

        store = SealedHoldoutStore(tmp_path / "p2_error_test.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="p2-error-test",
            start_time_ns=int(cube.timestamps_ns[-30]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=30,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec_p2",
        )
        store.create(manifest)
        config = CompoundEngineConfig()
        result = run_multiscale_compound_engine(
            market=cube, universe=universe,
            holdout_store=store, holdout_id="p2-error-test", config=config,
        )
        assert isinstance(result, CompoundEngineResult)
        assert result.l2.verdict == L2GateVerdict.NO_EVIDENCE
        assert any("p2_pipeline_error" in r for r in result.l2.reasons)
        assert result.l3.verdict == DeploymentVerdict.REJECT
        assert "l2_not_pass" in result.l3.reasons

    def test_l2_pass_consumes_sealed_holdout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cube = _make_cube(500)
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "l2_pass_consumes.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="l2-pass-test",
            start_time_ns=int(cube.timestamps_ns[-30]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=30,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec_p2",
        ))

        import src.domain.futures.compound.engine as eng

        passing_categories = tuple(
            L2CategoryResult(category=f"category-{i}", passed=True, reasons=())
            for i in range(5)
        )
        passing_l2 = L2Evaluation(
            verdict=L2GateVerdict.PASS,
            benchmark_id="test",
            annualized_log_growth=0.0,
            cagr=0.0,
            excess_growth_lcb90=0.0,
            excess_growth_probability=1.0,
            stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0,
            sharpe=0.0,
            sharpe_probability=1.0,
            deflated_sharpe_probability=1.0,
            candidate_count=1,
            calmar=0.0,
            max_drawdown=0.0,
            daily_cvar95=0.0,
            annual_volatility=0.0,
            annual_turnover=0.0,
            cost_drag_ratio=0.0,
            capacity_utilisation_p95=0.0,
            active_days_ratio=1.0,
            rebalance_count=30,
            positive_outer_folds=3,
            oos_days=365,
            category_results=passing_categories,
            integrity_ok=True,
            reasons=(),
        )
        monkeypatch.setattr(eng, "evaluate_l2_walk_forward", lambda **_: passing_l2)
        monkeypatch.setattr(
            eng,
            "evaluate_l3_sealed_holdout",
            lambda **_: L3ValidationResult(
                verdict=DeploymentVerdict.SHADOW,
                posterior_growth_probability=0.5,
                holdout_days=30,
                max_drawdown=0.0,
                daily_cvar95=0.0,
                reasons=(),
            ),
        )

        result = run_multiscale_compound_engine(
            market=cube,
            universe=universe,
            holdout_store=store,
            holdout_id="l2-pass-test",
            config=CompoundEngineConfig(),
        )
        assert result.l2.verdict == L2GateVerdict.PASS
        assert result.l3.verdict == DeploymentVerdict.SHADOW
