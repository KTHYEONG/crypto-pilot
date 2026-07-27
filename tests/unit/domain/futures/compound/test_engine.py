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
        mock_panel.valid_3d = np.ones((256, 5, 3), dtype=bool)
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
            mu_2d=np.column_stack([
                np.full(256, 0.020, dtype=np.float32),
                np.full(256, 0.010, dtype=np.float32),
                np.full(256, 0.005, dtype=np.float32),
                np.full(256, 0.003, dtype=np.float32),
                np.full(256, -0.002, dtype=np.float32),
            ]).astype(np.float32),
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
        mock_panel.valid_3d = np.ones((256, 5, 3), dtype=bool)
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

    def test_engine_passes_per_symbol_cost_array_to_simulator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cube = _make_cube(500)
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "cost_array_wiring.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="cost-array-test",
            start_time_ns=int(cube.timestamps_ns[-30]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=30,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec_cost",
        ))

        import src.domain.futures.compound.engine as eng
        real_sim = eng.simulate_dense_portfolio
        captured: dict[str, object] = {}

        def _spy_sim(*, bars_4h, target_weights_2d, funding_1h_2d, cost_bps, config):
            captured["cost_bps"] = cost_bps
            return real_sim(
                bars_4h=bars_4h, target_weights_2d=target_weights_2d,
                funding_1h_2d=funding_1h_2d, cost_bps=cost_bps, config=config,
            )

        monkeypatch.setattr(eng, "simulate_dense_portfolio", _spy_sim)

        run_multiscale_compound_engine(
            market=cube, universe=universe, holdout_store=store,
            holdout_id="cost-array-test", config=CompoundEngineConfig(),
        )

        assert "cost_bps" in captured
        cost_arg = captured["cost_bps"]
        assert isinstance(cost_arg, np.ndarray)
        n_bars_4h = 500 // 4
        assert cost_arg.shape == (n_bars_4h, len(cube.symbols))

    def test_engine_dry_run_does_not_consume_sealed_holdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cube = _make_cube(500)
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "dry_run_guard.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="dry-run-test",
            start_time_ns=int(cube.timestamps_ns[-30]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=30,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec_dry_run",
        ))

        import src.domain.futures.compound.engine as eng

        passing_categories = tuple(
            L2CategoryResult(category=f"category-{i}", passed=True, reasons=())
            for i in range(5)
        )
        passing_l2 = L2Evaluation(
            verdict=L2GateVerdict.PASS, benchmark_id="test",
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=1.0, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=1.0,
            deflated_sharpe_probability=1.0, candidate_count=1, calmar=0.0,
            max_drawdown=0.0, daily_cvar95=0.0, annual_volatility=0.0,
            annual_turnover=0.0, cost_drag_ratio=0.0, capacity_utilisation_p95=0.0,
            active_days_ratio=1.0, rebalance_count=30, positive_outer_folds=3,
            oos_days=365, category_results=passing_categories, integrity_ok=True,
            reasons=(),
        )
        monkeypatch.setattr(eng, "evaluate_l2_walk_forward", lambda **_: passing_l2)

        def _fail_if_consumed(**_kwargs: object) -> None:
            raise AssertionError("consume() must not be called in L2_DRY_RUN mode")

        monkeypatch.setattr(store, "consume", _fail_if_consumed)
        monkeypatch.setenv("L2_DRY_RUN", "1")

        result = run_multiscale_compound_engine(
            market=cube, universe=universe, holdout_store=store,
            holdout_id="dry-run-test", config=CompoundEngineConfig(),
        )

        assert result.l2.verdict == L2GateVerdict.PASS
        assert result.l3.verdict == DeploymentVerdict.SHADOW
        assert result.l3.reasons == ("dry_run_holdout_not_consumed",)

    def test_engine_wires_aligned_benchmark_and_trial_multiplicity(
        self, tmp_path: Path, mocker, small_cube: MarketFeatureCube,
    ) -> None:
        import src.domain.futures.compound.engine as eng

        spy_benchmark = mocker.spy(eng, "build_causal_l2_benchmark")
        spy_daily_market = mocker.spy(eng, "build_daily_market_returns")
        spy_trial_returns = mocker.spy(eng, "build_candidate_trial_returns")
        spy_multiplicity = mocker.spy(eng, "compute_trial_multiplicity")
        spy_l2_eval = mocker.spy(eng, "evaluate_l2_walk_forward")

        universe = type("Universe", (), {
            "symbols": small_cube.symbols, "snapshots": (),
        })()
        store = SealedHoldoutStore(tmp_path / "engine_wiring_test.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="wiring-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)

        result = run_multiscale_compound_engine(
            market=small_cube,
            universe=universe,
            holdout_store=store,
            holdout_id="wiring-test",
            config=CompoundEngineConfig(),
        )

        assert isinstance(result, CompoundEngineResult)
        spy_daily_market.assert_called_once()
        spy_benchmark.assert_called_once()
        spy_trial_returns.assert_called_once()
        spy_multiplicity.assert_called_once()
        spy_l2_eval.assert_called_once()

        benchmark_arg = spy_benchmark.call_args.kwargs["window_timestamps_ns"]
        l2_ledger_arg = spy_l2_eval.call_args.kwargs["ledger"]
        l2_daily_ts = eng._daily_timestamps_from_4h(l2_ledger_arg.timestamps_ns)
        np.testing.assert_array_equal(benchmark_arg, l2_daily_ts)

        trial_multiplicity_arg = spy_l2_eval.call_args.kwargs["trial_multiplicity"]
        assert trial_multiplicity_arg is spy_multiplicity.spy_return


class TestAggregateTrial4hToDaily:
    def test_zero_complete_days_returns_empty_daily_array(self) -> None:
        import src.domain.futures.compound.engine as eng

        trial_returns = np.zeros((3, 5), dtype=np.float64)
        result = eng._aggregate_trial_4h_to_daily(trial_returns)
        assert result.shape == (3, 0)

    def test_one_complete_day_compounds_log_returns(self) -> None:
        import src.domain.futures.compound.engine as eng

        trial_returns = np.full((1, 6), 0.01, dtype=np.float64)
        result = eng._aggregate_trial_4h_to_daily(trial_returns)
        expected = float(np.expm1(6 * np.log1p(0.01)))
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(expected, rel=1e-10)


class TestBuildDeploymentCandidate:
    """Tests 1-4: _build_deployment_candidate unit tests per L2 turnover spec."""

    @pytest.fixture
    def valid_args(self) -> dict:
        from src.domain.futures.compound.contracts import (
            CalibratedForecastPanel,
            HandoffAdmissionEvidence,
            HandoffResult,
            L2Evaluation,
            L2GateVerdict,
            L2CategoryResult,
            RawSignalPanel,
            SealedHoldoutManifest,
            SignalDescriptor,
        )
        import numpy as np

        desc_a = SignalDescriptor(signal_id="trend_ema:fast", family="trend", speed="fast", lookback_hours=48, native_timeframe="1h")
        desc_b = SignalDescriptor(signal_id="momentum_ts:slow", family="momentum", speed="slow", lookback_hours=168, native_timeframe="4h")
        desc_c = SignalDescriptor(signal_id="mean_rev:mean", family="mean_rev", speed="mean", lookback_hours=72, native_timeframe="1h")
        unique_ids = ("trend_ema:fast", "momentum_ts:slow", "mean_rev:mean")
        # 18x fast + 10x slow + 5x mean = 33 total, 3 unique
        ids_multiset = ("trend_ema:fast",) * 18 + ("momentum_ts:slow",) * 10 + ("mean_rev:mean",) * 5

        evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.1, growth_lcb90=0.05, growth_2x_cost=0.05,
            max_drawdown=0.1, annual_volatility=0.15, positive_outer_folds=5,
            effective_breadth=3.0, active_signal_ids=ids_multiset,
            admitted=True, reasons=(),
        )
        handoff_result = HandoffResult(
            forecast=CalibratedForecastPanel(
                decision_timestamps_ns=np.array([0], dtype=np.int64),
                symbols=("A",),
                mu_2d=np.zeros((1, 1), dtype=np.float32),
                se_2d=np.zeros((1, 1), dtype=np.float32),
                family_mu_3d=np.zeros((1, 1, 1), dtype=np.float32),
                family_ids=("f",),
                admitted_signal_ids=unique_ids,
                fold_manifest_hash="h1",
            ),
            evidence=evidence,
        )
        panel = RawSignalPanel(
            decision_timestamps_ns=np.array([0], dtype=np.int64),
            symbols=("A",),
            descriptors=(desc_a, desc_b, desc_c),
            z_3d=np.zeros((1, 1, 3), dtype=np.float32),
            valid_3d=np.ones((1, 1, 3), dtype=np.bool_),
            sigma_2d=np.ones((1, 1), dtype=np.float32),
        )
        passing_categories = tuple(
            L2CategoryResult(category=f"cat-{i}", passed=True, reasons=())
            for i in range(3)
        )
        l2_eval = L2Evaluation(
            verdict=L2GateVerdict.PASS, benchmark_id="b1",
            annualized_log_growth=0.1, cagr=0.05, excess_growth_lcb90=0.05,
            excess_growth_probability=0.95, stressed_excess_growth_lcb90=0.03,
            equity_multiple=1.2, sharpe=1.5, sharpe_probability=0.95,
            deflated_sharpe_probability=0.95, candidate_count=42, calmar=0.5,
            max_drawdown=0.05, daily_cvar95=-0.01, annual_volatility=0.15,
            annual_turnover=6.63, cost_drag_ratio=0.03, capacity_utilisation_p95=0.05,
            active_days_ratio=1.0, rebalance_count=30, positive_outer_folds=5,
            oos_days=365, category_results=passing_categories, integrity_ok=True,
            reasons=(),
        )
        manifest = SealedHoldoutManifest(
            holdout_id="test", start_time_ns=0, end_time_ns=1,
            holdout_days=90, model_version="v2", data_manifest_hash="dh1",
            strategy_spec_hash="sh1",
        )
        forecast = CalibratedForecastPanel(
            decision_timestamps_ns=np.array([0], dtype=np.int64),
            symbols=("A",),
            mu_2d=np.zeros((1, 1), dtype=np.float32),
            se_2d=np.zeros((1, 1), dtype=np.float32),
            family_mu_3d=np.zeros((1, 1, 1), dtype=np.float32),
            family_ids=("f",),
            admitted_signal_ids=unique_ids,
            fold_manifest_hash="fh1",
        )
        return {
            "handoff_result": handoff_result,
            "panel": panel,
            "l2_eval": l2_eval,
            "manifest": manifest,
            "forecast": forecast,
            "strategy_spec_hash": "sh1",
            "fold_manifest_hash": "fh1",
        }

    def test_build_deployment_candidate_deduplicates_and_frequency_weights(self, valid_args: dict) -> None:
        from src.domain.futures.compound.engine import _build_deployment_candidate
        result = _build_deployment_candidate(**valid_args)
        assert result is not None
        assert len(result.active_signal_ids) == 3
        assert sum(result.vote_weights) == pytest.approx(1.0)
        assert result.vote_weights[0] == pytest.approx(18 / 33)
        assert result.vote_weights[1] == pytest.approx(10 / 33)
        assert result.vote_weights[2] == pytest.approx(5 / 33)

    def test_build_deployment_candidate_returns_none_when_not_admitted_or_not_pass(self, valid_args: dict) -> None:
        from src.domain.futures.compound.engine import _build_deployment_candidate
        from src.domain.futures.compound.contracts import (
            HandoffAdmissionEvidence, HandoffResult,
        )
        ev = valid_args["handoff_result"].evidence
        not_admitted = HandoffAdmissionEvidence(
            annualized_log_growth=ev.annualized_log_growth,
            growth_lcb90=ev.growth_lcb90, growth_2x_cost=ev.growth_2x_cost,
            max_drawdown=ev.max_drawdown, annual_volatility=ev.annual_volatility,
            positive_outer_folds=ev.positive_outer_folds,
            effective_breadth=ev.effective_breadth,
            active_signal_ids=ev.active_signal_ids,
            admitted=False, reasons=("test",),
        )
        args = dict(valid_args)
        args["handoff_result"] = HandoffResult(
            forecast=valid_args["handoff_result"].forecast,
            evidence=not_admitted,
        )
        assert _build_deployment_candidate(**args) is None

    def test_l2_not_pass_returns_none(self, valid_args: dict) -> None:
        from src.domain.futures.compound.engine import _build_deployment_candidate
        from src.domain.futures.compound.contracts import L2CategoryResult, L2Evaluation, L2GateVerdict
        fail_categories = (
            L2CategoryResult(category="cat-0", passed=False, reasons=("fail",)),
            L2CategoryResult(category="cat-1", passed=True, reasons=()),
        )
        fail_eval = L2Evaluation(
            verdict=L2GateVerdict.FAIL, benchmark_id="b1",
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=0.0, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=0.0,
            deflated_sharpe_probability=0.0, candidate_count=0, calmar=0.0,
            max_drawdown=0.0, daily_cvar95=0.0, annual_volatility=0.0,
            annual_turnover=0.0, cost_drag_ratio=0.0, capacity_utilisation_p95=0.0,
            active_days_ratio=1.0, rebalance_count=30, positive_outer_folds=3,
            oos_days=365, category_results=fail_categories,
            integrity_ok=True, reasons=("test_fail",),
        )
        args = dict(valid_args)
        args["l2_eval"] = fail_eval
        assert _build_deployment_candidate(**args) is None

    def test_build_deployment_candidate_unmatched_signal_id_raises_value_error(self, valid_args: dict) -> None:
        from src.domain.futures.compound.engine import _build_deployment_candidate
        from src.domain.futures.compound.contracts import (
            HandoffAdmissionEvidence, HandoffResult,
        )
        bad_evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.1, growth_lcb90=0.05, growth_2x_cost=0.05,
            max_drawdown=0.1, annual_volatility=0.15, positive_outer_folds=5,
            effective_breadth=1.0,
            active_signal_ids=("trend_ema:fast", "nonexistent:sig"),
            admitted=True, reasons=(),
        )
        bad_handoff = HandoffResult(
            forecast=valid_args["handoff_result"].forecast,
            evidence=bad_evidence,
        )
        args = dict(valid_args)
        args["handoff_result"] = bad_handoff
        with pytest.raises(ValueError, match="unmatched signal ids"):
            _build_deployment_candidate(**args)


class TestEngineL2PassBuildsDeploymentCandidate:
    def test_engine_l2_pass_builds_deployment_candidate_without_crash(
        self, tmp_path, mocker, small_cube: MarketFeatureCube,
    ) -> None:
        from src.domain.futures.compound.contracts import (
            CalibratedForecastPanel,
            CausalFold,
            HandoffResult,
            HandoffAdmissionEvidence,
            L2CategoryResult,
            L2Evaluation,
            L2GateVerdict,
            RawSignalPanel,
        )

        desc_a = mocker.Mock(spec=SignalDescriptor, signal_id="trend_ema:fast", target_horizon_hours=4)
        desc_b = mocker.Mock(spec=SignalDescriptor, signal_id="momentum_ts:slow", target_horizon_hours=4)
        mock_panel = mocker.Mock(spec=RawSignalPanel)
        mock_panel.z_3d = np.zeros((256, 5, 3))
        mock_panel.valid_3d = np.ones((256, 5, 3), dtype=bool)
        mock_panel.sigma_2d = np.full((256, 5), 0.01, dtype=np.float32)
        mock_panel.descriptors = (desc_a, desc_b)
        mocker.patch(
            "src.domain.futures.compound.engine.build_raw_signal_panel",
            return_value=mock_panel,
        )
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
            admitted_signal_ids=("trend_ema:fast", "momentum_ts:slow"),
            fold_manifest_hash="test",
        )
        # duplicated multiset: 3x fast, 1x slow (mirrors the real cluster/fold sleeve fan-out)
        ids_multiset = ("trend_ema:fast",) * 3 + ("momentum_ts:slow",) * 1
        evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.1, growth_lcb90=0.05, growth_2x_cost=0.05,
            max_drawdown=0.1, annual_volatility=0.15, positive_outer_folds=5,
            effective_breadth=2.0, active_signal_ids=ids_multiset,
            admitted=True, reasons=(),
        )
        handoff_result = HandoffResult(forecast=forecast_panel, evidence=evidence)
        mocker.patch(
            "src.domain.futures.compound.engine.build_exit_aware_handoff",
            return_value=handoff_result,
        )

        passing_categories = tuple(
            L2CategoryResult(category=f"category-{i}", passed=True, reasons=())
            for i in range(5)
        )
        passing_l2 = L2Evaluation(
            verdict=L2GateVerdict.PASS, benchmark_id="test",
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=1.0, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=1.0,
            deflated_sharpe_probability=1.0, candidate_count=1, calmar=0.0,
            max_drawdown=0.0, daily_cvar95=0.0, annual_volatility=0.0,
            annual_turnover=0.0, cost_drag_ratio=0.0, capacity_utilisation_p95=0.0,
            active_days_ratio=1.0, rebalance_count=30, positive_outer_folds=3,
            oos_days=365, category_results=passing_categories, integrity_ok=True,
            reasons=(),
        )
        import src.domain.futures.compound.engine as eng
        mocker.patch.object(eng, "evaluate_l2_walk_forward", return_value=passing_l2)
        mocker.patch.object(
            eng, "evaluate_l3_sealed_holdout",
            return_value=L3ValidationResult(
                verdict=DeploymentVerdict.SHADOW, posterior_growth_probability=0.5,
                holdout_days=90, max_drawdown=0.0, daily_cvar95=0.0, reasons=(),
            ),
        )

        universe = type("Universe", (), {"symbols": small_cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "l2_pass_deployment_candidate.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="l2-pass-deploy-test",
            start_time_ns=int(small_cube.timestamps_ns[-180]),
            end_time_ns=int(small_cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)

        result = run_multiscale_compound_engine(
            market=small_cube, universe=universe,
            holdout_store=store, holdout_id="l2-pass-deploy-test",
            config=CompoundEngineConfig(),
        )

        assert result.l2.verdict == L2GateVerdict.PASS
        assert result.deployment_candidate is not None
        assert result.deployment_candidate.active_signal_ids == ("trend_ema:fast", "momentum_ts:slow")
        assert result.deployment_candidate.vote_weights[0] == pytest.approx(3 / 4)
        assert result.deployment_candidate.vote_weights[1] == pytest.approx(1 / 4)


    def test_window_none_without_holdout_id_raises(self) -> None:
        from src.domain.futures.compound.config import CompoundEngineConfig
        from src.domain.futures.compound.contracts import MarketFeatureCube

        n_bars, n_syms = 10, 2
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * _NS_PER_HOUR,
            symbols=("A", "B"),
            fields_2d={"close": np.ones((n_bars, n_syms), dtype=np.float32) * 100.0},
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.zeros((n_bars, n_syms), dtype=np.float64),
            execution_cost_bps_2d=np.zeros((n_bars, n_syms), dtype=np.float32),
            data_manifest_hash="h",
        )
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()
        from unittest.mock import MagicMock
        store = MagicMock()
        with pytest.raises(ValueError, match="window=None requires holdout_id"):
            run_multiscale_compound_engine(
                market=cube, universe=universe,
                window=None, holdout_id=None,
                holdout_store=store, config=CompoundEngineConfig(),
            )


    def test_run_multiscale_compound_engine_requires_window_or_holdout_id(self) -> None:
        self.test_window_none_without_holdout_id_raises()


    def test_l3_prior_slices_to_most_recent_cap_days(self, tmp_path, mocker, small_cube) -> None:
        from src.domain.futures.compound.contracts import (
            CalibratedForecastPanel, HandoffResult, HandoffAdmissionEvidence, CausalFold,
            L2CategoryResult, L2Evaluation, L2GateVerdict, L3ValidationResult, DeploymentVerdict,
            RawSignalPanel, SignalDescriptor,
        )
        import src.domain.futures.compound.engine as eng

        desc = mocker.Mock(spec=SignalDescriptor, signal_id="sig1", target_horizon_hours=4)
        mock_panel = mocker.Mock(spec=RawSignalPanel)
        mock_panel.z_3d = np.zeros((256, 5, 1))
        mock_panel.valid_3d = np.ones((256, 5, 1), dtype=bool)
        mock_panel.sigma_2d = np.full((256, 5), 0.01, dtype=np.float32)
        mock_panel.descriptors = (desc,)
        mocker.patch("src.domain.futures.compound.engine.build_raw_signal_panel", return_value=mock_panel)
        mock_folds = (CausalFold(0, 0, 50, 48, 50, 52, 102, 2, 42),)
        mocker.patch("src.domain.futures.compound.engine.build_folds_4h", return_value=mock_folds)
        mocker.patch("src.domain.futures.compound.engine.build_multi_horizon_targets", return_value={})
        mocker.patch("src.domain.futures.compound.engine.align_costs_to_decision_grid", return_value=np.full((256, 5), 8.0, dtype=np.float32))

        forecast_panel = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(256, dtype=np.int64),
            symbols=small_cube.symbols,
            mu_2d=np.zeros((256, 5), dtype=np.float32),
            se_2d=np.full((256, 5), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((256, 5, 1), dtype=np.float32),
            family_ids=(), admitted_signal_ids=("sig1",), fold_manifest_hash="test",
        )
        evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.1, growth_lcb90=0.05, growth_2x_cost=0.05,
            max_drawdown=0.1, annual_volatility=0.15, positive_outer_folds=5,
            effective_breadth=1.0, active_signal_ids=("sig1",), admitted=False, reasons=("test",),
        )
        hidden = HandoffResult(forecast=forecast_panel, evidence=evidence)
        mocker.patch("src.domain.futures.compound.engine.build_exit_aware_handoff", return_value=hidden)

        # A real quarterly L2 window (365 days) always exceeds l2_prior_effective_days_cap
        # (60 by default): the cap bounds how much recent history feeds the L3 posterior,
        # it is not a maximum-plausible-length sanity check. A prior implementation wrongly
        # raised whenever daily_len > cap, which crashed every production run (engine.py).
        from dataclasses import replace
        config = replace(
            CompoundEngineConfig(),
            l3=replace(CompoundEngineConfig().l3, l2_prior_effective_days_cap=5),
        )
        universe = type("Universe", (), {"symbols": small_cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "l3_prior_check.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="prior-check", start_time_ns=int(small_cube.timestamps_ns[-30]),
            end_time_ns=int(small_cube.timestamps_ns[-1]), holdout_days=30,
            model_version="v1", data_manifest_hash="h1", strategy_spec_hash="spec",
        ))
        daily_prior = np.arange(100, dtype=np.float64) * 0.001
        spy = mocker.patch.object(eng, "evaluate_l3_sealed_holdout", wraps=eng.evaluate_l3_sealed_holdout)
        with mocker.patch.object(eng, "aggregate_returns_to_utc_days", return_value=daily_prior):
            run_multiscale_compound_engine(
                market=small_cube, universe=universe, holdout_store=store,
                holdout_id="prior-check", config=config,
            )
        passed_prior = spy.call_args.kwargs["l2_prior_returns"]
        assert passed_prior.shape == (5,)
        np.testing.assert_array_equal(passed_prior, daily_prior[-5:])

    def test_l3_prior_empty_daily_aggregation_falls_back_to_zero(self, tmp_path, mocker, small_cube) -> None:
        from src.domain.futures.compound.contracts import (
            CalibratedForecastPanel, HandoffResult, HandoffAdmissionEvidence, CausalFold,
            RawSignalPanel, SignalDescriptor,
        )
        import src.domain.futures.compound.engine as eng

        desc = mocker.Mock(spec=SignalDescriptor, signal_id="sig1", target_horizon_hours=4)
        mock_panel = mocker.Mock(spec=RawSignalPanel)
        mock_panel.z_3d = np.zeros((256, 5, 1))
        mock_panel.valid_3d = np.ones((256, 5, 1), dtype=bool)
        mock_panel.sigma_2d = np.full((256, 5), 0.01, dtype=np.float32)
        mock_panel.descriptors = (desc,)
        mocker.patch("src.domain.futures.compound.engine.build_raw_signal_panel", return_value=mock_panel)
        mock_folds = (CausalFold(0, 0, 50, 48, 50, 52, 102, 2, 42),)
        mocker.patch("src.domain.futures.compound.engine.build_folds_4h", return_value=mock_folds)
        mocker.patch("src.domain.futures.compound.engine.build_multi_horizon_targets", return_value={})
        mocker.patch("src.domain.futures.compound.engine.align_costs_to_decision_grid", return_value=np.full((256, 5), 8.0, dtype=np.float32))

        forecast_panel = CalibratedForecastPanel(
            decision_timestamps_ns=np.arange(256, dtype=np.int64),
            symbols=small_cube.symbols,
            mu_2d=np.zeros((256, 5), dtype=np.float32),
            se_2d=np.full((256, 5), 0.01, dtype=np.float32),
            family_mu_3d=np.zeros((256, 5, 1), dtype=np.float32),
            family_ids=(), admitted_signal_ids=("sig1",), fold_manifest_hash="test",
        )
        evidence = HandoffAdmissionEvidence(
            annualized_log_growth=0.1, growth_lcb90=0.05, growth_2x_cost=0.05,
            max_drawdown=0.1, annual_volatility=0.15, positive_outer_folds=5,
            effective_breadth=1.0, active_signal_ids=("sig1",), admitted=False, reasons=("test",),
        )
        hidden = HandoffResult(forecast=forecast_panel, evidence=evidence)
        mocker.patch("src.domain.futures.compound.engine.build_exit_aware_handoff", return_value=hidden)

        universe = type("Universe", (), {"symbols": small_cube.symbols, "snapshots": ()})()
        store = SealedHoldoutStore(tmp_path / "l3_prior_empty.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="prior-empty", start_time_ns=int(small_cube.timestamps_ns[-30]),
            end_time_ns=int(small_cube.timestamps_ns[-1]), holdout_days=30,
            model_version="v1", data_manifest_hash="h1", strategy_spec_hash="spec",
        ))
        with mocker.patch.object(eng, "aggregate_returns_to_utc_days", return_value=np.zeros(0, dtype=np.float64)):
            result = run_multiscale_compound_engine(
                market=small_cube, universe=universe, holdout_store=store,
                holdout_id="prior-empty", config=CompoundEngineConfig(),
            )
        # empty daily aggregation must fall back to a zero-length-1 prior (engine.py L425)
        # instead of raising or leaving prior_returns undefined; the run completes and
        # l2_not_pass (cash-only, admitted=False) drives a deterministic REJECT.
        assert result.l3.verdict == DeploymentVerdict.REJECT
        assert result.l3.reasons == ("l2_not_pass",)
