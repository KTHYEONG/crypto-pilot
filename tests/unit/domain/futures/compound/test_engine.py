from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
    SignalDescriptor,
)
from src.domain.futures.compound.engine import run_multiscale_compound_engine
from src.domain.futures.compound.holdout_store import SealedHoldoutStore

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

    def test_admitted_signals_path_produces_weights(self, tmp_path, mocker, small_cube: MarketFeatureCube) -> None:
        from src.domain.futures.compound.contracts import (
            CalibratedForecastPanel,
            RawSignalPanel,
        )

        mock_panel = mocker.Mock(spec=RawSignalPanel)
        mock_panel.z_3d = np.zeros((256, 5, 3))
        mock_panel.sigma_2d = np.ones((256, 5), dtype=np.float32)
        mock_panel.descriptors = (mocker.Mock(spec=SignalDescriptor, target_horizon_hours=4),)
        mocker.patch(
            "src.domain.futures.compound.engine.build_raw_signal_panel",
            return_value=mock_panel,
        )
        mocker.patch("src.domain.futures.compound.engine.build_folds_4h", return_value=())
        mocker.patch("src.domain.futures.compound.engine.build_multi_horizon_targets", return_value={4: mocker.Mock()})
        mocker.patch("src.domain.futures.compound.engine.calibrate_signals", return_value=())
        mocker.patch("src.domain.futures.compound.engine.evaluate_signal_admission", return_value=())
        mocker.patch(
            "src.domain.futures.compound.engine.combine_admitted_forecasts",
            return_value=CalibratedForecastPanel(
                decision_timestamps_ns=np.arange(256, dtype=np.int64),
                symbols=small_cube.symbols,
                mu_2d=np.ones((256, len(small_cube.symbols)), dtype=np.float32) * 0.001,
                se_2d=np.full((256, len(small_cube.symbols)), 0.01, dtype=np.float32),
                family_mu_3d=np.zeros((256, len(small_cube.symbols), 1), dtype=np.float32),
                family_ids=("test",),
                admitted_signal_ids=("sig1",),
                fold_manifest_hash="test",
            ),
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
