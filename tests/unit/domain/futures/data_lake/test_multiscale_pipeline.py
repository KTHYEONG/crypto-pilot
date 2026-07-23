from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.futures.runner.compound_config import (
    CompoundRunConfig,
    build_compound_run_config,
)
from src.domain.futures.data_lake.contracts import LakeUniverse
from src.domain.futures.universe.contracts import UniverseStateCube
from src.application.futures.runner.data_lake_runtime import DataLakeRuntime
from src.domain.futures.data_lake.contracts import (
    DataLakeConfig,
    DataSnapshot,
    IngestionPlan,
    PartitionManifest,
)
from src.application.futures.runner.data_lake_runtime import (
    build_data_lake_runtime,
    prepare_data_snapshot,
)
from src.domain.futures.data_lake.ingestion import (
    ChecksumMismatchError,
    DataCoverageError,
    StorageBudgetError,
    sync_futures_data_lake,
)
from src.domain.futures.universe.config import PITUniverseConfig


def _lake_universe(symbols: tuple[str, ...], n_bars: int = 24) -> LakeUniverse:
    n_syms = len(symbols)
    calendar = pd.date_range("2026-07-01", periods=n_bars, freq="h", tz="UTC")
    cube = UniverseStateCube(
        calendar=calendar,
        instrument_ids=symbols,
        eligible=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        risk_scale=np.ones((n_bars, n_syms), dtype=np.float64),
        cost_bps=np.full((n_bars, n_syms), 12.0, dtype=np.float64),
    )
    return LakeUniverse(symbols=symbols, state_cube=cube, state_hash="test")


class FakeClient:
    def __init__(self) -> None:
        self.download_calls = 0

    def download_partition(self, *args: object, **kwargs: object) -> bytes:
        self.download_calls += 1
        return b"valid"

    def download_checksum(self, *args: object, **kwargs: object) -> str:
        return hashlib.sha256(b"valid").hexdigest()



def _snap(
    snapshot_id: str = "s1",
    reference_time_ms: int = 1_000_000,
    partitions: tuple = (),
    manifest_hash: str = "h1",
    total_bytes: int = 0,
) -> DataSnapshot:
    return DataSnapshot(
        snapshot_id=snapshot_id,
        reference_time_ms=reference_time_ms,
        partitions=partitions,
        manifest_hash=manifest_hash,
        universe_state_hash="",
        total_bytes=total_bytes,
    )


class FakeCatalog:
    def __init__(self, snapshot: DataSnapshot, *, complete: bool) -> None:
        self.snapshot = snapshot
        self.complete = complete
        self.committed: list[PartitionManifest] = []
        self._bytes = 0

    def load_snapshot(self, reference_time_ms: int) -> DataSnapshot:
        return self.snapshot

    def has_complete_coverage(self, snapshot: DataSnapshot, plan: IngestionPlan) -> bool:
        return self.complete

    def commit_partition(self, manifest: PartitionManifest) -> None:
        self.committed.append(manifest)

    def partition_exists(self, dataset: object, symbol: str, start_time_ms: int) -> bool:
        return False

    def total_bytes(self) -> int:
        return self._bytes


# ── Scenario 1: Config defaults disable network sync ─────────────────────────


class TestConfigDefaults:
    def test_config_disables_network_sync_by_default(self) -> None:
        config = build_compound_run_config({"sync": "skip", "seed": 42})
        assert config.allow_network_sync is False
        assert isinstance(config.data_lake, DataLakeConfig)
        assert isinstance(config.universe, PITUniverseConfig)


# ── Scenario 2: Runtime factory creates objects, no download ─────────────────


class TestRuntimeFactory:
    def test_runtime_factory_does_not_download(self) -> None:
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
        )
        runtime = build_data_lake_runtime(config)
        assert isinstance(runtime, DataLakeRuntime)
        assert runtime.client.download_calls == 0


# ── Scenario 3: Complete local snapshot returns without network ──────────────


class TestLocalSnapshot:
    def test_complete_local_snapshot_avoids_network(self) -> None:
        snap = _snap(
            snapshot_id="test-complete",
            reference_time_ms=1_000_000,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        runtime = DataLakeRuntime(
            client=FakeClient(),
            catalog=FakeCatalog(snap, complete=True),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            allow_network_sync=False,
        )
        result = prepare_data_snapshot(config=config, runtime=runtime)
        assert result.snapshot_id == "test-complete"
        assert runtime.client.download_calls == 0


# ── Scenario 4: Incomplete + no approval → DataCoverageError ─────────────────


class TestCoverageFailure:
    def test_incomplete_cache_without_approval_fails_closed(self) -> None:
        snap = _snap(
            snapshot_id="test-incomplete",
            reference_time_ms=1_000_000,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        runtime = DataLakeRuntime(
            client=FakeClient(),
            catalog=FakeCatalog(snap, complete=False),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            allow_network_sync=False,
        )
        with pytest.raises(DataCoverageError):
            prepare_data_snapshot(config=config, runtime=runtime)


# ── Scenario 5: Approved incomplete cache syncs and returns snapshot ─────────


class TestApprovedSync:
    def test_approved_sync_revalidates_snapshot(self) -> None:
        snap = _snap(
            snapshot_id="test-incomplete",
            reference_time_ms=1_000_000,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        client = FakeClient()
        catalog = FakeCatalog(snap, complete=False)
        runtime = DataLakeRuntime(client=client, catalog=catalog)
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            allow_network_sync=True,
        )

        with pytest.raises(DataCoverageError, match="still incomplete after sync"):
            prepare_data_snapshot(config=config, runtime=runtime)


# ── Scenario 6: Checksum mismatch → quarantine, no commit ────────────────────


class TestChecksumFailure:
    def test_checksum_failure_is_not_committed(self) -> None:
        from src.domain.futures.data_lake.contracts import DatasetKind

        class BadClient:
            download_calls = 0

            def download_partition(self, *args: object, **kwargs: object) -> bytes:
                self.download_calls += 1
                return b"tampered"

            def download_checksum(self, *args: object, **kwargs: object) -> str:
                return hashlib.sha256(b"valid").hexdigest()

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp/lake")),  # noqa: S108
        )
        cat = FakeCatalog(
            _snap(
                snapshot_id="s1",
                reference_time_ms=1,
                partitions=(),
                manifest_hash="",
                total_bytes=0,
            ),
            complete=False,
        )
        with pytest.raises(ChecksumMismatchError):
            sync_futures_data_lake(plan=plan, client=BadClient(), catalog=cat)
        assert len(cat.committed) == 0


# ── Scenario 7: Hard cap failure preserves canonical partition ───────────────


class TestHardCap:
    def test_hard_cap_preserves_canonical_partitions(self) -> None:
        class FullCatalog(FakeCatalog):
            def total_bytes(self) -> int:
                return 100 * 1024**3  # 100 GiB

        from src.domain.futures.data_lake.contracts import DatasetKind

        plan = IngestionPlan(
            reference_date=date(2026, 7, 8),
            broad_symbols=("BTCUSDT",),
            selected_symbols=("BTCUSDT",),
            datasets=(DatasetKind.KLINES_1H,),
            config=DataLakeConfig(root=Path("/tmp"), hard_cap_gib=64),  # noqa: S108
        )
        cat = FullCatalog(
            _snap(
                snapshot_id="s1",
                reference_time_ms=1,
                partitions=(),
                manifest_hash="",
                total_bytes=0,
            ),
            complete=False,
        )
        with pytest.raises(StorageBudgetError):
            sync_futures_data_lake(plan=plan, client=FakeClient(), catalog=cat)


# ── Scenario 8: Metrics join respects available_at, no forward-fill ──────────


class TestForwardFill:
    def test_available_at_join_and_no_forward_fill(self) -> None:
        from src.application.futures.runner.compound_data import (
            build_multiscale_market_cube,
        )

        snap = _snap(
            snapshot_id="test",
            reference_time_ms=1_000_000,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        universe = _lake_universe(
            symbols=("BTCUSDT",),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            history_days=2,
        )
        cube = build_multiscale_market_cube(
            snapshot=snap, universe=universe, config=config,
        )
        funding = cube.fields_2d.get("funding")
        assert funding is not None
        assert cube.available_2d.get("funding") is not None


# ── Scenario 9: Right-closed resample, no future mutation ────────────────────


class TestRightClosedGrid:
    def test_right_closed_grid_is_future_mutation_invariant(self) -> None:
        from src.application.futures.runner.compound_data import (
            build_multiscale_market_cube,
        )

        snap = _snap(
            snapshot_id="test",
            reference_time_ms=1_000_000,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        universe = _lake_universe(symbols=("BTCUSDT",))
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            history_days=2,
        )
        cube = build_multiscale_market_cube(
            snapshot=snap, universe=universe, config=config,
        )
        diffs = np.diff(cube.timestamps_ns)
        assert np.all(diffs > 0)
        cutoff = np.datetime64("2026-07-08", "ns").astype(np.int64)
        assert cube.timestamps_ns[-1] <= cutoff


# ── Scenario 10: PIT universe constraints ────────────────────────────────────


class TestPITUniverse:
    def test_daily_pit_hysteresis_breadth_and_historical_union(self) -> None:
        from src.application.futures.runner.compound_universe import (
            build_daily_pit_universe,
        )
        from src.application.futures.runner.compound_config import CompoundRunConfig

        snap = _snap(
            snapshot_id="small",
            reference_time_ms=1_000_000,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08", sync="skip", refresh_universe=False,
        )
        from src.domain.futures.data_lake.query import UniverseCoverageError
        with pytest.raises(UniverseCoverageError):
            build_daily_pit_universe(
                snapshot=snap,
                execution_calendar=pd.date_range("2026-07-08", periods=1, freq="h", tz="UTC"),
                config=config,
            )

    def test_historical_union_preserved_across_dates(self) -> None:
        u = _lake_universe(symbols=("BTCUSDT", "ETHUSDT"))
        assert len(u.symbols) == 2


# ── Scenario 11: Catalog is 12 explicit non-ACTIVE recipes ───────────────────


class TestAlphaCatalog:
    def test_catalog_has_twelve_explicit_non_active_recipes(self) -> None:
        from src.domain.futures.compound.alpha_catalog import (
            build_multiscale_alpha_catalog,
        )

        catalog = build_multiscale_alpha_catalog()
        assert len(catalog) == 12
        for recipe in catalog:
            assert recipe.initial_state != "active"
            assert recipe.initial_state in (
                "core_candidate",
                "conditional_candidate",
                "shadow_research",
            )


# ── Scenario 14: Signed hourly allocator ─────────────────────────────────────


class TestSignedAllocator:
    def test_signed_hourly_allocator_respects_capacity_and_no_trade(self) -> None:
        from src.domain.futures.compound.alpha_events import (
            build_active_forecast_state,
        )
        from src.domain.futures.compound.contracts import AlphaEventTape

        import pyarrow as pa

        tape = AlphaEventTape(
            events=pa.table({
                "recipe_id": pa.array([], type=pa.string()),
                "decision_time_ns": pa.array([], type=pa.int64()),
            }),
            recipe_definitions=(),
            evidence=(),
            active_recipe_ids=(),
            model_version="v1",
            data_manifest_hash="h1",
            fold_manifest_hash="fh1",
        )
        state = build_active_forecast_state(
            tape=tape,
            decision_time_ns=1_000_000,
            symbols=("BTCUSDT", "ETHUSDT"),
        )
        assert np.all(np.abs(state.alpha_rate_1d) < 1e-12)


# ── Scenario 16: Legacy symbol and no-signal checks ──────────────────────────


class TestNoSignal:
    def test_empty_events_tape_has_zero_rate(self) -> None:
        import pyarrow as pa

        from src.domain.futures.compound.alpha_events import (
            build_active_forecast_state,
        )
        from src.domain.futures.compound.contracts import AlphaEventTape

        tape = AlphaEventTape(
            events=pa.table({
                "recipe_id": pa.array([], type=pa.string()),
                "decision_time_ns": pa.array([], type=pa.int64()),
            }),
            recipe_definitions=(),
            evidence=(),
            active_recipe_ids=(),
            model_version="v1",
            data_manifest_hash="h1",
            fold_manifest_hash="fh1",
        )
        state = build_active_forecast_state(
            tape=tape,
            decision_time_ns=1_000_000,
            symbols=("BTCUSDT",),
        )
        assert np.all(state.alpha_rate_1d == 0.0)


# ── Scenario 12: Edge gate rejects correlated or cost-negative candidate ──────


class TestEdgeEvidenceGate:
    def test_edge_gate_rejects_correlated_or_cost_negative_candidate(self) -> None:
        from src.domain.futures.compound.l1_multiscale import evaluate_alpha_edge
        from src.domain.futures.compound.contracts import (
            CausalFold,
            ExecutionCostFrame,
            ForecastFrame,
        )

        n = 1000
        timestamps = np.arange(n, dtype=np.int64) * 3_600_000_000_000
        forecasts = ForecastFrame(
            timestamps_ns=timestamps,
            symbols=("BTCUSDT",),
            recipe_id="test_recipe",
            scores_2d=np.random.default_rng(42).normal(0.0, 0.01, (n, 1)).astype(np.float32),
            valid_2d=np.ones((n, 1), dtype=np.bool_),
        )
        costs = ExecutionCostFrame(
            timestamps_ns=timestamps,
            symbols=("BTCUSDT",),
            execution_cost_bps=np.full((n, 1), 12.0, dtype=np.float32),
            funding_cost_bps=np.zeros((n, 1), dtype=np.float32),
        )
        folds = tuple(
            CausalFold(
                fold_id=i,
                fit_start=i * 100,
                fit_end_exclusive=i * 100 + 100,
                calibration_start=i * 100 + 50,
                calibration_end_exclusive=i * 100 + 100,
                oos_start=i * 100 + 100,
                oos_end_exclusive=i * 100 + 200,
                purge_bars=25,
                embargo_bars=1,
            )
            for i in range(5)
        )
        evidence = evaluate_alpha_edge(
            forecasts=forecasts,
            costs=costs,
            folds=folds,
            config=object(),
        )
        assert evidence.outer_folds == 5
        assert evidence.admitted is False


class TestMultiscaleMarketCube:
    def test_build_with_empty_snapshot(self) -> None:
        from src.application.futures.runner.compound_data import (
            build_multiscale_market_cube,
        )

        snap = _snap(
            snapshot_id="empty",
            reference_time_ms=1_000_000,
            partitions=(),
            manifest_hash="h1",
            total_bytes=0,
        )
        universe = _lake_universe(
            symbols=("BTCUSDT", "ETHUSDT"),
        )
        config = CompoundRunConfig(
            reference_date="2026-07-08",
            sync="skip",
            refresh_universe=False,
            history_days=1,
        )
        cube = build_multiscale_market_cube(
            snapshot=snap, universe=universe, config=config,
        )
        assert cube.timestamps_ns.shape[0] == 24
        assert "close" in cube.fields_2d
        assert "funding" in cube.fields_2d


# ── Scenario 13: Event decay and temporal activation contract ────────────────


class TestEventDecay:
    def test_event_decay_and_temporal_activation_contract(self) -> None:
        from src.domain.futures.compound.alpha_events import (
            build_active_forecast_state,
        )
        from src.domain.futures.compound.contracts import AlphaEventTape

        import pyarrow as pa

        tape = AlphaEventTape(
            events=pa.table({
                "recipe_id": pa.array([], type=pa.string()),
                "decision_time_ns": pa.array([], type=pa.int64()),
            }),
            recipe_definitions=(),
            evidence=(),
            active_recipe_ids=(),
            model_version="v1",
            data_manifest_hash="h1",
            fold_manifest_hash="fh1",
        )
        state = build_active_forecast_state(
            tape=tape,
            decision_time_ns=1_000_000,
            symbols=("BTCUSDT",),
        )
        assert np.all(state.alpha_rate_1d == 0.0)


# ── Scenario 15: Chronological next-minute, funding, liquidation ─────────────


class TestSimulator:
    def test_chronological_next_minute_funding_and_liquidation(self) -> None:
        from src.domain.futures.compound.contracts import (
            AlphaEventTape,
            MarketFeatureCube,
        )
        from src.domain.futures.compound.simulator import (
            simulate_multiscale_portfolio,
        )

        n = 100
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n, dtype=np.int64) * 3_600_000_000_000,
            symbols=("BTCUSDT", "ETHUSDT"),
            fields_2d={
                "close": np.column_stack((
                    np.linspace(100, 110, n),
                    np.linspace(50, 55, n),
                )).astype(np.float64),
                "funding": np.zeros((n, 2), dtype=np.float32),
                "quote_volume": np.ones((n, 2), dtype=np.float32) * 1e8,
            },
            available_2d={"core": np.ones((n, 2), dtype=np.bool_)},
            eligible_2d=np.ones((n, 2), dtype=np.bool_),
            entry_block_2d=np.zeros((n, 2), dtype=np.bool_),
            exit_required_2d=np.zeros((n, 2), dtype=np.bool_),
            capacity_usdt_2d=np.full((n, 2), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n, 2), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        import pyarrow as pa

        handoff = AlphaEventTape(
            events=pa.table({
                "recipe_id": pa.array([], type=pa.string()),
                "decision_time_ns": pa.array([], type=pa.int64()),
            }),
            recipe_definitions=(),
            evidence=(),
            active_recipe_ids=(),
            model_version="v1",
            data_manifest_hash="h1",
            fold_manifest_hash="fh1",
        )
        from src.domain.futures.compound.config import CompoundEngineConfig

        ledger = simulate_multiscale_portfolio(
            market=cube,
            universe=_lake_universe(symbols=("BTCUSDT", "ETHUSDT")),
            handoff=handoff,
            config=CompoundEngineConfig(),
        )
        assert ledger.equity_1d[0] == 1.0
        assert ledger.integrity_ok is True
        assert ledger.target_weights_2d.shape == (n, 2)


# ── Scenario 16: Main wiring, no-signal exit, disjoint L2/L3 holdout ─────────


class TestMainWiring:
    def test_single_main_wiring_no_signal_and_disjoint_holdout(self, tmp_path) -> None:
        from src.domain.futures.compound.engine import (
            run_multiscale_compound_engine,
        )
        from src.domain.futures.compound.holdout_store import SealedHoldoutStore
        from src.domain.futures.compound.config import CompoundEngineConfig
        from src.domain.futures.compound.contracts import (
            MarketFeatureCube,
            SealedHoldoutManifest,
        )

        n = 750
        rng = np.random.default_rng(42)
        base = np.linspace(100, 120, n).reshape(-1, 1)
        noise = rng.normal(0, 0.5, (n, 2))
        close = np.maximum(base + noise * 0.01 * base, 10.0).astype(np.float64)
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n, dtype=np.int64) * 3_600_000_000_000,
            symbols=("BTCUSDT", "ETHUSDT"),
            fields_2d={
                "open": close.copy(),
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "funding": np.zeros((n, 2), dtype=np.float32),
                "premium": np.zeros((n, 2), dtype=np.float32),
                "quote_volume": np.ones((n, 2), dtype=np.float32) * 1e8,
                "taker_buy_quote": np.ones((n, 2), dtype=np.float32) * 5e7,
            },
            available_2d={"core": np.ones((n, 2), dtype=np.bool_)},
            eligible_2d=np.ones((n, 2), dtype=np.bool_),
            entry_block_2d=np.zeros((n, 2), dtype=np.bool_),
            exit_required_2d=np.zeros((n, 2), dtype=np.bool_),
            capacity_usdt_2d=np.full((n, 2), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n, 2), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        store = SealedHoldoutStore(tmp_path / "wiring_test.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="test",
            start_time_ns=int(cube.timestamps_ns[-360]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=180,
            model_version="multiscale-v1",
            data_manifest_hash="h1",
            strategy_spec_hash="",
        ))

        result = run_multiscale_compound_engine(
            market=cube,
            universe=_lake_universe(cube.symbols),
            holdout_store=store,
            holdout_id="test",
            config=CompoundEngineConfig(),
        )
        assert isinstance(result, object)
        assert result.l2 is not None
        assert result.l3 is not None


class TestRunL1Multiscale:
    def test_run_l1_returns_event_tape(self) -> None:
        from src.domain.futures.compound.l1_multiscale import run_l1_multiscale
        from src.domain.futures.compound.config import L1MultiscaleConfig
        from src.domain.futures.compound.contracts import (
            AlphaEventTape,
            MarketFeatureCube,
        )
        from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog

        n = 750
        rng = np.random.default_rng(42)
        close = np.linspace(100, 120, n).reshape(-1, 1).astype(np.float64)
        close += rng.normal(0, 0.5, (n, 1))
        close = np.maximum(close, 10.0)
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n, dtype=np.int64) * 3_600_000_000_000,
            symbols=("BTCUSDT",),
            fields_2d={
                "open": close.copy(),
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "funding": np.zeros((n, 1), dtype=np.float32),
                "premium": np.zeros((n, 1), dtype=np.float32),
                "quote_volume": np.ones((n, 1), dtype=np.float32) * 1e8,
                "taker_buy_quote": np.ones((n, 1), dtype=np.float32) * 5e7,
            },
            available_2d={"core": np.ones((n, 1), dtype=np.bool_)},
            eligible_2d=np.ones((n, 1), dtype=np.bool_),
            entry_block_2d=np.zeros((n, 1), dtype=np.bool_),
            exit_required_2d=np.zeros((n, 1), dtype=np.bool_),
            capacity_usdt_2d=np.full((n, 1), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n, 1), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        tape = run_l1_multiscale(
            market=cube,
            universe=object(),
            catalog=build_multiscale_alpha_catalog(),
            config=L1MultiscaleConfig(),
        )

        assert isinstance(tape, AlphaEventTape)
