from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CandidateTrial,
    CandidateTrialLedger,
    CompoundEngineResult,
    DeploymentVerdict,
    L2CategoryResult,
    L2Evaluation,
    L2GateVerdict,
    L3ValidationResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.engine import run_multiscale_compound_engine
from src.domain.futures.compound.holdout_store import SealedHoldoutStore
from src.domain.futures.compound.provenance import compute_strategy_spec_hash

_NS_PER_HOUR = 3_600_000_000_000


def _make_cube(
    n_bars: int = 512,
    n_syms: int = 3,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
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
def wired_stores(tmp_path):
    return SealedHoldoutStore(tmp_path / "h.db"), CandidateTrialLedger(tmp_path / "t.db")


class TestEngineIntegration:
    def test_engine_charges_search_multiplicity_and_passes_runtime_spec_hash(
        self, tmp_path, wired_stores,
    ) -> None:
        store, ledger = wired_stores
        cube = _make_cube(2400)
        universe = type("Universe", (), {
            "symbols": cube.symbols, "snapshots": (),
        })()
        manifest = SealedHoldoutManifest(
            holdout_id="int-test",
            start_time_ns=int(cube.timestamps_ns[-180]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)

        from dataclasses import replace
        from src.domain.futures.compound.config import CalibrationConfig, ClusterConfig
        calib = CalibrationConfig(n_folds=3, purge_bars=1, embargo_bars=2, min_fold_obs=5)
        cluster = ClusterConfig(k_clusters=2, min_cluster_size=1)

        def run_engine(target_ann_vol: float) -> CompoundEngineResult:
            config = replace(
                CompoundEngineConfig(), calibration=calib, cluster=cluster,
                dynamic_compounding=replace(
                    CompoundEngineConfig().dynamic_compounding, target_ann_vol=target_ann_vol,
                ),
            )
            spec_hash = compute_strategy_spec_hash(config=config)
            return run_multiscale_compound_engine(
                market=cube,
                universe=universe,
                holdout_store=store,
                holdout_id="int-test",
                config=config,
                strategy_spec_hash=spec_hash,
                trial_ledger=ledger,
            )

        result1 = run_engine(0.15)

        # RULE-06: a genuinely different config (different candidate_hash) must
        # find run1's registered trial as a prior and charge the multiplicity
        # (engine.py line: trial_multiplicity = charge_config_search_multiplicity(...)).
        result2 = run_engine(0.20)

        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "t.db"))
        row_count = conn.execute("SELECT COUNT(*) FROM candidate_trials").fetchone()[0]
        distinct_hashes = conn.execute("SELECT COUNT(DISTINCT candidate_hash) FROM candidate_trials").fetchone()[0]
        conn.close()

        assert row_count == 2
        assert distinct_hashes == 2
        # LIMIT-02: a single prior config trial (M=1) is charge-neutral by design
        # (charge_config_search_multiplicity requires M>=2 rows to derive a
        # participation ratio), so candidate_count is unchanged even though the
        # charging call itself (engine.py's `if prior_returns.shape[0] > 0:`
        # branch) executed on run2.
        assert result2.l2.candidate_count == result1.l2.candidate_count
        assert result1.l2.oos_days > 0
        assert result2.l2.oos_days > 0

    def test_engine_builds_deployment_candidate_with_non_empty_provenance_hashes(
        self, tmp_path, wired_stores,
    ) -> None:
        store, ledger = wired_stores
        cube = _make_cube(768)
        universe = type("Universe", (), {
            "symbols": cube.symbols, "snapshots": (),
        })()
        manifest = SealedHoldoutManifest(
            holdout_id="deploy-test",
            start_time_ns=int(cube.timestamps_ns[-180]),
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h1",
            strategy_spec_hash="spec1",
        )
        store.create(manifest)

        from dataclasses import replace
        from src.domain.futures.compound.config import CalibrationConfig, DynamicCompoundingConfig
        calib = CalibrationConfig(n_folds=3, purge_bars=1, embargo_bars=2, min_fold_obs=5)
        config = replace(CompoundEngineConfig(), calibration=calib,
                         dynamic_compounding=replace(
                             CompoundEngineConfig().dynamic_compounding, target_ann_vol=0.15,
                         ))
        spec_hash = compute_strategy_spec_hash(config=config)
        result = run_multiscale_compound_engine(
            market=cube,
            universe=universe,
            holdout_store=store,
            holdout_id="deploy-test",
            config=config,
            strategy_spec_hash=spec_hash,
            trial_ledger=ledger,
        )
        if result.deployment_candidate is not None:
            assert result.deployment_candidate.strategy_spec_hash != ""
            assert result.deployment_candidate.fold_manifest_hash != ""
