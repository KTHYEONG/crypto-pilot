from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.futures.universe.contracts import UniverseStateCube
from src.domain.futures.universe.models import UniverseRunManifest
from src.domain.futures.universe.store import (
    _cube_from_df,
    _cube_to_df,
    build_decision_frame,
    compute_universe_run_id,
    gc_stale_store_runs,
    load_universe_store_run,
    materialize_snapshot_from_store,
    write_universe_store_run,
)


def _manifest() -> UniverseRunManifest:
    return UniverseRunManifest(
        as_of="2025-01-01",
        tf="4h",
        schema_version=1,
        run_id=compute_universe_run_id(
            as_of="2025-01-01",
            tf="4h",
            config_hash="cfg",
            data_manifest_hash="manifest",
        ),
        config_hash="cfg",
        data_manifest_hash="manifest",
        generated_at_utc="2025-01-01T00:00:00+00:00",
        ledger_confidence="high",
        basket_ref=("BTCUSDT",),
        basket_weights=(1.0,),
        n_stage0=3,
        n_stage1_pass=3,
        n_stage2_pass=3,
        n_stage3_pass=2,
        n_stage4_pass=2,
        n_stage5_pass=2,
        n_stage6_selected=1,
    )


def test_store_roundtrip_preserves_stage5_and_stage6_rows(tmp_path: Path) -> None:
    manifest = _manifest()
    stage5 = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "role": "anchor",
                "rank": 1,
                "tradeable_score": 0.9,
                "execution_pool_score": 0.8,
                "adv_usdt_median": 100_000_000.0,
                "execution_cost_bps": 8.0,
                "funding_rate_8h": 0.001,
                "beta_vs_market": 1.2,
                "cluster_id": 4,
                "cluster_size": 6.0,
                "anchor_cluster_member": 1.0,
                "basis_annualized_mean": 0.01,
                "basis_vol": 0.02,
                "capacity_clip_usdt_list": (1000.0,),
            },
            {
                "symbol": "ETHUSDT",
                "role": "regular",
                "rank": 2,
                "tradeable_score": 0.7,
                "execution_pool_score": 0.6,
                "adv_usdt_median": 80_000_000.0,
                "execution_cost_bps": 10.0,
                "funding_rate_8h": 0.0,
                "beta_vs_market": 0.9,
                "cluster_id": 4,
                "cluster_size": 6.0,
                "anchor_cluster_member": 1.0,
                "basis_annualized_mean": None,
                "basis_vol": None,
                "capacity_clip_usdt_list": (900.0,),
            },
        ]
    )
    stage6 = stage5.iloc[[0]].copy()
    report = pd.DataFrame(
        [
            {"symbol": "BTCUSDT", "stage": "stage6_selection", "passed": True, "reason": "selected"},
            {"symbol": "ETHUSDT", "stage": "stage6_selection", "passed": False, "reason": "not_selected"},
        ]
    )

    decisions = build_decision_frame(
        manifest=manifest,
        stage5_frame=stage5,
        stage6_frame=stage6,
        report=report,
    )
    write_universe_store_run(
        manifest=manifest,
        decisions=decisions,
        report=report,
        root=tmp_path,
    )

    loaded = load_universe_store_run(
        as_of="2025-01-01",
        tf="4h",
        config_hash="cfg",
        data_manifest_hash="manifest",
        root=tmp_path,
    )
    assert loaded is not None
    loaded_manifest, loaded_decisions, loaded_report, cube = loaded
    assert cube is None  # no snapshot with cube was passed
    snapshot, selected_frame, _ = materialize_snapshot_from_store(
        manifest=loaded_manifest,
        decisions=loaded_decisions,
        report=loaded_report,
    )

    assert tuple(loaded_decisions.loc[loaded_decisions["stage5_pass"], "symbol"]) == (
        "BTCUSDT",
        "ETHUSDT",
    )
    assert tuple(selected_frame["symbol"].tolist()) == ("BTCUSDT",)
    assert snapshot.selected[0].cluster_size == 6.0
    assert snapshot.selected[0].anchor_cluster_member == 1.0


def test_load_universe_store_run_requires_exact_hash_match(tmp_path: Path) -> None:
    manifest = _manifest()
    decisions = pd.DataFrame(columns=[
        "as_of",
        "tf",
        "run_id",
        "config_hash",
        "data_manifest_hash",
        "symbol",
        "stage5_pass",
        "stage6_selected",
        "stage",
        "selection_reason",
        "role",
        "rank",
        "tradeable_score",
        "execution_pool_score",
        "adv_usdt_median",
        "execution_cost_bps",
        "funding_rate_8h",
        "beta_vs_market",
        "cluster_id",
        "cluster_size",
        "anchor_cluster_member",
        "basis_annualized_mean",
        "basis_vol",
        "capacity_clip_usdt_list",
        "reject_code",
        "final_rank",
        "generated_at_utc",
    ])
    report = pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])
    write_universe_store_run(manifest=manifest, decisions=decisions, report=report, root=tmp_path)

    assert (
        load_universe_store_run(
            as_of="2025-01-01",
            tf="4h",
            config_hash="different",
            data_manifest_hash="manifest",
            root=tmp_path,
        )
        is None
    )


def _make_cube(n_bar: int = 10, n_inst: int = 5) -> UniverseStateCube:
    calendar = pd.date_range("2024-01-01", periods=n_bar, freq="h", tz="UTC")
    instrument_ids = tuple(f"binance_usdt_perpetual:SYM{i}" for i in range(n_inst))
    rng = np.random.default_rng(42)
    return UniverseStateCube(
        calendar=calendar,
        instrument_ids=instrument_ids,
        eligible=rng.random((n_bar, n_inst)) > 0.5,
        entry_block=rng.random((n_bar, n_inst)) > 0.5,
        exit_required=rng.random((n_bar, n_inst)) > 0.5,
        capacity_usdt=rng.random((n_bar, n_inst)) * 10_000_000.0,
        risk_scale=rng.random((n_bar, n_inst)) * 2.0,
        cost_bps=rng.random((n_bar, n_inst)) * 50.0,
    )


def test_cube_serialization_roundtrip(tmp_path: Path) -> None:
    cube = _make_cube()
    df = _cube_to_df(cube)
    assert not df.empty
    assert df.shape[0] == 1  # single row

    restored = _cube_from_df(df)
    assert restored.instrument_ids == cube.instrument_ids
    assert np.array_equal(restored.eligible, cube.eligible)
    assert np.array_equal(restored.entry_block, cube.entry_block)
    assert np.array_equal(restored.exit_required, cube.exit_required)
    assert np.array_equal(restored.capacity_usdt, cube.capacity_usdt)
    assert np.array_equal(restored.risk_scale, cube.risk_scale)
    assert np.array_equal(restored.cost_bps, cube.cost_bps)
    assert restored.calendar.equals(cube.calendar)


def test_cube_persistence_through_store_run(tmp_path: Path) -> None:
    """Cube.parquet written via snapshot param, loaded via load_universe_store_run."""
    manifest = _manifest()
    cube = _make_cube()
    stage5 = pd.DataFrame([{
        "symbol": "BTCUSDT", "role": "anchor", "rank": 1,
        "tradeable_score": 0.9, "execution_pool_score": 0.8,
        "adv_usdt_median": 100_000_000.0, "execution_cost_bps": 8.0,
        "funding_rate_8h": 0.001, "beta_vs_market": 1.2,
        "cluster_id": 4, "cluster_size": 6.0, "anchor_cluster_member": 1.0,
        "basis_annualized_mean": 0.01, "basis_vol": 0.02,
        "capacity_clip_usdt_list": (1000.0,),
    }])
    decisions = build_decision_frame(
        manifest=manifest,
        stage5_frame=stage5,
        stage6_frame=stage5,
        report=pd.DataFrame([{"symbol": "BTCUSDT", "stage": "stage6_selection", "passed": True, "reason": "selected"}]),
    )
    report = pd.DataFrame([{"symbol": "BTCUSDT", "stage": "stage6_selection", "passed": True, "reason": "selected"}])
    snapshot = materialize_snapshot_from_store(
        manifest=manifest, decisions=decisions, report=report, cube=cube,
    )[0]
    assert snapshot.pit_state_cube is not None

    write_universe_store_run(
        manifest=manifest,
        decisions=decisions,
        report=report,
        snapshot=snapshot,
        root=tmp_path,
    )
    loaded = load_universe_store_run(
        as_of="2025-01-01",
        tf="4h",
        config_hash="cfg",
        data_manifest_hash="manifest",
        root=tmp_path,
    )
    assert loaded is not None
    _, _, _, loaded_cube = loaded
    assert loaded_cube is not None
    assert loaded_cube.instrument_ids == cube.instrument_ids
    assert np.array_equal(loaded_cube.eligible, cube.eligible)
    assert loaded_cube.calendar.equals(cube.calendar)


def test_gc_stale_store_runs_removes_old_dirs(tmp_path: Path) -> None:
    decisions = pd.DataFrame(columns=[
        "as_of", "tf", "run_id", "config_hash", "data_manifest_hash",
        "symbol", "stage5_pass", "stage6_selected", "stage", "selection_reason",
        "role", "rank", "tradeable_score", "execution_pool_score",
        "adv_usdt_median", "execution_cost_bps", "funding_rate_8h",
        "beta_vs_market", "cluster_id", "cluster_size", "anchor_cluster_member",
        "basis_annualized_mean", "basis_vol", "capacity_clip_usdt_list",
        "reject_code", "final_rank", "generated_at_utc",
    ])
    report = pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])

    for suffix in ("a", "b", "c"):
        m = UniverseRunManifest(
            as_of="2025-01-01",
            tf="4h",
            schema_version=1,
            run_id=compute_universe_run_id(
                as_of="2025-01-01",
                tf="4h",
                config_hash=suffix,
                data_manifest_hash="m",
            ),
            config_hash=suffix,
            data_manifest_hash="m",
            generated_at_utc="2025-01-01T00:00:00+00:00",
            ledger_confidence="high",
            basket_ref=(),
            basket_weights=(),
            n_stage0=0,
            n_stage1_pass=0,
            n_stage2_pass=0,
            n_stage3_pass=0,
            n_stage4_pass=0,
            n_stage5_pass=0,
            n_stage6_selected=0,
        )
        write_universe_store_run(
            manifest=m, decisions=decisions, report=report, root=tmp_path,
        )
        time.sleep(0.01)

    deleted = gc_stale_store_runs(tf="4h", as_of=date(2025, 1, 1), root=tmp_path)
    assert deleted == 2

    remaining = gc_stale_store_runs(tf="4h", as_of=date(2025, 1, 1), root=tmp_path)
    assert remaining == 0
