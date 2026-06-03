from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain.futures.universe.models import UniverseRunManifest
from src.domain.futures.universe.store import (
    build_decision_frame,
    compute_universe_run_id,
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
    loaded_manifest, loaded_decisions, loaded_report = loaded
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
    assert snapshot.stage5_research_panel == ("BTCUSDT", "ETHUSDT")
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
