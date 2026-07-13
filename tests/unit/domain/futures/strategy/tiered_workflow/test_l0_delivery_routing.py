from __future__ import annotations

import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    L0DeliveryContractError,
    L0StrategyDeliveryManifest,
    L0TfDeliveryRoute,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import select_l1_delivery_events


def _manifest(*ids: str) -> L0StrategyDeliveryManifest:
    end_ns = int(pd.Timestamp("2026-01-02T00:00").value)
    route = L0TfDeliveryRoute(
        timeframe="4h",
        selected_recipe_ids=tuple(ids),
        allocated_budget_units=len(ids),
        evidence_end_ns=end_ns,
    )
    return L0StrategyDeliveryManifest(
        run_id_prefix="test",
        reports_by_tf={},
        independence_audit=None,
        final_selected_recipe_ids=tuple(ids),
        total_l1_verification_budget=max(len(ids), 1),
        routes=(route,),
    )


def test_exact_recipe_route_does_not_expand_by_family() -> None:
    events = pd.DataFrame(
        {
            "native_tf": ["4h", "4h", "12h"],
            "l0_recipe_id": ["r1", "r2", "r1"],
            "family": ["trend_donchian"] * 3,
            "variant": ["lb20"] * 3,
            "entry_idx": [10, 11, 12],
        }
    )

    selected = select_l1_delivery_events(
        labeled_events=events,
        tf="4h",
        manifest=_manifest("r1"),
    )

    assert selected[["native_tf", "l0_recipe_id"]].to_dict("records") == [
        {"native_tf": "4h", "l0_recipe_id": "r1"}
    ]


def test_gate_manifest_missing_recipe_identity_fails_closed() -> None:
    events = pd.DataFrame({"native_tf": ["4h"], "family": ["trend_donchian"]})

    with pytest.raises(L0DeliveryContractError, match="l0_recipe_id"):
        select_l1_delivery_events(
            labeled_events=events,
            tf="4h",
            manifest=_manifest("r1"),
        )


def test_gate_manifest_empty_events_without_identity_returns_empty() -> None:
    selected = select_l1_delivery_events(
        labeled_events=pd.DataFrame(),
        tf="4h",
        manifest=_manifest("r1"),
    )

    assert selected.empty


def test_none_manifest_preserves_legacy_native_tf_filter() -> None:
    events = pd.DataFrame(
        {"native_tf": ["4h", "12h"], "entry_idx": [1, 2]}
    )

    selected = select_l1_delivery_events(
        labeled_events=events,
        tf="4h",
        manifest=None,
    )

    pd.testing.assert_frame_equal(selected.reset_index(drop=True), events.iloc[[0]].reset_index(drop=True))


def test_budget_overflow_raises_contract_error() -> None:
    end_ns = int(pd.Timestamp("2026-01-02T00:00").value)
    route = L0TfDeliveryRoute(
        timeframe="4h",
        selected_recipe_ids=("r1", "r2"),
        allocated_budget_units=10,
        evidence_end_ns=end_ns,
    )
    manifest = L0StrategyDeliveryManifest(
        run_id_prefix="test",
        reports_by_tf={},
        independence_audit=None,
        final_selected_recipe_ids=("r1", "r2"),
        total_l1_verification_budget=5,
        routes=(route,),
    )

    with pytest.raises(L0DeliveryContractError, match="budget"):
        select_l1_delivery_events(
            labeled_events=pd.DataFrame({"native_tf": ["4h"], "l0_recipe_id": ["r1"], "entry_idx": [1]}),
            tf="4h",
            manifest=manifest,
        )


def test_probe_zero_winners_does_not_remove_route() -> None:
    events = pd.DataFrame(
        {
            "native_tf": ["1d", "1d"],
            "l0_recipe_id": ["r1", "r2"],
            "entry_idx": [1, 2],
        }
    )
    manifest = _manifest("r1", "r2")
    manifest_r1 = L0TfDeliveryRoute(
        timeframe="1d",
        selected_recipe_ids=("r1", "r2"),
        allocated_budget_units=2,
        evidence_end_ns=int(pd.Timestamp("2026-01-02T00:00").value),
    )
    manifest = L0StrategyDeliveryManifest(
        run_id_prefix="test",
        reports_by_tf={},
        independence_audit=None,
        final_selected_recipe_ids=("r1", "r2"),
        total_l1_verification_budget=2,
        routes=(manifest_r1,),
    )
    selected = select_l1_delivery_events(
        labeled_events=events,
        tf="1d",
        manifest=manifest,
    )
    assert len(selected) == 2
