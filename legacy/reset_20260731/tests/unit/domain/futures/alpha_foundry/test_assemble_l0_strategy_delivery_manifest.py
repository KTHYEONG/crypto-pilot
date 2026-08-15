from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.domain.futures.alpha_foundry.bridge_helpers import (
    AlphaFoundryL0Result,
    assemble_l0_strategy_delivery_manifest,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaArchetype,
    CrossBucketDiversityResult,
    CrossTFPairEvidence,
    L0IndependenceAudit,
    L0SignalCandidate,
)


def _candidate(recipe_id: str, archetype: AlphaArchetype, timeframe: str, priority: float) -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test",
        timeframe=timeframe,
        family="btc_regime_pullback",
        variant="v",
        recipe_id=recipe_id,
        archetype=archetype,
        source="synthetic_recipe",
        n_events=100,
        effective_n=50.0,
        mean_net_bps=priority,
        block_lcb_bps=priority * 0.5,
        nw_tstat=1.5,
        bootstrap_lcb_bps=0.0,
        bootstrap_agree=True,
        cost_drag_ratio=0.3,
        turnover_per_year=50.0,
        max_abs_corr_in_bucket=0.0,
        tf_coverage_count=0,
        sign_agreement_ratio=0.0,
        corroboration_tier="single_tf_strict",
        discovery_tier="candidate",
        l1_priority_score=priority,
        l1_budget_units=1,
        hard_reject_reasons=(),
        soft_flags=(),
    )


def _real_aligned() -> Any:
    class _Aligned:
        datetimes = np.array([0, 1, 2], dtype="datetime64[ns]")
        active_mask = np.ones((3, 1), dtype=np.bool_)
        warm_mask = np.ones((3, 1), dtype=np.bool_)
        entry_block_mask = np.zeros((3, 1), dtype=np.bool_)
        kill_mask = np.zeros((3, 1), dtype=np.bool_)

    return _Aligned()


def _mock_panel(recipe_id: str) -> Any:
    class _Panel:
        metadata: dict[str, str]
        signed_score_2d: np.ndarray
        valid_mask_2d: np.ndarray

        def __init__(self) -> None:
            self.metadata = {"recipe_id": recipe_id}
            self.datetimes = np.array(["2026-01-01T00:00:00", "2026-01-02T00:00:00"], dtype="datetime64[ns]")
            self.signed_score_2d = np.zeros((2, 1), dtype=np.float64)
            self.valid_mask_2d = np.ones((2, 1), dtype=bool)

    return _Panel()


FIXED_AUDIT = L0IndependenceAudit(
    n_selected_total=2,
    n_distinct_thesis_ids=1,
    n_independent_clusters=1,
    cluster_members={0: ("r1",), 1: ("r2",)},
    demoted_recipe_ids=(),
    demoted_reason_by_id={},
    canonical_tf="1h",
    max_corr_threshold=0.70,
)

FIXED_CROSS_RESULT = CrossBucketDiversityResult(
    final_selected_recipe_ids=("r1",),
    demoted_recipe_ids=("r2",),
    demoted_reason_by_id={"r2": "r1"},
    cross_bucket_corr=np.array([[1.0, 0.9], [0.9, 1.0]]),
    global_eff_test_count=1.2,
    pair_evidence=(
        CrossTFPairEvidence(
            recipe_id_a="r1",
            recipe_id_b="r2",
            score_corr=0.9,
            shared_directional_entries=15,
            directional_entry_jaccard=0.6,
            is_redundant=True,
        ),
    ),
    canonical_tf="1h",
    common_start_ns=1,
    common_end_ns=100,
    n_common_active_bars=480,
)


def _base_multi_results() -> dict[str, AlphaFoundryL0Result]:
    """Two candidates from same TF to avoid TF floor re-admission."""
    c1 = _candidate("r1", "trend", "4h", priority=10.0)
    c2 = _candidate("r2", "trend", "4h", priority=5.0)
    return {
        "4h": AlphaFoundryL0Result(
            panels_for_l1=(_mock_panel("r1"), _mock_panel("r2")),
            summary_report=None,
            gate_results=(),
            panel_bindings=(),
            candidates_for_l1=(c1, c2),
        ),
    }


def _base_aligned() -> dict[str, Any]:
    return {"4h": _real_aligned()}


class TestAssembleL0StrategyDeliveryManifest:
    def test_passthrough_when_both_disabled(self) -> None:
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        pruned, manifest = assemble_l0_strategy_delivery_manifest(
            multi_results=multi_results,
            aligned_by_tf=aligned_by_tf,
            run_id_prefix="test",
            enable_audit=False,
            enable_pruning=False,
        )

        assert manifest.independence_audit is None
        assert dict(pruned) == multi_results

    def test_audit_only_populates_audit_and_returns_unpruned(self) -> None:
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with (
            patch("src.domain.futures.alpha_foundry.diversity.audit_l0_selected_recipe_independence") as mock_audit,
            patch("src.domain.futures.alpha_foundry.diversity.resolve_cross_tf_shared_context") as mock_ctx,
        ):
            mock_audit.return_value = FIXED_AUDIT
            mock_ctx.return_value = MagicMock()

            pruned, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test",
                enable_audit=True,
                enable_pruning=False,
            )

            assert manifest.independence_audit is FIXED_AUDIT
            mock_audit.assert_called_once()
            mock_ctx.assert_called_once()
        assert dict(pruned) == multi_results

    def test_pruning_enabled_returns_pruned_results(self) -> None:
        """With both candidates from same TF, floor doesn't re-admit r2.
        Assert manifest.final_selected_recipe_ids reflects the pruned set."""
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with (
            patch("src.domain.futures.alpha_foundry.diversity.audit_l0_selected_recipe_independence") as mock_audit,
            patch("src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy") as mock_compute,
            patch("src.domain.futures.alpha_foundry.diversity.resolve_cross_tf_shared_context") as mock_ctx,
        ):
            mock_audit.return_value = FIXED_AUDIT
            mock_compute.return_value = FIXED_CROSS_RESULT
            mock_ctx.return_value = MagicMock()

            pruned, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test",
                enable_audit=True,
                enable_pruning=True,
                max_novelty_corr=0.70,
            )

        assert manifest.independence_audit is FIXED_AUDIT
        assert manifest.pruning_status == "applied"
        assert "r1" in manifest.final_selected_recipe_ids
        assert len(pruned["4h"].candidates_for_l1) == 1
        assert pruned["4h"].candidates_for_l1[0].recipe_id == "r1"

    def test_pruning_enabled_audit_disabled_returns_pruned_no_audit(self) -> None:
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with (
            patch("src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy") as mock_compute,
            patch("src.domain.futures.alpha_foundry.diversity.resolve_cross_tf_shared_context") as mock_ctx,
        ):
            mock_compute.return_value = FIXED_CROSS_RESULT
            mock_ctx.return_value = MagicMock()

            _, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test",
                enable_audit=False,
                enable_pruning=True,
            )

        assert manifest.independence_audit is None
        assert manifest.pruning_status == "applied"
        assert "r1" in manifest.final_selected_recipe_ids
        assert "r2" not in manifest.final_selected_recipe_ids

    def test_value_error_fallback_returns_unpruned(self) -> None:
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with (
            patch("src.domain.futures.alpha_foundry.diversity.audit_l0_selected_recipe_independence") as mock_audit,
            patch("src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy") as mock_compute,
            patch("src.domain.futures.alpha_foundry.diversity.resolve_cross_tf_shared_context") as mock_ctx,
        ):
            mock_audit.return_value = FIXED_AUDIT
            mock_compute.side_effect = ValueError("simulated-calendar-mismatch")
            mock_ctx.return_value = MagicMock()

            pruned, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test",
                enable_audit=True,
                enable_pruning=True,
            )

        assert dict(pruned) == multi_results
        assert manifest.pruning_status == "fail_open"

    def test_audit_only_when_no_demotions(self) -> None:
        """When pruning runs but finds no redundant pairs, status is audit_only."""
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        no_demotion_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=("r1", "r2"),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
            global_eff_test_count=2.0,
            pair_evidence=(
                CrossTFPairEvidence(
                    recipe_id_a="r1",
                    recipe_id_b="r2",
                    score_corr=0.3,
                    shared_directional_entries=3,
                    directional_entry_jaccard=0.1,
                    is_redundant=False,
                ),
            ),
            canonical_tf="1h",
            common_start_ns=1,
            common_end_ns=100,
            n_common_active_bars=480,
        )

        with (
            patch("src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy") as mock_compute,
            patch("src.domain.futures.alpha_foundry.diversity.resolve_cross_tf_shared_context") as mock_ctx,
        ):
            mock_compute.return_value = no_demotion_result
            mock_ctx.return_value = MagicMock()

            _, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test",
                enable_audit=False,
                enable_pruning=True,
            )

        assert manifest.pruning_status == "audit_only"


# ─── Fix C: Shared Context Gate Relaxation ────────────────────────────


class TestSharedContextGateRelaxation:
    """Verify that resolve_cross_tf_shared_context is called when
    either audit or pruning is enabled (not only when both are enabled).

    [ADR_20260713_L0_L1_ASSET_GROWTH_RESTRUCTURE] Fix C
    """

    def test_shared_context_built_when_only_pruning_enabled(self, mocker: Any) -> None:
        spy = mocker.patch(
            "src.domain.futures.alpha_foundry.diversity.resolve_cross_tf_shared_context",
            return_value=mocker.MagicMock(),
        )
        assemble_l0_strategy_delivery_manifest(
            multi_results=_base_multi_results(),
            aligned_by_tf=_base_aligned(),
            run_id_prefix="test",
            enable_audit=False,
            enable_pruning=True,
            total_l1_verification_budget=30,
            max_novelty_corr=0.70,
            min_survivors_per_archetype=1,
            min_survivors_per_tf=1,
        )
        spy.assert_called_once()

    def test_shared_context_built_when_only_audit_enabled(self, mocker: Any) -> None:
        spy = mocker.patch(
            "src.domain.futures.alpha_foundry.diversity.resolve_cross_tf_shared_context",
            return_value=mocker.MagicMock(),
        )
        assemble_l0_strategy_delivery_manifest(
            multi_results=_base_multi_results(),
            aligned_by_tf=_base_aligned(),
            run_id_prefix="test",
            enable_audit=True,
            enable_pruning=False,
            total_l1_verification_budget=30,
            max_novelty_corr=0.70,
            min_survivors_per_archetype=1,
            min_survivors_per_tf=1,
        )
        spy.assert_called_once()

    def test_manifest_contains_route_fields(self) -> None:
        _, manifest = assemble_l0_strategy_delivery_manifest(
            multi_results=_base_multi_results(),
            aligned_by_tf=_base_aligned(),
            run_id_prefix="test",
            enable_audit=False,
            enable_pruning=False,
            total_l1_verification_budget=30,
            evidence_end_ns=1000000,
        )
        assert hasattr(manifest, "routes")
        assert manifest.routes is not None
        for route in manifest.routes:
            assert hasattr(route, "timeframe")
            assert hasattr(route, "selected_recipe_ids")
            assert hasattr(route, "allocated_budget_units")
            assert hasattr(route, "evidence_end_ns")

    def test_manifest_route_budget_within_global_budget(self) -> None:
        _, manifest = assemble_l0_strategy_delivery_manifest(
            multi_results=_base_multi_results(),
            aligned_by_tf=_base_aligned(),
            run_id_prefix="test",
            enable_audit=False,
            enable_pruning=False,
            total_l1_verification_budget=30,
            evidence_end_ns=1000000,
        )
        total_allocated = sum(r.allocated_budget_units for r in manifest.routes)
        assert total_allocated <= manifest.total_l1_verification_budget
