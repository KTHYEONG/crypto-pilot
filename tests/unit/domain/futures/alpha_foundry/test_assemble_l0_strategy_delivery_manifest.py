from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.bridge_helpers import (
    AlphaFoundryL0Result,
    assemble_l0_strategy_delivery_manifest,
)
from src.domain.futures.alpha_foundry.contracts import (
    CrossBucketDiversityResult,
    L0IndependenceAudit,
    L0SignalCandidate,
)


def _candidate(recipe_id: str, archetype: str, timeframe: str, priority: float) -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test", timeframe=timeframe, family="btc_regime_pullback", variant="v",
        recipe_id=recipe_id, archetype=archetype, source="synthetic_recipe",
        n_events=100, effective_n=50.0, mean_net_bps=priority, block_lcb_bps=priority * 0.5,
        nw_tstat=1.5, bootstrap_lcb_bps=0.0, bootstrap_agree=True, cost_drag_ratio=0.3,
        turnover_per_year=50.0, max_abs_corr_in_bucket=0.0, tf_coverage_count=0,
        sign_agreement_ratio=0.0, corroboration_tier="single_tf_strict", discovery_tier="candidate",
        l1_priority_score=priority, l1_budget_units=1, hard_reject_reasons=(), soft_flags=(),
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

        def __init__(self) -> None:
            self.metadata = {"recipe_id": recipe_id}
    return _Panel()


FIXED_AUDIT = L0IndependenceAudit(
    n_selected_total=2, n_distinct_thesis_ids=1, n_independent_clusters=1,
    cluster_members={0: ("r1",), 1: ("r2",)},
    demoted_recipe_ids=(), demoted_reason_by_id={},
    canonical_tf="4h", max_corr_threshold=0.70,
)

FIXED_CROSS_RESULT = CrossBucketDiversityResult(
    final_selected_recipe_ids=("r1",),
    demoted_recipe_ids=("r2",),
    demoted_reason_by_id={"r2": "r1"},
    cross_bucket_corr=np.array([[1.0, 0.9], [0.9, 1.0]]),
    global_eff_test_count=1.2,
)


def _base_multi_results() -> dict[str, AlphaFoundryL0Result]:
    """Two candidates from same TF to avoid TF floor re-admission."""
    c1 = _candidate("r1", "trend", "4h", priority=10.0)
    c2 = _candidate("r2", "trend", "4h", priority=5.0)
    return {
        "4h": AlphaFoundryL0Result(
            panels_for_l1=(_mock_panel("r1"), _mock_panel("r2")),
            summary_report=None, gate_results=(), panel_bindings=(),
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
            canonical_tf="4h",
            run_id_prefix="test",
            enable_audit=False,
            enable_pruning=False,
        )

        assert manifest.independence_audit is None
        assert dict(pruned) == multi_results

    def test_audit_only_populates_audit_and_returns_unpruned(self) -> None:
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with patch("src.domain.futures.alpha_foundry.diversity.audit_l0_selected_recipe_independence") as mock_audit:
            mock_audit.return_value = FIXED_AUDIT

            pruned, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                canonical_tf="4h",
                run_id_prefix="test",
                enable_audit=True,
                enable_pruning=False,
            )

            assert manifest.independence_audit is FIXED_AUDIT
            mock_audit.assert_called_once()
        assert dict(pruned) == multi_results

    def test_pruning_enabled_returns_pruned_results(self) -> None:
        """With both candidates from same TF, floor doesn't re-admit r2.
        Assert manifest.final_selected_recipe_ids reflects the pruned set."""
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with (
            patch("src.domain.futures.alpha_foundry.diversity.audit_l0_selected_recipe_independence") as mock_audit,
            patch("src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy") as mock_compute,
        ):
            mock_audit.return_value = FIXED_AUDIT
            mock_compute.return_value = FIXED_CROSS_RESULT

            pruned, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                canonical_tf="4h",
                run_id_prefix="test",
                enable_audit=True,
                enable_pruning=True,
                max_novelty_corr=0.70,
            )

        assert manifest.independence_audit is FIXED_AUDIT
        assert "r1" in manifest.final_selected_recipe_ids
        assert len(pruned["4h"].candidates_for_l1) == 1
        assert pruned["4h"].candidates_for_l1[0].recipe_id == "r1"

    def test_pruning_enabled_audit_disabled_returns_pruned_no_audit(self) -> None:
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with patch("src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy") as mock_compute:
            mock_compute.return_value = FIXED_CROSS_RESULT

            _, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                canonical_tf="4h",
                run_id_prefix="test",
                enable_audit=False,
                enable_pruning=True,
            )

        assert manifest.independence_audit is None
        assert "r1" in manifest.final_selected_recipe_ids
        assert "r2" not in manifest.final_selected_recipe_ids

    def test_key_error_when_canonical_tf_missing(self) -> None:
        mock_result = AlphaFoundryL0Result(
            panels_for_l1=(),
            summary_report=None, gate_results=(), panel_bindings=(),
        )
        multi_results = {"4h": mock_result}
        aligned_by_tf: dict[str, Any] = {"12h": _real_aligned()}

        with pytest.raises(KeyError):
            assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                canonical_tf="4h",
                run_id_prefix="test",
                enable_audit=False,
                enable_pruning=False,
            )

    def test_value_error_fallback_returns_unpruned(self) -> None:
        multi_results = _base_multi_results()
        aligned_by_tf = _base_aligned()

        with (
            patch("src.domain.futures.alpha_foundry.diversity.audit_l0_selected_recipe_independence") as mock_audit,
            patch("src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy") as mock_compute,
        ):
            mock_audit.return_value = FIXED_AUDIT
            mock_compute.side_effect = ValueError("simulated-calendar-mismatch")

            pruned, _manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                canonical_tf="4h",
                run_id_prefix="test",
                enable_audit=True,
                enable_pruning=True,
            )

        assert dict(pruned) == multi_results
