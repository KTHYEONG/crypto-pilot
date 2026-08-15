"[LIMIT-01] floor \ubd95\uad34 \uc2dc manifest.final_selected_recipe_ids \uc640 routes \uc77c\uad00\uc131 \uac80\uc99d."

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.domain.futures.alpha_foundry.bridge_helpers import (
    AlphaFoundryL0Result,
    assemble_l0_strategy_delivery_manifest,
)
from src.domain.futures.alpha_foundry.contracts import (
    CrossBucketDiversityResult,
    L0StrategyDeliveryManifest,
    L0TfDeliveryRoute,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    select_l1_delivery_events,
)


def _make_aligned(*, tf_hours: int, bars: int = 200) -> object:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData

    dt = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-01T00:00:00") + np.timedelta64(tf_hours * bars, "h"),
        np.timedelta64(tf_hours, "h"),
        dtype="datetime64[ns]",
    )
    t = dt.shape[0]
    close = 100.0 * np.exp(0.001 * np.arange(t, dtype=np.float64))[:, None] * np.ones((1, 2))
    mask = np.ones((t, 2), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, 2), 1000.0),
        funding_2d=np.full((t, 2), 0.00005),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros((t, 2), dtype=np.bool_),
        kill_mask=np.zeros((t, 2), dtype=np.bool_),
    )


def _make_l0_candidate(recipe_id: str, timeframe: str = "4h", score: float = 1.0) -> object:
    from src.domain.futures.alpha_foundry.contracts import L0SignalCandidate

    return L0SignalCandidate(
        run_id="test",
        timeframe=timeframe,
        family="fam",
        variant="var",
        recipe_id=recipe_id,
        archetype="trend",
        source="synthetic_recipe",
        n_events=100,
        effective_n=50.0,
        mean_net_bps=score,
        block_lcb_bps=score * 0.5,
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
        l1_priority_score=score,
        l1_budget_units=1,
        hard_reject_reasons=(),
        soft_flags=(),
    )


def _build_multi_results(
    tfs: tuple[str, ...] = ("4h", "8h", "12h"),
) -> tuple[dict[str, AlphaFoundryL0Result], dict[str, object]]:
    multi_results: dict[str, AlphaFoundryL0Result] = {}
    aligned_by_tf: dict[str, object] = {}
    for i, tf in enumerate(tfs):
        tf_hours = int(tf.replace("h", ""))
        aligned = _make_aligned(tf_hours=tf_hours, bars=200)
        aligned_by_tf[tf] = aligned
        bars = 200
        score = np.full((bars, 2), 0.0)
        for start in range(10, bars, 20):
            score[start, :] = 1.0 * (1.0 if i % 2 == 0 else -1.0)
        panel = CandidateSignalPanel(
            family="fam",
            variant=f"var_{tf}",
            params={"lookback": 20},
            datetimes=np.arange(bars, dtype=np.int64),
            symbols=("BTCUSDT", "ETHUSDT"),
            signed_score_2d=score,
            side_hint_2d=np.where(score > 0, np.int8(1), np.int8(-1)),
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((bars, 2), dtype=np.float64),
            valid_mask_2d=np.ones((bars, 2), dtype=np.bool_),
            metadata={"recipe_id": f"r{i}"},
        )
        cand = _make_l0_candidate(f"r{i}", timeframe=tf, score=2.0 - i)
        multi_results[tf] = AlphaFoundryL0Result(
            panels_for_l1=(panel,),
            summary_report=None,
            gate_results=(),
            panel_bindings=(),
            candidates_for_l1=(cand,),
        )
    return multi_results, aligned_by_tf


class TestManifestRouteConsistency:
    """[LIMIT-01] floor collapse routes vs final_selected_recipe_ids consistency."""

    def test_assemble_manifest_when_floor_collapses_to_zero_routes_match_final_ids(
        self,
    ) -> None:
        multi_results, aligned_by_tf = _build_multi_results()

        collapsed_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=(),
            demoted_recipe_ids=tuple(c.recipe_id for res in multi_results.values() for c in res.candidates_for_l1),
            demoted_reason_by_id={},
            cross_bucket_corr=np.zeros((1, 1)),
            global_eff_test_count=0.0,
            canonical_tf="8h",
            common_start_ns=0,
            common_end_ns=0,
            n_common_active_bars=0,
        )

        with (
            patch(
                "src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy",
                return_value=collapsed_result,
            ),
            patch(
                "src.domain.futures.alpha_foundry.diversity.apply_cross_tf_survival_floor",
                return_value=collapsed_result,
            ),
        ):
            pruned_multi_results, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test_run",
                enable_audit=False,
                enable_pruning=True,
                total_l1_verification_budget=30,
                min_common_active_bars=0,
            )

        original_ids = {c.recipe_id for res in multi_results.values() for c in res.candidates_for_l1}
        route_ids = {rid for route in manifest.routes for rid in route.selected_recipe_ids}

        assert manifest.pruning_status == "fail_open"
        assert set(manifest.final_selected_recipe_ids) == original_ids
        assert route_ids == original_ids, (
            "manifest.routes must cover the same recipe ids as "
            "final_selected_recipe_ids -- otherwise select_l1_delivery_events "
            "blocks every TF despite a non-empty top-level id list"
        )
        for tf_k in multi_results:
            assert len(pruned_multi_results[tf_k].candidates_for_l1) == len(multi_results[tf_k].candidates_for_l1)

    def test_assemble_manifest_happy_path_partial_demotion_routes_match_final_ids(
        self,
    ) -> None:
        multi_results, aligned_by_tf = _build_multi_results()

        all_ids = [c.recipe_id for res in multi_results.values() for c in res.candidates_for_l1]
        kept_ids = tuple(all_ids[:2])
        demoted_ids = tuple(all_ids[2:])

        partial_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=kept_ids,
            demoted_recipe_ids=demoted_ids,
            demoted_reason_by_id={},
            cross_bucket_corr=np.zeros((1, 1)),
            global_eff_test_count=0.0,
            canonical_tf="8h",
            common_start_ns=0,
            common_end_ns=0,
            n_common_active_bars=0,
        )

        with (
            patch(
                "src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy",
                return_value=partial_result,
            ),
            patch(
                "src.domain.futures.alpha_foundry.diversity.apply_cross_tf_survival_floor",
                return_value=partial_result,
            ),
        ):
            pruned_multi_results, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test_run",
                enable_audit=False,
                enable_pruning=True,
                total_l1_verification_budget=30,
                min_common_active_bars=0,
            )

        route_ids = {rid for route in manifest.routes for rid in route.selected_recipe_ids}

        assert manifest.pruning_status in ("applied", "audit_only")
        assert set(manifest.final_selected_recipe_ids) == set(kept_ids)
        assert route_ids == set(kept_ids), (
            "manifest.routes must cover exactly the kept recipe ids when pruning partially demotes"
        )
        for tf_k in multi_results:
            pruned_ids = {c.recipe_id for c in pruned_multi_results[tf_k].candidates_for_l1}
            assert pruned_ids == set(kept_ids) & {c.recipe_id for c in multi_results[tf_k].candidates_for_l1}

    def test_assemble_manifest_audit_only_when_no_redundant_pairs(
        self,
    ) -> None:
        multi_results, aligned_by_tf = _build_multi_results()

        all_ids = [c.recipe_id for res in multi_results.values() for c in res.candidates_for_l1]

        no_redundancy_result = CrossBucketDiversityResult(
            final_selected_recipe_ids=tuple(all_ids),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.zeros((1, 1)),
            global_eff_test_count=0.0,
            canonical_tf="8h",
            common_start_ns=0,
            common_end_ns=0,
            n_common_active_bars=0,
        )

        with (
            patch(
                "src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy",
                return_value=no_redundancy_result,
            ),
            patch(
                "src.domain.futures.alpha_foundry.diversity.apply_cross_tf_survival_floor",
                return_value=no_redundancy_result,
            ),
        ):
            _pruned, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test_run",
                enable_audit=False,
                enable_pruning=True,
                total_l1_verification_budget=30,
                min_common_active_bars=0,
            )

        assert manifest.pruning_status == "audit_only"
        assert manifest.pruning_reason == "no redundant pairs found"
        assert set(manifest.final_selected_recipe_ids) == set(all_ids)

    def test_assemble_manifest_valueerror_failopen_routes_match_final_ids(
        self,
    ) -> None:
        multi_results, aligned_by_tf = _build_multi_results()

        with (
            patch(
                "src.domain.futures.alpha_foundry.diversity.compute_cross_tf_redundancy",
                side_effect=ValueError("simulated pruning error"),
            ),
        ):
            pruned_multi_results, manifest = assemble_l0_strategy_delivery_manifest(
                multi_results=multi_results,
                aligned_by_tf=aligned_by_tf,
                run_id_prefix="test_run",
                enable_audit=False,
                enable_pruning=True,
                total_l1_verification_budget=30,
                min_common_active_bars=0,
            )

        original_ids = {c.recipe_id for res in multi_results.values() for c in res.candidates_for_l1}
        route_ids = {rid for route in manifest.routes for rid in route.selected_recipe_ids}

        assert manifest.pruning_status == "fail_open"
        assert set(manifest.final_selected_recipe_ids) == original_ids
        assert route_ids == original_ids, (
            "manifest.routes must cover all original recipe ids after ValueError fail-open"
        )
        for tf_k in multi_results:
            assert len(pruned_multi_results[tf_k].candidates_for_l1) == len(multi_results[tf_k].candidates_for_l1)

    def test_select_l1_delivery_events_routes_events_after_floor_collapse_fallback(
        self,
    ) -> None:
        labeled_events = pd.DataFrame(
            {
                "native_tf": ["8h", "8h", "4h"],
                "l0_recipe_id": ["r1", "r2", "r3"],
            }
        )
        manifest = L0StrategyDeliveryManifest(
            run_id_prefix="test_run",
            reports_by_tf={},
            independence_audit=None,
            final_selected_recipe_ids=("r1", "r2", "r3"),
            total_l1_verification_budget=30,
            pruning_status="fail_open",
            pruning_reason="collapsed",
            routes=(
                L0TfDeliveryRoute(
                    timeframe="8h",
                    selected_recipe_ids=("r1", "r2"),
                    allocated_budget_units=2,
                    evidence_end_ns=1,
                ),
                L0TfDeliveryRoute(
                    timeframe="4h",
                    selected_recipe_ids=("r3",),
                    allocated_budget_units=1,
                    evidence_end_ns=1,
                ),
            ),
        )

        result = select_l1_delivery_events(labeled_events=labeled_events, tf="8h", manifest=manifest)

        assert not result.empty
        assert set(result["l0_recipe_id"]) == {"r1", "r2"}
