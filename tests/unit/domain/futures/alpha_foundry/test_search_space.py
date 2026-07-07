"""Tests for alpha foundry search space module.

Covers Scenarios S1-1, S1-2, E2-1, E2-2, E2-3, E2-11, X3-3, X3-4, X3-5.
"""

from __future__ import annotations

import pytest

from src.domain.futures.alpha_foundry.contracts import (
    AlphaFeatureBlueprint,
    AlphaFoundryRuntimeConfig,
    AlphaHypothesis,
    AlphaSearchPolicyState,
    AlphaSignalBlueprint,
    L0SearchCell,
    L0SignalCandidate,
)
from src.domain.futures.alpha_foundry.search_space import (
    apply_cost_prior_screen,
    build_alpha_hypotheses,
    build_feature_blueprints,
    build_l0_search_cells,
    update_search_policy_state,
)


def _make_bp(
    family: str = "sparse_breakout_retest_liquidity",
    variant: str = "sbrl_40",
    timeframe: str = "4h",
    entry_mode: str = "sparse",
    holding_bars: int = 6,
) -> AlphaSignalBlueprint:
    return AlphaSignalBlueprint(
        family=family,
        variant=variant,
        archetype="trend",
        timeframe=timeframe,
        required_fields=("close", "high", "low"),
        causal_lag_bars=1,
        lookback_bars=(40,),
        holding_bars=holding_bars,
        max_turnover_per_year=180.0,
        entry_mode=entry_mode,
        side_rule_id="breakout_retest",
        exit_policy_id="atr_trail_2",
    )


# ── S1-1: AlphaHypothesis/AlphaFeatureBlueprint/AlphaSearchPolicyState 생성 ──


class TestAlphaHypothesis:
    def test_valid_hypothesis_creates_with_pending_status(self) -> None:
        h = AlphaHypothesis(
            hypothesis_id="h1",
            family="sparse_breakout_retest_liquidity",
            variant="sbrl_40",
            archetype="trend",
            timeframe="4h",
            data_scope=("global",),
            entry_mode="sparse",
            causal_lag_bars=1,
            holding_bars=6,
            turnover_budget_per_year=180.0,
            prior_score=0.0,
        )
        assert h.status == "pending"
        assert h.hypothesis_id == "h1"

    def test_rejects_causal_lag_below_one(self) -> None:
        with pytest.raises(ValueError, match="causal_lag_bars must be >= 1"):
            AlphaHypothesis(
                hypothesis_id="h_bad",
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                data_scope=("global",),
                entry_mode="sparse",
                causal_lag_bars=0,
                holding_bars=6,
                turnover_budget_per_year=180.0,
                prior_score=0.0,
            )

    def test_rejects_negative_turnover_budget(self) -> None:
        with pytest.raises(ValueError, match=r"turnover_budget_per_year must be >= 0\.0"):
            AlphaHypothesis(
                hypothesis_id="h_bad",
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                data_scope=("global",),
                entry_mode="sparse",
                causal_lag_bars=1,
                holding_bars=6,
                turnover_budget_per_year=-1.0,
                prior_score=0.0,
            )


class TestAlphaFeatureBlueprint:
    def test_valid_blueprint_creates_and_preserves_thresholds(self) -> None:
        bp = AlphaFeatureBlueprint(
            blueprint_id="fb1",
            hypothesis_id="h1",
            feature_family="liquidity",
            lookback_bars=(12, 24),
            thresholds={"z_entry": 1.5, "z_exit": 0.5},
            direction_rule="fade_crowding",
            required_fields=("close",),
            validity_mask_name="active",
            max_compute_cost_score=1.0,
        )
        assert bp.thresholds["z_entry"] == 1.5
        assert bp.direction_rule == "fade_crowding"
        assert bp.lookback_bars == (12, 24)

    def test_rejects_zero_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookback_bars must be >= 1"):
            AlphaFeatureBlueprint(
                blueprint_id="fb_bad",
                hypothesis_id="h1",
                feature_family="price_structure",
                lookback_bars=(0,),
                thresholds={},
                direction_rule="trend_follow",
                required_fields=("close",),
                validity_mask_name="active",
                max_compute_cost_score=1.0,
            )

    def test_rejects_non_positive_compute_cost(self) -> None:
        with pytest.raises(ValueError, match=r"max_compute_cost_score must be > 0\.0"):
            AlphaFeatureBlueprint(
                blueprint_id="fb_bad",
                hypothesis_id="h1",
                feature_family="price_structure",
                lookback_bars=(12,),
                thresholds={},
                direction_rule="trend_follow",
                required_fields=("close",),
                validity_mask_name="active",
                max_compute_cost_score=0.0,
            )


class TestAlphaSearchPolicyState:
    def test_default_next_budget_is_one(self) -> None:
        s = AlphaSearchPolicyState(family="f", timeframe="4h")
        assert s.next_budget == 1

    def test_rejects_negative_tested_count(self) -> None:
        with pytest.raises(ValueError, match="tested_count must be >= 0"):
            AlphaSearchPolicyState(family="f", timeframe="4h", tested_count=-1)

    def test_rejects_out_of_range_pass_rate(self) -> None:
        with pytest.raises(ValueError, match="posterior_pass_rate must be in"):
            AlphaSearchPolicyState(family="f", timeframe="4h", posterior_pass_rate=1.5)


# ── S1-2: cost prior screen ──


class TestApplyCostPriorScreen:
    def test_high_cost_30m_cell_gets_retired(self) -> None:
        cell_30m = L0SearchCell(
            blueprint_id="b1",
            family="f",
            variant="v",
            timeframe="30m",
            tf_minutes=30,
            symbol_scope="global",
            cost_floor_bps=25.0,
            expected_event_rate=0.5,
            family_prior_score=0.0,
            turnover_budget_per_year=730.0,
        )
        cell_8h = L0SearchCell(
            blueprint_id="b2",
            family="f",
            variant="v2",
            timeframe="8h",
            tf_minutes=480,
            symbol_scope="global",
            cost_floor_bps=3.0,
            expected_event_rate=0.1,
            family_prior_score=0.0,
            turnover_budget_per_year=120.0,
        )
        config = AlphaFoundryRuntimeConfig(
            mode="gate",
            cost_prior_floor_by_tf={"30m": 25.0, "8h": 3.0},
        )
        result = apply_cost_prior_screen(cells=(cell_30m, cell_8h), runtime_config=config)
        result_map = {c.timeframe: c for c in result}
        assert result_map["30m"].status == "retired"
        assert result_map["30m"].retire_reason == "cost_prior_failed"
        assert result_map["8h"].status == "pending"

    def test_all_cells_pending_when_cost_manageable(self) -> None:
        cell = L0SearchCell(
            blueprint_id="b1",
            family="f",
            variant="v",
            timeframe="4h",
            tf_minutes=240,
            symbol_scope="global",
            cost_floor_bps=2.0,
            expected_event_rate=0.25,
            family_prior_score=0.0,
            turnover_budget_per_year=365.0,
        )
        config = AlphaFoundryRuntimeConfig(mode="gate")
        result = apply_cost_prior_screen(cells=(cell,), runtime_config=config)
        assert result[0].status == "pending"


# ── build_alpha_hypotheses ──


class TestBuildAlphaHypotheses:
    def test_builds_hypotheses_from_blueprints(self) -> None:
        bps = [_make_bp()]
        hyps = build_alpha_hypotheses(
            blueprints=bps,
            family_prior_scores={"sparse_breakout_retest_liquidity": 0.5},
            timeframe_cost_floor_bps={"4h": 3.0},
        )
        assert len(hyps) == 1
        assert hyps[0].prior_score == 0.5
        assert hyps[0].status == "pending"

    def test_deduplicates_identical_blueprints(self) -> None:
        bps = [_make_bp(), _make_bp()]
        hyps = build_alpha_hypotheses(
            blueprints=bps,
            family_prior_scores={},
            timeframe_cost_floor_bps={},
        )
        assert len(hyps) == 1


# ── build_feature_blueprints ──


class TestBuildFeatureBlueprints:
    def test_creates_feature_blueprints(self) -> None:
        hyp = AlphaHypothesis(
            hypothesis_id="h1",
            family="sparse_breakout_retest_liquidity",
            variant="sbrl_40",
            archetype="trend",
            timeframe="4h",
            data_scope=("global",),
            entry_mode="sparse",
            causal_lag_bars=1,
            holding_bars=6,
            turnover_budget_per_year=180.0,
            prior_score=0.0,
        )
        fbs = build_feature_blueprints(
            hypotheses=(hyp,),
            feature_family_by_family={"sparse_breakout_retest_liquidity": "liquidity"},
            threshold_templates={"sparse_breakout_retest_liquidity": {"z_entry": 2.0}},
            compute_cost_by_family={"sparse_breakout_retest_liquidity": 1.5},
        )
        assert len(fbs) == 1
        assert fbs[0].feature_family == "liquidity"
        assert fbs[0].thresholds["z_entry"] == 2.0
        assert fbs[0].max_compute_cost_score == 1.5


# ── build_l0_search_cells ──


class TestBuildL0SearchCells:
    def test_creates_cells_with_new_fields(self) -> None:
        bps = [_make_bp()]
        cells = build_l0_search_cells(
            blueprints=bps,
            family_prior_scores={"sparse_breakout_retest_liquidity": 0.3},
            cost_floor_bps_by_tf={"4h": 3.0},
            feature_family_by_family={"sparse_breakout_retest_liquidity": "liquidity"},
            max_compute_cost_by_family={"sparse_breakout_retest_liquidity": 0.8},
        )
        assert len(cells) == 1
        assert cells[0].feature_family == "liquidity"
        assert cells[0].turnover_budget_per_year == 180.0
        assert cells[0].max_compute_cost_score == 0.8
        assert cells[0].status == "pending"

    def test_raises_on_missing_generator_family(self) -> None:
        bps = [_make_bp(family="unknown")]
        with pytest.raises(ValueError, match="missing from generator_exists_by_family"):
            build_l0_search_cells(
                blueprints=bps,
                family_prior_scores={},
                cost_floor_bps_by_tf={},
                generator_exists_by_family={"other": True},
            )


# ── update_search_policy_state ──


class TestUpdateSearchPolicyState:
    def test_tested_count_increases_with_candidates(self) -> None:
        cell = L0SearchCell(
            blueprint_id="b1",
            family="f",
            variant="v",
            timeframe="4h",
            tf_minutes=240,
            symbol_scope="global",
            cost_floor_bps=3.0,
            expected_event_rate=0.25,
            family_prior_score=0.0,
        )
        cand = L0SignalCandidate(
            run_id="r1",
            timeframe="4h",
            family="f",
            variant="v",
            recipe_id="r1",
            archetype="trend",
            source="catalog_exact",
            n_events=10,
            effective_n=8.0,
            mean_net_bps=5.0,
            block_lcb_bps=2.0,
            nw_tstat=1.5,
            bootstrap_lcb_bps=1.0,
            bootstrap_agree=True,
            cost_drag_ratio=0.5,
            turnover_per_year=100.0,
            max_abs_corr_in_bucket=0.0,
            tf_coverage_count=1,
            sign_agreement_ratio=1.0,
            corroboration_tier="single_tf_strict",
            discovery_tier="seed",
            l1_priority_score=1.0,
            l1_budget_units=0,
            hard_reject_reasons=(),
            soft_flags=("weak_tstat",),
        )
        result = update_search_policy_state(cells=(cell,), candidates=(cand,), min_trials=3)
        assert result[0].tested_count >= 1
        assert result[0].survivor_count >= 1


# ── E2-1: look-ahead defense ──


class TestAlphaHypothesisLookAhead:
    def test_causal_lag_bars_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="causal_lag_bars must be >= 1"):
            AlphaHypothesis(
                hypothesis_id="h1",
                family="f",
                variant="v",
                archetype="trend",
                timeframe="4h",
                data_scope=("global",),
                entry_mode="sparse",
                causal_lag_bars=0,
                holding_bars=1,
                turnover_budget_per_year=100.0,
                prior_score=0.0,
            )


# ── E2-2: lookback_bars validation ──


class TestLookbackBarsValidation:
    def test_lookback_with_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback_bars must be >= 1"):
            AlphaFeatureBlueprint(
                blueprint_id="fb1",
                hypothesis_id="h1",
                feature_family="price_structure",
                lookback_bars=(0,),
                thresholds={},
                direction_rule="trend_follow",
                required_fields=("close",),
                validity_mask_name="active",
                max_compute_cost_score=1.0,
            )


# ── E2-3: exploration_budget_fraction validation ──


class TestRuntimeConfigBudgetFraction:
    def test_zero_budget_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="exploration_budget_fraction must be in"):
            AlphaFoundryRuntimeConfig(mode="gate", exploration_budget_fraction=0.0)

    def test_one_budget_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="exploration_budget_fraction must be in"):
            AlphaFoundryRuntimeConfig(mode="gate", exploration_budget_fraction=1.0)


# ── X3-4: unsupported timeframe ──


class TestUnsupportedTimeframe:
    def test_invalid_timeframe_raises(self) -> None:
        from src.domain.futures.alpha_foundry.search_space import timeframe_to_minutes

        with pytest.raises(ValueError, match="unsupported timeframe"):
            timeframe_to_minutes("xyz")


# ── X3-5: invalid debug_reject_bucket_rows ──


class TestDebugRejectBucketRows:
    def test_zero_debug_reject_bucket_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="debug_reject_bucket_rows must be >= 1"):
            AlphaFoundryRuntimeConfig(mode="gate", debug_reject_bucket_rows=0)


# ── E2-11: repeated hard reject → retired ──


class TestRepeatedHardReject:
    def test_retired_after_repeated_rejects(self) -> None:
        cell = L0SearchCell(
            blueprint_id="b1",
            family="f",
            variant="v",
            timeframe="4h",
            tf_minutes=240,
            symbol_scope="global",
            cost_floor_bps=3.0,
            expected_event_rate=0.25,
            family_prior_score=0.0,
        )
        cands = [
            L0SignalCandidate(
                run_id="r1",
                timeframe="4h",
                family="f",
                variant="v",
                recipe_id="r1",
                archetype="trend",
                source="catalog_exact",
                n_events=5,
                effective_n=3.0,
                mean_net_bps=0.0,
                block_lcb_bps=0.0,
                nw_tstat=0.0,
                bootstrap_lcb_bps=0.0,
                bootstrap_agree=True,
                cost_drag_ratio=2.0,
                turnover_per_year=500.0,
                max_abs_corr_in_bucket=0.0,
                tf_coverage_count=0,
                sign_agreement_ratio=0.0,
                corroboration_tier="insufficient_coverage",
                discovery_tier="blocked",
                l1_priority_score=0.0,
                l1_budget_units=0,
                hard_reject_reasons=("excess_cost_drag",),
                soft_flags=(),
            )
            for _ in range(5)
        ]
        result = update_search_policy_state(cells=(cell,), candidates=tuple(cands), min_trials=3)
        assert result[0].retire_reason == "repeated_hard_reject"
        assert result[0].status == "retired"


# ── resolve_alpha_timeframe_grid ──


class TestResolveAlphaTimeframeGrid:
    def test_fast_timeframes_includes_all(self) -> None:
        from src.domain.futures.alpha_foundry.search_space import resolve_alpha_timeframe_grid

        result = resolve_alpha_timeframe_grid(enable_fast_timeframes=True, include_daily=True)
        assert "30m" in result
        assert "1d" in result

    def test_disable_fast_timeframes_excludes_30m_1h_2h(self) -> None:
        from src.domain.futures.alpha_foundry.search_space import resolve_alpha_timeframe_grid

        result = resolve_alpha_timeframe_grid(enable_fast_timeframes=False, include_daily=False)
        assert "30m" not in result
        assert "1h" not in result
        assert "2h" not in result
        assert "1d" not in result
        assert "4h" in result

    def test_exclude_daily_removes_1d(self) -> None:
        from src.domain.futures.alpha_foundry.search_space import resolve_alpha_timeframe_grid

        result = resolve_alpha_timeframe_grid(enable_fast_timeframes=True, include_daily=False)
        assert "1d" not in result


# ── mark_retired_search_cells ──


class TestMarkRetiredSearchCells:
    def test_marks_failed_cells_as_retired(self) -> None:
        from src.domain.futures.alpha_foundry.search_space import mark_retired_search_cells

        cell1 = L0SearchCell(
            blueprint_id="b1", family="f", variant="v1", timeframe="4h",
            tf_minutes=240, symbol_scope="global", cost_floor_bps=3.0,
            expected_event_rate=0.25, family_prior_score=0.0,
        )
        cell2 = L0SearchCell(
            blueprint_id="b2", family="f", variant="v2", timeframe="4h",
            tf_minutes=240, symbol_scope="global", cost_floor_bps=3.0,
            expected_event_rate=0.25, family_prior_score=0.0,
        )
        result = mark_retired_search_cells(
            cells=(cell1, cell2),
            failed_keys={("f", "4h", "v1")},
        )
        assert result[0].status == "retired"
        assert result[1].status == "pending"
