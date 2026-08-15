from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    DiversitySelectionResult,
    L0SignalCandidate,
)
from src.domain.futures.alpha_foundry.diversity import (
    compute_panel_correlation_matrix,
    resolve_cross_bucket_diversity,
    select_bucket_diverse_candidates,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel


def _make_panel(recipe_id: str, score: np.ndarray, *, valid: np.ndarray | None = None) -> CandidateSignalPanel:
    t, n = score.shape
    datetimes = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    return CandidateSignalPanel(
        family="ema_trend",
        variant=recipe_id,
        params={},
        datetimes=datetimes,
        symbols=tuple(f"s{i}" for i in range(n)),
        signed_score_2d=score.astype(np.float64),
        side_hint_2d=np.ones((t, n), dtype=np.int8),
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=valid if valid is not None else np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": recipe_id},
    )


def _make_candidate(recipe_id: str, *, priority_score: float) -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test",
        timeframe="4h",
        family="ema_trend",
        variant=recipe_id,
        recipe_id=recipe_id,
        archetype="trend",
        source="catalog_exact",
        n_events=100,
        effective_n=50.0,
        mean_net_bps=priority_score,
        block_lcb_bps=priority_score - 1.0,
        nw_tstat=2.0,
        bootstrap_lcb_bps=priority_score - 1.0,
        bootstrap_agree=True,
        cost_drag_ratio=0.3,
        turnover_per_year=100.0,
        max_abs_corr_in_bucket=0.0,
        tf_coverage_count=0,
        sign_agreement_ratio=0.0,
        corroboration_tier="insufficient_coverage",
        discovery_tier="candidate",
        l1_priority_score=priority_score,
        l1_budget_units=0,
        hard_reject_reasons=(),
        soft_flags=(),
    )


def _active_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=np.bool_)


class TestSelectBucketDiverseCandidates:
    def test_selects_top_k_by_priority_score(self) -> None:
        t, n = 100, 5
        panels = {
            "r1": _make_panel("r1", np.random.randn(t, n)),
            "r2": _make_panel("r2", np.random.randn(t, n)),
            "r3": _make_panel("r3", np.random.randn(t, n)),
        }
        candidates = [
            _make_candidate("r1", priority_score=3.0),
            _make_candidate("r2", priority_score=5.0),
            _make_candidate("r3", priority_score=1.0),
        ]
        active = _active_mask((t, n))
        result = select_bucket_diverse_candidates(
            bucket_key=("ema_trend", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            active_mask=active,
            top_k_per_family_tf=2,
            max_novelty_corr=0.85,
        )
        assert len(result.selected_recipe_ids) == 2
        assert result.selected_recipe_ids[0] == "r2"
        assert result.selected_recipe_ids[1] == "r1"
        assert "r3" in result.redundant_recipe_ids

    def test_single_candidate(self) -> None:
        t, n = 100, 5
        panels = {"r1": _make_panel("r1", np.random.randn(t, n))}
        candidates = [_make_candidate("r1", priority_score=3.0)]
        active = _active_mask((t, n))
        result = select_bucket_diverse_candidates(
            bucket_key=("ema_trend", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            active_mask=active,
            top_k_per_family_tf=5,
            max_novelty_corr=0.85,
        )
        assert result.selected_recipe_ids == ("r1",)
        assert result.bucket_eff_test_count == 1.0

    def test_top_k_early_exit(self) -> None:
        t, n = 100, 5
        rng = np.random.default_rng(42)
        panels = {f"r{i}": _make_panel(f"r{i}", rng.normal(0, 1, (t, n))) for i in range(5)}
        candidates = [_make_candidate(f"r{i}", priority_score=float(5 - i)) for i in range(5)]
        active = _active_mask((t, n))
        result = select_bucket_diverse_candidates(
            bucket_key=("test", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            active_mask=active,
            top_k_per_family_tf=2,
            max_novelty_corr=0.85,
        )
        assert len(result.selected_recipe_ids) == 2
        assert len(result.redundant_recipe_ids) == 3

    def test_tie_break_by_recipe_id(self) -> None:
        t, n = 100, 5
        score = np.random.randn(t, n)
        panels = {
            "a": _make_panel("a", score),
            "b": _make_panel("b", score),
        }
        candidates = [
            _make_candidate("b", priority_score=3.0),
            _make_candidate("a", priority_score=3.0),
        ]
        active = _active_mask((t, n))
        result = select_bucket_diverse_candidates(
            bucket_key=("test", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            active_mask=active,
            top_k_per_family_tf=2,
            max_novelty_corr=0.85,
        )
        # Same priority_score -> recipe_id asc: "a" first
        assert result.ranked_recipe_ids[0] == "a"

    def test_empty_candidates(self) -> None:
        active = _active_mask((10, 3))
        result = select_bucket_diverse_candidates(
            bucket_key=("test", "4h"),
            candidates=[],
            panel_by_recipe_id={},
            active_mask=active,
            top_k_per_family_tf=5,
            max_novelty_corr=0.85,
        )
        assert result.selected_recipe_ids == ()
        assert result.bucket_eff_test_count == 0.0

    def test_missing_panel_falls_back_to_identity_corr(self) -> None:
        t, n = 100, 3
        score = np.random.randn(t, n)
        panels = {
            "r1": _make_panel("r1", score),
        }
        candidates = [
            _make_candidate("r1", priority_score=5.0),
            _make_candidate("r2", priority_score=3.0),
        ]
        active = _active_mask((t, n))
        result = select_bucket_diverse_candidates(
            bucket_key=("test", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            active_mask=active,
            top_k_per_family_tf=5,
            max_novelty_corr=0.0,
        )
        # r2 has no panel -> falls into line 178 continue path and line 201 identity corr
        assert len(result.selected_recipe_ids) >= 1


class TestResolveCrossBucketDiversity:
    def test_keeps_all_when_uncorrelated(self) -> None:
        t, n = 100, 5
        rng = np.random.default_rng(1)
        panels = {
            "a1": _make_panel("a1", rng.normal(0, 1, (t, n))),
            "b1": _make_panel("b1", rng.normal(0, 1, (t, n))),
        }
        cand_a1 = _make_candidate("a1", priority_score=5.0)
        cand_b1 = _make_candidate("b1", priority_score=3.0)
        bucket_results = [
            DiversitySelectionResult(
                bucket_key=("family_a", "4h"),
                ranked_recipe_ids=("a1",),
                selected_recipe_ids=("a1",),
                redundant_recipe_ids=(),
                redundant_reason_by_id={},
                bucket_corr=np.array([[1.0]]),
                bucket_eff_test_count=1.0,
            ),
            DiversitySelectionResult(
                bucket_key=("family_b", "4h"),
                ranked_recipe_ids=("b1",),
                selected_recipe_ids=("b1",),
                redundant_recipe_ids=(),
                redundant_reason_by_id={},
                bucket_corr=np.array([[1.0]]),
                bucket_eff_test_count=1.0,
            ),
        ]
        active = _active_mask((t, n))
        result = resolve_cross_bucket_diversity(
            bucket_results=bucket_results,
            panel_by_recipe_id=panels,
            candidate_by_recipe_id={"a1": cand_a1, "b1": cand_b1},
            active_mask=active,
            max_novelty_corr=0.85,
        )
        assert "a1" in result.final_selected_recipe_ids
        assert "b1" in result.final_selected_recipe_ids
        assert len(result.demoted_recipe_ids) == 0

    def test_demotes_highly_correlated_cross_bucket(self) -> None:
        t, n = 100, 5
        common_score = np.random.randn(t, n)
        panels = {
            "a1": _make_panel("a1", common_score),
            "b1": _make_panel("b1", common_score),
        }
        cand_a1 = _make_candidate("a1", priority_score=5.0)
        cand_b1 = _make_candidate("b1", priority_score=3.0)
        bucket_results = [
            DiversitySelectionResult(
                bucket_key=("family_a", "4h"),
                ranked_recipe_ids=("a1",),
                selected_recipe_ids=("a1",),
                redundant_recipe_ids=(),
                redundant_reason_by_id={},
                bucket_corr=np.array([[1.0]]),
                bucket_eff_test_count=1.0,
            ),
            DiversitySelectionResult(
                bucket_key=("family_b", "4h"),
                ranked_recipe_ids=("b1",),
                selected_recipe_ids=("b1",),
                redundant_recipe_ids=(),
                redundant_reason_by_id={},
                bucket_corr=np.array([[1.0]]),
                bucket_eff_test_count=1.0,
            ),
        ]
        active = _active_mask((t, n))
        result = resolve_cross_bucket_diversity(
            bucket_results=bucket_results,
            panel_by_recipe_id=panels,
            candidate_by_recipe_id={"a1": cand_a1, "b1": cand_b1},
            active_mask=active,
            max_novelty_corr=0.85,
        )
        assert "a1" in result.final_selected_recipe_ids
        assert "b1" not in result.final_selected_recipe_ids
        assert "b1" in result.demoted_recipe_ids

    def test_empty_bucket_results(self) -> None:
        active = _active_mask((10, 3))
        result = resolve_cross_bucket_diversity(
            bucket_results=[],
            panel_by_recipe_id={},
            candidate_by_recipe_id={},
            active_mask=active,
            max_novelty_corr=0.85,
        )
        assert result.final_selected_recipe_ids == ()
        assert result.global_eff_test_count == 0.0

    def test_raises_on_empty_panels_corr(self) -> None:
        active = _active_mask((10, 3))
        with pytest.raises(ValueError, match="empty"):
            compute_panel_correlation_matrix([], active)

    def test_candidate_lookup_fallback_path(self) -> None:
        """candidate_by_recipe_id missing an entry -> line 261 fallback, line 293 _best_key fallback."""
        t, n = 100, 5
        common_score = np.random.randn(t, n)  # identical -> high correlation
        panels = {
            "a1": _make_panel("a1", common_score),
            "b1": _make_panel("b1", common_score),
        }
        cand_a1 = _make_candidate("a1", priority_score=5.0)
        bucket_results = [
            DiversitySelectionResult(
                bucket_key=("family_a", "4h"),
                ranked_recipe_ids=("a1", "b1"),
                selected_recipe_ids=("a1", "b1"),
                redundant_recipe_ids=(),
                redundant_reason_by_id={},
                bucket_corr=np.array([[1.0]]),
                bucket_eff_test_count=1.0,
            ),
        ]
        active = _active_mask((t, n))
        result = resolve_cross_bucket_diversity(
            bucket_results=bucket_results,
            panel_by_recipe_id=panels,
            candidate_by_recipe_id={"a1": cand_a1},
            active_mask=active,
            max_novelty_corr=0.5,
        )
        assert "a1" in result.final_selected_recipe_ids

    def test_panel_missing_fallback_path(self) -> None:
        t, n = 100, 5
        common_score = np.random.randn(t, n)
        panels = {
            "a1": _make_panel("a1", common_score),
        }
        cand_a1 = _make_candidate("a1", priority_score=5.0)
        cand_b1 = _make_candidate("b1", priority_score=3.0)  # no panel for b1

        bucket_results = [
            DiversitySelectionResult(
                bucket_key=("family_a", "4h"),
                ranked_recipe_ids=("a1", "b1"),
                selected_recipe_ids=("a1", "b1"),
                redundant_recipe_ids=(),
                redundant_reason_by_id={},
                bucket_corr=np.array([[1.0]]),
                bucket_eff_test_count=1.0,
            ),
        ]
        active = _active_mask((t, n))
        result = resolve_cross_bucket_diversity(
            bucket_results=bucket_results,
            panel_by_recipe_id=panels,
            candidate_by_recipe_id={"a1": cand_a1, "b1": cand_b1},
            active_mask=active,
            max_novelty_corr=0.85,
        )
        # b1 has no panel -> falls into line 250 len(panels) < 2 path
        assert "a1" in result.final_selected_recipe_ids
