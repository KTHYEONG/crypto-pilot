from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    CheapGateEvidence,
    DiversitySelectionResult,
)
from src.domain.futures.alpha_foundry.diversity import (
    compute_panel_correlation_matrix,
    resolve_cross_bucket_diversity,
    select_bucket_diverse_recipes,
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


def _make_evidence(recipe_id: str, *, block_lcb_bps: float) -> CheapGateEvidence:
    return CheapGateEvidence(
        recipe_id=recipe_id,
        timeframe="4h",
        symbol_scope="symbol",
        n_events=100,
        effective_n=50.0,
        mean_net_bps=block_lcb_bps + 1.0,
        nw_tstat=2.0,
        block_lcb_bps=block_lcb_bps,
        rank_ic=0.05,
        cost_drag_ratio=0.3,
        turnover_per_year=100.0,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        gate_passed=True,
        reject_reasons=(),
        bootstrap_lcb_bps=block_lcb_bps,
        bootstrap_agree=True,
    )


def _active_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=np.bool_)


class TestSelectBucketDiverseRecipes:
    # Scenario 1.3: 후보 3개, 상관 낮음, top_k=2
    def test_selects_top_k_by_block_lcb(self) -> None:
        t, n = 100, 5
        panels = {
            "r1": _make_panel("r1", np.random.randn(t, n)),
            "r2": _make_panel("r2", np.random.randn(t, n)),
            "r3": _make_panel("r3", np.random.randn(t, n)),
        }
        candidates = [
            _make_evidence("r1", block_lcb_bps=3.0),
            _make_evidence("r2", block_lcb_bps=5.0),
            _make_evidence("r3", block_lcb_bps=1.0),
        ]
        active = _active_mask((t, n))
        result = select_bucket_diverse_recipes(
            bucket_key=("ema_trend", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            fwd_ret_by_recipe_id={},
            active_mask=active,
            top_k_per_family_tf=2,
            max_novelty_corr=0.85,
            fdr_alpha=0.10,
            min_conviction_lcb_bps=0.0,
        )
        assert len(result.selected_recipe_ids) == 2
        assert result.selected_recipe_ids[0] == "r2"  # block_lcb_bps=5.0
        assert result.selected_recipe_ids[1] == "r1"  # block_lcb_bps=3.0
        assert "r3" in result.redundant_recipe_ids

    # Scenario 2.2: 버킷 1개 후보
    def test_single_candidate(self) -> None:
        t, n = 100, 5
        panels = {"r1": _make_panel("r1", np.random.randn(t, n))}
        candidates = [_make_evidence("r1", block_lcb_bps=3.0)]
        active = _active_mask((t, n))
        result = select_bucket_diverse_recipes(
            bucket_key=("ema_trend", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            fwd_ret_by_recipe_id={},
            active_mask=active,
            top_k_per_family_tf=5,
            max_novelty_corr=0.85,
            fdr_alpha=0.10,
            min_conviction_lcb_bps=0.0,
        )
        assert result.selected_recipe_ids == ("r1",)
        assert result.bucket_eff_test_count == 1.0

    # Scenario 2.3: top_k 조기종료 — 상관 계산 mock spy
    def test_top_k_early_exit(self) -> None:
        t, n = 100, 5
        rng = np.random.default_rng(42)
        panels = {f"r{i}": _make_panel(f"r{i}", rng.normal(0, 1, (t, n))) for i in range(5)}
        candidates = [_make_evidence(f"r{i}", block_lcb_bps=float(5 - i)) for i in range(5)]
        active = _active_mask((t, n))
        result = select_bucket_diverse_recipes(
            bucket_key=("test", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            fwd_ret_by_recipe_id={},
            active_mask=active,
            top_k_per_family_tf=2,
            max_novelty_corr=0.85,
            fdr_alpha=0.10,
            min_conviction_lcb_bps=0.0,
        )
        assert len(result.selected_recipe_ids) == 2
        assert len(result.redundant_recipe_ids) == 3

    # Scenario 2.4: 동률 tie-break
    def test_tie_break_by_recipe_id(self) -> None:
        t, n = 100, 5
        score = np.random.randn(t, n)
        panels = {
            "a": _make_panel("a", score),
            "b": _make_panel("b", score),
        }
        candidates = [
            _make_evidence("b", block_lcb_bps=3.0),
            _make_evidence("a", block_lcb_bps=3.0),
        ]
        active = _active_mask((t, n))
        result = select_bucket_diverse_recipes(
            bucket_key=("test", "4h"),
            candidates=candidates,
            panel_by_recipe_id=panels,
            fwd_ret_by_recipe_id={},
            active_mask=active,
            top_k_per_family_tf=2,
            max_novelty_corr=0.85,
            fdr_alpha=0.10,
            min_conviction_lcb_bps=0.0,
        )
        # 동일 block_lcb_bps → recipe_id asc: "a" 먼저
        assert result.ranked_recipe_ids[0] == "a"

    # Scenario 2.5: top_k=0 방어 — 이미 contracts에서 검증
    # Scenario 3.1: 빈 panels
    def test_empty_candidates(self) -> None:
        active = _active_mask((10, 3))
        result = select_bucket_diverse_recipes(
            bucket_key=("test", "4h"),
            candidates=[],
            panel_by_recipe_id={},
            fwd_ret_by_recipe_id={},
            active_mask=active,
            top_k_per_family_tf=5,
            max_novelty_corr=0.85,
            fdr_alpha=0.10,
            min_conviction_lcb_bps=0.0,
        )
        assert result.selected_recipe_ids == ()
        assert result.bucket_eff_test_count == 0.0


class TestResolveCrossBucketDiversity:
    # Scenario 1.4: 서로 다른 family, 버킷간 상관 낮음
    def test_keeps_all_when_uncorrelated(self) -> None:
        t, n = 100, 5
        rng = np.random.default_rng(1)
        panels = {
            "a1": _make_panel("a1", rng.normal(0, 1, (t, n))),
            "b1": _make_panel("b1", rng.normal(0, 1, (t, n))),
        }
        ev_a1 = _make_evidence("a1", block_lcb_bps=5.0)
        ev_b1 = _make_evidence("b1", block_lcb_bps=3.0)
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
            evidence_by_recipe_id={"a1": ev_a1, "b1": ev_b1},
            active_mask=active,
            max_novelty_corr=0.85,
        )
        assert "a1" in result.final_selected_recipe_ids
        assert "b1" in result.final_selected_recipe_ids
        assert len(result.demoted_recipe_ids) == 0

    # Scenario 2.7: 교차버킷 강등
    def test_demotes_highly_correlated_cross_bucket(self) -> None:
        t, n = 100, 5
        common_score = np.random.randn(t, n)
        panels = {
            "a1": _make_panel("a1", common_score),
            "b1": _make_panel("b1", common_score),  # 동일 점수 → 높은 상관
        }
        ev_a1 = _make_evidence("a1", block_lcb_bps=5.0)
        ev_b1 = _make_evidence("b1", block_lcb_bps=3.0)
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
            evidence_by_recipe_id={"a1": ev_a1, "b1": ev_b1},
            active_mask=active,
            max_novelty_corr=0.85,
        )
        # 상관 높음 → block_lcb_bps 낮은 b1 강등
        assert "a1" in result.final_selected_recipe_ids
        assert "b1" not in result.final_selected_recipe_ids
        assert "b1" in result.demoted_recipe_ids

    # Scenario 2.2-like: 빈 bucket_results
    def test_empty_bucket_results(self) -> None:
        active = _active_mask((10, 3))
        result = resolve_cross_bucket_diversity(
            bucket_results=[],
            panel_by_recipe_id={},
            evidence_by_recipe_id={},
            active_mask=active,
            max_novelty_corr=0.85,
        )
        assert result.final_selected_recipe_ids == ()
        assert result.global_eff_test_count == 0.0

    # Scenario 3.1: 빈 panels correlation
    def test_raises_on_empty_panels_corr(self) -> None:
        active = _active_mask((10, 3))
        with pytest.raises(ValueError, match="empty"):
            compute_panel_correlation_matrix([], active)
