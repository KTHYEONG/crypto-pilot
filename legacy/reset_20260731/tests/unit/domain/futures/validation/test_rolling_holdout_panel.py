"""Tests for L3 Rolling Holdout Panel + ADR-Level Deflation.

Covers:
- ValidationEpisode + build_validation_episode_panel (opt_config.py)
- EpisodeOutcome + RollingConsistencyVerdict + evaluate_rolling_holdout_consistency (gates.py)
- ADR Sharpe pool (run_tracker.py)
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from src.domain.futures.optimization.observability.run_tracker import (
    adr_sharpe_pool_study_name,
    compute_adr_level_deflated_sharpe,
    get_adr_sharpe_pool,
    record_adr_evaluation,
)
from src.domain.futures.optimization.opt_config import (
    REGIME_FLOOR,
    LayeredWindow,
    build_validation_episode_panel,
)
from src.domain.futures.validation.gates import (
    EpisodeOutcome,
    evaluate_rolling_holdout_consistency,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def in_memory_storage() -> optuna.storages.InMemoryStorage:
    return optuna.storages.InMemoryStorage()


# ===========================================================================
# Scenario 1 — Happy Path
# ===========================================================================


class TestBuildValidationEpisodePanel:
    """1.1 build_validation_episode_panel: promotion 2개 + stress 1개 생성."""

    def test_happy_path_3_episodes(self) -> None:
        result = build_validation_episode_panel(
            promotion_reference_dates=["2026-01-01", "2026-04-01"],
            stress_reference_dates=["2023-01-01"],
            l1_months=9,
            l2_months=6,
            holdout_months=3,
        )
        assert len(result) == 3
        # first two are promotion
        assert result[0].role == "promotion"
        assert result[0].window.regime_floor == REGIME_FLOOR
        assert result[1].role == "promotion"
        # third is stress_only
        assert result[2].role == "stress_only"
        assert result[2].window.l1_start < REGIME_FLOOR  # 2023 이전 데이터 사용
        # episode_id deterministic
        assert result[0].episode_id == "promotion_2026-01-01"
        assert result[2].episode_id == "stress_only_2023-01-01"

    def test_episode_id_format(self) -> None:
        result = build_validation_episode_panel(
            promotion_reference_dates=["2025-06-15"],
            l1_months=9,
            l2_months=6,
            holdout_months=3,
        )
        assert result[0].episode_id == "promotion_2025-06-15"

    def test_stress_window_bypasses_regime_floor(self) -> None:
        """stress episode는 regime_floor=date.min으로 클램프 우회."""
        result = build_validation_episode_panel(
            promotion_reference_dates=[],
            stress_reference_dates=["2022-10-01"],
            l1_months=9,
            l2_months=6,
            holdout_months=3,
        )
        assert len(result) == 1
        assert result[0].role == "stress_only"
        assert result[0].window.l1_start < REGIME_FLOOR


class TestEvaluateRollingHoldoutConsistency:
    """1.2 evaluate_rolling_holdout_consistency: 전 promotion episode 개선 시 통과."""

    def test_all_promotion_improve_passes(self) -> None:
        outcomes = [
            EpisodeOutcome("promotion_A", "promotion", 0.05, -0.02),
            EpisodeOutcome("promotion_B", "promotion", 0.03, -0.01),
            EpisodeOutcome("stress_C", "stress_only", -0.10, -0.30),
        ]
        verdict = evaluate_rolling_holdout_consistency(outcomes)
        assert verdict.consistent_improvement is True
        assert verdict.stress_generalization_pass is True  # stress도 candidate>baseline
        assert verdict.failing_episode_ids == ()
        assert verdict.n_promotion_episodes == 2

    def test_stress_fails_but_promotion_passes(self) -> None:
        """stress_only 실패해도 promotion 일관성에는 영향 없음."""
        outcomes = [
            EpisodeOutcome("promotion_A", "promotion", 0.05, 0.02),
            EpisodeOutcome("stress_B", "stress_only", -0.10, 0.30),
        ]
        verdict = evaluate_rolling_holdout_consistency(outcomes)
        assert verdict.consistent_improvement is True
        assert verdict.stress_generalization_pass is False
        assert verdict.failing_episode_ids == ()


class TestAdrSharpePool:
    """1.3 record_adr_evaluation + compute_adr_level_deflated_sharpe."""

    def test_record_adr_evaluation_keeps_losing_attempts(
        self,
        in_memory_storage: optuna.storages.BaseStorage,
    ) -> None:
        record_adr_evaluation("4h", in_memory_storage, sharpe=0.5, adr_id="adr_0")
        record_adr_evaluation("4h", in_memory_storage, sharpe=-0.3, adr_id="adr_1")
        record_adr_evaluation("4h", in_memory_storage, sharpe=0.2, adr_id="adr_2")

        pool = get_adr_sharpe_pool("4h", in_memory_storage)
        assert len(pool) == 3
        assert set(pool.tolist()) == {0.5, -0.3, 0.2}

    def test_compute_adr_level_deflated_sharpe_with_pool(
        self,
        in_memory_storage: optuna.storages.BaseStorage,
    ) -> None:
        record_adr_evaluation("1h", in_memory_storage, sharpe=0.3, adr_id="adr_0")
        record_adr_evaluation("1h", in_memory_storage, sharpe=1.2, adr_id="adr_1")
        candidate = np.array([0.001, 0.002, 0.0015], dtype=np.float64)
        result = compute_adr_level_deflated_sharpe(
            candidate,
            tag="1h",
            storage=in_memory_storage,
            tf="4h",
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# Scenario 2 — Edge Cases
# ===========================================================================


class TestEdgeCases:
    """2.1 stress episode window가 promotion과 비중첩."""

    def test_stress_window_independent_from_promotion(self) -> None:
        result = build_validation_episode_panel(
            promotion_reference_dates=["2026-01-01"],
            stress_reference_dates=["2023-01-01"],
            l1_months=9,
            l2_months=6,
            holdout_months=3,
        )
        promo_fetch = result[0].window.fetch_start
        stress_fetch = result[1].window.fetch_start
        # stress episode window가 promotion과 완전히 다른 구간
        assert stress_fetch < result[1].window.l1_start
        assert promo_fetch < result[0].window.l1_start
        # stress의 l1_start가 promotion의 fetch_start보다 이전 (비중첩)
        assert result[1].window.l1_start < promo_fetch

    def test_consistency_magnitude_independent(self) -> None:
        """2.2 부호 일관성 판정 — magnitude가 아닌 순서/부호만 판정."""
        small = [
            EpisodeOutcome("a", "promotion", 0.001, 0.0),
            EpisodeOutcome("b", "promotion", 0.001, -0.001),
        ]
        large = [
            EpisodeOutcome("a", "promotion", 100.0, 0.0),
            EpisodeOutcome("b", "promotion", 200.0, -100.0),
        ]
        assert evaluate_rolling_holdout_consistency(small).consistent_improvement is True
        assert evaluate_rolling_holdout_consistency(large).consistent_improvement is True

    def test_consistency_fails_on_one_promotion_decline(self) -> None:
        """하나의 promotion episode라도 candidate < baseline이면 실패."""
        outcomes = [
            EpisodeOutcome("good", "promotion", 0.05, 0.01),
            EpisodeOutcome("bad", "promotion", -0.02, 0.01),
        ]
        verdict = evaluate_rolling_holdout_consistency(outcomes)
        assert verdict.consistent_improvement is False
        assert "bad" in verdict.failing_episode_ids

    def test_stress_episode_not_in_failing(self) -> None:
        """stress_only 실패는 failing_episode_ids에 미포함."""
        outcomes = [
            EpisodeOutcome("promo_A", "promotion", 0.05, -0.01),
            EpisodeOutcome("stress_X", "stress_only", -0.50, 0.10),
        ]
        verdict = evaluate_rolling_holdout_consistency(outcomes)
        assert verdict.consistent_improvement is True
        assert "stress_X" not in verdict.failing_episode_ids
        assert verdict.stress_generalization_pass is False

    def test_no_promotion_episodes_with_stress(self) -> None:
        """promotion 0개, stress만 있으면 consistent_improvement=False."""
        outcomes = [
            EpisodeOutcome("stress_X", "stress_only", 0.10, 0.05),
        ]
        verdict = evaluate_rolling_holdout_consistency(outcomes)
        assert verdict.consistent_improvement is False
        assert verdict.failing_episode_ids == ("no_promotion_episodes",)

    def test_regime_floor_policy_stress_only(self) -> None:
        """2.3 build_validation_episode_panel은 순수 데이터 생성 — fit 호출 안 함."""
        result = build_validation_episode_panel(
            promotion_reference_dates=[],
            stress_reference_dates=["2022-06-01"],
            l1_months=9,
            l2_months=6,
            holdout_months=3,
        )
        # window 존재만 확인 (호출자가 fit 파이프라인에 넘기지 않는 책임)
        assert isinstance(result[0].window, LayeredWindow)
        assert result[0].window.l1_start < REGIME_FLOOR

    def test_adr_pool_respects_tag_isolation(self) -> None:
        """2.4 tag별 pool 격리."""
        rec = record_adr_evaluation
        get = get_adr_sharpe_pool
        storage = optuna.storages.InMemoryStorage()
        rec("tag_a", storage, sharpe=0.5, adr_id="a1")
        rec("tag_b", storage, sharpe=0.9, adr_id="b1")
        pool_a = get("tag_a", storage)
        pool_b = get("tag_b", storage)
        assert len(pool_a) == 1
        assert pool_a[0] == 0.5
        assert len(pool_b) == 1
        assert pool_b[0] == 0.9


# ===========================================================================
# Scenario 3 — Error Handling
# ===========================================================================


class TestErrorHandling:
    """3.1 빈 outcomes → consistent_improvement=False."""

    def test_empty_outcomes_returns_failure(self) -> None:
        verdict = evaluate_rolling_holdout_consistency([])
        assert verdict.consistent_improvement is False
        assert verdict.failing_episode_ids == ("no_promotion_episodes",)
        assert verdict.n_promotion_episodes == 0
        assert verdict.stress_generalization_pass is None

    def test_get_adr_sharpe_pool_unknown_tag_returns_empty(
        self,
        in_memory_storage: optuna.storages.BaseStorage,
    ) -> None:
        """3.2 존재하지 않는 tag → 빈 배열, 예외 없음."""
        pool = get_adr_sharpe_pool("nonexistent_tag", in_memory_storage)
        assert isinstance(pool, np.ndarray)
        assert len(pool) == 0

    def test_compute_adr_level_deflated_sharpe_empty_pool(
        self,
        in_memory_storage: optuna.storages.BaseStorage,
    ) -> None:
        """3.3 pool 비어있음 (첫 ADR 시도) → fallback 경로."""
        candidate = np.array([0.001, 0.002, 0.0015, 0.0018], dtype=np.float64)
        result = compute_adr_level_deflated_sharpe(
            candidate,
            tag="fresh",
            storage=in_memory_storage,
            tf="4h",
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_empty_promotion_dates_with_stress(self) -> None:
        """3.4 빈 promotion_reference_dates → promotion 없음, stress 정상 생성."""
        result = build_validation_episode_panel(
            promotion_reference_dates=[],
            stress_reference_dates=["2023-01-01"],
            l1_months=9,
            l2_months=6,
            holdout_months=3,
        )
        promo = [e for e in result if e.role == "promotion"]
        stress = [e for e in result if e.role == "stress_only"]
        assert len(promo) == 0
        assert len(stress) == 1

    def test_adr_sharpe_pool_study_name_format(self) -> None:
        assert adr_sharpe_pool_study_name("4h") == "adr_sharpe_pool_4h"
        assert adr_sharpe_pool_study_name("test") == "adr_sharpe_pool_test"
