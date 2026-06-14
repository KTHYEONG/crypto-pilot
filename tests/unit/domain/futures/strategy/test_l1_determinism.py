"""
L1 재현성(Determinism) 회귀 테스트.

검증 대상:
  - threadpool_limits BLAS 단일스레드화로 run-to-run 재현성 보장
  - resolve_safe_nested_workers의 pinned 파라미터 동작
  - l1_nested_workers config 검증 (ValueError 발생 조건)
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

# ---------------------------------------------------------------------------
# Scenario 3: Config 검증 / Error Handling
# ---------------------------------------------------------------------------

class TestL1NestedWorkersConfig:
    def test_l1_nested_workers_zero_raises_value_error(self) -> None:
        # __post_init__ 검증: l1_nested_workers=0은 ValueError
        with pytest.raises(ValueError, match="l1_nested_workers"):
            CandidateStrategyConfig(l1_nested_workers=0)

    def test_l1_nested_workers_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="l1_nested_workers"):
            CandidateStrategyConfig(l1_nested_workers=-1)

    def test_l1_nested_workers_none_is_valid(self) -> None:
        # None은 동적 모드 (기본값), 생성 가능해야 함
        cfg = CandidateStrategyConfig(l1_nested_workers=None)
        assert cfg.l1_nested_workers is None

    def test_l1_nested_workers_positive_is_valid(self) -> None:
        cfg = CandidateStrategyConfig(l1_nested_workers=2)
        assert cfg.l1_nested_workers == 2


# ---------------------------------------------------------------------------
# Scenario 3 (추가): resolve_safe_nested_workers pinned 동작
# ---------------------------------------------------------------------------

class TestResolveNestedWorkersPinned:
    def test_pinned_caps_at_n_tasks(self) -> None:
        # Arrange: pinned=10 이지만 n_tasks=3이므로 min(3,10)=3 반환
        result = resolve_safe_nested_workers(n_tasks=3, frame_memory_bytes=0, pinned=10)
        assert result == 3

    def test_pinned_one_always_returns_one(self) -> None:
        result = resolve_safe_nested_workers(n_tasks=8, frame_memory_bytes=0, pinned=1)
        assert result == 1

    def test_pinned_none_falls_through_to_dynamic(self) -> None:
        # pinned=None이면 동적 계산 (psutil 의존), 최소 1 이상이어야 함
        result = resolve_safe_nested_workers(n_tasks=4, frame_memory_bytes=0, pinned=None)
        assert result >= 1
        assert result <= 4  # n_tasks 초과 불가

    def test_pinned_equals_n_tasks(self) -> None:
        result = resolve_safe_nested_workers(n_tasks=4, frame_memory_bytes=0, pinned=4)
        assert result == 4

    def test_pinned_larger_than_max_worker_cap(self) -> None:
        # pinned이 n_tasks보다 크면 n_tasks로 클램핑
        result = resolve_safe_nested_workers(n_tasks=2, frame_memory_bytes=0, pinned=100)
        assert result == 2


# ---------------------------------------------------------------------------
# Scenario 4: threadpool_limits 적용 검증 (화이트박스)
# ---------------------------------------------------------------------------

class TestFitFoldThreadpoolLimits:
    def test_threadpool_limits_called_with_blas_single(self) -> None:
        """_fit_and_predict_single_fold 호출 시 threadpool_limits(limits=1, user_api='blas') 진입 확인."""
        from src.domain.futures.strategy.candidate_workflow import _fit_and_predict_single_fold

        call_kwargs: list[dict[str, object]] = []

        class _FakeCtx:
            def __enter__(self) -> _FakeCtx:
                return self

            def __exit__(self, *args: object) -> None:
                raise RuntimeError("_STOP_INNER_EXECUTION")

        def fake_threadpool_limits(**kwargs: object) -> _FakeCtx:
            call_kwargs.append(dict(kwargs))
            return _FakeCtx()

        import pandas as pd

        from src.domain.futures.strategy.common.alignment import AlignedMarketData
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.walk_forward import WFFold

        fold = WFFold(
            fit_start=0, fit_end=50, cal_start=50, cal_end=70,
            oos_start=70, oos_end=100,
        )
        events = pd.DataFrame()
        cfg = CandidateStrategyConfig()
        size = 200
        arr = np.ones((size, 1), dtype=np.float64)
        aligned = AlignedMarketData(
            symbols=("BTC",),
            datetimes=np.array(pd.date_range("2026-01-01", periods=size, freq="4h")),
            open_2d=arr,
            high_2d=arr,
            low_2d=arr,
            close_2d=arr,
            volume_2d=arr,
            funding_2d=np.zeros((size, 1)),
            active_mask=np.ones((size, 1), dtype=bool),
            warm_mask=np.ones((size, 1), dtype=bool),
            entry_block_mask=np.zeros((size, 1), dtype=bool),
            kill_mask=np.zeros((size, 1), dtype=bool),
        )

        # Act: threadpool_limits를 fake로 교체 → _FakeCtx.__exit__이 RuntimeError 발생 (내부 실행 중단)
        with patch(
            "src.domain.futures.strategy.candidate_workflow.threadpool_limits",
            side_effect=fake_threadpool_limits,
        ), pytest.raises(RuntimeError, match="_STOP_INNER_EXECUTION"):
            _fit_and_predict_single_fold(0, fold, events, aligned, cfg, 0)

        # Assert: limits=1, user_api='blas'로 호출됐는지 확인
        assert len(call_kwargs) == 1
        assert call_kwargs[0]["limits"] == 1
        assert call_kwargs[0]["user_api"] == "blas"


# ---------------------------------------------------------------------------
# Scenario 1 (단위): moving_block_bootstrap_mean 재현성
# ---------------------------------------------------------------------------

class TestBootstrapReproducibility:
    def test_same_seed_gives_identical_output(self) -> None:
        """동일 seed → bootstrap 분포 bitwise 동일."""
        from src.domain.futures.strategy.tiered_workflow.metrics import moving_block_bootstrap_mean

        rng = np.random.default_rng(42)
        values = rng.standard_normal(50).astype(np.float64)
        idx = np.arange(50, dtype=np.int64)

        result1 = moving_block_bootstrap_mean(values, idx, block_bars=5, n_bootstrap=100, seed=99)
        result2 = moving_block_bootstrap_mean(values, idx, block_bars=5, n_bootstrap=100, seed=99)

        assert np.array_equal(result1, result2), "동일 seed에서 bootstrap 결과가 다름"

    def test_different_seeds_produce_different_output(self) -> None:
        """다른 seed → 다른 bootstrap 분포 (확률적으로 항상 다름)."""
        from src.domain.futures.strategy.tiered_workflow.metrics import moving_block_bootstrap_mean

        rng = np.random.default_rng(0)
        values = rng.standard_normal(60).astype(np.float64)
        idx = np.arange(60, dtype=np.int64)

        result_a = moving_block_bootstrap_mean(values, idx, block_bars=6, n_bootstrap=200, seed=1)
        result_b = moving_block_bootstrap_mean(values, idx, block_bars=6, n_bootstrap=200, seed=9999)

        assert not np.array_equal(result_a, result_b), "다른 seed인데 bootstrap 결과가 같음 (예상치 못한 충돌)"
