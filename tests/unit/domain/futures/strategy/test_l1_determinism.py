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
# Scenario 1 (단위): moving_block_bootstrap_mean 재현성 및 가속 확인
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

    def test_bootstrap_numba_performance(self) -> None:
        """Numba 가속 후 속도가 충분히 빠른지 검증 (웜업 포함)."""
        import time

        from src.domain.futures.strategy.tiered_workflow.metrics import moving_block_bootstrap_mean

        rng = np.random.default_rng(123)
        values = rng.standard_normal(100).astype(np.float64)
        idx = np.arange(100, dtype=np.int64)

        # 웜업 컴파일용 호출
        moving_block_bootstrap_mean(values, idx, block_bars=5, n_bootstrap=10, seed=1)

        t0 = time.perf_counter()
        # 대량 연산 실행
        moving_block_bootstrap_mean(values, idx, block_bars=5, n_bootstrap=5000, seed=1)
        duration = time.perf_counter() - t0

        # 대량 연산이 Numba 덕에 매우 순식간(0.2초 이내)에 끝나야 함
        assert duration < 0.2, f"Bootstrap execution is too slow: {duration:.4f}s"


# ---------------------------------------------------------------------------
# Scenario 3 (캐시): 캐시 프라이밍 및 중복 연산 방지 검증
# ---------------------------------------------------------------------------

class TestFeatureCachePriming:
    def test_prime_aligned_feature_cache_populates_global_cache(self) -> None:
        """prime_aligned_feature_cache 호출 시 _ALIGNED_FEATURE_CACHE에 정상 등록 확인."""
        import pandas as pd

        from src.domain.futures.strategy.candidate_dataset import _ALIGNED_FEATURE_CACHE, prime_aligned_feature_cache
        from src.domain.futures.strategy.common.alignment import AlignedMarketData
        from src.domain.futures.strategy.config import CandidateStrategyConfig

        # mock aligned data
        size = 100
        arr = np.ones((size, 1), dtype=np.float64)
        aligned = AlignedMarketData(
            symbols=("BTC",),
            datetimes=np.array(pd.date_range("2026-01-01", periods=size, freq="4h")),
            open_2d=arr, high_2d=arr, low_2d=arr, close_2d=arr, volume_2d=arr,
            funding_2d=np.zeros((size, 1)),
            active_mask=np.ones((size, 1), dtype=bool),
            warm_mask=np.ones((size, 1), dtype=bool),
            entry_block_mask=np.zeros((size, 1), dtype=bool),
            kill_mask=np.zeros((size, 1), dtype=bool),
        )

        labeled_events = pd.DataFrame({
            "entry_idx": [25, 50, 75],
            "exit_idx": [30, 60, 80],
            "symbol": ["BTC", "BTC", "BTC"],
            "side": [1, -1, 1],
            "score": [0.5, -0.2, 0.8]
        })

        cfg = CandidateStrategyConfig(market_state_features_enabled=True)

        aligned_id = id(aligned)
        if aligned_id in _ALIGNED_FEATURE_CACHE:
            del _ALIGNED_FEATURE_CACHE[aligned_id]

        prime_aligned_feature_cache(labeled_events, aligned, cfg)

        # Assert: 캐시 엔트리가 생성되었고 핵심 피처가 캐싱되었는지 확인
        assert aligned_id in _ALIGNED_FEATURE_CACHE
        cache = _ALIGNED_FEATURE_CACHE[aligned_id]
        assert "sym_ret_1" in cache
        assert "btc_ret_1_ser" in cache


# ---------------------------------------------------------------------------
# Scenario 4: PERF(15) 로그 커버리지 검증
# ---------------------------------------------------------------------------

class TestPerfLogCoverage:
    """L1 PERF 로그 출력 검증: 병목탐지용 로그가 정상 emit되는지 확인."""

    def test_perf_log_emit_in_compute_symbol_strategy_evidence(self, caplog: pytest.LogCaptureFixture) -> None:
        """compute_symbol_strategy_evidence에서 [SIGNAL-EVIDENCE] PERF 로그가 emit되는지 검증."""
        import pandas as pd

        from src.core.utils.utils import PERF
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.tiered_workflow.signal_selection import (
            compute_symbol_strategy_evidence,
        )

        caplog.set_level(PERF)

        cfg = CandidateStrategyConfig(
            l1_pair_min_effective_obs=3,
            l1_pair_min_folds=1,
            l1_pair_min_mean_gross_bps=-9999,
            l1_pair_min_incremental_bps=-9999,
            l1_pair_min_positive_fold_ratio=0.0,
            l1_pair_fdr_alpha=0.25,
            l1_bootstrap_block_bars=2,
            l1_bootstrap_samples=10,
            l1_quality_weight_enabled=False,
        )
        event_results = pd.DataFrame({
            "symbol": ["BTC", "ETH"],
            "strategy_id": ["mom:a", "mom:b"],
            "activation_context": ["all", "all"],
            "fold_id": [0, 0],
            "expected_holding_bars": [1, 1],
            "gross_event_bps": [10.0, 5.0],
            "incremental_bps": [8.0, 3.0],
            "uniqueness_weight": [1.0, 1.0],
            "decision_idx": [0, 0],
            "batch_size": [1, 1],
        })

        result = compute_symbol_strategy_evidence(
            event_results=event_results,
            cfg=cfg,
            seed=42,
            registry_as_of_idx=0,
        )

        assert len(result) > 0
        assert any("[SIGNAL-EVIDENCE]" in rec.message for rec in caplog.records)
        assert any("n_pairs=" in rec.message for rec in caplog.records)
        assert any("prep=" in rec.message for rec in caplog.records)

    def test_awf_perf_log_migrated_from_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """_run_awf_simulation 종료 시 [AWF-PERF] 로그가 PERF 레벨로 emit되는지 검증."""
        from unittest.mock import MagicMock

        import numpy as np

        from src.core.utils.utils import PERF
        from src.domain.futures.strategy.tiered_workflow import _run_awf_simulation
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2AllocationConfig,
        )

        caplog.set_level(PERF)

        cache = MagicMock()
        cache.tradeable_mask_2d = np.zeros((5, 2), dtype=bool)
        cache.signal_mask_2d = np.zeros((5, 2), dtype=bool)
        cache.hurdle_2d = np.zeros((5, 2), dtype=np.float64)
        cache.beta_1d = np.zeros(2, dtype=np.float64)
        cache.vol_matrix_2d = np.ones((5, 2), dtype=np.float64) * 0.02
        cache.expected_gross_bps_2d = np.zeros((5, 2), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((5, 2), dtype=np.float64)
        cache.holding_bars_2d = np.ones((5, 2), dtype=np.float64)
        cache.side_2d = np.ones((5, 2), dtype=np.float64)
        cache.quality_weight_2d = np.ones((5, 2), dtype=np.float64)
        cache.funding_2d = np.zeros((5, 2), dtype=np.float64)

        aligned = MagicMock()
        aligned.symbols = ("BTC", "ETH")
        aligned.close_2d = np.ones((6, 2), dtype=np.float64) * 100.0
        aligned.datetimes = np.array([f"2024-01-0{d}T00:00:00" for d in range(1, 7)], dtype="datetime64[ns]")

        from src.domain.futures.strategy.walk_forward import WFFold
        fold = WFFold(
            fit_start=0, fit_end=1,
            cal_start=0, cal_end=1,
            oos_start=1, oos_end=5,
        )
        awf_folds = (fold,)

        config = Layer2AllocationConfig(
            k_rank=2, rank_buffer=0, kelly_fraction=0.5,
            no_trade_band=0.0, rebalance_bars=1,
        )

        from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps

        result = _run_awf_simulation(
            signal_batch=MagicMock(),
            awf_folds=awf_folds,
            aligned=aligned,
            cache=cache,
            config=config,
            caps=PortfolioCaps(),
            tf="4h",
        )

        assert result is not None
        perf_messages = [rec.message for rec in caplog.records if rec.levelno == PERF]
        assert any("[AWF-PERF]" in m for m in perf_messages), (
            f"[AWF-PERF] not found in PERF logs. Got: {perf_messages}"
        )
        assert any("[AWF-FOLD]" in m for m in perf_messages), (
            f"[AWF-FOLD] not found in PERF logs. Got: {perf_messages}"
        )
        assert not any("L2-AWF-PROF" in m for m in perf_messages), (
            "Legacy [L2-AWF-PROF] DEBUG log should be migrated to PERF"
        )

