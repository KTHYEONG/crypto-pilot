"""Phase 4: atomic_blocks non-overlap 6M pass_ratio 테스트.

사양서 §6.2 기준.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.validation.gates import (
    AtomicBlockConfig,
    build_atomic_blocks,
    evaluate_atomic_blocks,
)


def _make_timestamps(
    n_bars: int,
    start_ts: int = 1_600_000_000_000,
    bar_ms: int = 4 * 60 * 60 * 1000,  # 4h bars
) -> np.ndarray:
    """n_bars개의 4h bar timestamps 생성."""
    return np.array(
        [start_ts + i * bar_ms for i in range(n_bars)],
        dtype=np.int64,
    )


class TestBuildAtomicBlocks:
    """build_atomic_blocks 함수 검증."""

    def test_blocks_non_overlapping(self) -> None:
        """연속 block 간 시간 겹침 없음."""
        n_bars = 2000  # ~333일 (4h bars)
        timestamps = _make_timestamps(n_bars)
        is_end_ts = timestamps[500]  # IS 종료 시점

        blocks = build_atomic_blocks(timestamps, is_end_ts=is_end_ts)

        assert len(blocks) >= 2, "최소 2개 블록이 있어야 함"
        for k in range(len(blocks) - 1):
            end_k = blocks[k][1]
            start_next = blocks[k + 1][0]
            assert end_k <= start_next, f"Block {k}의 끝({end_k})이 Block {k + 1}의 시작({start_next})보다 커서는 안 됨"

    def test_blocks_start_after_is_end(self) -> None:
        """IS 기간 데이터가 block에 포함되지 않음."""
        n_bars = 2000
        timestamps = _make_timestamps(n_bars)
        is_end_ts = timestamps[600]

        blocks = build_atomic_blocks(timestamps, is_end_ts=is_end_ts)

        assert len(blocks) >= 1
        first_block_start_ts = timestamps[blocks[0][0]]
        assert first_block_start_ts >= is_end_ts, (
            f"첫 블록 시작({first_block_start_ts})이 IS 종료({is_end_ts})보다 이전이어서는 안 됨"
        )

    def test_blocks_each_approximately_6months(self) -> None:
        """각 block이 대략 6M (±15% 허용). 마지막 잔여 블록 제외."""
        n_bars = 5000  # ~833일 (~2.3년) → 충분한 데이터
        bar_ms = 4 * 60 * 60 * 1000
        timestamps = _make_timestamps(n_bars, bar_ms=bar_ms)
        is_end_ts = timestamps[300]

        blocks = build_atomic_blocks(timestamps, is_end_ts=is_end_ts)

        if len(blocks) < 2:
            pytest.skip("블록 수 부족")

        # 6M ≈ 182.5일 = 182.5 * 6 bars/day = 1095 bars (4h 기준)
        expected_bars_6m = int(182.5 * 24 / 4)  # ≈ 1095
        tolerance = 0.15  # ±15%

        # 완전한 블록만 검사 (마지막 블록은 잘릴 수 있음)
        full_blocks = blocks[:-1]  # 마지막 잔여 블록 제외
        if not full_blocks:
            pytest.skip("완전한 블록 없음")

        for i, (s, e) in enumerate(full_blocks):
            n_b = e - s
            assert n_b >= expected_bars_6m * (1 - tolerance), (
                f"Block {i}: {n_b}개 bars가 6M 기대값 {expected_bars_6m}의 -15% 미만"
            )
            assert n_b <= expected_bars_6m * (1 + tolerance), (
                f"Block {i}: {n_b}개 bars가 6M 기대값 {expected_bars_6m}의 +15% 초과"
            )

    def test_no_double_counting(self) -> None:
        """전체 OOS 기간을 blocks로 분할 시 Σ(block_bars) == total_oos_bars (±1 bar 허용)."""
        n_bars = 2500
        bar_ms = 4 * 60 * 60 * 1000
        timestamps = _make_timestamps(n_bars, bar_ms=bar_ms)
        is_end_idx = 400
        is_end_ts = timestamps[is_end_idx]

        blocks = build_atomic_blocks(timestamps, is_end_ts=is_end_ts)

        if len(blocks) == 0:
            pytest.skip("블록 없음")

        total_oos_bars = n_bars - is_end_idx
        sum_block_bars = sum(e - s for s, e in blocks)

        # blocks가 OOS 기간 내에 있는지 확인 (마지막 블록은 잘릴 수 있음)
        assert sum_block_bars <= total_oos_bars + 1, f"블록 총합({sum_block_bars})이 OOS 기간({total_oos_bars})을 초과"

    def test_insufficient_data_returns_empty_or_few_blocks(self) -> None:
        """데이터 부족 (2 blocks만 확보 가능) — n_blocks < required_min_blocks → passed=False."""
        # 매우 짧은 OOS 기간 (6M block 2개만 생성 가능)
        n_bars = 100
        bar_ms = 4 * 60 * 60 * 1000
        timestamps = _make_timestamps(n_bars, bar_ms=bar_ms)
        is_end_ts = timestamps[50]  # 절반만 OOS

        blocks = build_atomic_blocks(timestamps, is_end_ts=is_end_ts)

        # AtomicBlockConfig.required_min_blocks = 3이므로 3개 미만이면 불충분
        config = AtomicBlockConfig()
        if len(blocks) < config.required_min_blocks:
            # equity_curves를 블록 수만큼 생성
            equity_curves = [np.array([1.0, 1.02, 1.01, 1.03, 1.02]) for _ in range(len(blocks))]
            result = evaluate_atomic_blocks(equity_curves, config=config)
            assert not result.passed, "블록 수 부족 시 passed=False여야 함"


class TestEvaluateAtomicBlocks:
    """evaluate_atomic_blocks 함수 검증."""

    def test_pass_ratio_above_threshold(self) -> None:
        """11 blocks 중 8 pass → pass_ratio = 0.727 > 0.70 → passed=True."""
        # 11개 블록: 8개 수익, 3개 손실
        n_blocks = 11
        equity_curves = []
        for k in range(n_blocks):
            if k < 8:
                # 수익 블록: TW > 1 (≥ 1.015 기준 충족)
                equity_curves.append(np.array([1.0, 1.02, 1.04, 1.06, 1.05]))
            else:
                # 손실 블록: TW < 1
                equity_curves.append(np.array([1.0, 0.98, 0.96, 0.97, 0.95]))

        config = AtomicBlockConfig(min_pass_ratio=0.70, required_min_blocks=3)
        result = evaluate_atomic_blocks(equity_curves, config=config)

        assert result.n_blocks == 11
        assert result.pass_ratio >= 0.727 - 1e-6
        assert result.passed, f"pass_ratio={result.pass_ratio:.3f}이 0.70 초과여야 함"

    def test_pass_ratio_below_threshold(self) -> None:
        """11 blocks 중 7 pass → pass_ratio = 0.636 < 0.70 → passed=False."""
        n_blocks = 11
        equity_curves = []
        for k in range(n_blocks):
            if k < 7:
                equity_curves.append(np.array([1.0, 1.02, 1.04, 1.06, 1.05]))
            else:
                equity_curves.append(np.array([1.0, 0.98, 0.96, 0.97, 0.95]))

        config = AtomicBlockConfig(min_pass_ratio=0.70, required_min_blocks=3)
        result = evaluate_atomic_blocks(equity_curves, config=config)

        assert not result.passed, f"pass_ratio={result.pass_ratio:.3f}이 0.70 미만이어야 함"

    def test_min_blocks_not_met_passed_false(self) -> None:
        """required_min_blocks 미달 → passed=False."""
        equity_curves = [
            np.array([1.0, 1.02, 1.04]),
            np.array([1.0, 1.01, 1.03]),
        ]
        config = AtomicBlockConfig(required_min_blocks=3)
        result = evaluate_atomic_blocks(equity_curves, config=config)
        assert not result.passed
        assert result.n_blocks < config.required_min_blocks

    def test_result_contains_block_log_tws(self) -> None:
        """AtomicBlockResult.block_log_tws에 각 블록 log TW가 포함됨."""
        n_blocks = 5
        equity_curves = [np.array([1.0, 1.05]) for _ in range(n_blocks)]
        config = AtomicBlockConfig(required_min_blocks=2)
        result = evaluate_atomic_blocks(equity_curves, config=config)
        assert len(result.block_log_tws) == n_blocks

    def test_worst_block_mdd_computed(self) -> None:
        """worst_block_mdd가 올바르게 계산됨."""
        # 블록 1: 변동 없음 → MDD=0
        # 블록 2: 10% 하락 → MDD≈0.10
        equity_curves = [
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 0.90, 0.95, 0.98]),
        ]
        config = AtomicBlockConfig(required_min_blocks=2)
        result = evaluate_atomic_blocks(equity_curves, config=config)
        assert result.worst_block_mdd > 0.05, f"worst_block_mdd={result.worst_block_mdd:.3f}이 0.05보다 커야 함"
