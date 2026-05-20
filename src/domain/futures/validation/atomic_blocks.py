"""Non-overlapping 6M atomic block validation.

IS 이후 OOS 기간을 6M non-overlap blocks으로 분할하여 pass_ratio를 검증한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# 6M ≈ 182.5일 (밀리초 단위)
_MS_PER_DAY: int = 86_400_000
_DAYS_PER_6M: float = 182.5
_6M_MS: int = int(_DAYS_PER_6M * _MS_PER_DAY)


@dataclass(frozen=True)
class AtomicBlockConfig:
    """Atomic block 평가 설정.

    Attributes:
        block_months: 블록 기간 (고정 6M).
        min_pass_ratio: 통과 블록 비율 최소값.
        required_min_blocks: 판정을 위한 최소 block 수.
    """

    block_months: int = 6
    min_pass_ratio: float = 0.70
    required_min_blocks: int = 3


@dataclass
class AtomicBlockResult:
    """Atomic block 평가 결과.

    Attributes:
        n_blocks: 총 블록 수.
        n_passed: 통과 블록 수.
        pass_ratio: n_passed / n_blocks.
        passed: pass_ratio >= min_pass_ratio AND n_blocks >= required_min_blocks.
        block_log_tws: 각 블록의 log Terminal Wealth.
        worst_block_mdd: 최악 블록의 최대 낙폭 (0~1 scale).
        median_log_growth: 중앙값 log TW.
    """

    n_blocks: int
    n_passed: int
    pass_ratio: float
    passed: bool
    block_log_tws: list[float]
    worst_block_mdd: float
    median_log_growth: float


def build_atomic_blocks(
    timestamps: np.ndarray,
    is_end_ts: int,
    block_months: int = 6,
) -> list[tuple[int, int]]:
    """Non-overlapping 6M 시작/끝 인덱스 쌍 반환.

    IS 이후 타임스탬프부터 block_months 단위로 non-overlap 분할한다.

    Args:
        timestamps: decision bar timestamps (UTC unix ms), shape [T].
        is_end_ts: IS 종료 시점 (이 값 이상인 bar부터 OOS).
        block_months: 블록 길이 (기본 6).

    Returns:
        list of (start_idx, end_idx) — end_idx는 exclusive.
    """
    ts = np.asarray(timestamps, dtype=np.int64)
    n = int(ts.size)
    if n == 0:
        return []

    # IS 종료 이후 첫 인덱스 탐색
    oos_start_idx = int(np.searchsorted(ts, is_end_ts, side="left"))
    if oos_start_idx >= n:
        return []

    # 6M 밀리초
    block_ms = int(_DAYS_PER_6M * _MS_PER_DAY * (block_months / 6))

    blocks: list[tuple[int, int]] = []
    cur_start_idx = oos_start_idx

    while cur_start_idx < n:
        block_end_ts = ts[cur_start_idx] + block_ms
        # block_end_ts 이상인 첫 인덱스
        cur_end_idx = int(np.searchsorted(ts, block_end_ts, side="left"))

        if cur_end_idx > cur_start_idx:
            blocks.append((cur_start_idx, cur_end_idx))
        elif cur_end_idx == cur_start_idx:
            # 전진 불가 → 종료
            break

        cur_start_idx = cur_end_idx

    return blocks


def _calc_block_log_tw(equity_curve: np.ndarray) -> float:
    """블록 equity curve에서 log Terminal Wealth 계산."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    start = float(eq[0])
    end = float(eq[-1])
    if start <= 1e-15:
        return -10.0
    ratio = end / start
    if ratio <= 1e-15:
        return -10.0
    return float(math.log(ratio))


def _calc_block_mdd(equity_curve: np.ndarray) -> float:
    """블록 equity curve의 최대 낙폭 (0~1)."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    running_max = np.where(running_max < 1e-15, 1e-15, running_max)
    dd = (eq - running_max) / running_max
    return float(abs(np.min(dd)))


def evaluate_atomic_blocks(
    equity_curves: list[np.ndarray],
    config: AtomicBlockConfig = AtomicBlockConfig(),
) -> AtomicBlockResult:
    """블록별 equity curve를 평가하여 AtomicBlockResult 반환.

    Args:
        equity_curves: 각 block별 equity curve. len == n_blocks.
        config: AtomicBlockConfig 설정.

    Returns:
        AtomicBlockResult with pass/fail 판정.
    """
    n_blocks = len(equity_curves)

    if n_blocks < config.required_min_blocks:
        return AtomicBlockResult(
            n_blocks=n_blocks,
            n_passed=0,
            pass_ratio=0.0,
            passed=False,
            block_log_tws=[],
            worst_block_mdd=0.0,
            median_log_growth=0.0,
        )

    block_log_tws: list[float] = []
    block_mdds: list[float] = []
    n_passed = 0

    for ec in equity_curves:
        log_tw = _calc_block_log_tw(ec)
        mdd = _calc_block_mdd(ec)
        block_log_tws.append(log_tw)
        block_mdds.append(mdd)
        if math.exp(log_tw) >= 1.0:
            n_passed += 1

    pass_ratio = float(n_passed) / float(n_blocks) if n_blocks > 0 else 0.0
    worst_mdd = float(max(block_mdds)) if block_mdds else 0.0
    arr_log_tw = np.array(block_log_tws, dtype=np.float64)
    median_log = float(np.median(arr_log_tw)) if arr_log_tw.size > 0 else 0.0

    passed = pass_ratio >= config.min_pass_ratio

    return AtomicBlockResult(
        n_blocks=n_blocks,
        n_passed=n_passed,
        pass_ratio=pass_ratio,
        passed=passed,
        block_log_tws=block_log_tws,
        worst_block_mdd=worst_mdd,
        median_log_growth=median_log,
    )
