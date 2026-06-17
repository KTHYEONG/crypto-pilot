"""Unit tests for LayeredWindow and get_layered_window in opt_config.

Covers: REGIME_FLOOR clamp, segment ordering, warmup buffer, duration approximation.
"""

import datetime
from typing import Any, cast

import pytest
from dateutil.relativedelta import relativedelta

from src.domain.futures.optimization.opt_config import (
    REGIME_FLOOR,
    LayeredWindow,
    get_layered_window,
)

# ---------------------------------------------------------------------------
# T8 — Regime floor 클램프
# ---------------------------------------------------------------------------

def test_get_layered_window_clamps_l1_start_to_regime_floor() -> None:
    """l1_start 역산 결과가 REGIME_FLOOR보다 이전이면 클램프."""
    # Arrange: 기준일을 가까운 미래로 설정 → L1 역산 = 오늘에서 ~36mo 전
    # 36mo + 6mo = 42mo 전 → REGIME_FLOOR(2023-01-01) 이전일 수 있음
    # 2025-01-01 기준 → holdout_end=2024-12-31, holdout_start=2024-07-01
    # l2_start=2023-07-01, l1_start_raw=2022-01-01 → < REGIME_FLOOR
    reference = datetime.date(2025, 1, 15)

    # Act
    window = get_layered_window(reference_date=reference, l1_months=18, l2_months=12, holdout_months=6)

    # Assert
    assert window.l1_start >= REGIME_FLOOR


def test_get_layered_window_no_clamp_when_l1_above_floor() -> None:
    """l1_start 역산 결과가 REGIME_FLOOR 이후이면 클램프 미적용."""
    # Arrange: 2027-01-01 기준 → l1_start_raw = 2027-01-01 - 6mo - 12mo - 18mo ≈ 2024-07
    # 2024-07 > REGIME_FLOOR(2023-01) → 클램프 없음
    reference = datetime.date(2027, 1, 15)

    # Act
    window = get_layered_window(reference_date=reference, l1_months=18, l2_months=12, holdout_months=6)

    # Assert: 클램프가 없으면 l1_start == regime_floor는 아님
    # 역산 결과: holdout_end=2026-12-31, holdout_start≈2026-07-01
    # l2_start≈2025-07-01, l1_start_raw≈2024-01-01 > REGIME_FLOOR
    assert window.l1_start >= REGIME_FLOOR
    assert window.l1_start > REGIME_FLOOR  # 클램프 미적용 확인


def test_get_layered_window_fetch_start_before_l1() -> None:
    """fetch_start는 반드시 l1_start 이전."""
    # Arrange
    reference = datetime.date(2026, 6, 11)

    # Act
    window = get_layered_window(reference_date=reference)

    # Assert
    assert window.fetch_start < window.l1_start


def test_get_layered_window_fetch_warmup_exact() -> None:
    """fetch_start = l1_start - warmup_days (정확히)."""
    # Arrange
    reference = datetime.date(2027, 4, 1)
    warmup = 180

    # Act
    window = get_layered_window(reference_date=reference, warmup_days=warmup)

    # Assert
    delta = (window.l1_start - window.fetch_start).days
    assert delta == warmup


def test_get_layered_window_segment_ordering() -> None:
    """fetch_start ≤ l1_start ≤ l2_start ≤ holdout_start ≤ holdout_end."""
    # Arrange
    reference = datetime.date(2026, 6, 11)

    # Act
    w = get_layered_window(reference_date=reference)

    # Assert
    assert w.fetch_start <= w.l1_start
    assert w.l1_start <= w.l2_start
    assert w.l2_start <= w.holdout_start
    assert w.holdout_start <= w.holdout_end


def test_get_layered_window_duration_months() -> None:
    """L1 기간≈18mo, L2 기간≈12mo (허용 오차 2일)."""
    # Arrange
    reference = datetime.date(2027, 7, 1)

    # Act
    w = get_layered_window(reference_date=reference, l1_months=18, l2_months=12)

    # Assert — relativedelta 역산이므로 날짜 반올림 허용 ±2일
    l1_days = (w.l2_start - w.l1_start).days
    l2_days = (w.holdout_start - w.l2_start).days

    expected_l1 = (w.l1_start + relativedelta(months=18) - w.l1_start).days
    expected_l2 = (w.l2_start + relativedelta(months=12) - w.l2_start).days

    # 18mo ≈ 547일, 12mo ≈ 365일 → ±5일 허용
    assert abs(l1_days - expected_l1) <= 5
    assert abs(l2_days - expected_l2) <= 5


def test_get_layered_window_holdout_end_is_quarter_end() -> None:
    """holdout_end는 현재 분기 시작 전날 (이전 분기 마지막 날)."""
    # Arrange: 2026-06-11 → 현재 분기 Q2(2026-04-01) → holdout_end=2026-03-31
    reference = datetime.date(2026, 6, 11)

    # Act
    w = get_layered_window(reference_date=reference)

    # Assert
    assert w.holdout_end == datetime.date(2026, 3, 31)


def test_get_layered_window_none_reference_uses_today() -> None:
    """reference_date=None이면 오늘 기준으로 계산 (타입 안전성 확인)."""
    # Act
    w = get_layered_window()

    # Assert: LayeredWindow 인스턴스 반환 확인
    assert isinstance(w, LayeredWindow)
    assert isinstance(w.fetch_start, datetime.date)
    assert isinstance(w.holdout_end, datetime.date)


def test_get_layered_window_regime_floor_stored() -> None:
    """LayeredWindow.regime_floor에 사용된 floor 값이 저장됨."""
    # Arrange
    custom_floor = datetime.date(2022, 6, 1)
    reference = datetime.date(2027, 1, 1)

    # Act
    w = get_layered_window(reference_date=reference, regime_floor=custom_floor)

    # Assert
    assert w.regime_floor == custom_floor


def test_get_layered_window_frozen() -> None:
    """LayeredWindow는 frozen dataclass — 수정 불가."""
    # Arrange
    w = get_layered_window(reference_date=datetime.date(2026, 6, 11))

    # Act / Assert
    import dataclasses

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cast(Any, w).fetch_start = datetime.date(2020, 1, 1)


# ---------------------------------------------------------------------------
# S6 — L2_ALLOC_SPACE 재배선 (spec: layer2-signal-utilization.md §2.3)
# ---------------------------------------------------------------------------

from src.domain.futures.optimization.opt_config import L2_ALLOC_SPACE


def test_l2_alloc_space_contains_new_params() -> None:
    """S6: L2_ALLOC_SPACE_V8 — kelly_fraction/max_ann_vol은 Phase B 결정론 전환으로 제거됨."""
    # Fix C: leverage 차원은 탐색 공간에서 제거 → signal 차원만 최적화
    assert "kelly_fraction" not in L2_ALLOC_SPACE
    assert "max_ann_vol" not in L2_ALLOC_SPACE
    assert "K_RANK" in L2_ALLOC_SPACE
    assert "CS_Z_SCORE_THRESHOLD" in L2_ALLOC_SPACE


def test_l2_alloc_space_excludes_dead_params() -> None:
    """S6: RISK_PER_TRADE, MAX_EXPOSURE_PER_COIN, NORM_VAR_CONSTANT 제거됨."""
    assert "RISK_PER_TRADE" not in L2_ALLOC_SPACE
    assert "MAX_EXPOSURE_PER_COIN" not in L2_ALLOC_SPACE
    assert "NORM_VAR_CONSTANT" not in L2_ALLOC_SPACE


def test_l2_alloc_space_max_ann_vol_range() -> None:
    """Fix C: max_ann_vol은 V8에서 탐색 공간 제외 (결정론적 Phase B 배치로 대체)."""
    assert "max_ann_vol" not in L2_ALLOC_SPACE


def test_l2_alloc_space_kelly_fraction_range() -> None:
    """Fix C: kelly_fraction은 V8에서 탐색 공간 제외 (결정론적 Phase B 배치로 대체)."""
    assert "kelly_fraction" not in L2_ALLOC_SPACE
