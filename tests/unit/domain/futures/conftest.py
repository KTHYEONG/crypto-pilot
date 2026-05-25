"""공통 pytest 픽스처 — futures 단위 테스트 전용."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def flat_market() -> dict:
    """가격 불변, 펀딩 없는 기본 환경."""
    n_bars, n_syms = 200, 5
    price = 100.0
    return {
        "open_1m": np.full((n_bars * 240, n_syms), price),
        "high_1m": np.full((n_bars * 240, n_syms), price * 1.001),
        "low_1m": np.full((n_bars * 240, n_syms), price * 0.999),
        "close_1m": np.full((n_bars * 240, n_syms), price),
        "mark_1m": np.full((n_bars * 240, n_syms), price),
        "funding_mask": np.zeros((n_bars * 240, n_syms), dtype=np.int8),
        "funding_rate": np.zeros((n_bars * 240, n_syms)),
        "kill": np.zeros((n_bars, n_syms)),
        "n_bars_4h": n_bars,
        "n_syms": n_syms,
    }


@pytest.fixture
def luna_crash_scenario() -> dict:
    """LUNA 2022-05-09 급락 재현 (mark vs last 괴리 시나리오).

    mark_price는 last_price보다 높게 유지 (실제 LUNA 현상).
    """
    n = 2880  # 48시간 x 60
    price = np.linspace(80.0, 0.01, n)
    mark = np.clip(price * 1.05, 0.01, None)  # mark > last
    return {"price_1m": price, "mark_1m": mark}


@pytest.fixture
def leg_log_tw_healthy() -> np.ndarray:
    """K=8 legs, 모두 양수, hard gate 모두 통과하는 기본 케이스."""
    return np.array([0.04, 0.06, 0.03, 0.05, 0.07, 0.04, 0.05, 0.06])


@pytest.fixture
def leg_log_tw_borderline() -> np.ndarray:
    """경계값: min_positive_leg_ratio=0.55 경계 (5/8=0.625)."""
    return np.array([0.04, -0.01, 0.03, 0.05, -0.02, 0.04, 0.05, 0.06])
