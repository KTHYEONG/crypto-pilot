# tests/unit/domain/futures/strategy/tiered_workflow/test_metrics.py
"""metrics.py 신규 함수(_sortino, _terminal_multiple) 단위테스트.

Scenarios:
    S1: _sortino happy path — 수기계산값과 일치.
    S2: _sortino 무손실 edge — dd=0 → 0.0 (inf 방어).
    S3: _terminal_multiple — 복리 배수 및 전손(prod<=0) 케이스.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.metrics import (
    _sortino,
    _terminal_multiple,
)

# ---------------------------------------------------------------------------
# S1: _sortino happy path
# ---------------------------------------------------------------------------

def test_sortino_happy_path_matches_hand_calculation() -> None:
    """S1: 알려진 downside dd 수기계산값과 일치 (rel=1e-4)."""
    # Arrange (Given)
    rets = [0.02, -0.01, 0.03, -0.02]
    bars_per_year = 2190.0
    expected = 14.798648586948742  # mean=0.005, dd=sqrt(mean([0.0001,0.0004]))

    # Act (When)
    result = _sortino(rets, bars_per_year=bars_per_year)

    # Assert (Then)
    assert result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# S2: _sortino 무손실 edge (inf 방어)
# ---------------------------------------------------------------------------

def test_sortino_returns_zero_when_no_losses() -> None:
    """S2: 전부 양수 수익률 → downside=empty → dd=0 → 0.0 반환 (inf 방어)."""
    # Arrange (Given)
    rets = [0.01, 0.02, 0.03]

    # Act (When)
    result = _sortino(rets)

    # Assert (Then)
    assert result == 0.0


def test_sortino_returns_zero_for_insufficient_data() -> None:
    """S2 boundary: 데이터 1개 미만(size<2) → 0.0."""
    # Arrange (Given)
    rets: list[float] = [0.01]

    # Act (When)
    result = _sortino(rets)

    # Assert (Then)
    assert result == 0.0


def test_sortino_handles_target_offset() -> None:
    """target!=0.0일 때 하방편차 기준점이 정확히 이동하는지 확인."""
    # Arrange (Given): target=0.01 → -0.01,0.0 모두 하방 후보 (r < target)
    rets = [0.02, -0.01, 0.03, -0.02]
    target = 0.01

    # Act (When)
    result = _sortino(rets, target=target)

    # Assert (Then): downside = [r for r in rets if r < target] = [-0.01, -0.02]
    downside = np.array([-0.01, -0.02])
    dd = float(np.sqrt(np.mean(np.square(downside - target))))
    mean_r = float(np.mean(rets))
    expected = (mean_r - target) / dd * np.sqrt(2190.0)
    assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# S3: _terminal_multiple
# ---------------------------------------------------------------------------

def test_terminal_multiple_compounds_correctly() -> None:
    """S3: [0.1, 0.1] → 1.1*1.1 = 1.21."""
    # Arrange (Given)
    rets = [0.1, 0.1]

    # Act (When)
    result = _terminal_multiple(rets)

    # Assert (Then)
    assert result == pytest.approx(1.21, rel=1e-9)


def test_terminal_multiple_returns_zero_on_total_loss() -> None:
    """S3: 전손(prod<=0) → 0.0 반환."""
    # Arrange (Given)
    rets = [-0.5, -1.0]

    # Act (When)
    result = _terminal_multiple(rets)

    # Assert (Then)
    assert result == 0.0


def test_terminal_multiple_returns_one_for_empty_array() -> None:
    """S3 edge: 빈 배열 → 1.0 (no-op multiple)."""
    # Arrange (Given)
    rets: list[float] = []

    # Act (When)
    result = _terminal_multiple(rets)

    # Assert (Then)
    assert result == 1.0
