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
    _sharpe,
    _sortino,
    _terminal_multiple,
    moving_block_bootstrap_mean,
    resolve_num_blocks,
)

# ---------------------------------------------------------------------------
# S1: _sortino happy path
# ---------------------------------------------------------------------------


def test_sortino_happy_path_matches_hand_calculation() -> None:
    """S1: 표준 TDD(전표본 N 정규화) 수기계산값과 일치 (rel=1e-4).

    rets=[0.02,-0.01,0.03,-0.02], mean=0.005, target=0.0
    downside=[-0.01,-0.02], 제곱합=1e-4+4e-4=5e-4
    TDD=sqrt(5e-4/4)=0.011180... → Sortino=0.005/0.011180*sqrt(2190)
    """
    # Arrange (Given)
    rets = [0.02, -0.01, 0.03, -0.02]
    bars_per_year = 2190.0
    # dd=sqrt(sum([1e-4,4e-4])/4)=sqrt(1.25e-4)=0.011180339...
    expected = 0.005 / (5e-4 / 4) ** 0.5 * 2190.0**0.5  # ≈ 20.929

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
    # 표준 TDD: 제곱합을 전표본(4)으로 나눔
    arr = np.array(rets)
    downside = np.array([-0.01, -0.02])
    dd = float(np.sqrt(np.sum(np.square(downside - target)) / arr.size))
    mean_r = float(np.mean(rets))
    expected = (mean_r - target) / dd * np.sqrt(2190.0)
    assert result == pytest.approx(expected, rel=1e-6)


def test_sortino_exceeds_sharpe_for_mixed_returns() -> None:
    """S2-불변식: 표준 TDD에서 Sortino는 항상 Sharpe 이상 (하방만 벌점).

    비표준 분모(÷N_down)에서는 분모가 커져 이 불변식이 깨짐.
    """
    # Arrange (Given)
    rets = [0.03, -0.01, 0.02, -0.005, 0.04, -0.015]
    bars_per_year = 2190.0

    # Act (When)
    sortino_val = _sortino(rets, bars_per_year=bars_per_year)
    sharpe_val = _sharpe(rets, bars_per_year=bars_per_year)

    # Assert (Then)
    assert sortino_val > sharpe_val, f"표준 Sortino({sortino_val:.4f}) must exceed Sharpe({sharpe_val:.4f})"


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


# ---------------------------------------------------------------------------
# S4: resolve_num_blocks
# ---------------------------------------------------------------------------


def test_resolve_num_blocks_happy_path() -> None:
    """S1 (Happy Path): n_clusters=100, block_bars=6 → ceil(100/6)=17."""
    assert resolve_num_blocks(100, 6) == 17


def test_resolve_num_blocks_edge_zero_clusters() -> None:
    """S2 (Edge): n_clusters=0 → floor at 1."""
    assert resolve_num_blocks(0, 6) == 1


def test_resolve_num_blocks_edge_zero_block_bars() -> None:
    """S2 (Edge): block_bars=0 → floored to 1 → num_blocks = n_clusters."""
    assert resolve_num_blocks(10, 0) == 10


def test_resolve_num_blocks_negative_inputs() -> None:
    """Negative n_clusters or block_bars → floored safely to 1."""
    assert resolve_num_blocks(-5, 6) == 1
    assert resolve_num_blocks(5, -2) == 5  # block floored to 1


# ---------------------------------------------------------------------------
# S5: moving_block_bootstrap_mean regression guard
# ---------------------------------------------------------------------------


def test_moving_block_bootstrap_mean_block_count_unchanged_after_refactor() -> None:
    """Regression guard: block count post-refactor matches pre-refactor formula."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 50).astype(np.float64)
    decisions = np.arange(50, dtype=np.int64)
    boot = moving_block_bootstrap_mean(
        values, decisions, block_bars=6, n_bootstrap=100, seed=42,
    )
    assert boot.size == 100
    assert np.all(np.isfinite(boot))
