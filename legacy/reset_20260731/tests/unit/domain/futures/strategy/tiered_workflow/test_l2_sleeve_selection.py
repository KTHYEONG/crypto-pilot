from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    compute_per_sleeve_realized_edge,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_cache(
    signal_mask_2d: NDArray[np.bool_],
    side_2d: NDArray[np.float64],
    sleeve_to_sym: NDArray[np.int64],
) -> MagicMock:
    cache = MagicMock(spec=L2SimulationCache)
    cache.signal_mask_2d = signal_mask_2d
    cache.side_2d = side_2d
    cache.sleeve_to_sym = sleeve_to_sym
    return cache


def _make_aligned(close_2d: NDArray[np.float64]) -> MagicMock:
    aligned = MagicMock()
    aligned.close_2d = close_2d
    return aligned


# ─── S1 — compute_per_sleeve_realized_edge 방향정합 ─────────────────────────


class TestComputePerSleeveRealizedEdgeDirectional:
    """S1: 실현엣지 방향정합."""

    def test_directional_sign(self) -> None:
        """sleeve A(sym 상승, side=+1) > 0 > sleeve B(sym 하락, side=+1)."""
        T, N, S = 5, 2, 2
        mask = np.zeros((T, S), dtype=np.bool_)
        mask[1:4, :] = True
        side = np.zeros((T, S), dtype=np.float64)
        side[1:4, :] = 1.0
        sleeve_to_sym = np.array([0, 1], dtype=np.int64)

        close = np.zeros((T, N), dtype=np.float64)
        close[:, 0] = [100.0, 101.0, 102.0, 103.0, 104.0]
        close[:, 1] = [100.0, 99.0, 98.0, 97.0, 96.0]

        cache = _make_cache(mask, side, sleeve_to_sym)
        aligned = _make_aligned(close)

        edge = compute_per_sleeve_realized_edge(cache, aligned, 0, T)

        assert edge[0] > 0.0, f"edge[0]={edge[0]} should be positive"
        assert edge[1] < 0.0, f"edge[1]={edge[1]} should be negative"
        assert np.isfinite(edge[0])
        assert np.isfinite(edge[1])


# ─── S2 — 빈 active / t+1 경계 ──────────────────────────────────────────────


class TestPerSleeveRealizedEdgeEmptyAndBoundary:
    """S2: active 0 sleeve → NaN, 마지막 bar 경계 예외 없음."""

    def test_empty_active_returns_nan(self) -> None:
        """sleeve C(active 0개) → NaN."""
        T, N, S = 5, 1, 3
        mask = np.zeros((T, S), dtype=np.bool_)
        mask[1:4, 0] = True
        mask[1:4, 1] = True
        mask[:, 2] = False

        side = np.zeros((T, S), dtype=np.float64)
        side[1:4, 0] = 1.0
        side[1:4, 1] = 1.0

        sleeve_to_sym = np.array([0, 0, 0], dtype=np.int64)
        close = np.full((T, N), 100.0, dtype=np.float64)
        close[1:5, 0] = [101.0, 102.0, 103.0, 104.0]

        cache = _make_cache(mask, side, sleeve_to_sym)
        aligned = _make_aligned(close)

        edge = compute_per_sleeve_realized_edge(cache, aligned, 0, T)

        assert np.isfinite(edge[0]), f"edge[0]={edge[0]} should be finite"
        assert np.isfinite(edge[1]), f"edge[1]={edge[1]} should be finite"
        assert np.isnan(edge[2]), f"edge[2]={edge[2]} should be NaN"

    def test_last_bar_boundary_no_exception(self) -> None:
        """마지막 bar(t=T-1)에서 t+1>=T → skip, 예외 없음."""
        T, S = 3, 1
        mask = np.ones((T, S), dtype=np.bool_)
        side = np.ones((T, S), dtype=np.float64)
        sleeve_to_sym = np.zeros(S, dtype=np.int64)
        close = np.array([[100.0], [101.0], [102.0]], dtype=np.float64)

        cache = _make_cache(mask, side, sleeve_to_sym)
        aligned = _make_aligned(close)

        edge = compute_per_sleeve_realized_edge(cache, aligned, 0, T)
        assert np.isfinite(edge[0])


# ─── S3 — realized_ic 양/음 분별 ───────────────────────────────────────────


class TestSleeveRealizedIcSign:
    """S3: spearman IC 부호 정확성."""

    def _build_cache_with_edges(
        self,
        fit_edges: list[float],
        oos_edges: list[float],
    ) -> tuple[MagicMock, MagicMock, int, int, int, int]:
        """fit_edges[i]와 oos_edges[i]를 forward_ret으로 실현하는 cache+aligned.

        각 sleeve는 서로 다른 symbol에 매핑, 모든 bar에서 active.
        fit window: [0, fit_end), oos window: [oos_start, oos_end).
        """
        S = len(fit_edges)
        N = S
        T = 12
        fit_end = 6
        oos_start = 6
        oos_end = T

        mask = np.ones((T, S), dtype=np.bool_)
        side = np.ones((T, S), dtype=np.float64)
        sleeve_to_sym = np.arange(S, dtype=np.int64)

        close = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            c_fit = fit_edges[j]
            c_oos = oos_edges[j]
            base = 100.0
            for t in range(T):
                if t < fit_end:
                    close[t, j] = base * (1.0 + c_fit) ** t
                else:
                    t_off = t - oos_start
                    close[t, j] = base * (1.0 + c_fit) ** fit_end * (1.0 + c_oos) ** t_off

        cache = _make_cache(mask, side, sleeve_to_sym)
        aligned = _make_aligned(close)
        return cache, aligned, 0, fit_end, oos_start, oos_end

    def test_positive_ic(self) -> None:
        """동일 순서 → realized_ic ≈ +1."""
        S = 5
        fit_edges = [0.01 * (j + 1) for j in range(S)]
        oos_edges = [0.001 * (j + 1) for j in range(S)]

        cache, aligned, fit_start, fit_end, oos_start, oos_end = self._build_cache_with_edges(fit_edges, oos_edges)

        e_fit = compute_per_sleeve_realized_edge(cache, aligned, fit_start, fit_end)
        e_oos = compute_per_sleeve_realized_edge(cache, aligned, oos_start, oos_end)

        valid = np.isfinite(e_fit) & np.isfinite(e_oos)
        assert valid.sum() >= 5
        ic, _ = spearmanr(e_fit[valid], e_oos[valid])
        assert ic > 0.99, f"expected ic~+1, got {ic:.4f}"

    def test_negative_ic(self) -> None:
        """역순 → realized_ic ≈ -1."""
        S = 5
        fit_edges = [0.01 * (j + 1) for j in range(S)]
        oos_edges = [0.01 * (S - j) for j in range(S)]

        cache, aligned, fit_start, fit_end, oos_start, oos_end = self._build_cache_with_edges(fit_edges, oos_edges)

        e_fit = compute_per_sleeve_realized_edge(cache, aligned, fit_start, fit_end)
        e_oos = compute_per_sleeve_realized_edge(cache, aligned, oos_start, oos_end)

        valid = np.isfinite(e_fit) & np.isfinite(e_oos)
        assert valid.sum() >= 5
        ic, _ = spearmanr(e_fit[valid], e_oos[valid])
        assert ic < -0.99, f"expected ic~-1, got {ic:.4f}"
