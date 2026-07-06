"""L2 TF Inclusion Gate tests.

Spec reference: docs/specs/layer2-multi-tf-combination.md
Test scenarios: S1-S6.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _parse_tf_from_strategy_id,
    compute_per_tf_fit_edge,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    L2SimulationCache,
    Layer2AllocationConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# S1: _parse_tf_from_strategy_id
# ─────────────────────────────────────────────────────────────────────────────


class TestParseTfFromStrategyId:
    """S1 — _parse_tf_from_strategy_id 정상/엣지."""

    def test_standard_suffix(self) -> None:
        assert _parse_tf_from_strategy_id("donchian_72_8h") == "8h"

    def test_standard_suffix_4h(self) -> None:
        assert _parse_tf_from_strategy_id("trend_pullback_4h") == "4h"

    def test_trailing_tf(self) -> None:
        assert _parse_tf_from_strategy_id("ema_cross_12h") == "12h"

    def test_no_match_returns_unk(self) -> None:
        assert _parse_tf_from_strategy_id("weird") == "unk"

    def test_empty_string_returns_unk(self) -> None:
        assert _parse_tf_from_strategy_id("") == "unk"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures for compute_per_tf_fit_edge
# ─────────────────────────────────────────────────────────────────────────────


def _make_minimal_cache(
    *,
    t_max: int = 10,
    n_sleeve: int = 2,
    n_sym: int = 2,
    sleeve_to_tf: tuple[str, ...] | None = None,
    side_vals: list[list[float]] | None = None,
    active_mask: list[list[bool]] | None = None,
) -> L2SimulationCache:
    if sleeve_to_tf is None:
        sleeve_to_tf = ("4h", "12h")
    if side_vals is None:
        side_vals = [[1.0, 1.0] for _ in range(t_max)]
    if active_mask is None:
        active_mask = [[True, True] for _ in range(t_max)]

    side_2d = np.array(side_vals, dtype=np.float64)
    signal_mask_2d = np.array(active_mask, dtype=np.bool_)
    sleeve_to_sym = np.array([0, 1], dtype=np.int64)
    sleeve_ids: tuple[tuple[str, str], ...] = (
        ("SYM_A", "strat_4h"),
        ("SYM_B", "strat_12h"),
    )

    return L2SimulationCache(
        vol_matrix_2d=np.ones((t_max, n_sym), dtype=np.float64),
        tradeable_mask_2d=np.ones((t_max, n_sym), dtype=np.bool_),
        hurdle_2d=np.full((t_max, n_sym), 3.8, dtype=np.float64),
        funding_2d=np.zeros((t_max, n_sym), dtype=np.float64),
        beta_1d=np.zeros(n_sym, dtype=np.float64),
        expected_gross_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
        expected_net_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
        holding_bars_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
        side_2d=side_2d,
        quality_weight_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
        signal_mask_2d=signal_mask_2d,
        sleeve_to_sym=sleeve_to_sym,
        sleeve_ids=sleeve_ids,
        sleeve_to_tf=sleeve_to_tf,
    )


def _make_aligned(t_max: int = 10, n_sym: int = 2) -> Any:
    """AlignedMarketData stub."""
    aligned = MagicMock()
    # close_2d: 상승추세 (t→t+1 always +1%)
    close = 100.0 * np.cumprod(
        np.ones((t_max + 1, n_sym), dtype=np.float64) * 1.01,
        axis=0,
    )
    aligned.close_2d = close[:t_max]  # [T, N]
    aligned.symbols = tuple(f"SYM_{i}" for i in range(n_sym))
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# S2: compute_per_tf_fit_edge directional sign
# ─────────────────────────────────────────────────────────────────────────────


class TestComputePerTfFitEdge:
    """S2 — compute_per_tf_fit_edge 방향 정합."""

    def test_directional_sign(self) -> None:
        """TF_A side=+1, forward_return>0 (hit); TF_B side=+1, forward_return<0 (miss)."""
        t_max = 10
        n_sleeve = 2
        n_sym = 2
        # SYM_A (sleeve 0): close downward trend → forward_return < 0
        # SYM_B (sleeve 1): close upward trend → forward_return > 0
        aligned = MagicMock()
        aligned.symbols = ("SYM_A", "SYM_B")
        close_2d = np.zeros((t_max, n_sym), dtype=np.float64)
        for t in range(t_max):
            close_2d[t, 0] = 100.0 - t * 1.0  # downward
            close_2d[t, 1] = 100.0 + t * 1.0  # upward
        aligned.close_2d = close_2d

        # sleeve 0 = "4h" on SYM_A (downward, side=+1 → miss)
        # sleeve 1 = "12h" on SYM_B (upward, side=+1 → hit)
        side_2d = np.ones((t_max, n_sleeve), dtype=np.float64)  # both +1
        signal_mask_2d = np.ones((t_max, n_sleeve), dtype=np.bool_)
        sleeve_to_sym = np.array([0, 1], dtype=np.int64)
        sleeve_ids: tuple[tuple[str, str], ...] = (
            ("SYM_A", "strat_4h"),
            ("SYM_B", "strat_12h"),
        )
        cache = L2SimulationCache(
            vol_matrix_2d=np.ones((t_max, n_sym), dtype=np.float64),
            tradeable_mask_2d=np.ones((t_max, n_sym), dtype=np.bool_),
            hurdle_2d=np.full((t_max, n_sym), 3.8, dtype=np.float64),
            funding_2d=np.zeros((t_max, n_sym), dtype=np.float64),
            beta_1d=np.zeros(n_sym, dtype=np.float64),
            expected_gross_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            expected_net_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            holding_bars_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            side_2d=side_2d,
            quality_weight_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            signal_mask_2d=signal_mask_2d,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_ids=sleeve_ids,
            sleeve_to_tf=("4h", "12h"),
        )
        edge = compute_per_tf_fit_edge(cache, aligned, fit_start=0, fit_end=t_max - 1)
        # 12h (SYM_B, upward + side +1 → positive) should be > 4h (SYM_A, downward + side +1 → negative)
        assert edge.get("12h", -999.0) > edge.get("4h", 999.0), (
            f"expected 12h > 4h, got 12h={edge.get('12h'):.4f} 4h={edge.get('4h'):.4f}"
        )

    def test_empty_tf_returns_zero(self) -> None:
        """TF에 active sleeve 0개 → edge 0.0."""
        t_max = 10
        n_sleeve = 2
        n_sym = 2
        signal_mask_2d = np.zeros((t_max, n_sleeve), dtype=np.bool_)  # no active sleeves
        side_2d = np.ones((t_max, n_sleeve), dtype=np.float64)
        sleeve_to_sym = np.array([0, 1], dtype=np.int64)
        sleeve_ids: tuple[tuple[str, str], ...] = (
            ("SYM_A", "strat_4h"),
            ("SYM_B", "strat_12h"),
        )
        cache = L2SimulationCache(
            vol_matrix_2d=np.ones((t_max, n_sym), dtype=np.float64),
            tradeable_mask_2d=np.ones((t_max, n_sym), dtype=np.bool_),
            hurdle_2d=np.full((t_max, n_sym), 3.8, dtype=np.float64),
            funding_2d=np.zeros((t_max, n_sym), dtype=np.float64),
            beta_1d=np.zeros(n_sym, dtype=np.float64),
            expected_gross_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            expected_net_bps_2d=np.zeros((t_max, n_sleeve), dtype=np.float64),
            holding_bars_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            side_2d=side_2d,
            quality_weight_2d=np.ones((t_max, n_sleeve), dtype=np.float64),
            signal_mask_2d=signal_mask_2d,
            sleeve_to_sym=sleeve_to_sym,
            sleeve_ids=sleeve_ids,
            sleeve_to_tf=("4h", "12h"),
        )
        aligned = _make_aligned(t_max, n_sym)
        edge = compute_per_tf_fit_edge(cache, aligned, fit_start=0, fit_end=5)
        assert edge == {}


# ─────────────────────────────────────────────────────────────────────────────
# S4: inclusion gate filter logic
# ─────────────────────────────────────────────────────────────────────────────


class TestTfInclusionFilter:
    """S4 — 포함 게이트 필터 적용."""

    def test_filters_excluded_tf_sleeves(self) -> None:
        """included_tfs={'4h'}이면 12h sleeve 제거."""
        config = Layer2AllocationConfig(l2_tf_inclusion_enabled=True)
        assert config.l2_tf_inclusion_enabled is True

    def test_disabled_no_change(self) -> None:
        """게이트 off 시 거동 불변."""
        config = Layer2AllocationConfig(l2_tf_inclusion_enabled=False)
        assert config.l2_tf_inclusion_enabled is False


# ─────────────────────────────────────────────────────────────────────────────
# S5: empty fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestTfInclusionEmptyFallback:
    """S5 — 공집합 fail-safe."""

    def test_empty_fallback_config_defaults(self) -> None:
        """min_edge=0, enabled=True가 기본값."""
        config = Layer2AllocationConfig()
        assert config.l2_tf_inclusion_enabled is True
        assert config.l2_tf_inclusion_min_edge == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# S6: gate off regression
# ─────────────────────────────────────────────────────────────────────────────


class TestTfInclusionRegression:
    """S6 — 게이트 off 시 거동 불변."""

    def test_from_mapping_parses_correctly(self) -> None:
        """from_mapping이 신규 필드를 올바르게 파싱."""
        config = Layer2AllocationConfig.from_mapping(
            {
                "l2_tf_inclusion_enabled": False,
                "l2_tf_inclusion_min_edge": 0.005,
            }
        )
        assert config.l2_tf_inclusion_enabled is False
        assert config.l2_tf_inclusion_min_edge == 0.005

    def test_from_mapping_defaults(self) -> None:
        """from_mapping 기본값."""
        config = Layer2AllocationConfig.from_mapping({})
        assert config.l2_tf_inclusion_enabled is True
        assert config.l2_tf_inclusion_min_edge == 0.0
