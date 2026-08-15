"""Tests for L1→L2 confidence handoff: ValidatedSignalEvent LCB fields, cache, combine."""
from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.allocation_policy import compute_l1_confidence
from src.domain.futures.strategy.cs_rank import SymbolSignal


class TestSymbolSignalConfidenceFields:
    def test_symbol_signal_has_new_fields(self) -> None:
        sig = SymbolSignal(
            raw_mu=2.0,
            volatility=0.02,
            n_obs=10,
            t_stat=1.5,
            valid=True,
            quality_weight=0.8,
            l1_edge_margin_bps_per_bar=1.0,
            l1_confidence=0.4,
        )
        assert sig.l1_edge_margin_bps_per_bar == 1.0
        assert sig.l1_confidence == 0.4

    def test_symbol_signal_defaults_to_zero_fields(self) -> None:
        sig = SymbolSignal(
            raw_mu=2.0,
            volatility=0.02,
            n_obs=10,
            t_stat=1.5,
            valid=True,
        )
        assert sig.l1_edge_margin_bps_per_bar == 0.0
        assert sig.l1_confidence == 0.0


class TestComputeL1Confidence:
    def test_matches_formula(self) -> None:
        c = compute_l1_confidence(
            mu_bps=np.array([5.0, -3.0, 2.0, -4.0]),
            l1_edge_margin_bps_per_bar=np.array([2.0, -1.5, -1.0, 0.0]),
            quality_weight=np.array([0.9, 0.7, 0.6, 0.8]),
        )
        expected = np.array([
            1.0 * 0.9 * min(2.0 / 5.0, 1.0),
            1.0 * 0.7 * min(1.5 / 3.0, 1.0),
            0.0 * 0.6 * min(1.0 / 2.0, 1.0),
            0.0 * 0.8 * min(0.0 / 4.0, 1.0),
        ])
        expected[2] = 0.0
        expected[3] = 0.0
        np.testing.assert_allclose(c, expected)


def test_l1_lcb_is_scattered_atomically_with_strongest_event() -> None:
    """The strongest event carries its own LCB and breakeven pair."""
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _scatter_signals_jit

    expected_gross = np.zeros((2, 1), dtype=np.float64)
    expected_net = np.zeros((2, 1), dtype=np.float64)
    holding = np.zeros((2, 1), dtype=np.float64)
    side = np.zeros((2, 1), dtype=np.float64)
    quality = np.zeros((2, 1), dtype=np.float64)
    strength = np.zeros((2, 1), dtype=np.float64)
    lcb = np.zeros((2, 1), dtype=np.float64)
    breakeven = np.zeros((2, 1), dtype=np.float64)
    mask = np.ones((2, 1), dtype=np.bool_)
    _scatter_signals_jit(
        np.array([0]), np.array([1]), np.array([0]), np.array([2.0]), np.array([1.0]),
        np.array([1.0]), np.array([0.5]), np.array([9.0]), np.array([4.0]), np.array([2.0]),
        expected_gross, expected_net, holding, side, quality, strength,
        lcb, breakeven, mask, 2,
    )
    np.testing.assert_allclose(lcb[:, 0], [0.0, 4.0])
    np.testing.assert_allclose(breakeven[:, 0], [0.0, 2.0])
