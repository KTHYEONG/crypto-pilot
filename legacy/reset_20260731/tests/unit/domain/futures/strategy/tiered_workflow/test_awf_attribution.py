"""Tests for Layer2 Fold Attribution Diagnostics.

Scenarios:
  S1 - _assemble_fold_attribution reconciliation (Happy)
  S2 - friction ratio zero-division 방어 (Edge)
  S3 - throttle/exposure 빈 리스트 fallback (Edge)
  S4 - NaN 입력 방어 (Error/Robust)
  S5 - _resolve_sleeve_signals_at_bar drop count (Boundary)
  S6 - _count_netting_symbols netting 정확성 (Logic)
  S7 - diag on/off 무로깅 (Integration, optional)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    Layer2FoldAttribution,
    _assemble_fold_attribution,
    _count_netting_symbols,
    _resolve_sleeve_signals_at_bar,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache

# ---------------------------------------------------------------------------
# S1: _assemble_fold_attribution reconciliation (Happy)
# ---------------------------------------------------------------------------


class TestAssembleFoldAttribution:
    def test_reconciles_total_and_gap(self) -> None:
        """S1: realized_total == price + funding - cost, alpha_gap == realized - expected."""
        attr = _assemble_fold_attribution(
            fold_idx=0,
            oos_bars=100,
            n_rebal=10,
            realized_price=0.05,
            realized_funding=-0.01,
            realized_cost=0.02,
            expected_net=0.03,
            gross_exps=[1.5, 1.6],
            net_exps=[0.2, 0.3],
            throttle_mults=[0.8, 1.0],
            sleeves_active=[5, 7],
            friction_pass_total=8,
            signal_total=10,
            dropped_below_cost=2,
            netting_events=1,
        )
        assert attr.realized_total == pytest.approx(0.02)
        assert attr.alpha_gap == pytest.approx(-0.01)

    def test_zero_signal_no_div_error(self) -> None:
        """S2: signal_total=0 → friction_pass_ratio=0.0, no ZeroDivisionError."""
        attr = _assemble_fold_attribution(
            fold_idx=0,
            oos_bars=50,
            n_rebal=5,
            realized_price=0.01,
            realized_funding=0.0,
            realized_cost=0.005,
            expected_net=0.0,
            gross_exps=[1.0],
            net_exps=[0.0],
            throttle_mults=[1.0],
            sleeves_active=[3],
            friction_pass_total=0,
            signal_total=0,
            dropped_below_cost=0,
            netting_events=0,
        )
        assert attr.friction_pass_ratio == 0.0
        assert np.isfinite(attr.realized_total)

    def test_empty_lists_use_safe_defaults(self) -> None:
        """S3: empty throttle_mults → 1.0, empty gross_exps/sleeves_active → 0.0."""
        attr = _assemble_fold_attribution(
            fold_idx=0,
            oos_bars=30,
            n_rebal=0,
            realized_price=0.0,
            realized_funding=0.0,
            realized_cost=0.0,
            expected_net=0.0,
            gross_exps=[],
            net_exps=[],
            throttle_mults=[],
            sleeves_active=[],
            friction_pass_total=0,
            signal_total=0,
            dropped_below_cost=0,
            netting_events=0,
        )
        assert attr.throttle_mult_mean == 1.0
        assert attr.mean_gross_exp == 0.0
        assert attr.mean_net_exp == 0.0
        assert attr.sleeves_active_mean == 0.0

    def test_nan_input_coerced_to_zero(self) -> None:
        """S4: NaN expected_net → 0.0, alpha_gap remains finite."""
        attr = _assemble_fold_attribution(
            fold_idx=0,
            oos_bars=40,
            n_rebal=2,
            realized_price=0.02,
            realized_funding=-0.005,
            realized_cost=0.01,
            expected_net=float("nan"),
            gross_exps=[1.2],
            net_exps=[0.1],
            throttle_mults=[1.0],
            sleeves_active=[4],
            friction_pass_total=2,
            signal_total=5,
            dropped_below_cost=0,
            netting_events=0,
        )
        assert attr.expected_net == 0.0
        assert np.isfinite(attr.alpha_gap)


# ---------------------------------------------------------------------------
# S5: _resolve_sleeve_signals_at_bar drop count (Boundary)
# ---------------------------------------------------------------------------


class TestResolveSleeveSignalsDropCount:
    def test_reports_below_cost_drop_count(self) -> None:
        """S5: 2 sleeves 중 gross < cost인 1개 drop → n_dropped=1, len(result)=1."""
        n_sym = 1
        n_sleeve = 2
        t = 0

        cache = L2SimulationCache(
            vol_matrix_2d=np.full((1, n_sym), 0.02, dtype=np.float64),
            tradeable_mask_2d=np.full((1, n_sym), True, dtype=np.bool_),
            hurdle_2d=np.full((1, n_sym), 3.8, dtype=np.float64),
            funding_2d=np.zeros((1, n_sym), dtype=np.float64),
            beta_1d=np.zeros(n_sym, dtype=np.float64),
            expected_gross_bps_2d=np.array([[3.0, 10.0]], dtype=np.float64),
            expected_net_bps_2d=np.array([[1.0, 5.0]], dtype=np.float64),
            holding_bars_2d=np.array([[1.0, 1.0]], dtype=np.float64),
            side_2d=np.array([[1.0, 1.0]], dtype=np.float64),
            quality_weight_2d=np.array([[1.0, 1.0]], dtype=np.float64),
            signal_mask_2d=np.full((1, n_sleeve), True, dtype=np.bool_),
            sleeve_to_sym=np.array([0, 0], dtype=np.int64),
            sleeve_ids=(("SYM", "strat1"), ("SYM", "strat2")),
            sleeve_to_tf=("4h", "4h"),
        )

        sigs, _edges, n_dropped = _resolve_sleeve_signals_at_bar(
            cache=cache,
            t=t,
            tradeable_mask=np.array([True], dtype=np.bool_),
            symbols=("SYM",),
            hurdle_row=np.array([3.8], dtype=np.float64),
            vol_row=np.array([0.02], dtype=np.float64),
            fixed_cost_safety_mult=1.25,
        )

        assert n_dropped == 1
        assert len(sigs) == 1
        # surviving sleeve는 gross=10 > cost(=3.8*1.25=4.75)인 쪽
        surviving_key = ("SYM", "strat2")
        assert surviving_key in sigs


# ---------------------------------------------------------------------------
# S6: _count_netting_symbols netting 정확성 (Logic)
# ---------------------------------------------------------------------------


class TestCountNettingSymbols:
    def test_flags_opposite_sign_cancellation(self) -> None:
        """S6: sym A에 +10, -10 sleeve → pooled≈0 → netting=1."""
        sleeve_sigs: dict[tuple[str, str], SymbolSignal] = {
            ("A", "s1"): SymbolSignal(
                raw_mu=10.0,
                volatility=0.02,
                n_obs=1,
                t_stat=0.0,
                valid=True,
                beta_btc=None,
                quality_weight=1.0,
            ),
            ("A", "s2"): SymbolSignal(
                raw_mu=-10.0,
                volatility=0.02,
                n_obs=1,
                t_stat=0.0,
                valid=True,
                beta_btc=None,
                quality_weight=1.0,
            ),
        }
        pooled: dict[str, SymbolSignal] = {
            "A": SymbolSignal(
                raw_mu=0.0,
                volatility=0.02,
                n_obs=2,
                t_stat=0.0,
                valid=True,
                beta_btc=None,
                quality_weight=2.0,
            ),
        }
        assert _count_netting_symbols(sleeve_sigs, pooled) == 1

    def test_same_sign_no_netting(self) -> None:
        """S6: 동일 부호 두 sleeve → netting=0."""
        sleeve_sigs: dict[tuple[str, str], SymbolSignal] = {
            ("A", "s1"): SymbolSignal(
                raw_mu=5.0,
                volatility=0.02,
                n_obs=1,
                t_stat=0.0,
                valid=True,
                beta_btc=None,
                quality_weight=1.0,
            ),
            ("A", "s2"): SymbolSignal(
                raw_mu=3.0,
                volatility=0.02,
                n_obs=1,
                t_stat=0.0,
                valid=True,
                beta_btc=None,
                quality_weight=1.0,
            ),
        }
        pooled: dict[str, SymbolSignal] = {
            "A": SymbolSignal(
                raw_mu=4.0,
                volatility=0.02,
                n_obs=2,
                t_stat=0.0,
                valid=True,
                beta_btc=None,
                quality_weight=2.0,
            ),
        }
        assert _count_netting_symbols(sleeve_sigs, pooled) == 0


# ---------------------------------------------------------------------------
# S7: diag on/off 로깅 검증 (Integration) — caplog 사용
# ---------------------------------------------------------------------------


class TestLayer2FoldAttributionDataclass:
    def test_frozen_and_slots(self) -> None:
        """Layer2FoldAttribution is frozen and has __slots__."""
        attr = Layer2FoldAttribution(
            fold_idx=0,
            oos_bars=100,
            n_rebal=10,
            realized_total=0.02,
            realized_price=0.05,
            realized_funding=-0.01,
            realized_cost=0.02,
            expected_net=0.03,
            alpha_gap=-0.01,
            mean_gross_exp=1.55,
            mean_net_exp=0.25,
            sleeves_active_mean=6.0,
            friction_pass_ratio=0.8,
            throttle_mult_mean=0.9,
            dropped_below_cost=2,
            netting_events=1,
        )
        assert attr.realized_total == pytest.approx(0.02)

    def test_fold_attribution_whipsaw_decomposition(self) -> None:
        """S6: 저ER=손실, 고ER=수익 → low_er_price 합산, corr>0."""
        attr = _assemble_fold_attribution(
            fold_idx=0,
            oos_bars=100,
            n_rebal=4,
            realized_price=0.0,
            realized_funding=0.0,
            realized_cost=0.0,
            expected_net=0.0,
            gross_exps=[1.0],
            net_exps=[0.0],
            throttle_mults=[1.0],
            sleeves_active=[1],
            friction_pass_total=1,
            signal_total=1,
            dropped_below_cost=0,
            netting_events=0,
            er_return_pairs=[(0.1, -50.0), (0.2, -40.0), (0.8, 30.0), (0.9, 25.0)],
            target=0.35,
        )
        assert attr.realized_price_low_er == pytest.approx(-90.0, abs=1e-6)
        assert attr.trend_efficiency_corr > 0.0
        assert attr.mean_trend_efficiency == pytest.approx(0.5, abs=1e-6)


def test_combine_sleeve_signals_optimized_logical_equivalence() -> None:
    from src.domain.futures.strategy.cs_rank import SymbolSignal
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _combine_sleeve_signals_to_symbol

    sleeve_signals = {
        ("BTC", "strat1"): SymbolSignal(
            raw_mu=5.0, volatility=0.02, n_obs=10, t_stat=1.5, valid=True, beta_btc=None, quality_weight=1.0
        ),
        ("BTC", "strat2"): SymbolSignal(
            raw_mu=8.0, volatility=0.02, n_obs=10, t_stat=2.0, valid=True, beta_btc=None, quality_weight=2.0
        ),
        ("ETH", "strat1"): SymbolSignal(
            raw_mu=-2.0, volatility=0.03, n_obs=10, t_stat=-1.0, valid=True, beta_btc=None, quality_weight=1.5
        ),
    }

    sleeve_edges = {
        ("BTC", "strat1"): (5.0, 1.0),
        ("BTC", "strat2"): (8.0, 1.0),
        ("ETH", "strat1"): (-2.0, 1.0),
    }

    combined, _friction = _combine_sleeve_signals_to_symbol(
        sleeve_signals=sleeve_signals,
        method="precision_weighted",
        conviction_cap_mult=1.5,
        sleeve_edges=sleeve_edges,
    )

    assert "BTC" in combined
    assert combined["BTC"].raw_mu == pytest.approx(7.0)
    assert combined["BTC"].quality_weight == pytest.approx(3.0)

    assert "ETH" in combined
    assert combined["ETH"].raw_mu == pytest.approx(-2.0)
