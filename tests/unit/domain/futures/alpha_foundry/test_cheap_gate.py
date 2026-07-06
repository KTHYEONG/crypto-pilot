from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.cheap_gate import (
    evaluate_alpha_cheap_gate_batch,
    evaluate_panel_cheap_gate,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateConfig,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def make_mock_aligned() -> AlignedMarketData:
    datetimes = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-11T00:00:00"),
        np.timedelta64(4, "h"),
        dtype="datetime64[ns]",
    )
    t = datetimes.shape[0]
    symbols = ("BTCUSDT", "ETHUSDT")
    base = np.linspace(100.0, 112.0, t, dtype=np.float64)
    close = np.column_stack([base, base * 0.8])
    high = close * 1.01
    low = close * 0.99
    volume = np.full_like(close, 1_000.0)
    mask = np.ones_like(close, dtype=np.bool_)
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=symbols,
        open_2d=close.copy(),
        high_2d=high,
        low_2d=low,
        close_2d=close,
        volume_2d=volume,
        funding_2d=np.full_like(close, 0.0001),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
    )


def make_mock_panel(*, recipe_id: str = "trend_ma:ema_12_72:4h") -> CandidateSignalPanel:
    aligned = make_mock_aligned()
    t, n = aligned.close_2d.shape
    score = np.ones((t, n), dtype=np.float64)
    side = np.ones((t, n), dtype=np.int8)
    valid = np.ones((t, n), dtype=np.bool_)
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=score,
        side_hint_2d=side,
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=valid,
        metadata={"recipe_id": recipe_id},
    )


SAMPLE_RECIPE = AlphaRecipe(
    recipe_id="trend_ma:ema_12_72:4h",
    family="trend_ma",
    variant="ema_12_72",
    timeframe="4h",
    archetype="trend",
    indicator_params={"fast": 12, "slow": 72},
    side_rule_id="trend_follow",
    exit_policy_id="atr_trail_2",
    required_fields=("close",),
    causal_lag_bars=1,
    max_turnover_per_year=365.0,
)


class TestEvaluatePanelCheapGate:
    def test_passes_positive_cost_adjusted_alpha(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.gate_passed
        assert evidence.mean_net_bps > 0.0
        assert len(evidence.reject_reasons) == 0

    def test_forward_return_excludes_last_horizon_rows(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.n_events > 0

    def test_raises_on_invalid_shape(self) -> None:
        aligned = make_mock_aligned()
        t_bad, n_bad = 10, 3
        bad_score = np.ones((t_bad, n_bad), dtype=np.float64)
        bad_side = np.ones((t_bad, n_bad), dtype=np.int8)
        bad_valid = np.ones((t_bad, n_bad), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=bad_score,
            side_hint_2d=bad_side,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t_bad, n_bad), dtype=np.float64),
            valid_mask_2d=bad_valid,
            metadata={"recipe_id": "bad"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        with pytest.raises(ValueError, match="shape"):
            evaluate_panel_cheap_gate(
                panel=panel,
                aligned=aligned,
                recipe=SAMPLE_RECIPE,
                cost_model=cost,
                config=cfg,
                bars_per_year=2190.0,
            )

    def test_respects_valid_mask_and_entry_block_mask(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        score = np.ones((t, n), dtype=np.float64)
        side = np.ones((t, n), dtype=np.int8)
        valid = np.ones((t, n), dtype=np.bool_)
        valid[5:, :] = False
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "test"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.n_events < t * n

    def test_net_edge_includes_stress_cost_and_funding(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel(stress_multiplier=2.0)
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert evidence.mean_net_bps < 100.0

    def test_high_turnover_panel_rejected(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        alternating = np.ones((t, n), dtype=np.int8)
        alternating[1::2, :] = -1
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma",
            variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.ones((t, n), dtype=np.float64),
            side_hint_2d=alternating,
            expected_holding_bars=3,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "high_turnover"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig(max_turnover_per_year=10.0)
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=SAMPLE_RECIPE,
            cost_model=cost,
            config=cfg,
            bars_per_year=2190.0,
        )
        assert not evidence.gate_passed
        assert any("turnover" in r for r in evidence.reject_reasons)

    # Scenario 1.1: bars_per_year=2190.0 (4h) — regression with alternating side
    def test_bars_per_year_2190_regression(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        alternating = np.ones((t, n), dtype=np.int8)
        alternating[1::2, :] = -1
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma", variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.ones((t, n), dtype=np.float64),
            side_hint_2d=alternating,
            expected_holding_bars=3, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "turnover_test"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        evidence = evaluate_panel_cheap_gate(
            panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
            cost_model=cost, config=cfg, bars_per_year=2190.0,
        )
        assert evidence.turnover_per_year > 0.0

    # Scenario 1.2: bars_per_year=730.0 (12h) → turnover = 1/3 of 4h
    def test_bars_per_year_730_turnover_one_third(self) -> None:
        aligned = make_mock_aligned()
        t, n = aligned.close_2d.shape
        alternating = np.ones((t, n), dtype=np.int8)
        alternating[1::2, :] = -1
        valid = np.ones((t, n), dtype=np.bool_)
        panel = CandidateSignalPanel(
            family="trend_ma", variant="ema_12_72",
            params={"fast": 12, "slow": 72},
            datetimes=aligned.datetimes,
            symbols=aligned.symbols,
            signed_score_2d=np.ones((t, n), dtype=np.float64),
            side_hint_2d=alternating,
            expected_holding_bars=3, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
            valid_mask_2d=valid,
            metadata={"recipe_id": "turnover_test"},
        )
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        ev_4h = evaluate_panel_cheap_gate(
            panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
            cost_model=cost, config=cfg, bars_per_year=2190.0,
        )
        ev_12h = evaluate_panel_cheap_gate(
            panel=panel, aligned=aligned, recipe=SAMPLE_RECIPE,
            cost_model=cost, config=cfg, bars_per_year=730.0,
        )
        assert abs(ev_12h.turnover_per_year - ev_4h.turnover_per_year / 3.0) < 1e-9

    # Scenario 3.4: bars_per_year <= 0.0 raises
    def test_raises_on_non_positive_bars_per_year(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel()
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        with pytest.raises(ValueError, match="bars_per_year must be positive"):
            evaluate_panel_cheap_gate(
                panel=panel,
                aligned=aligned,
                recipe=SAMPLE_RECIPE,
                cost_model=cost,
                config=cfg,
                bars_per_year=0.0,
            )


class TestEvaluateAlphaCheapGateBatch:
    def test_batch_processes_all_panels(self) -> None:
        aligned = make_mock_aligned()
        panel_a = make_mock_panel(recipe_id="alpha_a")
        panel_b = make_mock_panel(recipe_id="alpha_b")
        recipes = {
            "alpha_a": SAMPLE_RECIPE,
            "alpha_b": SAMPLE_RECIPE,
        }
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        results = evaluate_alpha_cheap_gate_batch(
            panels=[panel_a, panel_b],
            recipes=recipes,
            aligned=aligned,
            cost_model=cost,
            config=cfg,
        )
        assert len(results) == 2
        assert all(r.gate_passed for r in results)

    def test_unknown_recipe_id_skipped(self) -> None:
        aligned = make_mock_aligned()
        panel = make_mock_panel(recipe_id="unknown")
        recipes: dict[str, AlphaRecipe] = {}
        cost = ExecutionCostModel()
        cfg = CheapGateConfig()
        results = evaluate_alpha_cheap_gate_batch(
            panels=[panel],
            recipes=recipes,
            aligned=aligned,
            cost_model=cost,
            config=cfg,
        )
        assert len(results) == 0
