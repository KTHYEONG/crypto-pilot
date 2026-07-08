from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.conditional_cells import (
    ConditionalCellGateConfig,
    evaluate_conditional_l0_cells,
    generate_default_cell_specs,
)
from src.domain.futures.alpha_foundry.contracts import AlphaGateConfig, AlphaRecipe
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


class TestGenerateDefaultCellSpecs:
    def test_funding_polarity_axis_generates_buckets(self) -> None:
        n_bars = 6
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]").repeat(n_bars)
        close = np.ones((n_bars, 1), dtype=np.float64) * 100.0
        mask = np.ones((n_bars, 1), dtype=np.bool_)
        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=("BTCUSDT",),
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((n_bars, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((n_bars, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((n_bars, 1), dtype=np.bool_),
            kill_mask=np.zeros((n_bars, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=("BTCUSDT",),
            signed_score_2d=np.zeros((n_bars, 1), dtype=np.float64),
            side_hint_2d=np.zeros((n_bars, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((n_bars, 1), dtype=np.float64),
            valid_mask_2d=mask, metadata={}, archetype="trend",
        )
        specs = generate_default_cell_specs(
            panel=panel, aligned=aligned,
            config=ConditionalCellGateConfig(enabled=True, axes=("funding_polarity",)),
        )
        assert len(specs) > 0
        assert any("funding_polarity" in s.cell_id.lower() for s in specs)

    def test_volatility_regime_axis_generates_buckets(self) -> None:
        n_bars = 6
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]").repeat(n_bars)
        close = np.ones((n_bars, 1), dtype=np.float64) * 100.0
        mask = np.ones((n_bars, 1), dtype=np.bool_)
        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=("BTCUSDT",),
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((n_bars, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((n_bars, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((n_bars, 1), dtype=np.bool_),
            kill_mask=np.zeros((n_bars, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=("BTCUSDT",),
            signed_score_2d=np.zeros((n_bars, 1), dtype=np.float64),
            side_hint_2d=np.zeros((n_bars, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((n_bars, 1), dtype=np.float64),
            valid_mask_2d=mask, metadata={}, archetype="trend",
        )
        specs = generate_default_cell_specs(
            panel=panel, aligned=aligned,
            config=ConditionalCellGateConfig(enabled=True, axes=("volatility_regime",)),
        )
        assert len(specs) > 0


class TestEvaluateConditionalL0Cells:
    def test_conditional_cell_can_pass_when_parent_average_fails(self) -> None:
        n_bars = 14
        datetimes = np.array(
            [
                f"2026-01-{d:02d}T{h:02d}:00:00"
                for d in range(1, 4)
                for h in range(0, 24, 4)
            ][:n_bars],
            dtype="datetime64[ns]",
        )
        symbols = ("BTCUSDT", "ETHUSDT")
        close = np.zeros((n_bars, 2), dtype=np.float64)
        close[:, 0] = 100.0 + np.arange(n_bars, dtype=np.float64) * 1.5
        close[:, 1] = 100.0 - np.arange(n_bars, dtype=np.float64) * 0.8
        mask = np.ones((n_bars, 2), dtype=np.bool_)
        score = np.zeros((n_bars, 2), dtype=np.float64)
        side = np.zeros((n_bars, 2), dtype=np.int8)
        for i in range(1, n_bars - 1, 2):
            score[i, 0] = 2.0 + (i % 5) * 0.3
            score[i, 1] = max(0.05, 0.5 - i * 0.04)
            side[i, 0] = 1
            side[i, 1] = -1
        valid = side != 0

        aligned = AlignedMarketData(
            datetimes=datetimes,
            symbols=symbols,
            open_2d=close,
            high_2d=close * 1.01,
            low_2d=close * 0.99,
            close_2d=close,
            volume_2d=np.full((n_bars, 2), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((n_bars, 2), dtype=np.float64),
            active_mask=mask,
            warm_mask=mask,
            entry_block_mask=np.zeros((n_bars, 2), dtype=np.bool_),
            kill_mask=np.zeros((n_bars, 2), dtype=np.bool_),
            adv_usdt_2d=np.full((n_bars, 2), [10_000_000.0, 500_000.0], dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, 2), 4.0, dtype=np.float64),
        )
        panel = CandidateSignalPanel(
            family="mock_ltf",
            variant="mock_ltf_v1",
            params={"ltf": "15m"},
            datetimes=datetimes,
            symbols=symbols,
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=1,
            min_holding_bars=1,
            stop_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid,
            metadata={"source_tf": "15m", "entry_mode": "sparse"},
            archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="mock_ltf_v1",
            family="mock_ltf",
            variant="mock_ltf_v1",
            timeframe="4h",
            archetype="trend",
            indicator_params={"ltf": "15m"},
            side_rule_id="mock_side",
            exit_policy_id="mock_exit",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel,
            aligned=aligned,
            recipe=recipe,
            cost_model=ExecutionCostModel(maker_ratio=0.0, stress_multiplier=1.0),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True,
                axes=("score_quantile", "symbol_liquidity"),
                min_cell_events=1,
                min_cell_effective_n=1.0,
                max_cells_per_recipe=4,
                min_symbols_per_cell=1,
                allow_single_symbol_cells=True,
            ),
            bars_per_year=2190.0,
            run_id="unit",
        )

        assert any(c.gate_evidence.gate_passed for c in cells)
        assert any(c.cell_id for c in cells)

    def test_disabled_config_returns_empty(self) -> None:
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]")
        symbols = ("BTCUSDT",)
        close = np.array([[100.0]], dtype=np.float64)
        mask = np.ones((1, 1), dtype=np.bool_)
        valid = np.ones((1, 1), dtype=np.bool_)

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((1, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((1, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((1, 1), dtype=np.bool_),
            kill_mask=np.zeros((1, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=np.zeros((1, 1), dtype=np.float64),
            side_hint_2d=np.zeros((1, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(enabled=False),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) == 0

    def test_insufficient_sample_in_cell(self) -> None:
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]")
        symbols = ("BTCUSDT",)
        close = np.array([[100.0]], dtype=np.float64)
        mask = np.ones((1, 1), dtype=np.bool_)
        valid = np.zeros((1, 1), dtype=np.bool_)

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((1, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((1, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((1, 1), dtype=np.bool_),
            kill_mask=np.zeros((1, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=np.zeros((1, 1), dtype=np.float64),
            side_hint_2d=np.zeros((1, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True,
                axes=("score_quantile",),
                min_cell_events=100,
                min_cell_effective_n=50.0,
                min_symbols_per_cell=1,
                allow_single_symbol_cells=True,
            ),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) == 0

    def test_all_cells_fail_returns_failure_axis(self) -> None:
        datetimes = np.array(
            ["2026-01-01T00:00:00", "2026-01-01T04:00:00", "2026-01-01T08:00:00"],
            dtype="datetime64[ns]",
        )
        symbols = ("BTCUSDT",)
        close = np.array([[100.0], [101.0], [99.0]], dtype=np.float64)
        mask = np.ones((3, 1), dtype=np.bool_)
        score = np.array([[0.0], [0.5], [0.0]], dtype=np.float64)
        side = np.array([[0], [1], [0]], dtype=np.int8)
        valid = side != 0

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close * 1.01, low_2d=close * 0.99,
            close_2d=close,
            volume_2d=np.full((3, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((3, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((3, 1), dtype=np.bool_),
            kill_mask=np.zeros((3, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=score,
            side_hint_2d=side,
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True,
                axes=("score_quantile",),
                min_cell_events=1,
                min_cell_effective_n=1.0,
                min_symbols_per_cell=1,
                allow_single_symbol_cells=True,
            ),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) > 0
        assert any(c.failure_axis for c in cells)

    def test_insufficient_symbols_returns_cells_with_failure(self) -> None:
        datetimes = np.array(
            ["2026-01-01T00:00:00", "2026-01-01T04:00:00", "2026-01-01T08:00:00",
             "2026-01-01T12:00:00", "2026-01-01T16:00:00", "2026-01-01T20:00:00"],
            dtype="datetime64[ns]",
        )
        symbols = ("BTCUSDT",)
        close = np.array([[100.0], [101.0], [103.0], [104.0], [105.0], [106.0]], dtype=np.float64)
        mask = np.ones((6, 1), dtype=np.bool_)
        score = np.array([[0.0], [3.0], [0.0], [2.8], [0.0], [0.0]], dtype=np.float64)
        side = np.array([[0], [1], [0], [1], [0], [0]], dtype=np.int8)
        valid = side != 0

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close,
            volume_2d=np.full((6, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((6, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((6, 1), dtype=np.bool_),
            kill_mask=np.zeros((6, 1), dtype=np.bool_),
            adv_usdt_2d=np.full((6, 1), 10_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((6, 1), 4.0, dtype=np.float64),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(maker_ratio=0.0, stress_multiplier=1.0),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True,
                axes=("score_quantile",),
                min_cell_events=1, min_cell_effective_n=1.0,
                max_cells_per_recipe=10,
                min_symbols_per_cell=3, allow_single_symbol_cells=False,
            ),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) > 0
        assert any(c.failure_axis for c in cells)

    def test_cell_with_events_below_min_events_adds_insufficient_sample(self) -> None:
        datetimes = np.array(
            ["2026-01-01T00:00:00", "2026-01-01T04:00:00", "2026-01-01T08:00:00",
             "2026-01-01T12:00:00", "2026-01-01T16:00:00", "2026-01-01T20:00:00"],
            dtype="datetime64[ns]",
        )
        symbols = ("BTCUSDT",)
        close = np.array([[100.0], [101.0], [103.0], [104.0], [105.0], [106.0]], dtype=np.float64)
        mask = np.ones((6, 1), dtype=np.bool_)
        score = np.array([[0.0], [3.0], [0.0], [2.8], [0.0], [0.0]], dtype=np.float64)
        side = np.array([[0], [1], [0], [1], [0], [0]], dtype=np.int8)
        valid = side != 0

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close,
            volume_2d=np.full((6, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((6, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((6, 1), dtype=np.bool_),
            kill_mask=np.zeros((6, 1), dtype=np.bool_),
            adv_usdt_2d=np.full((6, 1), 10_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((6, 1), 4.0, dtype=np.float64),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(maker_ratio=0.0, stress_multiplier=1.0),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True,
                axes=("score_quantile",),
                min_cell_events=100, min_cell_effective_n=50.0,
                max_cells_per_recipe=10,
                min_symbols_per_cell=1, allow_single_symbol_cells=True,
            ),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) > 0
        assert any(c.failure_axis == "insufficient_sample" for c in cells)

    def test_cell_with_negative_returns_gets_failure_axis(self) -> None:
        datetimes = np.array(
            ["2026-01-01T00:00:00", "2026-01-01T04:00:00", "2026-01-01T08:00:00"],
            dtype="datetime64[ns]",
        )
        symbols = ("BTCUSDT",)
        close = np.array([[100.0], [101.0], [99.0]], dtype=np.float64)
        mask = np.ones((3, 1), dtype=np.bool_)
        score = np.array([[0.0], [2.0], [0.0]], dtype=np.float64)
        side = np.array([[0], [1], [0]], dtype=np.int8)
        valid = side != 0

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close,
            volume_2d=np.full((3, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((3, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((3, 1), dtype=np.bool_),
            kill_mask=np.zeros((3, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True,
                axes=("score_quantile",),
                min_cell_events=1, min_cell_effective_n=1.0,
                max_cells_per_recipe=10,
                min_symbols_per_cell=1, allow_single_symbol_cells=True,
            ),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) > 0
        assert any(c.failure_axis for c in cells)


class TestEvaluateConditionalL0CellsErrors:
    def test_shape_mismatch_raises_value_error(self) -> None:
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]")
        symbols = ("BTCUSDT",)
        close = np.array([[100.0]], dtype=np.float64)
        score = np.zeros((2, 1), dtype=np.float64)
        mask = np.ones((1, 1), dtype=np.bool_)

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((1, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((1, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((1, 1), dtype=np.bool_),
            kill_mask=np.zeros((1, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=score,
            side_hint_2d=np.zeros((2, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((2, 1), dtype=np.float64),
            valid_mask_2d=np.ones((2, 1), dtype=np.bool_),
            metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_conditional_l0_cells(
                panel=panel, aligned=aligned, recipe=recipe,
                cost_model=ExecutionCostModel(),
                gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
                cell_config=ConditionalCellGateConfig(
                    enabled=True, axes=("score_quantile",),
                    min_cell_events=1, min_cell_effective_n=1.0,
                    min_symbols_per_cell=1, allow_single_symbol_cells=True,
                ),
                bars_per_year=2190.0, run_id="unit",
            )

    def test_side_hint_2d_shape_mismatch_raises_value_error(self) -> None:
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]")
        symbols = ("BTCUSDT",)
        close = np.array([[100.0]], dtype=np.float64)
        mask = np.ones((1, 1), dtype=np.bool_)

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((1, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((1, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((1, 1), dtype=np.bool_),
            kill_mask=np.zeros((1, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=np.zeros((1, 1), dtype=np.float64),
            side_hint_2d=np.zeros((2, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
            valid_mask_2d=np.ones((1, 1), dtype=np.bool_),
            metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_conditional_l0_cells(
                panel=panel, aligned=aligned, recipe=recipe,
                cost_model=ExecutionCostModel(),
                gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
                cell_config=ConditionalCellGateConfig(enabled=False),
                bars_per_year=2190.0, run_id="unit",
            )

    def test_unknown_conditional_axis_raises_value_error(self) -> None:
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]")
        symbols = ("BTCUSDT",)
        close = np.array([[100.0]], dtype=np.float64)
        mask = np.ones((1, 1), dtype=np.bool_)
        valid = np.ones((1, 1), dtype=np.bool_)

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((1, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((1, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((1, 1), dtype=np.bool_),
            kill_mask=np.zeros((1, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=np.zeros((1, 1), dtype=np.float64),
            side_hint_2d=np.zeros((1, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        with pytest.raises(ValueError, match="unsupported conditional axis"):
            evaluate_conditional_l0_cells(
                panel=panel, aligned=aligned, recipe=recipe,
                cost_model=ExecutionCostModel(),
                gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
                cell_config=ConditionalCellGateConfig(
                    enabled=True, axes=("invalid_axis",),
                    min_cell_events=1, min_cell_effective_n=1.0,
                    min_symbols_per_cell=1, allow_single_symbol_cells=True,
                ),
                bars_per_year=2190.0, run_id="unit",
            )

    def test_valid_mask_shape_mismatch_raises_value_error(self) -> None:
        aligned = AlignedMarketData(
            datetimes=np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            symbols=("BTCUSDT",),
            open_2d=np.array([[100.0]], dtype=np.float64),
            high_2d=np.array([[101.0]], dtype=np.float64),
            low_2d=np.array([[99.0]], dtype=np.float64),
            close_2d=np.array([[100.0]], dtype=np.float64),
            volume_2d=np.full((1, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((1, 1), dtype=np.float64),
            active_mask=np.ones((1, 1), dtype=np.bool_),
            warm_mask=np.ones((1, 1), dtype=np.bool_),
            entry_block_mask=np.zeros((1, 1), dtype=np.bool_),
            kill_mask=np.zeros((1, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            symbols=("BTCUSDT",),
            signed_score_2d=np.zeros((1, 1), dtype=np.float64),
            side_hint_2d=np.zeros((1, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
            valid_mask_2d=np.ones((2, 1), dtype=np.bool_),
            metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_conditional_l0_cells(
                panel=panel, aligned=aligned, recipe=recipe,
                cost_model=ExecutionCostModel(),
                gate_config=AlphaGateConfig(),
                cell_config=ConditionalCellGateConfig(enabled=False),
                bars_per_year=2190.0, run_id="unit",
            )

    def test_empty_axes_tuple_returns_empty_cells(self) -> None:
        n_bars = 6
        datetimes = np.array(
            ["2026-01-01T00:00:00"] * n_bars, dtype="datetime64[ns]",
        )
        close = np.ones((n_bars, 1), dtype=np.float64) * 100.0
        mask = np.ones((n_bars, 1), dtype=np.bool_)
        side = np.zeros((n_bars, 1), dtype=np.int8)
        side[1] = 1
        valid = side != 0
        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=("BTCUSDT",),
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((n_bars, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((n_bars, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((n_bars, 1), dtype=np.bool_),
            kill_mask=np.zeros((n_bars, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=("BTCUSDT",),
            signed_score_2d=np.zeros((n_bars, 1), dtype=np.float64),
            side_hint_2d=side,
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(enabled=True, axes=()),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) == 0

    def test_cell_without_adv_runs_liquidity_fallback(self) -> None:
        n_bars = 6
        datetimes = np.array(
            ["2026-01-01T00:00:00"] * n_bars, dtype="datetime64[ns]",
        )
        close = np.ones((n_bars, 1), dtype=np.float64) * 100.0
        mask = np.ones((n_bars, 1), dtype=np.bool_)
        side = np.zeros((n_bars, 1), dtype=np.int8)
        side[1] = 1
        side[3] = 1
        valid = side != 0
        score = np.zeros((n_bars, 1), dtype=np.float64)
        score[1] = 2.0
        score[3] = 3.0
        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=("BTCUSDT",),
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((n_bars, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((n_bars, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((n_bars, 1), dtype=np.bool_),
            kill_mask=np.zeros((n_bars, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=("BTCUSDT",),
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True, axes=("score_quantile", "symbol_liquidity"),
                min_cell_events=1, min_cell_effective_n=1.0,
                max_cells_per_recipe=10, min_symbols_per_cell=1,
                allow_single_symbol_cells=True,
            ),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) > 0

    def test_negative_bars_per_year_raises_value_error(self) -> None:
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]")
        symbols = ("BTCUSDT",)
        close = np.array([[100.0]], dtype=np.float64)
        mask = np.ones((1, 1), dtype=np.bool_)
        valid = np.ones((1, 1), dtype=np.bool_)

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((1, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((1, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((1, 1), dtype=np.bool_),
            kill_mask=np.zeros((1, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=np.zeros((1, 1), dtype=np.float64),
            side_hint_2d=np.zeros((1, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
            valid_mask_2d=valid, metadata={}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        with pytest.raises(ValueError, match="bars_per_year must be positive"):
            evaluate_conditional_l0_cells(
                panel=panel, aligned=aligned, recipe=recipe,
                cost_model=ExecutionCostModel(),
                gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
                cell_config=ConditionalCellGateConfig(enabled=False),
                bars_per_year=-1.0, run_id="unit",
            )

    def test_generate_cell_specs_with_disabled_config_returns_empty(self) -> None:
        n_bars = 6
        datetimes = np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]").repeat(n_bars)
        close = np.ones((n_bars, 1), dtype=np.float64) * 100.0
        mask = np.ones((n_bars, 1), dtype=np.bool_)
        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=("BTCUSDT",),
            open_2d=close, high_2d=close, low_2d=close, close_2d=close,
            volume_2d=np.full((n_bars, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((n_bars, 1), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((n_bars, 1), dtype=np.bool_),
            kill_mask=np.zeros((n_bars, 1), dtype=np.bool_),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=("BTCUSDT",),
            signed_score_2d=np.zeros((n_bars, 1), dtype=np.float64),
            side_hint_2d=np.zeros((n_bars, 1), dtype=np.int8),
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=np.zeros((n_bars, 1), dtype=np.float64),
            valid_mask_2d=mask, metadata={}, archetype="trend",
        )
        specs = generate_default_cell_specs(
            panel=panel, aligned=aligned,
            config=ConditionalCellGateConfig(enabled=False),
        )
        assert len(specs) == 0

    def test_two_axis_combination_cells_generated(self) -> None:
        n_bars = 10
        datetimes = np.array(
            ["2026-01-01T00:00:00"] * n_bars, dtype="datetime64[ns]",
        )
        datetimes[0] = np.datetime64("2026-01-01T00:00:00")
        for i in range(1, n_bars):
            datetimes[i] = datetimes[i - 1] + np.timedelta64(4, "h")
        symbols = ("BTCUSDT", "ETHUSDT")
        close = np.zeros((n_bars, 2), dtype=np.float64)
        close[:, 0] = 100.0 + np.arange(n_bars, dtype=np.float64) * 1.0
        close[:, 1] = 100.0 - np.arange(n_bars, dtype=np.float64) * 0.5
        mask = np.ones((n_bars, 2), dtype=np.bool_)
        side = np.zeros((n_bars, 2), dtype=np.int8)
        score = np.zeros((n_bars, 2), dtype=np.float64)
        for i in range(1, n_bars - 1, 2):
            side[i, 0] = 1
            side[i, 1] = -1
            score[i, 0] = 2.0 + i * 0.2
            score[i, 1] = 0.5 - i * 0.05
        valid = side != 0

        aligned = AlignedMarketData(
            datetimes=datetimes, symbols=symbols,
            open_2d=close, high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close,
            volume_2d=np.full((n_bars, 2), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((n_bars, 2), dtype=np.float64),
            active_mask=mask, warm_mask=mask,
            entry_block_mask=np.zeros((n_bars, 2), dtype=np.bool_),
            kill_mask=np.zeros((n_bars, 2), dtype=np.bool_),
            adv_usdt_2d=np.full((n_bars, 2), [10_000_000.0, 500_000.0], dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, 2), 4.0, dtype=np.float64),
        )
        panel = CandidateSignalPanel(
            family="m", variant="v", params={},
            datetimes=datetimes, symbols=symbols,
            signed_score_2d=score, side_hint_2d=side,
            expected_holding_bars=1, min_holding_bars=1,
            stop_atr_mult=2.0, take_profit_atr_mult=4.0,
            turnover_proxy_2d=valid.astype(np.float64),
            valid_mask_2d=valid, metadata={"source_tf": "1h"}, archetype="trend",
        )
        recipe = AlphaRecipe(
            recipe_id="v", family="m", variant="v", timeframe="4h",
            archetype="trend", indicator_params={},
            side_rule_id="s", exit_policy_id="e",
            required_fields=("close",), causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )

        cells = evaluate_conditional_l0_cells(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(maker_ratio=0.0, stress_multiplier=1.0),
            gate_config=AlphaGateConfig(min_events=1, min_effective_n=1.0, min_nw_tstat=0.0),
            cell_config=ConditionalCellGateConfig(
                enabled=True,
                axes=("score_quantile", "symbol_liquidity"),
                min_cell_events=1, min_cell_effective_n=1.0,
                max_cells_per_recipe=20, max_axes_per_cell=2,
                min_symbols_per_cell=1, allow_single_symbol_cells=True,
            ),
            bars_per_year=2190.0, run_id="unit",
        )
        assert len(cells) > 0
