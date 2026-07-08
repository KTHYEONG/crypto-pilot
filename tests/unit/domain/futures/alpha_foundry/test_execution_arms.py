from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.contracts import AlphaRecipe
from src.domain.futures.alpha_foundry.execution_arms import (
    ExecutionArmConfig,
    ExecutionCostArm,
    estimate_execution_arm_cost_bps,
    resolve_execution_cost_arms,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


class TestResolveExecutionCostArms:
    def test_disabled_returns_taker_now_only(self) -> None:
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

        arms = resolve_execution_cost_arms(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            config=ExecutionArmConfig(enabled=False),
        )
        assert len(arms) == 1
        assert arms[0].style == "taker_now"

    def test_unknown_execution_style_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unsupported execution style"):
            ExecutionArmConfig(enabled=True, styles=("unknown_style",))  # type: ignore[arg-type]

    def test_enabled_with_all_styles(self) -> None:
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

        arms = resolve_execution_cost_arms(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            config=ExecutionArmConfig(
                enabled=True,
                styles=("taker_now", "maker_retest", "maker_or_cancel", "hybrid"),
                max_arm_count_per_cell=4,
            ),
        )
        styles = {a.style for a in arms}
        assert "taker_now" in styles
        assert "maker_retest" in styles
        assert "maker_or_cancel" in styles
        assert "hybrid" in styles

    def test_enabled_with_max_arm_count_cap(self) -> None:
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

        arms = resolve_execution_cost_arms(
            panel=panel, aligned=aligned, recipe=recipe,
            cost_model=ExecutionCostModel(),
            config=ExecutionArmConfig(
                enabled=True,
                styles=("taker_now", "maker_retest", "maker_or_cancel", "hybrid"),
                max_arm_count_per_cell=2,
            ),
        )
        assert len(arms) == 2


class TestEstimateExecutionArmCostBps:
    def test_returns_per_event_costs(self) -> None:
        arm = ExecutionCostArm(
            style="taker_now",
            fill_probability=1.0,
            base_round_trip_bps=10.0,
            adverse_selection_bps=0.0,
            unfilled_opportunity_cost_bps=0.0,
        )
        aligned = AlignedMarketData(
            datetimes=np.array(["2026-01-01T00:00:00", "2026-01-01T04:00:00"], dtype="datetime64[ns]"),
            symbols=("BTCUSDT",),
            open_2d=np.array([[100.0], [101.0]], dtype=np.float64),
            high_2d=np.array([[101.0], [102.0]], dtype=np.float64),
            low_2d=np.array([[99.0], [100.0]], dtype=np.float64),
            close_2d=np.array([[100.0], [101.0]], dtype=np.float64),
            volume_2d=np.full((2, 1), 1000.0, dtype=np.float64),
            funding_2d=np.zeros((2, 1), dtype=np.float64),
            active_mask=np.ones((2, 1), dtype=np.bool_),
            warm_mask=np.ones((2, 1), dtype=np.bool_),
            entry_block_mask=np.zeros((2, 1), dtype=np.bool_),
            kill_mask=np.zeros((2, 1), dtype=np.bool_),
        )
        event_mask = np.array([[True], [False]], dtype=np.bool_)

        costs = estimate_execution_arm_cost_bps(
            event_mask_2d=event_mask,
            arm=arm,
            aligned=aligned,
            holding_bars=1,
        )
        assert costs.shape == (2, 1)
        assert np.isfinite(costs[0, 0])
        assert costs[0, 0] >= 0.0

    def test_maker_retest_low_fill_prob_blocked(self) -> None:
        arm = ExecutionCostArm(
            style="maker_retest",
            fill_probability=0.0,
            base_round_trip_bps=5.0,
            adverse_selection_bps=2.0,
            unfilled_opportunity_cost_bps=3.0,
        )
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
        event_mask = np.array([[True]], dtype=np.bool_)

        costs = estimate_execution_arm_cost_bps(
            event_mask_2d=event_mask,
            arm=arm,
            aligned=aligned,
            holding_bars=1,
        )
        assert not np.isfinite(costs[0, 0])

    def test_empty_event_mask_returns_all_nan(self) -> None:
        arm = ExecutionCostArm(
            style="taker_now",
            fill_probability=1.0,
            base_round_trip_bps=10.0,
            adverse_selection_bps=0.0,
            unfilled_opportunity_cost_bps=0.0,
        )
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
        event_mask = np.array([[False]], dtype=np.bool_)

        costs = estimate_execution_arm_cost_bps(
            event_mask_2d=event_mask,
            arm=arm,
            aligned=aligned,
            holding_bars=1,
        )
        assert not np.isfinite(costs[0, 0])
