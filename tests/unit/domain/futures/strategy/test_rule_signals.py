from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel, SignalExitPolicy
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.market_regime import MarketRegimeContext
from src.domain.futures.strategy.rule_signals import (
    _attach_signal_context,
    _entry_rising_edge_2d,
    _rolling_max_2d,
    _rolling_min_2d,
    build_rule_signal_panels,
    candidate_panels_to_events,
)


def _make_aligned(t: int = 150, n: int = 2) -> AlignedMarketData:
    base = np.linspace(100.0, 130.0, t * n, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=base.copy(),
        high_2d=base * 1.01,
        low_2d=base * 0.99,
        close_2d=base.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        basis_2d=np.zeros((t, n), dtype=np.float64),
        taker_buy_2d=np.full((t, n), 500.0, dtype=np.float64),
        trades_2d=np.full((t, n), 100.0, dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )


def test_build_rule_signal_panels_returns_expected_tuple() -> None:
    aligned = _make_aligned()
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    assert isinstance(panels, tuple)
    assert len(panels) == 28

    expected_families = {
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "bollinger_reversion",
        "rsi_reversion",
        "funding_carry",
        "btc_regime_pullback",
        "funding_zscore_carry",
        "vol_regime_reversion",
        "trend_pullback_continuation",
        "dual_momentum",
        "residual_reversion",
        "mtf_trend_pullback",
        "mtf_breakout_retest",
        "taker_imbalance_momentum",
        "funding_extreme_reversal",
        "vol_term_structure_gate",
    }

    for p in panels:
        assert p.family in expected_families
        assert p.signed_score_2d.shape == (150, 2)
        assert p.side_hint_2d.shape == (150, 2)
        assert p.valid_mask_2d.shape == (150, 2)
        assert isinstance(p.symbols, tuple)
        assert len(p.symbols) == 2


def test_rolling_max_min_trailing_exclusive() -> None:
    # Monotonically increasing high: close[t] must be able to exceed prior high (trailing-exclusive).
    # If shift(1) is absent, roll_max[t] == high[t] and close[t] > roll_max[t] is always False.
    t, n = 50, 1
    high = np.linspace(100.0, 150.0, t).reshape(t, n)
    close = high * 1.001  # close slightly above same-bar high

    max_2d = _rolling_max_2d(high, window=10)
    min_2d = _rolling_min_2d(high, window=10)

    # After shift(1): roll_max[t] = max(high[t-10:t]) — excludes current bar.
    # close[t] = 1.001 * high[t] > prior_high is possible for t > 1.
    long_break = (close > max_2d)[2:].any()
    short_break = (close < min_2d)[2:].any()

    assert long_break, "Donchian long breakout must be possible with trailing-exclusive rolling max"
    assert not short_break, "Donchian short breakout must not fire when price is monotonically rising"


def test_candidate_panels_to_events_creates_dataframe() -> None:
    aligned = _make_aligned()
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    # Convert to events
    events = candidate_panels_to_events(panels, min_abs_score=0.0)

    assert isinstance(events, pd.DataFrame)
    if not events.empty:
        required_cols = {
            "datetime",
            "symbol",
            "family",
            "variant",
            "side",
            "raw_score",
            "score_z",
            "expected_holding_bars",
            "min_holding_bars",
            "stop_atr_mult",
            "take_profit_atr_mult",
            "turnover_proxy",
            "cost_floor_bps",
            "entry_idx",
            "exit_policy_id",
            "signal_cell",
            "archetype",
            "entry_regime",
            "entry_regime_code",
        }
        assert required_cols.issubset(events.columns)
        assert (events["entry_idx"] > 0).all()
        assert events["signal_cell"].astype(str).str.contains(":").all()


def test_candidate_panels_to_events_normalizes_score_z_per_bar() -> None:
    datetimes = np.array(
        [
            np.datetime64("2025-01-01T00"),
            np.datetime64("2025-01-01T04"),
        ]
    )
    scores = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    panel = CandidateSignalPanel(
        family="trend_ma",
        variant="unit",
        params={},
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        signed_score_2d=scores,
        side_hint_2d=np.array([[0, 0], [1, 1]], dtype=np.int8),
        expected_holding_bars=8,
        min_holding_bars=1,
        stop_atr_mult=1.0,
        take_profit_atr_mult=1.0,
        turnover_proxy_2d=np.zeros_like(scores),
        valid_mask_2d=np.array([[False, False], [True, True]], dtype=bool),
        metadata={},
        archetype="trend",
        allowed_regimes=("bull_quiet",),
        exit_policies=(
            SignalExitPolicy(
                policy_id="base",
                archetype="trend",
                stop_atr_mult=1.0,
                take_profit_atr_mult=1.0,
                expected_holding_bars=8,
                min_holding_bars=1,
                description="base",
            ),
        ),
        regime_code_1d=np.array([4, 0], dtype=np.int8),
        regime_name_by_code=("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash"),
    )

    events = candidate_panels_to_events((panel,), min_abs_score=0.0)

    assert events["raw_score"].tolist() == [1.0, 2.0]
    assert events["score_z"].iloc[0] == pytest.approx(-0.6744907594765952)
    assert events["score_z"].iloc[1] == pytest.approx(0.6744907594765952)


def test_candidate_panels_to_events_uses_max_of_policy_floor_and_physical_cost() -> None:
    aligned = _make_aligned()
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    execution_cost_bps_2d = np.full((150, 2), 40.0, dtype=np.float64)
    events = candidate_panels_to_events(
        panels,
        min_abs_score=0.0,
        cost_floor_bps=24.0,
        execution_cost_bps_2d=execution_cost_bps_2d,
    )

    assert not events.empty
    assert float(events["cost_floor_bps"].min()) >= 40.0


def test_build_rule_signal_panels_respects_entry_warm_and_block_masks() -> None:
    aligned = _make_aligned()
    t, n = aligned.close_2d.shape
    blocked = np.zeros((t, n), dtype=bool)
    blocked[:20] = True
    warm = np.ones((t, n), dtype=bool)
    warm[:20] = False
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=aligned.active_mask,
        warm_mask=warm,
        entry_block_mask=blocked,
        kill_mask=aligned.kill_mask,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )

    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())

    for panel in panels:
        assert not panel.valid_mask_2d[:20].any()


def test_build_rule_signal_panels_prefers_inference_active_mask_for_l1_scope() -> None:
    aligned = _make_aligned()
    t, n = aligned.close_2d.shape
    active_mask = np.zeros((t, n), dtype=bool)
    inference_active_mask = np.ones((t, n), dtype=bool)
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        inference_active_mask=inference_active_mask,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )

    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())

    assert panels
    for panel in panels:
        assert panel.valid_mask_2d.any()


def test_build_rule_signal_panels_applies_optional_eligibility_masks() -> None:
    aligned = _make_aligned()
    t, n = aligned.close_2d.shape
    execution_eligibility_mask = np.ones((t, n), dtype=bool)
    strategy_readiness_mask = np.ones((t, n), dtype=bool)
    promotion_active_mask = np.ones((t, n), dtype=bool)
    execution_eligibility_mask[:, 0] = False
    strategy_readiness_mask[30:60, 1] = False
    promotion_active_mask[80:, :] = False
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=aligned.active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        execution_eligibility_mask=execution_eligibility_mask,
        strategy_readiness_mask=strategy_readiness_mask,
        promotion_active_mask=promotion_active_mask,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )

    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())

    assert panels
    for panel in panels:
        assert not panel.valid_mask_2d[:, 0].any()
        assert not panel.valid_mask_2d[30:60, 1].any()
        assert not panel.valid_mask_2d[80:, :].any()


def test_new_signal_families_shapes_and_side_hints() -> None:
    # Arrange
    aligned = _make_aligned(t=200)
    cfg = CandidateStrategyConfig()

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    # Assert: 각 신규 family 패널의 shape 및 side_hint 범위 검증
    new_families = {
        "funding_zscore_carry",
        "vol_regime_reversion",
    }
    new_panels = [p for p in panels if p.family in new_families]
    assert len(new_panels) == 5  # 3+2

    for p in new_panels:
        assert p.signed_score_2d.shape == (200, 2), f"{p.family}:{p.variant} score shape mismatch"
        assert p.side_hint_2d.shape == (200, 2), f"{p.family}:{p.variant} side shape mismatch"
        assert p.valid_mask_2d.shape == (200, 2)
        # side_hint must be in {-1, 0, 1}
        unique_sides = {int(v) for v in np.unique(p.side_hint_2d)}
        assert unique_sides.issubset({-1, 0, 1}), (
            f"{p.family}:{p.variant} invalid side_hint values: {unique_sides}"
        )
        # signed_score must be in [-1, 1]
        finite_scores = p.signed_score_2d[np.isfinite(p.signed_score_2d)]
        if finite_scores.size > 0:
            assert float(np.max(np.abs(finite_scores))) <= 1.0 + 1e-6, (
                f"{p.family}:{p.variant} score out of [-1,1]"
            )




def test_dual_momentum_no_lookahead() -> None:
    aligned = _make_aligned(t=140)
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    dm_panels = [p for p in panels if p.family == "dual_momentum"]
    assert len(dm_panels) == 2

    for panel in dm_panels:
        long_lb = int(panel.params["long_lookback"])
        assert (panel.side_hint_2d[:long_lb] == 0).all(), (
            f"dual_momentum:{panel.variant} has non-zero side_hint in warmup bars [0:{long_lb}]"
        )


def test_new_signal_families_include_metadata_contract() -> None:
    aligned = _make_aligned(t=220)
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    expected_new_families = {
        "trend_pullback_continuation",
        "dual_momentum",
        "residual_reversion",
        "mtf_trend_pullback",
        "mtf_breakout_retest",
        "taker_imbalance_momentum",
        "funding_extreme_reversal",
        "vol_term_structure_gate",
    }
    matched = [panel for panel in panels if panel.family in expected_new_families]
    assert len(matched) == 14
    for panel in matched:
        assert panel.metadata["archetype"]
        assert panel.metadata["regime"]
        assert panel.metadata["edge_hypothesis"]
        assert panel.exit_policies
        assert panel.allowed_regimes
        assert panel.regime_code_1d is not None


def test_candidate_panels_to_events_uses_entry_regime_for_exit_policy() -> None:
    aligned = _make_aligned(t=6, n=1)
    panel = CandidateSignalPanel(
        family="trend_pullback_continuation",
        variant="tpc_20_100",
        params={},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=np.ones((6, 1), dtype=np.float64),
        side_hint_2d=np.array([[0], [1], [1], [1], [0], [0]], dtype=np.int8),
        expected_holding_bars=8,
        min_holding_bars=2,
        stop_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=np.zeros((6, 1), dtype=np.float64),
        valid_mask_2d=np.ones((6, 1), dtype=bool),
        archetype="trend",
        allowed_regimes=("bull_quiet", "bull_volatile", "crash"),
        exit_policies=(
            SignalExitPolicy("trend_grind", "trend", 1.25, 3.5, 8, 2),
            SignalExitPolicy("trend_fast_fail", "trend", 0.9, 2.25, 8, 2),
        ),
        regime_code_1d=np.array([0, 1, 5, 0, 0, 0], dtype=np.int8),
        regime_name_by_code=("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash"),
    )
    events = candidate_panels_to_events((panel,), min_abs_score=0.0)

    assert not events.empty
    volatile_events = events.loc[events["entry_regime"].isin(["bull_volatile", "bear_volatile", "crash"])]
    if not volatile_events.empty:
        assert "trend_fast_fail" in set(volatile_events["exit_policy_id"].astype(str))


# ---------------------------------------------------------------------------
# Group A: _entry_rising_edge_2d unit tests
# ---------------------------------------------------------------------------


def test_entry_rising_edge_2d_only_false_to_true_transition() -> None:
    # Arrange
    # Rows: F, T, T, T, F, T — only rows 1 and 5 are transitions
    cond = np.array(
        [[False, False], [True, True], [True, True], [True, True], [False, False], [True, True]],
        dtype=bool,
    )

    # Act
    result = _entry_rising_edge_2d(cond)

    # Assert: only row 1 and row 5 should be True
    expected = np.array(
        [[False, False], [True, True], [False, False], [False, False], [False, False], [True, True]],
        dtype=bool,
    )
    np.testing.assert_array_equal(result, expected)


def test_entry_rising_edge_2d_first_row_always_false() -> None:
    # Arrange — first row is True
    cond = np.ones((5, 3), dtype=bool)

    # Act
    result = _entry_rising_edge_2d(cond)

    # Assert: first row must always be False regardless of condition value
    assert not result[0].any(), "first row must always be False (no prior state)"


def test_entry_rising_edge_2d_shape_preserved() -> None:
    # Arrange
    t, n = 30, 4
    cond = np.random.default_rng(42).integers(0, 2, size=(t, n)).astype(bool)

    # Act
    result = _entry_rising_edge_2d(cond)

    # Assert
    assert result.shape == (t, n)
    assert result.dtype == bool


def test_entry_rising_edge_2d_multiple_transitions() -> None:
    # Arrange — alternating F/T pattern: each True is a new transition
    cond = np.array(
        [[False], [True], [False], [True], [False], [True]],
        dtype=bool,
    )

    # Act
    result = _entry_rising_edge_2d(cond)

    # Assert: rows 1, 3, 5 are transitions
    expected = np.array([[False], [True], [False], [True], [False], [True]], dtype=bool)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Group B: family-level sparse-entry (no-refire) tests
# ---------------------------------------------------------------------------


def test_bollinger_reversion_no_refire_during_persistent_state() -> None:
    # Arrange — craft price series that stays below BB lower band for many bars
    t, n = 100, 2
    # Start at 100, then drop sharply and stay low so bb_z < -2.0 for many consecutive bars
    close_flat = np.full((t, n), 100.0, dtype=np.float64)
    close_flat[30:, :] = 60.0  # persistent extreme low

    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close_flat.copy(),
        high_2d=close_flat * 1.01,
        low_2d=close_flat * 0.99,
        close_2d=close_flat.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    bb_panels = [p for p in panels if p.family == "bollinger_reversion"]
    assert bb_panels, "bollinger_reversion panel must exist"

    for panel in bb_panels:
        long_entries = panel.side_hint_2d == 1
        # In a persistent extreme-low regime, the same symbol/column must not have
        # more than 1 long entry per consecutive run (rising-edge semantics)
        for col in range(n):
            col_entries = long_entries[30:, col]
            # Count consecutive runs of 1s — each run must have exactly 1 True
            if col_entries.any():
                # No two adjacent bars should both be 1
                adjacent_fires = bool(np.any(col_entries[:-1] & col_entries[1:]))
                assert not adjacent_fires, (
                    f"bollinger_reversion col={col} re-fires on adjacent bars in persistent extreme state"
                )


def test_dual_momentum_no_refire_during_persistent_agreement() -> None:
    # Arrange — price series with strong persistent uptrend so momentum agreement stays on
    t, n = 150, 2
    # Monotonically increasing: ret_short_z > 0.5 and ret_long_z > 0.5 for many bars
    close = np.cumsum(np.ones((t, n), dtype=np.float64), axis=0) + 100.0
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    dm_panels = [p for p in panels if p.family == "dual_momentum"]
    assert dm_panels, "dual_momentum panel must exist"

    for panel in dm_panels:
        long_entries = panel.side_hint_2d == 1
        for col in range(n):
            col_entries = long_entries[:, col]
            if col_entries.any():
                adjacent_fires = bool(np.any(col_entries[:-1] & col_entries[1:]))
                assert not adjacent_fires, (
                    f"dual_momentum col={col} re-fires on adjacent bars in persistent agreement"
                )


def test_vol_regime_reversion_no_refire_during_persistent_high_vol() -> None:
    # Arrange — constant extreme volatility with consistent price direction
    # t >= 97 required: _log_return_2d max lag = 96
    t, n = 120, 2
    close = np.full((t, n), 100.0, dtype=np.float64)
    # Add large consistent moves to ensure ATR z > threshold persistently
    close[20:, :] = np.cumsum(np.full((t - 20, n), 5.0), axis=0) + 100.0
    high = close * 1.05
    low = close * 0.95
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=high,
        low_2d=low,
        close_2d=close.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    vr_panels = [p for p in panels if p.family == "vol_regime_reversion"]
    assert vr_panels, "vol_regime_reversion panel must exist"

    for panel in vr_panels:
        for side_val in (1, -1):
            side_entries = panel.side_hint_2d == side_val
            for col in range(n):
                col_entries = side_entries[:, col]
                if col_entries.any():
                    adjacent_fires = bool(np.any(col_entries[:-1] & col_entries[1:]))
                    assert not adjacent_fires, (
                        f"vol_regime_reversion col={col} side={side_val} re-fires on adjacent bars"
                    )




# ---------------------------------------------------------------------------
# Group C: score-side decoupling
# ---------------------------------------------------------------------------


def test_bollinger_reversion_score_side_decoupled() -> None:
    # Arrange — price stays below BB lower for multiple bars
    # t >= 97 required: _log_return_2d max lag = 96
    t, n = 120, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[25:, :] = 55.0  # stays at extreme low persistently
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    bb_panels = [p for p in panels if p.family == "bollinger_reversion"]
    assert bb_panels

    panel = bb_panels[0]
    # score must remain non-zero (level) while side_hint can be 0 after first entry bar
    zero_side_mask = panel.side_hint_2d[:, 0] == 0
    nonzero_score_mask = np.abs(panel.signed_score_2d[:, 0]) > 1e-6

    # In the persistent extreme zone, there should be bars where side=0 but score≠0
    decoupled_exists = bool(np.any(zero_side_mask & nonzero_score_mask))
    assert decoupled_exists, (
        "bollinger_reversion: expected bars where side_hint=0 but score≠0 (score-side decoupled)"
    )


# ---------------------------------------------------------------------------
# Group D: metadata contract tests
# ---------------------------------------------------------------------------


def test_touched_panels_expose_required_metadata() -> None:
    # Arrange
    aligned = _make_aligned(t=200)
    cfg = CandidateStrategyConfig()

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    required_keys = {"edge_hypothesis", "causal_inputs", "expected_failure_mode"}
    target_families = {"bollinger_reversion", "vol_regime_reversion", "btc_corr_regime"}

    matched = [p for p in panels if p.family in target_families]
    assert matched, "no panels found for target families"

    for panel in matched:
        missing = required_keys - set(panel.metadata.keys())
        assert not missing, (
            f"{panel.family}:{panel.variant} missing metadata keys: {missing}"
        )


def test_mean_rev_gated_out_of_trending_regime() -> None:
    panel = CandidateSignalPanel(
        family="bollinger_reversion",
        variant="unit",
        params={},
        datetimes=np.array([np.datetime64("2025-01-01T00"), np.datetime64("2025-01-01T04")]),
        symbols=("BTCUSDT",),
        signed_score_2d=np.ones((2, 1), dtype=np.float64),
        side_hint_2d=np.ones((2, 1), dtype=np.int8),
        expected_holding_bars=4,
        min_holding_bars=1,
        stop_atr_mult=1.0,
        take_profit_atr_mult=1.0,
        turnover_proxy_2d=np.zeros((2, 1), dtype=np.float64),
        valid_mask_2d=np.ones((2, 1), dtype=bool),
        metadata={},
    )
    regime_ctx = MarketRegimeContext(
        code_1d=np.array([1, 4], dtype=np.int8),
        name_by_code=("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash"),
        trend_score_1d=np.zeros(2, dtype=np.float64),
        vol_z_1d=np.zeros(2, dtype=np.float64),
        dispersion_z_1d=np.zeros(2, dtype=np.float64),
    )

    out = _attach_signal_context(
        (panel,),
        cfg=CandidateStrategyConfig(mean_rev_gating_enabled=True),
        regime_ctx=regime_ctx,
    )

    assert out[0].side_hint_2d[:, 0].tolist() == [0, 1]
