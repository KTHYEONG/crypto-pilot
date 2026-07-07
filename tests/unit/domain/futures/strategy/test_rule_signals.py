from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.signals.rules import ALL_SIGNAL_FAMILIES
from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel, SignalExitPolicy
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.market_regime import MarketRegimeContext
from src.domain.futures.strategy.rule_signals import (
    _attach_signal_context,
    _beta_residual_return_2d,
    _cross_sectional_rank_signed_2d,
    _cross_sectional_robust_zscore,
    _entry_rising_edge_2d,
    _rolling_max_2d,
    _rolling_min_2d,
    _safe_taker_imbalance_2d,
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


def _make_flow_aligned(
    *,
    close: np.ndarray,
    funding: np.ndarray,
    taker_buy: np.ndarray | None,
    volume: np.ndarray | None = None,
    oi: np.ndarray | None = None,
    lsr: np.ndarray | None = None,
) -> AlignedMarketData:
    """Build a deterministic aligned fixture for flow-family tests."""
    t, n = close.shape
    if volume is None:
        volume = np.full((t, n), 1000.0, dtype=np.float64)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=tuple(f"SYM{i}" for i in range(n)),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close.copy(),
        volume_2d=volume,
        funding_2d=funding,
        oi_2d=oi,
        lsr_2d=lsr,
        taker_buy_2d=taker_buy,
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
    assert len(panels) > 0

    expected_families = {
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "funding_carry",
        "btc_regime_pullback",
        "trend_pullback_continuation",
        "dual_momentum",
        "residual_reversion",
        "xs_momentum",
        "xs_carry",
        "xs_flow",
        "xs_oi_skew",
        "mtf_trend_pullback",
        "mtf_breakout_retest",
        "taker_imbalance_momentum",
        "funding_flow_carry",
        "funding_flow_unwind",
        "flow_exhaustion_reversal",
        "funding_extreme_reversal",
        "vol_term_structure_gate",
        "funding_term_structure_carry",
        "flow_trend_continuation",
        "lsr_oi_regime_filter",
        "macd_4h",
        "supertrend",
        "ichimoku_trend",
        "positioning_unwind",
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


def test_candidate_strategy_config_defaults_include_new_flow_families() -> None:
    cfg = CandidateStrategyConfig()

    expected_families = {
        "funding_flow_carry",
        "funding_flow_unwind",
        "flow_exhaustion_reversal",
        "funding_term_structure_carry",
        "flow_trend_continuation",
        "lsr_oi_regime_filter",
    }
    assert expected_families.issubset(set(cfg.candidate_families))
    assert expected_families.issubset(set(cfg.ensemble_variant_prior_families))


def test_safe_taker_imbalance_2d_handles_valid_and_invalid_inputs() -> None:
    volume = np.array([[100.0, 100.0]], dtype=np.float64)
    taker_buy = np.array([[75.0, 25.0]], dtype=np.float64)

    imbalance, valid = _safe_taker_imbalance_2d(taker_buy, volume)

    np.testing.assert_allclose(imbalance, np.array([[0.5, -0.5]], dtype=np.float64))
    np.testing.assert_array_equal(valid, np.ones((1, 2), dtype=bool))

    invalid_inputs: tuple[tuple[np.ndarray | None, np.ndarray], ...] = (
        (None, volume),
        (np.array([[np.nan, np.nan]], dtype=np.float64), volume),
        (np.array([[10.0, 10.0]], dtype=np.float64), np.array([[0.0, 0.0]], dtype=np.float64)),
        (np.array([[-1.0, -1.0]], dtype=np.float64), volume),
        (np.array([[103.0, 103.0]], dtype=np.float64), volume),
    )

    for taker, vol in invalid_inputs:
        imbalance_i, valid_i = _safe_taker_imbalance_2d(taker, vol)
        assert imbalance_i.shape == vol.shape
        assert valid_i.shape == vol.shape
        assert imbalance_i.dtype == np.float64
        assert valid_i.dtype == bool
        assert not valid_i.any()
        assert not np.any(imbalance_i)


def test_safe_taker_imbalance_2d_rejects_shape_mismatch() -> None:
    volume = np.ones((2, 2), dtype=np.float64)
    taker_buy = np.ones((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="taker_buy and volume shapes must match"):
        _safe_taker_imbalance_2d(taker_buy, volume)


def test_funding_flow_carry_emits_short_entry_with_shifted_event_offset() -> None:
    t, n = 220, 1
    close = np.linspace(100.0, 124.0, t, dtype=np.float64).reshape(t, n)
    funding = np.zeros((t, n), dtype=np.float64)
    funding[130:, :] = 4.0
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[130:170, :] = 400.0
    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)

    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    carry_panels = [panel for panel in panels if panel.family == "funding_flow_carry"]

    assert len(carry_panels) == 1
    panel = carry_panels[0]
    fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)
    assert fired.tolist() == [133]
    assert panel.side_hint_2d[133, 0] == -1
    assert panel.side_hint_2d[134:, 0].sum() == 0

    events = candidate_panels_to_events((panel,), min_abs_score=0.0)
    assert events["entry_idx"].tolist() == [134]
    assert events["side"].tolist() == [-1]


def test_funding_flow_carry_rejects_opposite_flow_confirmation() -> None:
    t, n = 220, 1
    close = np.linspace(100.0, 124.0, t, dtype=np.float64).reshape(t, n)
    funding = np.zeros((t, n), dtype=np.float64)
    funding[130:, :] = 4.0
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[130:170, :] = 600.0
    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)

    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    carry_panels = [panel for panel in panels if panel.family == "funding_flow_carry"]

    assert len(carry_panels) == 1
    assert not carry_panels[0].side_hint_2d.any()
    assert candidate_panels_to_events((carry_panels[0],), min_abs_score=0.0).empty


def test_funding_flow_unwind_emits_short_entry_only_on_reversal_bar() -> None:
    t, n = 260, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[150:170, :] = np.linspace(100.0, 140.0, 20, dtype=np.float64).reshape(20, 1)
    close[170, 0] = close[169, 0] - 0.5
    close[171:, 0] = close[170, 0]
    funding = np.zeros((t, n), dtype=np.float64)
    funding[130:, :] = 4.0
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[150:170, :] = 1000.0
    taker_buy[170, :] = 0.0
    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)

    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    unwind_panels = [panel for panel in panels if panel.family == "funding_flow_unwind"]

    assert len(unwind_panels) == 1
    panel = unwind_panels[0]
    fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)
    assert fired.tolist() == [170]
    assert panel.side_hint_2d[170, 0] == -1

    events = candidate_panels_to_events((panel,), min_abs_score=0.0)
    assert events["entry_idx"].tolist() == [171]
    assert events["side"].tolist() == [-1]


def test_flow_exhaustion_reversal_emits_short_entry_on_same_bar_exhaustion_reversal() -> None:
    t, n = 240, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[150:170, :] = np.linspace(100.0, 140.0, 20, dtype=np.float64).reshape(20, 1)
    close[170, 0] = close[169, 0] - 0.5
    close[171:, 0] = close[170, 0]
    funding = np.zeros((t, n), dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[150:169, 0] = 500.0
    taker_buy[170, 0] = 1000.0
    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)

    panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
    exhaustion_panels = [panel for panel in panels if panel.family == "flow_exhaustion_reversal"]

    assert len(exhaustion_panels) == 1
    panel = exhaustion_panels[0]
    assert panel.side_hint_2d[0, 0] == 0
    fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)
    assert fired.tolist() == [170]
    assert panel.side_hint_2d[170, 0] == -1

    events = candidate_panels_to_events((panel,), min_abs_score=0.0)
    assert events["entry_idx"].tolist() == [171]
    assert events["side"].tolist() == [-1]


def test_flow_exhaustion_reversal_rejects_opposite_direction_flow_and_missing_funding() -> None:
    t, n = 240, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[150:170, :] = np.linspace(100.0, 140.0, 20, dtype=np.float64).reshape(20, 1)
    close[170, 0] = close[169, 0] - 0.5
    close[171:, 0] = close[170, 0]
    funding = np.full((t, n), np.nan, dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[150:170, 0] = 1000.0
    taker_buy[170, 0] = 500.0

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("flow_exhaustion_reversal",)),
    )

    assert len(panels) == 1
    assert not panels[0].side_hint_2d.any()


def test_flow_exhaustion_reversal_score_preserves_magnitude_ordering() -> None:
    t, n = 260, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[150:170, :] = np.linspace(100.0, 140.0, 20, dtype=np.float64).reshape(20, 1)
    close[170, 0] = close[169, 0] - 0.5
    close[200:220, :] = np.linspace(100.0, 125.0, 20, dtype=np.float64).reshape(20, 1)
    close[220, 0] = close[219, 0] - 0.25
    close[221:, 0] = close[220, 0]
    funding = np.zeros((t, n), dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[150:170, 0] = 500.0
    taker_buy[170, 0] = 1000.0
    taker_buy[200:220, 0] = 500.0
    taker_buy[220, 0] = 1000.0

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)
    panel = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("flow_exhaustion_reversal",)),
    )[0]
    fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)

    assert fired.tolist() == [170]
    assert 0.0 < abs(panel.signed_score_2d[170, 0]) < 1.0


def test_positioning_unwind_requires_joint_positioning_inputs() -> None:
    t, n = 260, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[150:170, :] = np.linspace(100.0, 140.0, 20, dtype=np.float64).reshape(20, 1)
    close[170, 0] = close[169, 0] - 0.5
    close[171:, 0] = close[170, 0]
    funding = np.zeros((t, n), dtype=np.float64)
    funding[130:, :] = 20.0
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[150:170, :] = 1000.0
    taker_buy[170, :] = 0.0
    oi = np.full((t, n), 100.0, dtype=np.float64)
    oi[160:170, :] = np.linspace(100.0, 5000.0, 10, dtype=np.float64).reshape(10, 1)
    oi[170, :] = 40000.0
    lsr = np.full((t, n), 1.0, dtype=np.float64)
    lsr[160:170, :] = np.linspace(1.0, 15.0, 10, dtype=np.float64).reshape(10, 1)
    lsr[170, :] = 20.0

    panel = build_rule_signal_panels(
        aligned=_make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy, oi=oi, lsr=lsr),
        cfg=CandidateStrategyConfig(candidate_families=("positioning_unwind",)),
    )[0]
    fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)

    assert panel.family == "positioning_unwind"
    assert fired.tolist() == [170]
    assert panel.side_hint_2d[170, 0] == -1


def test_flow_families_become_empty_when_taker_data_is_missing() -> None:
    t, n = 220, 1
    close = np.linspace(100.0, 124.0, t, dtype=np.float64).reshape(t, n)
    funding = np.zeros((t, n), dtype=np.float64)
    aligned_none = _make_flow_aligned(close=close, funding=funding, taker_buy=None)
    aligned_nan = _make_flow_aligned(
        close=close,
        funding=funding,
        taker_buy=np.full((t, n), np.nan, dtype=np.float64),
    )

    for aligned in (aligned_none, aligned_nan):
        panels = build_rule_signal_panels(aligned=aligned, cfg=CandidateStrategyConfig())
        flow_panels = [
            panel
            for panel in panels
            if panel.family
            in {
                "funding_flow_carry",
                "funding_flow_unwind",
                "flow_exhaustion_reversal",
            }
        ]
        assert len(flow_panels) == 3
        for panel in flow_panels:
            assert not panel.side_hint_2d.any()
            assert candidate_panels_to_events((panel,), min_abs_score=0.0).empty


def test_build_rule_signal_panels_filters_new_flow_family_by_config() -> None:
    aligned = _make_flow_aligned(
        close=np.linspace(100.0, 124.0, 220, dtype=np.float64).reshape(220, 1),
        funding=np.zeros((220, 1), dtype=np.float64),
        taker_buy=np.full((220, 1), 500.0, dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(candidate_families=("funding_flow_carry",))

    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    assert {panel.family for panel in panels} == {"funding_flow_carry"}
    assert {panel.variant for panel in panels} == {"ffc_96"}


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
    trend_panel = next(panel for panel in panels if panel.family == "trend_ma")
    assert trend_panel.valid_mask_2d.any()


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
        "xs_momentum",
        "xs_carry",
        "xs_flow",
        "xs_oi_skew",
        "funding_flow_carry",
        "funding_flow_unwind",
        "flow_exhaustion_reversal",
    }
    new_panels = [p for p in panels if p.family in new_families]

    for p in new_panels:
        assert p.signed_score_2d.shape == (200, 2), f"{p.family}:{p.variant} score shape mismatch"
        assert p.side_hint_2d.shape == (200, 2), f"{p.family}:{p.variant} side shape mismatch"
        assert p.valid_mask_2d.shape == (200, 2)
        # side_hint must be in {-1, 0, 1}
        unique_sides = {int(v) for v in np.unique(p.side_hint_2d)}
        assert unique_sides.issubset({-1, 0, 1}), f"{p.family}:{p.variant} invalid side_hint values: {unique_sides}"
        # signed_score must be in [-1, 1]
        finite_scores = p.signed_score_2d[np.isfinite(p.signed_score_2d)]
        if finite_scores.size > 0:
            assert float(np.max(np.abs(finite_scores))) <= 1.0 + 1e-6, f"{p.family}:{p.variant} score out of [-1,1]"


def test_new_flow_signal_families_include_metadata_contract() -> None:
    aligned = _make_aligned(t=220)
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    flow_panels = [
        panel
        for panel in panels
        if panel.family
        in {
            "funding_flow_carry",
            "funding_flow_unwind",
            "flow_exhaustion_reversal",
        }
    ]
    assert len(flow_panels) == 3

    for panel in flow_panels:
        assert panel.metadata["archetype"] in {"carry_rev", "unwind", "flow_rev"}
        assert panel.metadata["regime"]
        assert panel.metadata["causal_inputs"]
        assert panel.metadata["edge_hypothesis"]
        assert panel.exit_policies
        assert panel.allowed_regimes
        assert panel.regime_code_1d is not None


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
                assert not adjacent_fires, f"dual_momentum col={col} re-fires on adjacent bars in persistent agreement"


# ---------------------------------------------------------------------------
# Group D: metadata contract tests
# ---------------------------------------------------------------------------


def test_touched_panels_expose_required_metadata() -> None:
    # Arrange
    aligned = _make_aligned(t=200)
    cfg = CandidateStrategyConfig()

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    target_families = {"funding_flow_carry", "funding_flow_unwind"}

    matched = [p for p in panels if p.family in target_families]
    assert matched, "no panels found for target families"

    for panel in matched:
        assert "edge_hypothesis" in panel.metadata
        assert "causal_inputs" in panel.metadata


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


def _beta_neut_panel() -> CandidateSignalPanel:
    return CandidateSignalPanel(
        family="residual_reversion",
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
        metadata={"archetype": "beta_neut"},
    )


def _regime_ctx(codes: list[int]) -> MarketRegimeContext:
    return MarketRegimeContext(
        code_1d=np.array(codes, dtype=np.int8),
        name_by_code=("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash"),
        trend_score_1d=np.zeros(len(codes), dtype=np.float64),
        vol_z_1d=np.zeros(len(codes), dtype=np.float64),
        dispersion_z_1d=np.zeros(len(codes), dtype=np.float64),
    )


def test_beta_neut_gated_allowed_in_bull_quiet() -> None:
    panel = _beta_neut_panel()
    regime_ctx = _regime_ctx([0, 0])
    out = _attach_signal_context(
        (panel,),
        cfg=CandidateStrategyConfig(beta_neut_gating_enabled=True),
        regime_ctx=regime_ctx,
    )
    assert out[0].side_hint_2d[:, 0].tolist() == [1, 1]


def test_beta_neut_gated_blocked_in_bear_quiet() -> None:
    panel = _beta_neut_panel()
    regime_ctx = _regime_ctx([2, 2])
    out = _attach_signal_context(
        (panel,),
        cfg=CandidateStrategyConfig(beta_neut_gating_enabled=True),
        regime_ctx=regime_ctx,
    )
    assert out[0].side_hint_2d[:, 0].tolist() == [0, 0]


def test_beta_neut_gating_disabled_by_default_no_regression() -> None:
    panel = _beta_neut_panel()
    regime_ctx = _regime_ctx([2, 2])
    out = _attach_signal_context(
        (panel,),
        cfg=CandidateStrategyConfig(),
        regime_ctx=regime_ctx,
    )
    assert out[0].side_hint_2d[:, 0].tolist() == [1, 1]


def test_beta_neut_masked_by_global_regime_signal_gating_override() -> None:
    panel = _beta_neut_panel()
    regime_ctx = _regime_ctx([2, 2])
    out = _attach_signal_context(
        (panel,),
        cfg=CandidateStrategyConfig(regime_signal_gating_enabled=True, beta_neut_gating_enabled=False),
        regime_ctx=regime_ctx,
    )
    assert out[0].side_hint_2d[:, 0].tolist() == [0, 0]


# ── G9e. funding_term_structure_carry ─────────────────────────────────────


def test_funding_term_structure_carry_entry_detection() -> None:
    """Scenario 1: funding_z_96 > funding_z_168, same sign, above threshold -> entry.

    Funding drops from 10.0 to 0.0 at bar 168-200, creating:
    - z_96 (short-term) less negative than z_168 (long-term has higher mean)
    - slope = z_96 - z_168 > 0 (positive acceleration in bearish direction)
    """
    t, n = 400, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    funding = np.full((t, n), 0.0, dtype=np.float64)
    funding[:168, 0] = 10.0  # high plateau
    funding[168:200, 0] = np.linspace(10.0, 0.0, 32)  # sharp decline
    funding[200:, 0] = 0.0  # low plateau
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("funding_term_structure_carry",)),
    )
    panel = panels[0]
    assert panel.family == "funding_term_structure_carry"
    fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)
    assert len(fired) >= 1, "expected at least one entry under funding acceleration"
    # After sharp drop, long-term z is negative -> side = -1
    assert panel.side_hint_2d[fired[0], 0] == -1  # short entry (bearish acceleration)
    # Score magnitude negative because side=-1 (short); check abs > 0
    assert np.abs(panel.signed_score_2d[fired[0], 0]) > 0.0


def test_funding_term_structure_carry_opposite_direction() -> None:
    """Scenario 2: funding_z_96 and funding_z_168 have opposite signs -> no entry."""
    t, n = 260, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    funding = np.full((t, n), 0.0, dtype=np.float64)
    funding[50:100, 0] = np.linspace(0.0, -0.8, 50, dtype=np.float64)  # long-term negative
    funding[100:150, 0] = np.linspace(-0.8, 2.0, 50, dtype=np.float64)  # short-term positive flip
    funding[150:, 0] = 2.0
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("funding_term_structure_carry",)),
    )
    panel = panels[0]
    assert panel.family == "funding_term_structure_carry"
    assert not panel.side_hint_2d[:, 0].any(), "opposite signs should block all entries"


# ── G9f. flow_trend_continuation ─────────────────────────────────────────


def test_flow_trend_continuation_entry_detection() -> None:
    """Scenario 3: flow_z >= 1.0, ret_12 > 0, ret_1 > 0 -> long-only entry."""
    t, n = 250, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[120:, 0] = np.linspace(100.0, 130.0, t - 120, dtype=np.float64)  # rising
    funding = np.zeros((t, n), dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[100:130, 0] = np.linspace(500.0, 950.0, 30, dtype=np.float64)
    taker_buy[130:150, 0] = 950.0  # sustained high flow

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("flow_trend_continuation",)),
    )
    panel = panels[0]
    assert panel.family == "flow_trend_continuation"
    assert panel.metadata["archetype"] == "ts_mom", "flow_trend_continuation must route to ts_mom not flow_rev"
    fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)
    assert len(fired) >= 1, "expected continuation entries with strong flow + uptrend"
    assert panel.side_hint_2d[fired[0], 0] == 1  # long-only
    assert panel.signed_score_2d[fired[0], 0] > 0.0


def test_flow_trend_continuation_rejects_negative_trend() -> None:
    """Scenario 4: ret_12 < 0 (downtrend) -> no entry despite strong flow."""
    t, n = 250, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[120:, 0] = np.linspace(100.0, 70.0, t - 120, dtype=np.float64)  # falling
    funding = np.zeros((t, n), dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[100:150, 0] = np.linspace(500.0, 950.0, 50, dtype=np.float64)

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy)
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("flow_trend_continuation",)),
    )
    panel = panels[0]
    assert panel.family == "flow_trend_continuation"
    assert not panel.side_hint_2d[:, 0].any(), "downtrend must block all continuations"


# ── G9g. lsr_oi_regime_filter ────────────────────────────────────────────


def test_lsr_oi_regime_filter_activation() -> None:
    """Scenario 5: lsr_log_z_42 extreme + oi_build_z_42 rising -> regime active.

    Uses accelerating exponential growth (exp(quadratic)) so that log-change
    increases over time, keeping z-scores positive after warm-up clears.
    """
    t, n = 280, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    funding = np.full((t, n), 0.0, dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    oi = np.full((t, n), 100.0, dtype=np.float64)
    lsr = np.full((t, n), 1.0, dtype=np.float64)
    # Accelerating exponential growth from bar 100 to 180
    ramp = np.arange(80, dtype=np.float64)
    oi_rise = 100.0 * np.exp(5.0 * (ramp / 80.0) ** 3)
    lsr_rise = 1.0 * np.exp(4.0 * (ramp / 80.0) ** 3)
    oi[100:180, 0] = oi_rise
    lsr[100:180, 0] = lsr_rise
    oi[180:, 0] = oi_rise[-1]
    lsr[180:, 0] = lsr_rise[-1]

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy, oi=oi, lsr=lsr)
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("lsr_oi_regime_filter",)),
    )
    panel = panels[0]
    assert panel.family == "lsr_oi_regime_filter"
    fired = np.flatnonzero(panel.signed_score_2d[:, 0] > 0.0)
    assert len(fired) >= 1, "expected regime activation with extreme LSR + OI building"
    # Regime filter must produce directional side (fade the crowded LSR side)
    side_fired = np.flatnonzero(panel.side_hint_2d[:, 0] != 0)
    assert len(side_fired) >= 1, "lsr_oi_regime_filter must produce side_hint trades after activation"
    # LSR is extreme positive (crowded long) -> side = -1 (fade)
    assert panel.side_hint_2d[fired[0], 0] == -1


def test_lsr_oi_regime_filter_below_threshold() -> None:
    """Scenario 7: lsr_log_z_42 < 1.0, oi_build_z_42 < 0.5 -> no activation."""
    t, n = 260, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    funding = np.full((t, n), 0.0, dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    oi = np.full((t, n), 100.0, dtype=np.float64)
    lsr = np.full((t, n), 1.0, dtype=np.float64)  # flat LSR, never extreme

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy, oi=oi, lsr=lsr)
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("lsr_oi_regime_filter",)),
    )
    panel = panels[0]
    assert panel.family == "lsr_oi_regime_filter"
    assert not panel.signed_score_2d.any(), "below-threshold must produce zero score"


def test_lsr_oi_regime_filter_generates_live_events() -> None:
    """Scenario 5: lsr_oi_regime_filter must produce non-empty events via candidate_panels_to_events."""
    t, n = 280, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    funding = np.full((t, n), 0.0, dtype=np.float64)
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    oi = np.full((t, n), 100.0, dtype=np.float64)
    lsr = np.full((t, n), 1.0, dtype=np.float64)
    ramp = np.arange(80, dtype=np.float64)
    oi[100:180, 0] = 100.0 * np.exp(5.0 * (ramp / 80.0) ** 3)
    lsr[100:180, 0] = 1.0 * np.exp(4.0 * (ramp / 80.0) ** 3)
    oi[180:, 0] = oi[179, 0]
    lsr[180:, 0] = lsr[179, 0]

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy, oi=oi, lsr=lsr)
    panel = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("lsr_oi_regime_filter",)),
    )[0]
    events = candidate_panels_to_events((panel,), min_abs_score=0.0)
    assert not events.empty, "lsr_oi_regime_filter must produce events with active side_hint"
    assert events["family"].str.contains("lsr_oi_regime_filter").any()


# ── positioning_unwind warm-up barrier ────────────────────────────────────


def test_positioning_unwind_warm_up_barrier() -> None:
    """Scenario 4: bar_index < 168 blocked even when all features valid."""
    t, n = 280, 1
    close = np.full((t, n), 100.0, dtype=np.float64)
    close[60:80, :] = np.linspace(100.0, 140.0, 20, dtype=np.float64).reshape(20, 1)
    close[80, 0] = close[79, 0] - 0.5
    close[81:, 0] = close[80, 0]
    funding = np.full((t, n), 0.0, dtype=np.float64)
    funding[40:, :] = 20.0
    taker_buy = np.full((t, n), 500.0, dtype=np.float64)
    taker_buy[60:80, :] = 1000.0
    taker_buy[80, :] = 0.0
    oi = np.full((t, n), 100.0, dtype=np.float64)
    oi[50:80, :] = np.linspace(100.0, 5000.0, 30, dtype=np.float64).reshape(30, 1)
    oi[80, :] = 40000.0
    lsr = np.full((t, n), 1.0, dtype=np.float64)
    lsr[50:80, :] = np.linspace(1.0, 15.0, 30, dtype=np.float64).reshape(30, 1)
    lsr[80, :] = 20.0

    aligned = _make_flow_aligned(close=close, funding=funding, taker_buy=taker_buy, oi=oi, lsr=lsr)
    panel = build_rule_signal_panels(
        aligned=aligned,
        cfg=CandidateStrategyConfig(candidate_families=("positioning_unwind",)),
    )[0]

    # All bars < 168 must have valid_mask_2d=False
    assert not panel.valid_mask_2d[:168, :].any(), "bars before warm-up (index < 168) must be masked out"

    # Bars >= 168 may have valid_mask_2d=True depending on feature availability
    assert panel.valid_mask_2d[168:, 0].any() or panel.valid_mask_2d[168:, :].any(), (
        "bars after warm-up should be eligible when all features are valid"
    )


# =============================================================================
# TF-Specific Signal Pool tests (Spec: tf-signal-pools-v2.md)
# =============================================================================


def test_build_rule_signal_panels_family_filter() -> None:
    aligned = _make_aligned(t=100, n=2)
    cfg = CandidateStrategyConfig(
        candidate_families=("trend_ma", "funding_carry", "macd_4h", "supertrend"),
    )
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=cfg,
        family_filter=("trend_ma", "funding_carry"),
    )
    families = {p.family for p in panels}
    assert "trend_ma" in families
    assert "funding_carry" in families
    assert "macd_4h" not in families
    assert "supertrend" not in families


def test_build_rule_signal_panels_family_filter_none_backward_compat() -> None:
    aligned = _make_aligned(t=100, n=2)
    cfg = CandidateStrategyConfig()
    panels_without = build_rule_signal_panels(
        aligned=aligned,
        cfg=cfg,
        family_filter=None,
    )
    assert len(panels_without) >= 20  # all families (allow minor variance)


def test_build_rule_signal_panels_per_family_params() -> None:
    aligned = _make_aligned(t=100, n=2)
    overrides = {
        "funding_carry:funding_24": {"window": 8, "entry_z": 0.5},
    }
    cfg_with = CandidateStrategyConfig(per_family_params=overrides)
    cfg_without = CandidateStrategyConfig()

    panels_with = build_rule_signal_panels(
        aligned=aligned,
        cfg=cfg_with,
        family_filter=("funding_carry",),
    )
    panels_without = build_rule_signal_panels(
        aligned=aligned,
        cfg=cfg_without,
        family_filter=("funding_carry",),
    )

    p_with = next(p for p in panels_with if p.variant == "funding_24")
    p_without = next(p for p in panels_without if p.variant == "funding_24")

    assert p_with.params["window"] == 8
    assert p_with.params["entry_z"] == 0.5
    # default values differ
    assert p_without.params["window"] != 8 or p_without.params["entry_z"] != 0.5


def test_build_rule_signal_panels_supertrend_direction() -> None:
    from src.domain.futures.strategy.rule_signals import _supertrend_2d

    t = 60
    n = 1
    up = np.linspace(100.0, 120.0, t, dtype=np.float64).reshape(t, n)
    down = np.linspace(100.0, 80.0, t, dtype=np.float64).reshape(t, n)

    trend = _supertrend_2d(up * 1.01, up * 0.99, up, period=10, multiplier=2.5)
    # trend[0] = 0 (initial), after sustained uptrend should flip to +1
    assert trend[-1, 0] == 1, f"Expected +1 for steady uptrend, got {trend[-1, 0]}"

    trend_dn = _supertrend_2d(down * 1.01, down * 0.99, down, period=10, multiplier=2.5)
    assert trend_dn[-1, 0] == -1  # steady downtrend → -1


def test_build_rule_signal_panels_full_per_tf_pool() -> None:
    from src.domain.futures.strategy.config import _DEFAULT_PER_TF_FAMILIES

    aligned = _make_aligned(t=300, n=2)
    # Include all pool families so filter_rule_signal_panels doesn't strip them
    all_pool_families = tuple(sorted({f for pool in _DEFAULT_PER_TF_FAMILIES.values() for f in pool}))
    cfg = CandidateStrategyConfig(candidate_families=all_pool_families)
    pool_1h = _DEFAULT_PER_TF_FAMILIES["1h"]
    panels = build_rule_signal_panels(
        aligned=aligned,
        cfg=cfg,
        family_filter=pool_1h,
    )
    families = {p.family for p in panels}
    # 1h pool: residual_reversion + funding_carry + flow_exhaustion_reversal
    assert "residual_reversion" in families
    assert "funding_carry" in families
    assert "flow_exhaustion_reversal" in families
    # Trend families should NOT be in 1h pool
    assert "trend_ma" not in families
    assert "trend_donchian" not in families
    assert "dual_momentum" not in families


# ─── OPT: _robust_zscore_numba equivalence ─────────────────────────────────


def test_robust_zscore_numba_matches_original() -> None:
    """OPT-C: _robust_zscore_numba produces bit-exact results vs reference."""
    from scipy.stats import median_abs_deviation

    rng = np.random.default_rng(42)
    raw_scores = rng.normal(0, 1, (200,)).astype(np.float64)
    groups = np.repeat(np.arange(4, dtype=np.int64), 50)

    # Reference: pure-python median/MAD group loop
    expected = np.zeros(200, dtype=np.float64)
    for g in np.unique(groups):
        mask = groups == g
        vals = raw_scores[mask]
        fin = vals[np.isfinite(vals)]
        if fin.size == 0:
            continue
        med = float(np.median(fin))
        mad = float(median_abs_deviation(fin, scale=1.0)) * 1.4826
        out = np.zeros(vals.shape[0], dtype=np.float64)
        if mad > 1e-9:
            out[np.isfinite(vals)] = (vals[np.isfinite(vals)] - med) / mad
        expected[mask] = np.clip(out, -3.0, 3.0)

    actual = _cross_sectional_robust_zscore(raw_scores, groups)
    np.testing.assert_array_almost_equal(actual, expected, decimal=12)


def test_robust_zscore_numba_empty() -> None:
    """OPT-C: empty input produces empty output."""
    result = _cross_sectional_robust_zscore(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64),
    )
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_robust_zscore_numba_all_nan() -> None:
    """OPT-C: all-NaN groups produce zeros (no finite values to normalize)."""
    raw = np.full(20, np.nan, dtype=np.float64)
    groups = np.repeat(np.arange(2, dtype=np.int64), 10)
    result = _cross_sectional_robust_zscore(raw, groups)
    np.testing.assert_array_equal(result, np.zeros(20, dtype=np.float64))


# ─── OPT: candidate_panels_to_events output schema ─────────────────────────


def test_candidate_panels_to_events_output_columns_unchanged() -> None:
    """OPT-A/B: 출력 컬럼 스키마 변경 없음."""
    aligned = _make_aligned()
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)
    events = candidate_panels_to_events(panels, min_abs_score=0.0)
    expected_cols = {
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
        "side_flipped",
        "exit_policy_id",
        "signal_cell",
        "archetype",
        "entry_regime",
        "entry_regime_code",
    }
    assert expected_cols.issubset(events.columns), f"missing columns: {expected_cols - set(events.columns)}"


def test_candidate_panels_to_events_no_sort() -> None:
    """OPT-B: sort_values(\"datetime\") 제거 → crash 없이 실행."""
    aligned = _make_aligned()
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)
    events = candidate_panels_to_events(panels, min_abs_score=0.0)
    assert isinstance(events, pd.DataFrame)
    assert len(events) > 0


# ─── OPT: regime x policy pre-extraction correctness ───────────────────────


def test_candidate_panels_to_events_regime_policy_count() -> None:
    """OPT-A: regime x policy pre-extraction 이 event 수에 영향을 주지 않음."""
    aligned = _make_aligned()
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)
    events = candidate_panels_to_events(panels, min_abs_score=0.0)
    unique_cells = events["signal_cell"].nunique()
    assert unique_cells > 0
    assert events["exit_policy_id"].nunique() >= 1


# ─── XS Alpha helpers ──────────────────────────────────────────────────


def test_cross_sectional_rank_signed_monotonic_and_tercile() -> None:
    """Scenario 1: _cross_sectional_rank_signed_2d monotonic rank & tercile side."""
    raw_2d = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)
    valid_2d = np.ones_like(raw_2d, dtype=np.bool_)
    signed, side = _cross_sectional_rank_signed_2d(
        raw_2d,
        valid_2d,
        min_cross_section=5,
        top_q=0.70,
        bot_q=0.30,
    )
    assert signed.shape == (1, 6)
    assert side.shape == (1, 6)
    assert signed[0, 0] < signed[0, 1] < signed[0, 2] < signed[0, 3] < signed[0, 4] < signed[0, 5]
    assert signed[0, 0] == pytest.approx(-4.0 / 6.0, abs=1e-2)
    assert signed[0, 5] == pytest.approx(1.0, abs=1e-2)
    expected_side = np.array([[-1, 0, 0, 0, 1, 1]], dtype=np.int8)
    np.testing.assert_array_equal(side, expected_side)


def test_cross_sectional_rank_below_min_cross_section_zeroed() -> None:
    """Scenario 2: min_cross_section guard zeroes rows with insufficient valid symbols."""
    raw_2d = np.array(
        [[1.0, 2.0, 3.0, 4.0, np.nan, np.nan], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        dtype=np.float64,
    )
    valid_2d = np.array(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]],
        dtype=np.bool_,
    )
    signed, side = _cross_sectional_rank_signed_2d(
        raw_2d,
        valid_2d,
        min_cross_section=5,
        top_q=0.70,
        bot_q=0.30,
    )
    np.testing.assert_array_equal(signed[0], np.zeros(6, dtype=np.float64))
    np.testing.assert_array_equal(side[0], np.zeros(6, dtype=np.int8))
    assert np.any(signed[1] != 0)


def test_xs_momentum_common_beta_move_produces_no_signal() -> None:
    """Scenario 3: beta-neutrality — common BTC move yields ~0 residual return."""
    t, n = 500, 4
    btc_idx = 0
    rng = np.random.default_rng(42)
    btc_ret = rng.normal(0.0, 0.01, size=(t, 1))
    beta = 1.5
    alt_ret = beta * btc_ret
    log_ret = np.zeros((t, n), dtype=np.float64)
    log_ret[:, 0:1] = btc_ret
    log_ret[:, 1:] = alt_ret
    cum_ret = np.cumsum(log_ret, axis=0)
    close = np.exp(cum_ret) * 100.0
    lookback = 48
    resid_sum = _beta_residual_return_2d(close, btc_idx, lookback)
    nonzero_cols = resid_sum[lookback:, 1:]
    max_abs = np.max(np.abs(nonzero_cols))
    assert max_abs < 1e-10, f"Expected near-zero residual returns, got max_abs={max_abs}"


# =============================================================================
# Fix 1: _resolve_panel_archetype btc_regime_pullback → trend
# =============================================================================


def _make_panel_fixture(
    *, family: str, variant: str, metadata: dict[str, object]
) -> CandidateSignalPanel:
    n_bars, n_sym = 10, 2
    zeros_f = np.zeros((n_bars, n_sym), dtype=np.float64)
    zeros_i8 = np.zeros((n_bars, n_sym), dtype=np.int8)
    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={},
        datetimes=np.arange(n_bars).astype("datetime64[h]"),
        symbols=("BTCUSDT", "ETHUSDT"),
        signed_score_2d=zeros_f,
        side_hint_2d=zeros_i8,
        expected_holding_bars=18,
        min_holding_bars=6,
        stop_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=zeros_f,
        valid_mask_2d=np.ones((n_bars, n_sym), dtype=bool),
        metadata=metadata,
    )


def test_resolve_panel_archetype_btc_regime_pullback_returns_trend() -> None:
    from src.domain.futures.strategy.rule_signals import _resolve_panel_archetype

    panel = _make_panel_fixture(family="btc_regime_pullback", variant="btc_pullback_50", metadata={})

    archetype = _resolve_panel_archetype(panel)

    assert archetype == "trend"


def test_resolve_panel_archetype_identical_across_dual_modules() -> None:
    from src.domain.futures.signals.rules import _resolve_panel_archetype as resolve_a
    from src.domain.futures.strategy.rule_signals import _resolve_panel_archetype as resolve_b

    for family in ALL_SIGNAL_FAMILIES:
        panel_a = _make_panel_fixture(family=family, variant="v", metadata={})
        panel_b = _make_panel_fixture(family=family, variant="v", metadata={})
        assert resolve_a(panel_a) == resolve_b(panel_b), f"archetype drift for family={family}"


def test_resolve_panel_archetype_explicit_metadata_overrides_family_inference() -> None:
    from src.domain.futures.strategy.rule_signals import _resolve_panel_archetype

    panel = _make_panel_fixture(
        family="btc_regime_pullback", variant="v", metadata={"archetype": "carry_rev"}
    )

    assert _resolve_panel_archetype(panel) == "carry_rev"
