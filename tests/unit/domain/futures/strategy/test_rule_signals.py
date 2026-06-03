from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_signals import (
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
    # 8 base families (13 variants) + 4 new families (11 variants: F1x3+F2x3+F3x2+F4x3) = 24
    # + 3 new families (6 variants: fac x2 + brm x2 + oib x2) = 30
    assert len(panels) == 30

    expected_families = {
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "bollinger_reversion",
        "rsi_reversion",
        "funding_carry",
        "oi_volume_impulse",
        "btc_regime_pullback",
        "cross_sectional_momentum",
        "funding_zscore_carry",
        "vol_regime_reversion",
        "btc_corr_regime",
        "funding_acceleration_carry",
        "btc_residual_momentum",
        "oi_volume_confirmed_breakout",
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
        }
        assert required_cols.issubset(events.columns)
        assert (events["entry_idx"] > 0).all()


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


def test_new_signal_families_shapes_and_side_hints() -> None:
    # Arrange
    aligned = _make_aligned(t=200)
    cfg = CandidateStrategyConfig()

    # Act
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    # Assert: 각 신규 family 패널의 shape 및 side_hint 범위 검증
    new_families = {
        "cross_sectional_momentum",
        "funding_zscore_carry",
        "vol_regime_reversion",
        "btc_corr_regime",
    }
    new_panels = [p for p in panels if p.family in new_families]
    assert len(new_panels) == 11  # 3+3+2+3

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


def test_cross_sectional_momentum_no_lookahead() -> None:
    # The first `lookback` rows should have side_hint == 0 (warmup guard).
    aligned = _make_aligned(t=100)
    cfg = CandidateStrategyConfig()
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)

    cs_panels = [p for p in panels if p.family == "cross_sectional_momentum"]
    assert len(cs_panels) == 3

    for p in cs_panels:
        lb = int(p.params["lookback"])
        warmup_sides = p.side_hint_2d[:lb]
        assert (warmup_sides == 0).all(), (
            f"cross_sectional_momentum:{p.variant} has non-zero side_hint in warmup bars [0:{lb}]"
        )
