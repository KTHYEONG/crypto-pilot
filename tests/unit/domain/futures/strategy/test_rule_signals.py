from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_signals import build_rule_signal_panels, candidate_panels_to_events


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
    assert len(panels) == 8  # 8 rule families

    expected_families = {
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "bollinger_reversion",
        "rsi_reversion",
        "funding_carry",
        "oi_volume_impulse",
        "btc_regime_pullback",
    }

    for p in panels:
        assert p.family in expected_families
        assert p.signed_score_2d.shape == (150, 2)
        assert p.side_hint_2d.shape == (150, 2)
        assert p.valid_mask_2d.shape == (150, 2)
        assert isinstance(p.symbols, tuple)
        assert len(p.symbols) == 2


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
