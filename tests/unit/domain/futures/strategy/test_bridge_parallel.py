from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_signals import (
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


@pytest.fixture
def dummy_aligned() -> AlignedMarketData:
    return _make_aligned(t=150, n=2)


@pytest.fixture
def default_cfg() -> CandidateStrategyConfig:
    return CandidateStrategyConfig(
        timeframe="4h",
        signal_only=True,
    )


def test_bridge_parallel_happy_path(dummy_aligned: AlignedMarketData, default_cfg: CandidateStrategyConfig) -> None:
    """Scenario 1: Happy Path - parallel signal calculation succeeds and sorting is correct."""
    panels = build_rule_signal_panels(aligned=dummy_aligned, cfg=default_cfg)
    assert len(panels) > 0
    
    # Ensure sorted by family, variant
    for i in range(len(panels) - 1):
        p1 = panels[i]
        p2 = panels[i + 1]
        assert (p1.family, p1.variant) <= (p2.family, p2.variant)


def test_bridge_parallel_family_filter(dummy_aligned: AlignedMarketData, default_cfg: CandidateStrategyConfig) -> None:
    """Scenario 2: Edge Case - family filter only calculates targeted families."""
    family_filter = ("trend_ma",)
    panels = build_rule_signal_panels(aligned=dummy_aligned, cfg=default_cfg, family_filter=family_filter)
    
    assert len(panels) > 0
    for p in panels:
        assert p.family == "trend_ma"


def test_bridge_parallel_empty_path(default_cfg: CandidateStrategyConfig) -> None:
    """Scenario 3: Edge Case - empty/all NaN alignment returns clean result without crashes."""
    t, n = 150, 2
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    empty_aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=np.full((t, n), np.nan),
        high_2d=np.full((t, n), np.nan),
        low_2d=np.full((t, n), np.nan),
        close_2d=np.full((t, n), np.nan),
        volume_2d=np.full((t, n), 0.0),
        funding_2d=np.full((t, n), 0.0),
        basis_2d=np.full((t, n), 0.0),
        taker_buy_2d=np.full((t, n), 0.0),
        trades_2d=np.full((t, n), 0.0),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )
    
    # Just verify that it doesn't crash and returns output
    panels = build_rule_signal_panels(aligned=empty_aligned, cfg=default_cfg)
    assert len(panels) > 0


def test_bridge_parallel_determinism(dummy_aligned: AlignedMarketData, default_cfg: CandidateStrategyConfig) -> None:
    """Scenario 5: Determinism - repeated runs yield identical results."""
    panels_1 = build_rule_signal_panels(aligned=dummy_aligned, cfg=default_cfg)
    panels_2 = build_rule_signal_panels(aligned=dummy_aligned, cfg=default_cfg)
    
    assert len(panels_1) == len(panels_2)
    for p1, p2 in zip(panels_1, panels_2, strict=True):
        assert p1.family == p2.family
        assert p1.variant == p2.variant
        np.testing.assert_array_equal(p1.signed_score_2d, p2.signed_score_2d)


def test_candidate_panels_to_events_parallel(
    dummy_aligned: AlignedMarketData,
    default_cfg: CandidateStrategyConfig,
) -> None:
    """Validate that candidate_panels_to_events parallelization matches correctly."""
    panels = build_rule_signal_panels(aligned=dummy_aligned, cfg=default_cfg)
    
    events_1 = candidate_panels_to_events(panels, min_abs_score=0.1, n_workers=1)
    events_4 = candidate_panels_to_events(panels, min_abs_score=0.1, n_workers=4)
    
    assert len(events_1) == len(events_4)
    pd.testing.assert_frame_equal(events_1, events_4)
