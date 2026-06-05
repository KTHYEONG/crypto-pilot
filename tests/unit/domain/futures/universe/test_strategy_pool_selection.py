from __future__ import annotations

import pandas as pd

from src.domain.futures.universe.config import UniverseConfig
from src.domain.futures.universe.selection import apply_selection_stage


def test_universe_config_strategy_pool_defaults() -> None:
    cfg = UniverseConfig()
    assert cfg.strategy_pool_mode == "stage6_selected"
    assert cfg.stage6_is_alpha_rank is False


def test_stage6_outputs_execution_pool_score() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "adv_usdt_median": 100_000_000.0,
                "execution_cost_bps": 4.0,
                "last_60d_coverage": 0.99,
                "listing_age_days": 2000,
            },
            {
                "symbol": "ETHUSDT",
                "adv_usdt_median": 80_000_000.0,
                "execution_cost_bps": 5.0,
                "last_60d_coverage": 0.99,
                "listing_age_days": 1800,
            },
            {
                "symbol": "SOLUSDT",
                "adv_usdt_median": 50_000_000.0,
                "execution_cost_bps": 7.0,
                "last_60d_coverage": 0.98,
                "listing_age_days": 900,
            },
        ]
    )
    selected, _ = apply_selection_stage(frame, max_symbols=3)
    assert "execution_pool_score" in selected.columns
    assert "tradeable_score" in selected.columns

