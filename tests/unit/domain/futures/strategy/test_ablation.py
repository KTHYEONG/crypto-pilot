from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.strategy.ablation import run_candidate_ablation
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_mock_data_maps(t: int = 150) -> dict[str, dict[str, Any]]:
    symbols = ["BTCUSDT", "ETHUSDT"]
    datetimes = pd.date_range("2025-01-01", periods=t, freq="4h")
    
    maps = {}
    for sym in symbols:
        base = np.linspace(100.0, 130.0, t) if sym == "BTCUSDT" else np.linspace(10.0, 13.0, t)
        df = pd.DataFrame({
            "datetime": datetimes,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.full(t, 1000.0, dtype=np.float64),
            "funding_rate": np.zeros(t, dtype=np.float64),
            "universe_active_mask": np.ones(t, dtype=bool),
            "universe_entry_warm_mask": np.ones(t, dtype=bool),
            "entry_block_mask": np.zeros(t, dtype=bool),
            "kill_signal": np.zeros(t, dtype=bool),
        })
        maps[sym] = {"4h": df}
    return maps


def test_run_candidate_ablation_returns_correct_ablation_dataframe() -> None:
    data_maps = _make_mock_data_maps(250)
    cfg = CandidateStrategyConfig(
        timeframe="4h",
        min_candidate_obs=10,  # lower observation threshold for testing
        min_rule_net_bps=0.0,
        kelly_fraction=0.1,
        gross_cap=1.2,
    )

    df_ablation = run_candidate_ablation(
        data_maps=data_maps,
        symbols=("BTCUSDT", "ETHUSDT"),
        tf="4h",
        cfg=cfg,
    )

    assert isinstance(df_ablation, pd.DataFrame)
    if not df_ablation.empty:
        assert df_ablation.shape[0] == 6  # 6 variants
        required_cols = {
            "variant",
            "mean_log_growth",
            "cagr",
            "max_drawdown",
            "mar",
            "turnover",
            "final_equity",
            "pass_compound_gate",
        }
        assert required_cols.issubset(df_ablation.columns)
