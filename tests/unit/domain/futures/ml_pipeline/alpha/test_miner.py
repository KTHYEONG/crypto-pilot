from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = str(Path(__file__).resolve().parents[6])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.domain.futures.ml_pipeline.alpha.miner as miner_module
from src.domain.futures.ml_pipeline.alpha.miner import MLAlphaMiner


def test_miner_labels_preparation():
    """Tests the label preparation logic in MLAlphaMiner."""
    miner = MLAlphaMiner()
    
    # Mock data
    n_rows = 100
    target = pd.Series(np.linspace(0, 1, n_rows))
    raw_returns = np.random.normal(0, 0.02, n_rows)
    atr = np.full(n_rows, 0.02)
    
    labels_long = miner._prepare_labels(target, raw_returns=raw_returns, atr_24h_pct=atr, short_oriented=False)
    labels_short = miner._prepare_labels(target, raw_returns=raw_returns, atr_24h_pct=atr, short_oriented=True)
    
    assert len(labels_long) == n_rows
    assert len(labels_short) == n_rows
    assert not np.isnan(labels_long).any()
    assert not np.isnan(labels_short).any()
    
    # Check if sum is roughly 1.0
    np.testing.assert_allclose(labels_long + labels_short, 1.0, atol=1e-6)


def test_miner_records_elite_zero_after_survival(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(miner_module, "add_macro_interaction_features", lambda df: df)

    def _fake_filter(alpha_wide, panel_df, **kwargs):
        return alpha_wide, {
            "n_surviving": 2.0,
            "n_surviving_long": 2.0,
            "n_surviving_short": 0.0,
            "survived_cols": [],
            "survived_long_cols": [],
            "survived_short_cols": [],
            "ic_by_slot": {},
            "ic_weight_by_slot": {},
        }

    monkeypatch.setattr(miner_module, "filter_alpha_components", _fake_filter)

    dt = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    syms = ["BTCUSDT", "ETHUSDT"]
    idx = pd.MultiIndex.from_product([dt, syms], names=["datetime", "symbol"])
    panel_df = pd.DataFrame(index=idx)
    panel_df["close"] = 100.0
    panel_df["high"] = 101.0
    panel_df["low"] = 99.0
    panel_df["target"] = 0.5

    miner = MLAlphaMiner(slots_per_theme=1)
    out = miner.mine_alphas_cs(panel_df, is_end_date=dt[10].isoformat(), filter_options={})

    meta = out.attrs.get("alpha_component_filter", {})
    agg = out.attrs.get("alpha_final_aggregation_counts", {})
    assert float(meta.get("pre_agg_surviving_long_count", 0.0)) == 2.0
    assert float(meta.get("post_agg_selected_long_count", 1.0)) == 0.0
    assert float(meta.get("elite_zero_after_survival", 0.0)) == 1.0
    assert float(meta.get("final_selection_fail_long", 0.0)) == 2.0
    assert float(agg.get("elite_zero_after_survival", 0.0)) == 1.0
