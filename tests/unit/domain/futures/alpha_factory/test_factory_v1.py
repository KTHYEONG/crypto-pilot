from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.alpha_factory import AlphaFactoryV1


def _build_panel_df() -> pd.DataFrame:
    dt = pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT"]
    idx = pd.MultiIndex.from_product([dt, symbols], names=["datetime", "symbol"])
    n = len(idx)
    out = pd.DataFrame(index=idx)
    out["ret_6"] = np.array([0.05, -0.02, 0.04, -0.01, 0.03, -0.03], dtype=np.float64)
    out["ret_24"] = np.array([0.10, -0.03, 0.08, -0.02, 0.06, -0.04], dtype=np.float64)
    out["funding_z_72"] = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    out["taker_imbalance_z_24"] = np.linspace(-0.5, 0.5, n, dtype=np.float64)
    out["cvd_divergence_24h"] = np.linspace(-0.2, 0.2, n, dtype=np.float64)
    out["range_pos_24"] = np.array([0.7, 0.4, 0.6, 0.3, 0.55, 0.45], dtype=np.float64)
    out["hmm_prob_bull"] = 0.40
    out["hmm_prob_bear"] = 0.30
    out["hmm_prob_chop"] = 0.20
    out["hmm_prob_crisis"] = 0.10
    return out


def test_factory_v1_mine_alphas_cs_4h_guard() -> None:
    factory = AlphaFactoryV1(timeframe="1h")
    with pytest.raises(ValueError, match="supports only 4h timeframe"):
        factory.mine_alphas_cs(_build_panel_df())


def test_factory_v1_mine_alphas_cs_contract() -> None:
    factory = AlphaFactoryV1(timeframe="4h")
    panel_df = _build_panel_df()
    out = factory.mine_alphas_cs(panel_df)

    required_output_cols = {
        "alpha_long_00",
        "alpha_short_00",
        "alpha_long",
        "alpha_short",
        "alpha_net",
        "alpha_confidence",
    }
    assert required_output_cols.issubset(out.columns)
    assert out.index.equals(panel_df.index)

    assert out["alpha_long"].between(0.0, 1.0).all()
    assert out["alpha_short"].between(0.0, 1.0).all()
    assert out["alpha_confidence"].between(0.0, 1.0).all()
    assert out["alpha_net"].between(-1.0, 1.0).all()
    np.testing.assert_allclose(
        out["alpha_long"].to_numpy(dtype=np.float64, copy=False)
        + out["alpha_short"].to_numpy(dtype=np.float64, copy=False),
        np.ones(len(out), dtype=np.float64),
        atol=1e-12,
    )

    alpha_filter = out.attrs.get("alpha_component_filter", {})
    required_attr_keys = {
        "n_components",
        "n_surviving",
        "n_surviving_long",
        "n_surviving_short",
        "post_agg_selected_long_count",
        "post_agg_selected_short_count",
        "survived_long_cols",
        "survived_short_cols",
        "post_agg_selected_long_cols",
        "post_agg_selected_short_cols",
        "elite_zero_after_survival",
    }
    assert required_attr_keys.issubset(alpha_filter)
    assert float(alpha_filter["n_components"]) == 1.0
    assert 0.0 <= float(alpha_filter["n_surviving"]) <= float(alpha_filter["n_components"])
    assert 0.0 <= float(alpha_filter["n_surviving_long"]) <= float(alpha_filter["n_components"])
    assert 0.0 <= float(alpha_filter["n_surviving_short"]) <= float(alpha_filter["n_components"])
    assert alpha_filter["survived_long_cols"] == ["alpha_long_00"]
    assert alpha_filter["survived_short_cols"] == ["alpha_short_00"]
    assert alpha_filter["post_agg_selected_long_cols"] == ["alpha_long_00"]
    assert alpha_filter["post_agg_selected_short_cols"] == ["alpha_short_00"]
