from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.legacy.ml_pipeline import pipeline_runner


def _build_panel_df() -> pd.DataFrame:
    dt = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    syms = ["BTCUSDT", "ETHUSDT"]
    idx = pd.MultiIndex.from_product([dt, syms], names=["datetime", "symbol"])
    df = pd.DataFrame(index=idx)
    df["open"] = 1.0
    df["high"] = 1.0
    df["low"] = 1.0
    df["close"] = 1.0
    df["volume"] = 1.0
    df["target"] = 0.5
    df["hmm_prob_crisis"] = 0.2
    return df


def test_alpha_cache_hit_with_identical_key() -> None:
    pipeline_runner._alpha_cache_store.clear()
    panel_df = _build_panel_df()
    cfg = {"FUTURES_ML_ALPHA_SLOTS_PER_THEME": 5}
    filter_options = {"fdr_q": 0.1}
    key = pipeline_runner._build_alpha_cache_key(
        panel_df=panel_df,
        tf="4h",
        is_end_date="2026-01-05T00:00:00+00:00",
        seed=42,
        cfg=cfg,
        horizons=(3, 6, 12),
        slots_per_theme=5,
        filter_options=filter_options,
        alpha_backend="jax",
    )
    alpha_panel = pd.DataFrame(index=panel_df.index, data={"alpha_long": 0.5, "alpha_short": 0.5})
    store_meta = pipeline_runner._alpha_cache_put(
        key,
        alpha_panel,
        max_items=2,
        alpha_backend="jax",
    )
    hit_panel, hit_meta = pipeline_runner._alpha_cache_get(key)

    assert store_meta["cache_state"] == "miss_stored"
    assert hit_panel is not None
    assert hit_meta is not None
    assert hit_meta["cache_state"] == "hit"
    assert hit_meta["cache_key"] == key
    assert hit_panel.equals(alpha_panel)


def test_alpha_cache_miss_when_hyperparameter_changes() -> None:
    pipeline_runner._alpha_cache_store.clear()
    panel_df = _build_panel_df()
    cfg = {"FUTURES_ML_ALPHA_SLOTS_PER_THEME": 5}
    key_base = pipeline_runner._build_alpha_cache_key(
        panel_df=panel_df,
        tf="4h",
        is_end_date="2026-01-05T00:00:00+00:00",
        seed=42,
        cfg=cfg,
        horizons=(3, 6, 12),
        slots_per_theme=5,
        filter_options={"fdr_q": 0.1},
        alpha_backend="jax",
    )
    key_changed = pipeline_runner._build_alpha_cache_key(
        panel_df=panel_df,
        tf="4h",
        is_end_date="2026-01-05T00:00:00+00:00",
        seed=42,
        cfg=cfg,
        horizons=(3, 6, 12),
        slots_per_theme=6,
        filter_options={"fdr_q": 0.1},
        alpha_backend="jax",
    )
    pipeline_runner._alpha_cache_put(
        key_base,
        pd.DataFrame(index=panel_df.index, data={"alpha_long": 0.5}),
        max_items=2,
        alpha_backend="jax",
    )
    miss_panel, miss_meta = pipeline_runner._alpha_cache_get(key_changed)
    assert key_base != key_changed
    assert miss_panel is None
    assert miss_meta is None


def test_alpha_cache_miss_when_data_snapshot_changes() -> None:
    pipeline_runner._alpha_cache_store.clear()
    panel_df = _build_panel_df()
    panel_df_changed = panel_df.copy()
    panel_df_changed.loc[(slice(None), "BTCUSDT"), "close"] = 2.0
    cfg = {"FUTURES_ML_ALPHA_SLOTS_PER_THEME": 5}
    key_base = pipeline_runner._build_alpha_cache_key(
        panel_df=panel_df,
        tf="4h",
        is_end_date="2026-01-05T00:00:00+00:00",
        seed=42,
        cfg=cfg,
        horizons=(3, 6, 12),
        slots_per_theme=5,
        filter_options={"fdr_q": 0.1},
        alpha_backend="jax",
    )
    key_changed = pipeline_runner._build_alpha_cache_key(
        panel_df=panel_df_changed,
        tf="4h",
        is_end_date="2026-01-05T00:00:00+00:00",
        seed=42,
        cfg=cfg,
        horizons=(3, 6, 12),
        slots_per_theme=5,
        filter_options={"fdr_q": 0.1},
        alpha_backend="jax",
    )
    pipeline_runner._alpha_cache_put(
        key_base,
        pd.DataFrame(index=panel_df.index, data={"alpha_long": 0.5}),
        max_items=2,
        alpha_backend="jax",
    )
    miss_panel, miss_meta = pipeline_runner._alpha_cache_get(key_changed)
    assert key_base != key_changed
    assert miss_panel is None
    assert miss_meta is None
