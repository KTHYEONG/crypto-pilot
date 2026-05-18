from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.universe.config import Stage6Config
from src.domain.futures.universe.selection import apply_selection_stage


def _base_row(symbol: str, score_seed: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "adv_usdt_median": 30_000_000.0 + score_seed,
        "execution_cost_bps": 5.0 + score_seed * 0.01,
        "last_60d_coverage": 0.98,
        "listing_age_days": 500 + score_seed,
    }


def test_selection_computes_beta_and_cluster_from_return_vector() -> None:
    btc = np.array([0.01, 0.02, 0.00, -0.01], dtype=float)
    eth = np.array([0.015, 0.01, -0.005, -0.02], dtype=float)
    sol = np.array([0.02, 0.0, 0.01, -0.015], dtype=float)
    weights = np.array([0.45, 0.25, 0.08], dtype=float)
    market = (weights / weights.sum()) @ np.vstack([btc, eth, sol])
    xrp = market * 1.5
    ada = np.array([-0.02, -0.01, 0.02, 0.015], dtype=float)

    frame = pd.DataFrame(
        [
            {**_base_row("BTC/USDT", 10.0), "return_vector": btc},
            {**_base_row("ETH/USDT", 9.0), "return_vector": eth},
            {**_base_row("SOL/USDT", 8.0), "return_vector": sol},
            {**_base_row("XRP/USDT", 7.0), "return_vector": xrp},
            {**_base_row("ADA/USDT", 6.0), "return_vector": ada},
        ]
    )
    selected, _ = apply_selection_stage(
        frame,
        config=Stage6Config(
            k_in=5,
            k_out=5,
            anchor_symbols=("BTC/USDT", "ETH/USDT"),
            basket_ref=("BTC/USDT", "ETH/USDT", "SOL/USDT"),
            basket_weights=(0.45, 0.25, 0.08),
        ),
        max_symbols=5,
    )

    by_symbol = selected.set_index("symbol")
    assert np.isclose(float(by_symbol.loc["XRP/USDT", "beta_vs_market"]), 1.5, atol=1e-9)
    btc_cluster = int(by_symbol.loc["BTC/USDT", "cluster_id"])
    assert int(by_symbol.loc["XRP/USDT", "cluster_id"]) == btc_cluster
    assert int(by_symbol.loc["ADA/USDT", "cluster_id"]) != btc_cluster


def test_selection_fallback_for_missing_return_vectors_and_anchor_forcing() -> None:
    frame = pd.DataFrame([{**_base_row("XRP/USDT", 1.0)}])
    selected, report = apply_selection_stage(
        frame,
        config=Stage6Config(
            k_in=3,
            k_out=3,
            anchor_symbols=("BTC/USDT", "ETH/USDT"),
        ),
        max_symbols=3,
    )

    by_symbol = selected.set_index("symbol")
    assert float(by_symbol.loc["XRP/USDT", "beta_vs_market"]) == 0.0
    assert int(by_symbol.loc["XRP/USDT", "cluster_id"]) == -1
    assert by_symbol.loc["BTC/USDT", "hysteresis_state"] == "anchor_forced"
    assert by_symbol.loc["ETH/USDT", "hysteresis_state"] == "anchor_forced"
    assert by_symbol.loc["BTC/USDT", "role"] == "anchor"
    assert by_symbol.loc["ETH/USDT", "role"] == "anchor"
    assert set(report["symbol"].tolist()) >= {"BTC/USDT", "ETH/USDT", "XRP/USDT"}
