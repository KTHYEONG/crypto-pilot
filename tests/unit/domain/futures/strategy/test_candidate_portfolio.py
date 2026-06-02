from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
from src.domain.futures.strategy.candidate_portfolio import (
    build_candidate_alpha_panel,
    build_candidate_target_weights,
    compute_selection_sensitivity,
    select_candidate_events_for_portfolio,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_sample_events() -> pd.DataFrame:
    # 2 symbols, 2 variants at same datetime
    return pd.DataFrame({
        "datetime": ["2025-01-01T00:00:00", "2025-01-01T00:00:00", "2025-01-01T00:00:00"],
        "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
        "family": ["trend_ma", "rsi_reversion", "trend_donchian"],
        "variant": ["ema_12_72", "rsi_14", "donchian_36"],
        "side": [1, -1, 1],  # BTC long vs BTC short conflict!
        "raw_score": [0.6, -0.7, 0.5],
        "score_z": [0.6, -0.7, 0.5],
        "expected_holding_bars": [18, 12, 24],
        "min_holding_bars": [6, 4, 8],
        "stop_atr_mult": [2.0, 2.0, 2.0],
        "take_profit_atr_mult": [4.0, 3.0, 4.0],
        "turnover_proxy": [0.05, 0.08, 0.04],
        "cost_floor_bps": [24.0, 24.0, 24.0],
        "entry_idx": [10, 10, 10],
    })


def test_select_candidate_events_for_portfolio_filters_by_thresholds() -> None:
    events = _make_sample_events()
    # Mock model outputs
    # Index 0: BTC long passes, utility = 10.0
    # Index 1: BTC short passes, utility = 12.0 (BTC short will win!)
    # Index 2: ETH long passes, utility = 8.0
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.8, 0.9, 0.85], dtype=np.float64),
        mu_gross_bps=np.array([50.0, 60.0, 40.0], dtype=np.float64),
        mu_net_decision_bps=np.array([26.0, 36.0, 16.0], dtype=np.float64),
        q10_net_bps=np.array([-5.0, -8.0, -10.0], dtype=np.float64),
        q90_net_bps=np.array([40.0, 45.0, 30.0], dtype=np.float64),
        utility_score=np.array([10.0, 12.0, 8.0], dtype=np.float64),
    )

    cfg = CandidateStrategyConfig(
        min_gate_probability=0.75,
        min_expected_net_bps=10.0,
        max_expected_shortfall_bps=50.0,
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    # conflicting BTC variants resolved: BTC short wins because of higher utility (12.0 > 10.0)
    assert selected.shape[0] == 2  # 1 for BTC, 1 for ETH
    btc_row = selected[selected["symbol"] == "BTCUSDT"].iloc[0]
    assert int(btc_row["side"]) == -1  # short won
    assert btc_row["family"] == "rsi_reversion"

    eth_row = selected[selected["symbol"] == "ETHUSDT"].iloc[0]
    assert int(eth_row["side"]) == 1
    assert eth_row["family"] == "trend_donchian"


def test_select_candidate_events_for_portfolio_penalty_only_mode_ignores_q10_filter() -> None:
    events = _make_sample_events()
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.8, 0.9, 0.85], dtype=np.float64),
        mu_gross_bps=np.array([50.0, 60.0, 40.0], dtype=np.float64),
        mu_net_decision_bps=np.array([26.0, 36.0, 16.0], dtype=np.float64),
        q10_net_bps=np.array([-120.0, -140.0, -10.0], dtype=np.float64),
        q90_net_bps=np.array([40.0, 45.0, 30.0], dtype=np.float64),
        utility_score=np.array([10.0, 12.0, 8.0], dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(
        min_gate_probability=0.75,
        min_expected_net_bps=10.0,
        max_expected_shortfall_bps=50.0,
        selection_shortfall_mode="penalty_only",
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    assert selected.shape[0] == 2
    assert set(selected["symbol"]) == {"BTCUSDT", "ETHUSDT"}


def test_compute_selection_sensitivity_returns_grid_counts() -> None:
    events = _make_sample_events()
    events["p_pass"] = [0.40, 0.60, 0.80]
    events["mu_net_decision_bps"] = [0.0, 2.0, 8.0]
    events["q10_net_bps"] = [-300.0, -120.0, -20.0]

    sensitivity = compute_selection_sensitivity(
        events=events,
        gate_grid=(0.40, 0.55),
        edge_grid_bps=(0.0, 5.0),
        q10_grid_bps=(80.0, 250.0),
    )

    assert sensitivity.shape[0] == 8
    target = sensitivity.loc[
        (sensitivity["gate_threshold"] == 0.55)
        & (sensitivity["edge_threshold_bps"] == 5.0)
        & (sensitivity["q10_shortfall_bps"] == 250.0)
    ].iloc[0]
    assert int(target["all_pass"]) == 1
    assert str(target["top_variant"]) == "trend_donchian:donchian_36"


def test_build_candidate_target_weights_applies_kelly_and_caps() -> None:
    events = _make_sample_events()
    # Add passing features directly
    events["p_pass"] = [0.9, 0.9, 0.9]
    events["mu_net_decision_bps"] = [40.0, 40.0, 40.0]
    events["q10_net_bps"] = [-5.0, -5.0, -5.0]
    events["utility_score"] = [10.0, 10.0, 10.0]

    # Use resolved selection: 1 BTC short, 1 ETH long
    # t = 9 (entry_idx = 10)
    selected = events.iloc[[1, 2]].copy().reset_index(drop=True)

    t_steps = 20
    close_2d = np.full((t_steps, 2), 100.0, dtype=np.float64)
    # create some tiny variance in returns to allow Kelly Sizing
    close_2d[5:, 0] = 101.0
    close_2d[5:, 1] = 99.0

    symbols = ("BTCUSDT", "ETHUSDT")
    cfg = CandidateStrategyConfig(
        kelly_fraction=0.1,
        max_symbol_weight=0.15,
        gross_cap=1.2,
        net_cap=0.3,
        beta_cap=0.5,
        target_ann_vol=0.3,
    )

    target_weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )

    assert target_weights.shape == (20, 2)
    # execution timestamp is entry_idx = 10
    assert target_weights[10, 0] < 0.0  # BTC short
    assert target_weights[10, 1] > 0.0  # ETH long
    
    # Check that caps are respected
    # gross cap = 1.2, net cap = 0.3, symbol cap = 0.15
    assert abs(target_weights[10, 0]) <= 0.15
    assert abs(target_weights[10, 1]) <= 0.15
    assert np.sum(np.abs(target_weights[10])) <= 1.2
    assert abs(np.sum(target_weights[10])) <= 0.3


def test_build_candidate_alpha_panel_formats_correctly() -> None:
    events = _make_sample_events()
    events["p_pass"] = [0.8, 0.9, 0.85]
    events["mu_net_decision_bps"] = [26.0, 36.0, 16.0]
    events["q10_net_bps"] = [-5.0, -8.0, -10.0]
    events["utility_score"] = [10.0, 12.0, 8.0]

    selected = events.iloc[[1, 2]].copy().reset_index(drop=True)

    target_weights_2d = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.10, 0.12],  # execution bar t = entry_idx = 2
    ])
    selected.loc[0, "entry_idx"] = 2
    selected.loc[1, "entry_idx"] = 2

    datetimes = np.array(["2025-01-01T00", "2025-01-01T01", "2025-01-01T02"], dtype="datetime64[ns]")
    selected.loc[0, "datetime"] = "2025-01-01T01:00:00"
    selected.loc[1, "datetime"] = "2025-01-01T01:00:00"

    symbols = ("BTCUSDT", "ETHUSDT")

    panel = build_candidate_alpha_panel(
        selected_events=selected,
        target_weights_2d=target_weights_2d,
        datetimes=datetimes,
        symbols=symbols,
    )

    assert isinstance(panel, pd.DataFrame)
    assert panel.shape[0] == 6  # 3 timesteps * 2 symbols
    assert "target_weight" in panel.columns
    assert "alpha_long" in panel.columns
    assert "alpha_short" in panel.columns

    # Check mapping at execution bar t = 2
    btc_idx = (datetimes[2], "BTCUSDT")
    eth_idx = (datetimes[2], "ETHUSDT")

    assert panel.loc[btc_idx, "alpha_short"] == 0.10
    assert panel.loc[btc_idx, "alpha_long"] == 0.0
    assert panel.loc[btc_idx, "candidate_family"] == "rsi_reversion"

    assert panel.loc[eth_idx, "alpha_long"] == 0.12
    assert panel.loc[eth_idx, "alpha_short"] == 0.0
    assert panel.loc[eth_idx, "candidate_family"] == "trend_donchian"
