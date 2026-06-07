from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
from src.domain.futures.strategy.candidate_portfolio import (
    build_candidate_alpha_panel,
    build_candidate_target_weights,
    compute_selection_sensitivity,
    compute_selection_waterfall,
    compute_shadow_selection_profiles,
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
        min_expected_net_bps=10.0,
        max_expected_shortfall_bps=50.0,
        selection_policy="hard",
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
        min_expected_net_bps=10.0,
        max_expected_shortfall_bps=50.0,
        selection_shortfall_mode="penalty_only",
        selection_policy="hard",
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    assert selected.shape[0] == 2
    assert set(selected["symbol"]) == {"BTCUSDT", "ETHUSDT"}


def test_select_candidate_events_for_portfolio_supports_stop_relative_shortfall_thresholds() -> None:
    events = _make_sample_events()
    events["sl_thr_bps"] = [40.0, 300.0, 40.0]
    events["ex_ante_cost_bps"] = [10.0, 10.0, 10.0]
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.8, 0.9, 0.85], dtype=np.float64),
        mu_gross_bps=np.array([50.0, 60.0, 40.0], dtype=np.float64),
        mu_net_decision_bps=np.array([26.0, 36.0, 16.0], dtype=np.float64),
        q10_net_bps=np.array([-80.0, -200.0, -10.0], dtype=np.float64),
        q90_net_bps=np.array([40.0, 45.0, 30.0], dtype=np.float64),
        utility_score=np.array([10.0, 12.0, 8.0], dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(
        min_expected_net_bps=10.0,
        max_expected_shortfall_bps=50.0,
        catastrophic_shortfall_bps=120.0,
        shortfall_threshold_basis="stop_relative",
        catastrophic_shortfall_stop_mult=1.0,
        selection_policy="hard",
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    assert selected.shape[0] == 2
    assert set(selected["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    diagnostics = selected.attrs["candidate_selection_diagnostics"]
    assert diagnostics["shortfall_basis"] == "stop_relative"


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
    assert int(target["gate_pass"]) == 3
    assert int(target["all_pass"]) == 1
    assert str(target["top_variant"]) == "trend_donchian:donchian_36"


def test_select_candidate_events_for_portfolio_uses_validation_quantile_policy() -> None:
    events = _make_sample_events()
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.2, 0.3, 0.4], dtype=np.float64),
        mu_gross_bps=np.array([10.0, 20.0, 30.0], dtype=np.float64),
        mu_net_decision_bps=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        q10_net_bps=np.array([-10.0, -15.0, -20.0], dtype=np.float64),
        q90_net_bps=np.array([20.0, 25.0, 35.0], dtype=np.float64),
        utility_score=np.array([1.0, 5.0, 9.0], dtype=np.float64),
        selection_thresholds={"utility_min": 5.0},
    )
    cfg = CandidateStrategyConfig(selection_policy="validation_quantile", selection_top_quantile=0.5)

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    assert set(selected["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    btc_row = selected[selected["symbol"] == "BTCUSDT"].iloc[0]
    assert btc_row["family"] == "rsi_reversion"


def test_selection_stays_zero_when_all_mu_below_breakeven_floor() -> None:
    events = _make_sample_events()
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.9, 0.9, 0.9], dtype=np.float64),
        mu_gross_bps=np.array([5.0, 6.0, 7.0], dtype=np.float64),
        mu_net_decision_bps=np.array([5.0, 6.0, 7.0], dtype=np.float64),
        q10_net_bps=np.array([-20.0, -20.0, -20.0], dtype=np.float64),
        q90_net_bps=np.array([15.0, 16.0, 17.0], dtype=np.float64),
        utility_score=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(
        selection_policy="utility_topk",
        cost_floor_bps=24.0,
        min_net_floor_cost_fraction=0.5,
        catastrophic_shortfall_bps=300.0,
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    assert selected.empty
    diagnostics = selected.attrs["candidate_selection_diagnostics"]
    assert diagnostics["zero_reason"] == "no_eligible_after_breakeven_floor"
    assert diagnostics["eligible"] == 0


def test_utility_topk_hard_floor_no_longer_blocks_selection() -> None:
    events = _make_sample_events()
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.30, 0.34, 0.20], dtype=np.float64),
        mu_gross_bps=np.array([40.0, 50.0, 60.0], dtype=np.float64),
        mu_net_decision_bps=np.array([40.0, 50.0, 60.0], dtype=np.float64),
        q10_net_bps=np.array([-20.0, -20.0, -20.0], dtype=np.float64),
        q90_net_bps=np.array([60.0, 70.0, 80.0], dtype=np.float64),
        utility_score=np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(
        selection_policy="utility_topk",
        cost_floor_bps=24.0,
        min_net_floor_cost_fraction=0.5,
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    assert not selected.empty
    assert set(selected["symbol"]) == {"ETHUSDT"}
    assert selected.attrs["candidate_selection_diagnostics"]["gate_mode"] == "off"


def test_utility_topk_soft_floor_keeps_positive_expected_utility_candidate() -> None:
    events = _make_sample_events()
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.30, 0.60, 0.55], dtype=np.float64),
        mu_gross_bps=np.array([80.0, 20.0, 18.0], dtype=np.float64),
        mu_net_decision_bps=np.array([80.0, 20.0, 18.0], dtype=np.float64),
        q10_net_bps=np.array([-5.0, -80.0, -60.0], dtype=np.float64),
        q90_net_bps=np.array([90.0, 30.0, 25.0], dtype=np.float64),
        utility_score=np.array([18.0, 5.0, 4.0], dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(
        selection_policy="utility_topk",
        cost_floor_bps=10.0,
        min_net_floor_cost_fraction=0.5,
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    assert not selected.empty
    assert "BTCUSDT" in set(selected["symbol"])


def test_utility_topk_caps_single_variant_concentration() -> None:
    # Arrange: same timestamp, distinct symbols. Variant cap must apply inside
    # this timestamp only.
    events = pd.DataFrame({
        "datetime": ["2025-01-01T00:00:00"] * 5,
        "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"],
        "family": ["trend_ma", "trend_ma", "trend_ma", "trend_ma", "rsi_reversion"],
        "variant": ["ema_12_72", "ema_12_72", "ema_12_72", "ema_12_72", "rsi_14"],
        "side": [1, 1, 1, 1, 1],
        "raw_score": [0.6, 0.6, 0.6, 0.6, 0.5],
        "score_z": [0.6, 0.6, 0.6, 0.6, 0.5],
        "expected_holding_bars": [18, 18, 18, 18, 12],
        "min_holding_bars": [6, 6, 6, 6, 4],
        "stop_atr_mult": [2.0, 2.0, 2.0, 2.0, 2.0],
        "take_profit_atr_mult": [4.0, 4.0, 4.0, 4.0, 3.0],
        "turnover_proxy": [0.05, 0.05, 0.05, 0.05, 0.05],
        "cost_floor_bps": [10.0, 10.0, 10.0, 10.0, 10.0],
        "entry_idx": [10, 10, 10, 10, 10],
    })
    # Dominant variant holds the 4 highest utilities; the other variant is lowest.
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64),
        mu_gross_bps=np.array([60.0, 55.0, 50.0, 45.0, 40.0], dtype=np.float64),
        mu_net_decision_bps=np.array([60.0, 55.0, 50.0, 45.0, 40.0], dtype=np.float64),
        q10_net_bps=np.array([-5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float64),
        q90_net_bps=np.array([90.0, 85.0, 80.0, 75.0, 70.0], dtype=np.float64),
        utility_score=np.array([60.0, 55.0, 50.0, 45.0, 40.0], dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(
        selection_policy="utility_topk",
        selection_top_quantile=1.0,
        max_variant_selection_fraction=0.5,
        cost_floor_bps=10.0,
        min_net_floor_cost_fraction=0.5,
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)

    # n_keep = ceil(5 * 1.0) = 5; max_per_variant = ceil(5 * 0.5) = 3.
    # The dominant variant must be capped to 3 within the timestamp.
    dominant = selected[selected["variant"] == "ema_12_72"]
    assert dominant.shape[0] == 3
    assert "rsi_14" in set(selected["variant"])


def test_compute_selection_waterfall_exposes_expected_utility_terms() -> None:
    events = _make_sample_events()
    events["p_pass"] = [0.30, 0.75, 0.55]
    events["mu_net_decision_bps"] = [50.0, 18.0, 14.0]
    events["q10_net_bps"] = [-10.0, -120.0, -40.0]
    cfg = CandidateStrategyConfig(
        selection_min_expected_utility_bps=0.0,
        cost_floor_bps=10.0,
        min_net_floor_cost_fraction=0.5,
    )

    diagnostics = compute_selection_waterfall(events=events, cfg=cfg)

    assert diagnostics["expected_utility_raw_p90_bps"] is not None
    assert diagnostics["downside_drag_p90_bps"] is not None
    assert diagnostics["expected_utility_ge_floor"] is not None
    assert diagnostics["all_eligible"] is not None
    assert int(diagnostics["expected_utility_ge_floor"]) >= 1
    assert int(diagnostics["all_eligible"]) >= 1


def test_shadow_selection_profiles_do_not_change_production_selection() -> None:
    events = _make_sample_events()
    events["edge_after_hurdle_bps"] = [25.0, 22.0, 18.0]
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.array([0.30, 0.34, 0.20], dtype=np.float64),
        mu_gross_bps=np.array([40.0, 50.0, 60.0], dtype=np.float64),
        mu_net_decision_bps=np.array([40.0, 50.0, 60.0], dtype=np.float64),
        q10_net_bps=np.array([-20.0, -20.0, -20.0], dtype=np.float64),
        q90_net_bps=np.array([60.0, 70.0, 80.0], dtype=np.float64),
        utility_score=np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    cfg = CandidateStrategyConfig(
        selection_policy="utility_topk",
        cost_floor_bps=24.0,
        min_net_floor_cost_fraction=0.5,
        selection_shadow_utility_floors_bps=(-20.0, 0.0),
        selection_shadow_breakeven_floor_fractions=(0.0, 0.5),
    )

    selected = select_candidate_events_for_portfolio(model_output=model_output, cfg=cfg)
    profiles = compute_shadow_selection_profiles(
        events=selected.attrs["candidate_selection_diagnostics"] and model_output.events.assign(
            p_pass=model_output.p_pass,
            mu_net_decision_bps=model_output.mu_net_decision_bps,
            q10_net_bps=model_output.q10_net_bps,
            utility_score=model_output.utility_score,
        ),
        cfg=cfg,
    )

    assert not selected.empty
    assert int(selected.attrs["candidate_selection_diagnostics"]["selected_total"]) > 0
    assert int(profiles["selected_total"].max()) > 0


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


def test_build_candidate_target_weights_uses_stop_risk_and_kelly_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_portfolio.project_all_caps",
        lambda *, w, btc_beta, sigma_port, bars_per_year, caps: w,
    )
    selected_prob = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, 1],
            "entry_idx": [10, 10],
            "expected_holding_bars": [10, 10],
            "p_pass": [0.4, 0.8],
            "mu_net_decision_bps": [40.0, 40.0],
            "q10_net_bps": [-5.0, -5.0],
            "risk_unit_bps": [400.0, 100.0],
            "utility_score": [10.0, 10.0],
        }
    )
    selected_q10 = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, 1],
            "entry_idx": [10, 10],
            "expected_holding_bars": [10, 10],
            "p_pass": [0.8, 0.8],
            "mu_net_decision_bps": [40.0, 40.0],
            "q10_net_bps": [-5.0, -1000.0],
            "risk_unit_bps": [100.0, 100.0],
            "utility_score": [10.0, 10.0],
        }
    )
    close_2d = np.full((20, 2), 100.0, dtype=np.float64)
    symbols = ("BTCUSDT", "ETHUSDT")
    sigma_3d = np.zeros((20, 2, 2), dtype=np.float64)
    sigma_3d[:, 0, 0] = 1e-8
    sigma_3d[:, 1, 1] = 1e-8
    cfg = CandidateStrategyConfig(
        kelly_fraction=0.1,
        max_symbol_weight=10.0,
        gross_cap=20.0,
        net_cap=20.0,
        beta_cap=20.0,
        target_ann_vol=10.0,
    )
    cfg_kelly = CandidateStrategyConfig(
        sizing_mode="calibrated_event_kelly",
        kelly_fraction=0.1,
        max_symbol_weight=10.0,
        gross_cap=20.0,
        net_cap=20.0,
        beta_cap=20.0,
        target_ann_vol=10.0,
    )

    prob_weights = build_candidate_target_weights(
        selected_events=selected_prob,
        close_2d=close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=sigma_3d,
        cfg=cfg,
    )
    q10_weights = build_candidate_target_weights(
        selected_events=selected_q10,
        close_2d=close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=sigma_3d,
        cfg=cfg_kelly,
    )

    assert prob_weights[10, 1] > prob_weights[10, 0]
    assert q10_weights[10, 0] > q10_weights[10, 1]


def test_overlay_mult_scales_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_portfolio.project_all_caps",
        lambda *, w, btc_beta, sigma_port, bars_per_year, caps: w,
    )
    selected = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [10],
            "expected_holding_bars": [5],
            "risk_unit_bps": [100.0],
            "overlay_mult": [0.5],
            "mu_net_decision_bps": [10.0],
            "q10_net_bps": [-5.0],
        }
    )
    cfg = CandidateStrategyConfig(
        max_symbol_weight=10.0,
        gross_cap=20.0,
        net_cap=20.0,
        beta_cap=20.0,
        target_ann_vol=10.0,
        event_risk_budget=0.01,
    )

    weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=np.full((20, 1), 100.0, dtype=np.float64),
        symbols=("BTCUSDT",),
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )

    assert weights[10, 0] == pytest.approx(0.5)


def test_crisis_floor_caps_gross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_portfolio.project_all_caps",
        lambda *, w, btc_beta, sigma_port, bars_per_year, caps: w,
    )
    selected = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, 1],
            "entry_idx": [10, 10],
            "expected_holding_bars": [5, 5],
            "risk_unit_bps": [100.0, 100.0],
            "overlay_mult": [0.15, 0.15],
            "crisis_active": [True, True],
            "mu_net_decision_bps": [10.0, 10.0],
            "q10_net_bps": [-5.0, -5.0],
        }
    )
    cfg = CandidateStrategyConfig(
        max_symbol_weight=10.0,
        gross_cap=20.0,
        net_cap=20.0,
        beta_cap=20.0,
        target_ann_vol=10.0,
        event_risk_budget=0.01,
    )

    weights = build_candidate_target_weights(
        selected_events=selected,
        close_2d=np.full((20, 2), 100.0, dtype=np.float64),
        symbols=("BTCUSDT", "ETHUSDT"),
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )

    assert np.sum(np.abs(weights[10])) == pytest.approx(0.3)


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


def test_build_candidate_alpha_panel_forward_fills_metadata_with_target_weights() -> None:
    selected = _make_sample_events().iloc[[2]].copy().reset_index(drop=True)
    selected["p_pass"] = [0.8]
    selected["mu_net_decision_bps"] = [16.0]
    selected["q10_net_bps"] = [-10.0]
    selected["utility_score"] = [8.0]
    selected["entry_idx"] = [1]
    selected["expected_holding_bars"] = [3]

    target_weights_2d = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.12],
            [0.0, 0.12],
            [0.0, 0.12],
        ],
        dtype=np.float64,
    )
    datetimes = np.array(
        ["2025-01-01T00", "2025-01-01T01", "2025-01-01T02", "2025-01-01T03"],
        dtype="datetime64[ns]",
    )
    panel = build_candidate_alpha_panel(
        selected_events=selected,
        target_weights_2d=target_weights_2d,
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        cfg=CandidateStrategyConfig(candidate_metadata_forward_fill=True),
    )

    assert panel.loc[(datetimes[2], "ETHUSDT"), "candidate_family"] == "trend_donchian"
    assert panel.loc[(datetimes[2], "ETHUSDT"), "candidate_take_profit_atr_mult"] == 4.0


def test_build_candidate_alpha_panel_when_selected_events_empty_returns_zero_panel() -> None:
    datetimes = np.array(["2025-01-01T00", "2025-01-01T01"], dtype="datetime64[ns]")
    target_weights_2d = np.zeros((2, 2), dtype=np.float64)

    panel = build_candidate_alpha_panel(
        selected_events=pd.DataFrame(),
        target_weights_2d=target_weights_2d,
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
    )

    assert panel.shape[0] == 4
    assert panel.index.names == ["datetime", "symbol"]
    assert float(panel["target_weight"].abs().sum()) == 0.0
