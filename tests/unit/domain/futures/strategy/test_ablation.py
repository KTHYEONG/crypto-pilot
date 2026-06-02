from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.strategy.ablation import (
    _build_rule_equal_size_weights,
    _build_uncapped_kelly_edge_weights,
    run_candidate_ablation,
)
from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
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


def test_run_candidate_ablation_returns_correct_ablation_dataframe(monkeypatch: Any) -> None:
    data_maps = _make_mock_data_maps(250)
    cfg = CandidateStrategyConfig(
        timeframe="4h",
        min_candidate_obs=10,  # lower observation threshold for testing
        min_rule_net_bps=0.0,
        kelly_fraction=0.1,
        gross_cap=1.2,
    )

    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.fit_candidate_gate",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.predict_candidate_gate",
        lambda *, dataset, **__: np.full(dataset.X.shape[0], 0.9, dtype=np.float64),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.fit_candidate_edge_models",
        lambda **_: object(),
    )

    def _fake_predict_candidate_edges(*, dataset: Any, p_pass: np.ndarray, **__: Any) -> CandidateModelOutput:
        n = dataset.X.shape[0]
        return CandidateModelOutput(
            events=dataset.event_index,
            p_pass=p_pass,
            mu_gross_bps=np.full(n, 40.0, dtype=np.float64),
            mu_net_decision_bps=np.full(n, 16.0, dtype=np.float64),
            q10_net_bps=np.full(n, -10.0, dtype=np.float64),
            q90_net_bps=np.full(n, 30.0, dtype=np.float64),
            utility_score=np.full(n, 8.0, dtype=np.float64),
        )

    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.predict_candidate_edges",
        _fake_predict_candidate_edges,
    )

    df_ablation = run_candidate_ablation(
        data_maps=data_maps,
        symbols=("BTCUSDT", "ETHUSDT"),
        tf="4h",
        cfg=cfg,
    )

    assert isinstance(df_ablation, pd.DataFrame)
    if not df_ablation.empty:
        assert df_ablation.shape[0] == 10  # 6 base variants + 4 OOS-only ablation rows
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


def test_build_rule_equal_size_weights_uses_entry_idx_bar() -> None:
    close_2d = np.zeros((20, 2), dtype=np.float64)
    events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [10],
        }
    )

    weights = _build_rule_equal_size_weights(
        raw_events=events,
        close_2d=close_2d,
        symbols=("BTCUSDT", "ETHUSDT"),
        max_symbol_weight=0.1,
    )

    assert weights[9, 0] == 0.0
    assert weights[10, 0] == 0.1


def test_build_uncapped_kelly_edge_weights_scales_by_holding_bars() -> None:
    base = np.linspace(100.0, 130.0, 40, dtype=np.float64)
    close_2d = np.stack([base, base], axis=1)
    events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, 1],
            "entry_idx": [25, 25],
            "expected_holding_bars": [2, 20],
            "mu_net_decision_bps": [40.0, 40.0],
        }
    )

    weights = _build_uncapped_kelly_edge_weights(
        selected_events=events,
        close_2d=close_2d,
        symbols=("BTCUSDT", "ETHUSDT"),
        kelly_fraction=0.1,
    )

    short_hold = weights[25, 0]
    long_hold = weights[25, 1]
    assert short_hold > 0.0
    assert long_hold > 0.0
    assert np.isclose(short_hold / long_hold, 10.0, rtol=1e-6)
