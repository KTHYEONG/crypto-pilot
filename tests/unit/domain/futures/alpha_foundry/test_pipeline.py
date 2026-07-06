from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateConfig,
    L2PosteriorPolicyConfig,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.pipeline import run_alpha_foundry_pipeline
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _make_aligned() -> AlignedMarketData:
    t, n = 60, 2
    datetimes = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    close = np.linspace(100, 112, t).reshape(-1, 1) * np.ones((1, n))
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full_like(close, 1000.0),
        funding_2d=np.full_like(close, 0.0001),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(close, dtype=np.bool_),
        kill_mask=np.zeros_like(close, dtype=np.bool_),
    )


def _make_panel() -> CandidateSignalPanel:
    aligned = _make_aligned()
    t, n = aligned.close_2d.shape
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=np.ones((t, n), dtype=np.float64),
        side_hint_2d=np.ones((t, n), dtype=np.int8),
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": "r1"},
    )


SAMPLE_RECIPE = AlphaRecipe(
    recipe_id="r1",
    family="trend_ma",
    variant="ema_12_72",
    timeframe="4h",
    archetype="trend",
    indicator_params={"fast": 12, "slow": 72},
    side_rule_id="trend_follow",
    exit_policy_id="atr_trail_2",
    required_fields=("close",),
    causal_lag_bars=1,
    max_turnover_per_year=365.0,
)


class TestRunAlphaFoundryPipeline:
    def test_pipeline_produces_all_stage_outputs(self) -> None:
        panel = _make_panel()
        aligned = _make_aligned()
        cheap_ev, l1_units, posterior, sleeves = run_alpha_foundry_pipeline(
            panels=[panel],
            recipes={"r1": SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            posterior_gate_config=PosteriorGateConfig(),
            l2_config=L2PosteriorPolicyConfig(),
            symbols=("BTCUSDT",),
        )
        assert len(cheap_ev) > 0
        assert len(l1_units) > 0
        assert len(posterior) > 0
        assert len(sleeves) > 0

    def test_pipeline_empty_panels_returns_empty(self) -> None:
        aligned = _make_aligned()
        cheap_ev, l1_units, posterior, sleeves = run_alpha_foundry_pipeline(
            panels=[],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            posterior_gate_config=PosteriorGateConfig(),
            l2_config=L2PosteriorPolicyConfig(),
            symbols=("BTCUSDT",),
        )
        assert len(cheap_ev) == 0
        assert len(l1_units) == 0
        assert len(posterior) == 0
        assert len(sleeves) == 0
