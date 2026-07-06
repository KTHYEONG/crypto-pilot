from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateConfig,
    L2PosteriorPolicyConfig,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.pipeline import (
    build_l2_sleeves_from_posterior,
    build_posterior_from_l1_fold_rows,
    run_alpha_foundry_l0_pipeline,
    run_alpha_foundry_pipeline,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _make_aligned() -> AlignedMarketData:
    # 600 bars — long enough for the 8-on/8-off entry pattern in _make_panel()
    # to clear min_events=40 while staying under the default turnover cap.
    t, n = 600, 2
    datetimes = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    # Constant per-bar log-drift so the mean 3-bar gross return stays
    # comfortably above the ~11.25bps stress round-trip cost.
    close = (100.0 * np.exp(0.002 * np.arange(t, dtype=np.float64))).reshape(-1, 1) * np.ones((1, n))
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
    # 8-on/8-off cycle: sparse-entry semantics need real flat/reversal
    # transitions (a constant side=1 array yields zero entries).
    side = np.zeros((t, n), dtype=np.int8)
    for start in range(0, t, 16):
        side[start : start + 8, :] = 1
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=side.astype(np.float64),
        side_hint_2d=side,
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
        cheap_ev, l1_units, _post, _slv = run_alpha_foundry_pipeline(
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

    def test_pipeline_empty_panels_returns_empty(self) -> None:
        aligned = _make_aligned()
        cheap_ev, l1_units, _post, _slv = run_alpha_foundry_pipeline(
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
        assert len(_post) == 0
        assert len(_slv) == 0


class TestRunAlphaFoundryL0Pipeline:
    def test_l0_pipeline_returns_artifacts(self) -> None:
        panel = _make_panel()
        aligned = _make_aligned()
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[panel],
            recipes={"r1": SAMPLE_RECIPE},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
        )
        assert len(artifacts.evidences) > 0
        assert isinstance(artifacts.passed_recipe_ids, tuple)
        assert isinstance(artifacts.reject_reason_counts, dict)

    def test_l0_pipeline_empty_panels_returns_empty_artifacts(self) -> None:
        aligned = _make_aligned()
        artifacts = run_alpha_foundry_l0_pipeline(
            panels=[],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
        )
        assert len(artifacts.evidences) == 0
        assert len(artifacts.passed_recipe_ids) == 0


class TestBuildPosteriorFromL1FoldRows:
    def test_empty_raw_rows_returns_empty(self) -> None:
        config = PosteriorGateConfig()
        result = build_posterior_from_l1_fold_rows(
            raw_rows=pd.DataFrame(columns=["symbol", "recipe_id", "family", "timeframe",
                                           "activation_context", "net_bps", "fold_id", "effective_weight"]),
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) == 0

    def test_rejects_missing_net_bps_column(self) -> None:
        config = PosteriorGateConfig()
        with pytest.raises(ValueError, match="raw_rows missing required columns"):
            build_posterior_from_l1_fold_rows(
                raw_rows=pd.DataFrame({"symbol": ["SYM0USDT"]}),
                cost_model=ExecutionCostModel(),
                config=config,
            )

    def test_rejects_missing_symbol_column(self) -> None:
        config = PosteriorGateConfig()
        with pytest.raises(ValueError, match="raw_rows missing required columns"):
            build_posterior_from_l1_fold_rows(
                raw_rows=pd.DataFrame({"net_bps": [1.0]}),
                cost_model=ExecutionCostModel(),
                config=config,
            )

    def test_real_fold_rows_produces_posterior(self) -> None:
        config = PosteriorGateConfig()
        raw_rows = pd.DataFrame([
            {"symbol": "SYM0USDT", "recipe_id": "trend_ma__ema_12_72__4h",
             "family": "trend_ma", "timeframe": "4h", "activation_context": "pooled",
             "net_bps": 4.2, "fold_id": 0, "effective_weight": 1.0},
            {"symbol": "SYM0USDT", "recipe_id": "trend_ma__ema_12_72__4h",
             "family": "trend_ma", "timeframe": "4h", "activation_context": "pooled",
             "net_bps": 3.6, "fold_id": 1, "effective_weight": 1.0},
        ])
        result = build_posterior_from_l1_fold_rows(
            raw_rows=raw_rows,
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) > 0
        assert all(hasattr(r, "posterior_mu_bps") for r in result)


class TestL2SleevesFromPosterior:
    def test_empty_posterior_returns_empty(self) -> None:
        config = L2PosteriorPolicyConfig()
        result = build_l2_sleeves_from_posterior(
            posterior=(),
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) == 0

    def test_real_posterior_produces_l2_sleeves(self) -> None:
        config = L2PosteriorPolicyConfig()
        raw_rows = pd.DataFrame([
            {"symbol": "SYM0USDT", "recipe_id": "trend_ma__ema_12_72__4h",
             "family": "trend_ma", "timeframe": "4h", "activation_context": "pooled",
             "net_bps": 4.2, "fold_id": 0, "effective_weight": 1.0},
        ])
        posterior = build_posterior_from_l1_fold_rows(
            raw_rows=raw_rows,
            cost_model=ExecutionCostModel(),
            config=PosteriorGateConfig(),
        )
        result = build_l2_sleeves_from_posterior(
            posterior=posterior,
            cost_model=ExecutionCostModel(),
            config=config,
        )
        assert len(result) > 0
