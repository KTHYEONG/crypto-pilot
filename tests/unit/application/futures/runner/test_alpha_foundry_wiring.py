from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_foundry.budget import build_l1_verification_units
from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_cheap_gate_batch
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateConfig,
    L2PosteriorPolicyConfig,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.diversity import compute_panel_correlation_matrix
from src.domain.futures.alpha_foundry.l2_policy import (
    convert_posterior_to_l2_sleeves,
    resolve_staged_search_budget,
)
from src.domain.futures.alpha_foundry.pipeline import run_alpha_foundry_pipeline
from src.domain.futures.alpha_foundry.posterior import shrink_l1_evidence_hierarchical
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _aligned_4h() -> AlignedMarketData:
    t, n = 100, 2
    dt = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    close = np.linspace(100, 112, t).reshape(-1, 1) * np.ones((1, n))
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt,
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


def _panel(rid: str = "r1") -> CandidateSignalPanel:
    a = _aligned_4h()
    t, n = a.close_2d.shape
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
        datetimes=a.datetimes,
        symbols=a.symbols,
        signed_score_2d=np.ones((t, n), dtype=np.float64),
        side_hint_2d=np.ones((t, n), dtype=np.int8),
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((t, n), dtype=np.float64),
        valid_mask_2d=np.ones((t, n), dtype=np.bool_),
        metadata={"recipe_id": rid},
    )


R1 = AlphaRecipe(
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


class TestAlphaFoundryWiring:
    """Verify L0→L1→L2 bridge wiring end-to-end."""

    def test_l0_cheap_gate_feeds_l1_units(self) -> None:
        a = _aligned_4h()
        panel = _panel()
        evs = evaluate_alpha_cheap_gate_batch(
            panels=[panel],
            recipes={"r1": R1},
            aligned=a,
            cost_model=ExecutionCostModel(),
            config=CheapGateConfig(),
        )
        assert any(e.gate_passed for e in evs)
        units = build_l1_verification_units(
            evidences=evs,
            recipes={"r1": R1},
            symbols=("BTCUSDT",),
            top_k_per_family_tf=5,
            initial_fold_budget=3,
        )
        assert len(units) > 0

    def test_l1_posterior_feeds_l2_sleeves(self) -> None:
        a = _aligned_4h()
        panel = _panel()
        evs = evaluate_alpha_cheap_gate_batch(
            panels=[panel],
            recipes={"r1": R1},
            aligned=a,
            cost_model=ExecutionCostModel(),
            config=CheapGateConfig(),
        )
        import pandas as pd

        rows = [
            {
                "symbol": "BTCUSDT",
                "recipe_id": e.recipe_id,
                "family": "trend_ma",
                "timeframe": e.timeframe,
                "activation_context": "pooled",
                "net_bps": e.mean_net_bps,
                "fold_id": 0,
                "effective_weight": 1.0,
            }
            for e in evs
        ]
        post = shrink_l1_evidence_hierarchical(
            raw_rows=pd.DataFrame(rows),
            cost_model=ExecutionCostModel(),
            config=PosteriorGateConfig(),
        )
        sleeves = convert_posterior_to_l2_sleeves(
            posterior=post,
            cost_model=ExecutionCostModel(),
            config=L2PosteriorPolicyConfig(),
        )
        assert len(sleeves) > 0

    def test_full_pipeline_end_to_end(self) -> None:
        a = _aligned_4h()
        panel = _panel()
        *_, sleeves = run_alpha_foundry_pipeline(
            panels=[panel],
            recipes={"r1": R1},
            aligned=a,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=CheapGateConfig(),
            posterior_gate_config=PosteriorGateConfig(),
            l2_config=L2PosteriorPolicyConfig(),
            symbols=("BTCUSDT",),
        )
        assert len(sleeves) > 0

    def test_diversity_and_search_spaces_wired(self) -> None:
        panels = [_panel("r1"), _panel("r2")]
        corr = compute_panel_correlation_matrix(panels)
        assert corr.shape == (2, 2)
        budgets = resolve_staged_search_budget(
            n_dimensions={"signal": 5, "risk": 3, "regime": 4, "deployment": 2},
            requested_trials=200,
            seed_count=1,
        )
        assert len(budgets) == 4
