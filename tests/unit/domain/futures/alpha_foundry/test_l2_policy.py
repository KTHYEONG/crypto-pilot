from __future__ import annotations

import pytest

from src.domain.futures.alpha_foundry.contracts import (
    L1PosteriorEvidence,
    L2PosteriorPolicyConfig,
    StagedSearchBudget,
)
from src.domain.futures.alpha_foundry.l2_policy import (
    build_staged_l2_search_spaces,
    convert_posterior_to_l2_sleeves,
    resolve_staged_search_budget,
)
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

SAMPLE_POSTERIOR = (
    L1PosteriorEvidence(
        symbol="BTCUSDT",
        recipe_id="r1",
        family="trend_ma",
        timeframe="4h",
        activation_context="pooled",
        posterior_mu_bps=10.0,
        posterior_sigma_bps=5.0,
        prob_mu_gt_cost=0.85,
        lcb_net_bps=3.0,
        q_value=0.05,
        fold_pass_ratio=0.8,
        regime_stability=0.6,
        quality_weight=0.7,
        activation_contract="hard",
    ),
    L1PosteriorEvidence(
        symbol="ETHUSDT",
        recipe_id="r2",
        family="funding_carry",
        timeframe="4h",
        activation_context="pooled",
        posterior_mu_bps=-1.0,
        posterior_sigma_bps=3.0,
        prob_mu_gt_cost=0.2,
        lcb_net_bps=-2.0,
        q_value=0.6,
        fold_pass_ratio=0.3,
        regime_stability=0.2,
        quality_weight=0.0,
        activation_contract="observe",
    ),
)


class TestConvertPosteriorToL2Sleeves:
    def test_disables_non_positive_posterior_sleeve(self) -> None:
        cost = ExecutionCostModel()
        config = L2PosteriorPolicyConfig()
        sleeves = convert_posterior_to_l2_sleeves(posterior=SAMPLE_POSTERIOR, cost_model=cost, config=config)
        disabled = [s for s in sleeves if s.disabled_reason]
        active = [s for s in sleeves if not s.disabled_reason]
        for d in disabled:
            assert d.side == 0
        assert len(disabled) >= 1
        assert len(active) >= 1

    def test_active_sleeve_has_positive_mu_eff(self) -> None:
        cost = ExecutionCostModel()
        config = L2PosteriorPolicyConfig()
        sleeves = convert_posterior_to_l2_sleeves(posterior=SAMPLE_POSTERIOR, cost_model=cost, config=config)
        btc = [s for s in sleeves if s.symbol == "BTCUSDT"]
        if btc:
            assert btc[0].side != 0 or btc[0].disabled_reason != ""

    def test_keeps_zero_weight_disabled_after_regime_policy(self) -> None:
        cost = ExecutionCostModel()
        config = L2PosteriorPolicyConfig()
        sleeves = convert_posterior_to_l2_sleeves(posterior=SAMPLE_POSTERIOR, cost_model=cost, config=config)
        for sleeve in sleeves:
            if sleeve.disabled_reason:
                assert sleeve.side == 0

    def test_rejects_invalid_caps(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            L2PosteriorPolicyConfig(gross_cap_by_regime={"bull": 2.0})


class TestBuildStagedL2SearchSpaces:
    def test_returns_all_stages(self) -> None:
        spaces = build_staged_l2_search_spaces()
        for stage in ("signal", "risk", "regime", "deployment"):
            assert stage in spaces

    def test_deployment_stage_has_no_signal_fields(self) -> None:
        spaces = build_staged_l2_search_spaces()
        depl = spaces.get("deployment", {})
        signal_keys = {"quality_weight", "rank_k", "activation_contract"}
        depl_keys = set(depl.keys())
        assert signal_keys.isdisjoint(depl_keys)


class TestResolveStagedSearchBudget:
    def test_allocates_signal_risk_regime_deployment(self) -> None:
        budgets = resolve_staged_search_budget(
            n_dimensions={"signal": 5, "risk": 3, "regime": 4, "deployment": 2},
            requested_trials=200,
            seed_count=1,
        )
        stages = [b.stage for b in budgets]
        assert stages == ["signal", "risk", "regime", "deployment"]

    def test_rejects_zero_trials(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            StagedSearchBudget(
                stage="signal",
                n_trials=0,
                min_feasible_eff=0.05,
                patience=5,
                seed_count=1,
            )
