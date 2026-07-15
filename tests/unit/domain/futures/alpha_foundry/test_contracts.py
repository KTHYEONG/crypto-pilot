from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaRecipe,
    CheapGateConfig,
    ConditionalCellGateConfig,
    ExecutionArmConfig,
    L0DiagnosticConfig,
    L1VerificationUnit,
    L2PosteriorPolicyConfig,
    L2PosteriorSleeve,
    PosteriorGateConfig,
    StagedSearchBudget,
)


class TestAlphaRecipe:
    def test_valid_recipe_creates_successfully(self) -> None:
        recipe = AlphaRecipe(
            recipe_id="trend_ma:ema_12_72:4h",
            family="trend_ma",
            variant="ema_12_72",
            timeframe="4h",
            archetype="trend",
            indicator_params={"fast": 12, "slow": 72},
            side_rule_id="long_only",
            exit_policy_id="atr_trail_2",
            required_fields=("close",),
            causal_lag_bars=1,
            max_turnover_per_year=365.0,
        )
        assert recipe.recipe_id == "trend_ma:ema_12_72:4h"
        assert recipe.causal_lag_bars == 1

    def test_rejects_negative_causal_lag(self) -> None:
        with pytest.raises(ValueError, match="causal_lag_bars must be >= 1"):
            AlphaRecipe(
                recipe_id="bad",
                family="f",
                variant="v",
                timeframe="4h",
                archetype="trend",
                indicator_params={},
                side_rule_id="s",
                exit_policy_id="e",
                required_fields=("close",),
                causal_lag_bars=0,
                max_turnover_per_year=100.0,
            )

    def test_rejects_negative_turnover(self) -> None:
        with pytest.raises(ValueError, match="max_turnover_per_year must be non-negative"):
            AlphaRecipe(
                recipe_id="bad",
                family="f",
                variant="v",
                timeframe="4h",
                archetype="trend",
                indicator_params={},
                side_rule_id="s",
                exit_policy_id="e",
                required_fields=("close",),
                causal_lag_bars=1,
                max_turnover_per_year=-1.0,
            )

    def test_rejects_empty_recipe_id(self) -> None:
        with pytest.raises(ValueError, match="recipe_id must not be empty"):
            AlphaRecipe(
                recipe_id="",
                family="f",
                variant="v",
                timeframe="4h",
                archetype="trend",
                indicator_params={},
                side_rule_id="s",
                exit_policy_id="e",
                required_fields=("close",),
                causal_lag_bars=1,
                max_turnover_per_year=100.0,
            )


class TestL1VerificationUnit:
    def test_is_frozen_dataclass(self) -> None:
        unit = L1VerificationUnit(
            unit_id="u1",
            recipe_id="r1",
            timeframe="4h",
            scope_symbols=("BTCUSDT",),
            prior_mu_bps=0.0,
            prior_sigma_bps=10.0,
            allocated_fold_budget=5,
            early_stop_state="pending",
        )
        with pytest.raises(AttributeError):
            unit.allocated_fold_budget = 10  # type: ignore[misc]


class TestL2PosteriorPolicyConfig:
    def test_valid_config_creates_successfully(self) -> None:
        cfg = L2PosteriorPolicyConfig()
        assert cfg.kelly_fraction == 0.25
        assert cfg.cov_mode == "diagonal"

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("kelly_fraction", 0.0, "kelly_fraction must be positive"),
            ("kelly_fraction", -0.1, "kelly_fraction must be positive"),
            ("cost_safety_mult", 0.5, "cost_safety_mult must be >= 1.0"),
        ],
    )
    def test_rejects_invalid_values(self, field: str, value: float, match: str) -> None:
        kwargs: dict[str, Any] = {
            "k_rank": 3,
            "rebalance_bars": 3,
            "kelly_fraction": 0.25,
            "posterior_z": 0.50,
            "risk_budget_target": 0.50,
            "gross_cap_by_regime": {"bull": 1.0, "bear": 0.35, "crisis": 0.25},
            "cov_mode": "diagonal",
            "cost_safety_mult": 1.25,
            "turnover_penalty": 0.0,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=re.escape(match)):
            L2PosteriorPolicyConfig(**kwargs)

    @pytest.mark.parametrize(
        ("caps", "match"),
        [
            ({"bull": 1.5}, r"regime cap for 'bull' must be in \[0, 1\], got 1.5"),
            ({"bear": -0.1}, r"regime cap for 'bear' must be in \[0, 1\], got -0.1"),
        ],
    )
    def test_rejects_out_of_range_regime_caps(self, caps: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            L2PosteriorPolicyConfig(gross_cap_by_regime=caps)


class TestL2PosteriorSleeve:
    def test_disabled_sleeve_has_side_zero(self) -> None:
        sleeve = L2PosteriorSleeve(
            symbol="BTCUSDT",
            recipe_id="r1",
            family="trend_ma",
            timeframe="4h",
            activation_context="pooled",
            mu_eff_bps=0.0,
            sigma_bps=10.0,
            quality_weight=0.0,
            side=0,
            disabled_reason="non_positive_lcb",
        )
        assert sleeve.side == 0
        assert sleeve.disabled_reason == "non_positive_lcb"


class TestStagedSearchBudget:
    def test_valid_budget_creates_successfully(self) -> None:
        budget = StagedSearchBudget(stage="signal", n_trials=100, min_feasible_eff=0.05, patience=5, seed_count=1)
        assert budget.stage == "signal"
        assert budget.n_trials == 100

    def test_rejects_zero_trials(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            StagedSearchBudget(
                stage="signal",
                n_trials=0,
                min_feasible_eff=0.05,
                patience=5,
                seed_count=1,
            )

    def test_rejects_negative_patience(self) -> None:
        with pytest.raises(ValueError, match="patience must be non-negative"):
            StagedSearchBudget(
                stage="signal",
                n_trials=100,
                min_feasible_eff=0.05,
                patience=-1,
                seed_count=1,
            )


class TestAlphaFoundryRuntimeConfig:
    def test_default_config_creates_successfully(self) -> None:
        cfg = AlphaFoundryRuntimeConfig()
        assert cfg.mode == "off"
        assert cfg.max_recipes_per_family == 64
        assert cfg.top_k_per_family_tf == 5
        assert cfg.initial_fold_budget == 3

    def test_valid_audit_mode(self) -> None:
        cfg = AlphaFoundryRuntimeConfig(mode="audit")
        assert cfg.mode == "audit"

    def test_valid_gate_mode(self) -> None:
        cfg = AlphaFoundryRuntimeConfig(mode="gate")
        assert cfg.mode == "gate"

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="invalid alpha_foundry mode"):
            AlphaFoundryRuntimeConfig(mode="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_recipes_per_family", 0),
            ("top_k_per_family_tf", 0),
            ("initial_fold_budget", 0),
        ],
    )
    def test_rejects_zero_budget_values(self, field: str, value: int) -> None:
        kwargs: dict[str, object] = {
            "mode": "audit",
            "report_dir": Path("/tmp/af_test"),  # noqa: S108
            "max_recipes_per_family": 64,
            "top_k_per_family_tf": 5,
            "initial_fold_budget": 3,
            "cheap_gate": CheapGateConfig(),
            "posterior_gate": PosteriorGateConfig(),
            "l2_policy": L2PosteriorPolicyConfig(),
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=f"{field} must be >= 1"):
            AlphaFoundryRuntimeConfig(**kwargs)  # type: ignore[arg-type]

    def test_rejects_negative_budget_values(self) -> None:
        with pytest.raises(ValueError, match="max_recipes_per_family must be >= 1"):
            AlphaFoundryRuntimeConfig(max_recipes_per_family=-1)

    def test_rejects_empty_include_families_after_trim(self) -> None:
        with pytest.raises(ValueError, match="include_families contains empty string after trim"):
            AlphaFoundryRuntimeConfig(include_families=("", "trend_ma"))

    def test_rejects_empty_exclude_families_after_trim(self) -> None:
        with pytest.raises(ValueError, match="exclude_families contains empty string after trim"):
            AlphaFoundryRuntimeConfig(exclude_families=(" ",))


    def test_rejects_invalid_observability_mode(self) -> None:
        """S3-3: invalid observability_mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid observability_mode"):
            AlphaFoundryRuntimeConfig(observability_mode="file")  # type: ignore[arg-type]

    def test_rejects_invalid_gate_schema(self) -> None:
        """S3-4: gate_schema must be 'unified'."""
        with pytest.raises(ValueError, match="gate_schema must be 'unified'"):
            AlphaFoundryRuntimeConfig(gate_schema="v2")  # type: ignore[arg-type]

    def test_rejects_zero_debug_top_k_rows(self) -> None:
        """S3-5: debug_top_k_rows must be >= 1."""
        with pytest.raises(ValueError, match="debug_top_k_rows must be >= 1"):
            AlphaFoundryRuntimeConfig(debug_top_k_rows=0)

    def test_new_fields_default_correctly(self) -> None:
        cfg = AlphaFoundryRuntimeConfig()
        assert cfg.observability_mode == "debug_log"
        assert cfg.debug_top_k_rows == 10
        assert cfg.artifact_write_enabled is False
        assert cfg.gate_schema == "unified"

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("min_conviction_lcb_bps", -0.1, "min_conviction_lcb_bps must be >= 0.0"),
            ("total_l1_verification_budget", 0, "total_l1_verification_budget must be >= 1"),
            ("exploration_budget_fraction", 0.0, "exploration_budget_fraction must be in (0.0, 1.0)"),
            ("exploration_budget_fraction", 1.0, "exploration_budget_fraction must be in (0.0, 1.0)"),
            ("debug_reject_bucket_rows", 0, "debug_reject_bucket_rows must be >= 1"),
            ("max_discovery_units_for_l1", 0, "max_discovery_units_for_l1 must be >= 1"),
            ("max_discovery_event_jaccard", -0.1, "max_discovery_event_jaccard must be in [0.0, 1.0]"),
            ("max_discovery_event_jaccard", 1.1, "max_discovery_event_jaccard must be in [0.0, 1.0]"),
            ("min_discovery_unit_lcb_bps", -0.1, "min_discovery_unit_lcb_bps must be >= 0.0"),
        ],
    )
    def test_rejects_invalid_numeric_fields(self, field: str, value: float | int, match: str) -> None:
        kwargs: dict[str, object] = {
            "mode": "audit",
            "report_dir": Path("/tmp/af_test"),  # noqa: S108
            "max_recipes_per_family": 64,
            "include_families": (),
            "exclude_families": (),
            "top_k_per_family_tf": 5,
            "initial_fold_budget": 3,
            "enable_synthetic_recipes": True,
            "min_conviction_lcb_bps": 5.0,
            "total_l1_verification_budget": 30,
            "cheap_gate": CheapGateConfig(),
            "posterior_gate": PosteriorGateConfig(),
            "l2_policy": L2PosteriorPolicyConfig(),
            "enable_fast_discovery_timeframes": False,
            "fast_discovery_timeframes": ("1h", "2h"),
            "enable_correlation_audit": False,
            "observability_mode": "debug_log",
            "debug_top_k_rows": 10,
            "artifact_write_enabled": False,
            "gate_schema": "unified",
            "enable_cost_aware_generation": True,
            "exploration_budget_fraction": 0.15,
            "cost_prior_floor_by_tf": {},
            "use_all_timeframes_in_l0": True,
            "debug_reject_bucket_rows": 5,
            "enable_failure_attribution": False,
            "enable_conditional_l0_cells": False,
            "enable_execution_arms": False,
            "enable_horizon_sweep": False,
            "conditional_cell": ConditionalCellGateConfig(),
            "execution_arm": ExecutionArmConfig(),
            "horizon_sweep_bars": (1, 2, 3, 6),
            "diagnostic": L0DiagnosticConfig(),
            "enable_discovery_unit_handoff": False,
            "max_discovery_units_for_l1": 12,
            "max_discovery_event_jaccard": 0.80,
            "min_discovery_unit_lcb_bps": 0.0,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=re.escape(match)):
            AlphaFoundryRuntimeConfig(**kwargs)  # type: ignore[arg-type]

    def test_rejects_bad_conditional_cell_config(self) -> None:
        with pytest.raises(ValueError, match="min_cell_events must be >= 1"):
            ConditionalCellGateConfig(min_cell_events=0)

    def test_rejects_bad_execution_arm_config(self) -> None:
        with pytest.raises(ValueError, match="unsupported execution style"):
            ExecutionArmConfig(styles=("bogus_style",))  # type: ignore[arg-type]

    def test_rejects_bad_diagnostic_config(self) -> None:
        with pytest.raises(ValueError, match="calibration_fraction must be in \\(0,1\\)"):
            L0DiagnosticConfig(calibration_fraction=1.0)


class TestVersionSprawlControl:
    """S2-17: V2 suffix contracts should not be exposed."""

    def test_alpha_gate_evidence_v2_not_exposed(self) -> None:
        import src.domain.futures.alpha_foundry.contracts as c
        assert not hasattr(c, "AlphaGateEvidenceV2")

    def test_evaluate_panel_gate_v2_not_exposed_via_public_import(self) -> None:
        from src.domain.futures.alpha_foundry.cheap_gate import evaluate_panel_gate
        assert evaluate_panel_gate is not None
        assert not hasattr(evaluate_panel_gate, "__name__") or "v2" not in evaluate_panel_gate.__name__


class TestAlphaGateConfig:
    """l0_signal_yield_improvement: Track B gate config changes."""

    def test_funding_flow_carry_family_event_floor_raised(self) -> None:
        from src.domain.futures.alpha_foundry.contracts import AlphaGateConfig

        cfg = AlphaGateConfig()
        assert cfg.family_event_floors.get("funding_flow_carry", 0) >= 200
