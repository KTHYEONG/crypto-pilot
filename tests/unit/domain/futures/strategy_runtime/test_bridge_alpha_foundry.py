from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from src.domain.futures.alpha_foundry.bridge_helpers import (
    bind_panels_to_alpha_recipes,
    run_alpha_foundry_l0_gate,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaRecipe,
    CheapGateConfig,
    CheapGateEvidence,
    L2PosteriorPolicyConfig,
    PanelRecipeBinding,
    PosteriorGateConfig,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _b(panel_index: int, recipe_id: str = "r1", family: str = "trend_ma",
       variant: str = "ema_12_72_4h", source: str = "catalog_exact") -> PanelRecipeBinding:
    return PanelRecipeBinding(
        panel_index=panel_index, recipe_id=recipe_id, family=family,
        variant=variant, source=source,
    )


def make_aligned_market_data(t: int = 128, n: int = 2) -> AlignedMarketData:
    dt = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-01T00:00:00") + np.timedelta64(t, "h"),
        np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )
    close = np.linspace(100.0, 120.0, t, dtype=np.float64).reshape(-1, 1)
    close = np.repeat(close, n, axis=1)
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt,
        symbols=tuple(f"SYM{i}USDT" for i in range(n)),
        open_2d=close.copy(),
        high_2d=close * 1.001,
        low_2d=close * 0.999,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.full((t, n), 0.00005, dtype=np.float64),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros((t, n), dtype=np.bool_),
        kill_mask=np.zeros((t, n), dtype=np.bool_),
    )


def make_panel(
    *,
    family: str = "trend_ma",
    variant: str = "ema_12_72_4h",
    recipe_id: str = "trend_ma__ema_12_72__4h",
) -> CandidateSignalPanel:
    aligned = make_aligned_market_data(t=12, n=2)
    t, n = aligned.close_2d.shape
    return CandidateSignalPanel(
        family=family,
        variant=variant,
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
        metadata={"recipe_id": recipe_id},
        archetype="trend",
    )


def make_recipe() -> AlphaRecipe:
    return AlphaRecipe(
        recipe_id="trend_ma__ema_12_72__4h",
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


def make_passing_evidence(recipe_id: str = "trend_ma__ema_12_72__4h") -> CheapGateEvidence:
    return CheapGateEvidence(
        recipe_id=recipe_id,
        timeframe="4h",
        symbol_scope="global",
        n_events=100,
        effective_n=80.0,
        mean_net_bps=5.0,
        nw_tstat=2.5,
        block_lcb_bps=2.0,
        rank_ic=0.05,
        monotonic_bucket_score=0.6,
        regime_edges_bps={},
        cost_drag_ratio=0.3,
        turnover_per_year=50.0,
        novelty_corr_max=0.3,
        incremental_rank_ic=0.02,
        compute_cost_score=0.1,
        gate_passed=True,
        reject_reasons=(),
    )


def make_rejected_evidence(recipe_id: str = "bad_recipe") -> CheapGateEvidence:
    return CheapGateEvidence(
        recipe_id=recipe_id,
        timeframe="4h",
        symbol_scope="global",
        n_events=5,
        effective_n=3.0,
        mean_net_bps=-1.0,
        nw_tstat=0.5,
        block_lcb_bps=-2.0,
        rank_ic=0.01,
        monotonic_bucket_score=0.3,
        regime_edges_bps={},
        cost_drag_ratio=0.9,
        turnover_per_year=500.0,
        novelty_corr_max=0.9,
        incremental_rank_ic=0.0,
        compute_cost_score=0.5,
        gate_passed=False,
        reject_reasons=("insufficient_events", "weak_tstat"),
    )


def make_runtime_config(tmp_path: Path, mode: str = "audit") -> AlphaFoundryRuntimeConfig:
    return AlphaFoundryRuntimeConfig(
        mode=mode,
        report_dir=tmp_path,
        max_recipes_per_family=64,
        top_k_per_family_tf=5,
        initial_fold_budget=3,
        cheap_gate=CheapGateConfig(),
        posterior_gate=PosteriorGateConfig(),
        l2_policy=L2PosteriorPolicyConfig(),
    )


class TestAlphaFoundryOffMode:
    """S1-2: off ↔ legacy behavior unchanged."""

    def test_off_mode_returns_no_report(self) -> None:
        panel = make_panel()
        recipe = make_recipe()
        aligned = make_aligned_market_data(t=12, n=2)
        config = make_runtime_config(Path("/tmp/af_test"), mode="off")  # noqa: S108

        result = run_alpha_foundry_l0_gate(
            panels=[panel],
            bindings=(),
            recipes={recipe.recipe_id: recipe},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            runtime_config=config,
            run_id="test_off",
            timeframe="4h",
        )

        assert result.report is None
        assert len(result.panels_for_l1) == 1
        assert result.panels_for_l1[0].variant == panel.variant


class TestAlphaFoundryAuditMode:
    """S1-3: audit generates report, all panels preserved."""

    def test_audit_all_panels_preserved(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=[
                make_passing_evidence("r1"),
                make_rejected_evidence("r2"),
            ],
        )

        panel_pass = make_panel(variant="ema_12_72_4h", recipe_id="r1")
        panel_reject = make_panel(family="bad_family", variant="bad_variant", recipe_id="r2")
        panel_unmatched = make_panel(family="unknown", variant="no_match_4h", recipe_id="r3")

        bindings = [
            _b(0),
            _b(1, recipe_id="r2", family="bad_family", variant="bad_variant"),
        ]

        aligned = make_aligned_market_data(t=12, n=2)
        config = make_runtime_config(Path("/tmp/af_test"), mode="audit")  # noqa: S108
        recipe_pass = make_recipe()
        recipe_pass = dataclasses.replace(recipe_pass, recipe_id="r1")

        result = run_alpha_foundry_l0_gate(
            panels=[panel_pass, panel_reject, panel_unmatched],
            bindings=bindings,
            recipes={"r1": recipe_pass},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            runtime_config=config,
            run_id="test_audit",
            timeframe="4h",
        )

        assert result.report is not None
        assert result.report.mode == "audit"
        assert result.report.n_panels_in == 3
        assert result.report.n_bound_panels == 2
        assert result.report.n_evidence == 2
        assert result.report.n_passed == 1
        assert result.report.n_rejected == 1
        assert len(result.panels_for_l1) == 3


class TestAlphaFoundryGateMode:
    """S1-4: gate filters to passed bound panels only."""

    def test_gate_keeps_only_passed_bound_panels(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=[
                make_passing_evidence("r1"),
                make_rejected_evidence("r2"),
            ],
        )

        panel_pass = make_panel(variant="ema_12_72_4h", recipe_id="r1")
        panel_reject = make_panel(family="bad_family", variant="bad_variant", recipe_id="r2")
        panel_unmatched = make_panel(family="unknown", variant="no_match_4h", recipe_id="r3")

        bindings = [
            _b(0),
            _b(1, recipe_id="r2", family="bad_family", variant="bad_variant"),
        ]

        aligned = make_aligned_market_data(t=12, n=2)
        config = make_runtime_config(Path("/tmp/af_test"), mode="gate")  # noqa: S108
        recipe_pass = make_recipe()
        recipe_pass = dataclasses.replace(recipe_pass, recipe_id="r1")

        result = run_alpha_foundry_l0_gate(
            panels=[panel_pass, panel_reject, panel_unmatched],
            bindings=bindings,
            recipes={"r1": recipe_pass},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            runtime_config=config,
            run_id="test_gate",
            timeframe="4h",
        )

        assert result.report is not None
        assert result.report.mode == "gate"
        assert result.report.n_panels_in == 3
        assert result.report.n_bound_panels == 2
        assert result.report.n_passed == 1
        assert result.report.n_rejected == 1
        assert len(result.panels_for_l1) == 1
        assert result.panels_for_l1[0].variant == "ema_12_72_4h"

    def test_gate_zero_survivors_returns_empty_panels(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=[make_rejected_evidence("r1")],
        )

        panel = make_panel(recipe_id="r1")
        bindings = [
            _b(0),
        ]
        aligned = make_aligned_market_data(t=12, n=2)
        config = make_runtime_config(Path("/tmp/af_test"), mode="gate")  # noqa: S108

        result = run_alpha_foundry_l0_gate(
            panels=[panel],
            bindings=bindings,
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            runtime_config=config,
            run_id="test_gate_zero",
            timeframe="4h",
        )

        assert len(result.panels_for_l1) == 0
        assert result.report is not None
        assert result.report.n_passed == 0


class TestAlphaFoundryReportFailure:
    """S3-3: report write failure is fail-closed."""

    def test_report_write_failure_raises_oserror(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=[make_passing_evidence("r1")],
        )
        mocker.patch(
            "pathlib.Path.mkdir",
            side_effect=OSError("permission denied"),
        )

        panel = make_panel(recipe_id="r1")
        bindings = [
            _b(0),
        ]
        aligned = make_aligned_market_data(t=12, n=2)
        config = make_runtime_config(Path("/no-perm"), mode="audit")

        with pytest.raises(OSError, match="permission denied"):
            run_alpha_foundry_l0_gate(
                panels=[panel],
                bindings=bindings,
                recipes={},
                aligned=aligned,
                cost_model=ExecutionCostModel(),
                runtime_config=config,
                run_id="test_fail",
                timeframe="4h",
            )


class TestBindPanelsToAlphaRecipes:
    """S1-5: suffix normalization for recipe binding."""

    def test_exact_variant_binding(self) -> None:
        recipe = make_recipe()
        panel = make_panel(variant="ema_12_72_4h")

        bindings = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            timeframe="4h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=(),
        )

        assert len(bindings) == 1
        assert bindings[0].recipe_id == "trend_ma__ema_12_72__4h"

    def test_suffix_normalized_binding(self) -> None:
        recipe = make_recipe()
        panel = make_panel(variant="ema_12_72_8h")

        bindings = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            timeframe="8h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=(),
        )

        assert len(bindings) == 1

    def test_exclude_families_filters_panel(self) -> None:
        recipe = make_recipe()
        panel = make_panel(family="excluded_family", variant="test_4h")

        bindings = bind_panels_to_alpha_recipes(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            timeframe="4h",
            max_recipes_per_family=64,
            include_families=(),
            exclude_families=("excluded_family",),
        )

        assert len(bindings) == 0

    def test_max_recipes_per_family_respected(self) -> None:
        recipe1 = make_recipe()
        recipe1 = dataclasses.replace(recipe1, recipe_id="r1")
        recipe2 = make_recipe()
        recipe2 = dataclasses.replace(recipe2, recipe_id="r2")
        recipes = {"r1": recipe1, "r2": recipe2}
        panel1 = make_panel(variant="ema_12_72_4h", recipe_id="r1")
        panel2 = make_panel(variant="ema_12_72_8h", recipe_id="r2")

        bindings = bind_panels_to_alpha_recipes(
            panels=[panel1, panel2],
            recipes=recipes,
            timeframe="4h",
            max_recipes_per_family=1,
            include_families=(),
            exclude_families=(),
        )

        assert len(bindings) <= 1


class TestRunAlphaFoundryGateWithBindings:
    """S2-10: unmatched panels excluded from evidence."""

    def test_unmatched_audit_panel_excluded_from_evidence(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=[make_passing_evidence("r1")],
        )
        panel_pass = make_panel(recipe_id="r1")
        panel_unmatched = make_panel(family="unknown", variant="no_match_4h", recipe_id="r3")
        bindings = [
            _b(0),
        ]
        aligned = make_aligned_market_data(t=12, n=2)
        config = make_runtime_config(Path("/tmp/af_test"), mode="audit")  # noqa: S108
        recipe_pass = make_recipe()
        recipe_pass = dataclasses.replace(recipe_pass, recipe_id="r1")

        result = run_alpha_foundry_l0_gate(
            panels=[panel_pass, panel_unmatched],
            bindings=bindings,
            recipes={"r1": recipe_pass},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            runtime_config=config,
            run_id="test_audit_unmatched",
            timeframe="4h",
        )

        assert result.report is not None
        assert result.report.n_bound_panels == 1
        assert result.report.n_evidence == 1
        assert len(result.panels_for_l1) == 2
