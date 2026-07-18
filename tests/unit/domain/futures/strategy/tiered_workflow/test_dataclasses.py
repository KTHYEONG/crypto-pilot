"""Contract tests for tiered-workflow result dataclasses."""

from __future__ import annotations

import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result


def _minimal_layer1_result(*, selected_timeframe: str | None = None) -> Layer1Result:
    return Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=False,
        n_valid=0,
        n_total=0,
        selected_timeframe=selected_timeframe,
    )


def test_selected_timeframe_defaults_to_none() -> None:
    result = _minimal_layer1_result()

    assert result.selected_timeframe is None


def test_selected_timeframe_is_preserved() -> None:
    result = _minimal_layer1_result(selected_timeframe="4h")

    assert result.selected_timeframe == "4h"


def test_layer2_allocation_config_worst_fold_gate_defaults_to_enabled() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig()
    assert cfg.l2_deploy_worst_fold_gate_enabled is True


def test_from_mapping_worst_fold_gate_key_absent_resolves_to_dataclass_default() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig.from_mapping({})
    assert cfg.l2_deploy_worst_fold_gate_enabled is True


def test_from_mapping_worst_fold_gate_explicit_false_still_opts_out() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig.from_mapping(
        {"l2_deploy_worst_fold_gate_enabled": False}
    )
    assert cfg.l2_deploy_worst_fold_gate_enabled is False


def test_from_mapping_kelly_safety_fraction_default_unaffected() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig.from_mapping({})
    assert cfg.l2_deploy_kelly_safety_fraction is None


def test_layer2_allocation_config_from_mapping_rejects_crisis_mdd_margin_out_of_range() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    with pytest.raises(ValueError, match="l2_deploy_crisis_mdd_margin"):
        Layer2AllocationConfig.from_mapping({"l2_deploy_crisis_mdd_margin": 1.5})


def test_layer2_allocation_config_from_mapping_rejects_crisis_mdd_margin_zero() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    with pytest.raises(ValueError, match="l2_deploy_crisis_mdd_margin"):
        Layer2AllocationConfig.from_mapping({"l2_deploy_crisis_mdd_margin": 0.0})


def test_layer2_allocation_config_from_mapping_legacy_params_without_new_fields_uses_defaults() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig.from_mapping({})
    assert cfg.l2_deploy_crisis_mdd_margin == 0.30
    assert cfg.l2_deploy_oos_budget_blend == 0.5
    assert cfg.l2_deploy_oos_floor_cap == 4.0


def test_layer2_allocation_config_explicit_new_fields() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig.from_mapping({
        "l2_deploy_crisis_mdd_margin": 0.25,
        "l2_deploy_oos_budget_blend": 0.6,
        "l2_deploy_oos_floor_cap": 3.0,
    })
    assert cfg.l2_deploy_crisis_mdd_margin == 0.25
    assert cfg.l2_deploy_oos_budget_blend == 0.6
    assert cfg.l2_deploy_oos_floor_cap == 3.0


def test_layer2_allocation_config_from_mapping_reads_l2_wf_n_folds() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig.from_mapping({"l2_wf_n_folds": 6})
    assert cfg.l2_wf_n_folds == 6


def test_layer2_allocation_config_from_mapping_rejects_l2_wf_n_folds_below_two() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    with pytest.raises(ValueError, match="l2_wf_n_folds"):
        Layer2AllocationConfig.from_mapping({"l2_wf_n_folds": 0})


def test_layer2_allocation_config_from_mapping_legacy_params_without_l2_wf_n_folds_uses_default() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    cfg = Layer2AllocationConfig.from_mapping({})
    assert cfg.l2_wf_n_folds == 4
