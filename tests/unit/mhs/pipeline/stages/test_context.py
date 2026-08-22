"""PipelineContext carrier-field tests (S7 fold-scoped growth-budget mapping)."""

from __future__ import annotations

from src.mhs.pipeline.context import PipelineContext


def test_fold_growth_budget_target_vol_defaults_to_none() -> None:
    # The fold-scoped growth-budget target-vol carrier mirrors
    # _fold_committee_weights -- defaulting to None so every non-growth_budget
    # run stays byte-identical.
    field = PipelineContext.__dataclass_fields__["_fold_growth_budget_target_vol"]
    assert field.default is None
