"""Tests for cross-timeframe diagnostic classification."""

from __future__ import annotations

import pytest

from src.domain.futures.alpha_foundry.contracts import CrossTfStageSnapshot
from src.domain.futures.strategy.tiered_workflow.cross_tf_diagnostics import (
    CrossTfDiagnosticInputError,
    diagnose_snapshots,
)

_STAGES = (
    "native_panels",
    "cheap_evidence",
    "fusion_evidence",
    "canonical_l0",
    "manifest_route",
    "native_labeled_events",
    "l1_delivery_events",
    "terminal_event_audit",
    "outer_folds",
    "l1_result",
)


def _snapshot(*, run: str, stage: str, timeframe: str, digest: str) -> CrossTfStageSnapshot:
    return CrossTfStageSnapshot(
        schema_version=1,
        run=run,  # type: ignore[arg-type]
        stage=stage,  # type: ignore[arg-type]
        timeframe=timeframe,
        digest_sha256=digest,
        item_count=1,
        identity_keys=("recipe_a",),
        metrics=(("sym_count", 3.0),),
    )


def _four_run_snapshots(*, changed_from: str | None, ablation_restores: bool) -> list[CrossTfStageSnapshot]:
    snapshots: list[CrossTfStageSnapshot] = []
    change_index = _STAGES.index(changed_from) if changed_from is not None else len(_STAGES)
    for stage_index, stage in enumerate(_STAGES):
        control = _snapshot(run="control", stage=stage, timeframe="6h", digest=f"control:{stage}")
        repeat = _snapshot(run="control_repeat", stage=stage, timeframe="6h", digest=control.digest_sha256)
        snapshots.extend((control, repeat))
        treatment_digest = f"treatment:{stage}" if stage_index >= change_index else control.digest_sha256
        ablation_digest = control.digest_sha256 if ablation_restores else treatment_digest
        snapshots.append(_snapshot(run="treatment", stage=stage, timeframe="6h", digest=treatment_digest))
        snapshots.append(_snapshot(run="fusion_ablation", stage=stage, timeframe="6h", digest=ablation_digest))
    return snapshots


def test_diagnose_snapshots_when_fusion_ablation_restores_control_confirms_corroboration() -> None:
    # Arrange (Given)
    snapshots = _four_run_snapshots(changed_from="fusion_evidence", ablation_restores=True)

    # Act (When)
    diagnosis = diagnose_snapshots(snapshots=snapshots, common_timeframes=("6h",))

    # Assert (Then)
    assert diagnosis.classification == "confirmed_tf_corroboration_coupling"
    assert diagnosis.first_divergence_stage_by_tf == (("6h", "fusion_evidence"),)


def test_diagnose_snapshots_when_control_repeat_diverges_classifies_nondeterministic() -> None:
    # Arrange (Given)
    snapshots = _four_run_snapshots(changed_from="fusion_evidence", ablation_restores=True)
    snapshots[1] = _snapshot(run="control_repeat", stage="native_panels", timeframe="6h", digest="different")

    # Act (When)
    diagnosis = diagnose_snapshots(snapshots=snapshots, common_timeframes=("6h",))

    # Assert (Then)
    assert diagnosis.classification == "nondeterministic"
    assert diagnosis.complete is True


def test_diagnose_snapshots_when_required_stage_is_missing_returns_incomplete_trace() -> None:
    # Arrange (Given)
    snapshots = _four_run_snapshots(changed_from="fusion_evidence", ablation_restores=True)
    snapshots = [
        snapshot
        for snapshot in snapshots
        if not (snapshot.run == "treatment" and snapshot.stage == "l1_result")
    ]

    # Act (When)
    diagnosis = diagnose_snapshots(snapshots=snapshots, common_timeframes=("6h",))

    # Assert (Then)
    assert diagnosis.classification == "incomplete_trace"
    assert diagnosis.complete is False


def test_diagnose_snapshots_when_duplicate_stage_raises_input_error() -> None:
    # Arrange (Given)
    snapshots = _four_run_snapshots(changed_from=None, ablation_restores=True)
    snapshots.append(snapshots[0])

    # Act (When) / Assert (Then)
    with pytest.raises(CrossTfDiagnosticInputError, match="duplicate snapshot"):
        diagnose_snapshots(snapshots=snapshots, common_timeframes=("6h",))
