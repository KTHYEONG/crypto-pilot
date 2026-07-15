"""Deterministic cross-timeframe diagnostic snapshots and comparison."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from src.domain.futures.alpha_foundry.contracts import (
    CrossTfDiagnosticRun,
    CrossTfDiagnosticStage,
    CrossTfStageSnapshot,
)

CrossTfDiagnosisClass: TypeAlias = Literal[
    "nondeterministic",
    "panel_construction_coupling",
    "cheap_gate_shared_state",
    "confirmed_tf_corroboration_coupling",
    "manifest_budget_or_pruning_coupling",
    "labeling_or_delivery_coupling",
    "l1_runtime_shared_state",
    "multiple_or_unresolved_couplings",
    "incomplete_trace",
    "no_coupling_reproduced",
]

_STAGE_ORDER: tuple[CrossTfDiagnosticStage, ...] = (
    "native_panels",
    "cheap_evidence",
    "fusion_evidence",
    "canonical_l0",
    "manifest_route",
    "native_labeled_events",
    "l1_delivery_events",
    "outer_folds",
    "l1_result",
)
_REQUIRED_RUNS: tuple[CrossTfDiagnosticRun, ...] = (
    "control",
    "control_repeat",
    "treatment",
    "fusion_ablation",
)


class CrossTfDiagnosticInputError(ValueError):
    """Raised when diagnostic snapshots cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class CrossTfStageComparison:
    """[ADR_20260715_L0_L1_NATIVE_CONTRACT] A/B digest comparison at one timeframe and pipeline stage."""

    timeframe: str
    stage: CrossTfDiagnosticStage
    control_digest: str
    treatment_digest: str
    ablation_digest: str
    treatment_matches_control: bool
    ablation_restores_control: bool
    changed_identity_keys: tuple[str, ...]
    changed_metric_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CrossTfDiagnosis:
    """[ADR_20260715_L0_L1_NATIVE_CONTRACT] Classified result of control/treatment/ablation comparison."""

    schema_version: int
    classification: CrossTfDiagnosisClass
    first_divergence_stage_by_tf: tuple[tuple[str, CrossTfDiagnosticStage | None], ...]
    comparisons: tuple[CrossTfStageComparison, ...]
    input_fingerprint: str
    complete: bool
    notes: tuple[str, ...]


class CrossTfSnapshotRecorder:
    """[ADR_20260715_L0_L1_NATIVE_CONTRACT] Collect snapshots through the diagnostic sink contract."""

    @property
    def snapshots(self) -> tuple[CrossTfStageSnapshot, ...]:
        return tuple(self._snapshots)

    def __init__(self) -> None:
        self._snapshots: list[CrossTfStageSnapshot] = []

    def __call__(
        self,
        *,
        run: CrossTfDiagnosticRun,
        stage: CrossTfDiagnosticStage,
        timeframe: str,
        payload: object,
    ) -> None:
        if not isinstance(payload, CrossTfStageSnapshot):
            raise TypeError("diagnostic payload must be CrossTfStageSnapshot")
        if (payload.run, payload.stage, payload.timeframe) != (run, stage, timeframe):
            raise CrossTfDiagnosticInputError("sink payload identity does not match invocation")
        self._snapshots.append(payload)


def diagnose_snapshots(
    *,
    snapshots: Sequence[CrossTfStageSnapshot],
    common_timeframes: tuple[str, ...] = ("2h", "4h", "6h", "8h", "12h", "1d"),
) -> CrossTfDiagnosis:
    """Compare snapshots from the four prescribed diagnostic runs."""
    indexed: dict[tuple[CrossTfDiagnosticRun, CrossTfDiagnosticStage, str], CrossTfStageSnapshot] = {}
    for snapshot in snapshots:
        key = (snapshot.run, snapshot.stage, snapshot.timeframe)
        if key in indexed:
            raise CrossTfDiagnosticInputError(f"duplicate snapshot: {key}")
        indexed[key] = snapshot

    missing: list[str] = []
    for timeframe in common_timeframes:
        for stage in _STAGE_ORDER:
            missing.extend(
                f"{run}:{stage}:{timeframe}"
                for run in _REQUIRED_RUNS
                if (run, stage, timeframe) not in indexed
            )
    if missing:
        return CrossTfDiagnosis(
            schema_version=1,
            classification="incomplete_trace",
            first_divergence_stage_by_tf=tuple((timeframe, None) for timeframe in common_timeframes),
            comparisons=(),
            input_fingerprint="",
            complete=False,
            notes=(f"missing snapshots: {', '.join(missing)}",),
        )

    comparisons: list[CrossTfStageComparison] = []
    first_divergences: list[tuple[str, CrossTfDiagnosticStage | None]] = []
    nondeterministic = False
    corroboration_confirmed = False
    unresolved = False

    for timeframe in common_timeframes:
        first_stage: CrossTfDiagnosticStage | None = None
        for stage in _STAGE_ORDER:
            control = indexed[("control", stage, timeframe)]
            repeat = indexed[("control_repeat", stage, timeframe)]
            treatment = indexed[("treatment", stage, timeframe)]
            ablation = indexed[("fusion_ablation", stage, timeframe)]
            if control.digest_sha256 != repeat.digest_sha256:
                nondeterministic = True
            treatment_matches = control.digest_sha256 == treatment.digest_sha256
            ablation_restores = control.digest_sha256 == ablation.digest_sha256
            if not treatment_matches and first_stage is None:
                first_stage = stage
            control_metric_names = {name for name, _ in control.metrics}
            treatment_metric_names = {name for name, _ in treatment.metrics}
            changed_metric_keys = tuple(sorted(control_metric_names ^ treatment_metric_names))
            changed_identity_keys = tuple(sorted(set(control.identity_keys) ^ set(treatment.identity_keys)))
            comparisons.append(
                CrossTfStageComparison(
                    timeframe=timeframe,
                    stage=stage,
                    control_digest=control.digest_sha256,
                    treatment_digest=treatment.digest_sha256,
                    ablation_digest=ablation.digest_sha256,
                    treatment_matches_control=treatment_matches,
                    ablation_restores_control=ablation_restores,
                    changed_identity_keys=changed_identity_keys,
                    changed_metric_keys=changed_metric_keys,
                )
            )
        first_divergences.append((timeframe, first_stage))
        if first_stage in {"fusion_evidence", "canonical_l0"}:
            first_comparison = next(
                comparison
                for comparison in comparisons
                if comparison.timeframe == timeframe and comparison.stage == first_stage
            )
            corroboration_confirmed = corroboration_confirmed or first_comparison.ablation_restores_control
            unresolved = unresolved or not first_comparison.ablation_restores_control
        elif first_stage is not None:
            unresolved = True

    if nondeterministic:
        classification: CrossTfDiagnosisClass = "nondeterministic"
    elif all(stage is None for _, stage in first_divergences):
        classification = "no_coupling_reproduced"
    elif corroboration_confirmed and not unresolved:
        classification = "confirmed_tf_corroboration_coupling"
    else:
        classification = "multiple_or_unresolved_couplings"

    return CrossTfDiagnosis(
        schema_version=1,
        classification=classification,
        first_divergence_stage_by_tf=tuple(first_divergences),
        comparisons=tuple(comparisons),
        input_fingerprint="",
        complete=True,
        notes=(),
    )


def write_cross_tf_diagnosis(*, diagnosis: CrossTfDiagnosis, output_path: Path) -> None:
    """Write a compact diagnosis artifact atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(asdict(diagnosis), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
