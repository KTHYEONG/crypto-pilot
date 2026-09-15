"""Backtest reliability certification (P0_STOP_THE_LINE + P3 validation tracks).

A report is COMPLETE yet ``UNCERTIFIED`` while any accounting, point-in-time,
tradability, independence, or input-seal defect survives: certification gates
deployment readiness without rewriting the alpha Research-GO verdict
(GATE-DEPLOYMENT). Retrospective deployed-config replay and independent
walk-forward evidence are disclosed as separate tracks that must never be
conflated (INV-TWO-EVIDENCE-TRACKS).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import pandas as pd

from src.mhs.data_provenance import DataEvidenceTier, DataProvenanceResult
from src.mhs.evidence import DeploymentReadinessResult


class BacktestCertificationLevel(StrEnum):
    """Certification ladder; only the top two rungs admit deployment."""

    UNCERTIFIED = "uncertified"
    HISTORICAL_ROBUSTNESS = "historical_robustness"
    REPRODUCIBLE = "reproducible"
    FORWARD_VALIDATED = "forward_validated"


@dataclasses.dataclass(frozen=True, slots=True)
class BacktestReliabilityResult:
    """Reliability verdict attached additively to the diagnostic report."""

    eligible: bool
    certification_level: BacktestCertificationLevel
    reason_codes: tuple[str, ...]
    limitations: tuple[str, ...]
    input_manifest_digest: str | None
    forward_observation_count: int


def evaluate_backtest_reliability(
    *,
    primary_valid: bool,
    primary_invalid_reasons: tuple[str, ...],
    selection_overlap_fraction: float,
    fold_committee_weight_leak: Mapping[str, float] | None,
    input_provenance: DataProvenanceResult,
    data_limitations: tuple[str, ...],
    forward_provenance: DataProvenanceResult | None = None,
) -> BacktestReliabilityResult:
    """Certify one backtest run from validity, independence, and provenance."""
    reasons: list[str] = []
    if not primary_valid:
        reasons.append("PRIMARY_EXECUTION_INVALID")
        reasons.extend(primary_invalid_reasons)
    overlap = float(selection_overlap_fraction)
    if overlap > 0.0:
        reasons.append("SELECTION_WINDOW_OVERLAP")
    leaks = dict(fold_committee_weight_leak) if fold_committee_weight_leak else {}
    if any(float(value) > 0.0 for value in leaks.values()):
        reasons.append("FOLD_COMMITTEE_WEIGHT_LEAK")
    sealed = bool(input_provenance.valid) and input_provenance.tier is DataEvidenceTier.REPRODUCIBLE_ARCHIVE
    if not sealed:
        reasons.extend(input_provenance.reason_codes)
    forward_ok = (
        forward_provenance is not None
        and bool(forward_provenance.valid)
        and forward_provenance.tier is DataEvidenceTier.FORWARD_OBSERVED
    )
    if not forward_ok:
        reasons.append("FORWARD_EVIDENCE_INCOMPLETE")
    if not primary_valid:
        level = BacktestCertificationLevel.UNCERTIFIED
    elif overlap > 0.0 or any(float(value) > 0.0 for value in leaks.values()) or not sealed:
        level = BacktestCertificationLevel.HISTORICAL_ROBUSTNESS
    elif forward_ok:
        level = BacktestCertificationLevel.FORWARD_VALIDATED
    else:
        level = BacktestCertificationLevel.REPRODUCIBLE
    eligible = level in (BacktestCertificationLevel.REPRODUCIBLE, BacktestCertificationLevel.FORWARD_VALIDATED)
    forward_count = int(forward_provenance.files_checked) if forward_provenance is not None else 0
    return BacktestReliabilityResult(
        eligible=eligible,
        certification_level=level,
        reason_codes=tuple(reasons),
        limitations=tuple(data_limitations),
        input_manifest_digest=input_provenance.manifest_digest,
        forward_observation_count=forward_count,
    )


def gate_deployment_readiness(
    deployment: DeploymentReadinessResult, reliability: BacktestReliabilityResult
) -> DeploymentReadinessResult:
    """Combine the alpha Research-GO verdict with the reliability gate.

    Deployment stays ready only when both agree; the input object is never
    mutated (frozen additive gating).
    """
    allowed = bool(reliability.eligible)
    return dataclasses.replace(
        deployment,
        research_go_eligible=bool(deployment.research_go_eligible) and allowed,
        execution_go_eligible=bool(deployment.execution_go_eligible) and allowed,
        pilot_go_eligible=bool(deployment.pilot_go_eligible) and allowed,
        scale_go_eligible=bool(deployment.scale_go_eligible) and allowed,
    )


def build_validation_track_disclosure(
    *,
    selection_overlap_fraction: float,
    fold_committee_weight_leak: Mapping[str, float] | None,
    top_level_boundary: pd.Timestamp,
    fold_boundaries: Sequence[pd.Timestamp],
) -> dict[str, Any]:
    """Disclose the retrospective and independent walk-forward tracks."""
    overlap = float(selection_overlap_fraction)
    leaks = dict(fold_committee_weight_leak) if fold_committee_weight_leak else {}
    leaked = any(float(value) > 0.0 for value in leaks.values())
    limitations: list[str] = []
    if overlap > 0.0:
        limitations.append("SELECTION_WINDOW_OVERLAP")
    if leaked:
        limitations.append("FOLD_COMMITTEE_WEIGHT_LEAK")
    independent = bool(overlap == 0.0 and not leaked)
    return {
        "retrospective_deployed_config": {
            "independent_oos": False,
            "boundary": top_level_boundary.isoformat(),
            "selection_overlap_fraction": overlap,
        },
        "independent_walk_forward": {
            "independent_oos": independent,
            "fold_boundaries": [boundary.isoformat() for boundary in fold_boundaries],
            "fold_committee_weight_leak": leaks,
            "limitations": limitations,
        },
    }
