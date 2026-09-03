from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

from src.research.evaluation.reliability import FoldDistributionResult, ReliabilityGateResult

PromotionStatus = Literal["REJECTED", "OBSERVATION_PASS", "HOLDOUT_PASS"]
GateVerdict = Literal["PASS", "FAIL", "PENDING"]


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Append-only identity of a pre-registered, independent return-source hypothesis."""

    hypothesis_id: str
    code_hash: str
    parameters: dict[str, object]
    data_start: str | None
    data_end: str | None
    return_source: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must not be empty")
        if not self.code_hash:
            raise ValueError("code_hash must not be empty")
        if not self.return_source:
            raise ValueError("return_source must not be empty")


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Composed fail-closed promotion verdict with its individual gate evidence."""

    status: PromotionStatus
    observation_verdict: GateVerdict
    fold_gate_pass: bool
    stress_verdict: GateVerdict
    holdout_verdict: GateVerdict | None
    candidate: CandidateIdentity | None = None


def compose_promotion_verdict(
    observation: ReliabilityGateResult,
    folds: FoldDistributionResult,
    stress: ReliabilityGateResult,
    holdout: ReliabilityGateResult | None,
) -> PromotionResult:
    """Compose existing gate evidence into a fail-closed promotion verdict.

    REJECTED unless observation and stress are PASS and the fold gate is true;
    OBSERVATION_PASS when those hold without a passing holdout; HOLDOUT_PASS
    additionally requires a PASS holdout. Absent or non-PASS holdout evidence
    can never yield HOLDOUT_PASS. This function only composes existing evidence:
    it computes no returns and modifies no gate thresholds.
    """
    if observation.verdict != "PASS" or not folds.gate_pass or stress.verdict != "PASS":
        status: PromotionStatus = "REJECTED"
    elif holdout is None or holdout.verdict != "PASS":
        status = "OBSERVATION_PASS"
    else:
        status = "HOLDOUT_PASS"

    return PromotionResult(
        status=status,
        observation_verdict=observation.verdict,
        fold_gate_pass=folds.gate_pass,
        stress_verdict=stress.verdict,
        holdout_verdict=holdout.verdict if holdout is not None else None,
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen promotion contract surface."""
    assert {f.name for f in fields(CandidateIdentity)} == {
        "hypothesis_id", "code_hash", "parameters",
        "data_start", "data_end", "return_source",
    }
    assert {f.name for f in fields(PromotionResult)} == {
        "status", "observation_verdict", "fold_gate_pass",
        "stress_verdict", "holdout_verdict", "candidate",
    }
    assert compose_promotion_verdict.__name__ == "compose_promotion_verdict"


_check_contract()
