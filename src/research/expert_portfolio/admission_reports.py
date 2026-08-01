"""Admission and proposal-backtest report serialization.

Owns the JSON/report dataclasses plus the canonical-bytes helper. May depend on
both ``models`` and ``admission_types``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import cast

from src.research.evaluation.metrics import Metrics
from src.research.evaluation.promotion import PromotionResult
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateResult,
)
from src.research.expert_portfolio.admission_types import (
    AdmissionProposal,
    CandidateAdmissionResult,
    LibraryAdmissionConfig,
)
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class LibraryAdmissionReport:
    """JSON-safe diagnostic result of one library admission screen.

    ``status`` is ``"COMPLETE"`` when the structural search was exact and
    bounded and ``"FAIL_CLOSED"`` when the exact feasible combination count
    exceeds ``max_combinations``; a fail-closed report never proposes a subset.
    ``code_hash`` and ``data_hashes`` are provenance enrichment supplied by the
    application layer. ``execution_workers`` and ``wall_seconds`` are telemetry
    and never participate in the admission or proposal fingerprint.
    """

    status: str
    window_start: str
    window_end: str
    experts: tuple[ExpertDefinition, ...]
    candidates: tuple[CandidateAdmissionResult, ...]
    proposals: tuple[AdmissionProposal, ...]
    context_coverage: Mapping[str, int]
    covered_states: int
    coverage_sufficient: bool
    router: ContextualRouterSpec
    admission: LibraryAdmissionConfig
    code_hash: str = ""
    data_hashes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    execution_workers: int = 0
    wall_seconds: float = 0.0

    def fingerprint(self) -> dict[str, object]:
        """Canonical admission/proposal fingerprint excluding execution telemetry."""
        payload: dict[str, object] = {
            "experts": [
                {
                    "expert_id": e.expert_id,
                    "return_source": e.return_source,
                    "family": e.family,
                    "symbols": list(e.symbols),
                    "runner": e.runner,
                    "code_hash": e.code_hash,
                }
                for e in self.experts
            ],
            "window_start": self.window_start,
            "window_end": self.window_end,
            "router": asdict(self.router),
            "admission": self.admission.fingerprint(),
            "code_hash": self.code_hash,
            "data_hashes": {
                symbol: dict(hashes)
                for symbol, hashes in sorted(self.data_hashes.items())
            },
        }
        return cast(
            "dict[str, object]", json.loads(_canonical_bytes(payload))
        )

    def to_report_dict(self) -> dict[str, object]:
        """Deterministic JSON-safe representation for CLI stdout."""
        return {
            "status": self.status,
            "window": {"start": self.window_start, "end": self.window_end},
            "router": asdict(self.router),
            "admission": self.admission.fingerprint(),
            "experts": [
                {
                    "expert_id": e.expert_id,
                    "return_source": e.return_source,
                    "family": e.family,
                    "symbols": list(e.symbols),
                    "runner": e.runner,
                    "code_hash": e.code_hash,
                }
                for e in self.experts
            ],
            "candidates": [
                {
                    "expert_id": c.expert_id,
                    "closed_trades": c.closed_trades,
                    "active_return_bars": c.active_return_bars,
                    "admitted": c.admitted,
                    "reason": c.reason,
                }
                for c in self.candidates
            ],
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "expert_ids": list(p.expert_ids),
                    "eligible": p.eligible,
                }
                for p in self.proposals
            ],
            "context_coverage": dict(self.context_coverage),
            "covered_states": self.covered_states,
            "coverage_sufficient": self.coverage_sufficient,
            "code_hash": self.code_hash,
            "data_hashes": {
                symbol: dict(hashes)
                for symbol, hashes in sorted(self.data_hashes.items())
            },
            "execution_workers": self.execution_workers,
            "wall_seconds": self.wall_seconds,
            "fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True, slots=True)
class LibraryAdmissionBacktestReport:
    """In-memory base/stress result for one admission proposal.

    This report deliberately carries no ledger record or registration id.  It
    is a diagnostic execution of a proposal, not an ACTIVE-library evaluation.
    """

    status: str
    proposal_id: str
    expert_ids: tuple[str, ...]
    router: ContextualRouterSpec
    window_start: str
    window_end: str
    observation_metrics: Metrics
    observation_gate: ReliabilityGateResult
    observation_folds: FoldDistributionResult
    stress_metrics: Metrics
    stress_gate: ReliabilityGateResult
    stress_folds: FoldDistributionResult
    promotion: PromotionResult
    allocation_cost_total: float
    stress_allocation_cost_total: float
    execution_workers: int
    code_hash: str = ""
    data_hashes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def to_report_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe CLI output."""
        return {
            "status": self.status,
            "proposal_id": self.proposal_id,
            "expert_ids": list(self.expert_ids),
            "router": asdict(self.router),
            "window": {"start": self.window_start, "end": self.window_end},
            "observation": {
                "metrics": asdict(self.observation_metrics),
                "gate": asdict(self.observation_gate),
                "folds": asdict(self.observation_folds),
                "allocation_cost_total": self.allocation_cost_total,
            },
            "stress": {
                "metrics": asdict(self.stress_metrics),
                "gate": asdict(self.stress_gate),
                "folds": asdict(self.stress_folds),
                "allocation_cost_total": self.stress_allocation_cost_total,
            },
            "promotion": asdict(self.promotion),
            "execution_workers": self.execution_workers,
            "code_hash": self.code_hash,
            "data_hashes": {
                symbol: dict(hashes)
                for symbol, hashes in sorted(self.data_hashes.items())
            },
        }


__all__ = [
    "LibraryAdmissionBacktestReport",
    "LibraryAdmissionReport",
]
