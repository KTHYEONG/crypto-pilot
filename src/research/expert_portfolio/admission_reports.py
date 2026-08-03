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
    ExitSweepSetting,
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
    structural_combinations: int = 0
    generated_nodes: int = 0
    generation_limit: int = 0
    generation_status: str = "NOT_APPLICABLE"

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
            "structural_combinations": self.structural_combinations,
            "generated_nodes": self.generated_nodes,
            "generation_limit": self.generation_limit,
            "generation_status": self.generation_status,
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
class LibraryAdmissionPipelineReport:
    """Aggregate one-execution result of a frozen admission pipeline profile.

    Carries the resolved window metadata, selection counts, the deterministic
    size-stratified shortlist, and every child out-of-sample proposal backtest
    report. It is an exploration artifact: it never registers or promotes a
    library, and every child run executed with ``log_run=False``.
    """

    status: str
    profile: str
    requested_start: str | None
    common_start: str
    effective_start: str
    selection_end: str
    evaluation_start: str
    evaluation_end: str
    structural_combinations: int
    pair_compatible_count: int
    shortlist: tuple[AdmissionProposal, ...]
    backtests: tuple[LibraryAdmissionBacktestReport, ...] = ()

    @property
    def shortlist_count(self) -> int:
        return len(self.shortlist)

    def to_report_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe aggregate CLI output."""
        return {
            "status": self.status,
            "profile": self.profile,
            "window": {
                "requested_start": self.requested_start,
                "common_start": self.common_start,
                "effective_start": self.effective_start,
                "selection_end": self.selection_end,
            },
            "evaluation": {
                "start": self.evaluation_start,
                "end": self.evaluation_end,
            },
            "selection_counts": {
                "structural_combinations": self.structural_combinations,
                "pair_compatible": self.pair_compatible_count,
                "shortlist": self.shortlist_count,
            },
            "shortlist": [proposal.to_report_dict() for proposal in self.shortlist],
            "backtests": [backtest.to_report_dict() for backtest in self.backtests],
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
    # Telemetry: measured wall-clock seconds of this proposal's base+stress
    # backtest. Never part of fingerprint()/promotion semantics, matching
    # execution_workers' existing treatment.
    wall_seconds: float = 0.0
    code_hash: str = ""
    data_hashes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # Selection-window diversification rank key of the proposing subset. It is
    # advisory provenance supplied by the rolling service and never changes the
    # report's own gate semantics; legacy callers leave it empty.
    diversification_rank_key: tuple[float, float, float, float, str] = (0.0, 0.0, 0.0, 0.0, "")

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
            "diversification_rank_key": list(self.diversification_rank_key),
        }


@dataclass(frozen=True, slots=True)
class ExitSweepCellResult:
    """One (candidate, symbol, timeframe, exit-setting) sweep cell result."""

    candidate: str
    symbol: str
    timeframe: str
    setting: ExitSweepSetting
    cagr: float
    lcb90_cagr: float
    gate_pass: bool
    trade_count: int


@dataclass(frozen=True, slots=True)
class ExitSweepFamilySummary:
    """Pre-aggregated per-(candidate, timeframe, setting) sweep summary."""

    candidate: str
    timeframe: str
    setting: ExitSweepSetting
    symbol_count: int
    mean_cagr: float
    median_lcb90_cagr: float
    gate_pass_count: int


@dataclass(frozen=True, slots=True)
class TechnicalExpertExitSweepReport:
    """Full exit-mechanism sweep result: raw cells plus aggregated summaries.

    ``execution_workers`` and ``wall_seconds`` are execution telemetry only and
    never participate in any gate or admission decision.
    """

    cells: tuple[ExitSweepCellResult, ...]
    family_summary: tuple[ExitSweepFamilySummary, ...]
    execution_workers: int
    wall_seconds: float

    def to_report_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe CLI output.

        Each cell's ``setting`` is flattened to its scalar sub-keys (mirroring
        the asdict-based flattening style used elsewhere in this file) rather
        than nested as an object.
        """

        def _setting_dict(setting: ExitSweepSetting) -> dict[str, object]:
            return {
                "stop_loss_mode": setting.stop_loss_mode,
                "stop_loss_value": setting.stop_loss_value,
                "trailing_stop": setting.trailing_stop,
                "setting_label": setting.label(),
            }

        return {
            "cells": [
                {
                    "candidate": cell.candidate,
                    "symbol": cell.symbol,
                    "timeframe": cell.timeframe,
                    "cagr": cell.cagr,
                    "lcb90_cagr": cell.lcb90_cagr,
                    "gate_pass": cell.gate_pass,
                    "trade_count": cell.trade_count,
                    **_setting_dict(cell.setting),
                }
                for cell in self.cells
            ],
            "family_summary": [
                {
                    "candidate": summary.candidate,
                    "timeframe": summary.timeframe,
                    "symbol_count": summary.symbol_count,
                    "mean_cagr": summary.mean_cagr,
                    "median_lcb90_cagr": summary.median_lcb90_cagr,
                    "gate_pass_count": summary.gate_pass_count,
                    **_setting_dict(summary.setting),
                }
                for summary in self.family_summary
            ],
            "execution_workers": self.execution_workers,
            "wall_seconds": self.wall_seconds,
        }


__all__ = [
    "ExitSweepCellResult",
    "ExitSweepFamilySummary",
    "LibraryAdmissionBacktestReport",
    "LibraryAdmissionPipelineReport",
    "LibraryAdmissionReport",
    "TechnicalExpertExitSweepReport",
]
