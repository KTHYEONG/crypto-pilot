"""Library-admission inputs, proposal IDs, and result types.

Owns the admission request/config dataclasses, the reversible proposal-id
encode/decode, and the per-candidate/proposal result types. May depend on
``models`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.evaluation.policy import HOLDOUT_CUTOFF
from src.research.expert_portfolio.models import ContextualRouterSpec
from src.research.technical_experts.catalog import resolve_technical_candidate


@dataclass(frozen=True, slots=True)
class LibraryAdmissionConfig:
    """Immutable pre-registered admission limits for one library screen.

    Every limit is caller-supplied: no value is tuned on data. ``min_experts``
    and ``max_experts`` bound the structural combination sizes,
    ``min_closed_trades`` and ``min_active_return_bars`` are the per-candidate
    activity evidence requirements, ``max_abs_pairwise_log_return_correlation``
    and ``max_joint_negative_return_rate`` cap the pairwise completed-return
    diversification limits, ``min_context_covered_states`` is the router state
    coverage requirement, and ``max_combinations`` is the exact structural
    combination budget beyond which the evaluator fails closed. ``max_workers``
    is execution telemetry only and never participates in an admission or
    proposal fingerprint.
    """

    min_experts: int
    max_experts: int
    min_closed_trades: int
    min_active_return_bars: int
    max_abs_pairwise_log_return_correlation: float
    max_joint_negative_return_rate: float
    min_context_covered_states: int
    max_combinations: int
    max_workers: int | None = None

    def __post_init__(self) -> None:
        if self.min_experts < 1:
            raise ValueError(f"min_experts must be >= 1, got {self.min_experts}")
        if self.max_experts < self.min_experts:
            raise ValueError(
                f"max_experts must be >= min_experts, got max={self.max_experts} "
                f"min={self.min_experts}"
            )
        if self.min_closed_trades < 0:
            raise ValueError(
                f"min_closed_trades must be >= 0, got {self.min_closed_trades}"
            )
        if self.min_active_return_bars < 0:
            raise ValueError(
                f"min_active_return_bars must be >= 0, got {self.min_active_return_bars}"
            )
        if not 0.0 <= self.max_abs_pairwise_log_return_correlation <= 1.0:
            raise ValueError(
                "max_abs_pairwise_log_return_correlation must be in [0, 1], got "
                f"{self.max_abs_pairwise_log_return_correlation}"
            )
        if not 0.0 <= self.max_joint_negative_return_rate <= 1.0:
            raise ValueError(
                f"max_joint_negative_return_rate must be in [0, 1], got "
                f"{self.max_joint_negative_return_rate}"
            )
        if not 0 <= self.min_context_covered_states <= 6:
            raise ValueError(
                "min_context_covered_states must be between 0 and the 6 known "
                f"states, got {self.min_context_covered_states}"
            )
        if self.max_combinations < 1:
            raise ValueError(
                f"max_combinations must be >= 1, got {self.max_combinations}"
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(
                f"max_workers must be None or >= 1, got {self.max_workers}"
            )

    def fingerprint(self) -> dict[str, object]:
        """JSON-safe admission limits; ``max_workers`` is deliberately excluded."""
        return {
            "min_experts": self.min_experts,
            "max_experts": self.max_experts,
            "min_closed_trades": self.min_closed_trades,
            "min_active_return_bars": self.min_active_return_bars,
            "max_abs_pairwise_log_return_correlation": (
                self.max_abs_pairwise_log_return_correlation
            ),
            "max_joint_negative_return_rate": self.max_joint_negative_return_rate,
            "min_context_covered_states": self.min_context_covered_states,
            "max_combinations": self.max_combinations,
        }


@dataclass(frozen=True, slots=True)
class TechnicalLibraryAdmissionRequest:
    """Immutable request for one sealed library admission diagnostic.

    ``candidate_sources`` and ``symbols`` define the explicit requested
    universe; duplicates, unknown sources, invalid admission bounds, and any
    ``end`` past the sealed holdout cutoff are rejected before execution. The
    diagnostic never exposes a holdout-unseal switch.
    """

    candidate_sources: tuple[str, ...]
    symbols: tuple[str, ...]
    router: ContextualRouterSpec
    admission: LibraryAdmissionConfig
    start: str | None = None
    end: str | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not self.candidate_sources:
            raise ValueError("candidate_sources must not be empty")
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        if len(self.candidate_sources) != len(set(self.candidate_sources)):
            raise ValueError(
                "candidate_sources must not contain duplicates, got "
                f"{self.candidate_sources}"
            )
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError(
                f"symbols must not contain duplicates, got {self.symbols}"
            )
        for source in self.candidate_sources:
            resolve_technical_candidate(source)
        if self.end is not None and pd.Timestamp(self.end, tz="UTC") > HOLDOUT_CUTOFF:
            raise RuntimeError(
                f"Holdout sealed: end {self.end} > {HOLDOUT_CUTOFF}. Library "
                "admission never unseals the holdout."
            )


def admission_proposal_id(expert_ids: tuple[str, ...]) -> str:
    """Return the reversible, deterministic id emitted for one proposal."""
    ordered = tuple(sorted(expert_ids))
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("expert_ids must be non-empty and unique")
    if any(not expert_id for expert_id in ordered):
        raise ValueError("expert_ids must not contain empty ids")
    return "lae-v1:" + "|".join(ordered)


def expert_ids_from_admission_proposal_id(proposal_id: str) -> tuple[str, ...]:
    """Decode a proposal id emitted by :func:`admission_proposal_id`."""
    prefix = "lae-v1:"
    if not proposal_id.startswith(prefix):
        raise ValueError(f"proposal_id must start with {prefix!r}")
    raw = tuple(part for part in proposal_id[len(prefix):].split("|") if part)
    if not raw:
        raise ValueError("proposal_id must include at least one expert id")
    expected = admission_proposal_id(raw)
    if proposal_id != expected:
        raise ValueError("proposal_id expert ids must be lexical and unique")
    return raw


@dataclass(frozen=True, slots=True)
class TechnicalLibraryAdmissionBacktestRequest:
    """Immutable in-memory backtest request for one admission proposal.

    The proposal is materialized directly from expert ids and never resolves a
    catalog or registration.  The request has no holdout-unseal switch; the
    shared sealed-end policy is always applied by the application service.
    """

    expert_ids: tuple[str, ...]
    router: ContextualRouterSpec
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    max_workers: int | None = None
    log_run: bool = False

    def __post_init__(self) -> None:
        if not self.expert_ids:
            raise ValueError("expert_ids must not be empty")
        if len(self.expert_ids) != len(set(self.expert_ids)):
            raise ValueError("expert_ids must be unique")
        if self.initial_equity <= 0:
            raise ValueError(
                f"initial_equity must be > 0, got {self.initial_equity}"
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.end is not None and pd.Timestamp(self.end, tz="UTC") > HOLDOUT_CUTOFF:
            raise RuntimeError(
                f"Holdout sealed: end {self.end} > {HOLDOUT_CUTOFF}. Proposal "
                "backtest never unseals the holdout."
            )


@dataclass(frozen=True, slots=True)
class CandidateAdmissionResult:
    """One candidate's integrity and activity admission evidence."""

    expert_id: str
    closed_trades: int
    active_return_bars: int
    admitted: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class AdmissionProposal:
    """One deterministic expert subset and its admission eligibility verdict."""

    expert_ids: tuple[str, ...]
    eligible: bool

    @property
    def proposal_id(self) -> str:
        return admission_proposal_id(self.expert_ids)


__all__ = [
    "AdmissionProposal",
    "CandidateAdmissionResult",
    "LibraryAdmissionConfig",
    "TechnicalLibraryAdmissionBacktestRequest",
    "TechnicalLibraryAdmissionRequest",
    "admission_proposal_id",
    "expert_ids_from_admission_proposal_id",
]
