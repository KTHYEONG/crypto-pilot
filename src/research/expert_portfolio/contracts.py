from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import cast

import pandas as pd

from src.research.evaluation.metrics import Metrics
from src.research.evaluation.policy import HOLDOUT_CUTOFF
from src.research.evaluation.promotion import PromotionResult
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateResult,
)
from src.research.technical_experts.catalog import resolve_technical_candidate

_LCB_Z_SCORES: dict[float, float] = {
    0.80: 0.8416212335729143,
    0.85: 1.0364333894937896,
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
    0.99: 2.3263478740408408,
}


def lcb_z_score(confidence: float) -> float:
    """Standard-normal one-sided lower-quantile for a pre-registered confidence level.

    The quantile is a mathematical constant, never a fitted parameter. Only the
    pre-registered levels are supported so the allocator stays deterministic and
    auditable; an unsupported level fails closed.
    """
    try:
        return _LCB_Z_SCORES[float(confidence)]
    except KeyError as exc:  # noqa: PERF203
        raise ValueError(
            f"confidence must be one of {sorted(_LCB_Z_SCORES)}, got {confidence}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ExpertDefinition:
    """Immutable pre-registered definition of one return source.

    ``expert_id`` is a stable identity, ``return_source`` names the economic
    hypothesis, ``family`` groups correlated parameter variants that must share
    an exposure budget, ``symbols`` names the underlying exposures, ``runner``
    is the existing runner that creates the causal return series, and
    ``code_hash`` pins the producing code. Invalid metadata fails closed; a
    definition rejected by anti-pattern evidence is never loadable.
    """

    expert_id: str
    return_source: str
    family: str
    symbols: tuple[str, ...]
    runner: str
    code_hash: str

    def __post_init__(self) -> None:
        if not self.expert_id:
            raise ValueError("expert_id must not be empty")
        if not self.return_source:
            raise ValueError("return_source must not be empty")
        if not self.family:
            raise ValueError("family must not be empty")
        if not self.symbols:
            raise ValueError("symbols must contain at least one underlying symbol")
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError(f"symbols must not contain duplicates, got {self.symbols}")
        if not self.runner:
            raise ValueError("runner must not be empty")
        if not self.code_hash:
            raise ValueError("code_hash must not be empty")


@dataclass(frozen=True, slots=True)
class ContextualRouterSpec:
    """Immutable pre-registered contextual winner router specification.

    ``context_symbol`` names the market whose OHLCV defines the decision
    context, ``trend_lookback_bars`` is the completed trend window,
    ``volatility_lookback_bars`` is the completed rolling-volatility window,
    ``min_context_history_bars`` is the minimum number of completed samples of
    one state before a conditional allocation is permitted, and ``confidence``
    is the lower-confidence-bound level shared with the LCB allocator. Every
    bar count and the confidence level are frozen before the holdout is seen;
    nothing is fitted.
    """

    context_symbol: str
    trend_lookback_bars: int
    volatility_lookback_bars: int
    min_context_history_bars: int
    confidence: float = 0.90

    def __post_init__(self) -> None:
        if not self.context_symbol:
            raise ValueError("context_symbol must not be empty")
        if self.trend_lookback_bars < 1:
            raise ValueError(
                f"trend_lookback_bars must be >= 1, got {self.trend_lookback_bars}"
            )
        if self.volatility_lookback_bars < 1:
            raise ValueError(
                f"volatility_lookback_bars must be >= 1, got {self.volatility_lookback_bars}"
            )
        if self.min_context_history_bars < 1:
            raise ValueError(
                f"min_context_history_bars must be >= 1, got {self.min_context_history_bars}"
            )
        lcb_z_score(self.confidence)


@dataclass(frozen=True, slots=True)
class ExpertPortfolioSpec:
    """Immutable pre-registered expert library and allocator constraints.

    ``experts`` is the eligible library. ``gross_exposure`` caps the total risky
    allocation, ``family_exposure_limit`` caps any single source-family, and
    ``symbol_exposure_limit`` caps any single underlying symbol: correlated
    parameter variants therefore share an exposure budget and cash is always
    feasible. ``min_history_bars`` is the completed-history requirement before
    an expert can receive capital, ``confidence`` is the block-aware lower
    confidence bound level, and ``router`` is the optional pre-registered
    contextual winner router; ``None`` preserves the causal LCB-mix behaviour
    exactly. No constraint is tuned on the sealed result.
    """

    experts: tuple[ExpertDefinition, ...]
    gross_exposure: float = 1.0
    family_exposure_limit: float = 1.0
    symbol_exposure_limit: float = 1.0
    min_history_bars: int = 30
    confidence: float = 0.90
    router: ContextualRouterSpec | None = None

    def __post_init__(self) -> None:
        if not self.experts:
            raise ValueError("experts must contain at least one expert")
        ids = [e.expert_id for e in self.experts]
        if len(ids) != len(set(ids)):
            raise ValueError(f"expert ids must be unique, got {ids}")
        if not 0.0 < self.gross_exposure <= 1.0:
            raise ValueError(
                f"gross_exposure must be in (0, 1], got {self.gross_exposure}"
            )
        if not 0.0 < self.family_exposure_limit <= 1.0:
            raise ValueError(
                f"family_exposure_limit must be in (0, 1], got {self.family_exposure_limit}"
            )
        if not 0.0 < self.symbol_exposure_limit <= 1.0:
            raise ValueError(
                f"symbol_exposure_limit must be in (0, 1], got {self.symbol_exposure_limit}"
            )
        if self.min_history_bars < 1:
            raise ValueError(
                f"min_history_bars must be >= 1, got {self.min_history_bars}"
            )
        lcb_z_score(self.confidence)

    def fingerprint(self) -> dict[str, object]:
        """Deterministic fingerprint over definitions, code hashes, and allocator config.

        A fingerprint changed after registration is a distinct candidate: the
        record binds the evaluation to the exact library that produced it.
        """
        return {
            "experts": [asdict(e) for e in self.experts],
            "gross_exposure": self.gross_exposure,
            "family_exposure_limit": self.family_exposure_limit,
            "symbol_exposure_limit": self.symbol_exposure_limit,
            "min_history_bars": self.min_history_bars,
            "confidence": self.confidence,
            "router": asdict(self.router) if self.router is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ExpertPortfolioEvaluationRequest:
    """Immutable request for one sealed pre-registered expert portfolio evaluation.

    Only a registered ``library_id`` may be supplied; the sealed window flags
    and logging option are the only other switches, so no candidate parameters
    can be tuned on the command line.
    """

    library_id: str
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    unseal_holdout: bool = False
    log_run: bool = True

    def __post_init__(self) -> None:
        if not self.library_id:
            raise ValueError("library_id must not be empty")


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


def _check_contract() -> None:
    """Executable assertions locking the frozen expert-portfolio contract surface."""
    assert ExpertDefinition(
        "pair_residual_v1", "cointegration_residual", "pair_residual",
        ("A", "B"), "run_pair_residual", "abc",
    ).symbols == ("A", "B")
    assert {f.name for f in fields(ExpertDefinition)} == {
        "expert_id", "return_source", "family", "symbols", "runner", "code_hash",
    }
    spec = ExpertPortfolioSpec(experts=(
        ExpertDefinition(
            "pair_residual_v1", "cointegration_residual", "pair_residual",
            ("A", "B"), "run_pair_residual", "abc",
        ),
        ExpertDefinition(
            "pair_residual_v2", "cointegration_residual", "pair_residual",
            ("C", "D"), "run_pair_residual", "def",
        ),
    ))
    assert spec.gross_exposure == 1.0
    assert "experts" in spec.fingerprint()
    assert spec.fingerprint()["router"] is None
    router = ContextualRouterSpec("BTCUSDT", 60, 20, 30)
    assert router.min_context_history_bars == 30
    assert router.trend_lookback_bars == 60
    assert {f.name for f in fields(ContextualRouterSpec)} == {
        "context_symbol", "trend_lookback_bars", "volatility_lookback_bars",
        "min_context_history_bars", "confidence",
    }
    routed = ExpertPortfolioSpec(experts=spec.experts, router=router)
    assert routed.fingerprint()["router"] == asdict(router)
    assert routed.fingerprint() != spec.fingerprint()
    assert {f.name for f in fields(ExpertPortfolioEvaluationRequest)} == {
        "library_id", "start", "end", "initial_equity", "unseal_holdout", "log_run",
    }


def _check_library_admission_contract() -> None:
    """Executable assertions locking the library admission contract surface."""
    config = LibraryAdmissionConfig(2, 4, 1, 1, 0.8, 0.5, 1, 100, max_workers=1)
    assert config.max_workers == 1
    assert "max_workers" not in config.fingerprint()
    request = TechnicalLibraryAdmissionRequest(
        ("technical_macd_histogram_regime_long_v1",),
        ("BTCUSDT",),
        ContextualRouterSpec("BTCUSDT", 60, 20, 30),
        LibraryAdmissionConfig(1, 1, 1, 1, 0.8, 0.5, 1, 1),
    )
    assert request.symbols == ("BTCUSDT",)
    assert {f.name for f in fields(LibraryAdmissionConfig)} == {
        "min_experts", "max_experts", "min_closed_trades", "min_active_return_bars",
        "max_abs_pairwise_log_return_correlation", "max_joint_negative_return_rate",
        "min_context_covered_states", "max_combinations", "max_workers",
    }
    assert {f.name for f in fields(TechnicalLibraryAdmissionRequest)} == {
        "candidate_sources", "symbols", "router", "admission", "start", "end",
    }
    proposal_id = admission_proposal_id(("b", "a"))
    assert proposal_id == "lae-v1:a|b"
    assert expert_ids_from_admission_proposal_id(proposal_id) == ("a", "b")
    backtest_request = TechnicalLibraryAdmissionBacktestRequest(
        ("technical_macd_histogram_regime_long_v1:BTCUSDT",),
        ContextualRouterSpec("BTCUSDT", 60, 20, 30),
        max_workers=1,
    )
    assert backtest_request.initial_equity == 10_000.0
    assert backtest_request.log_run is False


_check_contract()
_check_library_admission_contract()
