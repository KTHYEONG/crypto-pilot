"""Library-admission inputs, proposal IDs, and result types.

Owns the admission request/config dataclasses, the reversible proposal-id
encode/decode, and the per-candidate/proposal result types. May depend on
``models`` only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.market_data.storage.loaders import timeframe_scale_factor, validate_timeframe
from src.research.evaluation.policy import HOLDOUT_CUTOFF
from src.research.expert_portfolio.models import ContextualRouterSpec
from src.research.technical_experts.catalog import (
    TECHNICAL_CANDIDATES,
    resolve_technical_candidate,
)


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

def scale_admission_config(
    admission: LibraryAdmissionConfig, timeframe: str,
) -> LibraryAdmissionConfig:
    """Rescale the admission config's bar-count limit to a fixed calendar window.

    Only ``min_active_return_bars`` is a bar count and is scaled;
    ``min_closed_trades`` (a trade count) and ``min_context_covered_states`` (a
    state count) are not bar counts and pass through unchanged along with every
    structural bound.
    """
    scale = timeframe_scale_factor(timeframe)
    return dataclasses.replace(
        admission,
        min_active_return_bars=max(1, round(admission.min_active_return_bars * scale)),
    )


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
    timeframe: str = "4h"

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
        validate_timeframe(self.timeframe)
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
    timeframe: str = "4h"
    stop_loss_mode: Literal["fixed_pct", "atr_multiple"] | None = None
    stop_loss_value: float | None = None
    atr_period: int = 14
    trailing_stop: bool = False

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
        validate_timeframe(self.timeframe)
        if self.stop_loss_mode is not None:
            if self.stop_loss_value is None or self.stop_loss_value <= 0.0:
                raise ValueError(
                    f"stop_loss_value must be > 0.0 when stop_loss_mode is set, "
                    f"got {self.stop_loss_value}"
                )
            if self.stop_loss_mode == "fixed_pct" and self.stop_loss_value >= 1.0:
                raise ValueError(
                    f"fixed_pct stop_loss_value must be < 1.0, got "
                    f"{self.stop_loss_value}"
                )
        if self.atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {self.atr_period}")
        if self.end is not None and pd.Timestamp(self.end, tz="UTC") > HOLDOUT_CUTOFF:
            raise RuntimeError(
                f"Holdout sealed: end {self.end} > {HOLDOUT_CUTOFF}. Proposal "
                "backtest never unseals the holdout."
            )


@dataclass(frozen=True, slots=True)
class ExitSweepSetting:
    """One frozen stop-loss exit configuration in a sweep grid.

    Field names/types mirror ``run_technical_expert_backtest``'s own
    parameters for direct pass-through. Frozen/slots makes instances hashable
    so they work as aggregation keys; ``label()`` is a human-readable JSON
    output helper and never participates in gate or admission logic.
    """

    stop_loss_mode: Literal["fixed_pct", "atr_multiple"] | None
    stop_loss_value: float | None
    trailing_stop: bool

    def label(self) -> str:
        if self.stop_loss_mode is None:
            return "baseline_no_stop"
        suffix = "trailing" if self.trailing_stop else "static"
        return f"{self.stop_loss_mode}_{self.stop_loss_value}_{suffix}"


@dataclass(frozen=True, slots=True)
class TechnicalExpertExitSweepRequest:
    """Immutable in-process grid sweep of stop-loss exit settings.

    The sweep evaluates every (candidate, symbol, timeframe, setting) cell
    without any portfolio, router, stress, or promotion machinery; the settings
    grid is derived deterministically from the value tuples. The request never
    constructs an ``ExpertPortfolioSpec`` or ``ContextualRouterSpec``.
    """

    candidate_sources: tuple[str, ...]
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    fixed_pct_values: tuple[float, ...] = (0.03, 0.05, 0.08)
    atr_multiple_values: tuple[float, ...] = (1.5, 2.5, 4.0)
    atr_period: int = 14
    include_baseline: bool = True
    start: str | None = None
    end: str | pd.Timestamp | None = None
    max_workers: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_sources:
            raise ValueError("candidate_sources must not be empty")
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        if not self.timeframes:
            raise ValueError("timeframes must not be empty")
        for timeframe in self.timeframes:
            validate_timeframe(timeframe)
        if self.atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {self.atr_period}")
        for value in self.fixed_pct_values:
            if value <= 0.0 or value >= 1.0:
                raise ValueError(
                    f"fixed_pct_values entries must be in (0.0, 1.0), got {value}"
                )
        for value in self.atr_multiple_values:
            if value <= 0.0:
                raise ValueError(
                    f"atr_multiple_values entries must be > 0.0, got {value}"
                )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.end is not None and pd.Timestamp(self.end, tz="UTC") > HOLDOUT_CUTOFF:
            raise RuntimeError(
                f"Holdout sealed: end {self.end} > {HOLDOUT_CUTOFF}. The exit "
                "sweep never unseals the holdout."
            )

    def settings(self) -> tuple[ExitSweepSetting, ...]:
        """Return the deterministic exit-setting grid for this request.

        Baseline (no stop) first when enabled, then each ``fixed_pct`` value
        crossed with static/trailing, then each ``atr_multiple`` value crossed
        with static/trailing, preserving the declared value order.
        """
        grid: list[ExitSweepSetting] = []
        if self.include_baseline:
            grid.append(ExitSweepSetting(None, None, False))
        for value in self.fixed_pct_values:
            grid.append(ExitSweepSetting("fixed_pct", value, False))
            grid.append(ExitSweepSetting("fixed_pct", value, True))
        for value in self.atr_multiple_values:
            grid.append(ExitSweepSetting("atr_multiple", value, False))
            grid.append(ExitSweepSetting("atr_multiple", value, True))
        return tuple(grid)


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
    """One deterministic expert subset, its eligibility, and pair diagnostics.

    The four pair diagnostics are computed strictly from the selection-window
    member pairs (maximum and mean absolute completed log-return correlation,
    maximum and mean joint-negative-return rate); no portfolio metric,
    promotion verdict, or OOS return ever participates.
    """

    expert_ids: tuple[str, ...]
    eligible: bool
    max_abs_pair_log_return_correlation: float = 0.0
    max_pair_joint_negative_rate: float = 0.0
    mean_abs_pair_log_return_correlation: float = 0.0
    mean_pair_joint_negative_rate: float = 0.0

    @property
    def proposal_id(self) -> str:
        return admission_proposal_id(self.expert_ids)

    def rank_key(self) -> tuple[float, float, float, float, str]:
        """Lower-is-better bounded diversification rank over the member pairs."""
        return (
            self.max_abs_pair_log_return_correlation,
            self.max_pair_joint_negative_rate,
            self.mean_abs_pair_log_return_correlation,
            self.mean_pair_joint_negative_rate,
            self.proposal_id,
        )

    def to_report_dict(self) -> dict[str, object]:
        """Deterministic JSON-safe proposal representation for CLI stdout."""
        return {
            "proposal_id": self.proposal_id,
            "expert_ids": list(self.expert_ids),
            "eligible": self.eligible,
            "pair_diagnostics": {
                "max_abs_log_return_correlation": self.max_abs_pair_log_return_correlation,
                "max_joint_negative_rate": self.max_pair_joint_negative_rate,
                "mean_abs_log_return_correlation": self.mean_abs_pair_log_return_correlation,
                "mean_joint_negative_rate": self.mean_pair_joint_negative_rate,
            },
        }


@dataclass(frozen=True, slots=True)
class TechnicalLibraryAdmissionPipelineRequest:
    """Immutable one-execution discovery-plus-OOS-backtest request.

    ``selection`` is the sealed selection request, ``evaluation_start`` is the
    out-of-sample boundary that must be strictly later than ``selection.end``,
    ``evaluation_end`` must not exceed the sealed holdout cutoff,
    ``max_backtest_proposals`` is the positive shortlist budget, and
    ``initial_equity`` seeds every child OOS proposal backtest.
    """

    selection: TechnicalLibraryAdmissionRequest
    evaluation_start: str | pd.Timestamp
    evaluation_end: str | pd.Timestamp
    max_backtest_proposals: int
    initial_equity: float = 10_000.0

    def __post_init__(self) -> None:
        if self.max_backtest_proposals < 1:
            raise ValueError(
                f"max_backtest_proposals must be >= 1, got {self.max_backtest_proposals}"
            )
        if self.initial_equity <= 0:
            raise ValueError(
                f"initial_equity must be > 0, got {self.initial_equity}"
            )
        evaluation_start = pd.Timestamp(self.evaluation_start, tz="UTC")
        evaluation_end = pd.Timestamp(self.evaluation_end, tz="UTC")
        selection_end = (
            pd.Timestamp(self.selection.end, tz="UTC")
            if self.selection.end is not None
            else HOLDOUT_CUTOFF
        )
        if evaluation_start <= selection_end:
            raise ValueError(
                f"evaluation_start {self.evaluation_start} must be strictly later "
                f"than selection.end {self.selection.end}"
            )
        if evaluation_end <= evaluation_start:
            raise ValueError(
                "evaluation_end must be strictly later than evaluation_start, got "
                f"{self.evaluation_start} .. {self.evaluation_end}"
            )
        if evaluation_end > HOLDOUT_CUTOFF:
            raise RuntimeError(
                f"Holdout sealed: evaluation_end {self.evaluation_end} > "
                f"{HOLDOUT_CUTOFF}. The pipeline never unseals the holdout."
            )


def technical_5symbol_2022_v1_profile() -> TechnicalLibraryAdmissionRequest:
    """Return the first frozen admission-pipeline profile.

    The profile freezes the exact universe, dates, router, activity evidence,
    pair screen, proposal sizes, and combination budget stated in the
    specification; any future threshold change requires a new profile id and a
    fresh out-of-sample period.
    """
    return TechnicalLibraryAdmissionRequest(
        candidate_sources=tuple(
            candidate.return_source for candidate in TECHNICAL_CANDIDATES
        ),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        admission=LibraryAdmissionConfig(
            min_experts=2,
            max_experts=5,
            min_closed_trades=20,
            min_active_return_bars=200,
            max_abs_pairwise_log_return_correlation=0.50,
            max_joint_negative_return_rate=0.15,
            min_context_covered_states=6,
            max_combinations=1_000_000,
            max_workers=None,
        ),
        start="2022-04-01 00:00",
        end="2024-12-31 20:00",
    )


LIBRARY_ADMISSION_PROFILES: Mapping[str, Callable[[], TechnicalLibraryAdmissionRequest]] = {
    "technical-5symbol-2022-v1": technical_5symbol_2022_v1_profile,
}


def resolve_library_admission_profile(name: str) -> TechnicalLibraryAdmissionRequest:
    """Resolve a frozen profile by exact name; an unknown name fails closed."""
    try:
        builder = LIBRARY_ADMISSION_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown library admission profile {name!r}; known profiles: "
            f"{sorted(LIBRARY_ADMISSION_PROFILES)}"
        ) from None
    return builder()


def technical_5symbol_rolling_profile() -> TechnicalLibraryAdmissionRequest:
    """Return the quarterly rolling walk-forward admission profile.

    The universe, router, activity evidence, and pair screen are identical to
    the static 2022 profile; only the search algorithm differs, selected by the
    rolling config as the exact best-first per-size top-N priority search with
    the prefilter disabled by default so the full admitted universe reaches
    screening. The profile carries no fixed selection dates: every rebalance
    decision freezes its own ``as_of`` snapshot and the rolling service resolves
    each scored window causally.
    """
    return TechnicalLibraryAdmissionRequest(
        candidate_sources=tuple(
            candidate.return_source for candidate in TECHNICAL_CANDIDATES
        ),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
        router=ContextualRouterSpec("BTCUSDT", 48, 48, 96),
        admission=LibraryAdmissionConfig(
            min_experts=2,
            max_experts=5,
            min_closed_trades=20,
            min_active_return_bars=200,
            max_abs_pairwise_log_return_correlation=0.50,
            max_joint_negative_return_rate=0.15,
            min_context_covered_states=6,
            max_combinations=1_000_000,
            max_workers=None,
        ),
        start=None,
        end=None,
    )


ROLLING_LIBRARY_ADMISSION_PROFILES: Mapping[str, Callable[[], TechnicalLibraryAdmissionRequest]] = {
    "technical-5symbol-rolling": technical_5symbol_rolling_profile,
}


def resolve_rolling_library_admission_profile(name: str) -> TechnicalLibraryAdmissionRequest:
    """Resolve a frozen rolling profile by exact name; an unknown name fails closed."""
    try:
        builder = ROLLING_LIBRARY_ADMISSION_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown rolling library admission profile {name!r}; known profiles: "
            f"{sorted(ROLLING_LIBRARY_ADMISSION_PROFILES)}"
        ) from None
    return builder()


__all__ = [
    "LIBRARY_ADMISSION_PROFILES",
    "ROLLING_LIBRARY_ADMISSION_PROFILES",
    "AdmissionProposal",
    "CandidateAdmissionResult",
    "ExitSweepSetting",
    "LibraryAdmissionConfig",
    "TechnicalExpertExitSweepRequest",
    "TechnicalLibraryAdmissionBacktestRequest",
    "TechnicalLibraryAdmissionPipelineRequest",
    "TechnicalLibraryAdmissionRequest",
    "admission_proposal_id",
    "expert_ids_from_admission_proposal_id",
    "resolve_library_admission_profile",
    "resolve_rolling_library_admission_profile",
    "technical_5symbol_2022_v1_profile",
    "technical_5symbol_rolling_profile",
]
