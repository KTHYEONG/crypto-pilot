"""Pure quarterly rolling rebalance schedule and selection-key comparison.

Owns the immutable rolling config, the causally resolved ``RebalanceWindow``
construction, the appendable ``RollingSelectionRecord``, the stitched
``RollingLibraryAdmissionReport``, and the deterministic deployability screen
plus selection-key ordering. The schedule never reads the system clock: a
window is derived purely from ``common_start`` and ``as_of``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.research.contracts import CostModel
from src.research.expert_portfolio.admission_reports import (
    LibraryAdmissionBacktestReport,
    LibraryAdmissionReport,
)
from src.research.expert_portfolio.admission_types import AdmissionProposal

_GRID_PERIOD = pd.Timedelta(hours=4)
_SCORING_CALENDAR_MONTHS = 24
_REBALANCE_MONTHS = (1, 4, 7, 10)
_WARMUP_BARS = 201
_SHORTLIST_BUDGET = 24
_MIN_CONTEXT_SAMPLES = 96
_SWITCH_COST = CostModel().fee_rate + CostModel().slippage_rate


@dataclass(frozen=True, slots=True)
class RollingAdmissionConfig:
    """Immutable quarterly walk-forward policy for one rolling profile.

    ``scoring_months`` is the trailing completed calendar months of scored
    history, ``warmup_bars`` the completed 4h bars of indicator/router warm-up
    before the scored start, ``shortlist_budget`` the admission shortlist cap,
    ``min_context_samples`` the frozen per-state completed router samples,
    ``rebalance_months`` the UTC quarter-start months, ``switch_cost`` the
    single-turn fee plus slippage charged when a library changes at a
    rebalance boundary, and ``initial_equity`` the seed of the stitched master
    ledger. ``router_kind`` selects the causal winner allocation
    (``"global_winner_v1"`` or ``"per_symbol_winner_v2"``), ``proposal_search``
    selects exact legacy admission or the bounded same-symbol family-unique
    search, and ``base_delay_bars`` is the single base scenario delay shared by
    the candidate screen and every base proposal. No value is tuned on data.
    """

    profile: str = "technical-5symbol-rolling-v1"
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    scoring_months: int = _SCORING_CALENDAR_MONTHS
    warmup_bars: int = _WARMUP_BARS
    shortlist_budget: int = _SHORTLIST_BUDGET
    min_context_samples: int = _MIN_CONTEXT_SAMPLES
    rebalance_months: tuple[int, ...] = _REBALANCE_MONTHS
    switch_cost: float = _SWITCH_COST
    initial_equity: float = 10_000.0
    router_kind: Literal["global_winner_v1", "per_symbol_winner_v2"] = "global_winner_v1"
    proposal_search: Literal["exact_legacy_v1", "bounded_family_unique_v2"] = "exact_legacy_v1"
    base_delay_bars: int = 0

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile must not be empty")
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError(f"symbols must not contain duplicates, got {self.symbols}")
        if self.scoring_months < 1:
            raise ValueError(
                f"scoring_months must be >= 1, got {self.scoring_months}"
            )
        if self.warmup_bars < 1:
            raise ValueError(f"warmup_bars must be >= 1, got {self.warmup_bars}")
        if self.shortlist_budget < 1:
            raise ValueError(
                f"shortlist_budget must be >= 1, got {self.shortlist_budget}"
            )
        if self.min_context_samples < 1:
            raise ValueError(
                f"min_context_samples must be >= 1, got {self.min_context_samples}"
            )
        if len(self.rebalance_months) != len(set(self.rebalance_months)):
            raise ValueError(
                f"rebalance_months must not contain duplicates, got {self.rebalance_months}"
            )
        if any(not (1 <= month <= 12) for month in self.rebalance_months):
            raise ValueError(
                f"rebalance_months must be in 1..12, got {self.rebalance_months}"
            )
        if self.switch_cost < 0.0:
            raise ValueError(f"switch_cost must be >= 0, got {self.switch_cost}")
        if self.initial_equity <= 0:
            raise ValueError(
                f"initial_equity must be > 0, got {self.initial_equity}"
            )
        if self.router_kind not in ("global_winner_v1", "per_symbol_winner_v2"):
            raise ValueError(
                f"router_kind must be 'global_winner_v1' or 'per_symbol_winner_v2', "
                f"got {self.router_kind!r}"
            )
        if self.proposal_search not in ("exact_legacy_v1", "bounded_family_unique_v2"):
            raise ValueError(
                f"proposal_search must be 'exact_legacy_v1' or "
                f"'bounded_family_unique_v2', got {self.proposal_search!r}"
            )
        if self.base_delay_bars < 0:
            raise ValueError(
                f"base_delay_bars must be >= 0, got {self.base_delay_bars}"
            )
        if self.proposal_search == "bounded_family_unique_v2":
            if self.router_kind != "per_symbol_winner_v2":
                raise ValueError(
                    "proposal_search 'bounded_family_unique_v2' requires "
                    "router_kind 'per_symbol_winner_v2'"
                )
            if self.base_delay_bars < 1:
                raise ValueError(
                    "proposal_search 'bounded_family_unique_v2' requires "
                    "base_delay_bars >= 1 (the shared base scenario)"
                )

    @property
    def warmup_period(self) -> pd.Timedelta:
        """Completed 4h warm-up bars as a duration."""
        return _GRID_PERIOD * self.warmup_bars

    @property
    def scoring_offset(self) -> pd.DateOffset:
        """Trailing completed calendar months of scored history."""
        return pd.DateOffset(months=self.scoring_months)

    def fingerprint(self) -> dict[str, object]:
        """JSON-safe policy parameters; the profile name is the version id."""
        return {
            "profile": self.profile,
            "symbols": list(self.symbols),
            "scoring_months": self.scoring_months,
            "warmup_bars": self.warmup_bars,
            "shortlist_budget": self.shortlist_budget,
            "min_context_samples": self.min_context_samples,
            "rebalance_months": list(self.rebalance_months),
            "switch_cost": self.switch_cost,
            "initial_equity": self.initial_equity,
            "router_kind": self.router_kind,
            "proposal_search": self.proposal_search,
            "base_delay_bars": self.base_delay_bars,
        }


@dataclass(frozen=True, slots=True)
class RebalanceWindow:
    """One causally resolved quarterly selection and deployment window.

    For a rebalance timestamp ``R``: ``observed_end`` is ``R - 4h``,
    ``scored_start`` is ``R - 24 calendar months``, ``load_start`` supplies
    indicator/router history only (``scored_start - warmup``), ``deploy_start``
    is ``R`` and ``deploy_end`` is the next quarter's start minus one 4h bar.
    ``status`` is ``"closed"`` when the deployment quarter is fully complete at
    ``as_of`` and ``"live_or_partial"`` otherwise; an ineligible window (raw
    warm-up before ``common_start``) is never constructed.
    """

    profile: str
    rebalance_start: pd.Timestamp
    scored_start: pd.Timestamp
    observed_end: pd.Timestamp
    load_start: pd.Timestamp
    deploy_start: pd.Timestamp
    deploy_end: pd.Timestamp
    status: Literal["closed", "live_or_partial"]

    def to_report_dict(self) -> dict[str, object]:
        return {
            "rebalance_start": str(self.rebalance_start),
            "scored_start": str(self.scored_start),
            "observed_end": str(self.observed_end),
            "load_start": str(self.load_start),
            "deploy_start": str(self.deploy_start),
            "deploy_end": str(self.deploy_end),
            "status": self.status,
        }


def _as_utc(ts: str | pd.Timestamp | pd.DatetimeIndex) -> pd.Timestamp:
    """Normalize any timestamp to a tz-aware UTC Timestamp."""
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def quarter_boundaries(
    start: pd.Timestamp,
    end: pd.Timestamp,
    months: tuple[int, ...] = _REBALANCE_MONTHS,
) -> Iterator[pd.Timestamp]:
    """Yield every UTC quarter-start boundary in ``[start, end]`` in order."""
    start = _as_utc(start)
    end = _as_utc(end)
    for year in range(start.year - 1, end.year + 1):
        for month in sorted(months):
            boundary = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            if boundary < start:
                continue
            if boundary > end:
                return
            yield boundary


def next_quarter_boundary(
    ts: pd.Timestamp,
    months: tuple[int, ...] = _REBALANCE_MONTHS,
) -> pd.Timestamp:
    """Return the first quarter boundary strictly after ``ts``."""
    ts = _as_utc(ts)
    for boundary in quarter_boundaries(ts, ts + pd.DateOffset(years=1), months):
        if boundary > ts:
            return boundary
    raise ValueError(f"no quarter boundary found after {ts}")


def build_rolling_rebalance_schedule(
    common_start: pd.Timestamp,
    as_of: pd.Timestamp,
    config: RollingAdmissionConfig,
) -> tuple[RebalanceWindow, ...]:
    """Construct every eligible quarterly window up to ``as_of``.

    A window is returned only when its raw warm-up history begins on or after
    ``common_start``, and a deployment quarter whose ``deploy_end`` is still
    after ``as_of`` is classified ``live_or_partial`` so it never enters a
    historical OOS aggregate. ``as_of`` is the only temporal source.
    """
    common_start = _as_utc(common_start)
    as_of = _as_utc(as_of)
    windows: list[RebalanceWindow] = []
    for rebalance_start in quarter_boundaries(common_start, as_of, config.rebalance_months):
        observed_end = rebalance_start - _GRID_PERIOD
        scored_start = rebalance_start - config.scoring_offset
        load_start = scored_start - config.warmup_period
        if load_start < common_start:
            continue
        deploy_start = rebalance_start
        deploy_end = next_quarter_boundary(rebalance_start, config.rebalance_months) - _GRID_PERIOD
        status: Literal["closed", "live_or_partial"] = (
            "closed" if deploy_end <= as_of else "live_or_partial"
        )
        windows.append(
            RebalanceWindow(
                profile=config.profile,
                rebalance_start=rebalance_start,
                scored_start=scored_start,
                observed_end=observed_end,
                load_start=load_start,
                deploy_start=deploy_start,
                deploy_end=deploy_end,
                status=status,
            )
        )
    return tuple(windows)


@dataclass(frozen=True, slots=True)
class RollingSelectionRecord:
    """Immutable record of one rebalance decision, written to the append-only ledger.

    The record binds the frozen profile, the resolved window timestamps, the
    admission status, the selected proposal (``None`` means CASH), the
    incumbent-retention decision, and the code/data provenance. ``snapshot_key``
    is the deterministic hash of the exact inputs so an identical replay is
    idempotent and later data cannot rewrite it.
    """

    profile: str
    rebalance_start: str
    scored_start: str
    observed_end: str
    load_start: str
    deploy_start: str
    deploy_end: str
    status: str
    selection_status: str
    proposal_id: str | None
    expert_ids: tuple[str, ...]
    incumbent_kept: bool
    code_hash: str
    data_hashes: Mapping[str, Mapping[str, str]]
    snapshot_key: str
    recorded_at: str = ""

    def to_payload(self) -> dict[str, object]:
        """Deterministic JSON-safe payload for the rebalance ledger."""
        return {
            "profile": self.profile,
            "rebalance_start": self.rebalance_start,
            "scored_start": self.scored_start,
            "observed_end": self.observed_end,
            "load_start": self.load_start,
            "deploy_start": self.deploy_start,
            "deploy_end": self.deploy_end,
            "status": self.status,
            "selection_status": self.selection_status,
            "proposal_id": self.proposal_id,
            "expert_ids": list(self.expert_ids),
            "incumbent_kept": self.incumbent_kept,
            "code_hash": self.code_hash,
            "data_hashes": {
                symbol: dict(hashes)
                for symbol, hashes in sorted(self.data_hashes.items())
            },
            "snapshot_key": self.snapshot_key,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True, slots=True)
class RollingLibraryAdmissionReport:
    """Stitched result of one rolling replay.

    ``records`` carry every closed and live selection decision in window order
    and the fold metrics summarise only the stitched closed deployment quarters.
    ``oos_start``/``oos_end`` bound the stitched closed OOS period and
    ``oos_return`` is its cumulative gross return before switch costs.
    """

    status: str
    profile: str
    mode: str
    as_of: str
    common_start: str
    common_end: str
    windows: tuple[RebalanceWindow, ...]
    records: tuple[RollingSelectionRecord, ...]
    n_folds: int
    median_fold_cagr: float
    worst_fold_cagr: float
    median_fold_calmar: float
    max_period_contribution: float
    fold_gate_pass: bool
    oos_start: str
    oos_end: str
    oos_return: float

    @property
    def closed_quarter_count(self) -> int:
        return sum(1 for record in self.records if record.status == "closed")

    def to_report_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe CLI output."""
        return {
            "status": self.status,
            "profile": self.profile,
            "mode": self.mode,
            "as_of": self.as_of,
            "common_window": {
                "common_start": self.common_start,
                "common_end": self.common_end,
            },
            "windows": [window.to_report_dict() for window in self.windows],
            "records": [record.to_payload() for record in self.records],
            "stitched_oos": {
                "start": self.oos_start,
                "end": self.oos_end,
                "closed_quarters": self.closed_quarter_count,
                "return": self.oos_return,
                "n_folds": self.n_folds,
                "median_fold_cagr": self.median_fold_cagr,
                "worst_fold_cagr": self.worst_fold_cagr,
                "median_fold_calmar": self.median_fold_calmar,
                "max_period_contribution": self.max_period_contribution,
                "fold_gate_pass": self.fold_gate_pass,
            },
        }


def _is_deployable(report: LibraryAdmissionBacktestReport) -> bool:
    """A proposal is deployable only when every reliability gate passes."""
    return (
        report.observation_gate.verdict == "PASS"
        and report.stress_gate.verdict == "PASS"
        and report.observation_folds.gate_pass
    )


def selection_primary_key(report: LibraryAdmissionBacktestReport) -> float:
    """The first selection-key term: best robust worst-case LCB90, lower wins."""
    return -min(report.observation_gate.lcb90_cagr, report.stress_gate.lcb90_cagr)


def selection_key(report: LibraryAdmissionBacktestReport) -> tuple[object, ...]:
    """Lower-is-better deterministic selection key over deployable reports."""
    return (
        selection_primary_key(report),
        report.observation_folds.max_period_contribution,
        abs(report.observation_metrics.mdd),
        report.diversification_rank_key,
        report.proposal_id,
    )


def _proposal_from_report(report: LibraryAdmissionBacktestReport) -> AdmissionProposal:
    return AdmissionProposal(expert_ids=report.expert_ids, eligible=True)


@dataclass(frozen=True, slots=True)
class RollingCandidateAuditRecord:
    """Immutable, deterministic audit of one rebalance's candidate screen.

    The identity is exactly ``(profile, rebalance_start, snapshot_key)`` and the
    payload deliberately carries no wall-clock time, so an identical replay
    yields byte-stable JSON. It records the causal window, every candidate
    admission outcome, the screen proposal set with pair diagnostics and
    eligibility, the shortlist order, each shortlisted proposal's
    base/stress/fold gates plus selection key, the selected proposal, and the
    CASH/incumbent outcome. Writing an audit never mutates the rebalance ledger,
    current-profile pointer, catalog, or trading state.
    """

    profile: str
    rebalance_start: str
    snapshot_key: str
    window: Mapping[str, object]
    selection: Mapping[str, object]
    candidates: tuple[Mapping[str, object], ...]
    proposals: tuple[Mapping[str, object], ...]
    shortlist: tuple[str, ...]
    training: tuple[Mapping[str, object], ...]
    selected: Mapping[str, object] | None
    selection_status: str
    execution: Mapping[str, object] = field(default_factory=dict)
    incumbent_kept: bool = False
    cash_reason: str | None = None

    @classmethod
    def from_selection(
        cls,
        window: RebalanceWindow,
        selection: LibraryAdmissionReport,
        shortlist: tuple[AdmissionProposal, ...],
        training_reports: tuple[LibraryAdmissionBacktestReport, ...],
        selected: AdmissionProposal | None,
        snapshot_key: str,
    ) -> RollingCandidateAuditRecord:
        """Build one deterministic audit from a window's frozen screen outputs."""
        cash_reason: str | None = None
        if selected is None:
            cash_reason = (
                "no_shortlist_or_incumbent" if not training_reports
                else "no_deployable_proposal"
            )
        return cls(
            profile=window.profile,
            rebalance_start=str(window.rebalance_start),
            snapshot_key=snapshot_key,
            window=window.to_report_dict(),
            selection={
                "status": selection.status,
                "window_start": selection.window_start,
                "window_end": selection.window_end,
                "covered_states": selection.covered_states,
                "coverage_sufficient": selection.coverage_sufficient,
                "structural_combinations": selection.structural_combinations,
                "generated_nodes": selection.generated_nodes,
                "generation_limit": selection.generation_limit,
                "generation_status": selection.generation_status,
                "proposal_count": len(selection.proposals),
            },
            candidates=tuple(
                {
                    "expert_id": candidate.expert_id,
                    "closed_trades": candidate.closed_trades,
                    "active_return_bars": candidate.active_return_bars,
                    "admitted": candidate.admitted,
                    "reason": candidate.reason,
                }
                for candidate in selection.candidates
            ),
            proposals=tuple(
                {
                    "proposal_id": proposal.proposal_id,
                    "expert_ids": list(proposal.expert_ids),
                    "eligible": proposal.eligible,
                    "pair_diagnostics": {
                        "max_abs_log_return_correlation": (
                            proposal.max_abs_pair_log_return_correlation
                        ),
                        "max_joint_negative_rate": proposal.max_pair_joint_negative_rate,
                        "mean_abs_log_return_correlation": (
                            proposal.mean_abs_pair_log_return_correlation
                        ),
                        "mean_joint_negative_rate": (
                            proposal.mean_pair_joint_negative_rate
                        ),
                    },
                }
                for proposal in selection.proposals
            ),
            shortlist=tuple(proposal.proposal_id for proposal in shortlist),
            training=tuple(
                {
                    "proposal_id": report.proposal_id,
                    "observation_gate_verdict": report.observation_gate.verdict,
                    "observation_folds_pass": report.observation_folds.gate_pass,
                    "stress_gate_verdict": report.stress_gate.verdict,
                    "stress_folds_pass": report.stress_folds.gate_pass,
                    "selection_primary": selection_primary_key(report),
                    "diversification_rank_key": list(report.diversification_rank_key),
                }
                for report in training_reports
            ),
            selected=(
                {
                    "proposal_id": selected.proposal_id,
                    "expert_ids": list(selected.expert_ids),
                }
                if selected is not None
                else None
            ),
            selection_status=(
                "selected" if selected is not None else "cash"
            ),
            cash_reason=cash_reason,
        )

    def to_payload(self) -> dict[str, object]:
        """Deterministic JSON-safe payload; never contains wall-clock time."""
        return {
            "profile": self.profile,
            "rebalance_start": self.rebalance_start,
            "snapshot_key": self.snapshot_key,
            "window": dict(self.window),
            "selection": dict(self.selection),
            "candidates": [dict(entry) for entry in self.candidates],
            "proposals": [dict(entry) for entry in self.proposals],
            "shortlist": list(self.shortlist),
            "training": [dict(entry) for entry in self.training],
            "selected": (
                dict(self.selected) if self.selected is not None else None
            ),
            "selection_status": self.selection_status,
            "execution": dict(self.execution),
            "incumbent_kept": self.incumbent_kept,
            "cash_reason": self.cash_reason,
        }

    def to_canonical_bytes(self) -> bytes:
        """Byte-stable canonical serialization for idempotency and audit tests."""
        return json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        ).encode("utf-8")


def rolling_admission_config_for_profile(
    profile_name: str,
    symbols: tuple[str, ...],
) -> RollingAdmissionConfig:
    """Resolve the rolling config for a frozen profile name.

    ``technical-5symbol-rolling-v1`` keeps the exact global-winner router, the
    legacy all-combination admission, and its zero-delay base scenario while
    ``technical-5symbol-rolling-v2`` selects the per-symbol winner router, the
    bounded same-symbol family-unique search, and the shared one-bar base
    scenario. An unknown profile name fails closed with ``ValueError``.
    """
    if profile_name == "technical-5symbol-rolling-v1":
        return RollingAdmissionConfig(
            profile=profile_name,
            symbols=symbols,
            router_kind="global_winner_v1",
            proposal_search="exact_legacy_v1",
            base_delay_bars=0,
        )
    if profile_name == "technical-5symbol-rolling-v2":
        return RollingAdmissionConfig(
            profile=profile_name,
            symbols=symbols,
            router_kind="per_symbol_winner_v2",
            proposal_search="bounded_family_unique_v2",
            base_delay_bars=1,
        )
    raise ValueError(
        f"unknown rolling profile {profile_name!r}; known profiles: "
        "technical-5symbol-rolling-v1, technical-5symbol-rolling-v2"
    )


def select_rebalance_proposal(
    reports: tuple[LibraryAdmissionBacktestReport, ...],
    incumbent: AdmissionProposal | None,
) -> AdmissionProposal | None:
    """Select the deployable proposal for the next deployment quarter.

    Only reports whose observation, stress, and fold gates all pass are
    deployable; ranking is by the best robust worst-case LCB90 first, then
    lower concentration, lower absolute drawdown, the diversification key, and
    the lexical proposal id. A structurally valid incumbent is retained on an
    exact primary-key tie and replaced only by a strictly better challenger.
    When no proposal is deployable the result is ``None``, which the caller must
    treat as CASH for the entire quarter — a failed refresh never keeps a stale
    library alive.
    """
    deployable = [report for report in reports if _is_deployable(report)]
    if not deployable:
        return None
    best = min(deployable, key=selection_key)
    if incumbent is not None:
        incumbent_report = next(
            (report for report in deployable if report.proposal_id == incumbent.proposal_id),
            None,
        )
        if incumbent_report is not None:
            incumbent_primary = selection_primary_key(incumbent_report)
            if selection_primary_key(best) >= incumbent_primary:
                return incumbent
    return _proposal_from_report(best)


def _check_contract() -> None:
    """Executable assertions locking the rolling schedule/selection surface."""
    assert build_rolling_rebalance_schedule.__name__ == "build_rolling_rebalance_schedule"
    assert select_rebalance_proposal((), None) is None
    first = next(
        iter(
            build_rolling_rebalance_schedule(
                pd.Timestamp("2022-04-01", tz="UTC"),
                pd.Timestamp("2026-07-07 20:00", tz="UTC"),
                RollingAdmissionConfig(),
            )
        )
    )
    assert str(first.rebalance_start) == "2024-07-01 00:00:00+00:00"
    assert select_rebalance_proposal.__name__ == "select_rebalance_proposal"
    assert RollingCandidateAuditRecord.from_selection.__name__ == "from_selection"
    v1 = rolling_admission_config_for_profile("technical-5symbol-rolling-v1", ("BTCUSDT",))
    assert v1.router_kind == "global_winner_v1"
    v2 = rolling_admission_config_for_profile("technical-5symbol-rolling-v2", ("BTCUSDT",))
    assert v2.router_kind == "per_symbol_winner_v2"
    assert v2.base_delay_bars == 1


_check_contract()

__all__ = [
    "RebalanceWindow",
    "RollingAdmissionConfig",
    "RollingCandidateAuditRecord",
    "RollingLibraryAdmissionReport",
    "RollingSelectionRecord",
    "build_rolling_rebalance_schedule",
    "next_quarter_boundary",
    "quarter_boundaries",
    "rolling_admission_config_for_profile",
    "select_rebalance_proposal",
    "selection_key",
    "selection_primary_key",
]
