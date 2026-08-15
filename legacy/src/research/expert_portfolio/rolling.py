"""Pure quarterly rolling rebalance schedule and selection-key comparison.

Owns the immutable rolling config, the causally resolved ``RebalanceWindow``
construction, the appendable ``RollingSelectionRecord``, the stitched
``RollingLibraryAdmissionReport``, and the deterministic deployability screen
plus selection-key ordering. The schedule never reads the system clock: a
window is derived purely from ``common_start`` and ``as_of``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from src.common.config import FUTURES_DATA_DIR
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import (
    load_ohlcv_1h_as,
    timeframe_period,
    timeframe_scale_factor,
    validate_timeframe,
)
from src.research.contracts import CostModel
from src.research.expert_portfolio.admission_reports import (
    LibraryAdmissionBacktestReport,
    LibraryAdmissionReport,
)
from src.research.expert_portfolio.admission_types import AdmissionProposal
from src.research.portfolio.defaults import DEFAULT_SYMBOLS

_SCORING_CALENDAR_MONTHS = 24
_REBALANCE_MONTHS = (1, 4, 7, 10)
_WARMUP_BARS = 201
_MIN_CONTEXT_SAMPLES = 96
_SWITCH_COST = CostModel().fee_rate + CostModel().slippage_rate


@dataclass(frozen=True, slots=True)
class RollingAdmissionConfig:
    """Immutable quarterly walk-forward policy for one rolling profile.

    ``scoring_months`` is the trailing completed calendar months of scored
    history, ``warmup_bars`` the completed 4h bars of indicator/router warm-up
    before the scored start, ``min_shortlist_budget`` the structural floor on
    the shortlist size (2 candidates per structural size 2..5, the minimum
    needed for a meaningful within-size diversification comparison),
    ``max_backtest_wall_seconds_per_window`` the per-window backtest wall-clock
    time budget, ``min_context_samples`` the frozen per-state completed router
    samples, ``rebalance_months`` the UTC quarter-start months, ``switch_cost``
    the single-turn fee plus slippage charged when a library changes at a
    rebalance boundary, ``initial_equity`` the seed of the stitched master
    ledger, ``hurdle_cost_multiple`` the standard 2x cost-margin trading-system
    convention (edge must clear cost_multiple times the measured annualized
    allocation-turnover cost; not data-tuned), and ``base_delay_bars`` the
    shared one-bar base scenario delay. The rolling path always uses the single
    validated priority family-unique search with ``per_symbol_winner_v2``
    routing internally (hardcoded, not user-selectable), so the legacy
    ``proposal_search``/``router_kind`` selectors are gone. No value is tuned on
    data.
    """

    profile: str = "technical-5symbol-rolling"
    timeframe: str = "4h"
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    scoring_months: int = _SCORING_CALENDAR_MONTHS
    warmup_bars: int = _WARMUP_BARS
    min_shortlist_budget: int = 8
    max_backtest_wall_seconds_per_window: float = 120.0
    min_context_samples: int = _MIN_CONTEXT_SAMPLES
    rebalance_months: tuple[int, ...] = _REBALANCE_MONTHS
    switch_cost: float = _SWITCH_COST
    initial_equity: float = 10_000.0
    base_delay_bars: int = 1
    family_symbol_prefilter_top_k: int | None = None
    hurdle_cost_multiple: float = 2.0
    dynamic_symbol_selection: bool = False
    symbol_universe: tuple[str, ...] = DEFAULT_SYMBOLS
    symbol_top_k: int = 5

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile must not be empty")
        validate_timeframe(self.timeframe)
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
        if self.min_shortlist_budget < 1:
            raise ValueError(
                f"min_shortlist_budget must be >= 1, got {self.min_shortlist_budget}"
            )
        if self.max_backtest_wall_seconds_per_window <= 0:
            raise ValueError(
                "max_backtest_wall_seconds_per_window must be > 0, got "
                f"{self.max_backtest_wall_seconds_per_window}"
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
        if self.base_delay_bars < 1:
            raise ValueError(
                f"base_delay_bars must be >= 1 (the shared base scenario), got "
                f"{self.base_delay_bars}"
            )
        if self.family_symbol_prefilter_top_k is not None and (
            self.family_symbol_prefilter_top_k < 1
        ):
            raise ValueError(
                "family_symbol_prefilter_top_k must be None or >= 1, got "
                f"{self.family_symbol_prefilter_top_k}"
            )
        if self.hurdle_cost_multiple < 0:
            raise ValueError(
                f"hurdle_cost_multiple must be >= 0, got {self.hurdle_cost_multiple}"
            )
        if self.symbol_top_k < 1:
            raise ValueError(f"symbol_top_k must be >= 1, got {self.symbol_top_k}")
        if not self.symbol_universe:
            raise ValueError("symbol_universe must not be empty")
        if len(self.symbol_universe) != len(set(self.symbol_universe)):
            raise ValueError(
                f"symbol_universe must not contain duplicates, got {self.symbol_universe}"
            )

    @property
    def warmup_period(self) -> pd.Timedelta:
        """Completed bars of warm-up as a duration on the research grid.

        ``warmup_bars`` is a 4h-reference count; it is rescaled to the config
        timeframe so the loaded warm-up window is a fixed calendar duration
        instead of shrinking/expanding with the bar grid.
        """
        return timeframe_period(self.timeframe) * max(
            1, round(self.warmup_bars * timeframe_scale_factor(self.timeframe)),
        )

    @property
    def scoring_offset(self) -> pd.DateOffset:
        """Trailing completed calendar months of scored history."""
        return pd.DateOffset(months=self.scoring_months)

    def fingerprint(self) -> dict[str, object]:
        """JSON-safe policy parameters; the profile name is the version id."""
        return {
            "profile": self.profile,
            "timeframe": self.timeframe,
            "symbols": list(self.symbols),
            "scoring_months": self.scoring_months,
            "warmup_bars": self.warmup_bars,
            "min_shortlist_budget": self.min_shortlist_budget,
            "max_backtest_wall_seconds_per_window": self.max_backtest_wall_seconds_per_window,
            "min_context_samples": self.min_context_samples,
            "rebalance_months": list(self.rebalance_months),
            "switch_cost": self.switch_cost,
            "initial_equity": self.initial_equity,
            "base_delay_bars": self.base_delay_bars,
            "family_symbol_prefilter_top_k": self.family_symbol_prefilter_top_k,
            "hurdle_cost_multiple": self.hurdle_cost_multiple,
            "dynamic_symbol_selection": self.dynamic_symbol_selection,
            "symbol_universe": list(self.symbol_universe),
            "symbol_top_k": self.symbol_top_k,
        }


@dataclass(frozen=True, slots=True)
class RebalanceWindow:
    """One causally resolved quarterly selection and deployment window.

    For a rebalance timestamp ``R``: ``observed_end`` is ``R`` minus one
    research-grid bar, ``scored_start`` is ``R - 24 calendar months``,
    ``load_start`` supplies indicator/router history only
    (``scored_start - warmup``), ``deploy_start`` is ``R`` and ``deploy_end`` is
    the next quarter's start minus one grid bar. ``symbols`` is the frozen
    per-window universe (the fixed profile symbols by default, or the
    dynamically selected top-k when ``dynamic_symbol_selection`` is enabled).
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
    symbols: tuple[str, ...]

    def to_report_dict(self) -> dict[str, object]:
        return {
            "rebalance_start": str(self.rebalance_start),
            "scored_start": str(self.scored_start),
            "observed_end": str(self.observed_end),
            "load_start": str(self.load_start),
            "deploy_start": str(self.deploy_start),
            "deploy_end": str(self.deploy_end),
            "status": self.status,
            "symbols": list(self.symbols),
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
    *,
    data_root: Path | None = None,
) -> tuple[RebalanceWindow, ...]:
    """Construct every eligible quarterly window up to ``as_of``.

    A window is returned only when its raw warm-up history begins on or after
    ``common_start``, and a deployment quarter whose ``deploy_end`` is still
    after ``as_of`` is classified ``live_or_partial`` so it never enters a
    historical OOS aggregate. ``as_of`` is the only temporal source. With
    ``config.dynamic_symbol_selection`` enabled, each window freezes its own
    top-k symbols ranked by trailing quote-notional volume strictly before
    ``rebalance_start`` (PIT); otherwise every window carries the fixed profile
    symbols, byte-identical to the pre-dynamic baseline.
    """
    common_start = _as_utc(common_start)
    as_of = _as_utc(as_of)
    period = timeframe_period(config.timeframe)
    root = data_root if data_root is not None else FUTURES_DATA_DIR
    windows: list[RebalanceWindow] = []
    for rebalance_start in quarter_boundaries(common_start, as_of, config.rebalance_months):
        observed_end = rebalance_start - period
        scored_start = rebalance_start - config.scoring_offset
        load_start = scored_start - config.warmup_period
        if load_start < common_start:
            continue
        deploy_start = rebalance_start
        deploy_end = next_quarter_boundary(rebalance_start, config.rebalance_months) - period
        status: Literal["closed", "live_or_partial"] = (
            "closed" if deploy_end <= as_of else "live_or_partial"
        )
        if config.dynamic_symbol_selection:
            window_symbols = select_symbols_for_window(
                rebalance_start, config.symbol_universe, config.symbol_top_k, root,
            )
        else:
            window_symbols = config.symbols
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
                symbols=window_symbols,
            )
        )
    return tuple(windows)

def select_symbols_for_window(
    as_of: pd.Timestamp,
    universe: tuple[str, ...],
    top_k: int,
    data_root: Path,
    *,
    lookback_days: int = 90,
) -> tuple[str, ...]:
    """Rank ``universe`` by trailing quote-notional volume and return the top-k.

    Only bars strictly before ``as_of`` enter the ranking (PIT discipline, no
    lookahead), and the trailing ``lookback_days`` window must be fully covered
    by continuous history or the symbol is excluded rather than back-filled or
    zero-substituted. Ties are broken deterministically by ascending symbol
    name so the result is reproducible for ``fingerprint()``/snapshot keys. A
    missing or integrity-invalid parquet is treated as insufficient history and
    excluded.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    if not universe:
        raise ValueError("universe must not be empty")
    if len(universe) != len(set(universe)):
        raise ValueError(f"universe must not contain duplicates, got {universe}")
    as_of = _as_utc(as_of)
    window_start = as_of - pd.Timedelta(days=lookback_days)

    parts: list[pd.DataFrame] = []
    for symbol in universe:
        path = data_root / "ohlcv" / "1h" / f"{symbol}.parquet"
        if not path.exists():
            continue
        try:
            bars = load_ohlcv_1h_as(path, "1h")
        except DataIntegrityError:
            continue
        prior = bars[bars.index < as_of]
        if prior.empty or prior.index[0] > window_start:
            continue
        if "quote_vol" not in prior.columns or prior.index[-1] < window_start:
            continue
        window = prior[prior.index >= window_start]
        part = window[["quote_vol"]].copy()
        part["symbol"] = symbol
        parts.append(part)

    if not parts:
        return ()
    combined = pd.concat(parts)
    volume = combined.groupby("symbol")["quote_vol"].sum()
    ranked = sorted(volume.index, key=lambda symbol: (-volume[symbol], symbol))
    return tuple(ranked[:top_k])


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
    failure: Mapping[str, object] | None = None

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
            "failure": dict(self.failure) if self.failure else None,
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
                    "expert_count": len(report.expert_ids),
                    "observation_gate_verdict": report.observation_gate.verdict,
                    "observation_lcb90": report.observation_gate.lcb90_cagr,
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

    @classmethod
    def from_failure(
        cls,
        window: RebalanceWindow,
        snapshot_key: str,
        reason: str,
    ) -> RollingCandidateAuditRecord:
        """Minimal audit for a window that failed closed before any selection.

        Records an empty candidate/proposal/shortlist/training state with
        ``selection_status == "fail_closed"`` and ``cash_reason`` carrying the
        failure reason, so the failed window still gets one deterministic,
        append-only audit line and the surviving windows keep theirs.
        """
        return cls(
            profile=window.profile,
            rebalance_start=str(window.rebalance_start),
            snapshot_key=snapshot_key,
            window=window.to_report_dict(),
            selection={},
            candidates=(),
            proposals=(),
            shortlist=(),
            training=(),
            selected=None,
            selection_status="fail_closed",
            cash_reason=reason,
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


def resolve_dynamic_shortlist_budget(
    probe_wall_seconds: tuple[float, ...],
    max_backtest_wall_seconds_per_window: float,
    min_shortlist_budget: int,
) -> int:
    """Resolve the effective per-window shortlist budget from measured probes.

    ``avg = mean(probe_wall_seconds)``; the budget is
    ``max(min_shortlist_budget, floor(max_backtest_wall_seconds_per_window / avg))``.
    No upper clamp exists: the time budget itself is the natural ceiling. Pure
    and deterministic given identical probe measurements; the caller must supply
    real measured probe wall-times, never a hardcoded stand-in.
    """
    if len(probe_wall_seconds) == 0:
        raise ValueError("probe_wall_seconds must not be empty")
    if any(sec <= 0.0 for sec in probe_wall_seconds):
        raise ValueError(
            f"probe_wall_seconds must all be positive, got {probe_wall_seconds}"
        )
    if max_backtest_wall_seconds_per_window <= 0:
        raise ValueError(
            f"max_backtest_wall_seconds_per_window must be > 0, got "
            f"{max_backtest_wall_seconds_per_window}"
        )
    if min_shortlist_budget < 1:
        raise ValueError(
            f"min_shortlist_budget must be >= 1, got {min_shortlist_budget}"
        )
    avg = float(sum(probe_wall_seconds)) / len(probe_wall_seconds)
    return max(
        min_shortlist_budget,
        math.floor(max_backtest_wall_seconds_per_window / avg),
    )


def rolling_admission_config_for_profile(
    profile_name: str,
    symbols: tuple[str, ...],
    *,
    timeframe: str = "4h",
) -> RollingAdmissionConfig:
    """Resolve the rolling config for a frozen profile name.

    ``technical-5symbol-rolling`` is the single canonical rolling profile: the
    per-symbol-winner router, the priority family-unique best-first search, and
    the shared one-bar base scenario are hardcoded internally. An unknown
    profile name fails closed with ``ValueError``. ``timeframe`` defaults to
    ``"4h"`` so legacy callers resolve byte-identical to the baseline.
    """
    if profile_name == "technical-5symbol-rolling":
        return RollingAdmissionConfig(
            profile=profile_name,
            symbols=symbols,
            timeframe=timeframe,
        )
    raise ValueError(
        f"unknown rolling profile {profile_name!r}; known profiles: "
        "technical-5symbol-rolling"
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
    assert select_symbols_for_window.__name__ == "select_symbols_for_window"
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
    assert first.symbols == RollingAdmissionConfig().symbols
    assert select_rebalance_proposal.__name__ == "select_rebalance_proposal"
    assert RollingCandidateAuditRecord.from_selection.__name__ == "from_selection"
    assert RollingCandidateAuditRecord.from_failure.__name__ == "from_failure"
    canonical = rolling_admission_config_for_profile(
        "technical-5symbol-rolling", ("BTCUSDT",),
    )
    assert canonical.min_shortlist_budget == 8
    assert canonical.base_delay_bars == 1
    assert canonical.hurdle_cost_multiple == 2.0
    assert canonical.timeframe == "4h"
    assert canonical.dynamic_symbol_selection is False
    assert canonical.fingerprint()["family_symbol_prefilter_top_k"] is None
    assert resolve_dynamic_shortlist_budget((1.0, 1.0, 1.0, 1.0), 120.0, 8) == 120
    assert resolve_dynamic_shortlist_budget((100.0,), 120.0, 8) == 8


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
    "resolve_dynamic_shortlist_budget",
    "rolling_admission_config_for_profile",
    "select_rebalance_proposal",
    "select_symbols_for_window",
    "selection_key",
    "selection_primary_key",
]
