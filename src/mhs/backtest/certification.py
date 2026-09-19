"""Dependence-aware process validation and evidence-based approval."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.deploy_gate import DeployGateResult
from src.mhs.execution.contracts import StrategyExecutionReplayResult
from src.mhs.execution.integrity import replay_ledger_certified
from src.mhs.params import GrowthRiskEnvelope
from src.mhs.resources import MhsMemoryBudget

REQUIRED_CHECK_NAMES: tuple[str, ...] = (
    "input_seal",
    "availability",
    "pit_universe",
    "label_maturity",
    "selection_independence",
    "execution_consistency",
    "live_parity",
    "margin_survival",
    "capital_capacity",
    "joint_stress",
    "forward_independence",
)

HISTORICAL_PREREQUISITES: tuple[str, ...] = (
    "input_seal",
    "availability",
    "pit_universe",
    "label_maturity",
    "selection_independence",
    "execution_consistency",
)


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """Bind a named validated requirement to the exact procedure, inputs, code and
    evidence interval. Missing or incompatible evidence cannot become a passing flag."""

    requirement: str
    status: Literal["passed", "failed", "unverified"]
    procedure_digest: str
    input_manifest_digest: str | None
    code_digest: str
    interval_start: pd.Timestamp
    interval_end: pd.Timestamp
    artifact_digest: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.requirement not in REQUIRED_CHECK_NAMES:
            raise DataIntegrityError(f"unknown requirement '{self.requirement}'")
        if self.status not in ("passed", "failed", "unverified"):
            raise DataIntegrityError(f"invalid status '{self.status}'")
        if self.interval_start.tzinfo is None or self.interval_end.tzinfo is None:
            raise DataIntegrityError("evidence interval must be tz-aware")
        if not self.interval_start < self.interval_end:
            raise DataIntegrityError("evidence interval must be non-empty")


@dataclass(frozen=True, slots=True)
class ValidationInferenceSpec:
    """Freeze family-wise error spending and dependence-aware uncertainty controls.
    Related procedures, repeated looks and multiple acceptance endpoints share an
    explicit budget; their independence is not assumed."""

    family_alpha: float
    procedure_budget: int
    look_budget: int
    endpoint_budget: int
    bootstrap_paths: int
    seed: int
    resample_batch_paths: int
    minimum_block_days: int

    def __post_init__(self) -> None:
        if not 0.0 < self.family_alpha < 1.0:
            raise DataIntegrityError(f"family_alpha must be in (0, 1), got {self.family_alpha}")
        for name in ("procedure_budget", "look_budget", "bootstrap_paths", "resample_batch_paths", "minimum_block_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DataIntegrityError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.endpoint_budget, bool) or not isinstance(self.endpoint_budget, int) or self.endpoint_budget < 4:
            raise DataIntegrityError(f"endpoint_budget must be an integer >= 4, got {self.endpoint_budget!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DataIntegrityError(f"seed must be an integer, got {self.seed!r}")

    @property
    def local_alpha(self) -> float:
        """Conservative per-endpoint level without independence assumptions."""
        return float(self.family_alpha / (self.procedure_budget * self.look_budget * self.endpoint_budget))

    @property
    def required_paths(self) -> int:
        """Minimum draws resolving the budgeted tail."""
        return math.ceil(20.0 / self.local_alpha)


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Carry execution-independent evidence boundaries and a journal-reserved look.
    UTC evaluation dates do not establish independence without registration and
    complete consultation history."""

    role: Literal["historical", "forward"]
    procedure_digest: str
    code_digest: str
    input_manifest_digest: str | None
    interval_start: pd.Timestamp
    interval_end: pd.Timestamp
    registered_at: pd.Timestamp | None
    consulted_through: pd.Timestamp | None
    family_id: str | None
    look_ordinal: int | None
    inference_spec: ValidationInferenceSpec | None
    journal_complete: bool
    observed_through: pd.Timestamp

    def __post_init__(self) -> None:
        if self.role not in ("historical", "forward"):
            raise DataIntegrityError(f"invalid role '{self.role}'")
        if self.interval_start.tzinfo is None or self.interval_end.tzinfo is None:
            raise DataIntegrityError("evidence interval must be tz-aware")
        if not self.interval_start < self.interval_end:
            raise DataIntegrityError("evidence interval must be non-empty")
        if self.observed_through.tzinfo is None:
            raise DataIntegrityError("observed_through must be tz-aware")
        if not isinstance(self.journal_complete, bool):
            raise DataIntegrityError("journal_complete must be bool")
        if self.look_ordinal is not None and (
            isinstance(self.look_ordinal, bool) or not isinstance(self.look_ordinal, int) or self.look_ordinal < 1
        ):
            raise DataIntegrityError(f"look_ordinal must be a positive integer or None, got {self.look_ordinal!r}")


@dataclass(frozen=True, slots=True)
class DailyPortfolioEvidence:
    """Bind portfolio returns to the two actual equity observations and latest input
    publication they require, rather than assume that a calendar label is completion."""

    returns: pd.Series
    label_start: pd.DatetimeIndex
    label_end: pd.DatetimeIndex
    available_at: pd.DatetimeIndex

    def __post_init__(self) -> None:
        n = len(self.returns)
        if len(self.label_start) != n or len(self.label_end) != n or len(self.available_at) != n:
            raise DataIntegrityError("daily evidence arrays must share one length")


def inventory_daily_evidence(replay: StrategyExecutionReplayResult) -> DailyPortfolioEvidence:
    """Construct complete daily inventory return intervals from actual equity/timing rows.

    Args:
        replay: Engine-owned equity series and aligned input-availability evidence.
    Returns:
        Observed return values and full economic/publication dependencies. Incomplete
        boundary intervals remain outside formal inference, not zero-filled.
    Raises:
        DataIntegrityError: Equity, chronology or required timing evidence is missing.
    """
    ledger = getattr(replay, "ledger", None)
    if ledger is None:
        raise DataIntegrityError("replay has no ledger")
    equity = getattr(ledger, "equity", None)
    if equity is None or len(equity) == 0:
        raise DataIntegrityError("ledger equity is missing")
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise DataIntegrityError("ledger equity must have a DatetimeIndex")
    if equity.index.hasnans:
        raise DataIntegrityError("ledger equity index must not contain NaT")
    if equity.index.tz is None:
        raise DataIntegrityError("ledger equity index must be timezone-aware")
    if equity.index.has_duplicates or not equity.index.is_monotonic_increasing:
        raise DataIntegrityError("ledger equity index must be unique and increasing")
    values = equity.to_numpy(dtype="float64")
    if not bool(pd.notna(values).all()):
        raise DataIntegrityError("ledger equity must not be missing")
    avail = getattr(replay, "ledger_available_at", None)
    if avail is None:
        raise DataIntegrityError("ledger timing evidence is missing")
    if not isinstance(avail, pd.DatetimeIndex):
        raise DataIntegrityError("ledger timing evidence must be a DatetimeIndex")
    if len(avail) != len(equity):
        raise DataIntegrityError("ledger timing evidence must align with equity rows")
    if avail.hasnans:
        raise DataIntegrityError("ledger timing evidence must not contain NaT")
    if avail.tz is None:
        raise DataIntegrityError("ledger timing evidence must be timezone-aware")
    avail_utc = avail.tz_convert("UTC")
    equity_utc = equity.tz_convert("UTC")
    for event_ts, avail_ts in zip(equity_utc.index, avail_utc, strict=True):
        if avail_ts < event_ts:
            raise DataIntegrityError("ledger availability must be no earlier than its observation")
    days = equity_utc.index.normalize()
    unique_days = pd.DatetimeIndex(sorted(set(days)), tz="UTC")
    if len(unique_days) < 2:
        raise DataIntegrityError("ledger equity must span at least two calendar days")
    day_end: dict[pd.Timestamp, pd.Timestamp] = {}
    day_avail: dict[pd.Timestamp, pd.Timestamp] = {}
    day_level: dict[pd.Timestamp, float] = {}
    for day in unique_days:
        mask = days == day
        positions = equity_utc.index[mask]
        day_end[day] = positions.max()
        day_avail[day] = avail_utc[mask].max()
        day_level[day] = float(equity_utc.loc[mask].iloc[-1])
    ordered = list(unique_days)
    returns: list[float] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    avails: list[pd.Timestamp] = []
    for prev, cur in itertools.pairwise(ordered):
        prev_level = day_level[prev]
        cur_level = day_level[cur]
        if prev_level <= 0.0:
            raise DataIntegrityError("ledger equity levels must be positive")
        returns.append(float(cur_level / prev_level - 1.0))
        starts.append(day_end[prev])
        ends.append(day_end[cur])
        later = day_avail[prev] if day_avail[prev] > day_avail[cur] else day_avail[cur]
        avails.append(later)
    index = pd.DatetimeIndex(ordered[1:], tz="UTC")
    return DailyPortfolioEvidence(
        returns=pd.Series(returns, index=index, dtype="float64"),
        label_start=pd.DatetimeIndex(starts, tz="UTC"),
        label_end=pd.DatetimeIndex(ends, tz="UTC"),
        available_at=pd.DatetimeIndex(avails, tz="UTC"),
    )


@dataclass(frozen=True, slots=True)
class ProcessValidationResult:
    """Separate observable historical performance from reliable forward deployment
    eligibility without hiding the reason an evidence requirement is missing."""

    accounting_valid: bool
    historical_acceptance: Literal["passed", "failed", "unverified"]
    forward_acceptance: Literal["passed", "failed", "unverified"]
    requirements: tuple[EvidenceCheck, ...]
    gate: DeployGateResult
    diagnostic_metrics: Mapping[str, float | None]
    reason_codes: tuple[str, ...]


def _select_formal_returns(
    base_evidence: DailyPortfolioEvidence,
    stress_evidence: DailyPortfolioEvidence,
    context: EvaluationContext,
) -> tuple[pd.Series, pd.Series]:
    start = context.interval_start.tz_convert("UTC")
    end = context.interval_end.tz_convert("UTC")
    observed = context.observed_through.tz_convert("UTC")
    base_mask = [
        (s >= start) and (e <= end) and (a <= observed)
        for s, e, a in zip(
            base_evidence.label_start, base_evidence.label_end, base_evidence.available_at, strict=True
        )
    ]
    stress_mask = [
        (s >= start) and (e <= end) and (a <= observed)
        for s, e, a in zip(
            stress_evidence.label_start, stress_evidence.label_end, stress_evidence.available_at, strict=True
        )
    ]
    base_idx = base_evidence.returns.index[[i for i, keep in enumerate(base_mask) if keep]]
    stress_idx = stress_evidence.returns.index[[i for i, keep in enumerate(stress_mask) if keep]]
    base_ret = base_evidence.returns.loc[base_idx]
    stress_ret = stress_evidence.returns.loc[stress_idx]
    if len(base_ret) == 0 or len(stress_ret) == 0:
        raise DataIntegrityError("no complete eligible intervals inside the registered boundary")
    if not base_ret.index.equals(stress_ret.index):
        raise DataIntegrityError("base and stress evidence intervals disagree")
    expected = pd.date_range(start.normalize(), (end - pd.Timedelta(nanoseconds=1)).normalize(), freq="D", tz="UTC")
    observed_days = set(base_ret.index.normalize())
    required = expected[1:] if len(expected) > 1 else expected
    if any(day not in observed_days for day in required):
        raise DataIntegrityError("evidence coverage is incomplete for the registered interval")
    return (base_ret.astype("float64"), stress_ret.astype("float64"))


def assess_process_validation(
    base: StrategyExecutionReplayResult,
    stress: StrategyExecutionReplayResult,
    *,
    context: EvaluationContext,
    checks: tuple[EvidenceCheck, ...],
    envelope: GrowthRiskEnvelope,
    memory_budget: MhsMemoryBudget,
) -> ProcessValidationResult:
    """Assess economic validity, formal research evidence and forward deployment separately.

    Args:
        base: Primary inventory accounting result.
        stress: Same-decision cost-stress inventory result.
        context: Exact procedure, evidence interval and reserved formal-look controls.
        checks: Named source-validated evidence checks with matching identities.
        envelope: Existing registered growth and survival budgets.
        memory_budget: Resource controls for bounded inference batches.
    Returns:
        Separate acceptance states, transparent diagnostics and one deployment verdict.
    Raises:
        DataIntegrityError: Identities, intervals, requirement names or inference inputs disagree.
    """
    names = [c.requirement for c in checks]
    if len(set(names)) != len(names):
        raise DataIntegrityError("requirement names must be unique per assessment")
    for c in checks:
        if c.requirement not in REQUIRED_CHECK_NAMES:
            raise DataIntegrityError(f"unknown requirement '{c.requirement}'")
    if not isinstance(memory_budget, MhsMemoryBudget):
        raise DataIntegrityError(f"memory_budget must be MhsMemoryBudget, got {type(memory_budget).__name__}")
    accounting_valid = bool(replay_ledger_certified(base) and replay_ledger_certified(stress))
    reasons: list[str] = []
    matched: dict[str, EvidenceCheck] = {}
    for check in checks:
        identity_ok = check.procedure_digest == context.procedure_digest and check.code_digest == context.code_digest
        if context.input_manifest_digest is not None and check.input_manifest_digest is not None:
            identity_ok = identity_ok and check.input_manifest_digest == context.input_manifest_digest
        covers = check.interval_start <= context.interval_start and check.interval_end >= context.interval_end
        if check.status == "passed" and (not identity_ok or not covers or check.artifact_digest is None):
            code = "PROCEDURE_IDENTITY_MISMATCH" if not (identity_ok and covers) else "EVIDENCE_SOURCE_ABSENT"
            matched[check.requirement] = dataclasses.replace(
                check, status="unverified", reason_codes=tuple(sorted(set(check.reason_codes) | {code}))
            )
            reasons.append(f"REQUIREMENT_UNVERIFIED:{check.requirement}")
            reasons.append(code)
        else:
            matched[check.requirement] = check
            if check.status == "failed":
                reasons.append(f"REQUIREMENT_FAILED:{check.requirement}")
            elif check.status == "unverified":
                reasons.append(f"REQUIREMENT_UNVERIFIED:{check.requirement}")
    reasons.extend(f"REQUIREMENT_UNVERIFIED:{name}" for name in REQUIRED_CHECK_NAMES if name not in matched)
    equity_rows = getattr(getattr(base, "ledger", None), "equity", None)
    n_observed = len(equity_rows) if equity_rows is not None else 0
    diagnostics: dict[str, float | None] = {"observed_equity_rows": float(n_observed), "coverage_rows": float(n_observed)}
    infer_keys = (
        "base_ann_log_growth_lcb",
        "stress_ann_log_growth_lcb",
        "p_mdd_breach_ucb",
        "p_ruin_ucb",
    )
    for key in infer_keys:
        diagnostics[key] = None
    if not accounting_valid:
        reasons.append("ACCOUNTING_INVALID")
        reasons.append("INVENTORY_LEDGER_INVALID")
        gate = DeployGateResult(go=False, reason_codes=tuple(sorted(set(reasons))), metrics={"observed_equity_rows": float(n_observed)})
        ordered = tuple(matched[name] for name in REQUIRED_CHECK_NAMES if name in matched)
        return ProcessValidationResult(
            accounting_valid=False,
            historical_acceptance="failed",
            forward_acceptance="failed",
            requirements=ordered,
            gate=gate,
            diagnostic_metrics=dict(diagnostics),
            reason_codes=tuple(sorted(set(reasons))),
        )
    if not context.journal_complete:
        reasons.append("ACCESS_HISTORY_INCOMPLETE")
    hist_statuses = [matched[n].status if n in matched else "unverified" for n in HISTORICAL_PREREQUISITES]
    hist_failed = any(s == "failed" for s in hist_statuses)
    hist_missing = any(s != "passed" for s in hist_statuses)
    historical: Literal["passed", "failed", "unverified"] = "unverified"
    stats_go = False
    stats_failed = False
    if hist_failed:
        historical = "failed"
    elif hist_missing or not context.journal_complete:
        historical = "unverified"
    else:
        spec = context.inference_spec
        if spec is None:
            historical = "unverified"
        elif spec.bootstrap_paths < spec.required_paths:
            historical = "unverified"
            reasons.append("INFERENCE_TAIL_UNRESOLVED")
        else:
            try:
                base_evidence = inventory_daily_evidence(base)
                stress_evidence = inventory_daily_evidence(stress)
                formal_base, formal_stress = _select_formal_returns(base_evidence, stress_evidence, context)
            except DataIntegrityError:
                historical = "unverified"
                reasons.append("EVIDENCE_COVERAGE_INCOMPLETE")
            else:
                from src.mhs.deploy_gate import evaluate_continuous_growth_survival

                batch = min(spec.resample_batch_paths, spec.bootstrap_paths)
                try:
                    verdict = evaluate_continuous_growth_survival(
                        formal_base,
                        formal_stress,
                        envelope=envelope,
                        alpha=spec.local_alpha,
                        n_paths=spec.bootstrap_paths,
                        seed=spec.seed,
                        batch_paths=batch,
                        minimum_block_days=spec.minimum_block_days,
                        memory_budget=memory_budget,
                    )
                except DataIntegrityError:
                    historical = "unverified"
                    reasons.append("EVIDENCE_COVERAGE_INCOMPLETE")
                else:
                    for key in infer_keys:
                        value = verdict.metrics.get(key)
                        diagnostics[key] = float(value) if value is not None else None
                    diagnostics["observed_days"] = float(len(formal_base))
                    if verdict.go:
                        historical = "passed"
                        stats_go = True
                    else:
                        historical = "failed"
                        stats_failed = True
                        reasons.extend(verdict.reason_codes)
    if context.role == "historical":
        forward: Literal["passed", "failed", "unverified"] = "unverified"
        reasons.append("HISTORICAL_ONLY")
        go = False
    else:
        forward_ok = True
        if context.registered_at is None or not context.registered_at.tz_convert("UTC") < context.interval_start.tz_convert("UTC"):
            forward_ok = False
        if context.consulted_through is not None and not context.consulted_through.tz_convert("UTC") < context.interval_start.tz_convert(
            "UTC"
        ):
            forward_ok = False
            reasons.append("FORWARD_INTERVAL_CONSULTED")
        if not context.journal_complete:
            forward_ok = False
        look_ok = (
            context.look_ordinal is not None
            and context.inference_spec is not None
            and 1 <= context.look_ordinal <= context.inference_spec.look_budget
        )
        if not look_ok:
            forward_ok = False
            reasons.append("LOOK_BUDGET_EXHAUSTED")
        all_passed = all(matched.get(n) is not None and matched[n].status == "passed" for n in REQUIRED_CHECK_NAMES)
        if not all_passed:
            forward_ok = False
        any_failed_req = any(matched.get(n) is not None and matched[n].status == "failed" for n in REQUIRED_CHECK_NAMES)
        if forward_ok and stats_go and historical == "passed":
            forward = "passed"
        elif any_failed_req or stats_failed or "FORWARD_INTERVAL_CONSULTED" in reasons:
            forward = "failed"
            forward_ok = False
        else:
            forward = "unverified"
        go = bool(forward_ok and stats_go and historical == "passed" and forward == "passed" and all_passed)
    ordered_checks = tuple(matched[name] for name in REQUIRED_CHECK_NAMES if name in matched)
    finite_metrics = {k: v for k, v in diagnostics.items() if isinstance(v, float)}
    gate = DeployGateResult(go=bool(go), reason_codes=tuple(sorted(set(reasons))), metrics=dict(finite_metrics))
    return ProcessValidationResult(
        accounting_valid=True,
        historical_acceptance=historical,
        forward_acceptance=forward,
        requirements=ordered_checks,
        gate=gate,
        diagnostic_metrics=dict(diagnostics),
        reason_codes=tuple(sorted(set(reasons))),
    )
