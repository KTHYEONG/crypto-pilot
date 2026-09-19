"""Durable consultation and formal-look journal for process evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from src.backtests.contracts import JsonValue
from src.common.errors import DataIntegrityError
from src.mhs.backtest.certification import EvaluationContext, ValidationInferenceSpec
from src.mhs.backtest.labels import ProcessClockSpec
from src.mhs.backtest.selection import NestedSelectionSpec, TrainingWindowSpec
from src.mhs.params import MHS_FINAL_OOS_CUTOFF_2026H1, GrowthRiskEnvelope
from src.mhs.process import ProcessExecutionPolicy, ProcessRiskSizingSpec
from src.mhs.types import ExecutionSpec

JOURNAL_SCHEMA_VERSION = 1
PROCEDURE_SCHEMA_VERSION = 1

_HEX64 = re.compile(r"[0-9a-f]{64}")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS procedures(digest TEXT PRIMARY KEY, canonical TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS families(
    family_id TEXT PRIMARY KEY, procedure_digest TEXT NOT NULL,
    procedure_budget INTEGER NOT NULL, look_budget INTEGER NOT NULL,
    endpoint_budget INTEGER NOT NULL, look_endpoints TEXT NOT NULL,
    judging_start TEXT, registered_at TEXT NOT NULL, registration_digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consultations(
    id INTEGER PRIMARY KEY AUTOINCREMENT, procedure_digest TEXT,
    start TEXT NOT NULL, end TEXT NOT NULL, recorded_at TEXT NOT NULL,
    source TEXT NOT NULL, result_digest TEXT);
CREATE TABLE IF NOT EXISTS attempts(
    attempt_id TEXT PRIMARY KEY, family_id TEXT NOT NULL, role TEXT NOT NULL,
    procedure_digest TEXT NOT NULL, look_ordinal INTEGER, endpoint TEXT,
    requested_start TEXT NOT NULL, requested_end TEXT NOT NULL,
    reserved_at TEXT NOT NULL, pre_read_consulted TEXT NOT NULL, context TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outcomes(
    attempt_id TEXT PRIMARY KEY, status TEXT NOT NULL,
    result_digest TEXT, finished_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans(
    registration_digest TEXT PRIMARY KEY, family_id TEXT NOT NULL, role TEXT NOT NULL,
    procedure_digest TEXT NOT NULL, canonical TEXT NOT NULL, created_at TEXT NOT NULL);
"""


@dataclass(frozen=True, slots=True)
class ProcessProcedureDefinition:
    """Freeze every result-affecting decision and economic contract before evaluation.
    Fitted monthly weights may evolve causally; the estimator, candidate universe,
    clock, costs, sizing and evidence requirements cannot be rewritten by outcomes."""

    schema_version: int
    code_digest: str
    data_policy: str
    universe_partition: Literal["dev", "holdout", "all"]
    member_ids: tuple[str, ...]
    clock: ProcessClockSpec
    selection: NestedSelectionSpec
    inference: ValidationInferenceSpec
    execution_policy: ProcessExecutionPolicy
    risk_sizing: ProcessRiskSizingSpec | None
    sizing_source: Literal["hourly_proxy"]
    member_evidence_source: Literal["daily_step_proxy", "inventory_3m"]
    execution_spec: ExecutionSpec
    envelope: GrowthRiskEnvelope
    initial_equity: float
    smoothing_halflife_days: float
    required_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PROCEDURE_SCHEMA_VERSION:
            raise DataIntegrityError(f"unsupported procedure schema version {self.schema_version}")
        if not isinstance(self.code_digest, str) or not self.code_digest:
            raise DataIntegrityError("code_digest must be a nonempty identity")
        if not isinstance(self.data_policy, str) or not self.data_policy:
            raise DataIntegrityError("data_policy must be a nonempty identity")
        if self.universe_partition not in ("dev", "holdout", "all"):
            raise DataIntegrityError(f"invalid universe partition {self.universe_partition!r}")
        if not self.member_ids or any(not isinstance(m, str) or not m for m in self.member_ids):
            raise DataIntegrityError("member_ids must declare at least one nonempty member")
        if not isinstance(self.clock, ProcessClockSpec):
            raise DataIntegrityError("clock must be a ProcessClockSpec")
        if not isinstance(self.selection, NestedSelectionSpec):
            raise DataIntegrityError("selection must be a NestedSelectionSpec")
        if not isinstance(self.inference, ValidationInferenceSpec):
            raise DataIntegrityError("inference must be a ValidationInferenceSpec")
        if not isinstance(self.execution_policy, ProcessExecutionPolicy):
            raise DataIntegrityError("execution_policy must be a ProcessExecutionPolicy")
        if self.risk_sizing is not None and not isinstance(self.risk_sizing, ProcessRiskSizingSpec):
            raise DataIntegrityError("risk_sizing must be a ProcessRiskSizingSpec or None")
        if self.sizing_source != "hourly_proxy":
            raise DataIntegrityError(f"unsupported sizing source {self.sizing_source!r}")
        if self.member_evidence_source not in ("daily_step_proxy", "inventory_3m"):
            raise DataIntegrityError(f"unsupported member evidence source {self.member_evidence_source!r}")
        if not isinstance(self.execution_spec, ExecutionSpec):
            raise DataIntegrityError("execution_spec must be an ExecutionSpec")
        if not isinstance(self.envelope, GrowthRiskEnvelope):
            raise DataIntegrityError("envelope must be a GrowthRiskEnvelope")
        if isinstance(self.initial_equity, bool) or not isinstance(self.initial_equity, (int, float)):
            raise DataIntegrityError("initial_equity must be a finite positive amount")
        if not math.isfinite(float(self.initial_equity)) or float(self.initial_equity) <= 0.0:
            raise DataIntegrityError("initial_equity must be a finite positive amount")
        if isinstance(self.smoothing_halflife_days, bool) or not isinstance(self.smoothing_halflife_days, (int, float)):
            raise DataIntegrityError("smoothing_halflife_days must be finite and positive")
        if not math.isfinite(float(self.smoothing_halflife_days)) or float(self.smoothing_halflife_days) <= 0.0:
            raise DataIntegrityError("smoothing_halflife_days must be finite and positive")
        if not self.required_checks or any(not isinstance(c, str) or not c for c in self.required_checks):
            raise DataIntegrityError("required_checks must declare at least one named requirement")


@dataclass(frozen=True, slots=True)
class ProcessEvaluationPlan:
    """Request a specific registered or historical procedure evaluation without
    creating registration implicitly as a side effect of reading market data."""

    role: Literal["historical", "forward"]
    procedure: ProcessProcedureDefinition
    procedure_digest: str
    family_id: str
    look_endpoints: tuple[pd.Timestamp, ...]
    registration_digest: str | None
    judging_start: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.role not in ("historical", "forward"):
            raise DataIntegrityError(f"invalid evaluation role {self.role!r}")
        if not isinstance(self.procedure, ProcessProcedureDefinition):
            raise DataIntegrityError("procedure must be a ProcessProcedureDefinition")
        if not isinstance(self.procedure_digest, str) or _HEX64.fullmatch(self.procedure_digest) is None:
            raise DataIntegrityError("procedure_digest must be a 64-character lowercase hex identity")
        if not isinstance(self.family_id, str) or not self.family_id:
            raise DataIntegrityError("family_id must be a nonempty identity")
        if not self.look_endpoints:
            raise DataIntegrityError("look_endpoints must declare at least one endpoint")
        if any(not isinstance(e, pd.Timestamp) or e.tzinfo is None for e in self.look_endpoints):
            raise DataIntegrityError("look endpoints must be timezone-aware timestamps")
        if any(later <= earlier for earlier, later in zip(self.look_endpoints, self.look_endpoints[1:], strict=False)):
            raise DataIntegrityError("look endpoints must be strictly increasing")
        if self.registration_digest is not None and (
            not isinstance(self.registration_digest, str) or _HEX64.fullmatch(self.registration_digest) is None
        ):
            raise DataIntegrityError("registration_digest must be None or a 64-character lowercase hex identity")
        if self.judging_start is not None and (
            not isinstance(self.judging_start, pd.Timestamp) or self.judging_start.tzinfo is None
        ):
            raise DataIntegrityError("judging_start must be a timezone-aware timestamp or None")
        if self.judging_start is not None and any(
            not endpoint > self.judging_start for endpoint in self.look_endpoints
        ):
            raise DataIntegrityError("look endpoints must follow the judging start")
        if self.role == "forward" and self.judging_start is None:
            raise DataIntegrityError("forward evaluations require an explicit judging start")


@dataclass(frozen=True, slots=True)
class ResearchAttempt:
    """Reserved evidence request with its persisted pre-read consultation snapshot."""

    attempt_id: str
    context: EvaluationContext
    reserved_at: pd.Timestamp
    pre_read_consulted_through: pd.Timestamp
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise DataIntegrityError("attempt_id must be a nonempty identity")
        if not isinstance(self.context, EvaluationContext):
            raise DataIntegrityError("context must be an EvaluationContext")
        if self.reserved_at.tzinfo is None or self.pre_read_consulted_through.tzinfo is None:
            raise DataIntegrityError("reservation clocks must be timezone-aware")
        if self.requested_start.tzinfo is None or self.requested_end.tzinfo is None:
            raise DataIntegrityError("requested bounds must be timezone-aware")
        if not self.requested_start < self.requested_end:
            raise DataIntegrityError("requested interval must be non-empty")


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataIntegrityError(f"canonical values must be finite, got {value!r}")
        return value
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise DataIntegrityError("canonical timestamps must be timezone-aware")
        return {"__timestamp__": value.tz_convert("UTC").isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"__timedelta_ns__": value.value}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if dataclasses.is_dataclass(value):
        encoded: dict[str, Any] = {"__type__": type(value).__name__}
        for field in dataclasses.fields(value):
            encoded[field.name] = _canonical(getattr(value, field.name))
        return {key: encoded[key] for key in sorted(encoded)}
    raise DataIntegrityError(f"values of type {type(value).__name__} are not canonical JSON values")


def _require_typed_dict(data: Any, typename: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("__type__") != typename or set(data) != set(keys) | {"__type__"}:
        raise DataIntegrityError(f"invalid {typename} encoding")
    return data


def _as_float(node: Any, label: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise DataIntegrityError(f"{label} must be a number")
    value = float(node)
    if not math.isfinite(value):
        raise DataIntegrityError(f"{label} must be finite")
    return value


def _as_int(node: Any, label: str) -> int:
    if isinstance(node, bool) or not isinstance(node, int):
        raise DataIntegrityError(f"{label} must be an integer")
    return node


def _as_bool(node: Any, label: str) -> bool:
    if not isinstance(node, bool):
        raise DataIntegrityError(f"{label} must be a boolean")
    return node


def _as_timestamp(node: Any, label: str) -> pd.Timestamp:
    if not isinstance(node, dict) or set(node) != {"__timestamp__"} or not isinstance(node["__timestamp__"], str):
        raise DataIntegrityError(f"{label} must be an encoded timestamp")
    try:
        parsed = pd.Timestamp(node["__timestamp__"])
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"{label} must be an encoded timestamp") from exc
    if parsed.tzinfo is None:
        raise DataIntegrityError(f"{label} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _as_timedelta(node: Any, label: str) -> pd.Timedelta:
    if (
        not isinstance(node, dict)
        or set(node) != {"__timedelta_ns__"}
        or isinstance(node["__timedelta_ns__"], bool)
        or not isinstance(node["__timedelta_ns__"], int)
    ):
        raise DataIntegrityError(f"{label} must be an encoded duration")
    return pd.Timedelta(node["__timedelta_ns__"])


def _as_str_tuple(node: Any, label: str) -> tuple[str, ...]:
    if not isinstance(node, list) or any(not isinstance(item, str) for item in node):
        raise DataIntegrityError(f"{label} must be a list of strings")
    return tuple(node)


def _as_timestamp_tuple(node: Any, label: str) -> tuple[pd.Timestamp, ...]:
    if not isinstance(node, list):
        raise DataIntegrityError(f"{label} must be a list of timestamps")
    return tuple(_as_timestamp(item, label) for item in node)


def _decode_clock(data: Any) -> ProcessClockSpec:
    node = _require_typed_dict(data, "ProcessClockSpec", ("bar_completion_lag", "decision_period", "fit_latency"))
    try:
        return ProcessClockSpec(
            decision_period=_as_timedelta(node["decision_period"], "decision_period"),
            bar_completion_lag=_as_timedelta(node["bar_completion_lag"], "bar_completion_lag"),
            fit_latency=_as_timedelta(node["fit_latency"], "fit_latency"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid ProcessClockSpec: {exc}") from exc


def _decode_training_window(data: Any) -> TrainingWindowSpec:
    node = _require_typed_dict(data, "TrainingWindowSpec", ("kind", "months", "policy_id"))
    try:
        months = node["months"]
        return TrainingWindowSpec(
            policy_id=node["policy_id"],
            kind=node["kind"],
            months=None if months is None else _as_int(months, "months"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid TrainingWindowSpec: {exc}") from exc


def _decode_selection(data: Any) -> NestedSelectionSpec:
    node = _require_typed_dict(
        data,
        "NestedSelectionSpec",
        ("alpha", "bootstrap_paths", "control_policy_id", "fit_latency", "minimum_inner_labels", "policies", "seed"),
    )
    try:
        if not isinstance(node["policies"], list):
            raise DataIntegrityError("policies must be a list")
        return NestedSelectionSpec(
            policies=tuple(_decode_training_window(item) for item in node["policies"]),
            control_policy_id=node["control_policy_id"],
            minimum_inner_labels=_as_int(node["minimum_inner_labels"], "minimum_inner_labels"),
            alpha=_as_float(node["alpha"], "alpha"),
            bootstrap_paths=_as_int(node["bootstrap_paths"], "bootstrap_paths"),
            seed=_as_int(node["seed"], "seed"),
            fit_latency=_as_timedelta(node["fit_latency"], "fit_latency"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid NestedSelectionSpec: {exc}") from exc


def _decode_inference(data: Any) -> ValidationInferenceSpec:
    node = _require_typed_dict(
        data,
        "ValidationInferenceSpec",
        (
            "bootstrap_paths",
            "endpoint_budget",
            "family_alpha",
            "look_budget",
            "minimum_block_days",
            "procedure_budget",
            "resample_batch_paths",
            "seed",
        ),
    )
    try:
        return ValidationInferenceSpec(
            family_alpha=_as_float(node["family_alpha"], "family_alpha"),
            procedure_budget=_as_int(node["procedure_budget"], "procedure_budget"),
            look_budget=_as_int(node["look_budget"], "look_budget"),
            endpoint_budget=_as_int(node["endpoint_budget"], "endpoint_budget"),
            bootstrap_paths=_as_int(node["bootstrap_paths"], "bootstrap_paths"),
            seed=_as_int(node["seed"], "seed"),
            resample_batch_paths=_as_int(node["resample_batch_paths"], "resample_batch_paths"),
            minimum_block_days=_as_int(node["minimum_block_days"], "minimum_block_days"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid ValidationInferenceSpec: {exc}") from exc


def _decode_execution_policy(data: Any) -> ProcessExecutionPolicy:
    node = _require_typed_dict(data, "ProcessExecutionPolicy", ("tracking_error_threshold",))
    try:
        threshold = node["tracking_error_threshold"]
        return ProcessExecutionPolicy(
            tracking_error_threshold=None if threshold is None else _as_float(threshold, "tracking_error_threshold")
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid ProcessExecutionPolicy: {exc}") from exc


def _decode_sizing(data: Any) -> ProcessRiskSizingSpec | None:
    if data is None:
        return None
    node = _require_typed_dict(
        data, "ProcessRiskSizingSpec", ("annual_volatility_target", "ewma_halflife_days", "leverage_cap", "minimum_observations")
    )
    try:
        return ProcessRiskSizingSpec(
            annual_volatility_target=_as_float(node["annual_volatility_target"], "annual_volatility_target"),
            ewma_halflife_days=_as_int(node["ewma_halflife_days"], "ewma_halflife_days"),
            minimum_observations=_as_int(node["minimum_observations"], "minimum_observations"),
            leverage_cap=_as_float(node["leverage_cap"], "leverage_cap"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid ProcessRiskSizingSpec: {exc}") from exc


def _decode_execution_spec(data: Any) -> ExecutionSpec:
    node = _require_typed_dict(
        data,
        "ExecutionSpec",
        (
            "decision_anchor",
            "ladder_tranches",
            "liquidity_cost_model",
            "maker_fee_bps",
            "min_notional_probe_usdt",
            "name_drift_trim_interval_hours",
            "name_drift_trim_max_weight",
            "passive_timeout_minutes",
            "peg_chase_band_bps",
            "peg_chase_tranches",
            "peg_passive_fraction",
            "reference_equity_usdt",
            "require_trade_through",
            "spread_ewma_alpha",
            "taker_fee_bps",
            "taker_slippage_bps",
        ),
    )
    try:
        trim = node["name_drift_trim_max_weight"]
        return ExecutionSpec(
            maker_fee_bps=_as_float(node["maker_fee_bps"], "maker_fee_bps"),
            taker_fee_bps=_as_float(node["taker_fee_bps"], "taker_fee_bps"),
            taker_slippage_bps=_as_float(node["taker_slippage_bps"], "taker_slippage_bps"),
            passive_timeout_minutes=_as_int(node["passive_timeout_minutes"], "passive_timeout_minutes"),
            require_trade_through=_as_bool(node["require_trade_through"], "require_trade_through"),
            ladder_tranches=_as_int(node["ladder_tranches"], "ladder_tranches"),
            decision_anchor=node["decision_anchor"],
            peg_passive_fraction=_as_float(node["peg_passive_fraction"], "peg_passive_fraction"),
            peg_chase_band_bps=_as_float(node["peg_chase_band_bps"], "peg_chase_band_bps"),
            peg_chase_tranches=_as_int(node["peg_chase_tranches"], "peg_chase_tranches"),
            liquidity_cost_model=node["liquidity_cost_model"],
            spread_ewma_alpha=_as_float(node["spread_ewma_alpha"], "spread_ewma_alpha"),
            min_notional_probe_usdt=_as_float(node["min_notional_probe_usdt"], "min_notional_probe_usdt"),
            reference_equity_usdt=_as_float(node["reference_equity_usdt"], "reference_equity_usdt"),
            name_drift_trim_max_weight=None if trim is None else _as_float(trim, "name_drift_trim_max_weight"),
            name_drift_trim_interval_hours=_as_int(node["name_drift_trim_interval_hours"], "name_drift_trim_interval_hours"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid ExecutionSpec: {exc}") from exc


def _decode_envelope(data: Any) -> GrowthRiskEnvelope:
    node = _require_typed_dict(
        data,
        "GrowthRiskEnvelope",
        ("horizon_years", "leverage_ceiling", "max_drawdown", "max_drawdown_prob", "max_ruin_prob", "name", "ruin_fraction"),
    )
    try:
        return GrowthRiskEnvelope(
            name=node["name"],
            max_drawdown=_as_float(node["max_drawdown"], "max_drawdown"),
            max_drawdown_prob=_as_float(node["max_drawdown_prob"], "max_drawdown_prob"),
            ruin_fraction=_as_float(node["ruin_fraction"], "ruin_fraction"),
            max_ruin_prob=_as_float(node["max_ruin_prob"], "max_ruin_prob"),
            horizon_years=_as_float(node["horizon_years"], "horizon_years"),
            leverage_ceiling=_as_float(node["leverage_ceiling"], "leverage_ceiling"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid GrowthRiskEnvelope: {exc}") from exc


def procedure_to_json(procedure: ProcessProcedureDefinition) -> dict[str, JsonValue]:
    """Encode a frozen procedure definition as canonical JSON-safe values."""
    return cast('dict[str, JsonValue]', _canonical(procedure))


def procedure_from_json(data: Any) -> ProcessProcedureDefinition:
    """Decode a frozen procedure definition; reject unsupported or unknown fields."""
    node = _require_typed_dict(
        data,
        "ProcessProcedureDefinition",
        (
            "clock",
            "code_digest",
            "data_policy",
            "envelope",
            "execution_policy",
            "execution_spec",
            "inference",
            "initial_equity",
            "member_evidence_source",
            "member_ids",
            "required_checks",
            "risk_sizing",
            "schema_version",
            "selection",
            "sizing_source",
            "smoothing_halflife_days",
            "universe_partition",
        ),
    )
    try:
        return ProcessProcedureDefinition(
            schema_version=_as_int(node["schema_version"], "schema_version"),
            code_digest=node["code_digest"],
            data_policy=node["data_policy"],
            universe_partition=node["universe_partition"],
            member_ids=_as_str_tuple(node["member_ids"], "member_ids"),
            clock=_decode_clock(node["clock"]),
            selection=_decode_selection(node["selection"]),
            inference=_decode_inference(node["inference"]),
            execution_policy=_decode_execution_policy(node["execution_policy"]),
            risk_sizing=_decode_sizing(node["risk_sizing"]),
            sizing_source=node["sizing_source"],
            member_evidence_source=node["member_evidence_source"],
            execution_spec=_decode_execution_spec(node["execution_spec"]),
            envelope=_decode_envelope(node["envelope"]),
            initial_equity=_as_float(node["initial_equity"], "initial_equity"),
            smoothing_halflife_days=_as_float(node["smoothing_halflife_days"], "smoothing_halflife_days"),
            required_checks=_as_str_tuple(node["required_checks"], "required_checks"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid ProcessProcedureDefinition: {exc}") from exc


def process_procedure_digest(procedure: ProcessProcedureDefinition) -> str:
    """Return SHA-256 identity of all canonical procedure values, preserving typed meaning."""
    raw = json.dumps(_canonical(procedure), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def plan_to_json(plan: ProcessEvaluationPlan) -> dict[str, JsonValue]:
    """Encode a frozen evaluation plan as canonical JSON-safe values."""
    return cast('dict[str, JsonValue]', _canonical(plan))


def plan_from_json(data: Any) -> ProcessEvaluationPlan:
    """Load strict versioned controls; reject unsupported or identity-conflicting plans."""
    node = _require_typed_dict(
        data,
        "ProcessEvaluationPlan",
        ("family_id", "judging_start", "look_endpoints", "procedure", "procedure_digest", "registration_digest", "role"),
    )
    try:
        judging = node["judging_start"]
        registration = node["registration_digest"]
        return ProcessEvaluationPlan(
            role=node["role"],
            procedure=procedure_from_json(node["procedure"]),
            procedure_digest=node["procedure_digest"],
            family_id=node["family_id"],
            look_endpoints=_as_timestamp_tuple(node["look_endpoints"], "look_endpoints"),
            registration_digest=None if registration is None else str(registration),
            judging_start=None if judging is None else _as_timestamp(judging, "judging_start"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid ProcessEvaluationPlan: {exc}") from exc


def context_to_json(context: EvaluationContext) -> dict[str, JsonValue]:
    """Encode a reserved evaluation context as canonical JSON-safe values."""
    return cast('dict[str, JsonValue]', _canonical(context))


def context_from_json(data: Any) -> EvaluationContext:
    """Decode a persisted reservation context; reject unsupported encodings."""
    node = _require_typed_dict(
        data,
        "EvaluationContext",
        (
            "code_digest",
            "consulted_through",
            "family_id",
            "inference_spec",
            "input_manifest_digest",
            "interval_end",
            "interval_start",
            "journal_complete",
            "look_ordinal",
            "observed_through",
            "procedure_digest",
            "registered_at",
            "role",
        ),
    )
    try:
        manifest = node["input_manifest_digest"]
        registered = node["registered_at"]
        consulted = node["consulted_through"]
        family = node["family_id"]
        look = node["look_ordinal"]
        inference = node["inference_spec"]
        return EvaluationContext(
            role=node["role"],
            procedure_digest=node["procedure_digest"],
            code_digest=node["code_digest"],
            input_manifest_digest=None if manifest is None else str(manifest),
            interval_start=_as_timestamp(node["interval_start"], "interval_start"),
            interval_end=_as_timestamp(node["interval_end"], "interval_end"),
            registered_at=None if registered is None else _as_timestamp(registered, "registered_at"),
            consulted_through=None if consulted is None else _as_timestamp(consulted, "consulted_through"),
            family_id=None if family is None else str(family),
            look_ordinal=None if look is None else _as_int(look, "look_ordinal"),
            inference_spec=None if inference is None else _decode_inference(inference),
            journal_complete=_as_bool(node["journal_complete"], "journal_complete"),
            observed_through=_as_timestamp(node["observed_through"], "observed_through"),
        )
    except (DataIntegrityError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"invalid EvaluationContext: {exc}") from exc


def _endpoint_schedule(text: str) -> tuple[pd.Timestamp, ...]:
    try:
        items = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"look schedule is corrupt: {exc}") from exc
    if not isinstance(items, list):
        raise DataIntegrityError("look schedule is corrupt")
    return tuple(_as_timestamp(item, "look endpoint") for item in items)


def _require_utc(value: pd.Timestamp, label: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
        raise DataIntegrityError(f"{label} must be a timezone-aware timestamp")
    return value.tz_convert("UTC")


def _parse_ts(text: str, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"{label} is corrupt: {text!r}") from exc
    if parsed.tzinfo is None:
        raise DataIntegrityError(f"{label} is corrupt: {text!r}")
    return parsed.tz_convert("UTC")


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path), timeout=10.0)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise DataIntegrityError(f"research journal is missing: {path}")


def _require_journal(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.Error as exc:
        raise DataIntegrityError(f"research journal is corrupt: {exc}") from exc
    if row is None or str(row[0]) != str(JOURNAL_SCHEMA_VERSION):
        raise DataIntegrityError("research journal is incompatible or corrupt")


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _locked_horizon(conn: sqlite3.Connection) -> pd.Timestamp:
    latest = _parse_ts(str(_meta_get(conn, "consulted_through")), "consulted horizon")
    ends = [latest]
    ends.extend(_parse_ts(str(row[0]), "consultation") for row in conn.execute("SELECT end FROM consultations"))
    ends.extend(_parse_ts(str(row[0]), "attempt") for row in conn.execute("SELECT requested_end FROM attempts"))
    return max(ends)


def initialize_research_journal(
    path: Path, *, now: pd.Timestamp, legacy_consulted_through: pd.Timestamp, legacy_history_complete: bool
) -> None:
    """Establish durable procedure, consultation and attempt history without inventing
    completeness for missing or retained legacy records.

    Args:
        path: Owned SQLite research journal, separate from detail retention.
        now: Trusted UTC journal initialization time.
        legacy_consulted_through: Conservative legacy consultation boundary.
        legacy_history_complete: Result of supported historical migration checks.
    Returns:
        None; creates or validates the journal schema transactionally.
    Raises:
        DataIntegrityError: Existing journal is incompatible, corrupt or contradictory.
    """
    now_utc = _require_utc(now, "now")
    legacy_utc = _require_utc(legacy_consulted_through, "legacy_consulted_through")
    if not isinstance(legacy_history_complete, bool):
        raise DataIntegrityError("legacy_history_complete must be a boolean")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    try:
        try:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.Error as exc:
            raise DataIntegrityError(f"research journal is corrupt: {exc}") from exc
        if row is None:
            horizon = legacy_utc
            if not legacy_history_complete:
                horizon = max(horizon, MHS_FINAL_OOS_CUTOFF_2026H1, now_utc)
            conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(JOURNAL_SCHEMA_VERSION),))
            conn.execute("INSERT INTO meta(key, value) VALUES ('consulted_through', ?)", (horizon.isoformat(),))
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('journal_complete', ?)", ("1" if legacy_history_complete else "0",)
            )
        else:
            if str(row[0]) != str(JOURNAL_SCHEMA_VERSION):
                raise DataIntegrityError("research journal schema is incompatible")
            stored_raw = _meta_get(conn, "consulted_through")
            if stored_raw is None:
                raise DataIntegrityError("research journal is corrupt")
            stored = _parse_ts(stored_raw, "consulted horizon")
            advanced = stored if stored >= legacy_utc else legacy_utc
            if advanced != stored:
                conn.execute("UPDATE meta SET value=? WHERE key='consulted_through'", (advanced.isoformat(),))
        conn.commit()
    finally:
        conn.close()


def consulted_process_horizon(path: Path) -> pd.Timestamp:
    """Return the conservative latest consulted or attempted input horizon without resetting after pruning."""
    _require_file(path)
    conn = _connect(path)
    try:
        _require_journal(conn)
        return _locked_horizon(conn)
    finally:
        conn.close()


def record_research_consultation(
    path: Path,
    *,
    procedure_digest: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    now: pd.Timestamp,
    source: Literal["cache", "legacy", "external"],
    result_digest: str | None = None,
) -> None:
    """Append a conservative consultation independently of execution/cache reuse; absent
    procedure identity cannot establish forward independence."""
    start_utc = _require_utc(start, "start")
    end_utc = _require_utc(end, "end")
    now_utc = _require_utc(now, "now")
    if not end_utc > start_utc:
        raise DataIntegrityError("consulted interval must be non-empty")
    if source not in ("cache", "legacy", "external"):
        raise DataIntegrityError(f"invalid consultation source {source!r}")
    if procedure_digest is not None and (not isinstance(procedure_digest, str) or not procedure_digest):
        raise DataIntegrityError("procedure_digest must be a nonempty identity or None")
    if result_digest is not None and (not isinstance(result_digest, str) or not result_digest):
        raise DataIntegrityError("result_digest must be a nonempty identity or None")
    _require_file(path)
    conn = _connect(path)
    try:
        _require_journal(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO consultations(procedure_digest, start, end, recorded_at, source, result_digest)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (procedure_digest, start_utc.isoformat(), end_utc.isoformat(), now_utc.isoformat(), source, result_digest),
        )
        stored = _parse_ts(str(_meta_get(conn, "consulted_through")), "consulted horizon")
        advanced = stored if stored >= end_utc else end_utc
        if advanced != stored:
            conn.execute("UPDATE meta SET value=? WHERE key='consulted_through'", (advanced.isoformat(),))
        conn.commit()
    finally:
        conn.close()


def persist_process_registration(path: Path, plan: ProcessEvaluationPlan, *, now: pd.Timestamp) -> ProcessEvaluationPlan:
    """Persist one frozen process registration atomically before any look.

    Stores the same typed definition whose digest is checked by
    reserve_research_attempt, so a later evaluation cannot reuse the family
    slot under enlarged budgets or a rewritten procedure.
    """
    now_utc = _require_utc(now, "now")
    if process_procedure_digest(plan.procedure) != plan.procedure_digest:
        raise DataIntegrityError("procedure digest does not match the frozen definition")
    if plan.registration_digest is not None:
        raise DataIntegrityError("plan is already registered")
    _require_file(path)
    conn = _connect(path)
    try:
        _require_journal(conn)
        conn.execute("BEGIN IMMEDIATE")
        endpoints_json = json.dumps(
            [_canonical(endpoint) for endpoint in plan.look_endpoints], separators=(",", ":")
        )
        family = conn.execute(
            "SELECT procedure_digest, procedure_budget, look_budget, endpoint_budget,"
            " look_endpoints, registration_digest FROM families WHERE family_id=?",
            (plan.family_id,),
        ).fetchone()
        wanted = (
            plan.procedure_digest,
            plan.procedure.inference.procedure_budget,
            plan.procedure.inference.look_budget,
            plan.procedure.inference.endpoint_budget,
            endpoints_json,
        )
        if family is not None:
            if (str(family[0]), int(family[1]), int(family[2]), int(family[3]), str(family[4])) != wanted:
                raise DataIntegrityError(f"family {plan.family_id} is already frozen under a different definition")
            conn.commit()
            return dataclasses.replace(plan, registration_digest=str(family[5]))
        payload = json.dumps(
            {
                "kind": "process-registration",
                "endpoint_budget": plan.procedure.inference.endpoint_budget,
                "family_id": plan.family_id,
                "judging_start": None if plan.judging_start is None else plan.judging_start.tz_convert("UTC").isoformat(),
                "look_budget": plan.procedure.inference.look_budget,
                "look_endpoints": json.loads(endpoints_json),
                "procedure_budget": plan.procedure.inference.procedure_budget,
                "procedure_digest": plan.procedure_digest,
                "registered_at": now_utc.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        registration_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        conn.execute("INSERT OR IGNORE INTO procedures(digest, canonical) VALUES (?, ?)", (plan.procedure_digest, payload))
        conn.execute(
            "INSERT INTO families(family_id, procedure_digest, procedure_budget, look_budget, endpoint_budget,"
            " look_endpoints, judging_start, registered_at, registration_digest)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan.family_id,
                plan.procedure_digest,
                plan.procedure.inference.procedure_budget,
                plan.procedure.inference.look_budget,
                plan.procedure.inference.endpoint_budget,
                endpoints_json,
                None if plan.judging_start is None else plan.judging_start.tz_convert("UTC").isoformat(),
                now_utc.isoformat(),
                registration_digest,
            ),
        )
        conn.execute(
            "INSERT INTO plans(registration_digest, family_id, role, procedure_digest, canonical, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                registration_digest,
                plan.family_id,
                plan.role,
                plan.procedure_digest,
                json.dumps(plan_to_json(dataclasses.replace(plan, registration_digest=registration_digest))),
                now_utc.isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return dataclasses.replace(plan, registration_digest=registration_digest)


def _forward_look_slot(
    conn: sqlite3.Connection, plan: ProcessEvaluationPlan, *, end_utc: pd.Timestamp, horizon: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp, int, pd.Timestamp]:
    family = conn.execute(
        "SELECT procedure_digest, procedure_budget, look_budget, endpoint_budget,"
        " look_endpoints, judging_start, registered_at, registration_digest"
        " FROM families WHERE family_id=?",
        (plan.family_id,),
    ).fetchone()
    if family is None:
        raise DataIntegrityError(f"family {plan.family_id} is not registered")
    endpoints_json = json.dumps(
        [_canonical(endpoint_ts) for endpoint_ts in plan.look_endpoints], separators=(",", ":")
    )
    if (
        str(family[0]),
        int(family[1]),
        int(family[2]),
        int(family[3]),
        str(family[4]),
        None if plan.registration_digest is None else str(plan.registration_digest),
    ) != (
        plan.procedure_digest,
        plan.procedure.inference.procedure_budget,
        plan.procedure.inference.look_budget,
        plan.procedure.inference.endpoint_budget,
        endpoints_json,
        str(family[7]),
    ):
        raise DataIntegrityError(f"family {plan.family_id} does not match the registered procedure")
    registered_at = _parse_ts(str(family[6]), "registration")
    used = conn.execute("SELECT COUNT(*) FROM attempts WHERE family_id=?", (plan.family_id,)).fetchone()
    look_ordinal = int(used[0]) + 1
    if look_ordinal > plan.procedure.inference.look_budget:
        raise DataIntegrityError("formal look budget is exhausted")
    schedule = _endpoint_schedule(str(family[4]))
    if look_ordinal > len(schedule):
        raise DataIntegrityError("formal look schedule is exhausted")
    endpoint = schedule[look_ordinal - 1]
    if not end_utc <= endpoint:
        raise DataIntegrityError("requested evidence exceeds the permitted look endpoint")
    assert plan.judging_start is not None
    if look_ordinal == 1:
        judging_start = plan.judging_start.tz_convert("UTC")
    else:
        judging_start = schedule[look_ordinal - 2].normalize() + pd.Timedelta(days=1)
    if not (registered_at < judging_start and horizon < judging_start):
        raise DataIntegrityError("judging interval is already consulted")
    return judging_start, endpoint, look_ordinal, registered_at


def reserve_research_attempt(
    path: Path, plan: ProcessEvaluationPlan, *, start: pd.Timestamp, end: pd.Timestamp, now: pd.Timestamp, attempt_id: str
) -> ResearchAttempt:
    """Persist consultation and a permitted formal look before opening market inputs.

    Args:
        path: Initialized durable research journal.
        plan: Exact immutable procedure, role, family and permitted look endpoints.
        start: Earliest requested market input, including training/warmup.
        end: Latest requested evidence input.
        now: Trusted UTC reservation clock.
        attempt_id: Unique execution attempt identity, including standalone workers.
    Returns:
        A transactionally reserved attempt and the prior consultation snapshot.
    Raises:
        DataIntegrityError: Registration, family, schedule, identities or journal integrity fail.
    """
    start_utc = _require_utc(start, "start")
    end_utc = _require_utc(end, "end")
    now_utc = _require_utc(now, "now")
    if not end_utc > start_utc:
        raise DataIntegrityError("requested interval must be non-empty")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise DataIntegrityError("attempt_id must be a nonempty identity")
    if process_procedure_digest(plan.procedure) != plan.procedure_digest:
        raise DataIntegrityError("procedure digest does not match the frozen definition")
    _require_file(path)
    conn = _connect(path)
    try:
        _require_journal(conn)
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone() is not None:
            raise DataIntegrityError(f"attempt {attempt_id} is already reserved")
        horizon = _locked_horizon(conn)
        registered_at: pd.Timestamp | None = None
        look_ordinal: int | None = None
        endpoint: pd.Timestamp | None = None
        if plan.role == "historical":
            judging_start = plan.judging_start if plan.judging_start is not None else start_utc
            if not judging_start < end_utc:
                raise DataIntegrityError("historical judging interval must be non-empty")
        else:
            judging_start, endpoint, look_ordinal, registered_at = _forward_look_slot(
                conn, plan, end_utc=end_utc, horizon=horizon
            )
        context = EvaluationContext(
            role=plan.role,
            procedure_digest=plan.procedure_digest,
            code_digest=plan.procedure.code_digest,
            input_manifest_digest=None,
            interval_start=judging_start,
            interval_end=end_utc,
            registered_at=None if registered_at is None else registered_at,
            consulted_through=horizon,
            family_id=plan.family_id,
            look_ordinal=look_ordinal,
            inference_spec=plan.procedure.inference,
            journal_complete=_meta_get(conn, "journal_complete") == "1",
            observed_through=now_utc,
        )
        conn.execute(
            "INSERT INTO consultations(procedure_digest, start, end, recorded_at, source, result_digest)"
            " VALUES (?, ?, ?, ?, 'attempt', NULL)",
            (plan.procedure_digest, start_utc.isoformat(), end_utc.isoformat(), now_utc.isoformat()),
        )
        conn.execute(
            "INSERT INTO attempts(attempt_id, family_id, role, procedure_digest, look_ordinal, endpoint,"
            " requested_start, requested_end, reserved_at, pre_read_consulted, context)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                plan.family_id,
                plan.role,
                plan.procedure_digest,
                look_ordinal,
                None if endpoint is None else endpoint.isoformat(),
                start_utc.isoformat(),
                end_utc.isoformat(),
                now_utc.isoformat(),
                horizon.isoformat(),
                json.dumps(context_to_json(context)),
            ),
        )
        if end_utc > horizon:
            conn.execute("UPDATE meta SET value=? WHERE key='consulted_through'", (end_utc.isoformat(),))
        conn.commit()
    finally:
        conn.close()
    return ResearchAttempt(
        attempt_id=attempt_id,
        context=context,
        reserved_at=now_utc,
        pre_read_consulted_through=horizon,
        requested_start=start_utc,
        requested_end=end_utc,
    )


def finish_research_attempt(
    path: Path,
    attempt: ResearchAttempt,
    *,
    status: Literal["completed", "failed", "interrupted"],
    result_digest: str | None,
    now: pd.Timestamp,
) -> None:
    """Append attempt outcome without refunding consulted data or spent formal looks.

    Args:
        path: Journal owning the reserved attempt.
        attempt: Previously persisted attempt identity and context.
        status: Actual terminal execution outcome.
        result_digest: Immutable result identity, absent if execution/persistence failed.
        now: Trusted UTC finalization time.
    Returns:
        None; preserves original reservation and appends one compatible outcome.
    Raises:
        DataIntegrityError: Identity or outcome conflicts with existing history.
    """
    if status not in ("completed", "failed", "interrupted"):
        raise DataIntegrityError(f"invalid attempt status {status!r}")
    now_utc = _require_utc(now, "now")
    if result_digest is not None and (not isinstance(result_digest, str) or not result_digest):
        raise DataIntegrityError("result_digest must be a nonempty identity or None")
    if status == "completed" and result_digest is None:
        raise DataIntegrityError("completed attempts must carry a result digest")
    _require_file(path)
    conn = _connect(path)
    try:
        _require_journal(conn)
        row = conn.execute(
            "SELECT family_id, role, procedure_digest, requested_start, requested_end, reserved_at"
            " FROM attempts WHERE attempt_id=?",
            (attempt.attempt_id,),
        ).fetchone()
        if row is None:
            raise DataIntegrityError(f"attempt {attempt.attempt_id} was never reserved")
        if tuple(row) != (
            attempt.context.family_id,
            attempt.context.role,
            attempt.context.procedure_digest,
            attempt.requested_start.isoformat(),
            attempt.requested_end.isoformat(),
            attempt.reserved_at.isoformat(),
        ):
            raise DataIntegrityError(f"attempt {attempt.attempt_id} does not match the reservation")
        existing = conn.execute("SELECT status, result_digest FROM outcomes WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
        if existing is not None:
            if (existing[0], existing[1]) != (status, result_digest):
                raise DataIntegrityError(f"conflicting outcome for attempt {attempt.attempt_id}")
            return
        conn.execute(
            "INSERT INTO outcomes(attempt_id, status, result_digest, finished_at) VALUES (?, ?, ?, ?)",
            (attempt.attempt_id, status, result_digest, now_utc.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_process_evaluation_plan(path: Path) -> ProcessEvaluationPlan:
    """Load strict versioned controls; reject unsupported or identity-conflicting plans."""
    _require_file(path)
    conn = _connect(path)
    try:
        _require_journal(conn)
        row = conn.execute("SELECT procedure_digest, canonical FROM plans ORDER BY rowid DESC LIMIT 1").fetchone()
        if row is None:
            raise DataIntegrityError("research journal holds no registered plan")
        plan = plan_from_json(json.loads(str(row[1])))
        if plan.procedure_digest != str(row[0]) or process_procedure_digest(plan.procedure) != plan.procedure_digest:
            raise DataIntegrityError("stored plan conflicts with its registered identity")
        return plan
    finally:
        conn.close()


def load_research_attempt(path: Path, attempt_id: str) -> ResearchAttempt:
    """Load the original persisted reservation without inventing an outcome or a fresh look."""
    if not isinstance(attempt_id, str) or not attempt_id:
        raise DataIntegrityError("attempt_id must be a nonempty identity")
    _require_file(path)
    conn = _connect(path)
    try:
        _require_journal(conn)
        row = conn.execute(
            "SELECT context, reserved_at, pre_read_consulted, requested_start, requested_end"
            " FROM attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise DataIntegrityError(f"attempt {attempt_id} was never reserved")
        return ResearchAttempt(
            attempt_id=attempt_id,
            context=context_from_json(json.loads(str(row[0]))),
            reserved_at=_parse_ts(str(row[1]), "reservation"),
            pre_read_consulted_through=_parse_ts(str(row[2]), "consultation"),
            requested_start=_parse_ts(str(row[3]), "requested start"),
            requested_end=_parse_ts(str(row[4]), "requested end"),
        )
    finally:
        conn.close()
