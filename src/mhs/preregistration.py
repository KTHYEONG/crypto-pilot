"""Pre-registered forward evaluation protocol; procedures are frozen before the data that judges them exists."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.backtest.journal import (
    ProcessEvaluationPlan,
    consulted_process_horizon,
    persist_process_registration,
    process_procedure_digest,
)
from src.mhs.params import DISCOVERY_START, MHS_FINAL_OOS_CUTOFF_2026H1
from src.mhs.run_history import (
    _DEFAULT_HISTORY_DIR,
    RESEARCH_NEUTRAL_FLAGS,
    _iter_history_records,
    _parse_utc_timestamp,
)
from src.quant.evaluation.policy import HOLDOUT_CUTOFF

PROCEDURE_REGISTRY_PATH: Path = Path("docs") / "decisions" / "mhs_procedure_registry.jsonl"
EVENT_REGISTRATION: str = "registration"
EVENT_EVALUATION: str = "evaluation"
# 실행 창·실행 제어 필드는 절차(알파 결정 경로)가 아니므로 digest에서 제외한다.
PROCEDURE_RUN_CONTROL_FIELDS: frozenset[str] = frozenset({
    "start", "end", "final_oos_2026h1", "input_manifest_path",
    "forward_execution_quality_dir", "forward_strategy_digest", "forward_registration_digest",
})


@dataclass(frozen=True, slots=True)
class ProcedureRegistration:
    """One frozen procedure; ``effective_start`` is the later of the freeze clock and the consulted data horizon."""

    procedure_digest: str
    frozen_at: pd.Timestamp
    data_horizon: pd.Timestamp
    procedure: dict[str, Any]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", self.procedure_digest) is None:
            raise ValueError("procedure_digest must be 32 lowercase hex characters")
        if self.frozen_at.tzinfo is None or self.data_horizon.tzinfo is None:
            raise ValueError("frozen_at and data_horizon must be tz-aware")

    @property
    def effective_start(self) -> pd.Timestamp:
        return max(self.frozen_at, self.data_horizon)


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def procedure_payload(request: Any) -> dict[str, Any]:
    """JSON-safe alpha-relevant payload of one request.

    Run-window and telemetry fields are excluded; the resolved committee gross
    and the sealed params snapshot are included so a changed decision constant
    changes the digest.
    """
    from src.mhs.live_strategy import capture_params_snapshot
    from src.mhs.research_go import _resolved_committee_target_gross

    payload = {
        f.name: getattr(request, f.name)
        for f in fields(request)
        if f.name not in RESEARCH_NEUTRAL_FLAGS and f.name not in PROCEDURE_RUN_CONTROL_FIELDS
    }
    payload["committee_target_gross"] = _resolved_committee_target_gross(request)
    payload["params_snapshot"] = capture_params_snapshot()
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return json.loads(raw)  # type: ignore[no-any-return]


def procedure_identity_digest(request: Any) -> str:
    """32-hex SHA-256 digest of :func:`procedure_payload`."""
    raw = json.dumps(procedure_payload(request), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _read_events(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for i, line in enumerate(registry_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict) or event.get("event") not in (EVENT_REGISTRATION, EVENT_EVALUATION):
            raise DataIntegrityError(f"procedure registry line {i} is not a known event")
        events.append(event)
    return events


def _append_event(registry_path: Path, event: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open(mode="a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


def load_registrations(registry_path: Path = PROCEDURE_REGISTRY_PATH) -> tuple[ProcedureRegistration, ...]:
    """Registration events in file order.

    Raises:
        DataIntegrityError: an unknown event or a malformed registration event.
    """
    registrations: list[ProcedureRegistration] = []
    for event in _read_events(registry_path):
        if event["event"] != EVENT_REGISTRATION:
            continue
        try:
            registrations.append(
                ProcedureRegistration(
                    procedure_digest=str(event["procedure_digest"]),
                    frozen_at=_utc(event["frozen_at"]),
                    data_horizon=_utc(event["data_horizon"]),
                    procedure=dict(event["procedure"]),
                )
            )
        except KeyError as exc:
            raise DataIntegrityError(f"malformed registration event: {exc}") from exc
    return tuple(registrations)


def find_registration(digest: str, registry_path: Path = PROCEDURE_REGISTRY_PATH) -> ProcedureRegistration:
    """Registration for ``digest``.

    Raises:
        DataIntegrityError: the digest is not registered.
    """
    for registration in load_registrations(registry_path):
        if registration.procedure_digest == digest:
            return registration
    raise DataIntegrityError(f"procedure {digest} is not registered")


def consulted_data_horizon(
    history_dir: Path = _DEFAULT_HISTORY_DIR, registry_path: Path = PROCEDURE_REGISTRY_PATH
) -> pd.Timestamp:
    """Latest data timestamp any research look may have consulted.

    The sealed unseal ceilings always count as consulted because pruned
    run-history shards cannot prove otherwise; forward evaluation events in the
    registry advance the horizon.
    """
    # 봉인 해제 상한은 이력 샤드가 가지치기돼도 조회된 것으로 간주한다.
    ends = [HOLDOUT_CUTOFF, MHS_FINAL_OOS_CUTOFF_2026H1]
    if history_dir.exists():
        for record in _iter_history_records(history_dir):
            ts = _parse_utc_timestamp(record.get("resolved_end") or record.get("end"))
            if ts is not None:
                ends.append(ts)
    ends.extend(
        _utc(event["resolved_end"])
        for event in _read_events(registry_path)
        if event["event"] == EVENT_EVALUATION
    )
    return max(ends)


def register_procedure(
    request: Any,
    *,
    now: pd.Timestamp,
    registry_path: Path = PROCEDURE_REGISTRY_PATH,
    history_dir: Path = _DEFAULT_HISTORY_DIR,
) -> ProcedureRegistration:
    """Freeze one procedure before the data that will judge it exists.

    Raises:
        ValueError: ``now`` is naive or the request already references a registration.
        DataIntegrityError: the procedure is already registered or ``now`` does not
            follow the consulted data horizon.
    """
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    if getattr(request, "forward_registration_digest", None) is not None:
        raise ValueError("a registration request must not reference an existing registration")
    digest = procedure_identity_digest(request)
    if any(r.procedure_digest == digest for r in load_registrations(registry_path)):
        raise DataIntegrityError(f"procedure {digest} is already registered")
    horizon = consulted_data_horizon(history_dir, registry_path)
    now_utc = now.tz_convert("UTC")
    if now_utc <= horizon:
        raise DataIntegrityError("registration clock precedes consulted data horizon")
    registration = ProcedureRegistration(digest, now_utc, horizon, procedure_payload(request))
    _append_event(registry_path, {"event": EVENT_REGISTRATION, "procedure_digest": digest,
            "frozen_at": now_utc.isoformat(), "data_horizon": horizon.isoformat(),
            "procedure": registration.procedure})
    return registration


def register_process_procedure(
    plan: ProcessEvaluationPlan,
    *,
    now: pd.Timestamp,
    journal_path: Path,
    legacy_history_dir: Path = _DEFAULT_HISTORY_DIR,
    legacy_registry_path: Path = PROCEDURE_REGISTRY_PATH,
) -> ProcessEvaluationPlan:
    """Register a complete process procedure and permitted looks before judging data exists.

    Args:
        plan: Strict process definition, fixed family budget and future look schedule.
        now: Trusted UTC registration time.
        journal_path: Durable process research journal.
        legacy_history_dir: Existing consulted research history, when retained.
        legacy_registry_path: Existing forward registration/evaluation history.
    Returns:
        An immutable plan referencing the persisted process registration identity.
    Raises:
        ValueError: ``now`` is naive.
        DataIntegrityError: Procedure/family identity, consultation history or future schedule conflicts.
    """
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    if process_procedure_digest(plan.procedure) != plan.procedure_digest:
        raise DataIntegrityError("procedure digest does not match the frozen definition")
    now_utc = now.tz_convert("UTC")
    floor = consulted_data_horizon(legacy_history_dir, legacy_registry_path)
    journal_floor = consulted_process_horizon(journal_path)
    if journal_floor > floor:
        floor = journal_floor
    if now_utc <= floor:
        raise DataIntegrityError("registration clock precedes consulted data horizon")
    if plan.role == "forward" and (
        plan.judging_start is None or not plan.judging_start.tz_convert("UTC") > now_utc
    ):
        raise DataIntegrityError("forward judging schedule must follow registration")
    return persist_process_registration(journal_path, plan, now=now_utc)


def record_forward_evaluation(
    registration: ProcedureRegistration,
    resolved_end: pd.Timestamp,
    *,
    now: pd.Timestamp,
    registry_path: Path = PROCEDURE_REGISTRY_PATH,
) -> None:
    """Append one evaluation event; every forward look advances later registrations' data horizon.

    Raises:
        ValueError: ``now`` or ``resolved_end`` is naive.
    """
    if now.tzinfo is None or resolved_end.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    _append_event(registry_path, {"event": EVENT_EVALUATION,
            "procedure_digest": registration.procedure_digest,
            "resolved_end": resolved_end.tz_convert("UTC").isoformat(),
            "at": now.tz_convert("UTC").isoformat()})


def is_quarter_end_date(value: Any) -> bool:
    """True when ``value`` is midnight UTC on a calendar quarter-end date."""
    ts = _utc(value)
    if ts != ts.normalize():
        return False
    nxt = ts + pd.Timedelta(days=1)
    return nxt.day == 1 and nxt.month in (1, 4, 7, 10)


def forward_evaluation_end_ceiling(now: pd.Timestamp) -> pd.Timestamp:
    """Last quarter end strictly completed before ``now`` (23:59:59 UTC).

    Raises:
        ValueError: ``now`` is naive.
        DataIntegrityError: no quarter has completed before ``now``.
    """
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    now_utc = now.tz_convert("UTC")
    quarter_ends = pd.date_range(start=DISCOVERY_START.normalize(), end=now_utc.normalize(), freq="QE-DEC")
    complete = [q for q in quarter_ends if q + pd.Timedelta(days=1) <= now_utc]
    if not complete:
        raise DataIntegrityError("no quarter completed before now")
    return complete[-1] + pd.Timedelta(hours=23, minutes=59, seconds=59)
