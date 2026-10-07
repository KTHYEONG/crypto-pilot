"""Pre-registered forward evaluation protocol; procedures are frozen before the data that judges them exists."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import BACKTESTS_DIR, BASE_DIR
from src.mhs.backtest.journal import (
    ProcessEvaluationPlan,
    consulted_process_horizon,
    persist_process_registration,
    process_procedure_digest,
)
from src.mhs.params import DISCOVERY_START, MHS_FINAL_OOS_CUTOFF_2026H1
from src.mhs.run_history import (
    RESEARCH_NEUTRAL_FLAGS,
    _resolve_history_registry,
    consulted_registry_horizon,
)
from src.quant.evaluation.policy import HOLDOUT_CUTOFF

logger = logging.getLogger("MhsPreregistration")

# Runtime evidence, never documentation: the registry lives beside the run-history
# registry under the gitignored backtests root so it can never be tracked.
PROCEDURE_REGISTRY_PATH: Path = BACKTESTS_DIR / "procedure_registry.jsonl"
"""Operator procedure registry (``BACKTESTS_DIR / "procedure_registry.jsonl"``): CLI write default and read default."""
LEGACY_PROCEDURE_REGISTRY_PATH: Path = BASE_DIR / "docs" / "decisions" / "mhs_procedure_registry.jsonl"
"""Pre-2026-10 location under the documentation tree; only ``migrate_legacy_procedure_registry`` writes or deletes it."""
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
    changes the digest. Inert dependent fields are recorded at their canonical value so one decision path has one digest.
    """
    from src.mhs.live_strategy import capture_params_snapshot
    from src.mhs.validation import inert_dependent_overrides

    payload = {
        f.name: getattr(request, f.name)
        for f in fields(request)
        if f.name not in RESEARCH_NEUTRAL_FLAGS and f.name not in PROCEDURE_RUN_CONTROL_FIELDS
    }
    payload["params_snapshot"] = capture_params_snapshot()
    payload.update(inert_dependent_overrides(request))
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return json.loads(raw)  # type: ignore[no-any-return]


def procedure_identity_digest(request: Any) -> str:
    """32-hex SHA-256 digest of :func:`procedure_payload`."""
    raw = json.dumps(procedure_payload(request), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _reject_unmigrated_legacy(registry_path: Path) -> None:
    """Fail closed when the canonical registry is read while legacy evidence still exists."""
    if registry_path.resolve() == PROCEDURE_REGISTRY_PATH.resolve() and LEGACY_PROCEDURE_REGISTRY_PATH.is_file():
        raise DataIntegrityError(
            f"procedure registry is unmigrated: legacy {LEGACY_PROCEDURE_REGISTRY_PATH} still exists; "
            f"target {PROCEDURE_REGISTRY_PATH} must not be read as absent evidence; "
            f"run ops procedure-registry-migrate --legacy-path {LEGACY_PROCEDURE_REGISTRY_PATH} "
            f"--target-path {PROCEDURE_REGISTRY_PATH}"
        )


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def migrate_legacy_procedure_registry(
    *,
    legacy_path: Path,
    target_path: Path,
) -> bool:
    """Move the pre-2026-10 procedure registry out of the documentation tree exactly once.

    The registry is append-only runtime evidence (every registration and forward evaluation advances later
    consulted horizons), so it must never be silently forked, duplicated or dropped. Migration is an explicit
    operator action (``ops procedure-registry-migrate``); no read or append triggers it.

    Args:
        legacy_path: Legacy registry file.
        target_path: Destination registry file.
    Returns:
        True iff a file was moved by this call.
    Raises:
        TypeError: Either path is not a ``pathlib.Path``; raised before any filesystem access.
        DataIntegrityError: Both files exist (message names both paths), or the moved bytes do not hash
            identically to the source.
    """
    if not isinstance(legacy_path, Path) or not isinstance(target_path, Path):
        raise TypeError("legacy_path and target_path must be pathlib.Path")
    legacy = legacy_path
    target = target_path
    if not legacy.is_file():
        return False
    if target.exists():
        raise DataIntegrityError(f"procedure registry is ambiguous: both {legacy} and {target} exist")
    payload = legacy.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.staged-{os.getpid()}")
    with staged.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(staged, target)
    _fsync_directory(target.parent)
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    if target_sha != source_sha:
        raise DataIntegrityError(f"procedure registry migration changed content: {target}")
    legacy.unlink()
    logger.info(
        "[DATA] procedure_registry migrated legacy=%s target=%s bytes=%d sha256=%s",
        legacy, target, len(payload), target_sha,
    )
    return True


def _read_events(registry_path: Path | None = None) -> list[dict[str, Any]]:
    path = PROCEDURE_REGISTRY_PATH if registry_path is None else registry_path
    _reject_unmigrated_legacy(path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"procedure registry line {i} is not valid JSON") from exc
        if not isinstance(event, dict) or event.get("event") not in (EVENT_REGISTRATION, EVENT_EVALUATION):
            raise DataIntegrityError(f"procedure registry line {i} is not a known event")
        if event["event"] == EVENT_EVALUATION:
            try:
                _utc(event["resolved_end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataIntegrityError(f"procedure registry line {i} has no valid resolved_end") from exc
        events.append(event)
    return events


def _append_event(registry_path: Path | None, event: dict[str, Any]) -> None:
    if registry_path is None or not isinstance(registry_path, Path):
        raise TypeError("registry_path must be a pathlib.Path")
    path = registry_path
    _reject_unmigrated_legacy(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


def load_registrations(registry_path: Path | None = None) -> tuple[ProcedureRegistration, ...]:
    """Registration events in file order; ``None`` reads ``PROCEDURE_REGISTRY_PATH`` at call time (read default).

    Raises:
        DataIntegrityError: Unknown/malformed event, or the canonical registry is read while the legacy file
            still exists (unmigrated evidence must never be mistaken for absent evidence).
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


def find_registration(digest: str, registry_path: Path | None = None) -> ProcedureRegistration:
    """Registration for ``digest``; ``None`` reads ``PROCEDURE_REGISTRY_PATH`` at call time (read default).

    Raises:
        DataIntegrityError: The digest is not registered, or as ``load_registrations``.
    """
    for registration in load_registrations(registry_path):
        if registration.procedure_digest == digest:
            return registration
    raise DataIntegrityError(f"procedure {digest} is not registered")


def consulted_data_horizon(
    history_dir: Path | None = None, registry_path: Path | None = None, *, journal_path: Path | None = None
) -> pd.Timestamp:
    """Latest data timestamp any research look may have consulted (I-SOURCE-COMPLETE).

    The horizon is the maximum over every independent consultation record:
    the sealed unseal ceilings (always counted, because pruned or unmigrated
    history cannot prove they were not read), every dated look in the canonical
    run-history registry, every forward evaluation event of the procedure
    registry, and the process research journal's locked horizon when a journal
    is supplied. Appending evidence to any source can only raise the result
    (I-HORIZON-MONOTONE). Legacy JSONL shards are not evidence; they count only
    after import into the registry.

    Args:
        history_dir: Run-history directory; ``None`` reads the canonical operator
            registry (read-only default), otherwise ``<dir>/registry.sqlite3``.
        registry_path: Procedure registry (registration/evaluation JSONL events);
            ``None`` reads ``PROCEDURE_REGISTRY_PATH`` at call time (read-only default).
        journal_path: Process research journal; when given it must exist.
    Returns:
        A tz-aware UTC timestamp, never earlier than ``MHS_FINAL_OOS_CUTOFF_2026H1``.
    Raises:
        DataIntegrityError: any supplied or existing source is unreadable or
            corrupt (registry, procedure events, or a missing/corrupt journal).
    """
    # 봉인 해제 상한은 이력이 가지치기돼도 조회된 것으로 간주한다.
    ends = [HOLDOUT_CUTOFF, MHS_FINAL_OOS_CUTOFF_2026H1]
    history_registry = _resolve_history_registry(history_dir)
    registry_horizon = consulted_registry_horizon(history_registry)
    if registry_horizon is not None:
        ends.append(registry_horizon)
    ends.extend(
        _utc(event["resolved_end"])
        for event in _read_events(registry_path)
        if event["event"] == EVENT_EVALUATION
    )
    if journal_path is not None:
        ends.append(consulted_process_horizon(journal_path))
    horizon = max(ends)
    logger.debug(
        "[DATA] consulted_data_horizon horizon=%s registry=%s journal=%s",
        horizon.isoformat(), history_registry, journal_path,
    )
    return horizon


def register_procedure(
    request: Any,
    *,
    now: pd.Timestamp,
    registry_path: Path,
    history_dir: Path,
) -> ProcedureRegistration:
    """Freeze one procedure before the data that will judge it exists.

    Both stores are explicit: the registration event is written to ``registry_path`` and its frozen data horizon is
    derived from ``history_dir``; letting either default would let an ad-hoc caller register against, or into, the
    operator's evidence.

    Args:
        request: Alpha-relevant request flags whose identity is frozen.
        now: Trusted tz-aware UTC registration clock.
        registry_path: Procedure registry receiving the registration event.
        history_dir: Run-history directory whose registry bounds the consulted data horizon.
    Raises:
        TypeError: ``registry_path`` or ``history_dir`` is not a ``pathlib.Path``; raised before any read or write.
        ValueError: ``now`` is naive or the request already references a registration.
        DataIntegrityError: Already registered, ``now`` does not follow the consulted horizon, a consulted source
            is corrupt, or the canonical registry is used while the legacy file still exists.
    """
    if not isinstance(registry_path, Path) or not isinstance(history_dir, Path):
        raise TypeError("registry_path and history_dir must be pathlib.Path")
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
    legacy_history_dir: Path,
    legacy_registry_path: Path,
) -> ProcessEvaluationPlan:
    """Register a complete process procedure and permitted looks before judging data exists.

    Args:
        plan: Strict process definition, fixed family budget and future look schedule.
        now: Trusted UTC registration time.
        journal_path: Durable process research journal.
        legacy_history_dir: Run-history directory bounding the persisted horizon (required).
        legacy_registry_path: Existing forward registration/evaluation history (required).
    Returns:
        An immutable plan referencing the persisted process registration identity.
    Raises:
        TypeError: A non-``Path`` source; raised before any I/O.
        ValueError: ``now`` is naive.
        DataIntegrityError: Procedure/family identity, consultation history or future schedule conflicts.
    """
    if not isinstance(legacy_history_dir, Path) or not isinstance(legacy_registry_path, Path):
        raise TypeError("legacy_history_dir and legacy_registry_path must be pathlib.Path")
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    if process_procedure_digest(plan.procedure) != plan.procedure_digest:
        raise DataIntegrityError("procedure digest does not match the frozen definition")
    now_utc = now.tz_convert("UTC")
    floor = consulted_data_horizon(legacy_history_dir, legacy_registry_path, journal_path=journal_path)
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
    registry_path: Path,
) -> None:
    """Append one evaluation event; every forward look advances later registrations' data horizon.

    Raises:
        TypeError: ``registry_path`` is not a ``pathlib.Path``; raised before any read or write.
        ValueError: ``now`` or ``resolved_end`` is naive.
        DataIntegrityError: The canonical registry is used while the legacy file still exists.
    """
    if not isinstance(registry_path, Path):
        raise TypeError("registry_path must be a pathlib.Path")
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
