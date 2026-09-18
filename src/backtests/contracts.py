"""Execution registry value objects with strict UTC, identity and JSON invariants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Union

JsonValue = Union[None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]]  # noqa: UP007

_RUN_STATUS = ("completed", "failed", "timed_out", "signaled", "resource_rejected", "interrupted")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or "/" in run_id or "\\" in run_id:
        raise ValueError(f"run_id must be UUID hex without path components, got {run_id!r}")
    if len(run_id) != 32:
        raise ValueError(f"run_id must be UUID hex without path components, got {run_id!r}")
    if any(c not in "0123456789abcdefABCDEF" for c in run_id):
        raise ValueError(f"run_id must be UUID hex without path components, got {run_id!r}")


def _validate_utc_iso8601(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be timezone-aware UTC ISO8601, got {value!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{label} must be timezone-aware UTC ISO8601, got {value!r}") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC ISO8601, got {value!r}")


def _validate_json_value(value: JsonValue) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata must not contain non-finite float, got {value!r}")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"metadata keys must be strings, got {key!r}")
            _validate_json_value(item)
        return
    raise ValueError(f"metadata must be JSON values, got {value!r}")


def _validate_metadata(mapping: dict[str, JsonValue], label: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a JSON object, got {mapping!r}")
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings, got {key!r}")
        _validate_json_value(item)


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{label} must be SHA-256 hex, got {value!r}")


def _validate_evidence_id(value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"evidence_id must be a plain content identity, got {value!r}")


def _validate_budget(value: int | None, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a positive integer budget, got {value!r}")
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer budget, got {value!r}")


def utc_now_iso8601() -> str:
    """Return the current UTC time as a timezone-aware ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


@dataclass(frozen=True, slots=True)
class RunRegistration:
    """One execution attempt and its request provenance, independent of financial trial admission. Args: run identity, UTC registration time, request metadata and optional owned directory."""

    run_id: str
    strategy_id: str
    registered_at: str
    request: dict[str, JsonValue]
    managed_directory: Path | None

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if not isinstance(self.strategy_id, str) or not self.strategy_id:
            raise ValueError(f"strategy_id must be a non-empty string, got {self.strategy_id!r}")
        _validate_utc_iso8601(self.registered_at, "registered_at")
        _validate_metadata(self.request, "request")
        if self.managed_directory is not None and not isinstance(self.managed_directory, Path):
            raise ValueError(f"managed_directory must be a Path or None, got {self.managed_directory!r}")


@dataclass(frozen=True, slots=True)
class RunFinalization:
    """Observed execution outcome with financial validity kept separate. Unknown financial evidence remains null. Args: run identity, UTC finalization time, process status, financial flags and complete outcome metadata."""

    run_id: str
    status: Literal["completed", "failed", "timed_out", "signaled", "resource_rejected", "interrupted"]
    finalized_at: str
    primary_valid: bool | None
    terminal_certified: bool | None
    outcome: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if self.status not in _RUN_STATUS:
            raise ValueError(f"status must be one of {_RUN_STATUS}, got {self.status!r}")
        _validate_utc_iso8601(self.finalized_at, "finalized_at")
        if self.primary_valid is not None and not isinstance(self.primary_valid, bool):
            raise ValueError(f"primary_valid must be bool or None, got {self.primary_valid!r}")
        if self.terminal_certified is not None and not isinstance(self.terminal_certified, bool):
            raise ValueError(f"terminal_certified must be bool or None, got {self.terminal_certified!r}")
        _validate_metadata(self.outcome, "outcome")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Verified artifact ownership and content identity for safe shared-evidence retention. Args: owning run, artifact role, absolute path, SHA-256, nonnegative size, ownership and optional shared evidence identity."""

    run_id: str
    role: str
    path: Path
    sha256: str
    byte_count: int
    managed: bool
    evidence_id: str | None

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if not isinstance(self.role, str) or not self.role:
            raise ValueError(f"role must be a non-empty string, got {self.role!r}")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError(f"path must be an absolute Path, got {self.path!r}")
        _validate_sha256(self.sha256, "sha256")
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int):
            raise ValueError(f"byte_count must be a nonnegative integer, got {self.byte_count!r}")
        if self.byte_count < 0:
            raise ValueError(f"byte_count must be a nonnegative integer, got {self.byte_count!r}")
        if not isinstance(self.managed, bool):
            raise ValueError(f"managed must be bool, got {self.managed!r}")
        _validate_evidence_id(self.evidence_id)


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Explicit detail-retention budgets; absent budgets do not authorize evidence loss. Args: positive optional byte and run budgets. Raises: ValueError for invalid budgets."""

    max_detail_bytes: int | None = None
    max_detail_runs: int | None = None

    def __post_init__(self) -> None:
        _validate_budget(self.max_detail_bytes, "max_detail_bytes")
        _validate_budget(self.max_detail_runs, "max_detail_runs")


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """Detail reclamation observations that never authorize removal of run metadata, trial history or protected evidence. Args: evidence identities, byte observations and budget feasibility."""

    evidence_ids: tuple[str, ...]
    reclaimable_bytes: int
    protected_bytes: int
    budget_satisfied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_ids, tuple) or any(not isinstance(v, str) or not v for v in self.evidence_ids):
            raise ValueError(f"evidence_ids must be a tuple of non-empty strings, got {self.evidence_ids!r}")
        for label, count in (("reclaimable_bytes", self.reclaimable_bytes), ("protected_bytes", self.protected_bytes)):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{label} must be a nonnegative integer, got {count!r}")
        if not isinstance(self.budget_satisfied, bool):
            raise ValueError(f"budget_satisfied must be bool, got {self.budget_satisfied!r}")


@dataclass(frozen=True, slots=True)
class RetentionResult:
    """Detail reclamation observations that never authorize removal of run metadata, trial history or protected evidence. Args: evidence identities, byte observations and budget feasibility."""

    removed_evidence_ids: tuple[str, ...]
    reclaimed_bytes: int
    budget_satisfied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.removed_evidence_ids, tuple) or any(
            not isinstance(v, str) or not v for v in self.removed_evidence_ids
        ):
            raise ValueError(f"removed_evidence_ids must be a tuple of non-empty strings, got {self.removed_evidence_ids!r}")
        if isinstance(self.reclaimed_bytes, bool) or not isinstance(self.reclaimed_bytes, int) or self.reclaimed_bytes < 0:
            raise ValueError(f"reclaimed_bytes must be a nonnegative integer, got {self.reclaimed_bytes!r}")
        if not isinstance(self.budget_satisfied, bool):
            raise ValueError(f"budget_satisfied must be bool, got {self.budget_satisfied!r}")


__all__ = [
    "ArtifactReference",
    "JsonValue",
    "RetentionPlan",
    "RetentionPolicy",
    "RetentionResult",
    "RunFinalization",
    "RunRegistration",
    "utc_now_iso8601",
]
