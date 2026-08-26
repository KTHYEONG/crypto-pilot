"""Append-only, rotation-capped MHS run-history ledger.

One JSON-Lines shard per active run set plus immutable timestamped archives,
with ``latest.json`` holding the most recent run snapshot. No hardcoded
absolute paths: callers derive the history directory dynamically.

Each record is one JSON line in ``active.jsonl``; when appending exceeds
``RUN_HISTORY_SHARD_MAX_BYTES``, the shard rotates to an immutable archive.

The trial set behind the Deflated Sharpe Ratio denominator is defined here
exactly once: ``is_trial_record`` decides admission (outcome-blind) and
``trial_identity_key`` canonicalizes a record's flags into one identity key
(I-SAME-TRIAL-SET). Distinct keys also accumulate in ``trials_ledger.json``,
which archive rotation never touches (I-MONOTONE-TRIALS).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.mhs.params import MHS_FINAL_OOS_CUTOFF_2026H1, SEARCH_TRIALS_ATTEMPTED
from src.research.evaluation.policy import HOLDOUT_CUTOFF

RUN_HISTORY_SHARD_MAX_BYTES: int = 262144
RUN_HISTORY_MAX_SHARDS: int = 12

# Registered whitelist of request fields that never enter a strategy decision
# path (telemetry, resource guards, input-path pinning, opt-in extra replays).
# Fail-closed: any field NOT registered here always stays part of the trial
# identity key, so a newly added alpha flag can never silently merge trials.
RESEARCH_NEUTRAL_FLAGS: frozenset[str] = frozenset[str]({
    "log_run",
    "max_rss_bytes",
    "ram_guard",
    "data_root",
    "partition",
    "touch_diagnostic",
    "ladder_diagnostic",
    "peg_chase_diagnostic",
    "committee_growth_diagnostic",
    "committee_member_attribution",
    "discovery_gate_adjusted_net_t",
    "discovery_gate_regime_scaled_net_t",
})

# Pool-window admissibility for recorded trial outcomes: derived from the
# registered sealed-holdout extension width, never a literal day count.
TRIAL_POOL_WINDOW_TOLERANCE: pd.Timedelta = (
    MHS_FINAL_OOS_CUTOFF_2026H1 - HOLDOUT_CUTOFF
)

# Repository-canonical history directory used when a caller passes no explicit
# directory; mirrors the persist-time ``<target.parent>/mhs_run_history`` layout.
_DEFAULT_HISTORY_DIR = Path("docs") / "results" / "mhs_run_history"

_ACTIVE_FILE_NAME = "active.jsonl"
_LATEST_FILE_NAME = "latest.json"
_ARCHIVE_PREFIX = "mhs_run_history_"
_ARCHIVE_SUFFIX = ".jsonl"
_TRIALS_LEDGER_FILE_NAME = "trials_ledger.json"


def mhs_run_history_dir(target: Path) -> Path:
    """History directory derived from a persisted report target.

    Always ``target.parent / 'mhs_run_history'`` so test fixtures under
    ``tmp_path`` isolate their run history from the repository tree.
    """
    return target.parent / "mhs_run_history"


def _archive_path(history_dir: Path, utc_millis: int) -> Path:
    return history_dir / f"{_ARCHIVE_PREFIX}{utc_millis}{_ARCHIVE_SUFFIX}"


def _unique_archive_path(history_dir: Path) -> Path:
    """Rotated archive name that stays unique even for same-millisecond rotations."""
    utc_millis = int(time.time() * 1000)
    archive = _archive_path(history_dir, utc_millis)
    while archive.exists():
        utc_millis += 1
        archive = _archive_path(history_dir, utc_millis)
    return archive


def _serialize_record(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True)


def _prune_archives(history_dir: Path) -> None:
    archives = sorted(history_dir.glob(f"{_ARCHIVE_PREFIX}*{_ARCHIVE_SUFFIX}"))
    excess = len(archives) - RUN_HISTORY_MAX_SHARDS
    for stale in archives[:excess]:
        stale.unlink()


def append_run_history_record(record: Mapping[str, Any], history_dir: Path) -> Path:
    """Append one run record to the ledger, rotating/pruning as needed.

    Returns the active shard path. Rotation and pruning happen only when the
    append would push the active shard past the byte budget, so the steady
    state is a single small append with no directory scan per run. Valid trial
    records additionally upsert their identity key into ``trials_ledger.json``
    (first-seen wins); that file is never a pruning target, keeping the trial
    denominator monotone across archive rotation. A ledger failure is
    observational and never breaks the append itself.
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    active = history_dir / _ACTIVE_FILE_NAME
    line = _serialize_record(record) + "\n"

    if active.exists() and active.stat().st_size + len(line.encode("utf-8")) > RUN_HISTORY_SHARD_MAX_BYTES:
        active.rename(_unique_archive_path(history_dir))
        _prune_archives(history_dir)

    with active.open("a", encoding="utf-8") as fh:
        fh.write(line)

    latest = history_dir / _LATEST_FILE_NAME
    latest.write_text(line, encoding="utf-8")
    _upsert_trials_ledger(record, history_dir)
    return active


# --- trial-set definition (single source for N and V) ------------------------


def trial_identity_key(record: Mapping[str, Any]) -> str | None:
    """Canonical identity key of one recorded configuration.

    Normalizes the record's ``flags`` against the ``MhsDiagnosticRequest``
    field defaults (missing key -> default, explicit ``None`` -> default),
    drops the registered ``RESEARCH_NEUTRAL_FLAGS``, retains every other key
    (fail-closed against new alpha fields), and serializes canonically. Two
    records share a trial iff they denote the same strategy decision path,
    regardless of schema drift or telemetry-only flag differences.
    """
    if not isinstance(record, Mapping):
        return None
    from dataclasses import fields as dc_fields

    from src.application.research.mhs.contracts import MhsDiagnosticRequest

    flags = record.get("flags")
    flags = flags if isinstance(flags, Mapping) else {}
    normalized: dict[str, Any] = {}
    for field in dc_fields(MhsDiagnosticRequest):
        value = flags.get(field.name, field.default)
        if value is None:
            value = field.default
        if field.name not in RESEARCH_NEUTRAL_FLAGS:
            normalized[field.name] = value
    for key, value in flags.items():
        # Unknown keys are unregistered by construction: retain them fail-closed.
        if key not in normalized and key not in RESEARCH_NEUTRAL_FLAGS:
            normalized[key] = value
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda o: f"<{type(o).__module__}.{type(o).__qualname__}>",
    )


def _carries_data_integrity_code(record: Mapping[str, Any]) -> bool:
    """True when the record declares any registered data-integrity reason code."""
    research_go = record.get("research_go")
    if not isinstance(research_go, Mapping):
        return False
    codes: tuple[Any, ...] = ()
    for field in ("reason_codes", "data_integrity_reason_codes"):
        declared = research_go.get(field)
        if isinstance(declared, (list, tuple)):
            codes = (*codes, *declared)
    if not codes:
        return False
    from src.application.research.mhs.research_go import GO_REASON_DATA_INTEGRITY_CODES

    return not GO_REASON_DATA_INTEGRITY_CODES.isdisjoint(codes)


def _has_finite_blend_sharpe(record: Mapping[str, Any]) -> bool:
    blend = record.get("blend")
    sharpe = blend.get("primary_naive_sharpe") if isinstance(blend, Mapping) else None
    if not isinstance(sharpe, (int, float)) or isinstance(sharpe, bool):
        return False
    return math.isfinite(float(sharpe))


def is_trial_record(record: Mapping[str, Any]) -> bool:
    """Outcome-blind admissibility of one history record as a strategy trial.

    A record is a trial iff it completed, carries no registered data-integrity
    reason code, and reports a finite blend Sharpe. The Sharpe enters only
    through a finiteness check -- never its value, sign, or rank.
    """
    if not isinstance(record, Mapping):
        return False
    if record.get("status") != "COMPLETE":
        return False
    if _carries_data_integrity_code(record):
        return False
    return _has_finite_blend_sharpe(record)


def _parse_utc_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    try:
        if parsed.tz is None:
            return parsed.tz_localize("UTC")
        return parsed.tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def _load_trials_ledger(directory: Path) -> dict[str, str] | None:
    """Read the monotone ledger; ``None`` marks an unusable (corrupt) file."""
    path = directory / _TRIALS_LEDGER_FILE_NAME
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return {str(key): str(value) for key, value in loaded.items()}


def _upsert_trials_ledger(record: Mapping[str, Any], history_dir: Path) -> None:
    try:
        if not is_trial_record(record):
            return
        key = trial_identity_key(record)
        if key is None:
            return
        ledger = _load_trials_ledger(history_dir)
        if ledger is None:
            ledger = {}
        ledger.setdefault(key, datetime.now(UTC).isoformat())
        payload = json.dumps(ledger, ensure_ascii=False, sort_keys=True)
        (history_dir / _TRIALS_LEDGER_FILE_NAME).write_text(payload, encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass  # observational: the shard append itself already succeeded


def _iter_history_records(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from every JSONL shard; raises on unreadable IO."""
    for shard in sorted(directory.glob("*.jsonl")):
        with shard.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if isinstance(record, dict):
                    yield record


def derive_trials_attempted(history_dir: Path | str | None = None) -> tuple[int, str]:
    """Audit-trials denominator for the DSR from the run history itself.

    Counts the distinct trial identity keys admitted by ``is_trial_record``
    (the same predicate and equivalence key ``window_trial_sharpes`` uses --
    I-SAME-TRIAL-SET), unioned with the rotation-proof ``trials_ledger.json``
    so archived-away exploration keeps counting (I-MONOTONE-TRIALS). Returns
    ``(SEARCH_TRIALS_ATTEMPTED + counted, source)`` where ``source`` is
    ``'constant_plus_ledger'``, ``'constant_plus_history'`` (no usable
    ledger), or ``'constant_fallback'`` when no readable evidence exists at
    all. O(history_lines).
    """
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    ledger = _load_trials_ledger(directory)
    try:
        seen: set[str] = set()
        observed_records = 0
        for record in _iter_history_records(directory):
            observed_records += 1
            if not is_trial_record(record):
                continue
            key = trial_identity_key(record)
            if key is not None:
                seen.add(key)
    except (OSError, json.JSONDecodeError):
        return SEARCH_TRIALS_ATTEMPTED, "constant_fallback"
    union = (seen | set(ledger)) if ledger is not None else seen
    if ledger:
        return SEARCH_TRIALS_ATTEMPTED + len(union), "constant_plus_ledger"
    if observed_records == 0:
        # No readable history at all: the denominator's provenance must say so.
        return SEARCH_TRIALS_ATTEMPTED, "constant_fallback"
    return SEARCH_TRIALS_ATTEMPTED + len(union), "constant_plus_history"


def _matches_window(
    record: Mapping[str, Any],
    wanted_start: pd.Timestamp,
    wanted_end: pd.Timestamp,
) -> bool:
    start = _parse_utc_timestamp(record.get("start"))
    resolved_end = _parse_utc_timestamp(record.get("resolved_end"))
    if start is None or resolved_end is None:
        return False
    if start != wanted_start:
        return False
    gap = abs(resolved_end - wanted_end)
    return bool(gap <= TRIAL_POOL_WINDOW_TOLERANCE)


def window_trial_sharpes(
    window: tuple[str, str], history_dir: Path | str | None = None
) -> tuple[float, ...]:
    """Annualized blend Sharpe outcomes recorded for one evaluation window.

    Single pass over every JSONL shard using the shared trial-set definition:
    a record qualifies when ``is_trial_record`` admits it, its ``start``
    matches exactly, and its ``resolved_end`` lies within the registered
    ``TRIAL_POOL_WINDOW_TOLERANCE`` of ``window``'s end (so widening the
    evaluation window pools with the sealed window it extends). A re-run of
    one configuration with the same outcome collapses to a single entry;
    distinct outcomes of one configuration stay distinct entries. Returns the
    outcomes ascending; an unreadable or missing history yields ``()``.
    """
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    wanted_start = _parse_utc_timestamp(window[0])
    wanted_end = _parse_utc_timestamp(window[1])
    if wanted_start is None or wanted_end is None:
        return ()
    try:
        seen_entries: set[tuple[str, float]] = set()
        outcomes: list[float] = []
        for record in _iter_history_records(directory):
            if not is_trial_record(record):
                continue
            blend = record["blend"]
            sharpe = float(blend["primary_naive_sharpe"])  # finite: is_trial_record
            if not _matches_window(record, wanted_start, wanted_end):
                continue
            key = trial_identity_key(record)
            if key is None:
                continue
            entry = (key, sharpe)
            if entry in seen_entries:
                continue
            seen_entries.add(entry)
            outcomes.append(sharpe)
        return tuple(sorted(outcomes))
    except (OSError, json.JSONDecodeError):
        return ()


def trial_pool_disclosure(
    window: tuple[str, str], history_dir: Path | str | None = None
) -> dict[str, Any]:
    """Observational disclosure of how the DSR trial pool was assembled.

    Pure accounting over one O(history_lines) scan. Every readable record
    falls into exactly one bucket -- admitted trial, or excluded by exactly
    one registered ground (incomplete status / data-integrity code /
    non-finite blend Sharpe) -- so the counts sum to ``n_history_records``.
    ``distinct_trial_keys`` and ``pool_window_span_days`` describe the
    tolerance-merged pool actually matched for ``window`` (its end-date
    heterogeneity, in days). Emits no GO reason code and degrades to zeros
    with ``source='constant_fallback'`` on any unreadable input.
    """
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    wanted_start = _parse_utc_timestamp(window[0])
    wanted_end = _parse_utc_timestamp(window[1])
    if wanted_start is None or wanted_end is None:
        return {**_EMPTY_DISCLOSURE}
    ledger = _load_trials_ledger(directory)
    disclosure: dict[str, Any] = {**_EMPTY_DISCLOSURE}
    try:
        matched_keys: set[str] = set()
        matched_ends: list[pd.Timestamp] = []
        for record in _iter_history_records(directory):
            disclosure["n_history_records"] += 1
            flags = record.get("flags")
            if isinstance(flags, Mapping):
                disclosure["neutral_flags_dropped"] += sum(
                    1 for name in flags if name in RESEARCH_NEUTRAL_FLAGS
                )
            if record.get("status") != "COMPLETE":
                disclosure["excluded_not_complete"] += 1
                continue
            if _carries_data_integrity_code(record):
                disclosure["excluded_data_integrity"] += 1
                continue
            if not _has_finite_blend_sharpe(record):
                disclosure["excluded_nonfinite_blend"] += 1
                continue
            if _matches_window(record, wanted_start, wanted_end):
                resolved_end = _parse_utc_timestamp(record.get("resolved_end"))
                if resolved_end is not None:
                    matched_ends.append(resolved_end)
                key = trial_identity_key(record)
                if key is not None:
                    matched_keys.add(key)
            disclosure["n_trial_records"] += 1
    except (OSError, json.JSONDecodeError):
        return {**_EMPTY_DISCLOSURE}
    disclosure["distinct_trial_keys"] = len(matched_keys)
    if len(matched_ends) >= 2:
        span_seconds = (max(matched_ends) - min(matched_ends)).total_seconds()
        disclosure["pool_window_span_days"] = float(span_seconds / 86400.0)
    disclosure["ledger_size"] = len(ledger) if ledger is not None else 0
    if ledger:
        disclosure["source"] = "constant_plus_ledger"
    elif disclosure["n_history_records"] > 0:
        disclosure["source"] = "constant_plus_history"
    return disclosure


_EMPTY_DISCLOSURE: dict[str, Any] = {
    "n_history_records": 0,
    "n_trial_records": 0,
    "excluded_data_integrity": 0,
    "excluded_not_complete": 0,
    "excluded_nonfinite_blend": 0,
    "distinct_trial_keys": 0,
    "neutral_flags_dropped": 0,
    "pool_window_span_days": 0.0,
    "ledger_size": 0,
    "source": "constant_fallback",
}
