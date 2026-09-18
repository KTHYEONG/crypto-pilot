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

import dataclasses
import json
import logging
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.mhs.params import MHS_FINAL_OOS_CUTOFF_2026H1, SEARCH_TRIALS_ATTEMPTED
from src.quant.evaluation.policy import HOLDOUT_CUTOFF

logger = logging.getLogger("MhsRunHistory")

_REGISTRY_NAMESPACE = "mhs_legacy_horizon"
_LIVE_SOURCE_ID = "live"

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


def canonical_history_registry() -> Path:
    """Return the sole persistent registry for MHS trial-history provenance.

    Returns:
        The backtest registry path used for trial denominators and window outcomes.
    """
    from src.common.paths import BACKTESTS_DIR

    return BACKTESTS_DIR / "registry.sqlite3"


def _is_canonical_history_request(history_dir: Path | str | None) -> bool:
    if history_dir is None:
        return True
    if str(history_dir).endswith("docs/results/mhs_run_history"):
        return True
    return _resolve_history_registry(history_dir) == canonical_history_registry()


def _resolve_history_registry(history_dir: Path | str | None) -> Path:
    """Map a compatibility history location to its unified registry file."""
    if history_dir is None:
        return canonical_history_registry()
    text = str(history_dir)
    if text.endswith("docs/results/mhs_run_history"):
        return canonical_history_registry()
    return Path(text) / "registry.sqlite3"


def _load_registry_state(registry: Path) -> tuple[list[dict[str, Any]], dict[str, str]] | None:
    """Read registry history rows and trial ledger; None when no imported evidence."""
    if not registry.is_file():
        return None
    try:
        conn = sqlite3.connect(str(registry), timeout=5.0)
        try:
            rows = conn.execute(
                "SELECT record_json FROM history_records WHERE namespace = ? ORDER BY source_id, ordinal",
                (_REGISTRY_NAMESPACE,),
            ).fetchall()
            ledger_rows = conn.execute(
                "SELECT identity_key, first_seen FROM trials WHERE namespace = ?",
                (_REGISTRY_NAMESPACE,),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return None
    if not rows and not ledger_rows:
        return None
    records: list[dict[str, Any]] = []
    for (payload,) in rows:
        try:
            parsed = json.loads(str(payload))
        except ValueError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    ledger = {str(key): str(seen) for key, seen in ledger_rows}
    return records, ledger


def append_run_history_record(record: Mapping[str, Any], history_dir: Path | str | None) -> Path:
    """Persist one legacy-horizon summary into the unified transactional registry. Args: raw history record and compatibility history location. Returns: the registry file path. Raises: sqlite3.Error or OSError if durable persistence fails."""
    from src.backtests.registry import initialize_registry

    registry = _resolve_history_registry(history_dir)
    initialize_registry(registry)
    payload = json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
    admitted = is_trial_record(record)
    identity = trial_identity_key(record) if admitted else None
    first_seen = record.get("run_at") if isinstance(record.get("run_at"), str) else datetime.now(UTC).isoformat()
    conn = sqlite3.connect(str(registry), timeout=5.0, isolation_level="DEFERRED")
    try:
        with conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(ordinal), -1) FROM history_records WHERE source_id = ?",
                (_LIVE_SOURCE_ID,),
            ).fetchone()
            ordinal = int(row[0]) + 1
            conn.execute(
                "INSERT INTO history_records (source_id, ordinal, namespace, record_json, admitted, identity_key)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (_LIVE_SOURCE_ID, ordinal, _REGISTRY_NAMESPACE, payload, 1 if admitted else 0, identity),
            )
            if admitted and identity is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO trials (namespace, identity_key, first_seen, provenance_json)"
                    " VALUES (?, ?, ?, ?)",
                    (_REGISTRY_NAMESPACE, identity, str(first_seen), payload),
                )
    finally:
        conn.close()
    return registry


# --- trial-set definition (single source for N and V) ------------------------


def _identity_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda o: f"<{type(o).__module__}.{type(o).__qualname__}>",
    )


def _equals_field_default(value: Any, field: Any) -> bool:
    if field.default is dataclasses.MISSING:
        return False
    return _identity_dump(value) == _identity_dump(field.default)


def _sparse_identity_key(key: str) -> str:
    """Re-key one stored identity (dense or sparse) into the sparse form."""
    try:
        parsed = json.loads(key)
    except json.JSONDecodeError:
        return key
    if not isinstance(parsed, dict):
        return key
    from src.mhs.contracts import MhsDiagnosticRequest

    for field in dataclasses.fields(MhsDiagnosticRequest):
        if field.name in parsed and _equals_field_default(parsed[field.name], field):
            del parsed[field.name]
    return _identity_dump(parsed)


def trial_identity_key(record: Mapping[str, Any]) -> str | None:
    """Canonical identity key of one recorded configuration.

    Normalizes the record's ``flags`` against the ``MhsDiagnosticRequest``
    field defaults (missing key -> default, explicit ``None`` -> default),
    drops the registered ``RESEARCH_NEUTRAL_FLAGS``, retains every other key
    (fail-closed against new alpha fields), and serializes canonically. Two
    records share a trial iff they denote the same strategy decision path,
    regardless of schema drift or telemetry-only flag differences. Fields equal to their registered default are
    omitted, so adding a defaulted request field never re-keys existing configurations.
    """
    if not isinstance(record, Mapping):
        return None
    from dataclasses import fields as dc_fields

    from src.mhs.contracts import MhsDiagnosticRequest

    flags = record.get("flags")
    flags = flags if isinstance(flags, Mapping) else {}
    normalized: dict[str, Any] = {}
    registered = {f.name for f in dc_fields(MhsDiagnosticRequest)}
    for field in dc_fields(MhsDiagnosticRequest):
        value = flags.get(field.name, field.default)
        if value is None:
            value = field.default
        if field.name == "data_policy" and "data_policy" not in flags:
            # Migration: a data_policy-less legacy record keeps its legacy
            # identity instead of adopting the new zombie default.
            value = "legacy"
        if field.name not in RESEARCH_NEUTRAL_FLAGS and not _equals_field_default(value, field):
            normalized[field.name] = value
    for key, value in flags.items():
        # Unknown keys are unregistered by construction: retain them fail-closed.
        if key not in registered and key not in RESEARCH_NEUTRAL_FLAGS:
            normalized[key] = value  # noqa: PERF403
    snapshot = record.get("params_snapshot")
    if isinstance(snapshot, Mapping):
        # Mapped snapshot keys and values are canonically retained, so newly
        # sealed decision parameters cannot silently merge trials.
        normalized["params_snapshot"] = dict(snapshot)
    elif "params_snapshot" not in record:
        normalized["params_snapshot"] = {"__legacy_params_snapshot__": "missing"}
    else:
        normalized["params_snapshot"] = {
            "__legacy_params_snapshot__": f"non-mapping:{type(snapshot).__module__}.{type(snapshot).__qualname__}"
        }
    return _identity_dump(normalized)


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
    from src.mhs.research_go import GO_REASON_DATA_INTEGRITY_CODES

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
    rekeyed: dict[str, str] = {}
    for key, value in loaded.items():
        sparse = _sparse_identity_key(str(key))
        first = str(value)
        rekeyed[sparse] = min(rekeyed[sparse], first) if sparse in rekeyed else first
    return rekeyed


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
    all. O(history_lines). The unified registry preserves imported source provenance and monotone trial identity; source compatibility labels remain unchanged.
    """
    registry = _resolve_history_registry(history_dir)
    state = _load_registry_state(registry)
    if state is not None:
        records, ledger_map = state
        seen: set[str] = set()
        for record in records:
            if not is_trial_record(record):
                continue
            key = trial_identity_key(record)
            if key is not None:
                seen.add(key)
        union = seen | set(ledger_map)
        if ledger_map:
            return SEARCH_TRIALS_ATTEMPTED + len(union), "constant_plus_ledger"
        return SEARCH_TRIALS_ATTEMPTED + len(union), "constant_plus_history"
    if _is_canonical_history_request(history_dir):
        return SEARCH_TRIALS_ATTEMPTED, "constant_fallback"
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    ledger = _load_trials_ledger(directory)
    try:
        seen = set()
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
    outcomes ascending; an unreadable or missing history yields ``()``. The unified registry preserves imported source provenance and monotone trial identity; source compatibility labels remain unchanged.
    """
    registry = _resolve_history_registry(history_dir)
    state = _load_registry_state(registry)
    wanted_start = _parse_utc_timestamp(window[0])
    wanted_end = _parse_utc_timestamp(window[1])
    if wanted_start is None or wanted_end is None:
        return ()
    if state is not None:
        records, _ = state
        seen_entries: set[tuple[str, float]] = set()
        outcomes: list[float] = []
        for record in records:
            if not is_trial_record(record):
                continue
            blend = record["blend"]
            sharpe = float(blend["primary_naive_sharpe"])  # finite: is_trial_record
            if not _matches_window(record, wanted_start, wanted_end):
                continue
            identity = cast(str, trial_identity_key(record))
            entry = (identity, sharpe)
            if entry in seen_entries:
                continue
            seen_entries.add(entry)
            outcomes.append(sharpe)
        return tuple(sorted(outcomes))
    if _is_canonical_history_request(history_dir):
        return ()
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    try:
        seen_entries = set()
        outcomes = []
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
    with ``source='constant_fallback'`` on any unreadable input. The unified registry preserves imported source provenance and monotone trial identity; source compatibility labels remain unchanged.
    """
    registry = _resolve_history_registry(history_dir)
    state = _load_registry_state(registry)
    wanted_start = _parse_utc_timestamp(window[0])
    wanted_end = _parse_utc_timestamp(window[1])
    if wanted_start is None or wanted_end is None:
        return {**_EMPTY_DISCLOSURE}
    if state is not None:
        records, ledger_map = state
        disclosure: dict[str, Any] = {**_EMPTY_DISCLOSURE}
        matched_keys: set[str] = set()
        matched_ends: list[pd.Timestamp] = []
        for record in records:
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
        disclosure["distinct_trial_keys"] = len(matched_keys)
        if len(matched_ends) >= 2:
            span_seconds = (max(matched_ends) - min(matched_ends)).total_seconds()
            disclosure["pool_window_span_days"] = float(span_seconds / 86400.0)
        disclosure["ledger_size"] = len(ledger_map)
        if ledger_map:
            disclosure["source"] = "constant_plus_ledger"
        elif disclosure["n_history_records"] > 0:
            disclosure["source"] = "constant_plus_history"
        return disclosure
    if _is_canonical_history_request(history_dir):
        return {**_EMPTY_DISCLOSURE}
    directory = Path(history_dir) if history_dir is not None else _DEFAULT_HISTORY_DIR
    ledger = _load_trials_ledger(directory)
    disclosure = {**_EMPTY_DISCLOSURE}
    try:
        matched_keys = set()
        matched_ends = []
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
