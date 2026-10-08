"""MHS run-history trial set over the canonical SQLite registry.

Every research look is persisted as one ``history_records`` row of the
``mhs_legacy_horizon`` namespace in ``registry.sqlite3``. Reads default to the
canonical operator registry (read-only default); writes require an explicit
destination directory, so ad-hoc callers can never inflate operator evidence.

The trial set behind the Deflated Sharpe Ratio denominator is defined here
exactly once: ``is_trial_record`` decides admission (outcome-blind) and
``trial_identity_key`` canonicalizes a record's flags into one identity key
(I-SAME-TRIAL-SET). Readers fail closed on corrupt evidence wherever the
result feeds the DSR; legacy JSON-Lines history is evidence only after import
by ``src.backtests.migration``.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.params import MHS_FINAL_OOS_CUTOFF_2026H1, SEARCH_TRIALS_ATTEMPTED
from src.quant.evaluation.policy import HOLDOUT_CUTOFF

logger = logging.getLogger("MhsRunHistory")

_REGISTRY_NAMESPACE = "mhs_legacy_horizon"
_LIVE_SOURCE_ID = "live"

# Registered whitelist of request fields that never enter a strategy decision path (telemetry, resource guards, input-path pinning, opt-in extra replays, opt-in report-only diagnostics).
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
    "placebo_diagnostic",
    "phase_diagnostic",
    "signal_48h_diagnostic",
    "bootstrap_ci_diagnostic",
    "reference_books_diagnostic",
    "patient_reference_diagnostic",
})

# Pool-window admissibility for recorded trial outcomes: derived from the
# registered sealed-holdout extension width, never a literal day count.
TRIAL_POOL_WINDOW_TOLERANCE: pd.Timedelta = (
    MHS_FINAL_OOS_CUTOFF_2026H1 - HOLDOUT_CUTOFF
)


def canonical_history_registry() -> Path:
    """Return the operator run-history registry used as the READ default.

    Readers called with ``history_dir=None`` resolve here. Write paths never call this: the application boundary
    passes the operator directory (``BACKTESTS_DIR``) explicitly.

    Returns:
        ``BACKTESTS_DIR / "registry.sqlite3"`` (``BACKTESTS_DIR`` honours ``CRYPTO_PILOT_BACKTESTS_DIR``).
    """
    from src.common.paths import BACKTESTS_DIR

    return BACKTESTS_DIR / "registry.sqlite3"


def _resolve_history_registry(history_dir: Path | str | None) -> Path:
    """Registry file for one run-history location.

    ``None`` selects the canonical registry (read-only default for readers);
    any explicit location maps to ``<location>/registry.sqlite3`` so test and
    fixture histories stay isolated from the canonical registry by construction.
    Write paths never pass ``None``: ``append_run_history_record`` rejects it
    before any filesystem access.
    """
    if history_dir is None:
        return canonical_history_registry()
    return Path(history_dir) / "registry.sqlite3"


@dataclass(frozen=True, slots=True)
class HistoryRegistryState:
    """Verified snapshot of the run-history namespace of one registry file.

    ``records`` and ``stored_identity_keys`` are index-aligned (one entry per
    ``history_records`` row, ordered by ``source_id, ordinal``); the stored key
    is the ``identity_key`` column written at append/import time (``None`` for
    non-admitted rows). ``ledger`` maps every ``trials`` identity key of the
    namespace to its first-seen timestamp.
    """

    records: tuple[dict[str, Any], ...]
    stored_identity_keys: tuple[str | None, ...]
    ledger: Mapping[str, str]


def _load_registry_state(registry: Path) -> HistoryRegistryState | None:
    """Read the run-history namespace of one registry; ``None`` only when no evidence exists.

    Absence (no file, or a schema-valid file with zero history rows and zero
    trial rows) is the only ``None`` outcome. Anything that exists but cannot be
    read completely is corrupt evidence and must never be mistaken for absent
    evidence: a silently dropped row can hide a trial (understating the DSR
    denominator) or a consulted look (re-admitting contaminated forward data).

    Stored identities are verified, never trusted: every row's stored
    ``identity_key`` must equal the key recomputed from its record, and every
    ``trials`` key must already be in sparse baseline form. A mismatch means
    the identity definition drifted after the key was written; unioning such
    keys would inflate or merge trials, so it fails closed (I-ID-STABLE).

    Raises:
        DataIntegrityError: the file is not a readable SQLite registry, a required
            table/column is missing, any ``record_json`` row is not a JSON object,
            or identity drift.
    """
    if not registry.is_file():
        return None
    try:
        conn = sqlite3.connect(str(registry), timeout=5.0)
        try:
            rows = conn.execute(
                "SELECT source_id, ordinal, record_json, identity_key FROM history_records"
                " WHERE namespace = ? ORDER BY source_id, ordinal",
                (_REGISTRY_NAMESPACE,),
            ).fetchall()
            ledger_rows = conn.execute(
                "SELECT identity_key, first_seen FROM trials WHERE namespace = ?"
                " ORDER BY identity_key, first_seen",
                (_REGISTRY_NAMESPACE,),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        raise DataIntegrityError(f"run-history registry is unreadable: {registry}") from exc
    if not rows and not ledger_rows:
        return None
    records: list[dict[str, Any]] = []
    stored_keys: list[str | None] = []
    for source_id, ordinal, payload, stored_key in rows:
        try:
            parsed = json.loads(str(payload))
        except ValueError as exc:
            raise DataIntegrityError(
                f"corrupt history record {source_id}/{ordinal} in {registry}"
            ) from exc
        if not isinstance(parsed, dict):
            raise DataIntegrityError(
                f"history record {source_id}/{ordinal} is not a JSON object in {registry}"
            )
        records.append(parsed)
        stored_keys.append(None if stored_key is None else str(stored_key))
    ledger: dict[str, str] = {}
    for key, first_seen in ledger_rows:
        ledger.setdefault(str(key), str(first_seen))
    for (source_id, ordinal, _payload, stored_key), record in zip(rows, records, strict=True):
        if stored_key is None:
            continue
        if trial_identity_key(record) != str(stored_key):
            raise DataIntegrityError(
                f"trial identity drift: source_id={source_id} ordinal={ordinal}"
            )
    for key in ledger:
        if _sparse_identity_key(key) != key:
            raise DataIntegrityError(
                f"trial ledger key not in baseline form: {key[:80]}"
            )
    return HistoryRegistryState(tuple(records), tuple(stored_keys), ledger)


def _runs_request_ends(registry: Path) -> list[tuple[str, Any]]:
    """Registered ``runs`` request payloads as ``(run_id, request)`` in a deterministic order.

    Raises:
        DataIntegrityError: the ``runs`` table is unreadable or a ``request_json``
            value is not a JSON object.
    """
    try:
        conn = sqlite3.connect(str(registry), timeout=5.0)
        try:
            rows = conn.execute("SELECT run_id, request_json FROM runs ORDER BY run_id").fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        raise DataIntegrityError(f"run-history registry is unreadable: {registry}") from exc
    requests: list[tuple[str, Any]] = []
    for run_id, payload in rows:
        try:
            parsed = json.loads(str(payload))
        except ValueError as exc:
            raise DataIntegrityError(f"corrupt run request {run_id} in {registry}") from exc
        if not isinstance(parsed, dict):
            raise DataIntegrityError(f"run request {run_id} is not a JSON object in {registry}")
        requests.append((str(run_id), parsed))
    return requests


def consulted_registry_horizon(registry: Path) -> pd.Timestamp | None:
    """Latest data timestamp any look recorded in one registry may have consulted.

    Counts every ``history_records`` row of the run-history namespace regardless
    of status or trial admission (a failed or excluded run still read its data),
    using ``resolved_end`` and falling back to ``end``; and every ``runs`` row's
    registered request ``end`` (a registered backtest is treated as consulted from
    the moment it is registered, mirroring reserve-before-read). Returns ``None``
    when the registry is absent or holds no dated look.

    Raises:
        DataIntegrityError: the registry is unreadable (see ``_load_registry_state``),
            a ``runs.request_json`` is not a JSON object, a ``runs`` request has no
            non-null ``end`` (every sanctioned producer records one, so its absence
            is corruption that could hide a consulted look), or a present, non-null
            end value cannot be parsed as a timestamp.
    """
    if not registry.is_file():
        return None
    state = _load_registry_state(registry)
    ends: list[pd.Timestamp] = []
    for index, record in enumerate(state.records if state is not None else ()):
        candidate = record.get("resolved_end")
        if candidate is None:
            candidate = record.get("end")
        if candidate is None:
            continue
        parsed = _parse_utc_timestamp(candidate)
        if parsed is None:
            raise DataIntegrityError(
                f"unparseable consulted end {candidate!r} in history record {index} of {registry}"
            )
        ends.append(parsed)
    for run_id, request in _runs_request_ends(registry):
        end = request.get("end")
        if end is None:
            raise DataIntegrityError(f"run {run_id} in {registry} has no non-null request end")
        parsed = _parse_utc_timestamp(end)
        if parsed is None:
            raise DataIntegrityError(f"unparseable request end {end!r} for run {run_id} in {registry}")
        ends.append(parsed)
    return max(ends) if ends else None


def append_run_history_record(record: Mapping[str, Any], history_dir: Path | str) -> Path:
    """Persist one legacy-horizon summary into ``<history_dir>/registry.sqlite3``.

    The destination is mandatory because run history is DSR trial evidence: a write that silently resolved the
    operator registry would let any ad-hoc process (scratch script, foreign pytest file, subprocess without the
    storage redirect) inflate the operator's trial denominator and consulted horizon.

    Args:
        record: JSON-serializable history record.
        history_dir: Directory whose ``registry.sqlite3`` receives the record; initialized when absent. Operator
            callers pass ``BACKTESTS_DIR``.
    Returns:
        The registry file written.
    Raises:
        TypeError: ``history_dir`` is ``None`` or not ``str``/``Path``; raised before any filesystem access.
        sqlite3.Error, OSError: Durable persistence failed.
    """
    if history_dir is None or not isinstance(history_dir, (str, Path)):
        raise TypeError("history_dir must be a str or pathlib.Path")
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


# Fixed canonical value per registered request field for trial identity. A field whose recorded value dumps equal to its baseline is omitted from the identity key, so request defaults can change without re-keying recorded trials (I-DEFAULT-DECOUPLED). Append-only: a new field is appended with the value every record lacking the key actually ran under (its introduction default for new behavior; the prior unconditional behavior when a field makes existing behavior optional); an existing entry is never edited or removed (I-ID-STABLE). Values are JSON-native; the retired unset-gross sentinel is represented by its canonical dump string.
TRIAL_IDENTITY_BASELINE: Final[Mapping[str, object]] = MappingProxyType(
    {
        "start": None,
        "end": None,
        "partition": "dev",
        "data_root": None,
        "execution_timeframe": "3m",
        "execution_universe_size": 30,
        "max_rss_bytes": None,
        "log_run": True,
        "touch_diagnostic": False,
        "ladder_diagnostic": False,
        "peg_chase_diagnostic": False,
        "liquidity_cost_model": "flat",
        "passive_timeout_minutes": 30,
        "discovery_gate": False,
        "discovery_gate_adjusted_net_t": False,
        "discovery_gate_regime_scaled_net_t": False,
        "fold_safe_horizon_selection": False,
        "crash_regime_tilt_alpha": None,
        "slow_book_mode": "single_horizon",
        "fast_book_mode": "single_horizon",
        "rebalance_filter": "per_symbol_deadband",
        "beta_neutralize": False,
        "ensemble_signal": "raw",
        "trend_efficiency_overlay": False,
        "pnl_vol_target": True,
        "pnl_vol_target_mode": "median_relative",
        "trend_sleeve": False,
        "trend_sleeve_gross": 0.0,
        "multi_feature_book": False,
        "committee_book": False,
        "committee_kelly_sizing": False,
        "committee_growth_diagnostic": False,
        "committee_capital": False,
        "committee_member_set": "risk_premia",
        "committee_tranche_smoothing": False,
        "committee_regime_adaptive_tranche": False,
        "committee_tranche_count": 3,
        "committee_target_gross": "<builtins.object>",
        "committee_evidence_weighting": False,
        "funding_carry_sleeve": False,
        "funding_carry_weight": 0.0,
        "execution_coverage_gate": False,
        "exposure_scale_two_sided": False,
        "exposure_drawdown_brake": False,
        "name_drift_trim": False,
        "ram_guard": True,
        "growth_envelope": "conservative",
        "committee_member_attribution": False,
        "final_oos_2026h1": False,
        "forward_registration_digest": None,
        "data_policy": "zombie_mask_v1",
        "input_manifest_path": None,
        "forward_execution_quality_dir": None,
        "forward_strategy_digest": None,
        "placebo_diagnostic": True,
        "phase_diagnostic": True,
        "signal_48h_diagnostic": True,
        "bootstrap_ci_diagnostic": True,
        "reference_books_diagnostic": True,
        "patient_reference_diagnostic": True,
    }
)


def _identity_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda o: f"<{type(o).__module__}.{type(o).__qualname__}>",
    )


def _equals_baseline(name: str, value: Any) -> bool:
    """True when ``value`` is the fixed identity baseline of field ``name``; never true for unregistered names."""
    if name not in TRIAL_IDENTITY_BASELINE:
        return False
    return _identity_dump(value) == _identity_dump(TRIAL_IDENTITY_BASELINE[name])


def _sparse_identity_key(key: str) -> str:
    """Re-key one stored identity (dense or sparse) into the sparse baseline form; non-object keys pass through."""
    try:
        parsed = json.loads(key)
    except json.JSONDecodeError:
        return key
    if not isinstance(parsed, dict):
        return key
    for name in list(parsed.keys()):
        if name in TRIAL_IDENTITY_BASELINE and _equals_baseline(name, parsed[name]):
            del parsed[name]
    return _identity_dump(parsed)


def trial_identity_key(record: Mapping[str, Any]) -> str | None:
    """Canonical identity key of one recorded configuration.

    Normalizes the record's ``flags`` against the fixed
    ``TRIAL_IDENTITY_BASELINE`` (missing key or explicit ``None`` -> baseline;
    a ``data_policy``-less record keeps the ``legacy`` policy it ran under),
    drops ``RESEARCH_NEUTRAL_FLAGS``, omits values equal to their baseline,
    retains every other key including unregistered ones (fail-closed against
    new alpha fields), and serializes canonically with the ``params_snapshot``.
    Two records share a trial iff they denote the same decision path.
    Request dataclass defaults are deliberately not consulted: changing them
    must never re-key recorded trials.
    """
    if not isinstance(record, Mapping):
        return None
    flags = record.get("flags")
    flags = flags if isinstance(flags, Mapping) else {}
    normalized: dict[str, Any] = {}
    registered = set(TRIAL_IDENTITY_BASELINE)
    for name in TRIAL_IDENTITY_BASELINE:
        baseline = TRIAL_IDENTITY_BASELINE[name]
        value = flags.get(name, baseline)
        if value is None:
            value = baseline
        if name == "data_policy" and "data_policy" not in flags:
            # Migration: a data_policy-less legacy record keeps its legacy
            # identity instead of adopting the new zombie default.
            value = "legacy"
        if name not in RESEARCH_NEUTRAL_FLAGS and not _equals_baseline(name, value):
            normalized[name] = value
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


def derive_trials_attempted(history_dir: Path | str | None = None) -> tuple[int, str]:
    """Audit-trials denominator for the DSR from the run-history registry.

    ``history_dir=None`` reads the canonical operator registry (read-only default).

    Counts the distinct identity keys of records admitted by ``is_trial_record``
    (the same predicate and key ``window_trial_sharpes`` uses -- I-SAME-TRIAL-SET),
    unioned with the monotone ``trials`` ledger so retention never lowers the
    count (I-MONOTONE-TRIALS). Returns ``(SEARCH_TRIALS_ATTEMPTED + counted,
    source)`` with ``source`` = ``'constant_plus_ledger'`` (ledger non-empty),
    ``'constant_plus_history'`` (history rows but empty ledger) or
    ``'constant_fallback'`` (no registry evidence). O(history rows).

    Raises:
        DataIntegrityError: the registry exists but is unreadable or corrupt;
            an understated N would inflate the DSR, so no fallback is taken.
    """
    registry = _resolve_history_registry(history_dir)
    state = _load_registry_state(registry)
    if state is None:
        return SEARCH_TRIALS_ATTEMPTED, "constant_fallback"
    seen: set[str] = set()
    for record in state.records:
        if not is_trial_record(record):
            continue
        key = trial_identity_key(record)
        if key is not None:
            seen.add(key)
    union = seen | set(state.ledger)
    source = "constant_plus_ledger" if state.ledger else "constant_plus_history"
    return SEARCH_TRIALS_ATTEMPTED + len(union), source


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

    ``history_dir=None`` reads the canonical operator registry (read-only default).

    A registry record qualifies when ``is_trial_record`` admits it, its ``start``
    matches exactly, and its ``resolved_end`` lies within
    ``TRIAL_POOL_WINDOW_TOLERANCE`` of the window end. A re-run of one
    configuration with the same outcome collapses to one entry; distinct
    outcomes of one configuration stay distinct. Returns outcomes ascending;
    ``()`` when the window is unparseable or no registry evidence exists.

    Raises:
        DataIntegrityError: the registry exists but is unreadable or corrupt.
    """
    registry = _resolve_history_registry(history_dir)
    state = _load_registry_state(registry)
    wanted_start = _parse_utc_timestamp(window[0])
    wanted_end = _parse_utc_timestamp(window[1])
    if wanted_start is None or wanted_end is None or state is None:
        return ()
    seen_entries: set[tuple[str, float]] = set()
    outcomes: list[float] = []
    for record in state.records:
        if not is_trial_record(record):
            continue
        blend = record["blend"]
        sharpe = float(blend["primary_naive_sharpe"])  # finite: is_trial_record
        if not _matches_window(record, wanted_start, wanted_end):
            continue
        entry = (cast(str, trial_identity_key(record)), sharpe)
        if entry in seen_entries:
            continue
        seen_entries.add(entry)
        outcomes.append(sharpe)
    return tuple(sorted(outcomes))
