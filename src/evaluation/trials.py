"""Append-only trial ledger counting every evaluated configuration.

A run that fails still counts: its record carries ``daily_sharpe=None`` and
increments ``n_trials``. Records are never deleted or edited by code.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.evaluation.statistics import sharpe_sampling_variance


@dataclass(frozen=True, slots=True)
class TrialRecord:
    family: str
    spec_digest: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    daily_sharpe: float | None
    n_obs: int
    recorded_at: pd.Timestamp
    source: str


@dataclass(frozen=True, slots=True)
class TrialPopulation:
    family: str
    prior_trials: int
    sharpes: tuple[float, ...]
    n_trials: int
    sharpe_variance: float


def _require_utc(ts: pd.Timestamp, label: str) -> pd.Timestamp:
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        raise DataIntegrityError(f"{label} must be a valid timestamp")
    if ts.tzinfo is None or ts.utcoffset() is None or ts.utcoffset().total_seconds() != 0:
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return ts.tz_convert("UTC")


def _record_payload(record: TrialRecord) -> dict[str, object]:
    _require_utc(record.window_start, "window_start")
    _require_utc(record.window_end, "window_end")
    _require_utc(record.recorded_at, "recorded_at")
    if not isinstance(record.family, str) or not record.family:
        raise DataIntegrityError("family must be a non-empty string")
    if not isinstance(record.spec_digest, str) or not record.spec_digest:
        raise DataIntegrityError("spec_digest must be a non-empty string")
    if not isinstance(record.source, str) or not record.source:
        raise DataIntegrityError("source must be a non-empty string")
    if record.window_end <= record.window_start:
        raise DataIntegrityError("window_end must be after window_start")
    if isinstance(record.n_obs, bool) or not isinstance(record.n_obs, int) or record.n_obs < 0:
        raise DataIntegrityError("n_obs must be a non-negative integer")
    sharpe = record.daily_sharpe
    if sharpe is not None and (isinstance(sharpe, bool) or not math.isfinite(float(sharpe))):
        raise DataIntegrityError("daily_sharpe must be finite or None")
    return {
        "family": record.family,
        "spec_digest": record.spec_digest,
        "window_start": record.window_start.isoformat(),
        "window_end": record.window_end.isoformat(),
        "daily_sharpe": None if sharpe is None else float(sharpe),
        "n_obs": int(record.n_obs),
        "recorded_at": record.recorded_at.isoformat(),
        "source": record.source,
    }


def append_trial(record: TrialRecord, *, path: Path) -> bool:
    """Append one trial record; idempotent by ``(spec_digest, window_start, window_end)``."""
    payload = _record_payload(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    key = (payload["spec_digest"], payload["window_start"], payload["window_end"])
    with target.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"trial ledger corrupt: {target}") from exc
            if not isinstance(row, dict):
                raise DataIntegrityError(f"trial ledger corrupt: {target}")
            if (row.get("spec_digest"), row.get("window_start"), row.get("window_end")) == key:
                return False
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _parse_row(row: dict[str, object], path: Path) -> tuple[str, float | None]:
    try:
        digest = str(row["spec_digest"])
        raw_sharpe = row.get("daily_sharpe")
        sharpe = None if raw_sharpe is None else float(raw_sharpe)  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise DataIntegrityError(f"trial ledger corrupt: {path}") from exc
    if sharpe is not None and not math.isfinite(sharpe):
        raise DataIntegrityError(f"trial ledger corrupt: {path}")
    return digest, sharpe


def _invalidated_records(path: Path) -> set[str]:
    journal = path.with_name(path.name + ".invalidations.jsonl")
    if not journal.exists():
        return set()
    try:
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {str(row["record_sha256"]) for row in rows}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataIntegrityError(f"trial invalidation journal corrupt: {journal}") from exc


def trial_record_digest(row: dict[str, object]) -> str:
    """Identify one immutable record independently of JSON spacing and key order."""
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def trial_population(
    family: str,
    *,
    path: Path,
    candidate_sharpe: float = 0.0,
    candidate_n_obs: int = 2,
    candidate_skew: float = 0.0,
    candidate_kurtosis: float = 3.0,
) -> TrialPopulation:
    """Load the trial population for one family.

    ``n_trials`` is prior trials plus distinct recorded digests. The variance
    is the cross-trial variance of recorded daily Sharpes when at least two
    exist, floored by ``sharpe_sampling_variance`` of the candidate so DSR is
    always computable and never more lenient than sampling noise alone.
    """
    if not isinstance(family, str) or not family:
        raise DataIntegrityError("family must be a non-empty string")
    target = Path(path)
    digests: set[str] = set()
    sharpes: list[float] = []
    invalidated = _invalidated_records(target)
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"trial ledger corrupt: {target}") from exc
            if not isinstance(row, dict) or row.get("family") != family:
                continue
            if trial_record_digest(row) in invalidated:
                continue
            digest, sharpe = _parse_row(row, target)
            if row.get("source") == "documented_prior_search":
                continue
            fresh = digest not in digests
            digests.add(digest)
            if fresh and sharpe is not None:
                sharpes.append(sharpe)
    prior = 96 if family == "flow_mom" else 0
    floor = sharpe_sampling_variance(
        float(candidate_sharpe),
        int(candidate_n_obs),
        float(candidate_skew),
        float(candidate_kurtosis),
    )
    if len(sharpes) >= 2:
        cross = float(np.var(np.asarray(sharpes, dtype="float64"), ddof=1))
        variance = max(cross, floor)
    else:
        variance = floor
    return TrialPopulation(
        family=family,
        prior_trials=prior,
        sharpes=tuple(sharpes),
        n_trials=int(prior + len(digests)),
        sharpe_variance=float(variance),
    )
