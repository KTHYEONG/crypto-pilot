"""Observational disclosure of how the DSR trial pool was assembled from the run-history registry."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.lab.mhs.run_history import (
    RESEARCH_NEUTRAL_FLAGS,
    _carries_data_integrity_code,
    _has_finite_blend_sharpe,
    _load_registry_state,
    _matches_window,
    _parse_utc_timestamp,
    _resolve_history_registry,
    trial_identity_key,
)

# Shares the run-history logger so registry warnings keep their established routing.
logger = logging.getLogger("MhsRunHistory")


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


class _DisclosureBuckets:
    """Single-pass bucket accounting behind ``trial_pool_disclosure`` (I-DISCLOSURE)."""

    __slots__ = ("matched_ends", "matched_keys", "payload")

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {**_EMPTY_DISCLOSURE}
        self.matched_keys: set[str] = set()
        self.matched_ends: list[pd.Timestamp] = []

    def add(self, record: Mapping[str, Any], wanted_start: pd.Timestamp, wanted_end: pd.Timestamp) -> None:
        payload = self.payload
        payload["n_history_records"] += 1
        flags = record.get("flags")
        if isinstance(flags, Mapping):
            payload["neutral_flags_dropped"] += sum(
                1 for name in flags if name in RESEARCH_NEUTRAL_FLAGS
            )
        if record.get("status") != "COMPLETE":
            payload["excluded_not_complete"] += 1
            return
        if _carries_data_integrity_code(record):
            payload["excluded_data_integrity"] += 1
            return
        if not _has_finite_blend_sharpe(record):
            payload["excluded_nonfinite_blend"] += 1
            return
        if _matches_window(record, wanted_start, wanted_end):
            resolved_end = _parse_utc_timestamp(record.get("resolved_end"))
            if resolved_end is not None:
                self.matched_ends.append(resolved_end)
            key = trial_identity_key(record)
            if key is not None:
                self.matched_keys.add(key)
        payload["n_trial_records"] += 1

    def finalize(self, ledger_size: int) -> dict[str, Any]:
        payload = self.payload
        payload["distinct_trial_keys"] = len(self.matched_keys)
        if len(self.matched_ends) >= 2:
            span_seconds = (max(self.matched_ends) - min(self.matched_ends)).total_seconds()
            payload["pool_window_span_days"] = float(span_seconds / 86400.0)
        payload["ledger_size"] = ledger_size
        if ledger_size:
            payload["source"] = "constant_plus_ledger"
        elif payload["n_history_records"] > 0:
            payload["source"] = "constant_plus_history"
        return payload


def _disclosure_over(
    records: Iterable[Mapping[str, Any]],
    wanted_start: pd.Timestamp,
    wanted_end: pd.Timestamp,
    *,
    ledger_size: int,
) -> dict[str, Any]:
    buckets = _DisclosureBuckets()
    for record in records:
        buckets.add(record, wanted_start, wanted_end)
    return buckets.finalize(ledger_size)


def trial_pool_disclosure(
    window: tuple[str, str], history_dir: Path | str | None = None
) -> dict[str, Any]:
    """Observational disclosure of how the DSR trial pool was assembled.

    ``history_dir=None`` reads the canonical operator registry (read-only default).

    Every registry record falls into exactly one bucket -- admitted trial, or
    excluded by exactly one registered ground (incomplete status /
    data-integrity code / non-finite blend Sharpe) -- so the bucket counts sum
    to ``n_history_records``. ``distinct_trial_keys`` and
    ``pool_window_span_days`` describe the tolerance-merged pool matched for
    ``window``. Emits no GO reason code; on corrupt or absent registry evidence
    it returns zeros with ``source='constant_fallback'`` (corruption is logged).
    """
    registry = _resolve_history_registry(history_dir)
    wanted_start = _parse_utc_timestamp(window[0])
    wanted_end = _parse_utc_timestamp(window[1])
    if wanted_start is None or wanted_end is None:
        return {**_EMPTY_DISCLOSURE}
    try:
        state = _load_registry_state(registry)
    except DataIntegrityError as exc:
        logger.warning("[DATA] trial_pool_disclosure registry_unreadable registry=%s error=%s", registry, exc)
        return {**_EMPTY_DISCLOSURE}
    if state is None:
        return {**_EMPTY_DISCLOSURE}
    return _disclosure_over(state.records, wanted_start, wanted_end, ledger_size=len(state.ledger))
