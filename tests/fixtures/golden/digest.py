"""Compact golden-fixture builders: sha256 digest + row-count summary.

The monolithic payload golden (1.09 GB) exceeded GitHub's 100 MB limit and was
deleted, leaving the identity gate inert. Two small artifacts replace it with
no loss of strength:

* ``build_report_digest`` — per replay per ledger series ``{n, sha256,
  landmarks}``. SHA-256 over ``index.asi8.astype('int64').tobytes()`` followed
  by ``values.to_numpy(dtype='float64').tobytes()`` is bit-equality by
  construction; the current repr() comparison round-trips float64 exactly, so
  this is no weaker. Landmark indices are
  ``np.unique(np.linspace(0, n-1, min(landmarks, n)).astype(int))``.
* ``build_report_summary`` — the payload with every
  ``StrategyExecutionReplayResult`` field replaced by its
  ``{category: {'row_count': int}}`` stub (the exact shape ``_wire_compact_refs``
  produces), human-diffable in a PR.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.mhs.execution import SimulatedInventoryLedgerResult

#: The only Series-valued state on a replay result that feeds every scalar
#: metric and Research-GO decision: the six streamed-inventory ledger columns.
LEDGER_SERIES: tuple[str, ...] = (
    "equity",
    "net_returns",
    "mark_to_market_pnl",
    "funding_charge",
    "fee_charge",
    "fill_turnover",
)


def _series_digest(series: pd.Series, *, landmarks: int) -> dict[str, Any]:
    values = series.to_numpy(dtype="float64")
    digest = hashlib.sha256()
    digest.update(series.index.asi8.astype("int64").tobytes())
    digest.update(values.tobytes())
    n = len(values)
    landmark_idx = np.unique(np.linspace(0, n - 1, min(landmarks, n)).astype(int))
    return {
        "n": int(n),
        "sha256": digest.hexdigest(),
        "landmarks": {int(i): repr(float(values[i])) for i in landmark_idx},
    }


def build_report_digest(
    report: Any,
    *,
    landmarks: int = 64,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{replay_id: {series_name: {n, sha256, landmarks}}}`` for every replay.

    Covers every replay reachable from the report INCLUDING touch and ladder.
    """
    from src.mhs.report.persist import _collect_replay_entries

    digest: dict[str, dict[str, dict[str, Any]]] = {}
    for replay_id, replay in _collect_replay_entries(report):
        ledger: SimulatedInventoryLedgerResult = replay.ledger
        digest[replay_id] = {
            name: _series_digest(getattr(ledger, name), landmarks=landmarks)
            for name in LEDGER_SERIES
        }
    return digest


def build_report_summary(report: Any) -> dict[str, Any]:
    """Row-count-stubbed payload of ``report`` (<= ~100 KB for the golden shape)."""
    from src.mhs.report.persist import (
        _collect_replay_entries,
        _replay_category_row_counts,
        _stubbed_report_for_payload,
        _wire_compact_refs,
    )

    entries = _collect_replay_entries(report)
    row_counts = {
        replay_id: _replay_category_row_counts(replay) for replay_id, replay in entries
    }
    payload = _stubbed_report_for_payload(report, row_counts).to_payload()
    _wire_compact_refs(payload, report, entries, row_counts)
    payload["replay_ids"] = [replay_id for replay_id, _ in entries]
    return payload


def first_divergent_index(
    golden_series: Mapping[str, Any],
    actual_series: pd.Series,
) -> tuple[int | None, str | None, str | None]:
    """Locate the first divergent index against stored landmarks.

    Returns ``(index, golden_repr, actual_repr)``, or ``(None, None, None)``
    when every landmark matches (the divergence sits between landmarks).
    """
    values = actual_series.to_numpy(dtype="float64")
    for idx_str, g_repr in sorted(
        golden_series["landmarks"].items(), key=lambda kv: int(kv[0]),
    ):
        idx = int(idx_str)
        if idx >= len(values):
            break
        a_repr = repr(float(values[idx]))
        if a_repr != g_repr:
            return idx, g_repr, a_repr
    return None, None, None
