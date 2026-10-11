"""Top-ADV depth candidate sidecar for execution-window capture."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.common.durable_io import durable_write_text

DEPTH_CANDIDATES_NAME: str = "depth_candidates.json"

_logger = logging.getLogger(__name__)


def _normalize_day(decision_day: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(decision_day)
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    return day.normalize()


def write_depth_candidates(path: Path, decision_day: pd.Timestamp, ranked_symbols: Sequence[str]) -> None:
    """Persist ranked candidate symbols for a decision day."""
    day = _normalize_day(decision_day)
    payload = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "decision_day": day.isoformat(),
        "symbols": [str(symbol) for symbol in ranked_symbols],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    durable_write_text(Path(path), json.dumps(payload, indent=2, sort_keys=True))


def read_depth_candidates(path: Path, decision_day: pd.Timestamp, *, limit: int) -> tuple[str, ...]:
    """Return the first ``limit`` sidecar symbols for ``decision_day``."""
    if limit <= 0:
        return ()
    try:
        raw = Path(path).read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        stamp = payload["decision_day"]
        if not isinstance(stamp, str) or pd.isna(pd.Timestamp(stamp)):
            raise ValueError("decision_day must be a timestamp string")
        stored = _normalize_day(pd.Timestamp(stamp))
        if stored != _normalize_day(decision_day):
            _logger.info("[DATA] stage=exec_depth_candidates status=stale")
            return ()
        symbols = payload["symbols"]
        if not isinstance(symbols, list) or any(not isinstance(s, str) for s in symbols):
            raise ValueError("symbols must be a list of strings")
        return tuple(symbols[: int(limit)])
    except FileNotFoundError:
        _logger.info("[DATA] stage=exec_depth_candidates status=missing")
        return ()
    except Exception:
        _logger.info("[DATA] stage=exec_depth_candidates status=malformed")
        return ()
