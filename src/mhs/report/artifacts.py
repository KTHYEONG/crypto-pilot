"""Replay artifact references and readers for the MHS diagnostic report.

Holds the artifact-table checksum/reference helpers, the unified-file replay
reference builder, and ``load_mhs_replay_artifact`` (the reader contract that
consumes the unified Parquet artifact tables written by ``persist``).
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.execution import StrategyExecutionReplayResult
from src.mhs.params import ARTIFACT_SCHEMA_VERSION


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        if not value.index.is_unique:
            return _jsonable(value.to_dict(orient="records"))
        return {
            str(k): _jsonable(v)
            for k, v in value.astype(object).to_dict(orient="index").items()
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _to_timestamped_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Promote a DatetimeIndex into an explicit UTC timestamp column.

    The returned table carries a physical ``datetime64[ns, UTC]`` column named
    ``timestamp`` and a RangeIndex, so readers need no string-parsing guess.
    """
    out = frame.copy()
    if len(out):
        out.insert(0, "timestamp", out.index)
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    else:
        out["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    return out.reset_index(drop=True)


def _artifact_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_reference(table: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Row count, time bounds, schema version, and content checksum per table."""
    ts = (
        pd.to_datetime(table["timestamp"], utc=True)
        if "timestamp" in table.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    return {
        "file": path.name,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "row_count": len(table),
        "time_bounds": {
            "start": None if len(ts) == 0 else str(ts.iloc[0]),
            "end": None if len(ts) == 0 else str(ts.iloc[-1]),
        },
        "checksum_sha256": _artifact_checksum(path),
    }


def _verify_ledger_artifact(path: Path, replay_id: str, expected_rows: int) -> None:
    """Re-read a written unified ledger parquet and verify one replay partition fail-closed.

    Checksum-only provenance cannot detect a silently-written NULL or truncated
    equity column, so this pass re-reads the ``replay_id`` partition via PyArrow
    pushdown filtering and asserts the exact row count and a fully finite
    positive equity column (spec §3.3, ``fold_integrity``).
    """
    roundtrip = pd.read_parquet(path, filters=[("replay_id", "==", replay_id)])
    if len(roundtrip) != expected_rows:
        raise DataIntegrityError(
            f"ledger artifact row count mismatch path={path} replay_id={replay_id} "
            f"expected={expected_rows} got={len(roundtrip)}"
        )
    if expected_rows and "equity" in roundtrip.columns:
        equity = roundtrip["equity"].to_numpy(dtype="float64")
        if not np.isfinite(equity).all() or (equity <= 0).any():
            raise DataIntegrityError(
                f"ledger artifact equity must be finite and strictly positive "
                f"path={path} replay_id={replay_id}"
            )


def _build_replay_artifact_reference(
    replay_id: str,
    replay: StrategyExecutionReplayResult,
    tables: dict[str, pd.DataFrame],
    artifact_root: Path,
    unified_tables: dict[str, tuple[Path, pd.DataFrame]],
) -> dict[str, Any]:
    """Unified-file reference for one replay: per-category row/time provenance and
    the shared unified file checksum, so readers load via ``load_mhs_replay_artifact``."""
    return {
        "artifact_format": "parquet",
        "artifact_dir": str(artifact_root),
        "replay_id": replay_id,
        "fills": _artifact_reference(tables["fills"], unified_tables["fills"][0]),
        "units": _artifact_reference(tables["units"], unified_tables["units"][0]),
        "notional_weights": _artifact_reference(
            tables["notional_weights"], unified_tables["notional_weights"][0]
        ),
        "ledger": _artifact_reference(tables["ledger"], unified_tables["ledger"][0]),
        "times": _artifact_reference(tables["times"], unified_tables["times"][0]),
        "fill_source": replay.fill_source,
        "mark_source": replay.mark_source,
        "event_snapshots_retained": replay.event_snapshots_retained,
        "fill_count": replay.fill_count,
        "unfilled_count": replay.unfilled_count,
        "fallback_count": replay.fallback_count,
        "all_intent_shortfall_bps": replay.all_intent_shortfall_bps,
        "forced_exit_count": replay.forced_exit_count,
        "forced_exit_notional": replay.forced_exit_notional,
        "termination_counts": dict(replay.termination_counts),
        "unsupported_assumptions": list(replay.unsupported_assumptions),
        "elapsed_seconds": replay.elapsed_seconds,
        "data_gaps": [
            {
                "code": gap.code,
                "symbol": gap.symbol,
                "timestamp": _jsonable(gap.timestamp),
                "decision_time": _jsonable(gap.decision_time),
                "signal_time": _jsonable(gap.signal_time),
                "execution_bound": gap.execution_bound,
            }
            for gap in replay.data_gaps
        ],
    }


def load_mhs_replay_artifact(
    artifact_root: str | Path,
    replay_id: str,
    category: Literal["fills", "units", "notional_weights", "ledger", "times"],
) -> pd.DataFrame:
    """Load a specific replay's artifact table using PyArrow pushdown filtering."""
    path = Path(artifact_root) / f"{category}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Unified artifact table missing: {path}")
    return pd.read_parquet(path, filters=[("replay_id", "==", replay_id)]).drop(
        columns=["replay_id"]
    )
