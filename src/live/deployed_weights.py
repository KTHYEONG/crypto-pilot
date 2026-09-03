# ruff: noqa
"""Thin weights reader: rolling state file."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from pydantic import SecretStr

from src.common.paths import DATA_DIR
from src.common.errors import DataIntegrityError
from src.live.errors import StaleSignalError, ArtifactSealError


def default_weights_path() -> Path:
    return DATA_DIR / "state" / "deployed_target_weights.parquet"


def _read_parquet(path: Path, artifact_key: SecretStr | None) -> pd.DataFrame:
    if str(path).endswith(".enc"):
        if artifact_key is None:
            raise ArtifactSealError(f"sealed artifact requires a key: {path}")
        from src.live.crypto import derive_key, read_sealed_parquet

        return read_sealed_parquet(path, derive_key(artifact_key))
    return pd.read_parquet(path)


def _write_parquet(frame: pd.DataFrame, path: Path, artifact_key: SecretStr | None) -> None:
    is_enc = str(path).endswith(".enc")
    if artifact_key is not None or is_enc:
        from src.live.crypto import derive_key, seal_bytes

        dest = path if is_enc else Path(f"{path}.enc")
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=True)
        if artifact_key is None:
            raise DataIntegrityError(f"sealed artifact path requires a key: {dest}")
        sealed = seal_bytes(buffer.getvalue(), derive_key(artifact_key))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(sealed)
        os.replace(tmp, dest)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=True)
    os.replace(tmp, path)


def load_weights_frame(path: Path, *, artifact_key: SecretStr | None = None) -> pd.DataFrame:
    p = Path(path)
    candidate: Path | None = None
    if p.exists():
        candidate = p
    elif artifact_key is not None:
        enc = p if str(p).endswith(".enc") else Path(f"{p}.enc")
        if enc.exists():
            candidate = enc
        enc2 = Path(str(p) + ".enc")
        if enc2.exists() and candidate is None:
            candidate = enc2
    if candidate is None:
        return pd.DataFrame()
    try:
        df = _read_parquet(candidate, artifact_key)
    except (ArtifactSealError, DataIntegrityError):
        raise
    except Exception as exc:
        raise DataIntegrityError(f"weights frame corrupt: {candidate}: {exc}") from exc
    if not df.empty:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            raise DataIntegrityError(f"weights frame index must be tz-aware UTC: {candidate}")
        df.index = idx.tz_convert("UTC")
        df = df.sort_index()
    return df


def append_weight_row(path: Path, date: pd.Timestamp, row: pd.Series, *, artifact_key: SecretStr | None = None, keep_rows: int = 500) -> bool:
    p = Path(path)
    frame = load_weights_frame(p, artifact_key=artifact_key)
    dt = pd.Timestamp(date)
    if dt.tzinfo is None:
        raise DataIntegrityError("date must be tz-aware")
    dt = dt.tz_convert("UTC").normalize()
    if not frame.empty:
        idx = pd.DatetimeIndex(frame.index)
        if dt in idx:
            return False
    new_row_df = pd.DataFrame([row], index=pd.DatetimeIndex([dt]))
    if new_row_df.index.tz is None:
        new_row_df.index = new_row_df.index.tz_localize("UTC")
    if frame.empty:
        new_frame = new_row_df
    else:
        all_cols = sorted(set(frame.columns) | set(new_row_df.columns))
        new_frame = pd.concat([frame.reindex(columns=all_cols), new_row_df.reindex(columns=all_cols)])
        new_frame = new_frame.sort_index()
        if new_frame.index.tz is None:
            new_frame.index = new_frame.index.tz_localize("UTC")
    # truncate to trailing keep_rows
    if len(new_frame) > keep_rows:
        new_frame = new_frame.tail(keep_rows)
    _write_parquet(new_frame, p, artifact_key)
    return True


def weights_asof(frame: pd.DataFrame, decision_time: pd.Timestamp, *, max_staleness: pd.Timedelta) -> pd.Series:
    if frame.empty:
        raise DataIntegrityError("weights frame empty")
    dt = pd.Timestamp(decision_time)
    if dt.tzinfo is None:
        raise DataIntegrityError("decision_time must be tz-aware")
    dt = dt.tz_convert("UTC")
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is None:
        raise DataIntegrityError("weights frame index must be tz-aware UTC")
    # ensure sorted
    frame_sorted = frame.sort_index()
    idx_sorted = pd.DatetimeIndex(frame_sorted.index)
    # mask <= dt
    mask = idx_sorted <= dt
    if not mask.any():
        raise DataIntegrityError(f"no weights <= decision_time {dt}")
    # most recent prior
    candidate_idx = idx_sorted[mask][-1]
    row = frame_sorted.loc[candidate_idx]
    ser = pd.Series(row, index=frame_sorted.columns, dtype="float64", name=candidate_idx)
    # staleness check: dt - candidate_idx > max_staleness ?
    if dt - candidate_idx > max_staleness:
        raise StaleSignalError(f"signal for {candidate_idx} is stale: now={dt} max_staleness={max_staleness}")
    return ser
