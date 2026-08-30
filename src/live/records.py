"""Typed monthly partition storage — evidence-immutable append layer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def monthly_partition_path(
    directory: Path, prefix: str, when: pd.Timestamp, *, suffix: str = ".parquet"
) -> Path:
    ts = pd.Timestamp(when)
    if ts.tzinfo is None:
        raise ValueError("when must be tz-aware UTC")
    ts_utc = ts.tz_convert("UTC")
    yyyymm = ts_utc.strftime("%Y%m")
    return Path(directory) / f"{prefix}_{yyyymm}{suffix}"


def _enforce_dtypes(df: pd.DataFrame, dtypes: Mapping[str, str]) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        if "datetime64" in dtype:
            df[col] = pd.to_datetime(df[col], utc=True).astype(dtype)
        elif dtype in ("float64", "float32", "float"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif dtype.startswith("int"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
        elif dtype == "bool":
            df[col] = df[col].astype(bool)
        else:
            df[col] = df[col].astype(dtype)
    return df


def append_typed_frame(
    frame: pd.DataFrame,
    directory: Path,
    prefix: str,
    *,
    time_column: str,
    dtypes: Mapping[str, str],
    compression: str = "zstd",
) -> list[Path]:
    if frame.empty:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # ensure time_column is datetime
    frame = frame.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    # enforce dtypes
    frame = _enforce_dtypes(frame, dtypes)
    # group by YYYYMM of time_column UTC
    frame["_yyyymm"] = frame[time_column].dt.tz_convert("UTC").dt.strftime("%Y%m")
    written: list[Path] = []
    for yyyymm, group in frame.groupby("_yyyymm"):
        grp = group.drop(columns=["_yyyymm"])
        path = directory / f"{prefix}_{yyyymm}.parquet"
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, grp], ignore_index=True)
                combined = _enforce_dtypes(combined, dtypes)
                # ensure time column remains datetime
                if time_column in combined.columns:
                    combined[time_column] = pd.to_datetime(combined[time_column], utc=True).astype("datetime64[ns, UTC]")
                combined.to_parquet(path, index=False, compression=compression)
            except Exception:  # noqa: BLE001 - 손상된 기존 파티션은 신규 그룹으로 덮어써 append 지속
                grp.to_parquet(path, index=False, compression=compression)
        else:
            grp.to_parquet(path, index=False, compression=compression)
        written.append(path)
    return sorted(written)


def load_partitions(
    directory: Path,
    prefix: str,
    *,
    since: pd.Timestamp | None = None,
    suffix: str = ".parquet",
) -> pd.DataFrame:
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame()
    pattern = f"{prefix}_*{suffix}"
    shards = sorted(directory.glob(pattern))
    if not shards:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            df = pd.read_parquet(shard)
            if not df.empty:
                frames.append(df)
        except Exception:  # noqa: S112 - 읽을 수 없는 shard는 건너뛰고 나머지 로드
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if since is not None:
        try:
            since_ts = pd.Timestamp(since)
            since_ts = since_ts.tz_localize("UTC") if since_ts.tzinfo is None else since_ts.tz_convert("UTC")
            # find datetime column to filter
            dt_col: str | None = None
            for col in combined.columns:
                if pd.api.types.is_datetime64_any_dtype(combined[col]):
                    dt_col = col
                    break
                # try parse as datetime
                try:
                    parsed = pd.to_datetime(combined[col], utc=True, errors="coerce")
                    if parsed.notna().any():
                        dt_col = col
                        combined[col] = parsed
                        break
                except Exception:  # noqa: S112 - datetime 파싱 불가 컬럼은 후보에서 제외
                    continue
            if dt_col is not None:
                mask = pd.to_datetime(combined[dt_col], utc=True, errors="coerce") >= since_ts
                combined = combined[mask]
        except Exception:  # noqa: S110, BLE001 - since 필터 실패 시 원본 결합 결과를 그대로 반환
            pass
    return combined


def append_jsonl_partition(
    rows: Sequence[Mapping[str, Any]],
    directory: Path,
    prefix: str,
    *,
    time_key: str,
) -> list[Path]:
    if not rows:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        ts = pd.Timestamp(r[time_key])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        yyyymm = ts.strftime("%Y%m")
        buckets.setdefault(yyyymm, []).append(r)
    written: list[Path] = []
    for yyyymm in sorted(buckets.keys()):
        path = directory / f"{prefix}_{yyyymm}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for r in buckets[yyyymm]:
                # ensure time_key is isoformat for json
                rec = dict(r)
                val = rec.get(time_key)
                if isinstance(val, pd.Timestamp):
                    rec[time_key] = val.isoformat()
                # handle pd NaT or Timestamp
                line = json.dumps(rec, ensure_ascii=False, default=str)
                f.write(line + "\n")
        written.append(path)
    return sorted(written)
