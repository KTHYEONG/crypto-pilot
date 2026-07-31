from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_logger = logging.getLogger("OhlcvStore")

_OHLCV_1M_COLUMNS: tuple[str, ...] = (
    "timestamp", "open", "high", "low", "close", "volume",
    "taker_buy_base_volume", "taker_buy_quote_volume", "quote_vol",
)

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw kline frame into the shared UTC/numeric representation.

    Mirrors the historical futures cache normalisation so migrated files stay
    row-equivalent: a numeric ``timestamp`` yields a UTC ``datetime`` column,
    an existing ``datetime`` is coerced to tz-aware UTC, and object columns
    that parse numerically are converted in place. ``df`` is not mutated.
    """
    if df.empty:
        return df
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    elif "datetime" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        elif getattr(df["datetime"].dtype, "tz", None) is None:
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
        else:
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert("UTC")
    for col in df.columns:
        if col == "datetime":
            continue
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
    return df


def merge_ohlcv_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate raw kline frames, de-duplicate by ``timestamp``, sort by ms."""
    parts = [normalize_frame(f) for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined["timestamp"] = pd.to_numeric(combined["timestamp"], errors="coerce")
    return (
        combined.dropna(subset=["timestamp"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def write_ohlcv(path: Path, df: pd.DataFrame, *, timeframe: str) -> None:
    """Atomically persist a canonical kline frame to ``path`` (zstd Parquet).

    The 1m layout keeps the historical column order and adds missing columns as
    NaN; any other timeframe preserves the supplied column order. The ``datetime``
    helper column and OHLC are stored as integer millisecond ``timestamp`` plus
    float32 OHLC, byte-equivalent with the canonical futures lake. Writes to a
    temporary sibling and replaces atomically; an empty frame is a no-op.
    """
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_frame(df)
    if timeframe == "1m":
        df = df.rename(columns={
            "quote_volume": "quote_vol",
            "taker_buy_base": "taker_buy_base_volume",
            "taker_buy_quote": "taker_buy_quote_volume",
        })
        for column in _OHLCV_1M_COLUMNS:
            if column not in df.columns:
                df[column] = float("nan")
        df = df[list(_OHLCV_1M_COLUMNS)]
        for column in _OHLCV_1M_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = (
            df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
        )

    df_to_save = df.copy()
    if "datetime" in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=["datetime"])
    for col in _PRICE_COLUMNS:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].astype("float32")
    temp_path = path.with_suffix(".tmp.parquet")
    df_to_save.to_parquet(temp_path, index=False, compression="zstd")
    temp_path.replace(path)
    _logger.info(
        "write_ohlcv path=%s rows=%d cols=%s", path, len(df_to_save), list(df_to_save.columns),
        extra={"tag": "DATA"},
    )
