from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_logger = logging.getLogger("OhlcvStore")

_OHLCV_1M_COLUMNS: tuple[str, ...] = (
    "timestamp", "open", "high", "low", "close", "volume",
    "taker_buy_base_volume", "taker_buy_quote_volume", "quote_vol",
)

_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

_TAKER_CANONICAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("taker_buy_base", "taker_buy_base_volume"),
    ("taker_buy_quote", "taker_buy_quote_volume"),
)

_ROW_GROUP_DAYS: int = 31
_BARS_PER_DAY: dict[str, int] = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "1h": 24}


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


def is_temp_artifact(name: str) -> bool:
    return name.endswith(".tmp.parquet")


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
    else:
        # REST kline은 *_volume 명만, Vision 아카이브는 무접미사 명만 준다(같은 kline 9·10번 필드).
        # Vision 월이 없는 신규 상장 심볼은 무접미사 컬럼이 없어 load_base_panel 요청이 실패하므로
        # 저장 시 비어 있는 무접미사 값만 채운다. 리네임은 두 형식을 모두 가진 파일에서 중복 컬럼을 만든다.
        for canonical, suffixed in _TAKER_CANONICAL_SOURCES:
            if suffixed not in df.columns:
                continue
            df[canonical] = df[canonical].fillna(df[suffixed]) if canonical in df.columns else df[suffixed]

    df_to_save = df.copy()
    if "datetime" in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=["datetime"])
    for col in _PRICE_COLUMNS:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].astype("float32")
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    # 윈도우(31일) 필터 읽기가 row group을 건너뛰도록 — 수집기 재기록 때마다 레이아웃이 단일 그룹으로 되돌아가던 회귀 방지(실측 10.2→3.7ms/읽기).
    bars_per_day = _BARS_PER_DAY.get(timeframe)
    if bars_per_day is None:
        df_to_save.to_parquet(temp_path, index=False, compression="zstd")
    else:
        df_to_save.to_parquet(temp_path, index=False, compression="zstd", row_group_size=_ROW_GROUP_DAYS * bars_per_day)
    temp_path.replace(path)
    _logger.info(
        "write_ohlcv path=%s rows=%d cols=%s", path, len(df_to_save), list(df_to_save.columns),
        extra={"tag": "DATA"},
    )
