from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.core.settings import FUTURES_DATA_DIR

_logger = logging.getLogger(__name__)


def resolve_cached_symbols(
    timeframe: str = "1h",
    base_dir: Path | None = None,
) -> tuple[str, ...]:
    data_dir = base_dir or FUTURES_DATA_DIR
    ohlcv_dir = data_dir / "ohlcv" / timeframe
    if not ohlcv_dir.exists():
        return ()
    symbols: list[str] = []
    for p in sorted(ohlcv_dir.glob("*.parquet")):
        stem = p.stem
        symbols.append(stem)
    return tuple(symbols)


def load_hourly_data(
    config: CompoundRunConfig,
    symbols: tuple[str, ...],
    ref_dt: datetime | None = None,
    *,
    bars: int = 2048,
) -> dict[str, dict[str, pd.DataFrame]]:
    ref = ref_dt or datetime.now(UTC)
    end_dt = ref
    start_dt = pd.Timestamp(end_dt) - pd.Timedelta(hours=bars)
    data_maps: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        path = FUTURES_DATA_DIR / "ohlcv" / "1h" / f"{sym}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if "datetime" not in df.columns:
                    if "timestamp" in df.columns:
                        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    else:
                        raise ValueError("missing datetime/timestamp column")
                elif not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                if "quote_volume" not in df.columns and "quote_vol" in df.columns:
                    df["quote_volume"] = df["quote_vol"]
                if "taker_buy_quote" not in df.columns and "taker_buy_quote_volume" in df.columns:
                    df["taker_buy_quote"] = df["taker_buy_quote_volume"]
                df = df.sort_values("datetime").reset_index(drop=True)
                mask = (df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)
                data_maps[sym] = {"1h": df.loc[mask].copy()}
            except Exception as exc:
                _logger.warning("failed to load data for symbol=%s: %s", sym, exc)
                data_maps[sym] = {}
        else:
            data_maps[sym] = {}
    return data_maps


def check_data_readiness(
    data_maps: dict[str, dict[str, pd.DataFrame]],
    *,
    min_ready_pct: float = 0.80,
) -> bool:
    if not data_maps:
        return False
    ready = sum(1 for sym_data in data_maps.values() if "1h" in sym_data and not sym_data["1h"].empty)
    ratio = ready / max(len(data_maps), 1)
    if ratio < min_ready_pct:
        _logger.error(
            "data readiness %.1f%% < %.1f%% (%d/%d symbols ready)",
            ratio * 100, min_ready_pct * 100, ready, len(data_maps),
        )
        return False
    return True
