from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FUTURES_DATA_DIR = DATA_DIR / "futures"


def ohlcv_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_")
    return FUTURES_DATA_DIR / "ohlcv" / timeframe / f"{safe}.parquet"


def funding_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_")
    return FUTURES_DATA_DIR / "funding" / f"{safe}.parquet"
