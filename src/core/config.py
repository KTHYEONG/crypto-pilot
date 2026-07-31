from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FUTURES_DATA_DIR = DATA_DIR / "futures"
SPOT_DATA_DIR = DATA_DIR / "spot"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("_", "")


def ohlcv_path(symbol: str, timeframe: str) -> Path:
    return FUTURES_DATA_DIR / "ohlcv" / timeframe / f"{_safe_symbol(symbol)}.parquet"


def funding_path(symbol: str) -> Path:
    return FUTURES_DATA_DIR / "funding" / f"{_safe_symbol(symbol)}.parquet"


def spot_ohlcv_path(symbol: str, timeframe: str) -> Path:
    safe = _safe_symbol(symbol)
    return SPOT_DATA_DIR / "ohlcv" / timeframe / f"{safe}.parquet"


def borrow_path(symbol: str) -> Path:
    safe = _safe_symbol(symbol)
    return SPOT_DATA_DIR / "borrow" / f"{safe}.parquet"
