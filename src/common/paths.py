from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_ROOT: Path = BASE_DIR
"""Repository root. Live deployment defaults (frozen unit history, venue fallback) anchor here."""
DATA_DIR = BASE_DIR / "data"
FUTURES_DATA_DIR = DATA_DIR / "futures"
SPOT_DATA_DIR = DATA_DIR / "spot"
DEPLOY_MHS_DIR: Path = BASE_DIR / "deploy" / "mhs"
"""Repository delivery boundary for sealed MHS runtime artifacts."""
BACKTESTS_DIR: Path = DATA_DIR / "backtests"
FROZEN_BACKTESTS_DIR: Path = BACKTESTS_DIR / "frozen" / "runs"
VENUE_RULES_DIR: Path = FUTURES_DATA_DIR / "venue_rules"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("_", "")


def ohlcv_path(symbol: str, timeframe: str) -> Path:
    return FUTURES_DATA_DIR / "ohlcv" / timeframe / f"{_safe_symbol(symbol)}.parquet"


def funding_path(symbol: str) -> Path:
    return FUTURES_DATA_DIR / "funding" / f"{_safe_symbol(symbol)}.parquet"


def metrics_path(symbol: str) -> Path:
    return FUTURES_DATA_DIR / "metrics" / "1d" / f"{_safe_symbol(symbol)}.parquet"


def indicator_kline_path(dataset: str, symbol: str, timeframe: str) -> Path:
    safe = _safe_symbol(symbol)
    return FUTURES_DATA_DIR / dataset / timeframe / f"{safe}.parquet"


def bookdepth_path(symbol: str) -> Path:
    safe = _safe_symbol(symbol)
    return FUTURES_DATA_DIR / "bookdepth" / f"{safe}.parquet"


def spot_ohlcv_path(symbol: str, timeframe: str) -> Path:
    safe = _safe_symbol(symbol)
    return SPOT_DATA_DIR / "ohlcv" / timeframe / f"{safe}.parquet"


def borrow_path(symbol: str) -> Path:
    safe = _safe_symbol(symbol)
    return SPOT_DATA_DIR / "borrow" / f"{safe}.parquet"
