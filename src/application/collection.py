from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from src.common.config import BASE_DIR
from src.market_data.services.borrow_collection import (
    collect_binance_quote_borrow_history,
    import_quote_borrow_history,
)
from src.market_data.services.futures_collection import DataCollector
from src.market_data.services.spot_collection import (
    SpotDataCollector,
    repair_spot_ohlcv_gap,
)

_logger = logging.getLogger(__name__)


def collect_ohlcv(symbol: str, timeframe: str, start: str, end: str) -> None:
    """Incrementally collect futures OHLCV and persist it to the canonical lake."""
    collector = DataCollector()
    collector.ensure_ohlcv_data(symbol, timeframe, start, end)
    _logger.info("Data collection complete for %s %s (through %s)", symbol, timeframe, end)

def collect_funding(symbol: str, start: str, end: str) -> None:
    """Incrementally collect futures funding history and persist it to the canonical lake."""
    collector = DataCollector()
    collector.ensure_funding_data(symbol, start, end)
    _logger.info("Funding collection complete for %s (through %s)", symbol, end)


def collect_metrics(symbol: str, start: str, end: str) -> None:
    """Collect one symbol's daily futures metrics (open interest) into the canonical lake.

    Collects exactly one requested symbol; an orchestration caller must report
    each unavailable symbol explicitly rather than silently reducing the fixed
    universe.
    """
    collector = DataCollector()
    collector.ensure_metrics_data(symbol, start, end)
    _logger.info("Metrics collection complete for %s (through %s)", symbol, end)


def collect_spot_ohlcv(symbol: str, timeframe: str, start: str, end: str) -> None:
    """Incrementally collect spot OHLCV and persist it to the canonical spot lake."""
    SpotDataCollector().ensure_spot_ohlcv(symbol, timeframe, start, end)
    _logger.info("Spot data collection complete for %s %s (through %s)", symbol, timeframe, end)


def import_borrow(symbol: str, source: str | Path, source_id: str, rate_period: str) -> None:
    """Import a versioned historical quote-borrow export into the spot lake."""
    import_quote_borrow_history(symbol, source, source_id, rate_period)
    _logger.info("Borrow history imported for %s from %s", symbol, source)


def collect_borrow(symbol: str, asset: str, start: str, end: str) -> None:
    """Collect signed Binance Margin quote-borrow history into the spot lake."""
    load_dotenv(BASE_DIR / ".env")
    collect_binance_quote_borrow_history(symbol, asset, start, end)
    _logger.info("Binance Margin borrow history collected for %s (%s)", symbol, asset)


def repair_spot_gap(symbol: str, timeframe: str, timestamp: str) -> None:
    """Record one documented synthetic spot bar bridging an isolated gap."""
    repair_spot_ohlcv_gap(symbol, timeframe, timestamp)
    _logger.info("Spot OHLCV gap repaired for %s %s at %s", symbol, timeframe, timestamp)
