from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.data.collector import DataCollector

_logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OHLCV data from Binance")
    parser.add_argument("symbol", type=str, help="Symbol (e.g. BTCUSDT)")
    parser.add_argument("timeframe", type=str, help="Timeframe (e.g. 1h)")
    parser.add_argument("--start", default="2022-04-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: now)")
    args = parser.parse_args()

    # ensure_ohlcv_data does pd.to_datetime(end_date) internally; an empty
    # string parses to NaT, which makes every range comparison False and the
    # whole fetch silently no-ops. Default explicitly to "now" instead.
    end = args.end or str(pd.Timestamp.now(tz="UTC"))

    collector = DataCollector()
    collector.ensure_ohlcv_data(args.symbol, args.timeframe, args.start, end)
    _logger.info("Data collection complete for %s %s (through %s)", args.symbol, args.timeframe, end)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
