from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.application import collection

_logger = logging.getLogger(__name__)


def _ohlcv(args: argparse.Namespace) -> None:
    # ensure_ohlcv_data does pd.to_datetime(end_date) internally; an empty
    # string parses to NaT, which makes every range comparison False and the
    # whole fetch silently no-ops. Default explicitly to "now" instead.
    end = args.end or str(pd.Timestamp.now(tz="UTC"))
    collection.collect_ohlcv(args.symbol, args.timeframe, args.start, end)
    _logger.info("Data collection complete for %s %s (through %s)", args.symbol, args.timeframe, end)


def _spot_ohlcv(args: argparse.Namespace) -> None:
    end = args.end or str(pd.Timestamp.now(tz="UTC"))
    collection.collect_spot_ohlcv(args.symbol, args.timeframe, args.start, end)
    _logger.info("Spot data collection complete for %s %s (through %s)", args.symbol, args.timeframe, end)

def _funding(args: argparse.Namespace) -> None:
    collection.collect_funding(args.symbol, args.start, args.end)
    _logger.info("Funding collection complete for %s (through %s)", args.symbol, args.end)


def _import_borrow(args: argparse.Namespace) -> None:
    collection.import_borrow(args.symbol, args.source, args.source_id, args.rate_period)
    _logger.info("Borrow history imported for %s from %s", args.symbol, args.source)


def _collect_borrow(args: argparse.Namespace) -> None:
    collection.collect_borrow(args.symbol, args.asset, args.start, args.end)
    _logger.info(
        "Binance Margin borrow history collected for %s (%s)", args.symbol, args.asset,
    )


def _repair_spot_gap(args: argparse.Namespace) -> None:
    collection.repair_spot_gap(args.symbol, args.timeframe, args.timestamp)
    _logger.info(
        "Spot OHLCV gap repaired for %s %s at %s",
        args.symbol, args.timeframe, args.timestamp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect market data from Binance")
    sub = parser.add_subparsers(dest="command", required=True)

    ohlcv_p = sub.add_parser("ohlcv", help="Collect futures OHLCV")
    ohlcv_p.add_argument("symbol", type=str)
    ohlcv_p.add_argument("timeframe", type=str)
    ohlcv_p.add_argument("--start", default="2022-04-01")
    ohlcv_p.add_argument("--end", default=None)
    ohlcv_p.set_defaults(func=_ohlcv)

    spot_p = sub.add_parser("spot-ohlcv", help="Collect independent spot OHLCV")
    spot_p.add_argument("symbol", type=str)
    spot_p.add_argument("timeframe", type=str)
    spot_p.add_argument("--start", default="2022-04-01")
    spot_p.add_argument("--end", default=None)
    spot_p.set_defaults(func=_spot_ohlcv)

    funding_p = sub.add_parser("funding", help="Collect futures funding history")
    funding_p.add_argument("symbol", type=str)
    funding_p.add_argument("--start", default="2022-04-01")
    funding_p.add_argument("--end", required=True)
    funding_p.set_defaults(func=_funding)

    borrow_p = sub.add_parser("import-borrow", help="Import a versioned quote-borrow export")
    borrow_p.add_argument("symbol", type=str)
    borrow_p.add_argument("--source", required=True, help="Path to the borrow export parquet")
    borrow_p.add_argument("--source-id", required=True, help="Operator source locator/identifier")
    borrow_p.add_argument(
        "--rate-period", default="hourly",
        help="Source cadence: annual/daily/hourly, 1y/1d/1h, or integer seconds",
    )
    borrow_p.set_defaults(func=_import_borrow)

    collect_borrow_p = sub.add_parser(
        "collect-borrow", help="Collect signed Binance Margin quote-borrow history",
    )
    collect_borrow_p.add_argument("symbol", type=str)
    collect_borrow_p.add_argument("--asset", default="USDT")
    collect_borrow_p.add_argument("--start", default="2022-04-01")
    collect_borrow_p.add_argument("--end", required=True)
    collect_borrow_p.set_defaults(func=_collect_borrow)

    repair_p = sub.add_parser("repair-spot-gap", help="Record one documented synthetic spot bar")
    repair_p.add_argument("symbol", type=str)
    repair_p.add_argument("timeframe", type=str)
    repair_p.add_argument("timestamp", type=str)
    repair_p.set_defaults(func=_repair_spot_gap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
