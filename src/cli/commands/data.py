from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.application.data import collection

_logger = logging.getLogger(__name__)


def _ohlcv(args: argparse.Namespace) -> None:
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


def _metrics(args: argparse.Namespace) -> None:
    collection.collect_metrics(args.symbol, args.start, args.end)
    _logger.info("Metrics collection complete for %s (through %s)", args.symbol, args.end)


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


def add_data_commands(data_parser: argparse.ArgumentParser) -> None:
    """Attach the ``data collect <subcommand>`` group to the root parser."""
    collect = data_parser.add_subparsers(dest="data_command", required=True)
    collect_p = collect.add_parser("collect", help="Collect market data from Binance")
    collect_sub = collect_p.add_subparsers(dest="collect_command", required=True)

    futures = collect_sub.add_parser("futures-ohlcv", help="Collect futures OHLCV")
    futures.add_argument("symbol", type=str)
    futures.add_argument("timeframe", type=str)
    futures.add_argument("--start", default="2022-04-01")
    futures.add_argument("--end", default=None)
    futures.set_defaults(handler=_ohlcv)

    spot = collect_sub.add_parser("spot-ohlcv", help="Collect independent spot OHLCV")
    spot.add_argument("symbol", type=str)
    spot.add_argument("timeframe", type=str)
    spot.add_argument("--start", default="2022-04-01")
    spot.add_argument("--end", default=None)
    spot.set_defaults(handler=_spot_ohlcv)

    funding = collect_sub.add_parser("funding", help="Collect futures funding history")
    funding.add_argument("symbol", type=str)
    funding.add_argument("--start", default="2022-04-01")
    funding.add_argument("--end", required=True)
    funding.set_defaults(handler=_funding)

    metrics = collect_sub.add_parser(
        "metrics", help="Collect daily futures metrics (open interest) for one symbol",
    )
    metrics.add_argument("symbol", type=str)
    metrics.add_argument("--start", default="2022-04-01")
    metrics.add_argument("--end", required=True)
    metrics.set_defaults(handler=_metrics)

    import_borrow = collect_sub.add_parser(
        "import-borrow", help="Import a versioned quote-borrow export",
    )
    import_borrow.add_argument("symbol", type=str)
    import_borrow.add_argument("--source", required=True, help="Path to the borrow export parquet")
    import_borrow.add_argument("--source-id", required=True, help="Operator source locator/identifier")
    import_borrow.add_argument(
        "--rate-period", default="hourly",
        help="Source cadence: annual/daily/hourly, 1y/1d/1h, or integer seconds",
    )
    import_borrow.set_defaults(handler=_import_borrow)

    collect_borrow = collect_sub.add_parser(
        "collect-borrow", help="Collect signed Binance Margin quote-borrow history",
    )
    collect_borrow.add_argument("symbol", type=str)
    collect_borrow.add_argument("--asset", default="USDT")
    collect_borrow.add_argument("--start", default="2022-04-01")
    collect_borrow.add_argument("--end", required=True)
    collect_borrow.set_defaults(handler=_collect_borrow)

    repair = collect_sub.add_parser(
        "repair-spot-gap", help="Record one documented synthetic spot bar",
    )
    repair.add_argument("symbol", type=str)
    repair.add_argument("timeframe", type=str)
    repair.add_argument("timestamp", type=str)
    repair.set_defaults(handler=_repair_spot_gap)
