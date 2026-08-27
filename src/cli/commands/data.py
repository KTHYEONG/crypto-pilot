from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from src.application.data import collection
from src.market_data.storage.gap_report import detect_internal_gaps

_logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION_START = "2022-04-01"


def _ohlcv(args: argparse.Namespace) -> None:
    end = args.end or str(pd.Timestamp.now(tz="UTC"))
    collection.collect_ohlcv(args.symbol, args.timeframe, args.start, end)
    _logger.info("Data collection complete for %s %s (through %s)", args.symbol, args.timeframe, end)


def _mhs_execution(args: argparse.Namespace) -> None:
    from src.application.data.mhs_execution_collection import (
        build_mhs_execution_plan,
        collect_mhs_execution_data,
    )

    end = args.end or str(pd.Timestamp.now(tz="UTC"))
    plan = build_mhs_execution_plan(
        args.start, end, timeframe=args.timeframe,
        execution_universe_size=args.execution_universe_size,
    )
    result = collect_mhs_execution_data(plan, execute=args.execute, workers=args.workers)
    _logger.info(
        "MHS execution data %s: timeframe=%s symbols=%d manifest=%s",
        result["mode"], plan.timeframe, len(plan.symbols), plan.manifest_path,
    )


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


def _indicator_klines(args: argparse.Namespace) -> None:
    end = args.end or str(pd.Timestamp.now(tz="UTC"))
    collection.collect_indicator_klines(args.dataset, args.symbol, args.timeframe, args.start, end)
    _logger.info(
        "Indicator kline collection complete for %s %s %s (through %s)",
        args.dataset, args.symbol, args.timeframe, end,
    )


def _bookdepth(args: argparse.Namespace) -> None:
    collection.collect_bookdepth(args.symbol, args.start, args.end)
    _logger.info("Bookdepth collection complete for %s (through %s)", args.symbol, args.end)


def _import_borrow(args: argparse.Namespace) -> None:
    collection.import_borrow(args.symbol, args.source, args.source_id, args.rate_period)
    _logger.info("Borrow history imported for %s from %s", args.symbol, args.source)


def _collect_borrow(args: argparse.Namespace) -> None:
    collection.collect_borrow(args.symbol, args.asset, args.start, args.end)
    _logger.info(
        "Binance Margin borrow history collected for %s (%s)", args.symbol, args.asset,
    )


def _report_internal_gaps(args: argparse.Namespace) -> None:
    from src.common.config import FUTURES_DATA_DIR
    from src.mhs.panel import load_base_panel

    panel = load_base_panel(
        str(FUTURES_DATA_DIR / "ohlcv"), args.timeframe,
        ("open", "high", "low", "close", "quote_vol"),
        pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"), partition="all",
    )
    valid = panel["close"].notna()
    for col in ("open", "high", "low", "quote_vol"):
        valid &= panel[col].notna()
    gaps = detect_internal_gaps(valid)
    for sym, spans in sorted(gaps.items()):
        for start, end, length in spans:
            _logger.info("internal gap symbol=%s start=%s end=%s length_bars=%d", sym, start, end, length)


def _refresh_one_symbol_tail(collector: Any, symbol: str, start: str, end: str) -> bool:
    """Best-effort incremental tail top-up for one symbol. ``ensure_ohlcv_data``/
    ``ensure_funding_data`` are themselves idempotent (they check the existing
    cache and only fetch the missing tail), so a per-symbol network failure is
    logged and skipped rather than aborting the whole refresh."""
    try:
        collector.ensure_ohlcv_data(symbol, "1h", start, end)
        collector.ensure_funding_data(symbol, start, end)
        return True
    except Exception as exc:  # noqa: BLE001 - one symbol's flakiness must not abort the batch
        _logger.warning("[DATA] refresh_live_universe symbol=%s failed error=%s", symbol, exc)
        return False


def _refresh_live_universe(args: argparse.Namespace) -> None:
    """Incremental tail top-up: 1h/funding for every cached dev symbol, then
    the 3m roster for the deployed execution universe. 1h/funding MUST run
    first -- ``build_mhs_execution_plan`` ranks the roster by trailing 1h
    quote volume, so a stale 1h panel silently picks a stale roster."""
    import concurrent.futures
    import glob
    import os
    import time

    from src.common.config import FUTURES_DATA_DIR
    from src.mhs.params import SIGNAL_PANEL_WINDOW_DAYS

    t0 = time.perf_counter()
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=SIGNAL_PANEL_WINDOW_DAYS)

    ohlcv_paths = sorted(glob.glob(str(FUTURES_DATA_DIR / "ohlcv" / "1h" / "*.parquet")))
    symbols = [os.path.basename(p).removesuffix(".parquet") for p in ohlcv_paths]

    from src.market_data.services.futures_collection import DataCollector

    collector = DataCollector()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda sym: _refresh_one_symbol_tail(collector, sym, str(start), str(now)), symbols,
            )
        )
    failures = sum(1 for ok in results if not ok)

    from src.application.data.mhs_execution_collection import (
        build_mhs_execution_plan,
        collect_mhs_execution_data,
    )

    exec_size = int(getattr(args, "execution_universe_size", 60) or 60)
    plan = build_mhs_execution_plan(str(start), str(now), timeframe="3m", execution_universe_size=exec_size)
    collect_mhs_execution_data(plan, execute=True, workers=2)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _logger.info(
        "[DATA] stage=refresh_live_universe ohlcv_symbols=%d funding_symbols=%d roster_symbols=%d "
        "failures=%d elapsed_ms=%d",
        len(symbols), len(symbols), len(plan.symbols), failures, elapsed_ms,
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
    futures.add_argument("--start", default=_DEFAULT_COLLECTION_START)
    futures.add_argument("--end", default=None)
    futures.set_defaults(handler=_ohlcv)

    mhs_execution = collect_sub.add_parser(
        "mhs-execution", help="Plan or collect PIT MHS execution OHLCV (dry-run by default)",
    )
    mhs_execution.add_argument("--timeframe", choices=["1m", "3m", "5m"], default="3m")
    mhs_execution.add_argument("--start", default="2021-01-01")
    mhs_execution.add_argument("--end", default=None)
    mhs_execution.add_argument("--execution-universe-size", type=int, default=30)
    mhs_execution.add_argument("--workers", type=int, default=4)
    mhs_execution.add_argument(
        "--execute", action="store_true",
        help="Actually download data; without this flag only the plan manifest is written",
    )
    mhs_execution.set_defaults(handler=_mhs_execution)

    spot = collect_sub.add_parser("spot-ohlcv", help="Collect independent spot OHLCV")
    spot.add_argument("symbol", type=str)
    spot.add_argument("timeframe", type=str)
    spot.add_argument("--start", default=_DEFAULT_COLLECTION_START)
    spot.add_argument("--end", default=None)
    spot.set_defaults(handler=_spot_ohlcv)

    funding = collect_sub.add_parser("funding", help="Collect futures funding history")
    funding.add_argument("symbol", type=str)
    funding.add_argument("--start", default=_DEFAULT_COLLECTION_START)
    funding.add_argument("--end", required=True)
    funding.set_defaults(handler=_funding)

    metrics = collect_sub.add_parser(
        "metrics", help="Collect daily futures metrics (open interest) for one symbol",
    )
    metrics.add_argument("symbol", type=str)
    metrics.add_argument("--start", default=_DEFAULT_COLLECTION_START)
    metrics.add_argument("--end", required=True)
    metrics.set_defaults(handler=_metrics)

    indicator_klines = collect_sub.add_parser(
        "indicator-klines", help="Collect mark/index/premium klines from Vision archives",
    )
    indicator_klines.add_argument("dataset", type=str)
    indicator_klines.add_argument("symbol", type=str)
    indicator_klines.add_argument("timeframe", type=str)
    indicator_klines.add_argument("--start", default=_DEFAULT_COLLECTION_START)
    indicator_klines.add_argument("--end", default=None)
    indicator_klines.set_defaults(handler=_indicator_klines)

    bookdepth = collect_sub.add_parser(
        "bookdepth", help="Collect daily book depth from Vision archives",
    )
    bookdepth.add_argument("symbol", type=str)
    bookdepth.add_argument("--start", default=_DEFAULT_COLLECTION_START)
    bookdepth.add_argument("--end", default=None)
    bookdepth.set_defaults(handler=_bookdepth)

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
    collect_borrow.add_argument("--start", default=_DEFAULT_COLLECTION_START)
    collect_borrow.add_argument("--end", required=True)
    collect_borrow.set_defaults(handler=_collect_borrow)

    repair = collect_sub.add_parser(
        "repair-spot-gap", help="Record one documented synthetic spot bar",
    )
    repair.add_argument("symbol", type=str)
    repair.add_argument("timeframe", type=str)
    repair.add_argument("timestamp", type=str)
    repair.set_defaults(handler=_repair_spot_gap)

    report_gaps = collect_sub.add_parser(
        "report-internal-gaps", help="Read-only report of per-symbol internal OHLCV gaps (never deletes data)",
    )
    report_gaps.add_argument("--timeframe", default="1h")
    report_gaps.add_argument("--start", default="2019-01-01")
    report_gaps.add_argument("--end", required=True)
    report_gaps.set_defaults(handler=_report_internal_gaps)

    # wiring: refresh.set_defaults(handler=_refresh_live_universe)
    refresh = collect.add_parser("refresh-live-universe", help="Incremental tail top-up for live signal refresh (1h/funding + 3m roster)")
    refresh.set_defaults(handler=_refresh_live_universe)
