"""``data collect spot-universe``: PIT futures universe to spot companion mapping and collection."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

_logger = logging.getLogger(__name__)

_PROGRESS_INTERVAL_S = 30.0
# Binance "Invalid symbol": the pair never existed on spot, which is an answer, not a failure.
_INVALID_SYMBOL_CODE = -1121


class _Progress:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self._last = self.started

    def tick(self, done: int, total: int, ok: int, failed: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last >= _PROGRESS_INTERVAL_S:
            _logger.info(
                "[DATA] stage=spot_universe done=%d/%d ok=%d failed=%d elapsed_s=%.1f",
                done, total, ok, failed, now - self.started,
            )
            self._last = now


def _seated_spans(args: argparse.Namespace, start_date: date, end_date: date) -> dict[str, tuple[date, date]]:
    from src.common import paths as paths_mod
    from src.market_data.public_archive.catalog import MASK_POLICIES
    from src.market_data.public_archive.mask import load_daily_census, seat_day_mask
    from src.market_data.services.spot_universe import seated_spans

    daily_close, daily_quote_volume = load_daily_census(paths_mod.FUTURES_DATA_DIR / "ohlcv" / "1h")
    mask = seat_day_mask(
        daily_close,
        daily_quote_volume,
        breadth=int(args.breadth),
        preroll_days=MASK_POLICIES["aggTrades"].preroll_days,
    )
    return seated_spans(mask, first=start_date, last=end_date)


def _has_bars(client: Any, symbol: str, day: date, last: date) -> bool:
    from urllib.error import HTTPError

    from src.market_data.binance.futures import BinanceKlinePermanentError

    upper = min(day + timedelta(days=1), last + timedelta(days=1))
    try:
        frame = client.fetch_spot_ohlcv(
            symbol, "1h", day.isoformat(),
            str(pd.Timestamp(upper, tz="UTC") - pd.Timedelta(milliseconds=1)),
        )
    except BinanceKlinePermanentError as exc:
        cause = exc.__cause__
        if isinstance(cause, HTTPError) and json.loads(cause.read()).get("code") == _INVALID_SYMBOL_CODE:
            return False
        raise
    return not frame.empty


def _earliest_seated(companions: list[dict[str, Any]]) -> dict[str, str]:
    """First seated day per distinct mapped spot symbol across every futures symbol sharing it."""
    earliest: dict[str, str] = {}
    for c in companions:
        if c["status"] != "mapped":
            continue
        spot = str(c["spot_symbol"])
        first_seated = str(c["first_seated"])
        if spot not in earliest or first_seated < earliest[spot]:
            earliest[spot] = first_seated
    return earliest


def _collect(earliest: dict[str, str], args: argparse.Namespace, progress: _Progress) -> list[str]:
    """Collect each spot symbol once; failures are returned, never abort the remaining symbols."""
    from src.market_data.services import spot_collection as spot_collection_mod

    end_str = str(args.end)
    timeframe = str(getattr(args, "timeframe", "1h"))
    failures: list[str] = []
    for collected, spot in enumerate(sorted(earliest), start=1):
        try:
            spot_collection_mod.ensure_spot_ohlcv(spot, timeframe, earliest[spot], end_str)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{spot}: {exc}")
            _logger.warning("[DATA] stage=spot_universe symbol=%s status=FAILED detail=%s", spot, exc)
        progress.tick(collected, len(earliest), collected - len(failures), len(failures))
    return failures


def run_spot_universe(args: argparse.Namespace) -> None:
    """Write the companion mapping to ``--output`` and, with ``--execute``, collect spot OHLCV; exits 3 on any collection failure."""
    from dataclasses import asdict

    from src.common.durable_io import durable_write_text
    from src.market_data.binance.spot import BinanceSpotClient
    from src.market_data.services.spot_universe import resolve_spot_companions

    start_date = date.fromisoformat(str(args.start))
    end_date = date.fromisoformat(str(args.end))
    if start_date > end_date:
        raise ValueError(f"invalid range start={args.start} end={args.end}")
    spans = _seated_spans(args, start_date, end_date)
    client = BinanceSpotClient()
    progress = _Progress()

    def _probe(symbol: str, first: date, last: date) -> bool:
        progress.tick(0, len(spans), 0, 0)
        return _has_bars(client, symbol, first, last) or _has_bars(client, symbol, first + (last - first) // 2, last)

    companions = [asdict(companion) for companion in resolve_spot_companions(spans, _probe)]
    output = Path(str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    durable_write_text(output, json.dumps(companions, sort_keys=True, indent=2, default=str) + "\n")

    mapped = sum(1 for c in companions if c["status"] == "mapped")
    missing = sum(1 for c in companions if c["status"] == "no_spot_history")
    failures: list[str] = []
    if bool(getattr(args, "execute", False)) and mapped:
        failures = _collect(_earliest_seated(companions), args, progress)
    _logger.info(
        "[DATA] stage=spot_universe status=COMPLETED mapped=%d no_spot_history=%d failed=%d elapsed_s=%.1f",
        mapped, missing, len(failures), time.monotonic() - progress.started,
    )
    if failures:
        _logger.error("[DATA] stage=spot_universe failures=%s", "; ".join(failures))
        raise SystemExit(3)


def add_spot_universe_command(collect_sub: Any) -> None:
    """Register ``spot-universe`` on the ``data collect`` subparsers."""
    from src.common.paths import DATA_DIR
    from src.market_data.public_archive.catalog import MASK_POLICIES

    spot_universe = collect_sub.add_parser(
        "spot-universe", help="Map PIT futures universe to spot companions and collect spot OHLCV",
    )
    spot_universe.add_argument("--start", default="2023-01-01")
    spot_universe.add_argument(
        "--end",
        default=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date().isoformat(),
    )
    spot_universe.add_argument("--breadth", type=int, default=MASK_POLICIES["aggTrades"].breadth)
    spot_universe.add_argument("--timeframe", default="1h")
    spot_universe.add_argument("--execute", action="store_true", default=False)
    spot_universe.add_argument("--output", default=str(DATA_DIR / "spot" / "companions.json"))
    spot_universe.set_defaults(handler=run_spot_universe)
