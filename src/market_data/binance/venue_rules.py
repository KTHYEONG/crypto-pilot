"""Dated snapshots of Binance USD-M venue rules that govern margin and order admissibility: per-symbol leverage brackets and exchange order filters."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError

BRACKET_URL = "https://fapi.binance.com/fapi/v1/leverageBracket"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

VENUE_SNAPSHOT_SUFFIXES: tuple[str, ...] = (".json.gz", ".json")


@dataclass(frozen=True, slots=True)
class VenueBracket:
    """One notional tier of a symbol's leverage bracket ladder (USDT)."""

    notional_floor: float
    notional_cap: float
    maint_margin_ratio: float
    maint_amount: float
    initial_leverage: int


@dataclass(frozen=True, slots=True)
class VenueSymbolRules:
    """Margin ladder and order filters for one symbol at the snapshot time."""

    symbol: str
    brackets: tuple[VenueBracket, ...]
    step_size: float | None
    min_notional: float | None


@dataclass(frozen=True, slots=True)
class VenueRuleSnapshot:
    """All symbols' venue rules as observed at ``captured_at``; never valid for earlier dates by itself."""

    captured_at: pd.Timestamp
    symbols: Mapping[str, VenueSymbolRules]


def _parse_bracket_tiers(symbol: str, raw: Any) -> tuple[VenueBracket, ...]:
    tiers: list[VenueBracket] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise DataIntegrityError(f"leverageBracket bracket malformed for {symbol}")
        try:
            floor = float(entry["notionalFloor"])
            cap = float(entry["notionalCap"])
            ratio = float(entry["maintMarginRatio"])
            maint = float(entry["cum"] if "cum" in entry else entry["maintAmount"])
            leverage = int(entry["initialLeverage"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataIntegrityError(f"leverageBracket bracket missing required keys for {symbol}") from exc
        if ratio <= 0 or leverage <= 0 or not cap > floor:
            raise DataIntegrityError(f"leverageBracket bracket has non-positive ratio/leverage for {symbol}")
        tiers.append(
            VenueBracket(
                notional_floor=floor,
                notional_cap=cap,
                maint_margin_ratio=ratio,
                maint_amount=maint,
                initial_leverage=leverage,
            )
        )
    tiers.sort(key=lambda tier: tier.notional_floor)
    expected_floor = 0.0
    for tier in tiers:
        if tier.notional_floor != expected_floor:
            raise DataIntegrityError(f"leverageBracket ladder non-contiguous for {symbol}")
        expected_floor = tier.notional_cap
    return tuple(tiers)


def _extract_filters(entry: Mapping[str, Any]) -> tuple[float | None, float | None]:
    raw_filters = entry.get("filters")
    if not isinstance(raw_filters, list):
        return None, None
    step: float | None = None
    notional: float | None = None
    for item in raw_filters:
        filter_type = item.get("filterType")
        if filter_type == "LOT_SIZE" and "stepSize" in item:
            step = float(item["stepSize"])
        elif filter_type in ("MIN_NOTIONAL", "NOTIONAL"):
            key = "minNotional" if "minNotional" in item else "notional"
            if key in item:
                notional = float(item[key])
    return step, notional


def parse_venue_rules(
    bracket_payload: Any, exchange_info_payload: Any, *, captured_at: pd.Timestamp,
) -> VenueRuleSnapshot:
    """Join the leverageBracket and exchangeInfo responses into one snapshot.

    A symbol present only in the bracket payload keeps ``step_size``/``min_notional`` as None;
    a symbol present only in exchangeInfo is dropped (no margin ladder means it cannot be priced
    for margin). Brackets are sorted by notional floor and must tile [0, last cap) without gaps.

    Raises:
        DataIntegrityError: Malformed payloads, missing required bracket keys, non-positive
            ratios/leverage, or a non-contiguous ladder.
    """
    if not isinstance(bracket_payload, list):
        raise DataIntegrityError("leverageBracket payload must be a list")
    raw_symbols = exchange_info_payload.get("symbols") if isinstance(exchange_info_payload, Mapping) else None
    if not isinstance(raw_symbols, list):
        raise DataIntegrityError("exchangeInfo payload missing symbols list")
    filter_map: dict[str, tuple[float | None, float | None]] = {}
    for info_entry in raw_symbols:
        if not isinstance(info_entry, Mapping) or "symbol" not in info_entry:
            raise DataIntegrityError("exchangeInfo symbol entry malformed")
        filter_map[str(info_entry["symbol"])] = _extract_filters(info_entry)
    captured = pd.Timestamp(captured_at)
    captured = captured.tz_localize("UTC") if captured.tzinfo is None else captured.tz_convert("UTC")
    symbols: dict[str, VenueSymbolRules] = {}
    for row in bracket_payload:
        if not isinstance(row, Mapping) or "symbol" not in row:
            raise DataIntegrityError("leverageBracket row malformed")
        raw_tiers = row.get("brackets")
        if not isinstance(raw_tiers, list) or not raw_tiers:
            raise DataIntegrityError("leverageBracket row malformed")
        symbol = str(row["symbol"])
        step_size, min_notional = filter_map.get(symbol, (None, None))
        symbols[symbol] = VenueSymbolRules(
            symbol=symbol,
            brackets=_parse_bracket_tiers(symbol, raw_tiers),
            step_size=step_size,
            min_notional=min_notional,
        )
    return VenueRuleSnapshot(captured_at=captured, symbols=symbols)


def fetch_venue_rules(
    *, api_key: str | None = None, api_secret: str | None = None, timeout_seconds: float = 30.0
) -> VenueRuleSnapshot:
    """Fetch both endpoints (bracket endpoint is signed; exchangeInfo is public) and parse them.

    Raises:
        RuntimeError: HTTP failure or missing credentials.
        DataIntegrityError: Propagated from ``parse_venue_rules``.
    """
    key = api_key or os.getenv("BINANCE_API_KEY")
    secret = api_secret or os.getenv("BINANCE_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Binance venue-rules credentials are required")
    query = urllib.parse.urlencode({"timestamp": int(time.time() * 1000), "recvWindow": 5000})
    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{BRACKET_URL}?{query}&signature={signature}"
    request = urllib.request.Request(url, method="GET", headers={"X-MBX-APIKEY": key})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            bracket_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Binance venue-rules bracket request failed: {exc}") from exc
    try:
        with urllib.request.urlopen(EXCHANGE_INFO_URL, timeout=timeout_seconds) as response:  # noqa: S310
            exchange_info_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Binance venue-rules exchangeInfo request failed: {exc}") from exc
    return parse_venue_rules(
        bracket_payload, exchange_info_payload, captured_at=pd.Timestamp.now(tz="UTC")
    )


def write_venue_rule_snapshot(snapshot: VenueRuleSnapshot, root: Path) -> Path:
    """Persist ``<root>/<YYYYMMDD>.json.gz`` atomically (temp + os.replace); fresh-only per day.

    Gzip (deterministic, mtime=0) cuts the daily snapshot ~33x; content is the same JSON document as the
    legacy ``.json`` files.

    Raises:
        FileExistsError: a snapshot (either suffix) already exists for that UTC day.
    """
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    day = snapshot.captured_at.tz_convert("UTC").strftime("%Y%m%d")
    target = directory / f"{day}.json.gz"
    legacy = directory / f"{day}.json"
    if target.exists() or legacy.exists():
        existing = target if target.exists() else legacy
        raise FileExistsError(f"venue-rules snapshot already captured: {existing}")
    payload = {
        "captured_at": snapshot.captured_at.isoformat(),
        "symbols": {
            symbol: {
                "brackets": [
                    {
                        "notional_floor": tier.notional_floor,
                        "notional_cap": tier.notional_cap,
                        "maint_margin_ratio": tier.maint_margin_ratio,
                        "maint_amount": tier.maint_amount,
                        "initial_leverage": tier.initial_leverage,
                    }
                    for tier in rules.brackets
                ],
                "step_size": rules.step_size,
                "min_notional": rules.min_notional,
            }
            for symbol, rules in snapshot.symbols.items()
        },
    }
    tmp = target.with_name(f".{target.name}.tmp")
    blob = gzip.compress(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"), compresslevel=9, mtime=0)
    tmp.write_bytes(blob)
    os.replace(tmp, target)
    return target


def _snapshot_day(name: str) -> str | None:
    for suffix in VENUE_SNAPSHOT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def venue_rule_snapshot_paths(root: Path) -> list[Path]:
    """All snapshot files under ``root`` (both suffixes) sorted by capture day ascending.

    Raises:
        DataIntegrityError: two files claim the same day (``.json`` and ``.json.gz``).
    """
    directory = Path(root)
    if not directory.exists():
        return []
    by_day: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        day = _snapshot_day(path.name)
        if day is None or len(day) != 8 or not day.isdigit():
            continue
        if day in by_day:
            raise DataIntegrityError(f"duplicate venue-rules snapshot for day {day}")
        by_day[day] = path
    return [by_day[day] for day in sorted(by_day)]


def venue_rule_snapshot_exists(root: Path, day: str) -> bool:
    """Whether a snapshot for ``day`` (``YYYYMMDD``) exists under either suffix."""
    directory = Path(root)
    return (directory / f"{day}.json.gz").exists() or (directory / f"{day}.json").exists()


def load_venue_rule_snapshot(path: Path) -> VenueRuleSnapshot:
    """Load one persisted snapshot (``.json`` or ``.json.gz``) with ``parse_venue_rules`` validation."""
    src = Path(path)
    if src.name.endswith(".json.gz"):
        raw = json.loads(gzip.decompress(src.read_bytes()).decode("utf-8"))
    else:
        raw = json.loads(src.read_text(encoding="utf-8"))
    bracket_payload: list[Any] = []
    exchange_symbols: list[Any] = []
    for symbol, entry in raw["symbols"].items():
        bracket_payload.append(
            {
                "symbol": symbol,
                "brackets": [
                    {
                        "bracket": index + 1,
                        "initialLeverage": tier["initial_leverage"],
                        "notionalCap": tier["notional_cap"],
                        "notionalFloor": tier["notional_floor"],
                        "maintMarginRatio": tier["maint_margin_ratio"],
                        "cum": tier["maint_amount"],
                    }
                    for index, tier in enumerate(entry["brackets"])
                ],
            }
        )
        filters: list[Any] = []
        if entry.get("step_size") is not None:
            filters.append({"filterType": "LOT_SIZE", "stepSize": entry["step_size"]})
        if entry.get("min_notional") is not None:
            filters.append({"filterType": "MIN_NOTIONAL", "notional": entry["min_notional"]})
        if filters:
            exchange_symbols.append({"symbol": symbol, "filters": filters})
    return parse_venue_rules(
        bracket_payload, {"symbols": exchange_symbols}, captured_at=pd.Timestamp(raw["captured_at"])
    )


def latest_venue_rule_snapshot(root: Path) -> Path:
    """Most recent snapshot file under ``root`` (either suffix).

    Raises:
        FileNotFoundError: No snapshot exists.
    """
    files = venue_rule_snapshot_paths(root)
    if not files:
        raise FileNotFoundError(f"no venue-rules snapshot under {root}")
    return files[-1]
