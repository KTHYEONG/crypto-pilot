"""Point-in-time universe gap detection against the Binance Vision archive."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from src.market_data.storage.ohlcv import is_temp_artifact
from src.quant.universe.pit_universe import symbol_partition

EXCHANGE_INFO_URL: str = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def local_futures_symbols(root: Path, timeframe: str) -> frozenset[str]:
    """Symbols with a persisted ``<root>/<timeframe>/<SYMBOL>.parquet`` file.

    Temporary write artifacts are excluded so an interrupted collection is not
    mistaken for a collected symbol.
    """
    directory = Path(root) / timeframe
    names = [p.name for p in directory.glob("*.parquet")]
    return frozenset(
        Path(name).stem for name in names if not is_temp_artifact(name)
    )


def non_crypto_symbols(exchange_info: Mapping[str, Any]) -> frozenset[str]:
    """Currently-listed Binance UM futures symbols whose underlying is not a cryptocurrency.

    Binance UM futures lists tokenized equities (``EQUITY``/``KR_EQUITY``/
    ``HK_EQUITY``/``CN_EQUITY``/``PREMARKET``), commodities (``COMMODITY``) and
    index baskets (``INDEX``) under the same ``USDT``-quoted naming convention
    as crypto perpetuals (e.g. ``TSLAUSDT``, ``XAUUSDT``, ``HK0700USDT``); only
    ``underlyingType == "COIN"`` is a crypto asset. A symbol absent from this
    payload (delisted, e.g. ``MATICUSDT``/``LUNAUSDT``/``EOSUSDT``) is never
    excluded -- this can only ever narrow a gap list, never revive a
    historically-delisted crypto contract as a false exclusion.

    Raises:
        ValueError: ``exchange_info`` has no ``symbols`` list.
    """
    symbols = exchange_info.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError("exchange_info missing a symbols list")
    return frozenset(
        str(entry["symbol"])
        for entry in symbols
        if str(entry.get("underlyingType", "COIN")) != "COIN"
    )


def fetch_exchange_info(*, timeout: int = 20) -> dict[str, Any]:
    """Unauthenticated GET of the public Binance UM futures ``exchangeInfo`` endpoint."""
    with urllib.request.urlopen(EXCHANGE_INFO_URL, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())  # type: ignore[no-any-return]


def historical_universe_gaps(
    vision_symbols: Iterable[str],
    local_symbols: Iterable[str],
    *,
    partition: Literal["dev", "holdout", "all"],
    quote_asset: str = "USDT",
    exclude: Iterable[str] = (),
) -> tuple[str, ...]:
    """Vision-listed perpetuals of ``partition`` that are absent from the local lake.

    The archive keeps delisted and renamed contracts (e.g. MATICUSDT, LUNAUSDT),
    while a lake built from today's exchange listing does not; every missing
    contract is a survivorship bias in any historical cross-section. ``exclude``
    (typically :func:`non_crypto_symbols`) removes tokenized non-crypto
    products that share the Vision archive's USDT-perpetual naming convention.

    Raises:
        ValueError: ``vision_symbols`` is empty (a failed listing must never read
            as "no gaps") or ``partition`` is unknown.
    """
    materialized = list(vision_symbols)
    if not materialized:
        raise ValueError("vision_symbols must not be empty")
    if partition not in ("dev", "holdout", "all"):
        raise ValueError(f"unknown partition {partition!r}")
    local = set(local_symbols)
    excluded = set(exclude)
    gaps: list[str] = []
    for symbol in materialized:
        if not symbol.endswith(quote_asset):
            continue
        if partition != "all" and symbol_partition(symbol) != partition:
            continue
        if symbol in excluded:
            continue
        if symbol not in local:
            gaps.append(symbol)
    return tuple(sorted(gaps))
