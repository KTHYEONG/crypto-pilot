from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from src.market_data.public_archive.catalog import is_perpetual_usdt
from src.market_data.public_archive.mask import allowed_days

SPOT_PREFIX_MULTIPLIERS: tuple[tuple[str, int], ...] = (("1000000", 1_000_000), ("1000", 1_000))


@dataclass(frozen=True, slots=True)
class SpotCompanion:
    futures_symbol: str
    spot_symbol: str | None
    price_multiplier: int
    first_seated: date
    last_seated: date
    status: Literal["mapped", "no_spot_history"]


def spot_candidates(futures_symbol: str) -> tuple[tuple[str, int], ...]:
    """Ordered (spot_symbol, multiplier) candidates for one USDT perpetual: exact name first, then prefix-stripped names."""
    if not is_perpetual_usdt(futures_symbol):
        raise ValueError(f"not a USDT perpetual: {futures_symbol!r}")
    out: list[tuple[str, int]] = [(futures_symbol, 1)]
    seen = {futures_symbol}
    for prefix, multiplier in SPOT_PREFIX_MULTIPLIERS:
        if futures_symbol.startswith(prefix):
            stripped = futures_symbol[len(prefix):]
            if stripped.removesuffix("USDT") and stripped not in seen:
                seen.add(stripped)
                out.append((stripped, multiplier))
    return tuple(out)


def resolve_spot_companions(
    seated: Mapping[str, tuple[date, date]],
    probe: Callable[[str, date, date], bool],
) -> list[SpotCompanion]:
    """Map each seated futures symbol to the first spot candidate whose probe finds klines inside its seated span; sorted by futures symbol."""
    companions: list[SpotCompanion] = []
    for futures_symbol in sorted(seated):
        first, last = seated[futures_symbol]
        mapped: SpotCompanion | None = None
        for spot_symbol, multiplier in spot_candidates(futures_symbol):
            if probe(spot_symbol, first, last):
                mapped = SpotCompanion(
                    futures_symbol=futures_symbol,
                    spot_symbol=spot_symbol,
                    price_multiplier=multiplier,
                    first_seated=first,
                    last_seated=last,
                    status="mapped",
                )
                break
        if mapped is None:
            mapped = SpotCompanion(
                futures_symbol=futures_symbol,
                spot_symbol=None,
                price_multiplier=1,
                first_seated=first,
                last_seated=last,
                status="no_spot_history",
            )
        companions.append(mapped)
    return companions


def seated_spans(mask: pd.DataFrame, *, first: date, last: date) -> dict[str, tuple[date, date]]:
    """First/last seated day per symbol from a seat-day mask restricted to [first, last]."""
    allowed = allowed_days(mask, first=first, last=last)
    return {symbol: (min(days), max(days)) for symbol, days in allowed.items() if days}
