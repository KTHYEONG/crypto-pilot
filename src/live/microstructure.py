"""Microstructure capture — top-of-book + premiumIndex observability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.paths import DATA_DIR
from src.live.records import append_typed_frame

_MICROSTRUCTURE_DTYPES: Mapping[str, str] = {
    "decision_time": "datetime64[ns, UTC]",
    "symbol": "object",
    "mode": "object",
    "bid": "float64",
    "ask": "float64",
    "bid_qty": "float64",
    "ask_qty": "float64",
    "mid": "float64",
    "spread_bps": "float64",
    "mark_price": "float64",
    "index_price": "float64",
    "last_funding_rate": "float64",
    "next_funding_time": "datetime64[ns, UTC]",
}


@dataclass(frozen=True, slots=True)
class BookQuote:
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_qty: Decimal
    ask_qty: Decimal

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        if mid <= 0:
            return float("nan")
        return float((self.ask - self.bid) / mid * Decimal(10_000))


def parse_book_quote(symbol: str, entry: Mapping[str, Any]) -> BookQuote | None:
    """Validate one bookTicker entry into a two-sided top of book, or None when it is unusable.

    Binance encodes an empty book side as price 0 (e.g. a settled delivery contract returned
    ``bidPrice "0.0"`` on 2026-09-25); a crossed or locked book, a non-finite value or a missing
    price is equally unusable for marking or pricing. Returning None makes the symbol *absent*,
    which every caller already handles (no mark, no post), instead of producing a mark of 0 or a
    half-price mid.

    Returns:
        ``BookQuote`` with bid > 0, ask > 0, bid < ask, all finite; quantities default to 0 when
        missing but a present non-numeric quantity makes the quote invalid.
    """
    try:
        if not isinstance(entry, Mapping):
            return None
        bid_raw = entry.get("bidPrice", None)
        if bid_raw is None:
            bid_raw = entry.get("bid", None)
        ask_raw = entry.get("askPrice", None)
        if ask_raw is None:
            ask_raw = entry.get("ask", None)
        if bid_raw is None or ask_raw is None:
            return None
        bid = Decimal(str(bid_raw))
        ask = Decimal(str(ask_raw))
        if not bid.is_finite() or not ask.is_finite():
            return None
        if bid <= 0 or ask <= 0 or not bid < ask:
            return None
        bid_qty_raw = entry.get("bidQty", None)
        if bid_qty_raw is None:
            bid_qty_raw = entry.get("bid_qty", None)
        ask_qty_raw = entry.get("askQty", None)
        if ask_qty_raw is None:
            ask_qty_raw = entry.get("ask_qty", None)
        bid_qty = Decimal(0) if bid_qty_raw is None else Decimal(str(bid_qty_raw))
        ask_qty = Decimal(0) if ask_qty_raw is None else Decimal(str(ask_qty_raw))
        if not bid_qty.is_finite() or not ask_qty.is_finite():
            return None
        return BookQuote(symbol=symbol, bid=bid, ask=ask, bid_qty=bid_qty, ask_qty=ask_qty)
    except Exception:  # noqa: BLE001, S112 - any parse failure makes the quote unusable
        return None


def fetch_book_quotes(client: Any, symbols: Sequence[str]) -> dict[str, BookQuote]:
    wanted = list(symbols)
    result: dict[str, BookQuote] = {}
    # Try batch
    batch_getter = getattr(client, "book_tickers", None)
    if callable(batch_getter):
        payload = batch_getter()
        # payload is dict symbol -> dict
        if isinstance(payload, dict):
            for sym in wanted:
                entry = payload.get(sym)
                if entry is None:
                    continue
                quote = parse_book_quote(sym, entry)
                if quote is None:
                    continue
                result[sym] = quote
            return result
        # if list, handle
        if isinstance(payload, list):
            indexed: dict[str, dict[str, Any]] = {}
            for e in payload:
                if isinstance(e, dict) and "symbol" in e:
                    indexed[str(e["symbol"])] = e
            for sym in wanted:
                entry = indexed.get(sym)
                if entry is None:
                    continue
                quote = parse_book_quote(sym, entry)
                if quote is None:
                    continue
                result[sym] = quote
            return result
    # fallback per-symbol
    for sym in wanted:
        try:
            getter = getattr(client, "book_ticker", None)
            if not callable(getter):
                continue
            entry = getter(sym)
        except Exception:  # noqa: S112 - 심볼별 조회 실패는 건너뛰고 나머지 수집
            continue
        quote = parse_book_quote(sym, entry)
        if quote is None:
            continue
        result[sym] = quote
    return result


@dataclass(frozen=True, slots=True)
class MicrostructureRecord:
    decision_time: pd.Timestamp
    symbol: str
    mode: str
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    mid: float
    spread_bps: float
    mark_price: float | None
    index_price: float | None
    last_funding_rate: float | None
    next_funding_time: pd.Timestamp | None


def build_microstructure_records(
    decision_time: pd.Timestamp,
    mode: str,
    quotes: Mapping[str, BookQuote],
    premium: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[MicrostructureRecord, ...]:
    dt = pd.Timestamp(decision_time)
    dt = dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")
    records: list[MicrostructureRecord] = []
    for symbol in sorted(quotes.keys()):
        q = quotes[symbol]
        bid_f = float(q.bid)
        ask_f = float(q.ask)
        bid_qty_f = float(q.bid_qty)
        ask_qty_f = float(q.ask_qty)
        mid_f = float(q.mid)
        spread = q.spread_bps
        mark_price: float | None = None
        index_price: float | None = None
        last_funding_rate: float | None = None
        next_funding_time: pd.Timestamp | None = None
        if premium is not None and symbol in premium:
            p = premium[symbol]
            try:
                mp = p.get("markPrice")
                if mp is not None:
                    mark_price = float(mp)
            except Exception:
                mark_price = None
            try:
                ip = p.get("indexPrice")
                if ip is not None:
                    index_price = float(ip)
            except Exception:
                index_price = None
            try:
                lfr = p.get("lastFundingRate")
                if lfr is not None:
                    last_funding_rate = float(lfr)
            except Exception:
                last_funding_rate = None
            try:
                nft = p.get("nextFundingTime")
                if nft is not None:
                    # nft may be ms int or timestamp
                    if isinstance(nft, (int, float)):
                        nft_ts = pd.Timestamp(int(nft), unit="ms", tz="UTC")
                    else:
                        nft_ts = pd.Timestamp(nft)
                        nft_ts = nft_ts.tz_localize("UTC") if nft_ts.tzinfo is None else nft_ts.tz_convert("UTC")
                    next_funding_time = nft_ts
            except Exception:
                next_funding_time = None
        records.append(
            MicrostructureRecord(
                decision_time=dt,
                symbol=str(symbol),
                mode=str(mode),
                bid=bid_f,
                ask=ask_f,
                bid_qty=bid_qty_f,
                ask_qty=ask_qty_f,
                mid=mid_f,
                spread_bps=float(spread),
                mark_price=mark_price,
                index_price=index_price,
                last_funding_rate=last_funding_rate,
                next_funding_time=next_funding_time,
            )
        )
    return tuple(records)


def append_microstructure(
    records: Sequence[MicrostructureRecord], history_dir: Path
) -> list[Path]:
    if not records:
        return []
    rows = [
        {
            "decision_time": r.decision_time,
            "symbol": r.symbol,
            "mode": r.mode,
            "bid": r.bid,
            "ask": r.ask,
            "bid_qty": r.bid_qty,
            "ask_qty": r.ask_qty,
            "mid": r.mid,
            "spread_bps": r.spread_bps,
            "mark_price": r.mark_price,
            "index_price": r.index_price,
            "last_funding_rate": r.last_funding_rate,
            "next_funding_time": r.next_funding_time,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    # enforce dtypes via records module
    return append_typed_frame(df, Path(history_dir), "microstructure", time_column="decision_time", dtypes=dict(_MICROSTRUCTURE_DTYPES))


def default_microstructure_dir() -> Path:
    return DATA_DIR / "state" / "live_microstructure"
