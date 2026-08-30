"""Microstructure capture — top-of-book + premiumIndex observability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR
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
                try:
                    bid = Decimal(str(entry.get("bidPrice", entry.get("bid", "0"))))
                    ask = Decimal(str(entry.get("askPrice", entry.get("ask", "0"))))
                    bq_raw = entry.get("bidQty", entry.get("bid_qty", entry.get("bidQty", None)))
                    aq_raw = entry.get("askQty", entry.get("ask_qty", entry.get("askQty", None)))
                    # Handle missing keys: entry may not have bidQty/askQty
                    if bq_raw is None:
                        # try alternative keys
                        bq_raw = entry.get("bidQty", 0)
                        if "bidQty" not in entry and "bid_qty" not in entry:
                            bq_raw = Decimal(0)
                    if aq_raw is None:
                        aq_raw = Decimal(0)
                    bid_qty = Decimal(str(bq_raw)) if bq_raw is not None else Decimal(0)
                    ask_qty = Decimal(str(aq_raw)) if aq_raw is not None else Decimal(0)
                except Exception:  # noqa: S112 - 개별 심볼 파싱 실패는 건너뛰고 나머지 수집
                    continue
                result[sym] = BookQuote(symbol=sym, bid=bid, ask=ask, bid_qty=bid_qty, ask_qty=ask_qty)
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
                try:
                    bid = Decimal(str(entry.get("bidPrice", entry.get("bid", "0"))))
                    ask = Decimal(str(entry.get("askPrice", entry.get("ask", "0"))))
                    bq_raw = entry.get("bidQty", Decimal(0))
                    aq_raw = entry.get("askQty", Decimal(0))
                    bid_qty = Decimal(str(bq_raw)) if bq_raw is not None else Decimal(0)
                    ask_qty = Decimal(str(aq_raw)) if aq_raw is not None else Decimal(0)
                except Exception:  # noqa: S112 - 개별 심볼 파싱 실패는 건너뛰고 나머지 수집
                    continue
                result[sym] = BookQuote(symbol=sym, bid=bid, ask=ask, bid_qty=bid_qty, ask_qty=ask_qty)
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
        try:
            bid = Decimal(str(entry.get("bidPrice", entry.get("bid", "0"))))
            ask = Decimal(str(entry.get("askPrice", entry.get("ask", "0"))))
            bq_raw = entry.get("bidQty", entry.get("bid_qty", None))
            aq_raw = entry.get("askQty", entry.get("ask_qty", None))
            bid_qty = Decimal(str(bq_raw)) if bq_raw is not None else Decimal(0)
            ask_qty = Decimal(str(aq_raw)) if aq_raw is not None else Decimal(0)
        except Exception:  # noqa: S112 - 심볼별 파싱 실패는 건너뛰고 나머지 수집
            continue
        result[sym] = BookQuote(symbol=sym, bid=bid, ask=ask, bid_qty=bid_qty, ask_qty=ask_qty)
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
