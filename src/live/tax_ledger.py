"""Tax ledger — immutable JSONL with watermark idempotent collection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import DATA_DIR
from src.live.records import append_jsonl_partition

TAX_RECORD_KINDS: frozenset[str] = frozenset({"TRADE", "REALIZED_PNL", "FUNDING_FEE", "COMMISSION", "TRANSFER"})


@dataclass(frozen=True, slots=True)
class TaxRecord:
    record_id: str
    kind: str
    event_time: pd.Timestamp
    symbol: str
    side: str
    quantity: float
    price: float
    quote_qty: float
    fee: float
    fee_asset: str
    realized_pnl: float
    income_asset: str
    is_maker: bool
    venue_id: int
    source: str
    mode: str


@dataclass(frozen=True, slots=True)
class TaxWatermark:
    last_trade_id: dict[str, int]
    last_income_id: int
    last_collected_at: pd.Timestamp | None


def _income_kind(income_type: str) -> str:
    mapping = {
        "REALIZED_PNL": "REALIZED_PNL",
        "FUNDING_FEE": "FUNDING_FEE",
        "COMMISSION": "COMMISSION",
        "TRANSFER": "TRANSFER",
    }
    return mapping.get(income_type.upper(), income_type.upper())


def collect_tax_records(
    client: Any,
    symbols: Sequence[str],
    watermark: TaxWatermark,
    mode: str,
    *,
    now: pd.Timestamp,
) -> tuple[tuple[TaxRecord, ...], TaxWatermark]:
    now_ts = pd.Timestamp(now)
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    records: list[TaxRecord] = []
    new_last_trade: dict[str, int] = dict(watermark.last_trade_id)
    max_income_id = watermark.last_income_id
    max_trade_seen: dict[str, int] = {}
    # userTrades per symbol
    for sym in symbols:
        last_id = watermark.last_trade_id.get(sym)
        from_id = (last_id + 1) if last_id is not None else None
        try:
            trades = client.user_trades(sym, from_id=from_id) if from_id is not None else client.user_trades(sym)
        except TypeError:
            # fallback if client signature doesn't accept from_id keyword
            try:
                trades = client.user_trades(sym, from_id)
            except Exception:
                trades = []
        except Exception:
            trades = []
        if not isinstance(trades, list):
            continue
        for entry in trades:
            try:
                venue_id = int(entry.get("id", entry.get("tranId", 0)))
            except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
                continue
            # skip duplicates already seen (id <= last_id)
            if last_id is not None and venue_id <= last_id:
                continue
            if sym not in max_trade_seen or venue_id > max_trade_seen[sym]:
                max_trade_seen[sym] = venue_id
            if venue_id > max_income_id:
                # we track income separately, don't mix
                pass
            # parse fields
            try:
                price = float(entry.get("price", 0) or 0)
                qty = float(entry.get("qty", entry.get("quantity", 0)) or 0)
                quote_qty = float(entry.get("quoteQty", entry.get("quote_qty", 0)) or 0)
                if quote_qty == 0 and qty and price:
                    quote_qty = qty * price
                fee = float(entry.get("commission", 0) or 0)
                fee_asset = str(entry.get("commissionAsset", entry.get("commission_asset", "")) or "")
                realized_pnl = float(entry.get("realizedPnl", entry.get("realized_pnl", 0)) or 0)
                # time
                t_raw = entry.get("time", entry.get("event_time", now_ts))
                try:
                    if isinstance(t_raw, (int, float)):
                        event_time = pd.Timestamp(int(t_raw), unit="ms", tz="UTC")
                    else:
                        event_time = pd.Timestamp(t_raw)
                        if event_time.tzinfo is None:
                            event_time = event_time.tz_localize("UTC")
                        else:
                            event_time = event_time.tz_convert("UTC")
                except Exception:
                    event_time = now_ts
                symbol = str(entry.get("symbol", sym) or sym)
                # side from buyer or side
                is_buyer = entry.get("buyer", entry.get("isBuyer", None))
                if isinstance(is_buyer, bool):
                    side = "BUY" if is_buyer else "SELL"
                else:
                    side_raw = str(entry.get("side", "") or "").upper()
                    side = side_raw if side_raw in ("BUY", "SELL") else ""
                is_maker = bool(entry.get("maker", False))
                # income asset for TRADE is commissionAsset
                income_asset = fee_asset or "USDT"
            except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
                continue
            record_id = f"venue:TRADE:{venue_id}"
            # source is always venue for collected
            rec = TaxRecord(
                record_id=record_id,
                kind="TRADE",
                event_time=event_time,
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                quote_qty=quote_qty,
                fee=fee,
                fee_asset=fee_asset,
                realized_pnl=realized_pnl,
                income_asset=income_asset,
                is_maker=is_maker,
                venue_id=venue_id,
                source="venue",
                mode=str(mode),
            )
            # dedup by venue_id check (should not duplicate)
            records.append(rec)
        if max_trade_seen.get(sym) is not None:
            new_last_trade[sym] = max(new_last_trade.get(sym, -1), max_trade_seen[sym])
    # income
    start_ms: int | None = None
    if watermark.last_collected_at is not None:
        try:
            ts = pd.Timestamp(watermark.last_collected_at)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            start_ms = int(ts.timestamp() * 1000) + 1
        except Exception:
            start_ms = None
    try:
        incomes = client.income(start_time_ms=start_ms) if start_ms is not None else client.income()
    except TypeError:
        try:
            incomes = client.income(start_time_ms=start_ms) if start_ms is not None else client.income()
        except Exception:
            incomes = []
    except Exception:
        incomes = []
    if isinstance(incomes, list):
        for entry in incomes:
            try:
                tran_id_raw = entry.get("tranId", entry.get("id", entry.get("tran_id", 0)))
                venue_id = int(tran_id_raw) if tran_id_raw is not None else 0
            except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
                continue
            if venue_id <= watermark.last_income_id:
                continue
            if venue_id > max_income_id:
                max_income_id = venue_id
            try:
                income_type = str(entry.get("incomeType", entry.get("income_type", "TRANSFER")) or "TRANSFER")
                kind = _income_kind(income_type)
                if kind not in TAX_RECORD_KINDS:
                    kind = "TRANSFER"
                inc = float(entry.get("income", 0) or 0)
                asset = str(entry.get("asset", "") or "")
                symbol = str(entry.get("symbol", "") or "")
                t_raw = entry.get("time", entry.get("tranTime", now_ts))
                try:
                    if isinstance(t_raw, (int, float)):
                        event_time = pd.Timestamp(int(t_raw), unit="ms", tz="UTC")
                    else:
                        event_time = pd.Timestamp(t_raw)
                        if event_time.tzinfo is None:
                            event_time = event_time.tz_localize("UTC")
                        else:
                            event_time = event_time.tz_convert("UTC")
                except Exception:
                    event_time = now_ts
                # For income records: price/qty etc zero, realized_pnl = income
                realized_pnl = inc
            except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
                continue
            record_id = f"venue:{kind}:{venue_id}"
            rec = TaxRecord(
                record_id=record_id,
                kind=kind,
                event_time=event_time,
                symbol=symbol,
                side="",
                quantity=0.0,
                price=0.0,
                quote_qty=0.0,
                fee=0.0,
                fee_asset="",
                realized_pnl=realized_pnl,
                income_asset=asset,
                is_maker=False,
                venue_id=venue_id,
                source="venue",
                mode=str(mode),
            )
            records.append(rec)
    new_watermark = TaxWatermark(
        last_trade_id=new_last_trade,
        last_income_id=max_income_id,
        last_collected_at=now_ts,
    )
    # deduplicate by venue_id already, but also sort by event_time
    records_sorted = sorted(records, key=lambda r: r.event_time)
    return tuple(records_sorted), new_watermark


def simulated_tax_records(
    fill_events: Sequence[Any], mode: str
) -> tuple[TaxRecord, ...]:
    records: list[TaxRecord] = []
    seq = -1
    for ev in fill_events:
        try:
            qty_delta = float(ev.quantity_delta) if hasattr(ev, "quantity_delta") else 0.0
        except Exception:
            qty_delta = 0.0
        quantity = abs(qty_delta)
        side = "BUY" if qty_delta > 0 else "SELL" if qty_delta < 0 else ""
        try:
            price = float(ev.fill_price) if hasattr(ev, "fill_price") else 0.0
        except Exception:
            price = 0.0
        quote_qty = quantity * price
        try:
            fee_bps = float(getattr(ev, "fee_bps", 0) or 0)
        except Exception:
            fee_bps = 0.0
        fee = abs(quote_qty) * fee_bps / 10_000
        fee_asset = "USDT"
        # event_time
        try:
            ts_raw = getattr(ev, "timestamp", getattr(ev, "decision_time", None))
            event_time = pd.Timestamp(ts_raw)
            event_time = (
                event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
            )
        except Exception:  # noqa: BLE001 - 타임스탬프 파싱 실패 시 현재 시각으로 대체
            event_time = pd.Timestamp.now(tz="UTC")
        symbol = str(getattr(ev, "symbol", "") or "")
        liquidity = str(getattr(ev, "liquidity", "") or "")
        is_maker = liquidity == "maker"
        venue_id = seq
        seq -= 1
        # record_id 는 사이클 간 유일해야 한다: run_id(=decision_time YYYYMMDD)가
        # 없으면 event_time 초 단위로 대체. leg_index+cycle-seq 로 부분체결까지 구분.
        run_tag = str(getattr(ev, "run_id", "") or "") or event_time.strftime("%Y%m%dT%H%M%S")
        leg_index = int(getattr(ev, "leg_index", 0) or 0)
        record_id = f"simulated:TRADE:{run_tag}:{symbol}:{leg_index}:{-venue_id}"
        rec = TaxRecord(
            record_id=record_id,
            kind="TRADE",
            event_time=event_time,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            quote_qty=quote_qty,
            fee=fee,
            fee_asset=fee_asset,
            realized_pnl=0.0,
            income_asset=fee_asset,
            is_maker=is_maker,
            venue_id=venue_id,
            source="simulated",
            mode=str(mode),
        )
        records.append(rec)
    return tuple(records)


def append_tax_records(
    records: Sequence[TaxRecord], ledger_dir: Path
) -> list[Path]:
    if not records:
        return []
    rows: list[dict[str, Any]] = [
        {
            "record_id": r.record_id,
            "kind": r.kind,
            "event_time": r.event_time,
            "symbol": r.symbol,
            "side": r.side,
            "quantity": r.quantity,
            "price": r.price,
            "quote_qty": r.quote_qty,
            "fee": r.fee,
            "fee_asset": r.fee_asset,
            "realized_pnl": r.realized_pnl,
            "income_asset": r.income_asset,
            "is_maker": r.is_maker,
            "venue_id": r.venue_id,
            "source": r.source,
            "mode": r.mode,
        }
        for r in records
    ]
    return append_jsonl_partition(rows, Path(ledger_dir), "tax_ledger", time_key="event_time")


def load_tax_records(
    ledger_dir: Path | str | None = None, *, year: int | None = None
) -> pd.DataFrame:
    dir_path = Path(ledger_dir) if ledger_dir is not None else default_tax_ledger_dir()
    if not dir_path.exists():
        return pd.DataFrame()
    shards = sorted(dir_path.glob("tax_ledger_*.jsonl"))
    if not shards:
        # fallback glob
        shards = sorted(dir_path.glob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard in shards:
        try:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
                        continue
                    rid = obj.get("record_id")
                    if rid in seen:
                        continue
                    seen.add(rid)
                    # year filter
                    if year is not None:
                        try:
                            et = pd.Timestamp(obj.get("event_time"))
                            et = et.tz_localize("UTC") if et.tzinfo is None else et.tz_convert("UTC")
                            if et.year != year:
                                continue
                        except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
                            continue
                    rows.append(obj)
        except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # ensure event_time dtype
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    # deduplicate by record_id first-seen already
    return df


def summarize_tax_year(
    year: int,
    ledger_dir: Path | str | None = None,
    *,
    cost_basis: Literal["moving_average", "fifo"] = "moving_average",
    source: str = "venue",
) -> dict[str, Any]:
    df = load_tax_records(ledger_dir, year=year)
    if df.empty:
        return {
            "year": year,
            "source": source,
            "per_symbol": {},
            "total": {
                "acquisition_cost": 0.0,
                "disposal_proceeds": 0.0,
                "fees": 0.0,
                "funding": 0.0,
                "realized_pnl": 0.0,
            },
            "unclassified_income": {},
            "asset": "USDT",
        }
    # source purity check
    if "source" in df.columns:
        unique_sources = set(df["source"].dropna().unique())
        # if any row source != requested source -> fail closed
        mixed = any(s != source for s in unique_sources)
        if mixed:
            raise DataIntegrityError(f"mixed sources in year {year}: {unique_sources} vs requested {source}")
    # Prepare per_symbol aggregation
    # For cost basis, need to process TRADE rows chronologically per symbol
    # Filter to TRADE kind for cost basis
    trades = df[df["kind"] == "TRADE"].copy() if "kind" in df.columns else pd.DataFrame()
    if not trades.empty and "event_time" in trades.columns:
        trades = trades.sort_values("event_time")
    per_symbol: dict[str, dict[str, Any]] = {}
    total_acq = 0.0
    total_disp = 0.0
    total_fees = 0.0
    total_funding = 0.0
    total_realized = 0.0
    # compute fees/funding totals across all rows
    if "fee" in df.columns:
        try:
            total_fees = float(pd.to_numeric(df["fee"], errors="coerce").fillna(0).sum())
        except Exception:
            total_fees = 0.0
    # funding = sum of FUNDING_FEE realized_pnl
    if "kind" in df.columns and "realized_pnl" in df.columns:
        try:
            funding_mask = df["kind"] == "FUNDING_FEE"
            total_funding = float(pd.to_numeric(df.loc[funding_mask, "realized_pnl"], errors="coerce").fillna(0).sum())
        except Exception:
            total_funding = 0.0
    # Map symbol -> list of trades
    symbol_groups: dict[str, pd.DataFrame] = {}
    if not trades.empty:
        for sym, grp in trades.groupby("symbol"):
            symbol_groups[str(sym)] = grp.sort_values("event_time")
    else:
        symbol_groups = {}
    # Ensure all symbols from dataframe accounted even if only funding? collect symbols
    all_symbols: set[str] = set()
    if "symbol" in df.columns:
        all_symbols.update(str(s) for s in df["symbol"].dropna().unique() if str(s))
    for sym in sorted(all_symbols):
        grp = symbol_groups.get(sym, pd.DataFrame())
        # compute cost basis for this symbol
        if grp.empty:
            # no trades, but may have funding etc.
            per_symbol[sym] = {
                "acquisition_cost": 0.0,
                "disposal_proceeds": 0.0,
                "fees": float(pd.to_numeric(df[df["symbol"]==sym]["fee"], errors="coerce").fillna(0).sum()) if "fee" in df.columns else 0.0,
                "funding": float(pd.to_numeric(df[(df["symbol"]==sym) & (df["kind"]=="FUNDING_FEE")]["realized_pnl"], errors="coerce").fillna(0).sum()) if "kind" in df.columns else 0.0,
                "realized_pnl": float(pd.to_numeric(df[df["symbol"]==sym]["realized_pnl"], errors="coerce").fillna(0).sum()) if "realized_pnl" in df.columns else 0.0,
                "n_trades": 0,
                "closing_quantity": 0.0,
                "closing_cost_basis": 0.0,
            }
            continue
        acq = 0.0
        disp = 0.0
        n_trades = len(grp)
        qty_held = 0.0
        avg_cost = 0.0
        # fifo queue
        fifo_lots: list[list[float]] = []  # [qty, price]
        # for moving average, track total cost = avg*qty
        for _, row in grp.iterrows():
            try:
                qty = float(row.get("quantity", 0) or 0)
                price = float(row.get("price", 0) or 0)
                side = str(row.get("side", "") or "").upper()
            except Exception:  # noqa: S112 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
                continue
            if side == "BUY":
                acq += qty * price
                if cost_basis == "moving_average":
                    # update avg
                    total_cost_before = avg_cost * qty_held
                    qty_held += qty
                    avg_cost = (total_cost_before + qty * price) / qty_held if qty_held != 0 else 0.0
                else:  # fifo
                    fifo_lots.append([qty, price])
                    qty_held += qty
            elif side == "SELL":
                disp += qty * price
                if cost_basis == "moving_average":
                    qty_held -= qty
                    if qty_held < 0:
                        qty_held = 0
                        avg_cost = 0.0
                else:
                    # fifo consume
                    remaining = qty
                    while remaining > 0 and fifo_lots:
                        lot_qty, _lot_price = fifo_lots[0]
                        if lot_qty <= remaining + 1e-12:
                            remaining -= lot_qty
                            qty_held -= lot_qty
                            fifo_lots.pop(0)
                        else:
                            fifo_lots[0][0] = lot_qty - remaining
                            qty_held -= remaining
                            remaining = 0
                    if not fifo_lots and remaining > 0:
                        # sell more than held? goes negative? just subtract
                        qty_held -= remaining
            else:
                # unknown side skip
                continue
        if cost_basis == "moving_average":
            closing_qty = qty_held
            closing_basis = avg_cost * closing_qty if closing_qty > 0 else 0.0
        else:
            closing_qty = qty_held
            closing_basis = sum(q * p for q, p in fifo_lots)
            # if moving average still need qty_held for closing
        # fees/funding per symbol
        try:
            sym_fees = float(pd.to_numeric(df[df["symbol"]==sym]["fee"], errors="coerce").fillna(0).sum()) if "fee" in df.columns else 0.0
        except Exception:
            sym_fees = 0.0
        try:
            sym_funding = float(pd.to_numeric(df[(df["symbol"]==sym) & (df["kind"]=="FUNDING_FEE")]["realized_pnl"], errors="coerce").fillna(0).sum()) if "kind" in df.columns else 0.0
        except Exception:
            sym_funding = 0.0
        try:
            sym_realized = float(pd.to_numeric(df[df["symbol"]==sym]["realized_pnl"], errors="coerce").fillna(0).sum()) if "realized_pnl" in df.columns else 0.0
        except Exception:
            sym_realized = 0.0
        per_symbol[sym] = {
            "acquisition_cost": float(acq),
            "disposal_proceeds": float(disp),
            "fees": float(sym_fees),
            "funding": float(sym_funding),
            "realized_pnl": float(sym_realized),
            "n_trades": int(n_trades),
            "closing_quantity": float(closing_qty),
            "closing_cost_basis": float(closing_basis),
        }
        total_acq += acq
        total_disp += disp
        # total_fees already summed, but use per-symbol sum? avoid double count
    # if per_symbol was empty due to no trades but total_fees etc already computed
    # Fix total_fees/funding/realized recalc via sum of per_symbol? For simplicity keep totals from full df
    # total fees already computed above includes all symbols; total acq/disp summed above.
    # For consistency, compute total realized as sum
    try:
        total_realized = float(pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0).sum()) if "realized_pnl" in df.columns else 0.0
    except Exception:
        total_realized = 0.0
    # unclassified_income for TRANSFER
    unclassified: dict[str, float] = {}
    if "kind" in df.columns:
        try:
            transfer_mask = df["kind"] == "TRANSFER"
            if transfer_mask.any():
                transfer_sum = float(pd.to_numeric(df.loc[transfer_mask, "realized_pnl"], errors="coerce").fillna(0).sum())
                unclassified["TRANSFER"] = transfer_sum
        except Exception:  # noqa: S110 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
            pass
    # asset inference
    asset = "USDT"
    if "income_asset" in df.columns:
        try:
            vals = df["income_asset"].dropna().unique()
            if len(vals) > 0:
                asset = str(vals[0])
        except Exception:  # noqa: S110 - 개별 레코드 파싱 실패는 건너뛰고 수집 지속
            pass
    # mode inference
    mode_val = None
    if "mode" in df.columns:
        try:
            mode_val = str(df["mode"].dropna().iloc[0]) if not df["mode"].dropna().empty else source
        except Exception:
            mode_val = source
    return {
        "year": year,
        "source": source,
        "mode": mode_val if mode_val is not None else source,
        "per_symbol": per_symbol,
        "total": {
            "acquisition_cost": float(total_acq),
            "disposal_proceeds": float(total_disp),
            "fees": float(total_fees),
            "funding": float(total_funding),
            "realized_pnl": float(total_realized),
        },
        "unclassified_income": unclassified,
        "asset": asset,
    }


def default_tax_ledger_dir() -> Path:
    return DATA_DIR / "state" / "live_tax_ledger"
