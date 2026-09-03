"""Live per-fill ledger — backtest fills.parquet 동형 기록."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.common.paths import DATA_DIR

FILL_REASONS: frozenset[str] = frozenset({"maker_fill", "backstop_taker", "timeout_taker", "residual", "obsolete", "immediate_taker"})

__all__ = ["FILL_REASONS", "_FILL_COLUMNS", "FillEvent", "append_fills", "default_fills_dir", "load_fills"]

_FILL_COLUMNS = [
    "decision_time",
    "timestamp",
    "symbol",
    "quantity_delta",
    "fill_price",
    "fee_bps",
    "reason",
    "pre_trade_equity",
    "liquidity",
    "mode",
    "run_id",
    "leg_index",
    "client_order_id",
    "decision_mark",
    "sizing_anchor",
]


@dataclass(frozen=True, slots=True)
class FillEvent:
    decision_time: pd.Timestamp
    timestamp: pd.Timestamp
    symbol: str
    quantity_delta: Decimal
    fill_price: Decimal
    fee_bps: float
    reason: str
    pre_trade_equity: Decimal
    liquidity: str
    mode: str
    run_id: str
    leg_index: int
    client_order_id: str
    decision_mark: Decimal | None = None
    sizing_anchor: str = "book_mid"


def default_fills_dir() -> Path:
    return DATA_DIR / "state" / "live_fills"


def _event_to_row(ev: FillEvent) -> dict[str, object]:
    if ev.reason not in FILL_REASONS:
        raise ValueError(f"invalid fill reason {ev.reason!r} not in {FILL_REASONS}")
    if ev.liquidity not in {"maker", "taker"}:
        raise ValueError(f"invalid liquidity {ev.liquidity!r}")
    # decision_mark may be None -> NaN via float conversion later
    return {
        "decision_time": pd.Timestamp(ev.decision_time).tz_localize("UTC") if pd.Timestamp(ev.decision_time).tzinfo is None else pd.Timestamp(ev.decision_time).tz_convert("UTC"),
        "timestamp": pd.Timestamp(ev.timestamp).tz_localize("UTC") if pd.Timestamp(ev.timestamp).tzinfo is None else pd.Timestamp(ev.timestamp).tz_convert("UTC"),
        "symbol": str(ev.symbol),
        "quantity_delta": float(ev.quantity_delta),
        "fill_price": float(ev.fill_price),
        "fee_bps": float(ev.fee_bps),
        "reason": str(ev.reason),
        "pre_trade_equity": float(ev.pre_trade_equity),
        "liquidity": str(ev.liquidity),
        "mode": str(ev.mode),
        "run_id": str(ev.run_id),
        "leg_index": int(ev.leg_index),
        "client_order_id": str(ev.client_order_id),
        "decision_mark": float(ev.decision_mark) if ev.decision_mark is not None else float("nan"),
        "sizing_anchor": str(ev.sizing_anchor),
    }


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # datetime typed
    for col in ("decision_time", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True).astype("datetime64[ns, UTC]")
    for col in ("quantity_delta", "fill_price", "fee_bps", "pre_trade_equity", "decision_mark"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ("leg_index",):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
    for col in ("symbol", "reason", "liquidity", "mode", "run_id", "client_order_id", "sizing_anchor"):
        if col in df.columns:
            df[col] = df[col].astype("object")
    # ensure column order
    ordered = [c for c in _FILL_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def append_fills(events: Sequence[FillEvent], fills_dir: Path) -> Path | None:
    if not events:
        return None
    # validate all reasons first (fail fast)
    for ev in events:
        if ev.reason not in FILL_REASONS:
            raise ValueError(f"invalid fill reason {ev.reason!r}")
    fills_dir = Path(fills_dir)
    fills_dir.mkdir(parents=True, exist_ok=True)
    # group by YYYYMM
    buckets: dict[str, list[FillEvent]] = {}
    for ev in events:
        ts = pd.Timestamp(ev.timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")  # noqa: SIM108
        key = ts.strftime("%Y%m")
        buckets.setdefault(key, []).append(ev)
    last_written: Path | None = None
    for yyyymm, evs in sorted(buckets.items()):
        rows = [_event_to_row(ev) for ev in evs]
        df_new = pd.DataFrame(rows)
        df_new = _enforce_dtypes(df_new)
        path = fills_dir / f"fills_{yyyymm}.parquet"
        if path.exists():
            try:
                df_existing = pd.read_parquet(path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = _enforce_dtypes(df_combined)
                df_combined.to_parquet(path, index=False, compression="snappy")
            except Exception:
                # fallback overwrite
                df_new.to_parquet(path, index=False, compression="snappy")
        else:
            df_new.to_parquet(path, index=False, compression="snappy")
        last_written = path
    return last_written


def load_fills(fills_dir: Path | str | None = None, *, since: pd.Timestamp | None = None) -> pd.DataFrame:
    dir_path = Path(fills_dir) if fills_dir is not None else default_fills_dir()
    if not dir_path.exists():
        # return empty with schema
        empty = pd.DataFrame({c: [] for c in _FILL_COLUMNS})
        # set dtypes
        empty["decision_time"] = pd.to_datetime(empty["decision_time"], utc=True).astype("datetime64[ns, UTC]")
        empty["timestamp"] = pd.to_datetime(empty["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        for col in ("quantity_delta", "fill_price", "fee_bps", "pre_trade_equity", "decision_mark"):
            empty[col] = empty[col].astype("float64")
        empty["leg_index"] = empty["leg_index"].astype("int64")
        for col in ("symbol", "reason", "liquidity", "mode", "run_id", "client_order_id", "sizing_anchor"):
            empty[col] = empty[col].astype("object")
        if since is not None:
            return empty
        return empty
    shards = sorted(dir_path.glob("fills_*.parquet"))
    if not shards:
        empty = pd.DataFrame({c: [] for c in _FILL_COLUMNS})
        empty["decision_time"] = pd.to_datetime(empty["decision_time"], utc=True).astype("datetime64[ns, UTC]")
        empty["timestamp"] = pd.to_datetime(empty["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        for col in ("quantity_delta", "fill_price", "fee_bps", "pre_trade_equity", "decision_mark"):
            empty[col] = empty[col].astype("float64")
        empty["leg_index"] = empty["leg_index"].astype("int64")
        for col in ("symbol", "reason", "liquidity", "mode", "run_id", "client_order_id", "sizing_anchor"):
            empty[col] = empty[col].astype("object")
        return empty
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            df = pd.read_parquet(shard)
            if not df.empty:
                frames.append(df)
        except Exception:  # noqa: S112, BLE001
            continue
    if not frames:
        empty = pd.DataFrame({c: [] for c in _FILL_COLUMNS})
        empty["decision_time"] = pd.to_datetime(empty["decision_time"], utc=True).astype("datetime64[ns, UTC]")
        empty["timestamp"] = pd.to_datetime(empty["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        for col in ("quantity_delta", "fill_price", "fee_bps", "pre_trade_equity", "decision_mark"):
            empty[col] = empty[col].astype("float64")
        empty["leg_index"] = empty["leg_index"].astype("int64")
        for col in ("symbol", "reason", "liquidity", "mode", "run_id", "client_order_id", "sizing_anchor"):
            empty[col] = empty[col].astype("object")
        return empty
    combined = pd.concat(frames, ignore_index=True)
    combined = _enforce_dtypes(combined)
    # reorder to canonical order
    ordered = [c for c in _FILL_COLUMNS if c in combined.columns]
    extras = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + extras]
    if since is not None:
        try:
            since_ts = pd.Timestamp(since)
            since_ts = since_ts.tz_localize("UTC") if since_ts.tzinfo is None else since_ts.tz_convert("UTC")
            mask = pd.to_datetime(combined["timestamp"], utc=True) >= since_ts
            combined = combined[mask]
        except Exception:  # noqa: BLE001, S110
            pass
    return combined


_referenced_load_fills = load_fills  # noqa: F841
