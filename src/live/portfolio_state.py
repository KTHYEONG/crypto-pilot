"""사이클별 포트폴리오 상태 기록 — 가상 MTM 관측 레이어."""

from __future__ import annotations

import contextlib
import io
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR
from src.live.settings import ExecutionMode
from src.mhs.run_history import RUN_HISTORY_MAX_SHARDS, RUN_HISTORY_SHARD_MAX_BYTES

logger = logging.getLogger("PortfolioState")

_PORTFOLIO_ACTIVE_FILE_NAME = "active.parquet"
_PORTFOLIO_ARCHIVE_PREFIX = "portfolio_state_"
_PORTFOLIO_ARCHIVE_SUFFIX = ".parquet"


@dataclass(frozen=True, slots=True)
class PortfolioStateRecord:
    decision_time: pd.Timestamp
    mode: str
    equity_usdt: float
    equity_source: str
    cash_usdt: float | None
    wallet_balance_usdt: float | None
    unrealized_pnl_usdt: float | None
    equity_high_water_mark_usdt: float
    gross_notional_usdt: float
    n_holdings: int
    intent_count: int
    dropped_notional_fraction: float


def virtual_mtm_equity(
    cash_usdt: Decimal,
    positions: Mapping[str, Decimal],
    marks: Mapping[str, Decimal],
) -> Decimal:
    """cash + sum(qty * mark) — marks 없는 심볼은 0으로 무시."""
    total = Decimal(cash_usdt)
    for sym, qty in positions.items():
        mark = marks.get(sym)
        if mark is not None:
            total += qty * mark
    return total


def resolve_effective_equity(
    mode: ExecutionMode,
    venue_equity: Decimal,
    cash_usdt: Decimal | None,
    positions: Mapping[str, Decimal],
    marks: Mapping[str, Decimal],
) -> tuple[Decimal, str]:
    """모드에 따라 venue 또는 가상 MTM equity를 반환."""
    if mode.suppresses_mutations:
        cash = cash_usdt if cash_usdt is not None else Decimal(0)
        return (virtual_mtm_equity(cash, positions, marks), "virtual_mtm")
    return (venue_equity, "venue")


def default_portfolio_state_dir() -> Path:
    return DATA_DIR / "state" / "live_portfolio_state"


def _archive_path(history_dir: Path, utc_millis: int) -> Path:
    return history_dir / f"{_PORTFOLIO_ARCHIVE_PREFIX}{utc_millis}{_PORTFOLIO_ARCHIVE_SUFFIX}"


def _unique_archive_path(history_dir: Path) -> Path:
    utc_millis = int(time.time() * 1000)
    archive = _archive_path(history_dir, utc_millis)
    while archive.exists():
        utc_millis += 1
        archive = _archive_path(history_dir, utc_millis)
    return archive


def _prune_archives(history_dir: Path) -> None:
    archives = sorted(history_dir.glob(f"{_PORTFOLIO_ARCHIVE_PREFIX}*{_PORTFOLIO_ARCHIVE_SUFFIX}"))
    excess = len(archives) - RUN_HISTORY_MAX_SHARDS
    for stale in archives[:excess]:
        with contextlib.suppress(OSError):
            stale.unlink()


def _record_to_dataframe(record: PortfolioStateRecord) -> pd.DataFrame:
    # Ensure decision_time is tz-aware UTC
    ts = pd.Timestamp(record.decision_time)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    data = {
        "decision_time": [ts],
        "mode": [str(record.mode)],
        "equity_usdt": [float(record.equity_usdt)],
        "equity_source": [str(record.equity_source)],
        "cash_usdt": [float(record.cash_usdt) if record.cash_usdt is not None else float("nan")],
        "wallet_balance_usdt": [float(record.wallet_balance_usdt) if record.wallet_balance_usdt is not None else float("nan")],
        "unrealized_pnl_usdt": [float(record.unrealized_pnl_usdt) if record.unrealized_pnl_usdt is not None else float("nan")],
        "equity_high_water_mark_usdt": [float(record.equity_high_water_mark_usdt)],
        "gross_notional_usdt": [float(record.gross_notional_usdt)],
        "n_holdings": [int(record.n_holdings)],
        "intent_count": [int(record.intent_count)],
        "dropped_notional_fraction": [float(record.dropped_notional_fraction)],
    }
    df = pd.DataFrame(data)
    # Enforce typed dtypes
    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True).astype("datetime64[ns, UTC]")
    df["mode"] = df["mode"].astype("object")
    df["equity_source"] = df["equity_source"].astype("object")
    for col in ("equity_usdt", "cash_usdt", "wallet_balance_usdt", "unrealized_pnl_usdt", "equity_high_water_mark_usdt", "gross_notional_usdt", "dropped_notional_fraction"):
        df[col] = df[col].astype("float64")
    for col in ("n_holdings", "intent_count"):
        df[col] = df[col].astype("int64")
    return df


def append_portfolio_state(record: PortfolioStateRecord, history_dir: Path) -> Path | None:
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    active = history_dir / _PORTFOLIO_ACTIVE_FILE_NAME
    df_new = _record_to_dataframe(record)
    buf = io.BytesIO()
    df_new.to_parquet(buf, index=False, compression="snappy")
    new_bytes = buf.getvalue()
    if active.exists() and active.stat().st_size + len(new_bytes) > RUN_HISTORY_SHARD_MAX_BYTES:
        archive = _unique_archive_path(history_dir)
        active.rename(archive)
        _prune_archives(history_dir)
    if active.exists():
        try:
            df_existing = pd.read_parquet(active)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Ensure dtypes remain typed after concat
            df_combined["decision_time"] = pd.to_datetime(df_combined["decision_time"], utc=True).astype("datetime64[ns, UTC]")
            for col in ("equity_usdt", "cash_usdt", "wallet_balance_usdt", "unrealized_pnl_usdt", "equity_high_water_mark_usdt", "gross_notional_usdt", "dropped_notional_fraction"):
                if col in df_combined.columns:
                    df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce").astype("float64")
            for col in ("n_holdings", "intent_count"):
                if col in df_combined.columns:
                    df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce").astype("int64")
            df_combined.to_parquet(active, index=False, compression="snappy")
        except Exception:
            df_new.to_parquet(active, index=False, compression="snappy")
    else:
        df_new.to_parquet(active, index=False, compression="snappy")
    return active


def _load_all_frames(history_dir: Path) -> pd.DataFrame | None:
    if not history_dir.exists():
        return None
    shards = sorted(history_dir.glob("*.parquet"))
    if not shards:
        return None
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            df = pd.read_parquet(shard)
            if not df.empty:
                frames.append(df)
        except Exception:  # noqa: S112
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def summarize_portfolio_state(
    history_dir: Path | str | None = None,
    *,
    since: pd.Timestamp | None = None,
) -> dict[str, Any]:
    dir_path = Path(history_dir) if history_dir is not None else default_portfolio_state_dir()
    base: dict[str, Any] = {"n_cycles": 0, "by_mode": {}}
    if not dir_path.exists():
        return base
    try:
        combined = _load_all_frames(dir_path)
    except Exception:  # noqa: BLE001
        return base
    if combined is None or combined.empty:
        return base
    # Filter since if provided
    if since is not None:
        try:
            since_ts = pd.Timestamp(since)
            since_ts = since_ts.tz_localize("UTC") if since_ts.tzinfo is None else since_ts.tz_convert("UTC")
            dt_parsed = pd.to_datetime(combined["decision_time"], utc=True, errors="coerce")
            mask = dt_parsed >= since_ts
            combined = combined[mask]
            if combined.empty:
                return base
        except Exception:  # noqa: S110
            pass
    try:
        # n_cycles distinct decision_time
        try:
            n_cycles = int(combined["decision_time"].nunique())
        except Exception:
            n_cycles = len(combined)
        by_mode: dict[str, Any] = {}
        if "mode" in combined.columns:
            for mode_val, group in combined.groupby("mode"):
                # n_cycles per mode
                try:
                    mode_n = int(group["decision_time"].nunique())
                except Exception:
                    mode_n = len(group)
                # sort by decision_time
                try:
                    g_sorted = group.sort_values("decision_time")
                    last_row = g_sorted.iloc[-1]
                    last_dt = pd.Timestamp(last_row["decision_time"])
                    last_dt = last_dt.tz_localize("UTC") if last_dt.tzinfo is None else last_dt.tz_convert("UTC")
                    last_decision_time = last_dt.isoformat()
                    last_equity = float(last_row["equity_usdt"]) if pd.notna(last_row["equity_usdt"]) else None
                except Exception:
                    last_decision_time = None
                    last_equity = None
                # max drawdown fraction per mode
                max_dd: float | None = None
                try:
                    equities = pd.to_numeric(group.sort_values("decision_time")["equity_usdt"], errors="coerce").dropna()
                    if not equities.empty:
                        hwm = float("-inf")
                        worst = 0.0
                        has = False
                        for val in equities:
                            fval = float(val)
                            if fval > hwm:
                                hwm = fval
                            if hwm > 0:
                                dd = fval / hwm - 1.0
                                if not has or dd < worst:
                                    worst = dd
                                    has = True
                        max_dd = worst if has else 0.0
                except Exception:
                    max_dd = None
                by_mode[str(mode_val)] = {
                    "n_cycles": mode_n,
                    "last_decision_time": last_decision_time,
                    "last_equity_usdt": last_equity,
                    "max_drawdown_fraction": max_dd,
                }
        return {"n_cycles": n_cycles, "by_mode": by_mode}
    except Exception:  # noqa: BLE001
        return base
