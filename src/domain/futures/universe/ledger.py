"""Ledger helpers for offline-first point-in-time universe workflow."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_LEDGER_PATH = Path("data/futures/universe_ledger.parquet")


def _to_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def query_ledger_as_of(
    ledger: pd.DataFrame,
    *,
    as_of: str | date,
    tf: str,
    symbols: Iterable[str] | None = None,
    enforce_eligibility: bool = True,
) -> pd.DataFrame:
    """Filter ledger rows with PIT-safe constraints.

    Args:
        ledger: Source ledger rows.
        as_of: Evaluation date.
        tf: Timeframe (e.g., 4h, 1h).
        symbols: Optional symbol whitelist.
        enforce_eligibility: Enforce Stage0 eligibility ``is_listed & is_trading``.

    Returns:
        PIT-filtered rows where ``knowledge_date <= as_of`` and ``date <= as_of``.

    """
    if ledger.empty:
        return ledger.copy()

    as_of_date = _to_date(as_of)
    out = ledger.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.date
    out["knowledge_date"] = pd.to_datetime(
        out["knowledge_date"], utc=True, errors="coerce"
    ).dt.date
    mask = (out["tf"] == tf) & (out["date"] <= as_of_date) & (out["knowledge_date"] <= as_of_date)
    if symbols is not None:
        symbol_set = set(symbols)
        mask &= out["symbol"].isin(symbol_set)
    if enforce_eligibility:
        is_listed = (
            out.get("is_listed", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        )
        is_trading = (
            out.get("is_trading", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        )
        mask &= is_listed & is_trading
    out = out.loc[mask]
    out = out.sort_values(["symbol", "date", "knowledge_date"])
    return out


def load_ledger_slice(
    *,
    as_of: str | date,
    tf: str,
    columns: tuple[str, ...],
    symbols: tuple[str, ...] | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    enforce_eligibility: bool = True,
) -> pd.DataFrame:
    """Load partitioned parquet ledger and apply PIT query."""
    needed = set(columns) | {"symbol", "tf", "date", "knowledge_date"}
    if enforce_eligibility:
        needed |= {"is_listed", "is_trading"}
    dataset = pd.read_parquet(ledger_path, columns=sorted(needed))
    return query_ledger_as_of(
        dataset,
        as_of=as_of,
        tf=tf,
        symbols=symbols,
        enforce_eligibility=enforce_eligibility,
    )


def update_ledger(new_rows: pd.DataFrame, *, ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
    """Append new rows to ledger storage in an idempotent way."""
    if new_rows.empty:
        return
    if ledger_path.exists():
        old = pd.read_parquet(ledger_path)
        merged = pd.concat([old, new_rows], ignore_index=True)
    else:
        merged = new_rows.copy()
    merged = merged.drop_duplicates(subset=["symbol", "tf", "date", "knowledge_date"], keep="last")
    merged = merged.sort_values(["symbol", "tf", "date", "knowledge_date"])
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(ledger_path, index=False)
