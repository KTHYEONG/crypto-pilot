from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError


@dataclass(frozen=True, slots=True)
class LiquidRoster:
    version: int
    as_of: pd.Timestamp
    top_n: int
    lookback_days: int
    symbols: tuple[str, ...]


def roster_path(root: Path) -> Path:
    return Path(root) / "_roster" / "aggtrades_liquid.json"


def ever_liquid_symbols(
    daily_quote_volume: pd.DataFrame, *, top_n: int, lookback_days: int
) -> tuple[str, ...]:
    """Union per-date top-N symbols ranked by trailing median quote volume."""
    if top_n <= 0 or lookback_days <= 0 or daily_quote_volume.empty:
        return ()
    frame = daily_quote_volume.sort_index()
    symbols = [str(c) for c in frame.columns]
    if len(frame) <= lookback_days:
        return ()
    found: set[str] = set()
    frame.columns = symbols
    observed_dates = frame.index
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="D"))
    trails = frame.shift(1).rolling(lookback_days, min_periods=1).median()
    trails = trails.iloc[lookback_days:].reindex(observed_dates).dropna(how="all")
    for row in trails.itertuples(index=False, name=None):
        order = sorted(
            (j for j in range(len(symbols)) if pd.notna(row[j])),
            key=lambda j: (-row[j], symbols[j]),
        )
        found.update(symbols[j] for j in order[:top_n])
    return tuple(sorted(found))


def load_roster(path: Path) -> LiquidRoster | None:
    """Load a persisted roster, failing closed on malformed content."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise DataIntegrityError(f"malformed roster: {path}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"malformed roster: {path}") from exc
    if not isinstance(obj, dict):
        raise DataIntegrityError(f"malformed roster: {path}")
    try:
        version = obj["version"]
        as_of_raw = obj["as_of"]
        top_n = obj["top_n"]
        lookback_days = obj["lookback_days"]
        symbols_raw = obj["symbols"]
        if (
            type(version) is not int or version < 1
            or type(top_n) is not int or top_n <= 0
            or type(lookback_days) is not int or lookback_days <= 0
            or not isinstance(as_of_raw, str)
            or not isinstance(symbols_raw, list)
            or any(not isinstance(s, str) for s in symbols_raw)
        ):
            raise ValueError("invalid roster schema")
        symbols = tuple(symbols_raw)
        if len(set(symbols)) != len(symbols) or tuple(sorted(symbols)) != symbols:
            raise ValueError("roster symbols must be sorted unique")
        as_of = pd.Timestamp(as_of_raw)
        if pd.isna(as_of):
            raise ValueError("invalid roster as_of")
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed roster: {path}") from exc
    return LiquidRoster(
        version=version,
        as_of=as_of,
        top_n=top_n,
        lookback_days=lookback_days,
        symbols=symbols,
    )


def save_roster(path: Path, roster: LiquidRoster) -> None:
    """Persist a roster with an atomic sibling-replace write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(roster.as_of)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    payload = {
        "as_of": stamp.isoformat(),
        "lookback_days": roster.lookback_days,
        "symbols": list(roster.symbols),
        "top_n": roster.top_n,
        "version": roster.version,
    }
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def merge_roster(
    existing: LiquidRoster | None,
    discovered: tuple[str, ...],
    *,
    as_of: pd.Timestamp,
    top_n: int,
    lookback_days: int,
) -> LiquidRoster:
    """Union discovered symbols into a grow-only roster."""
    stamp = pd.Timestamp(as_of)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if existing is None:
        return LiquidRoster(
            version=1,
            as_of=stamp,
            top_n=top_n,
            lookback_days=lookback_days,
            symbols=tuple(sorted(set(discovered))),
        )
    merged = tuple(sorted(set(existing.symbols) | set(discovered)))
    version = existing.version if merged == existing.symbols else existing.version + 1
    return LiquidRoster(
        version=version,
        as_of=stamp,
        top_n=top_n,
        lookback_days=lookback_days,
        symbols=merged,
    )
