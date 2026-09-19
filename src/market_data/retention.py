"""Market data retention pruning with non-destructive guards."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.storage.ohlcv import is_temp_artifact
from src.mhs.params import SIGNAL_PANEL_WINDOW_DAYS

MARKET_DATA_MIN_RETENTION_DAYS: int = SIGNAL_PANEL_WINDOW_DAYS + 30
MARKET_DATA_MIN_KEPT_ROWS: int = 24

# Feeds still collected by MHS live refresh (1h trade OHLCV + settled funding).
# `markPriceKlines/1h` and `metrics/1d` are retired: ordinary retention never
# refreshes, prunes or validates them, and the 3m execution corpus owned by the
# backtest workflow is never a live-tail truncation target.
MHS_LIVE_RETENTION_FEEDS: tuple[str, ...] = ("ohlcv/1h", "funding")
MHS_RETIRED_FEEDS: tuple[str, ...] = ("markPriceKlines/1h", "metrics/1d")

_logger = logging.getLogger(__name__)


def prune_market_data(
    futures_root: Path, retention_days: int, *, now: pd.Timestamp
) -> dict[str, dict[str, int]]:
    """Apply the registered retention rule to data still collected by MHS. Historical input removal is a separate, explicitly targeted cleanup after active consumers and input manifests are migrated."""
    if retention_days < MARKET_DATA_MIN_RETENTION_DAYS:
        raise ValueError(
            f"data_retention_days must be >= {MARKET_DATA_MIN_RETENTION_DAYS}"
        )
    cutoff_ms = int(
        (pd.Timestamp(now).tz_convert("UTC") - pd.Timedelta(days=retention_days)).timestamp()
        * 1000
    )
    result: dict[str, dict[str, int]] = {}
    for rel in MHS_LIVE_RETENTION_FEEDS:
        d = Path(futures_root) / rel
        files_pruned = 0
        rows_removed = 0
        files_skipped = 0
        if not d.is_dir():
            result[rel] = {
                "files_pruned": 0,
                "rows_removed": 0,
                "files_skipped": 0,
            }
            continue
        for p in sorted(d.glob("*.parquet")):
            if is_temp_artifact(p.name):
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:
                files_skipped += 1
                continue
            if "timestamp" not in df.columns or not pd.api.types.is_integer_dtype(
                df["timestamp"]
            ):
                files_skipped += 1
                continue
            n0 = len(df)
            kept = df[df["timestamp"] >= cutoff_ms]
            if len(kept) == n0:
                continue
            if kept.empty or len(kept) < MARKET_DATA_MIN_KEPT_ROWS:
                files_skipped += 1
                continue
            tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.prune.tmp")
            kept.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(p)
            files_pruned += 1
            rows_removed += n0 - len(kept)
        result[rel] = {
            "files_pruned": files_pruned,
            "rows_removed": rows_removed,
            "files_skipped": files_skipped,
        }
    for rel in MHS_RETIRED_FEEDS:
        result[rel] = {
            "files_pruned": 0,
            "rows_removed": 0,
            "files_skipped": 0,
        }
    _logger.info(
        "[DATA] stage=prune_market_data retention_days=%d cutoff_ms=%d result=%s",
        retention_days,
        cutoff_ms,
        result,
    )
    return result


def prune_orderbook_history(
    orderbook_dir: Path, retention_days: int, *, now: pd.Timestamp
) -> int:
    if retention_days < 1:
        raise ValueError("orderbook_retention_days must be >= 1")
    cutoff = (
        pd.Timestamp(now).tz_convert("UTC") - pd.Timedelta(days=retention_days)
    ).strftime("%Y%m%d")
    d = Path(orderbook_dir)
    if not d.is_dir():
        return 0
    removed = 0
    for p in sorted(d.glob("live_orderbook_*.parquet")):
        tag = p.stem.removeprefix("live_orderbook_")
        if len(tag) == 8 and tag.isdigit() and tag < cutoff:
            p.unlink()
            removed += 1
    return removed


def check_orderbook_prune_impending(
    orderbook_dir: Path,
    retention_days: int,
    *,
    now: pd.Timestamp,
    warning_days: int = 7,
) -> tuple[bool, int, str | None]:
    """Check if any orderbook history files are within warning_days of being pruned.

    Returns (is_impending, days_left, earliest_date_str).
    """
    if retention_days < 1:
        raise ValueError("orderbook_retention_days must be >= 1")
    d = Path(orderbook_dir)
    if not d.is_dir():
        return False, 0, None
    tags: list[str] = []
    for p in d.glob("live_orderbook_*.parquet"):
        tag = p.stem.removeprefix("live_orderbook_")
        if len(tag) == 8 and tag.isdigit():
            tags.append(tag)
    if not tags:
        return False, 0, None

    earliest_tag = min(tags)
    earliest_dt = pd.Timestamp(earliest_tag, tz="UTC")
    t_now = pd.Timestamp(now)
    now_utc = t_now.tz_localize("UTC") if t_now.tzinfo is None else t_now.tz_convert("UTC")

    expiry_dt = earliest_dt + pd.Timedelta(days=retention_days)
    days_left = int((expiry_dt - now_utc).total_seconds() // 86400)

    if 0 <= days_left <= warning_days:
        return True, days_left, earliest_dt.strftime("%Y-%m-%d")
    return False, max(0, days_left), earliest_dt.strftime("%Y-%m-%d")


# Retired MHS feeds eligible only for explicitly targeted cleanup (never for
# automatic live-tail retention). Deletion targets are data files plus their
# collection sidecars; ohlcv/1h, ohlcv/3m, funding, backtest runs and unrelated
# data are never enumerated here.
RETIRED_MHS_CLEANUP_SUFFIXES: tuple[str, ...] = (".parquet", ".coverage.json")


def enumerate_retired_mhs_feed_files(futures_root: Path) -> list[Path]:
    """Enumerate the exact existing retired-feed files (mark + daily metrics)."""
    root = Path(futures_root)
    targets: list[Path] = []
    for rel in MHS_RETIRED_FEEDS:
        feed_dir = root / rel
        if not feed_dir.is_dir():
            continue
        for path in sorted(feed_dir.glob("*")):
            if not path.is_file() or is_temp_artifact(path.name):
                continue
            if not any(
                path.name.endswith(suffix) for suffix in RETIRED_MHS_CLEANUP_SUFFIXES
            ):
                continue
            targets.append(path)
    return targets


def retired_feed_active_readers() -> tuple[str, ...]:
    """Name the MHS/live readers that still consume retired mark files.

    Physical removal may proceed only after this returns empty (or the
    operator explicitly accepts the listed readers via ``allow_active_readers``).
    """
    reasons: list[str] = []
    return tuple(reasons)


def quarantine_retired_mhs_feeds(
    futures_root: Path,
    recovery_dir: Path,
    *,
    dry_run: bool = True,
    manifest_path: Path | None = None,
    allow_active_readers: bool = False,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Move retired mark/metrics files to a recovery location with a manifest.

    Enumeration is always safe; moving requires either no active readers or
    explicit operator acceptance. Reports removed file counts and bytes plus
    the recovery location. Never touches ohlcv/1h, ohlcv/3m, funding,
    existing backtest runs or unrelated data.
    """
    root = Path(futures_root)
    recovery = Path(recovery_dir)
    targets = enumerate_retired_mhs_feed_files(root)
    if not dry_run and not allow_active_readers:
        readers = retired_feed_active_readers()
        if readers:
            raise DataIntegrityError(
                "retired-feed cleanup blocked by active readers: " + "; ".join(readers)
            )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in targets:
        blob = path.read_bytes()
        size = len(blob)
        total_bytes += size
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    entries.sort(key=lambda entry: entry["relative_path"])
    moved = 0
    if not dry_run:
        recovery.mkdir(parents=True, exist_ok=True)
        for path in targets:
            rel = path.relative_to(root).as_posix()
            destination = recovery / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_name(destination.name + ".tmp")
            tmp.write_bytes(path.read_bytes())
            os.replace(tmp, destination)
            path.unlink()
            moved += 1
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else recovery / "retired_feeds_manifest.json"
    )
    payload = {
        "version": 1,
        "moved": not dry_run,
        "dry_run": dry_run,
        "futures_root": str(root),
        "recovery_dir": str(recovery),
        "created_at": (
            pd.Timestamp(now).tz_convert("UTC").isoformat()
            if now is not None
            else pd.Timestamp.now(tz="UTC").isoformat()
        ),
        "files": entries,
    }
    if not dry_run or manifest_path is not None:
        resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp_manifest = resolved_manifest.with_name(resolved_manifest.name + ".tmp")
        tmp_manifest.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(tmp_manifest, resolved_manifest)
    report = {
        "targets": len(targets),
        "moved": moved,
        "bytes": total_bytes,
        "recovery_dir": str(recovery),
        "manifest": str(resolved_manifest)
        if (not dry_run or manifest_path is not None)
        else None,
    }
    _logger.info(
        "[DATA] stage=quarantine_retired_feeds dry_run=%s targets=%d moved=%d bytes=%d recovery=%s",
        dry_run,
        len(targets),
        moved,
        total_bytes,
        recovery,
    )
    return report
