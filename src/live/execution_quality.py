"""Live execution quality recording — observability-only compression layer over audit logs."""

from __future__ import annotations

import contextlib
import io
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR
from src.live.audit import AUDIT_LOG_RETENTION_DAYS
from src.mhs.params import MEASURED_EXECUTION_COST_TIERS_BPS
from src.mhs.run_history import RUN_HISTORY_MAX_SHARDS, RUN_HISTORY_SHARD_MAX_BYTES

logger = logging.getLogger("ExecutionQuality")

EXECUTION_QUALITY_MIN_EVIDENCE_DAYS: int = AUDIT_LOG_RETENTION_DAYS
EXECUTION_QUALITY_SHARD_MAX_BYTES: int = RUN_HISTORY_SHARD_MAX_BYTES
EXECUTION_QUALITY_MAX_SHARDS: int = RUN_HISTORY_MAX_SHARDS

_ACTIVE_FILE_NAME = "active.parquet"
_ARCHIVE_PREFIX = "execution_quality_"
_ARCHIVE_SUFFIX = ".parquet"


@dataclass(frozen=True, slots=True)
class ExecutionQualityRecord:
    decision_time: pd.Timestamp
    symbol: str
    mode: str
    target_weight: float
    mark_price_at_decision: Decimal | None
    avg_fill_price: Decimal | None
    filled_qty: Decimal
    unfilled_qty: Decimal
    status: str
    chases: int
    slippage_bps: float | None


def _slippage_bps(side: str, mark: Decimal | None, fill: Decimal | None) -> float | None:
    if mark is None or fill is None:
        return None
    if mark == Decimal(0):
        return None
    if side == "BUY":
        return float((fill - mark) / mark * Decimal(10_000))
    if side == "SELL":
        return float((mark - fill) / mark * Decimal(10_000))
    return None


def build_execution_quality_records(
    decision_time: pd.Timestamp,
    mode: str,
    weights: pd.Series,
    marks: Mapping[str, Decimal],
    intents: Sequence[Any],
    outcomes: Sequence[Any],
) -> tuple[ExecutionQualityRecord, ...]:
    records: list[ExecutionQualityRecord] = []
    for intent, outcome in zip(intents, outcomes, strict=False):
        symbol = intent.symbol
        # weights is pd.Series; fallback to 0.0 if missing
        try:
            w = weights.get(symbol, 0.0) if hasattr(weights, "get") else 0.0
        except Exception:
            w = 0.0
        try:
            target_weight = float(w) if w is not None else 0.0
        except Exception:
            target_weight = 0.0
        mark = marks.get(symbol) if isinstance(marks, Mapping) else None
        avg_fill = outcome.avg_fill_price
        filled = outcome.filled_qty
        unfilled = outcome.unfilled_qty
        status = outcome.status
        chases = outcome.chases
        side = getattr(intent, "side", "")
        slippage = _slippage_bps(side, mark, avg_fill)
        # Ensure decision_time is tz-aware normalized
        ts = pd.Timestamp(decision_time)
        records.append(
            ExecutionQualityRecord(
                decision_time=ts,
                symbol=symbol,
                mode=mode,
                target_weight=target_weight,
                mark_price_at_decision=mark,
                avg_fill_price=avg_fill,
                filled_qty=filled,
                unfilled_qty=unfilled,
                status=status,
                chases=chases,
                slippage_bps=slippage,
            )
        )
    return tuple(records)


def default_execution_quality_dir() -> Path:
    return DATA_DIR / "state" / "live_execution_quality"


def _archive_path(history_dir: Path, utc_millis: int) -> Path:
    return history_dir / f"{_ARCHIVE_PREFIX}{utc_millis}{_ARCHIVE_SUFFIX}"


def _unique_archive_path(history_dir: Path) -> Path:
    utc_millis = int(time.time() * 1000)
    archive = _archive_path(history_dir, utc_millis)
    while archive.exists():
        utc_millis += 1
        archive = _archive_path(history_dir, utc_millis)
    return archive


def _prune_archives(history_dir: Path) -> None:
    archives = sorted(history_dir.glob(f"{_ARCHIVE_PREFIX}*{_ARCHIVE_SUFFIX}"))
    excess = len(archives) - EXECUTION_QUALITY_MAX_SHARDS
    for stale in archives[:excess]:
        with contextlib.suppress(OSError):
            stale.unlink()


def _records_to_dataframe(records: Sequence[ExecutionQualityRecord]) -> pd.DataFrame:
    rows = [
        {
            "decision_time": r.decision_time.isoformat() if isinstance(r.decision_time, pd.Timestamp) else str(r.decision_time),
            "symbol": r.symbol,
            "mode": r.mode,
            "target_weight": float(r.target_weight),
            "mark_price_at_decision": str(r.mark_price_at_decision) if r.mark_price_at_decision is not None else None,
            "avg_fill_price": str(r.avg_fill_price) if r.avg_fill_price is not None else None,
            "filled_qty": str(r.filled_qty),
            "unfilled_qty": str(r.unfilled_qty),
            "status": r.status,
            "chases": int(r.chases),
            "slippage_bps": float(r.slippage_bps) if r.slippage_bps is not None else None,
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def append_execution_quality(
    records: Sequence[ExecutionQualityRecord], history_dir: Path
) -> Path | None:
    if not records:
        return None
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    active = history_dir / _ACTIVE_FILE_NAME
    df_new = _records_to_dataframe(records)
    # Estimate new byte size via snappy parquet buffer
    buf = io.BytesIO()
    df_new.to_parquet(buf, index=False, compression="snappy")
    new_bytes = buf.getvalue()
    if active.exists() and active.stat().st_size + len(new_bytes) > EXECUTION_QUALITY_SHARD_MAX_BYTES:
        archive = _unique_archive_path(history_dir)
        active.rename(archive)
        _prune_archives(history_dir)
    if active.exists():
        try:
            df_existing = pd.read_parquet(active)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(active, index=False, compression="snappy")
        except Exception:
            # fallback to overwrite with new
            df_new.to_parquet(active, index=False, compression="snappy")
    else:
        df_new.to_parquet(active, index=False, compression="snappy")
    return active


def _load_all_frames(history_dir: Path) -> pd.DataFrame | None:
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


def summarize_execution_quality(
    history_dir: Path | str | None = None, *, since: pd.Timestamp | None = None
) -> dict[str, Any]:
    dir_path = Path(history_dir) if history_dir is not None else default_execution_quality_dir()
    base: dict[str, Any] = {
        "n_cycles": 0,
        "n_fill_records": 0,
        "fill_rate": 0.0,
        "slippage_bps_mean": None,
        "slippage_bps_median": None,
        "slippage_bps_p90": None,
        "by_mode": {},
        "vs_measured_cost_tiers": dict(MEASURED_EXECUTION_COST_TIERS_BPS),
        "n_days_span": 0,
        "sufficient_evidence": False,
    }
    if not dir_path.exists():
        return base
    combined = _load_all_frames(dir_path)
    if combined is None or combined.empty:
        return base
    # Filter since if provided
    if since is not None:
        try:
            since_ts = pd.Timestamp(since)
            since_ts = since_ts.tz_localize("UTC") if since_ts.tzinfo is None else since_ts.tz_convert("UTC")
            # parse decision_time column
            dt_parsed = pd.to_datetime(combined["decision_time"], utc=True, errors="coerce")
            mask = dt_parsed >= since_ts
            combined = combined[mask]
            if combined.empty:
                return base
        except Exception:  # noqa: S110
            pass
    # n_cycles distinct decision_time
    try:
        n_cycles = int(combined["decision_time"].nunique())
    except Exception:  # noqa: S110
        n_cycles = len(combined)
    # n_fill_records where avg_fill_price not null and not NaN
    try:
        fill_mask = combined["avg_fill_price"].notna()
        n_fill_records = int(fill_mask.sum())
    except Exception:  # noqa: S110
        n_fill_records = 0
    total = len(combined)
    fill_rate = float(n_fill_records / total) if total > 0 else 0.0
    # slippage stats where not null
    slippage_mean = None
    slippage_median = None
    slippage_p90 = None
    by_mode: dict[str, Any] = {}
    try:
        # overall slippage series
        slip_series = pd.to_numeric(combined["slippage_bps"], errors="coerce").dropna()
        if not slip_series.empty:
            slippage_mean = float(slip_series.mean())
            slippage_median = float(slip_series.median())
            slippage_p90 = float(slip_series.quantile(0.9))
        # per-mode grouping if mode column exists
        if "mode" in combined.columns:
            for mode_val, group in combined.groupby("mode"):
                g_slip = pd.to_numeric(group["slippage_bps"], errors="coerce").dropna()
                mode_stat: dict[str, Any] = {
                    "count": len(group),
                    "n_fill": int(group["avg_fill_price"].notna().sum()),
                }
                if not g_slip.empty:
                    mode_stat["slippage_bps_mean"] = float(g_slip.mean())
                    mode_stat["slippage_bps_median"] = float(g_slip.median())
                    mode_stat["slippage_bps_p90"] = float(g_slip.quantile(0.9))
                else:
                    mode_stat["slippage_bps_mean"] = None
                    mode_stat["slippage_bps_median"] = None
                    mode_stat["slippage_bps_p90"] = None
                by_mode[str(mode_val)] = mode_stat
    except Exception:  # noqa: S110
        pass
    # n_days_span
    n_days_span = 0
    try:
        dts = pd.to_datetime(combined["decision_time"], utc=True, errors="coerce").dropna()
        if not dts.empty:
            dts = dts.dt.tz_convert("UTC") if dts.dt.tz is not None else dts
            min_dt = dts.min()
            max_dt = dts.max()
            min_norm = min_dt.normalize()
            max_norm = max_dt.normalize()
            n_days_span = int((max_norm - min_norm).days)
    except Exception:  # noqa: S110
        n_days_span = 0
    sufficient = bool(n_days_span >= EXECUTION_QUALITY_MIN_EVIDENCE_DAYS)
    return {
        "n_cycles": n_cycles,
        "n_fill_records": n_fill_records,
        "fill_rate": fill_rate,
        "slippage_bps_mean": slippage_mean,
        "slippage_bps_median": slippage_median,
        "slippage_bps_p90": slippage_p90,
        "by_mode": by_mode,
        "vs_measured_cost_tiers": dict(MEASURED_EXECUTION_COST_TIERS_BPS),
        "n_days_span": n_days_span,
        "sufficient_evidence": sufficient,
    }
