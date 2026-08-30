"""Live execution quality recording — observability-only compression layer over audit logs."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_DIR
from src.live.audit import AUDIT_LOG_RETENTION_DAYS
from src.live.records import append_typed_frame
from src.mhs.params import MEASURED_EXECUTION_COST_TIERS_BPS
from src.mhs.run_history import RUN_HISTORY_MAX_SHARDS, RUN_HISTORY_SHARD_MAX_BYTES

logger = logging.getLogger("ExecutionQuality")

EXECUTION_QUALITY_MIN_EVIDENCE_DAYS: int = AUDIT_LOG_RETENTION_DAYS
EXECUTION_QUALITY_SHARD_MAX_BYTES: int = RUN_HISTORY_SHARD_MAX_BYTES
EXECUTION_QUALITY_MAX_SHARDS: int = RUN_HISTORY_MAX_SHARDS

_EXECUTION_QUALITY_DTYPES: Mapping[str, str] = {
    "decision_time": "datetime64[ns, UTC]",
    "mark_price_at_decision": "float64",
    "avg_fill_price": "float64",
    "filled_qty": "float64",
    "unfilled_qty": "float64",
    "target_weight": "float64",
    "slippage_bps": "float64",
    "latency_seconds": "float64",
    "maker_fill_fraction": "float64",
}


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
    leg_index: int
    run_id: str
    latency_seconds: float | None = None
    sizing_anchor: str = "book_mid"
    maker_fill_fraction: float | None = None


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
        leg_index = int(getattr(intent, "leg_index", 0))  # leg_index=intent.leg_index
        run_id = str(getattr(intent, "client_order_prefix", ""))
        latency_seconds = getattr(outcome, "latency_seconds", None)  # latency_seconds=outcome.latency_seconds
        filled_qty_dec = filled if isinstance(filled, Decimal) else Decimal(str(filled)) if filled is not None else Decimal(0)
        try:
            maker_qty = getattr(outcome, "maker_qty", Decimal(0))
            if filled_qty_dec > 0:
                m = maker_qty if isinstance(maker_qty, Decimal) else Decimal(str(maker_qty))
                maker_fill_fraction = float(m / filled_qty_dec) if filled_qty_dec != 0 else None
            else:
                maker_fill_fraction = None
        except Exception:  # noqa: BLE001
            maker_fill_fraction = None
        sizing_anchor = getattr(outcome, "sizing_anchor", "book_mid") if hasattr(outcome, "sizing_anchor") else "book_mid"
        if isinstance(marks, Mapping) and "sizing_anchor" in marks:
            sizing_anchor = str(marks.get("sizing_anchor", sizing_anchor))
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
                leg_index=leg_index,
                run_id=run_id,
                latency_seconds=latency_seconds,
                sizing_anchor=sizing_anchor,
                maker_fill_fraction=maker_fill_fraction,
            )
        )
    return tuple(records)


def default_execution_quality_dir() -> Path:
    return DATA_DIR / "state" / "live_execution_quality"


def _records_to_dataframe(records: Sequence[ExecutionQualityRecord]) -> pd.DataFrame:
    rows = [
        {
            "decision_time": pd.Timestamp(r.decision_time).tz_localize("UTC") if pd.Timestamp(r.decision_time).tzinfo is None else pd.Timestamp(r.decision_time).tz_convert("UTC"),
            "symbol": r.symbol,
            "mode": r.mode,
            "target_weight": float(r.target_weight),
            "mark_price_at_decision": float(r.mark_price_at_decision) if r.mark_price_at_decision is not None else float("nan"),
            "avg_fill_price": float(r.avg_fill_price) if r.avg_fill_price is not None else float("nan"),
            "filled_qty": float(r.filled_qty),
            "unfilled_qty": float(r.unfilled_qty),
            "status": r.status,
            "chases": int(r.chases),
            "slippage_bps": float(r.slippage_bps) if r.slippage_bps is not None else float("nan"),
            "leg_index": int(r.leg_index),
            "run_id": str(r.run_id),
            "latency_seconds": float(r.latency_seconds) if r.latency_seconds is not None else float("nan"),
            "sizing_anchor": str(r.sizing_anchor),
            "maker_fill_fraction": float(r.maker_fill_fraction) if r.maker_fill_fraction is not None else float("nan"),
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True).astype("datetime64[ns, UTC]")
        for col in ("mark_price_at_decision", "avg_fill_price", "filled_qty", "unfilled_qty", "target_weight", "slippage_bps", "latency_seconds", "maker_fill_fraction"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        for col in ("chases", "leg_index"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
    return df


def append_execution_quality(
    records: Sequence[ExecutionQualityRecord], history_dir: Path
) -> Path | None:
    if not records:
        return None
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    df_new = _records_to_dataframe(records)
    written = append_typed_frame(df_new, history_dir, 'execution_quality', time_column='decision_time', dtypes=_EXECUTION_QUALITY_DTYPES)
    if not written:
        return None
    return written[0]


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
        "latency_seconds_mean": None,
        "latency_seconds_median": None,
        "latency_seconds_p90": None,
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
        n_cycles = int(combined["decision_time"].nunique())
    except Exception:  # noqa: S110
        n_cycles = len(combined)
    try:
        fill_mask = combined["avg_fill_price"].notna()
        n_fill_records = int(fill_mask.sum())
    except Exception:  # noqa: S110
        n_fill_records = 0
    total = len(combined)
    fill_rate = float(n_fill_records / total) if total > 0 else 0.0
    slippage_mean = None
    slippage_median = None
    slippage_p90 = None
    latency_mean = None
    latency_median = None
    latency_p90 = None
    by_mode: dict[str, Any] = {}
    try:
        slip_series = pd.to_numeric(combined["slippage_bps"], errors="coerce").dropna()
        if not slip_series.empty:
            slippage_mean = float(slip_series.mean())
            slippage_median = float(slip_series.median())
            slippage_p90 = float(slip_series.quantile(0.9))
        if "latency_seconds" in combined.columns:
            lat_series = pd.to_numeric(combined["latency_seconds"], errors="coerce").dropna()
            if not lat_series.empty:
                latency_mean = float(lat_series.mean())
                latency_median = float(lat_series.median())
                latency_p90 = float(lat_series.quantile(0.9))
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
                if "latency_seconds" in group.columns:
                    g_lat = pd.to_numeric(group["latency_seconds"], errors="coerce").dropna()
                    if not g_lat.empty:
                        mode_stat["latency_seconds_mean"] = float(g_lat.mean())
                        mode_stat["latency_seconds_median"] = float(g_lat.median())
                        mode_stat["latency_seconds_p90"] = float(g_lat.quantile(0.9))
                    else:
                        mode_stat["latency_seconds_mean"] = None
                        mode_stat["latency_seconds_median"] = None
                        mode_stat["latency_seconds_p90"] = None
                else:
                    mode_stat["latency_seconds_mean"] = None
                    mode_stat["latency_seconds_median"] = None
                    mode_stat["latency_seconds_p90"] = None
                by_mode[str(mode_val)] = mode_stat
    except Exception:  # noqa: S110
        pass
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
        "latency_seconds_mean": latency_mean,
        "latency_seconds_median": latency_median,
        "latency_seconds_p90": latency_p90,
        "by_mode": by_mode,
        "vs_measured_cost_tiers": dict(MEASURED_EXECUTION_COST_TIERS_BPS),
        "n_days_span": n_days_span,
        "sufficient_evidence": sufficient,
    }
