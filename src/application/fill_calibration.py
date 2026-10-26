"""Tape-based fill calibration report for the OHLCV and paper fill rules."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.parquet_io import write_parquet_atomic
from src.core.resources import MhsMemoryBudget

_logger = logging.getLogger(__name__)

_SLICE_COLUMNS = ("transact_time_ms", "agg_trade_id", "price", "quantity", "is_buyer_maker")


def _utc_now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _round_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _round_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_json(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if np.isfinite(f):
            return round(f, 6)
        return None
    return value


@dataclass(frozen=True, slots=True)
class FillCalibrationRequest:
    """Typed controls for one tape-based fill calibration report."""

    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    archive_root: Path
    k_bars: tuple[int, ...] = (1, 3, 5, 10)
    touch_lookback_s: float = 60.0
    queue_ahead_fraction: float = 1.0
    bootstrap_samples: int = 2000
    seed: int = 7
    paper_run_dir: Path | None = None
    live_capture_root: Path | None = None
    data_root: Path | None = None
    memory_budget: MhsMemoryBudget | None = None

    def __post_init__(self) -> None:
        start, end = self.evaluation_start, self.evaluation_end
        for label, value in (("evaluation_start", start), ("evaluation_end", end)):
            if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
                raise ValueError(f"{label} must be a timezone-aware Timestamp, got {value!r}")
            if pd.Timestamp(value).utcoffset() != pd.Timedelta(0):
                raise ValueError(f"{label} must be UTC, got {value!r}")
        s_utc = pd.Timestamp(start).tz_convert("UTC")
        e_utc = pd.Timestamp(end).tz_convert("UTC")
        if not s_utc < e_utc:
            raise ValueError(f"require evaluation_start < evaluation_end, got {s_utc} {e_utc}")
        kb = self.k_bars
        if (
            not isinstance(kb, tuple)
            or not kb
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) <= 0 for v in kb)
        ):
            raise ValueError(f"k_bars must be a non-empty tuple of positive ints, got {kb!r}")
        ints = [int(v) for v in kb]
        if len(set(ints)) != len(ints) or tuple(sorted(ints)) != tuple(ints):
            raise ValueError(f"k_bars must be sorted unique positive ints, got {kb!r}")
        if (
            not isinstance(self.touch_lookback_s, (int, float))
            or isinstance(self.touch_lookback_s, bool)
            or not np.isfinite(float(self.touch_lookback_s))
            or not float(self.touch_lookback_s) > 0
        ):
            raise ValueError(f"touch_lookback_s must be > 0, got {self.touch_lookback_s!r}")
        if (
            not isinstance(self.bootstrap_samples, (int, np.integer))
            or isinstance(self.bootstrap_samples, bool)
            or int(self.bootstrap_samples) < 100
        ):
            raise ValueError(f"bootstrap_samples must be >= 100, got {self.bootstrap_samples!r}")
        paper_set = self.paper_run_dir is not None
        live_set = self.live_capture_root is not None
        if paper_set != live_set:
            raise ValueError("paper_run_dir and live_capture_root must be set together or both omitted")
        for label in ("archive_root", "paper_run_dir", "live_capture_root", "data_root"):
            value = getattr(self, label)
            if value is not None and not isinstance(value, Path):
                raise ValueError(f"{label} must be a Path, got {value!r}")
        from src.engine.execution.tape_fill import TapeFillSpec

        TapeFillSpec(queue_ahead_fraction=self.queue_ahead_fraction)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if int(self.touch_lookback_s * 1000) < 1:
            raise ValueError("touch_lookback_s must be at least one millisecond")


@dataclass(frozen=True, slots=True)
class FillCalibrationReport:
    """Persisted outcome of one fill calibration run."""

    backtest: dict[str, Any]
    paper: dict[str, Any] | None
    exclusions: dict[str, int]
    intents_path: Path
    report_path: Path


def _kind_of(q_prev: float, q_new: float) -> str:
    if q_prev == 0.0 and q_new != 0.0:
        return "open"
    if q_new == 0.0 and q_prev != 0.0:
        return "close"
    if q_prev * q_new < 0.0:
        return "flip"
    if abs(q_new) > abs(q_prev):
        return "increase"
    return "reduce"


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _bootstrap_diff_ci(
    days: np.ndarray,
    day_of_row: np.ndarray,
    metric_rule: np.ndarray,
    metric_base: np.ndarray,
    weights: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    uniq = np.unique(day_of_row)
    n_days = len(uniq)
    if n_days == 0:
        return float("nan"), float("nan"), float("nan")
    _, inverse = np.unique(day_of_row, return_inverse=True)
    numerators = np.bincount(inverse, weights=(metric_rule - metric_base) * weights)
    denominators = np.bincount(inverse, weights=weights)
    rng = np.random.default_rng(int(seed))
    point_num_r = float(np.sum(metric_rule * weights))
    point_num_b = float(np.sum(metric_base * weights))
    denom = float(np.sum(weights))
    point = (point_num_r - point_num_b) / denom if denom > 0 else float("nan")
    diffs = np.empty(samples, dtype="float64")
    batch_size = max(1, min(samples, 65536 // n_days))
    for start in range(0, samples, batch_size):
        picks = rng.integers(0, n_days, size=(min(batch_size, samples - start), n_days))
        num = numerators[picks].sum(axis=1)
        den = denominators[picks].sum(axis=1)
        diffs[start : start + len(picks)] = np.divide(num, den, out=np.full(len(picks), np.nan), where=den > 0)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return point, float("nan"), float("nan")
    return point, min(point, float(np.percentile(finite, 2.5))), max(point, float(np.percentile(finite, 97.5)))


def _load_backtest_inputs(request: FillCalibrationRequest) -> tuple[Any, Any, Any, Any, Any, pd.DatetimeIndex]:
    from src.application.strategy_account import AccountReplayRequest, build_account_replay_inputs
    from src.common.paths import STRATEGY_BACKTESTS_DIR, VENUE_RULES_DIR
    from src.core.params import DISCOVERY_START

    replay_request = AccountReplayRequest(
        source_start=DISCOVERY_START,
        evaluation_start=request.evaluation_start,
        evaluation_end=request.evaluation_end,
        runs_root=STRATEGY_BACKTESTS_DIR,
        venue_rules_root=VENUE_RULES_DIR,
        execution="maker",
        data_root=request.data_root,
        memory_budget=request.memory_budget,
    )
    return build_account_replay_inputs(replay_request)


def _build_backtest_intents(parts: tuple[Any, ...]) -> list[dict[str, Any]]:
    from src.strategy.targets import FLOW_MOM_TOP20

    unit_weights, marks, _funding, _adv, _sigma, anchor_times = parts
    cutoff = FLOW_MOM_TOP20.design_data_cutoff
    closes = marks.close
    bar_index = closes.index
    if not all(panel.index.equals(bar_index) and panel.columns.equals(closes.columns) for panel in (marks.high, marks.low)):
        raise DataIntegrityError("mark panels must have aligned indexes and columns")
    if not bar_index.is_unique or not bar_index.is_monotonic_increasing:
        raise DataIntegrityError("mark bars must be unique and ordered")
    anchor_pos = bar_index.get_indexer(anchor_times)
    intents: list[dict[str, Any]] = []
    q_prev: dict[str, float] = dict.fromkeys(unit_weights.columns, 0.0)
    dates = unit_weights.index
    if len(anchor_times) != len(dates) or not dates.is_unique or not dates.is_monotonic_increasing:
        raise DataIntegrityError("entry dates and anchors must be aligned and ordered")
    if not np.isfinite(unit_weights.to_numpy(dtype="float64")).all():
        raise DataIntegrityError("unit weights must be finite")
    for i in range(len(dates)):
        entry = dates[i]
        anchor = anchor_times[i]
        period = "holdout" if pd.Timestamp(entry).tz_convert("UTC") > cutoff else "design"
        pos = int(anchor_pos[i])
        if pos < 0:
            raise DataIntegrityError(f"anchor bar absent from mark panels: {anchor}")
        for symbol in unit_weights.columns:
            w = float(unit_weights.iloc[i][symbol])
            try:
                a = float(closes.iloc[pos][symbol])
            except (KeyError, IndexError, ValueError) as exc:
                raise DataIntegrityError(f"malformed mark panels: {exc}") from exc
            if not np.isfinite(a) or not a > 0:
                continue
            held = float(q_prev[symbol]) * a
            delta = float(w) - held
            if not np.isfinite(delta) or not abs(delta) > 0:
                continue
            q_new = float(w) / a
            qp = float(q_prev[symbol])
            side: Literal["BUY", "SELL"] = "BUY" if delta > 0 else "SELL"
            intents.append(
                {
                    "symbol": str(symbol),
                    "entry": pd.Timestamp(entry).tz_convert("UTC"),
                    "anchor": pd.Timestamp(anchor).tz_convert("UTC"),
                    "anchor_pos": pos,
                    "anchor_price": a,
                    "delta": delta,
                    "notional": abs(delta),
                    "side": side,
                    "kind": _kind_of(qp, q_new),
                    "period": period,
                    "year": int(pd.Timestamp(entry).tz_convert("UTC").year),
                    "day": pd.Timestamp(entry).tz_convert("UTC").normalize(),
                }
            )
            q_prev[symbol] = q_new
    return intents


def _read_tape_partition(path: Path) -> pd.DataFrame:
    """Read only the five slice columns, rejecting incomplete partition schemas."""
    import pyarrow.parquet as pq

    try:
        missing = set(_SLICE_COLUMNS) - set(pq.ParquetFile(path).schema_arrow.names)
        if missing:
            raise DataIntegrityError(f"tape partition {path} missing columns: {sorted(missing)}")
        return pd.read_parquet(path, columns=list(_SLICE_COLUMNS))
    except DataIntegrityError:
        raise
    except Exception as exc:
        raise DataIntegrityError(f"unreadable tape partition {path}: {exc}") from exc


def _slice_interval(day: date, slice_hours: tuple[int, ...]) -> tuple[pd.Timestamp, pd.Timestamp]:
    base = pd.Timestamp(day, tz="UTC")
    return base + pd.Timedelta(hours=slice_hours[0]), base + pd.Timedelta(hours=slice_hours[1])


def _intent_day_and_window(
    intent: dict[str, Any], bar_ms: int, max_k: int, slice_hours: tuple[int, ...]
) -> tuple[date, int, int]:
    anchor_ms = int(pd.Timestamp(intent["anchor"]).value // 10**6)
    active_from = anchor_ms + bar_ms
    active_until = anchor_ms + (max_k + 1) * bar_ms
    day = pd.Timestamp(active_from, unit="ms", tz="UTC").date()
    return day, active_from, active_until


def _score_backtest(
    request: FillCalibrationRequest,
    intents: list[dict[str, Any]],
    parts: tuple[Any, ...],
    exclusions: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.core.params import ACCOUNT_MAKER_FEE_BPS, ACCOUNT_TAKER_FEE_BPS
    from src.engine.execution.tape_fill import (
        RestingOrder,
        TapeFillSpec,
        TapeTrades,
        post_only_limit,
        simulate_resting_fill,
        touch_proxy,
    )
    from src.live.executor import EXECUTION_BAR_SECONDS
    from src.market_data.public_archive import catalog, store

    _unit, marks, _f, _a, _s, _t = parts
    bar_ms = int(float(EXECUTION_BAR_SECONDS) * 1000)
    slice_hours = tuple(catalog.TAPE_SLICE_HOURS)
    max_k = max(int(v) for v in request.k_bars)
    lookback_ms = int(float(request.touch_lookback_s) * 1000)
    coverage = store.load_coverage(request.archive_root, "aggTrades")
    closes = marks.close
    highs = marks.high
    lows = marks.low
    bar_index = closes.index

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, intent in enumerate(intents):
        _day, _af, _au = _intent_day_and_window(intent, bar_ms, max_k, slice_hours)
        month = pd.Timestamp(_af, unit="ms", tz="UTC").strftime("%Y%m")
        grouped[(intent["symbol"], month)].append(idx)

    total_groups = len(grouped)
    scored_rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    last_log = t0
    done = 0
    tape: TapeTrades | None = None
    ordered_keys = sorted(grouped)
    for symbol, month in ordered_keys:
        idxs = grouped[(symbol, month)]
        try:
            path = store.partition_path(request.archive_root, "aggTrades", symbol, month, kind="slice")
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError(f"malformed tape partition key: {exc}") from exc
        frame: pd.DataFrame | None = None
        for idx in idxs:
            now = time.monotonic()
            if now - last_log >= 60.0:
                _logger.info("[ALGO] stage=fill_calibration cohort=backtest done=%d total=%d elapsed_s=%.1f", done, total_groups, now - t0)
                last_log = now
            intent = intents[idx]
            day, active_from, active_until = _intent_day_and_window(intent, bar_ms, max_k, slice_hours)
            row = coverage.get((symbol, day))
            if row is None or row.status != "ok":
                exclusions["no_tape_coverage"] = exclusions.get("no_tape_coverage", 0) + 1
                continue
            slice_start, slice_end = _slice_interval(day, slice_hours)
            slice_start_ms = int(slice_start.value // 10**6)
            slice_end_ms = int(slice_end.value // 10**6)
            if not (active_from >= slice_start_ms and active_until <= slice_end_ms):
                exclusions["window_outside_slice"] = exclusions.get("window_outside_slice", 0) + 1
                continue
            if frame is None and path.exists():
                frame = _read_tape_partition(path)
            if frame is None:
                raise DataIntegrityError(f"tape partition missing for covered {(symbol, day)}: {path}")
            day_frame = frame[
                (frame["transact_time_ms"] >= slice_start_ms) & (frame["transact_time_ms"] < slice_end_ms)
            ]
            try:
                tape = TapeTrades.from_slice_frame(day_frame, coverage_ms=(slice_start_ms, slice_end_ms))
            except DataIntegrityError as exc:
                raise DataIntegrityError(f"malformed tape slice {(symbol, day)}: {exc}") from exc
            try:
                proxy = touch_proxy(tape, active_from, lookback_ms)
            except DataIntegrityError:
                proxy = None
            a = float(intent["anchor_price"])
            side = intent["side"]
            side_lit: Literal["BUY", "SELL"] = "BUY" if side == "BUY" else "SELL"
            marketable = False
            feasible_limit: float | None = None
            if proxy is not None:
                bid_px, ask_px = proxy
                marketable = (a >= ask_px) if side == "BUY" else (a <= bid_px)
                feasible_limit = post_only_limit(side_lit, a, proxy)
            else:
                exclusions["no_touch_proxy"] = exclusions.get("no_touch_proxy", 0) + 1
            pos = int(intent["anchor_pos"])
            n_bars = len(bar_index)
            expected = pd.date_range(intent["anchor"], periods=max_k + 1, freq=pd.Timedelta(milliseconds=bar_ms))
            if not bar_index[pos : pos + max_k + 1].equals(expected):
                raise DataIntegrityError("mark panels do not contain the complete calibration window")
            for k in request.k_bars:
                kk = int(k)
                w_end = min(pos + 1 + kk, n_bars)
                window_close = (
                    closes.iloc[pos + 1 : w_end][symbol].to_numpy(dtype="float64") if w_end > pos + 1 else np.empty(0)
                )
                window_low = (
                    lows.iloc[pos + 1 : w_end][symbol].to_numpy(dtype="float64") if w_end > pos + 1 else np.empty(0)
                )
                window_high = (
                    highs.iloc[pos + 1 : w_end][symbol].to_numpy(dtype="float64") if w_end > pos + 1 else np.empty(0)
                )
                if side == "BUY":
                    hit = np.flatnonzero(np.isfinite(window_low) & (window_low < a))
                else:
                    hit = np.flatnonzero(np.isfinite(window_high) & (window_high > a))
                ohlcv_fill = 1.0 if hit.size else 0.0
                ohlcv_time = (
                    float((int(hit[0]) + 1) * float(EXECUTION_BAR_SECONDS) / 60.0) if hit.size else float("nan")
                )
                finite_closes = window_close[np.isfinite(window_close)]
                c_k = float(finite_closes[-1]) if finite_closes.size else a
                ohlcv_is = (side_sign(side) * (a - a) / a * 1e4 + ACCOUNT_MAKER_FEE_BPS) * ohlcv_fill + (
                    1.0 - ohlcv_fill
                ) * (side_sign(side) * (c_k - a) / a * 1e4 + ACCOUNT_TAKER_FEE_BPS)
                scored_rows.append(
                    {
                        "cohort": "backtest",
                        "symbol": symbol,
                        "entry": intent["entry"],
                        "anchor": intent["anchor"],
                        "period": intent["period"],
                        "year": intent["year"],
                        "day": intent["day"],
                        "side": side,
                        "kind": intent["kind"],
                        "notional": float(intent["notional"]),
                        "anchor_price": a,
                        "k": kk,
                        "rule": "ohlcv_strict",
                        "fill": float(ohlcv_fill),
                        "fill_time_min": ohlcv_time,
                        "is_bps": float(ohlcv_is),
                        "limit": a,
                        "close_k": c_k,
                        "marketable_anchor": bool(marketable),
                    }
                )
                tape_specs: list[tuple[str, float, Literal["through_only", "front_of_queue"]]] = [
                    ("tape_through_anchor", a, "through_only")
                ]
                if feasible_limit is not None:
                    tape_specs.append(("tape_through_feasible", feasible_limit, "through_only"))
                    tape_specs.append(("tape_front_feasible", feasible_limit, "front_of_queue"))
                for rule, limit, unknown in tape_specs:
                    qty = float(intent["notional"]) / a if a > 0 else 0.0
                    order = RestingOrder(
                        side=side_lit,
                        limit_price=float(limit),
                        quantity=qty,
                        active_from_ms=active_from,
                        active_until_ms=anchor_ms(active_from, kk, bar_ms),
                        queue_ahead_qty=None,
                    )
                    result = simulate_resting_fill(order, tape, TapeFillSpec(unknown_queue=unknown))
                    frac = float(result.filled_qty / qty)
                    frac = min(max(frac, 0.0), 1.0)
                    tmin = (
                        float((int(result.first_fill_ms) - active_from) / 60000.0)
                        if result.first_fill_ms is not None
                        else float("nan")
                    )
                    is_v = frac * (side_sign(side) * (float(limit) - a) / a * 1e4 + ACCOUNT_MAKER_FEE_BPS) + (
                        1.0 - frac
                    ) * (side_sign(side) * (c_k - a) / a * 1e4 + ACCOUNT_TAKER_FEE_BPS)
                    scored_rows.append(
                        {
                            "cohort": "backtest",
                            "symbol": symbol,
                            "entry": intent["entry"],
                            "anchor": intent["anchor"],
                            "period": intent["period"],
                            "year": intent["year"],
                            "day": intent["day"],
                            "side": side,
                            "kind": intent["kind"],
                            "notional": float(intent["notional"]),
                            "anchor_price": a,
                            "k": kk,
                            "rule": rule,
                            "fill": frac,
                            "fill_time_min": tmin,
                            "is_bps": float(is_v),
                            "limit": float(limit),
                            "close_k": c_k,
                            "marketable_anchor": bool(marketable),
                        }
                    )
        done += 1
        frame = None
        tape = None
        day_frame = None
        now = time.monotonic()
        if now - last_log >= 60.0 or done == total_groups:
            _logger.info(
                "[ALGO] stage=fill_calibration cohort=backtest done=%d total=%d elapsed_s=%.1f",
                done,
                total_groups,
                now - t0,
            )
            last_log = now
    summary = _summarize_backtest(scored_rows, request, exclusions)
    _logger.info(
        "[ALGO] stage=fill_calibration cohort=backtest status=done intents=%d exclusions=%s",
        len(scored_rows),
        dict(exclusions),
    )
    return scored_rows, summary


def anchor_ms(active_from: int, k: int, bar_ms: int) -> int:
    return int(active_from + k * bar_ms)


def side_sign(side: str) -> float:
    return 1.0 if side == "BUY" else -1.0


def _summarize_backtest(
    rows: list[dict[str, Any]], request: FillCalibrationRequest, exclusions: dict[str, int]
) -> dict[str, Any]:
    if not rows:
        return {"periods": {}}
    frame = pd.DataFrame(rows)
    periods: dict[str, Any] = {}
    for period, g in frame.groupby("period"):
        per_rule: dict[str, Any] = {}
        market_w = g[g["rule"] == "ohlcv_strict"]["notional"].to_numpy(dtype="float64")
        market_flags = g[g["rule"] == "ohlcv_strict"]["marketable_anchor"].to_numpy(dtype=bool)
        market_share = (
            float(np.sum(market_w[market_flags]) / np.sum(market_w)) if np.sum(market_w) > 0 else float("nan")
        )
        for (rule, k), gg in g.groupby(["rule", "k"]):
            kk = int(k)
            fills = gg["fill"].to_numpy(dtype="float64")
            iss = gg["is_bps"].to_numpy(dtype="float64")
            w = gg["notional"].to_numpy(dtype="float64")
            times = gg["fill_time_min"].to_numpy(dtype="float64")
            filled_times = times[np.isfinite(times) & (fills > 0)]
            key = f"{rule}@K{kk}"
            base = g[(g["rule"] == "ohlcv_strict") & (g["k"] == kk)]
            entry: dict[str, Any] = {
                "n_intents": len(gg),
                "fill_fraction": _weighted_mean(fills, w),
                "is_bps": _weighted_mean(iss, w),
                "fill_time_p50_min": _percentile(filled_times, 50.0),
                "fill_time_p90_min": _percentile(filled_times, 90.0),
            }
            if rule != "ohlcv_strict" and len(base):
                paired = gg.merge(base, on=["symbol", "entry", "k"], suffixes=("", "_base"), validate="one_to_one")
                bfill = paired["fill_base"].to_numpy(dtype="float64")
                bis = paired["is_bps_base"].to_numpy(dtype="float64")
                bday = paired["day"].to_numpy()
                gday = gg["day"].to_numpy()
                pf, lo_f, hi_f = _bootstrap_diff_ci(
                    bday, gday, fills, bfill, w, samples=int(request.bootstrap_samples), seed=int(request.seed)
                )
                pi, lo_i, hi_i = _bootstrap_diff_ci(
                    bday, gday, iss, bis, w, samples=int(request.bootstrap_samples), seed=int(request.seed)
                )
                entry["diff_fill_vs_ohlcv"] = pf
                entry["diff_fill_ci95"] = [lo_f, hi_f]
                entry["diff_is_vs_ohlcv"] = pi
                entry["diff_is_ci95"] = [lo_i, hi_i]
                both = paired[(paired["fill"] > 0) & (paired["fill_base"] > 0)]
                time_diff, time_lo, time_hi = _bootstrap_diff_ci(
                    both["day"].to_numpy(), both["day"].to_numpy(),
                    both["fill_time_min"].to_numpy(dtype="float64"),
                    both["fill_time_min_base"].to_numpy(dtype="float64"),
                    both["notional"].to_numpy(dtype="float64"),
                    samples=request.bootstrap_samples, seed=request.seed,
                )
                entry["diff_fill_time_min_vs_ohlcv"] = time_diff
                entry["diff_fill_time_ci95"] = [time_lo, time_hi]
            per_rule[key] = entry
        per_year: dict[str, Any] = {}
        for (rule, k, year), gg in g.groupby(["rule", "k", "year"]):
            fills = gg["fill"].to_numpy(dtype="float64")
            iss = gg["is_bps"].to_numpy(dtype="float64")
            w = gg["notional"].to_numpy(dtype="float64")
            per_year[f"{rule}@K{int(k)}@{int(year)}"] = {
                "n_intents": len(gg),
                "fill_fraction": _weighted_mean(fills, w),
                "is_bps": _weighted_mean(iss, w),
            }
        periods[str(period)] = {
            "rules": per_rule,
            "marketable_anchor_share": market_share,
            "per_year": per_year,
        }
    return {"periods": periods}


def _parse_ts(value: Any) -> int:
    if value is None:
        raise DataIntegrityError("audit record missing ts")
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError("timestamp cannot be NaT")
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed audit ts: {value!r}") from exc
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.tz_convert("UTC").value // 10**6)


def _load_paper_posts(paper_run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    manifest_path = paper_run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"malformed paper run manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("mode") != "paper":
        raise DataIntegrityError(f"non-paper run directory rejected: {manifest_path}")
    audit_dir = paper_run_dir / "audit"
    if not audit_dir.is_dir():
        raise DataIntegrityError(f"paper audit directory missing: {audit_dir}")
    files = sorted(audit_dir.glob("*.jsonl"))
    if not files:
        raise DataIntegrityError(f"paper audit directory holds no jsonl: {audit_dir}")
    posts: dict[str, dict[str, Any]] = {}
    cancels: dict[str, int] = {}
    fills: dict[str, list[tuple[int, float]]] = defaultdict(list)
    sides: dict[str, str] = {}
    journal_path = paper_run_dir / "order_journal.jsonl"
    if journal_path.exists():
        try:
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("journal record must be an object")
                if record.get("event") == "submit" and record.get("side") in ("BUY", "SELL"):
                    sides[record["client_order_id"]] = record["side"]
        except (OSError, ValueError, KeyError) as exc:
            raise DataIntegrityError(f"malformed paper order journal: {exc}") from exc
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DataIntegrityError(f"unreadable paper audit {path}: {exc}") from exc
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise DataIntegrityError(f"malformed paper audit line: {line!r}") from exc
            if not isinstance(record, dict):
                raise DataIntegrityError(f"malformed paper audit line: {line!r}")
            event = record.get("event")
            cid = record.get("client_order_id")
            if not isinstance(cid, str) or not cid:
                continue
            if event == "order_posted":
                if record.get("time_in_force") != "GTX":
                    continue
                try:
                    price = float(record["price"])
                    qty = float(record["quantity"])
                except (KeyError, ValueError, TypeError) as exc:
                    raise DataIntegrityError(f"malformed order_posted record: {exc}") from exc
                if not (np.isfinite(price) and price > 0 and np.isfinite(qty) and qty > 0):
                    raise DataIntegrityError(f"malformed order_posted record: {record!r}")
                posts[cid] = {
                    "symbol": str(record.get("symbol")),
                    "price": price,
                    "quantity": qty,
                    "ts_ms": _parse_ts(record.get("ts")),
                    "side": None,
                }
                side_raw = str(record.get("side", "")).upper()
                if side_raw in ("BUY", "SELL"):
                    posts[cid]["side"] = side_raw
            elif event == "order_cancelled":
                try:
                    cancel = _parse_ts(record.get("ts"))
                    cancels[cid] = min(cancels.get(cid, cancel), cancel)
                except DataIntegrityError as exc:
                    raise DataIntegrityError(f"malformed order_cancelled record: {exc}") from exc
            elif event == "fill":
                try:
                    stamp = _parse_ts(record.get("ts"))
                    quantity = float(record["qty"])
                    if not np.isfinite(quantity) or quantity <= 0:
                        raise ValueError("fill quantity must be finite and positive")
                    fills[cid].append((stamp, quantity))
                except (KeyError, ValueError, TypeError) as exc:
                    raise DataIntegrityError(f"malformed fill record: {exc}") from exc
    for cid, post in posts.items():
        if post["side"] is None:
            post["side"] = sides.get(cid)
        elif cid in sides and sides[cid] != post["side"]:
            raise DataIntegrityError(f"paper side disagrees with journal for {cid}")
    return {"posts": posts, "cancels": cancels, "fills": fills}, {}


def _load_depth_index(live_capture_root: Path, days: set[str] | None = None) -> dict[str, pd.DataFrame]:
    base = live_capture_root / "exec_depth"
    if not base.exists():
        return {}
    paths = sorted(base.rglob("*.parquet"))
    by_symbol: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for path in paths:
        if days is not None and path.parent.name not in days:
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataIntegrityError(f"unreadable exec_depth {path}: {exc}") from exc
        if "symbol" not in frame.columns:
            raise DataIntegrityError(f"exec_depth {path} missing symbol column")
        time_col = None
        for candidate in ("received_at", "transact_time_ms", "event_time_ms"):
            if candidate in frame.columns:
                time_col = candidate
                break
        if time_col is None:
            raise DataIntegrityError(f"exec_depth {path} missing time column")
        if time_col == "received_at":
            times = pd.to_datetime(frame["received_at"], utc=True).dt.as_unit("ns").astype("int64") // 10**6
        else:
            times = pd.to_numeric(frame[time_col], errors="coerce")
        work = frame.copy()
        work["_row_ms"] = np.asarray(times, dtype="float64")
        for symbol, part in work.groupby("symbol"):
            by_symbol[str(symbol)].append(part)
    out: dict[str, pd.DataFrame] = {}
    for symbol, parts in by_symbol.items():
        merged = pd.concat(parts, ignore_index=True).sort_values("_row_ms", kind="mergesort").reset_index(drop=True)
        out[symbol] = merged
    return out


def _depth_levels(frame: pd.DataFrame) -> int:
    n = 0
    while n < 5 and f"bid_px_{n}" in frame.columns and f"ask_px_{n}" in frame.columns:
        n += 1
    return n


def _queue_ahead(side: str, limit: float, row: pd.Series, n_levels: int) -> tuple[float | None, float | None]:
    prefix = "bid" if side == "BUY" else "ask"
    try:
        quantities = [float(row[f"{prefix}_qty_{i}"]) for i in range(n_levels)]
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed exec_depth quantities: {exc}") from exc
    if any(q < 0 for q in quantities):
        raise DataIntegrityError("exec_depth quantities must be non-negative")
    if side == "BUY":
        bids = [(float(row[f"bid_px_{i}"]), float(row[f"bid_qty_{i}"])) for i in range(n_levels)]
        bids = [(p, q) for p, q in bids if np.isfinite(p) and np.isfinite(q) and p > 0]
        if not bids:
            return None, None
        if limit > bids[0][0]:
            return 0.0, None
        for _i, (px, qty) in enumerate(bids):
            if limit == px:
                return float(qty), float(qty)
        for pos in range(len(bids) - 1):
            if bids[pos][0] > limit > bids[pos + 1][0]:
                return 0.0, None
        return None, None
    asks = [(float(row[f"ask_px_{i}"]), float(row[f"ask_qty_{i}"])) for i in range(n_levels)]
    asks = [(p, q) for p, q in asks if np.isfinite(p) and np.isfinite(q) and p > 0]
    if not asks:
        return None, None
    if limit < asks[0][0]:
        return 0.0, None
    for px, qty in asks:
        if limit == px:
            return float(qty), float(qty)
    for i in range(len(asks) - 1):
        if asks[i][0] < limit < asks[i + 1][0]:
            return 0.0, None
    return None, None


def _score_paper(
    request: FillCalibrationRequest,
    exclusions: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    from src.engine.execution.tape_fill import RestingOrder, TapeFillSpec, TapeTrades, simulate_resting_fill
    from src.market_data.public_archive import catalog, store

    assert request.paper_run_dir is not None
    assert request.live_capture_root is not None
    bundle, _ = _load_paper_posts(request.paper_run_dir)
    posts, cancels, fills_map = bundle["posts"], bundle["cancels"], bundle["fills"]
    start_ms = int(request.evaluation_start.value // 10**6)
    end_ms = int(request.evaluation_end.value // 10**6)
    posts = {cid: post for cid, post in posts.items() if start_ms <= post["ts_ms"] < end_ms}
    depth_days = {
        (pd.Timestamp(post["ts_ms"], unit="ms", tz="UTC") - pd.Timedelta(days=offset)).strftime("%Y%m%d")
        for post in posts.values() for offset in (0, 1)
    }
    depth_index = _load_depth_index(request.live_capture_root, depth_days)
    coverage = store.load_coverage(request.archive_root, "aggTrades")
    slice_hours = tuple(catalog.TAPE_SLICE_HOURS)
    rows: list[dict[str, Any]] = []
    variants = ("through_only", "queue_observed", "front_of_queue")
    confusion: dict[str, dict[str, int]] = {
        v: {"both_filled": 0, "paper_only": 0, "tape_only": 0, "neither": 0} for v in variants
    }
    fallback_count = 0
    t0 = time.monotonic()
    last_log = t0
    items = sorted(posts.items(), key=lambda item: (item[1]["symbol"], item[1]["ts_ms"], item[0]))
    month_keys = sorted({(post["symbol"], pd.Timestamp(post["ts_ms"], unit="ms", tz="UTC").strftime("%Y%m")) for post in posts.values()})
    group_positions = {key: index for index, key in enumerate(month_keys)}
    partition_key: tuple[str, str] | None = None
    partition: pd.DataFrame | None = None
    for order_id, post in items:
        now = time.monotonic()
        if now - last_log >= 60.0:
            group_key = (post["symbol"], pd.Timestamp(post["ts_ms"], unit="ms", tz="UTC").strftime("%Y%m"))
            _logger.info("[ALGO] stage=fill_calibration cohort=paper done=%d total=%d elapsed_s=%.1f", group_positions[group_key], len(month_keys), now - t0)
            last_log = now
        symbol = post["symbol"]
        limit = float(post["price"])
        qty = float(post["quantity"])
        active_from = int(post["ts_ms"])
        cancel_ms = cancels.get(order_id)
        fill_list = sorted(fills_map.get(order_id, []))
        completing = None
        cumulative = 0.0
        for stamp, increment in fill_list:
            if stamp >= active_from and (cancel_ms is None or stamp <= cancel_ms):
                cumulative += increment
                if cumulative >= qty * (1.0 - 1e-12):
                    completing = stamp
                    break
        if completing is None and cancel_ms is None:
            exclusions["paper_post_unresolved"] = exclusions.get("paper_post_unresolved", 0) + 1
            continue
        ends: list[int] = []
        if cancel_ms is not None:
            ends.append(int(cancel_ms))
        if completing is not None:
            ends.append(int(completing))
        active_until = min(ends) + 1
        if not active_until > active_from:
            exclusions["paper_post_unresolved"] = exclusions.get("paper_post_unresolved", 0) + 1
            continue
        paper_filled = any(active_from <= stamp < active_until for stamp, _ in fill_list)
        day = pd.Timestamp(active_from, unit="ms", tz="UTC").date()
        row_cov = coverage.get((symbol, day))
        if row_cov is None or row_cov.status != "ok":
            exclusions["paper_no_tape_coverage"] = exclusions.get("paper_no_tape_coverage", 0) + 1
            continue
        slice_start, slice_end = _slice_interval(day, slice_hours)
        s_ms = int(slice_start.value // 10**6)
        e_ms = int(slice_end.value // 10**6)
        if not (active_from >= s_ms and active_until <= e_ms):
            exclusions["paper_no_tape_coverage"] = exclusions.get("paper_no_tape_coverage", 0) + 1
            continue
        month = pd.Timestamp(active_from, unit="ms", tz="UTC").strftime("%Y%m")
        try:
            path = store.partition_path(request.archive_root, "aggTrades", symbol, month, kind="slice")
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError(f"malformed tape partition key: {exc}") from exc
        if not path.exists():
            raise DataIntegrityError(f"tape partition missing for covered {(symbol, day)}: {path}")
        key = (symbol, month)
        if key != partition_key:
            partition = None
            tape = None
            day_frame = None
            partition = _read_tape_partition(path)
            partition_key = key
        assert partition is not None
        day_frame = partition[(partition["transact_time_ms"] >= s_ms) & (partition["transact_time_ms"] < e_ms)]
        try:
            tape = TapeTrades.from_slice_frame(day_frame.loc[:, list(_SLICE_COLUMNS)], coverage_ms=(s_ms, e_ms))
        except DataIntegrityError as exc:
            raise DataIntegrityError(f"malformed tape slice {(symbol, day)}: {exc}") from exc
        side = post.get("side")
        if side not in ("BUY", "SELL"):
            raise DataIntegrityError(f"paper order {order_id} has no side; require audit side or order_journal.jsonl submit")
        side_lit: Literal["BUY", "SELL"] = "BUY" if side == "BUY" else "SELL"
        depth_frame = depth_index.get(symbol)
        queue_est: float | None = None
        level_qty: float | None = None
        if depth_frame is not None and len(depth_frame):
            times = depth_frame["_row_ms"].to_numpy(dtype="float64")
            eligible = np.flatnonzero(np.isfinite(times) & (times <= active_from))
            if eligible.size:
                pos_idx = int(eligible[-1])
                row_time = float(times[pos_idx])
                if active_from - row_time <= 2000.0:
                    n_levels = _depth_levels(depth_frame)
                    if n_levels:
                        qe, lq = _queue_ahead(side, limit, depth_frame.iloc[pos_idx], n_levels)
                        queue_est, level_qty = qe, lq
        queue_used = queue_est if queue_est is not None else None
        if queue_used is None:
            fallback_count += 1
        specs: dict[str, tuple[float | None, Literal["through_only", "front_of_queue"]]] = {
            "through_only": (None, "through_only"),
            "queue_observed": (queue_used, "through_only"),
            "front_of_queue": (None, "front_of_queue"),
        }
        tape_frac: dict[str, float] = {}
        tape_first: dict[str, float] = {}
        for variant, (q_val, unknown) in specs.items():
            order = RestingOrder(
                side=side_lit,
                limit_price=limit,
                quantity=qty,
                active_from_ms=active_from,
                active_until_ms=active_until,
                queue_ahead_qty=None,
            )
            spec = TapeFillSpec(queue_ahead_fraction=float(request.queue_ahead_fraction), unknown_queue=unknown)
            if q_val is not None:
                order = RestingOrder(
                    side=side_lit,
                    limit_price=limit,
                    quantity=qty,
                    active_from_ms=active_from,
                    active_until_ms=active_until,
                    queue_ahead_qty=float(q_val),
                )
            result = simulate_resting_fill(order, tape, spec)
            frac = float(result.filled_qty / qty) if qty > 0 else 0.0
            tape_frac[variant] = min(max(frac, 0.0), 1.0)
            tape_first[variant] = (
                float((int(result.first_fill_ms) - active_from) / 60000.0)
                if result.first_fill_ms is not None
                else float("nan")
            )
        observed_times = [stamp for stamp, _ in fill_list if active_from <= stamp < active_until]
        paper_fill_min = (min(observed_times) - active_from) / 60000.0 if observed_times else float("nan")
        rows.extend(
            {
                "cohort": "paper",
                "period": "holdout" if pd.Timestamp(active_from, unit="ms", tz="UTC") > _design_cutoff() else "design",
                "symbol": symbol,
                "client_order_id": order_id,
                "side": side,
                "notional": float(limit * qty),
                "anchor_price": limit,
                "k": -1,
                "rule": variant,
                "fill": float(tape_frac[variant]),
                "fill_time_min": float(tape_first[variant]),
                "is_bps": float("nan"),
                "limit": limit,
                "paper_filled": bool(paper_filled),
                "paper_fill_time_min": paper_fill_min,
                "queue_ahead_qty": float(queue_used) if queue_used is not None else float("nan"),
                "queue_fallback": queue_used is None,
                "quantity_exceeds_level": bool(level_qty is not None and qty > level_qty),
                "level_observed": level_qty is not None,
            }
            for variant in variants
        )
    _logger.info(
        "[ALGO] stage=fill_calibration cohort=paper status=done posts=%d elapsed_s=%.1f",
        len(rows),
        time.monotonic() - t0,
    )
    if not rows and not posts:
        return rows, {
            "periods": {},
            "variants": {
                v: {"confusion": confusion[v], "fill_rate_gap": float("nan"), "median_fill_time_gap_min": float("nan")}
                for v in variants
            },
            "queue_fallback_count": fallback_count,
            "optimistic_full_fill_share": float("nan"),
        }
    periods = sorted({row["period"] for row in rows})
    per_period: dict[str, Any] = {}
    for period in periods:
        selected = [row for row in rows if row["period"] == period]
        variants_summary: dict[str, Any] = {}
        for variant in variants:
            subset = [row for row in selected if row["rule"] == variant]
            counts = {"both_filled": 0, "paper_only": 0, "tape_only": 0, "neither": 0}
            for row in subset:
                paper, tape_filled = row["paper_filled"], row["fill"] > 1e-9
                confusion_key = "both_filled" if paper and tape_filled else "paper_only" if paper else "tape_only" if tape_filled else "neither"
                counts[confusion_key] += 1
            variants_summary[variant] = {
                "confusion": counts,
                "fill_rate_gap": _weighted_mean(
                    np.array([row["fill"] - float(row["paper_filled"]) for row in subset]),
                    np.array([row["notional"] for row in subset]),
                ),
                "median_fill_time_gap_min": _percentile(np.array([
                    row["fill_time_min"] - row["paper_fill_time_min"] for row in subset
                    if row["paper_filled"] and row["fill"] > 0
                ]), 50),
            }
        unique = [row for row in selected if row["rule"] == "through_only"]
        observed = [row for row in unique if row["level_observed"]]
        per_period[period] = {
            "variants": variants_summary,
            "queue_fallback_count": sum(row["queue_fallback"] for row in unique),
            "optimistic_full_fill_share": float(np.mean([row["quantity_exceeds_level"] for row in observed])) if observed else float("nan"),
        }
    return rows, {**(per_period[periods[0]] if len(periods) == 1 else {}), "periods": per_period}


def _design_cutoff() -> pd.Timestamp:
    from src.strategy.targets import FLOW_MOM_TOP20

    return FLOW_MOM_TOP20.design_data_cutoff


def run_fill_calibration(request: FillCalibrationRequest, *, output_dir: Path) -> FillCalibrationReport:
    """Score OHLCV/paper fill rules against the tape model and persist report.json + intents.parquet under output_dir; raises DataIntegrityError on malformed inputs."""
    if not isinstance(request, FillCalibrationRequest):
        raise DataIntegrityError(f"request must be FillCalibrationRequest, got {type(request).__name__}")
    if not isinstance(output_dir, Path):
        raise DataIntegrityError(f"output_dir must be a Path, got {output_dir!r}")
    if not isinstance(request.archive_root, Path):
        raise DataIntegrityError(f"archive_root must be a Path, got {request.archive_root!r}")
    stamp = _utc_now_stamp()
    target = Path(output_dir) / stamp
    if os.path.lexists(target):
        raise DataIntegrityError(f"calibration output must be fresh: {target} already exists")
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DataIntegrityError(f"calibration output must be fresh: {target} already exists") from exc
    exclusions: dict[str, int] = {}
    try:
        parts = _load_backtest_inputs(request)
    except DataIntegrityError as exc:
        raise DataIntegrityError(f"malformed calibration inputs: {exc}") from exc
    intents = _build_backtest_intents(parts)
    backtest_rows, backtest_summary = _score_backtest(request, intents, parts, exclusions)
    paper_rows: list[dict[str, Any]] = []
    paper_summary: dict[str, Any] | None = None
    if request.paper_run_dir is not None:
        assert request.live_capture_root is not None
        paper_rows, paper_summary = _score_paper(request, exclusions)
    all_rows = [*backtest_rows, *paper_rows]
    frame = pd.DataFrame(all_rows) if all_rows else pd.DataFrame({"cohort": pd.Series(dtype="object")})
    intents_path = target / "intents.parquet"
    try:
        intents_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(frame, intents_path, compression="zstd")
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise DataIntegrityError(f"calibration intents write failed: {exc}") from exc
    payload = {"backtest": backtest_summary, "paper": paper_summary, "exclusions": dict(sorted(exclusions.items()))}
    rounded = _round_json(payload)
    report_path = target / "report.json"
    tmp = target / ".report.json.tmp"
    try:
        tmp.write_text(json.dumps(rounded, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, report_path)
    except OSError as exc:
        raise DataIntegrityError(f"calibration report write failed: {exc}") from exc
    return FillCalibrationReport(
        backtest=backtest_summary,
        paper=paper_summary,
        exclusions=dict(sorted(exclusions.items())),
        intents_path=intents_path,
        report_path=report_path,
    )
