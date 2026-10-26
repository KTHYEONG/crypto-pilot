"""Backtest cohort of the fill calibration report: OHLCV strict rule versus tape replays."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from src.application.fill_calibration_common import (
    PROGRESS_INTERVAL_S,
    bootstrap_diff_ci,
    bump,
    day_tape,
    percentile,
    read_tape_partition,
    side_sign,
    slice_bounds_ms,
    tape_partition_path,
    weighted_mean,
)
from src.common.errors import DataIntegrityError

if TYPE_CHECKING:
    from src.application.fill_calibration import FillCalibrationRequest
    from src.engine.execution.tape_fill import TapeTrades

_logger = logging.getLogger(__name__)

_UnknownQueue = Literal["through_only", "front_of_queue"]


def kind_of(q_prev: float, q_new: float) -> str:
    if q_prev == 0.0 and q_new != 0.0:
        return "open"
    if q_new == 0.0 and q_prev != 0.0:
        return "close"
    if q_prev * q_new < 0.0:
        return "flip"
    if abs(q_new) > abs(q_prev):
        return "increase"
    return "reduce"


def _validate_mark_panels(marks: Any, unit_weights: pd.DataFrame, anchor_times: pd.DatetimeIndex) -> None:
    closes = marks.close
    bar_index = closes.index
    if not all(panel.index.equals(bar_index) and panel.columns.equals(closes.columns) for panel in (marks.high, marks.low)):
        raise DataIntegrityError("mark panels must have aligned indexes and columns")
    if not bar_index.is_unique or not bar_index.is_monotonic_increasing:
        raise DataIntegrityError("mark bars must be unique and ordered")
    dates = unit_weights.index
    if len(anchor_times) != len(dates) or not dates.is_unique or not dates.is_monotonic_increasing:
        raise DataIntegrityError("entry dates and anchors must be aligned and ordered")
    if not np.isfinite(unit_weights.to_numpy(dtype="float64")).all():
        raise DataIntegrityError("unit weights must be finite")


def build_backtest_intents(parts: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Replay unit-weight targets into per-symbol rebalance intents at each anchor close; raises DataIntegrityError on misaligned panels."""
    from src.strategy.targets import FLOW_MOM_TOP20

    unit_weights, marks, _funding, _adv, _sigma, anchor_times = parts
    cutoff = FLOW_MOM_TOP20.design_data_cutoff
    closes = marks.close
    _validate_mark_panels(marks, unit_weights, anchor_times)
    anchor_pos = closes.index.get_indexer(anchor_times)
    intents: list[dict[str, Any]] = []
    q_prev: dict[str, float] = dict.fromkeys(unit_weights.columns, 0.0)
    for i, entry in enumerate(unit_weights.index):
        entry_utc = pd.Timestamp(entry).tz_convert("UTC")
        pos = int(anchor_pos[i])
        if pos < 0:
            raise DataIntegrityError(f"anchor bar absent from mark panels: {anchor_times[i]}")
        for symbol in unit_weights.columns:
            w = float(unit_weights.iloc[i][symbol])
            try:
                a = float(closes.iloc[pos][symbol])
            except (KeyError, IndexError, ValueError) as exc:
                raise DataIntegrityError(f"malformed mark panels: {exc}") from exc
            if not np.isfinite(a) or not a > 0:
                continue
            qp = float(q_prev[symbol])
            delta = w - qp * a
            if not np.isfinite(delta) or not abs(delta) > 0:
                continue
            q_new = w / a
            intents.append(
                {
                    "symbol": str(symbol),
                    "entry": entry_utc,
                    "anchor": pd.Timestamp(anchor_times[i]).tz_convert("UTC"),
                    "anchor_pos": pos,
                    "anchor_price": a,
                    "delta": delta,
                    "notional": abs(delta),
                    "side": "BUY" if delta > 0 else "SELL",
                    "kind": kind_of(qp, q_new),
                    "period": "holdout" if entry_utc > cutoff else "design",
                    "year": int(entry_utc.year),
                    "day": entry_utc.normalize(),
                }
            )
            q_prev[symbol] = q_new
    return intents


def _intent_window(intent: dict[str, Any], bar_ms: int, max_k: int) -> tuple[date, int, int]:
    anchor = int(pd.Timestamp(intent["anchor"]).value // 10**6)
    active_from = anchor + bar_ms
    active_until = anchor + (max_k + 1) * bar_ms
    return pd.Timestamp(active_from, unit="ms", tz="UTC").date(), active_from, active_until


def _blended_is_bps(frac: float, side: str, limit: float, anchor: float, close_k: float) -> float:
    """Maker leg at the limit for the filled fraction, taker cross at the window close for the rest."""
    from src.core.params import ACCOUNT_MAKER_FEE_BPS, ACCOUNT_TAKER_FEE_BPS

    sign = side_sign(side)
    maker = sign * (limit - anchor) / anchor * 1e4 + ACCOUNT_MAKER_FEE_BPS
    taker = sign * (close_k - anchor) / anchor * 1e4 + ACCOUNT_TAKER_FEE_BPS
    return frac * maker + (1.0 - frac) * taker


def _ohlcv_outcome(marks: Any, symbol: str, pos: int, k: int, side: str, anchor: float) -> tuple[float, float, float]:
    """Return (fill, fill_time_min, close_k) of the strict OHLCV rule over the k bars after the anchor."""
    from src.live.executor import EXECUTION_BAR_SECONDS

    window = slice(pos + 1, pos + 1 + k)
    closes = marks.close.iloc[window][symbol].to_numpy(dtype="float64")
    if side == "BUY":
        lows = marks.low.iloc[window][symbol].to_numpy(dtype="float64")
        hit = np.flatnonzero(np.isfinite(lows) & (lows < anchor))
    else:
        highs = marks.high.iloc[window][symbol].to_numpy(dtype="float64")
        hit = np.flatnonzero(np.isfinite(highs) & (highs > anchor))
    fill_time = float((int(hit[0]) + 1) * float(EXECUTION_BAR_SECONDS) / 60.0) if hit.size else float("nan")
    finite_closes = closes[np.isfinite(closes)]
    close_k = float(finite_closes[-1]) if finite_closes.size else anchor
    return (1.0 if hit.size else 0.0), fill_time, close_k


def _touch_context(
    tape: TapeTrades, intent: dict[str, Any], active_from: int, lookback_ms: int, exclusions: dict[str, int]
) -> tuple[bool, float | None]:
    """Return (marketable_anchor, post-only feasible limit) from the tape touch proxy, or (False, None) without one."""
    from src.engine.execution.tape_fill import post_only_limit, touch_proxy

    try:
        proxy = touch_proxy(tape, active_from, lookback_ms)
    except DataIntegrityError:
        proxy = None
    if proxy is None:
        bump(exclusions, "no_touch_proxy")
        return False, None
    a = float(intent["anchor_price"])
    side: Literal["BUY", "SELL"] = "BUY" if intent["side"] == "BUY" else "SELL"
    bid_px, ask_px = proxy
    marketable = (a >= ask_px) if side == "BUY" else (a <= bid_px)
    return marketable, post_only_limit(side, a, proxy)


def _tape_outcome(
    tape: TapeTrades, side: str, limit: float, qty: float, active_from: int, active_until: int, unknown: _UnknownQueue
) -> tuple[float, float]:
    """Return (filled fraction in [0, 1], first fill minutes after active_from or NaN)."""
    from src.engine.execution.tape_fill import RestingOrder, TapeFillSpec, simulate_resting_fill

    order = RestingOrder(
        side="BUY" if side == "BUY" else "SELL",
        limit_price=float(limit),
        quantity=qty,
        active_from_ms=active_from,
        active_until_ms=active_until,
        queue_ahead_qty=None,
    )
    result = simulate_resting_fill(order, tape, TapeFillSpec(unknown_queue=unknown))
    frac = min(max(float(result.filled_qty / qty), 0.0), 1.0)
    first = (
        float((int(result.first_fill_ms) - active_from) / 60000.0) if result.first_fill_ms is not None else float("nan")
    )
    return frac, first


def _score_intent(
    request: FillCalibrationRequest,
    intent: dict[str, Any],
    tape: TapeTrades,
    marks: Any,
    active_from: int,
    bar_ms: int,
    exclusions: dict[str, int],
) -> list[dict[str, Any]]:
    symbol = intent["symbol"]
    a = float(intent["anchor_price"])
    side = intent["side"]
    marketable, feasible_limit = _touch_context(
        tape, intent, active_from, int(float(request.touch_lookback_s) * 1000), exclusions
    )
    pos = int(intent["anchor_pos"])
    max_k = max(int(v) for v in request.k_bars)
    expected = pd.date_range(intent["anchor"], periods=max_k + 1, freq=pd.Timedelta(milliseconds=bar_ms))
    if not marks.close.index[pos : pos + max_k + 1].equals(expected):
        raise DataIntegrityError("mark panels do not contain the complete calibration window")
    base = {
        "cohort": "backtest",
        **{key: intent[key] for key in ("symbol", "entry", "anchor", "period", "year", "day", "side", "kind")},
        "notional": float(intent["notional"]),
        "anchor_price": a,
        "marketable_anchor": bool(marketable),
    }
    tape_rules: list[tuple[str, float, _UnknownQueue]] = [("tape_through_anchor", a, "through_only")]
    if feasible_limit is not None:
        tape_rules.append(("tape_through_feasible", feasible_limit, "through_only"))
        tape_rules.append(("tape_front_feasible", feasible_limit, "front_of_queue"))
    qty = float(intent["notional"]) / a
    rows: list[dict[str, Any]] = []
    for k in (int(v) for v in request.k_bars):
        fill, fill_time, close_k = _ohlcv_outcome(marks, symbol, pos, k, side, a)
        rows.append({
            **base, "k": k, "rule": "ohlcv_strict", "fill": fill, "fill_time_min": fill_time,
            "is_bps": float(_blended_is_bps(fill, side, a, a, close_k)), "limit": a, "close_k": close_k,
        })
        for rule, limit, unknown in tape_rules:
            frac, first = _tape_outcome(tape, side, limit, qty, active_from, active_from + k * bar_ms, unknown)
            rows.append({
                **base, "k": k, "rule": rule, "fill": frac, "fill_time_min": first,
                "is_bps": float(_blended_is_bps(frac, side, float(limit), a, close_k)), "limit": float(limit),
                "close_k": close_k,
            })
    return rows


def score_backtest(
    request: FillCalibrationRequest,
    intents: list[dict[str, Any]],
    parts: tuple[Any, ...],
    exclusions: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every covered intent under the OHLCV and tape rules; reads each (symbol, month) partition at most once."""
    from src.live.executor import EXECUTION_BAR_SECONDS
    from src.market_data.public_archive import store

    marks = parts[1]
    bar_ms = int(float(EXECUTION_BAR_SECONDS) * 1000)
    max_k = max(int(v) for v in request.k_bars)
    coverage = store.load_coverage(request.archive_root, "aggTrades")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, intent in enumerate(intents):
        _day, active_from, _until = _intent_window(intent, bar_ms, max_k)
        grouped[(intent["symbol"], pd.Timestamp(active_from, unit="ms", tz="UTC").strftime("%Y%m"))].append(idx)

    scored_rows: list[dict[str, Any]] = []
    t0 = last_log = time.monotonic()
    for done, (symbol, month) in enumerate(sorted(grouped), start=1):
        path = tape_partition_path(request.archive_root, symbol, month)
        partition: pd.DataFrame | None = None
        for idx in grouped[(symbol, month)]:
            intent = intents[idx]
            day, active_from, active_until = _intent_window(intent, bar_ms, max_k)
            row = coverage.get((symbol, day))
            if row is None or row.status != "ok":
                bump(exclusions, "no_tape_coverage")
                continue
            slice_start_ms, slice_end_ms = slice_bounds_ms(day)
            if not (active_from >= slice_start_ms and active_until <= slice_end_ms):
                bump(exclusions, "window_outside_slice")
                continue
            if partition is None:
                if not path.exists():
                    raise DataIntegrityError(f"tape partition missing for covered {(symbol, day)}: {path}")
                partition = read_tape_partition(path)
            tape = day_tape(partition, symbol, day)
            scored_rows.extend(_score_intent(request, intent, tape, marks, active_from, bar_ms, exclusions))
        now = time.monotonic()
        if now - last_log >= PROGRESS_INTERVAL_S or done == len(grouped):
            _logger.info(
                "[ALGO] stage=fill_calibration cohort=backtest done=%d total=%d elapsed_s=%.1f",
                done,
                len(grouped),
                now - t0,
            )
            last_log = now
    summary = summarize_backtest(scored_rows, request)
    _logger.info(
        "[ALGO] stage=fill_calibration cohort=backtest status=done intents=%d exclusions=%s",
        len(scored_rows),
        dict(exclusions),
    )
    return scored_rows, summary


def _rule_entry(g: pd.DataFrame, gg: pd.DataFrame, rule: str, k: int, request: FillCalibrationRequest) -> dict[str, Any]:
    fills = gg["fill"].to_numpy(dtype="float64")
    iss = gg["is_bps"].to_numpy(dtype="float64")
    w = gg["notional"].to_numpy(dtype="float64")
    times = gg["fill_time_min"].to_numpy(dtype="float64")
    filled_times = times[np.isfinite(times) & (fills > 0)]
    entry: dict[str, Any] = {
        "n_intents": len(gg),
        "fill_fraction": weighted_mean(fills, w),
        "is_bps": weighted_mean(iss, w),
        "fill_time_p50_min": percentile(filled_times, 50.0),
        "fill_time_p90_min": percentile(filled_times, 90.0),
    }
    base = g[(g["rule"] == "ohlcv_strict") & (g["k"] == k)]
    if rule == "ohlcv_strict" or not len(base):
        return entry
    samples, seed = int(request.bootstrap_samples), int(request.seed)
    paired = gg.merge(base, on=["symbol", "entry", "k"], suffixes=("", "_base"), validate="one_to_one")
    bday = paired["day"].to_numpy()
    gday = gg["day"].to_numpy()
    pf, lo_f, hi_f = bootstrap_diff_ci(
        bday, gday, fills, paired["fill_base"].to_numpy(dtype="float64"), w, samples=samples, seed=seed
    )
    pi, lo_i, hi_i = bootstrap_diff_ci(
        bday, gday, iss, paired["is_bps_base"].to_numpy(dtype="float64"), w, samples=samples, seed=seed
    )
    both = paired[(paired["fill"] > 0) & (paired["fill_base"] > 0)]
    time_diff, time_lo, time_hi = bootstrap_diff_ci(
        both["day"].to_numpy(), both["day"].to_numpy(),
        both["fill_time_min"].to_numpy(dtype="float64"),
        both["fill_time_min_base"].to_numpy(dtype="float64"),
        both["notional"].to_numpy(dtype="float64"),
        samples=samples, seed=seed,
    )
    entry["diff_fill_vs_ohlcv"] = pf
    entry["diff_fill_ci95"] = [lo_f, hi_f]
    entry["diff_is_vs_ohlcv"] = pi
    entry["diff_is_ci95"] = [lo_i, hi_i]
    entry["diff_fill_time_min_vs_ohlcv"] = time_diff
    entry["diff_fill_time_ci95"] = [time_lo, time_hi]
    return entry


def summarize_backtest(rows: list[dict[str, Any]], request: FillCalibrationRequest) -> dict[str, Any]:
    """Per-period rule@K summaries with day-block bootstrap differences against ohlcv_strict; periods never pool."""
    if not rows:
        return {"periods": {}}
    frame = pd.DataFrame(rows)
    periods: dict[str, Any] = {}
    for period, g in frame.groupby("period"):
        strict = g[g["rule"] == "ohlcv_strict"]
        market_w = strict["notional"].to_numpy(dtype="float64")
        market_flags = strict["marketable_anchor"].to_numpy(dtype=bool)
        market_share = (
            float(np.sum(market_w[market_flags]) / np.sum(market_w)) if np.sum(market_w) > 0 else float("nan")
        )
        per_rule = {
            f"{rule}@K{int(k)}": _rule_entry(g, gg, str(rule), int(k), request)
            for (rule, k), gg in g.groupby(["rule", "k"])
        }
        per_year = {
            f"{rule}@K{int(k)}@{int(year)}": {
                "n_intents": len(gg),
                "fill_fraction": weighted_mean(gg["fill"].to_numpy(dtype="float64"), gg["notional"].to_numpy(dtype="float64")),
                "is_bps": weighted_mean(gg["is_bps"].to_numpy(dtype="float64"), gg["notional"].to_numpy(dtype="float64")),
            }
            for (rule, k, year), gg in g.groupby(["rule", "k", "year"])
        }
        periods[str(period)] = {
            "rules": per_rule,
            "marketable_anchor_share": market_share,
            "per_year": per_year,
        }
    return {"periods": periods}
