"""Paper cohort of the fill calibration report: observed paper GTX posts versus tape replays."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from src.application.fill_calibration_common import (
    PROGRESS_INTERVAL_S,
    bump,
    day_tape,
    design_cutoff,
    percentile,
    read_tape_partition,
    slice_bounds_ms,
    tape_partition_path,
    weighted_mean,
)
from src.common.errors import DataIntegrityError

if TYPE_CHECKING:
    from src.application.fill_calibration import FillCalibrationRequest
    from src.engine.execution.tape_fill import TapeTrades

_logger = logging.getLogger(__name__)

VARIANTS: tuple[str, ...] = ("through_only", "queue_observed", "front_of_queue")
# A depth snapshot older than this before the post no longer describes the queue the order joined.
_MAX_DEPTH_STALENESS_MS = 2000.0
_FILL_QTY_RTOL = 1e-12


def parse_ts(value: Any) -> int:
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


def _audit_files(paper_run_dir: Path) -> list[Path]:
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
    return files


def _journal_sides(paper_run_dir: Path) -> dict[str, str]:
    journal_path = paper_run_dir / "order_journal.jsonl"
    sides: dict[str, str] = {}
    if not journal_path.exists():
        return sides
    try:
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("journal record must be an object")
            if record.get("event") == "submit" and record.get("side") in ("BUY", "SELL"):
                sides[record["client_order_id"]] = record["side"]
    except (OSError, ValueError, KeyError) as exc:
        raise DataIntegrityError(f"malformed paper order journal: {exc}") from exc
    return sides


def _audit_records(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataIntegrityError(f"unreadable paper audit {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise DataIntegrityError(f"malformed paper audit line: {line!r}") from exc
        if not isinstance(record, dict):
            raise DataIntegrityError(f"malformed paper audit line: {line!r}")
        records.append(record)
    return records


def _posted(record: dict[str, Any]) -> dict[str, Any]:
    try:
        price = float(record["price"])
        qty = float(record["quantity"])
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed order_posted record: {exc}") from exc
    if not (np.isfinite(price) and price > 0 and np.isfinite(qty) and qty > 0):
        raise DataIntegrityError(f"malformed order_posted record: {record!r}")
    side_raw = str(record.get("side", "")).upper()
    return {
        "symbol": str(record.get("symbol")),
        "price": price,
        "quantity": qty,
        "ts_ms": parse_ts(record.get("ts")),
        "side": side_raw if side_raw in ("BUY", "SELL") else None,
    }


def _fill_event(record: dict[str, Any]) -> tuple[int, float]:
    try:
        stamp = parse_ts(record.get("ts"))
        quantity = float(record["qty"])
        if not np.isfinite(quantity) or quantity <= 0:
            raise ValueError("fill quantity must be finite and positive")
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed fill record: {exc}") from exc
    return stamp, quantity


def load_paper_posts(paper_run_dir: Path) -> dict[str, Any]:
    """Read GTX posts, cancels and fills from a paper run's audit log; raises DataIntegrityError on any malformed record."""
    files = _audit_files(paper_run_dir)
    sides = _journal_sides(paper_run_dir)
    posts: dict[str, dict[str, Any]] = {}
    cancels: dict[str, int] = {}
    fills: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for path in files:
        for record in _audit_records(path):
            event = record.get("event")
            cid = record.get("client_order_id")
            if not isinstance(cid, str) or not cid:
                continue
            if event == "order_posted" and record.get("time_in_force") == "GTX":
                posts[cid] = _posted(record)
            elif event == "order_cancelled":
                try:
                    cancel = parse_ts(record.get("ts"))
                except DataIntegrityError as exc:
                    raise DataIntegrityError(f"malformed order_cancelled record: {exc}") from exc
                cancels[cid] = min(cancels.get(cid, cancel), cancel)
            elif event == "fill":
                fills[cid].append(_fill_event(record))
    for cid, post in posts.items():
        if post["side"] is None:
            post["side"] = sides.get(cid)
        elif cid in sides and sides[cid] != post["side"]:
            raise DataIntegrityError(f"paper side disagrees with journal for {cid}")
    return {"posts": posts, "cancels": cancels, "fills": fills}


def _depth_times_ms(frame: pd.DataFrame, path: Path) -> np.ndarray:
    for candidate in ("received_at", "transact_time_ms", "event_time_ms"):
        if candidate not in frame.columns:
            continue
        if candidate == "received_at":
            times = pd.to_datetime(frame["received_at"], utc=True).dt.as_unit("ns").astype("int64") // 10**6
        else:
            times = pd.to_numeric(frame[candidate], errors="coerce")
        return np.asarray(times, dtype="float64")
    raise DataIntegrityError(f"exec_depth {path} missing time column")


def load_depth_index(live_capture_root: Path, days: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """Per-symbol exec_depth snapshots ordered by receipt time, restricted to day directories in ``days``."""
    base = live_capture_root / "exec_depth"
    if not base.exists():
        return {}
    by_symbol: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for path in sorted(base.rglob("*.parquet")):
        if days is not None and path.parent.name not in days:
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataIntegrityError(f"unreadable exec_depth {path}: {exc}") from exc
        if "symbol" not in frame.columns:
            raise DataIntegrityError(f"exec_depth {path} missing symbol column")
        work = frame.copy()
        work["_row_ms"] = _depth_times_ms(frame, path)
        for symbol, part in work.groupby("symbol"):
            by_symbol[str(symbol)].append(part)
    return {
        symbol: pd.concat(parts, ignore_index=True).sort_values("_row_ms", kind="mergesort").reset_index(drop=True)
        for symbol, parts in by_symbol.items()
    }


def _depth_levels(frame: pd.DataFrame) -> int:
    n = 0
    while n < 5 and f"bid_px_{n}" in frame.columns and f"ask_px_{n}" in frame.columns:
        n += 1
    return n


def queue_ahead(side: str, limit: float, row: pd.Series, n_levels: int) -> tuple[float | None, float | None]:
    """Return (queue ahead qty, resting level qty) for a passive limit on its own book side.

    A limit inside the spread or between two visible levels joins an empty level: (0.0, None).
    A limit outside the visible book is unknown: (None, None).
    """
    prefix = "bid" if side == "BUY" else "ask"
    # Orient prices so "more aggressive" is always larger on both sides of the book.
    sign = 1.0 if side == "BUY" else -1.0
    try:
        quantities = [float(row[f"{prefix}_qty_{i}"]) for i in range(n_levels)]
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed exec_depth quantities: {exc}") from exc
    if any(q < 0 for q in quantities):
        raise DataIntegrityError("exec_depth quantities must be non-negative")
    levels = [(float(row[f"{prefix}_px_{i}"]), quantities[i]) for i in range(n_levels)]
    book = [(p, q) for p, q in levels if np.isfinite(p) and np.isfinite(q) and p > 0]
    if not book:
        return None, None
    if sign * limit > sign * book[0][0]:
        return 0.0, None
    for px, qty in book:
        if limit == px:
            return float(qty), float(qty)
    for (near, _), (far, _) in pairwise(book):
        if sign * near > sign * limit > sign * far:
            return 0.0, None
    return None, None


def _observed_queue(
    depth_frame: pd.DataFrame | None, side: str, limit: float, active_from: int
) -> tuple[float | None, float | None]:
    """Queue estimate from the last depth snapshot at or before the post, if fresh enough."""
    if depth_frame is None or not len(depth_frame):
        return None, None
    times = depth_frame["_row_ms"].to_numpy(dtype="float64")
    eligible = np.flatnonzero(np.isfinite(times) & (times <= active_from))
    if not eligible.size:
        return None, None
    pos_idx = int(eligible[-1])
    if active_from - float(times[pos_idx]) > _MAX_DEPTH_STALENESS_MS:
        return None, None
    n_levels = _depth_levels(depth_frame)
    if not n_levels:
        return None, None
    return queue_ahead(side, limit, depth_frame.iloc[pos_idx], n_levels)


def _post_lifetime(
    post: dict[str, Any], cancel_ms: int | None, fill_list: list[tuple[int, float]]
) -> int | None:
    """Exclusive end of the post's resting life (first cancel or completing fill), or None when unresolved."""
    active_from = int(post["ts_ms"])
    qty = float(post["quantity"])
    completing = None
    cumulative = 0.0
    for stamp, increment in fill_list:
        if stamp >= active_from and (cancel_ms is None or stamp <= cancel_ms):
            cumulative += increment
            if cumulative >= qty * (1.0 - _FILL_QTY_RTOL):
                completing = stamp
                break
    ends = [int(v) for v in (cancel_ms, completing) if v is not None]
    if not ends:
        return None
    active_until = min(ends) + 1
    return active_until if active_until > active_from else None


def _simulate_variants(
    tape: TapeTrades,
    side: str,
    limit: float,
    qty: float,
    window: tuple[int, int],
    queue: float | None,
    request: FillCalibrationRequest,
) -> dict[str, tuple[float, float]]:
    from src.engine.execution.tape_fill import RestingOrder, TapeFillSpec, simulate_resting_fill

    specs: dict[str, tuple[float | None, Literal["through_only", "front_of_queue"]]] = {
        "through_only": (None, "through_only"),
        "queue_observed": (queue, "through_only"),
        "front_of_queue": (None, "front_of_queue"),
    }
    out: dict[str, tuple[float, float]] = {}
    for variant, (q_val, unknown) in specs.items():
        order = RestingOrder(
            side="BUY" if side == "BUY" else "SELL",
            limit_price=limit,
            quantity=qty,
            active_from_ms=window[0],
            active_until_ms=window[1],
            queue_ahead_qty=float(q_val) if q_val is not None else None,
        )
        spec = TapeFillSpec(queue_ahead_fraction=float(request.queue_ahead_fraction), unknown_queue=unknown)
        result = simulate_resting_fill(order, tape, spec)
        frac = min(max(float(result.filled_qty / qty), 0.0), 1.0)
        first = (
            float((int(result.first_fill_ms) - window[0]) / 60000.0) if result.first_fill_ms is not None else float("nan")
        )
        out[variant] = (frac, first)
    return out


class _PartitionCache:
    """Holds one (symbol, month) tape partition; posts are visited in symbol/time order so each loads once."""

    def __init__(self) -> None:
        self._key: tuple[str, str] | None = None
        self._frame: pd.DataFrame | None = None

    def get(self, key: tuple[str, str], path: Path) -> pd.DataFrame:
        if key != self._key or self._frame is None:
            self._frame = None
            self._frame = read_tape_partition(path)
            self._key = key
        return self._frame


def _month_of(ts_ms: int) -> str:
    return str(pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y%m"))


def _score_post(
    request: FillCalibrationRequest,
    order_id: str,
    post: dict[str, Any],
    bundle: dict[str, Any],
    coverage: dict[Any, Any],
    depth_index: dict[str, pd.DataFrame],
    cache: _PartitionCache,
    exclusions: dict[str, int],
) -> list[dict[str, Any]] | None:
    symbol = post["symbol"]
    limit = float(post["price"])
    qty = float(post["quantity"])
    active_from = int(post["ts_ms"])
    fill_list = sorted(bundle["fills"].get(order_id, []))
    active_until = _post_lifetime(post, bundle["cancels"].get(order_id), fill_list)
    if active_until is None:
        bump(exclusions, "paper_post_unresolved")
        return None
    day = pd.Timestamp(active_from, unit="ms", tz="UTC").date()
    row_cov = coverage.get((symbol, day))
    s_ms, e_ms = slice_bounds_ms(day)
    if row_cov is None or row_cov.status != "ok" or not (active_from >= s_ms and active_until <= e_ms):
        bump(exclusions, "paper_no_tape_coverage")
        return None
    month = _month_of(active_from)
    path = tape_partition_path(request.archive_root, symbol, month)
    if not path.exists():
        raise DataIntegrityError(f"tape partition missing for covered {(symbol, day)}: {path}")
    tape = day_tape(cache.get((symbol, month), path), symbol, day)
    side = post.get("side")
    if side not in ("BUY", "SELL"):
        raise DataIntegrityError(f"paper order {order_id} has no side; require audit side or order_journal.jsonl submit")
    queue_used, level_qty = _observed_queue(depth_index.get(symbol), side, limit, active_from)
    outcomes = _simulate_variants(tape, side, limit, qty, (active_from, active_until), queue_used, request)
    observed_times = [stamp for stamp, _ in fill_list if active_from <= stamp < active_until]
    paper_fill_min = (min(observed_times) - active_from) / 60000.0 if observed_times else float("nan")
    period = "holdout" if pd.Timestamp(active_from, unit="ms", tz="UTC") > design_cutoff() else "design"
    return [
        {
            "cohort": "paper",
            "period": period,
            "symbol": symbol,
            "client_order_id": order_id,
            "side": side,
            "notional": float(limit * qty),
            "anchor_price": limit,
            "k": -1,
            "rule": variant,
            "fill": float(outcomes[variant][0]),
            "fill_time_min": float(outcomes[variant][1]),
            "is_bps": float("nan"),
            "limit": limit,
            "paper_filled": bool(observed_times),
            "paper_fill_time_min": paper_fill_min,
            "queue_ahead_qty": float(queue_used) if queue_used is not None else float("nan"),
            "queue_fallback": queue_used is None,
            "quantity_exceeds_level": bool(level_qty is not None and qty > level_qty),
            "level_observed": level_qty is not None,
        }
        for variant in VARIANTS
    ]


def _confusion(subset: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"both_filled": 0, "paper_only": 0, "tape_only": 0, "neither": 0}
    for row in subset:
        paper, tape_filled = row["paper_filled"], row["fill"] > 1e-9
        key = "both_filled" if paper and tape_filled else "paper_only" if paper else "tape_only" if tape_filled else "neither"
        counts[key] += 1
    return counts


def summarize_paper(rows: list[dict[str, Any]], *, n_posts: int) -> dict[str, Any]:
    """Per-period confusion matrices and notional-weighted fill-rate gaps per tape variant; periods never pool."""
    if not rows and not n_posts:
        return {
            "periods": {},
            "variants": {
                v: {
                    "confusion": _confusion([]),
                    "fill_rate_gap": float("nan"),
                    "median_fill_time_gap_min": float("nan"),
                }
                for v in VARIANTS
            },
            "queue_fallback_count": 0,
            "optimistic_full_fill_share": float("nan"),
        }
    periods = sorted({row["period"] for row in rows})
    per_period: dict[str, Any] = {}
    for period in periods:
        selected = [row for row in rows if row["period"] == period]
        variants_summary: dict[str, Any] = {}
        for variant in VARIANTS:
            subset = [row for row in selected if row["rule"] == variant]
            variants_summary[variant] = {
                "confusion": _confusion(subset),
                "fill_rate_gap": weighted_mean(
                    np.array([row["fill"] - float(row["paper_filled"]) for row in subset]),
                    np.array([row["notional"] for row in subset]),
                ),
                "median_fill_time_gap_min": percentile(np.array([
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
    return {**(per_period[periods[0]] if len(periods) == 1 else {}), "periods": per_period}


def score_paper(
    request: FillCalibrationRequest,
    exclusions: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay every paper GTX post inside the evaluation window on the tape; requires paper_run_dir and live_capture_root."""
    from src.market_data.public_archive import store

    assert request.paper_run_dir is not None
    assert request.live_capture_root is not None
    bundle = load_paper_posts(request.paper_run_dir)
    start_ms = int(request.evaluation_start.value // 10**6)
    end_ms = int(request.evaluation_end.value // 10**6)
    posts = {cid: post for cid, post in bundle["posts"].items() if start_ms <= post["ts_ms"] < end_ms}
    depth_days = {
        (pd.Timestamp(post["ts_ms"], unit="ms", tz="UTC") - pd.Timedelta(days=offset)).strftime("%Y%m%d")
        for post in posts.values() for offset in (0, 1)
    }
    depth_index = load_depth_index(request.live_capture_root, depth_days)
    coverage = store.load_coverage(request.archive_root, "aggTrades")
    month_keys = sorted({(post["symbol"], _month_of(post["ts_ms"])) for post in posts.values()})
    group_positions = {key: index for index, key in enumerate(month_keys)}
    cache = _PartitionCache()
    rows: list[dict[str, Any]] = []
    t0 = last_log = time.monotonic()
    for order_id, post in sorted(posts.items(), key=lambda item: (item[1]["symbol"], item[1]["ts_ms"], item[0])):
        now = time.monotonic()
        if now - last_log >= PROGRESS_INTERVAL_S:
            done = group_positions[(post["symbol"], _month_of(post["ts_ms"]))]
            _logger.info(
                "[ALGO] stage=fill_calibration cohort=paper done=%d total=%d elapsed_s=%.1f",
                done, len(month_keys), now - t0,
            )
            last_log = now
        scored = _score_post(request, order_id, post, bundle, coverage, depth_index, cache, exclusions)
        if scored is not None:
            rows.extend(scored)
    _logger.info(
        "[ALGO] stage=fill_calibration cohort=paper status=done posts=%d elapsed_s=%.1f",
        len(rows),
        time.monotonic() - t0,
    )
    return rows, summarize_paper(rows, n_posts=len(posts))
