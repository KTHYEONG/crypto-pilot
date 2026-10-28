"""Daily paper versus tape versus backtest-rule fill parity monitor."""

from __future__ import annotations

import fcntl
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError


@dataclass(frozen=True, slots=True)
class FillParityThresholds:
    """Breach limits for the trailing parity window."""

    trailing_days: int = 7
    min_agreement: float = 0.95
    max_abs_gap_bps: float = 1.5
    min_intents: int = 30


@dataclass(frozen=True, slots=True)
class FillParityRow:
    """One day's parity measurement."""

    day: date
    run_id: str
    n_intents: int
    n_posts: int
    through_agreement: float | None
    paper_is_bps: float | None
    rule_is_bps: float | None
    gap_bps: float | None
    status: Literal["OK", "SKIPPED"]
    detail: str | None


_EQ_REQUIRED = ("decision_time", "symbol", "leg_index", "mark_price_at_decision", "avg_fill_price", "filled_qty")
_FILLS_REQUIRED = ("decision_time", "symbol", "leg_index", "quantity_delta", "fill_price", "fee_bps")


def _load_eq_frames(paper_run_dir: Path) -> pd.DataFrame:
    eq_dir = paper_run_dir / "execution_quality"
    if not eq_dir.is_dir():
        return pd.DataFrame()
    shards = sorted(eq_dir.glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            frame = pd.read_parquet(shard)
        except Exception as exc:
            raise DataIntegrityError(f"unreadable execution_quality {shard}: {exc}") from exc
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_fills_frames(paper_run_dir: Path) -> pd.DataFrame:
    fills_dir = paper_run_dir / "fills"
    if not fills_dir.is_dir():
        return pd.DataFrame()
    shards = sorted(fills_dir.glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    for shard in shards:
        try:
            frame = pd.read_parquet(shard)
        except Exception as exc:
            raise DataIntegrityError(f"unreadable fills {shard}: {exc}") from exc
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _day_key(value: object) -> date:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed decision_time {value!r}: {exc}") from exc
    if pd.isna(stamp):
        raise DataIntegrityError(f"malformed decision_time {value!r}")
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    result: date = stamp.date()
    return result


def _as_ms(value: object) -> int:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed timestamp {value!r}: {exc}") from exc
    if pd.isna(stamp):
        raise DataIntegrityError(f"malformed timestamp {value!r}")
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return int(stamp.value // 10**6)


def _require_frame_columns(eq: pd.DataFrame, fills: pd.DataFrame) -> None:
    missing = [c for c in _EQ_REQUIRED if c not in eq.columns]
    if missing:
        raise DataIntegrityError(f"execution_quality missing columns: {missing}")
    if len(fills):
        missing_f = [c for c in _FILLS_REQUIRED if c not in fills.columns]
        if missing_f:
            raise DataIntegrityError(f"fills missing columns: {missing_f}")


def _group_fills(fills: pd.DataFrame) -> dict[tuple[int, str, int], pd.DataFrame]:
    if not len(fills):
        return {}
    try:
        key_ms = [_as_ms(v) for v in fills["decision_time"].tolist()]
        leg = [int(v) for v in fills["leg_index"].tolist()]
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed fills key: {exc}") from exc
    work = fills.copy()
    work["_key_ms"] = key_ms
    work["_leg"] = leg
    work["_sym"] = [str(v) for v in fills["symbol"].tolist()]
    return {
        (int(key[0]), str(key[1]), int(key[2])): part
        for key, part in work.groupby(["_key_ms", "_sym", "_leg"])
    }


def _eq_row_values(row: pd.Series) -> tuple[float, float, float, int] | None:
    """(ref, avg_fill, filled_qty, leg_index) of an execution-quality row, or None when nothing filled."""
    try:
        ref = float(row["mark_price_at_decision"])
        avg = float(row["avg_fill_price"])
        filled = float(row["filled_qty"])
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed execution_quality row: {exc}") from exc
    if not np.isfinite(filled) or filled < 0:
        raise DataIntegrityError(f"malformed execution_quality filled_qty: {row.to_dict()!r}")
    if filled == 0:
        return None
    if not (np.isfinite(ref) and np.isfinite(avg)):
        raise DataIntegrityError(f"malformed execution_quality prices: {row.to_dict()!r}")
    if not (ref > 0 and avg > 0):
        raise DataIntegrityError(f"malformed execution_quality non-positive price: {row.to_dict()!r}")
    try:
        leg_index = int(row["leg_index"])
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed execution_quality leg_index: {exc}") from exc
    return ref, avg, filled, leg_index


def _fills_side_and_fee(part: pd.DataFrame) -> tuple[str, float] | None:
    """(side, notional-weighted fee bps) of one intent's fills, or None when they net to zero."""
    try:
        deltas = pd.to_numeric(part["quantity_delta"], errors="coerce").to_numpy(dtype="float64")
        prices = pd.to_numeric(part["fill_price"], errors="coerce").to_numpy(dtype="float64")
        fees = pd.to_numeric(part["fee_bps"], errors="coerce").to_numpy(dtype="float64")
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed fills row: {exc}") from exc
    if not (np.isfinite(deltas).all() and np.isfinite(prices).all() and np.isfinite(fees).all()):
        raise DataIntegrityError("fills hold non-finite quantity, price or fee")
    if bool((prices <= 0).any()):
        raise DataIntegrityError("fills hold non-positive price")
    net = float(np.sum(deltas))
    if net == 0.0:
        return None
    weights = np.abs(deltas) * prices
    denom = float(np.sum(weights))
    if not np.isfinite(denom) or denom <= 0:
        raise DataIntegrityError("fills hold non-positive notional")
    fee_w = float(np.sum(weights * fees) / denom)
    if not np.isfinite(fee_w):
        raise DataIntegrityError("fills fee mean is non-finite")
    return ("BUY" if net > 0 else "SELL"), fee_w


def _collect_intents(eq: pd.DataFrame, fills: pd.DataFrame) -> list[dict[str, object]]:
    _require_frame_columns(eq, fills)
    grouped = _group_fills(fills)
    intents: list[dict[str, object]] = []
    for _, row in eq.iterrows():
        values = _eq_row_values(row)
        if values is None:
            continue
        ref, avg, filled, leg_index = values
        symbol = str(row["symbol"])
        part = grouped.get((_as_ms(row["decision_time"]), symbol, leg_index))
        if part is None or not len(part):
            raise DataIntegrityError(f"fills missing for {(row['decision_time'], symbol, leg_index)}")
        side_fee = _fills_side_and_fee(part)
        if side_fee is None:
            continue
        side, fee_w = side_fee
        sign = 1.0 if side == "BUY" else -1.0
        paper_is = sign * (avg - ref) / ref * 1e4 + fee_w
        if not np.isfinite(paper_is):
            raise DataIntegrityError("paper shortfall is non-finite")
        notional = float(filled * avg)
        if not np.isfinite(notional) or notional <= 0:
            raise DataIntegrityError("intent notional is non-positive")
        intents.append({
            "symbol": symbol, "side": side, "ref": float(ref),
            "notional": notional, "paper_is": float(paper_is),
        })
    return intents


def _release_window_ms(day: date) -> tuple[int, int]:
    from src.strategy.targets import FLOW_MOM_TOP20

    release = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=int(FLOW_MOM_TOP20.release_hour_utc))
    from src.core.params import ACCOUNT_PASSIVE_WINDOW_BARS
    from src.live.executor import EXECUTION_BAR_SECONDS

    bar_ms = int(float(EXECUTION_BAR_SECONDS) * 1000)
    start_ms = int(release.value // 10**6)
    return start_ms, start_ms + int(ACCOUNT_PASSIVE_WINDOW_BARS) * bar_ms


def _rule_outcome(
    intents: list[dict[str, object]], archive_root: Path, day: date
) -> tuple[float | None, int, str | None, list[dict[str, object]]]:
    from src.application.fill_calibration_common import day_tape, read_tape_partition
    from src.core.params import ACCOUNT_MAKER_FEE_BPS, ACCOUNT_PASSIVE_WINDOW_BARS, ACCOUNT_TAKER_FEE_BPS
    from src.live.executor import EXECUTION_BAR_SECONDS

    bar_ms = int(float(EXECUTION_BAR_SECONDS) * 1000)
    start_ms, end_ms = _release_window_ms(day)
    month = day.strftime("%Y%m")
    symbols = sorted({str(item["symbol"]) for item in intents})
    partitions: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        from src.market_data.public_archive import store

        path = store.partition_path(archive_root, "aggTrades", symbol, month, kind="slice")
        if not path.exists():
            raise FileNotFoundError(str(path))
        partition = read_tape_partition(path)
        day_tape(partition, symbol, day)
        partitions[symbol] = partition
    maker_outcomes: list[tuple[dict[str, object], bool, float]] = []
    excluded = 0
    for item in intents:
        symbol = str(item["symbol"])
        frame = partitions[symbol]
        times = frame["transact_time_ms"].to_numpy()
        mask = (times >= start_ms) & (times < end_ms)
        sub = frame.loc[mask].sort_values(["transact_time_ms", "agg_trade_id"], kind="mergesort")
        bars: list[tuple[float, float, float]] = []
        if len(sub):
            idx = ((sub["transact_time_ms"].to_numpy(dtype="float64") - start_ms) // bar_ms).astype(int)
            sub = sub.copy()
            sub["_bar"] = idx
            for _, group in sub.groupby("_bar", sort=True):
                prices = group["price"].to_numpy(dtype="float64")
                bars.append((float(np.min(prices)), float(np.max(prices)), float(prices[-1])))
        if len(bars) < int(ACCOUNT_PASSIVE_WINDOW_BARS):
            excluded += 1
            continue
        ref = float(cast(Any, item["ref"]))
        side = str(item["side"])
        hit = any(low < ref for low, _, _ in bars) if side == "BUY" else any(high > ref for _, high, _ in bars)
        maker_outcomes.append((item, bool(hit), float(bars[-1][2])))
    included: list[dict[str, object]] = []
    for item, hit, close in maker_outcomes:
        ref = float(cast(Any, item["ref"]))
        side = str(item["side"])
        sign = 1.0 if side == "BUY" else -1.0
        if hit:
            rule_is = float(ACCOUNT_MAKER_FEE_BPS)
        else:
            rule_is = sign * (float(close) - ref) / ref * 1e4 + float(ACCOUNT_TAKER_FEE_BPS)
        if not np.isfinite(rule_is):
            raise DataIntegrityError("rule shortfall is non-finite")
        included.append({**item, "rule_is": rule_is})
    if not included:
        detail = f"excluded_insufficient_bars={excluded}" if excluded else None
        return None, 0, detail, []
    notionals = np.array([float(cast(Any, item["notional"])) for item in included], dtype="float64")
    rules = np.array([float(cast(Any, item["rule_is"])) for item in included], dtype="float")
    rule_mean = float(np.sum(rules * notionals) / np.sum(notionals)) if float(np.sum(notionals)) > 0 else float("nan")
    detail = f"excluded_insufficient_bars={excluded}" if excluded else None
    return rule_mean, len(included), detail, included


def _score_through_agreement(
    day: date, *, paper_run_dir: Path, archive_root: Path, live_capture_root: Path
) -> tuple[float | None, int]:
    from src.application.fill_calibration import FillCalibrationRequest
    from src.application.fill_calibration_paper import score_paper

    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    request = FillCalibrationRequest(
        evaluation_start=start,
        evaluation_end=end,
        archive_root=archive_root,
        k_bars=(1,),
        bootstrap_samples=100,
        seed=7,
        paper_run_dir=paper_run_dir,
        live_capture_root=live_capture_root,
    )
    rows, _ = score_paper(request, {})
    through = [row for row in rows if row.get("rule") == "through_only"]
    n_posts = len(through)
    if not n_posts:
        return None, 0
    both = sum(1 for row in through if bool(row.get("paper_filled")) and float(row.get("fill", 0.0)) > 1e-9)
    neither = sum(1 for row in through if not bool(row.get("paper_filled")) and not float(row.get("fill", 0.0)) > 1e-9)
    return float((both + neither) / n_posts), int(n_posts)


def measure_fill_parity(
    day: date, *, paper_run_dir: Path, archive_root: Path, live_capture_root: Path
) -> FillParityRow:
    """Measure one day's agreement between paper fills, tape replay and the backtest rule."""
    if not isinstance(day, date) or isinstance(day, pd.Timestamp):
        raise DataIntegrityError(f"day must be a date, got {day!r}")
    paper_run_dir = Path(paper_run_dir)
    archive_root = Path(archive_root)
    live_capture_root = Path(live_capture_root)
    run_id = paper_run_dir.name
    eq = _load_eq_frames(paper_run_dir)
    if len(eq):
        try:
            if "decision_time" not in eq.columns:
                raise DataIntegrityError("execution_quality missing decision_time")
            days = [_day_key(v) for v in eq["decision_time"].tolist()]
        except DataIntegrityError:
            raise
        except Exception as exc:
            raise DataIntegrityError(f"malformed execution_quality decision_time: {exc}") from exc
        eq = eq.copy()
        eq["_day"] = days
        eq_day = eq[(eq["_day"] == day) & (pd.to_datetime(eq["decision_time"], utc=True) == pd.Timestamp(day, tz="UTC"))]
    else:
        eq_day = eq
    if not len(eq_day):
        return FillParityRow(
            day=day, run_id=run_id, n_intents=0, n_posts=0,
            through_agreement=None, paper_is_bps=None, rule_is_bps=None,
            gap_bps=None, status="SKIPPED", detail="no_intents",
        )
    fills = _load_fills_frames(paper_run_dir)
    intents = _collect_intents(eq_day, fills)
    if not intents:
        return FillParityRow(
            day=day, run_id=run_id, n_intents=0, n_posts=0,
            through_agreement=None, paper_is_bps=None, rule_is_bps=None,
            gap_bps=None, status="SKIPPED", detail="no_intents",
        )
    try:
        rule_mean, n_included, rule_detail, included = _rule_outcome(intents, archive_root, day)
    except FileNotFoundError:
        return FillParityRow(
            day=day, run_id=run_id, n_intents=0, n_posts=0,
            through_agreement=None, paper_is_bps=None, rule_is_bps=None,
            gap_bps=None, status="SKIPPED", detail="tape_missing",
        )
    try:
        agreement, n_posts = _score_through_agreement(
            day, paper_run_dir=paper_run_dir, archive_root=archive_root, live_capture_root=live_capture_root
        )
    except DataIntegrityError as exc:
        if "tape partition missing" in str(exc):
            return FillParityRow(
                day=day, run_id=run_id, n_intents=0, n_posts=0,
                through_agreement=None, paper_is_bps=None, rule_is_bps=None,
                gap_bps=None, status="SKIPPED", detail="tape_missing",
            )
        raise
    if not included:
        return FillParityRow(
            day=day, run_id=run_id, n_intents=0, n_posts=int(n_posts),
            through_agreement=agreement, paper_is_bps=None, rule_is_bps=None,
            gap_bps=None, status="OK", detail=rule_detail,
        )
    notionals = np.array([float(cast(Any, item["notional"])) for item in included], dtype="float64")
    papers = np.array([float(cast(Any, item["paper_is"])) for item in included], dtype="float64")
    paper_mean = float(np.sum(papers * notionals) / np.sum(notionals))
    gap = float(paper_mean - float(rule_mean)) if rule_mean is not None else None
    return FillParityRow(
        day=day, run_id=run_id, n_intents=int(n_included), n_posts=int(n_posts),
        through_agreement=agreement, paper_is_bps=float(paper_mean),
        rule_is_bps=float(rule_mean) if rule_mean is not None else None,
        gap_bps=gap, status="OK", detail=rule_detail,
    )


def _row_payload(row: FillParityRow) -> dict[str, object]:
    return {
        "day": row.day.isoformat(),
        "detail": row.detail,
        "gap_bps": row.gap_bps,
        "n_intents": int(row.n_intents),
        "n_posts": int(row.n_posts),
        "paper_is_bps": row.paper_is_bps,
        "rule_is_bps": row.rule_is_bps,
        "run_id": row.run_id,
        "status": row.status,
        "through_agreement": row.through_agreement,
    }


def append_parity_row(row: FillParityRow, path: Path) -> None:
    """Append one parity row as a sorted-key JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(_row_payload(row), sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line)
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    try:
        os.chmod(path, 0o644)
    except OSError as exc:
        raise DataIntegrityError(f"parity report chmod failed: {exc}") from exc


def _parse_row(line: str) -> dict[str, object]:
    try:
        obj = json.loads(line)
    except ValueError as exc:
        raise DataIntegrityError(f"malformed parity line: {line!r}") from exc
    if not isinstance(obj, dict):
        raise DataIntegrityError(f"malformed parity line: {line!r}")
    try:
        day = date.fromisoformat(str(obj["day"]))
        status = str(obj["status"])
        if status not in ("OK", "SKIPPED"):
            raise ValueError("invalid status")
        return {
            "day": day,
            "run_id": str(obj["run_id"]),
            "n_intents": int(cast(Any, obj["n_intents"])),
            "n_posts": int(cast(Any, obj["n_posts"])),
            "through_agreement": None if obj.get("through_agreement") is None else float(cast(Any, obj.get("through_agreement"))),
            "paper_is_bps": None if obj.get("paper_is_bps") is None else float(cast(Any, obj.get("paper_is_bps"))),
            "rule_is_bps": None if obj.get("rule_is_bps") is None else float(cast(Any, obj.get("rule_is_bps"))),
            "gap_bps": None if obj.get("gap_bps") is None else float(cast(Any, obj.get("gap_bps"))),
            "status": status,
            "detail": None if obj.get("detail") is None else str(obj.get("detail")),
        }
    except (KeyError, ValueError, TypeError) as exc:
        raise DataIntegrityError(f"malformed parity line: {line!r}") from exc


def trailing_breach(path: Path, thresholds: FillParityThresholds) -> str | None:
    """Check the trailing window for an agreement or gap breach."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DataIntegrityError(f"unreadable parity report {path}: {exc}") from exc
    latest: dict[tuple[date, str], dict[str, object]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parsed = _parse_row(raw)
        latest[(cast(date, parsed["day"]), cast(str, parsed["run_id"]))] = parsed
    ok_rows = [row for row in latest.values() if row["status"] == "OK"]
    if not ok_rows:
        return None
    days = sorted({cast(date, row["day"]) for row in ok_rows})
    window_days = set(days[-int(thresholds.trailing_days):])
    window = [row for row in ok_rows if row["day"] in window_days]
    total_intents = int(sum(int(cast(Any, row["n_intents"])) for row in window))
    if total_intents < int(thresholds.min_intents):
        return None
    post_weight = sum(int(cast(Any, row["n_posts"])) for row in window if row["through_agreement"] is not None and math.isfinite(float(cast(Any, row["through_agreement"]))))
    agreement_mean: float | None = None
    if post_weight > 0:
        agreement_mean = float(
            sum(float(cast(Any, row["through_agreement"])) * int(cast(Any, row["n_posts"])) for row in window if row["through_agreement"] is not None and math.isfinite(float(cast(Any, row["through_agreement"])))) / post_weight
        )
    gap_weight = sum(int(cast(Any, row["n_intents"])) for row in window if row["gap_bps"] is not None and math.isfinite(float(cast(Any, row["gap_bps"]))))
    gap_mean: float | None = None
    if gap_weight > 0:
        gap_mean = float(
            sum(float(cast(Any, row["gap_bps"])) * int(cast(Any, row["n_intents"])) for row in window if row["gap_bps"] is not None and math.isfinite(float(cast(Any, row["gap_bps"])))) / gap_weight
        )
    parts: list[str] = []
    if agreement_mean is not None and agreement_mean < float(thresholds.min_agreement):
        parts.append(f"agreement={agreement_mean:.4f} below {float(thresholds.min_agreement):.4f}")
    if gap_mean is not None and abs(gap_mean) > float(thresholds.max_abs_gap_bps):
        parts.append(f"gap_bps={gap_mean:.4f} exceeds {float(thresholds.max_abs_gap_bps):.4f}")
    if not parts:
        return None
    return "; ".join(parts)
