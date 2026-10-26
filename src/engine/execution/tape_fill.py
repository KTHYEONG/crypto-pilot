"""Queue-aware tape fill model: pure resting post-only fill primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import AGG_TRADES_SLICE_COLUMNS


def _int64_column(frame: pd.DataFrame, column: str) -> NDArray[np.int64]:
    """Convert integral slice values without rounding native int64 identifiers."""
    raw = frame[column].to_numpy()
    if raw.size:
        if raw.dtype.kind not in "iuf":
            raise DataIntegrityError(f"{column} must contain int64 values")
        if raw.dtype.kind == "f" and (not np.isfinite(raw).all() or not (raw == np.floor(raw)).all()):
            raise DataIntegrityError(f"{column} must contain finite integral values")
        if (raw < -(2**63)).any() or (raw >= 2**63).any():
            raise DataIntegrityError(f"{column} values exceed int64 range")
    return np.array(raw, dtype=np.int64, copy=True)


@dataclass(frozen=True, slots=True, eq=False)
class TapeTrades:
    """One symbol's aggTrades tape sorted by (time, trade id) with its completeness interval."""

    time_ms: NDArray[np.int64]
    price: NDArray[np.float64]
    qty: NDArray[np.float64]
    buyer_maker: NDArray[np.bool_]
    coverage_ms: tuple[int, int]

    @classmethod
    def from_slice_frame(cls, frame: pd.DataFrame, *, coverage_ms: tuple[int, int]) -> TapeTrades:
        "Validate one symbol's aggTrades slice rows and wrap them as sorted arrays; raises DataIntegrityError."
        if not isinstance(frame, pd.DataFrame):
            raise DataIntegrityError(f"slice frame must be a DataFrame, got {type(frame).__name__}")
        missing = [c for c in AGG_TRADES_SLICE_COLUMNS if c not in frame.columns]
        if missing:
            raise DataIntegrityError(f"slice frame missing columns: {missing}")
        try:
            start, end = coverage_ms
        except (TypeError, ValueError) as exc:
            raise DataIntegrityError(f"coverage_ms must be a (start, end) pair, got {coverage_ms!r}") from exc
        if (
            not all(
                isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                for value in (start, end)
            )
            or start >= end
        ):
            raise DataIntegrityError(f"coverage_ms must satisfy start < end, got {(start, end)}")
        time_ms = _int64_column(frame, "transact_time_ms")
        trade_id = _int64_column(frame, "agg_trade_id")
        try:
            price = np.asarray(frame["price"].to_numpy(), dtype="float64")
            qty = np.asarray(frame["quantity"].to_numpy(), dtype="float64")
            raw_bm = frame["is_buyer_maker"].to_numpy()
            if raw_bm.size and raw_bm.dtype.kind != "b":
                raise DataIntegrityError("is_buyer_maker must contain booleans")
            buyer_maker = np.asarray(raw_bm, dtype=bool)
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError(f"slice frame holds non-numeric trade rows: {exc}") from exc
        if price.size and not (np.isfinite(price).all() and bool((price > 0.0).all())):
            raise DataIntegrityError("price must be finite and positive")
        if qty.size and not (np.isfinite(qty).all() and bool((qty > 0.0).all())):
            raise DataIntegrityError("quantity must be finite and positive")
        if time_ms.size and bool(((time_ms < start) | (time_ms >= end)).any()):
            raise DataIntegrityError("slice rows must lie inside coverage_ms")
        order = np.lexsort((trade_id, time_ms))
        return cls(
            time_ms=np.ascontiguousarray(time_ms[order]),
            price=np.ascontiguousarray(price[order]),
            qty=np.ascontiguousarray(qty[order]),
            buyer_maker=np.ascontiguousarray(buyer_maker[order]),
            coverage_ms=(start, end),
        )


@dataclass(frozen=True, slots=True)
class RestingOrder:
    """One resting post-only limit order with its active window and optional queue-ahead estimate."""

    side: Literal["BUY", "SELL"]
    limit_price: float
    quantity: float
    active_from_ms: int
    active_until_ms: int
    queue_ahead_qty: float | None

    def __post_init__(self) -> None:
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {self.side!r}")
        if not math.isfinite(self.limit_price) or self.limit_price <= 0.0:
            raise ValueError(f"limit_price must be finite and > 0, got {self.limit_price}")
        if not math.isfinite(self.quantity) or self.quantity <= 0.0:
            raise ValueError(f"quantity must be finite and > 0, got {self.quantity}")
        if self.active_until_ms <= self.active_from_ms:
            raise ValueError(
                f"active window must satisfy from < until, got {(self.active_from_ms, self.active_until_ms)}"
            )
        if self.queue_ahead_qty is not None and (not math.isfinite(self.queue_ahead_qty) or self.queue_ahead_qty < 0.0):
            raise ValueError(f"queue_ahead_qty must be None or finite and >= 0, got {self.queue_ahead_qty}")


@dataclass(frozen=True, slots=True)
class TapeFillSpec:
    """Typed fill-model configuration; no magic numbers in the fill logic."""

    queue_ahead_fraction: float = 1.0
    unknown_queue: Literal["through_only", "front_of_queue"] = "through_only"
    qty_rtol: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.queue_ahead_fraction <= 1.0:
            raise ValueError(f"queue_ahead_fraction must lie in [0, 1], got {self.queue_ahead_fraction}")
        if self.unknown_queue not in ("through_only", "front_of_queue"):
            raise ValueError(f"unknown_queue must be through_only or front_of_queue, got {self.unknown_queue!r}")
        if not math.isfinite(self.qty_rtol) or not self.qty_rtol > 0.0:
            raise ValueError(f"qty_rtol must be finite and > 0, got {self.qty_rtol}")


@dataclass(frozen=True, slots=True)
class TapeFillResult:
    """Fill path of one resting order; fills carry (time_ms, qty) increments and price is limit_price."""

    filled_qty: float
    fills: tuple[tuple[int, float], ...]
    first_fill_ms: int | None
    completed_ms: int | None
    through_ms: int | None


def simulate_resting_fill(order: RestingOrder, tape: TapeTrades, spec: TapeFillSpec) -> TapeFillResult:
    "Fill path of one post-only resting order against the tape over its active window; raises DataIntegrityError if the window exceeds tape coverage."
    start, end = tape.coverage_ms
    if order.active_from_ms < start or order.active_until_ms > end:
        raise DataIntegrityError(
            f"order window {(order.active_from_ms, order.active_until_ms)} exceeds tape coverage {(start, end)}"
        )
    lo, hi = np.searchsorted(tape.time_ms, (order.active_from_ms, order.active_until_ms), side="left")
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return TapeFillResult(filled_qty=0.0, fills=(), first_fill_ms=None, completed_ms=None, through_ms=None)
    wtime = tape.time_ms[lo:hi]
    wprice = tape.price[lo:hi]
    wqty = tape.qty[lo:hi]
    wbm = tape.buyer_maker[lo:hi]
    limit = order.limit_price
    quantity = order.quantity
    if order.side == "BUY":
        through = wprice < limit
        eligible = (wprice == limit) & wbm
    else:
        through = wprice > limit
        eligible = (wprice == limit) & ~wbm
    hit = np.flatnonzero(through)
    through_ms = int(wtime[hit[0]]) if hit.size else None
    if order.queue_ahead_qty is None and spec.unknown_queue == "through_only":
        if through_ms is None:
            return TapeFillResult(filled_qty=0.0, fills=(), first_fill_ms=None, completed_ms=None, through_ms=None)
        fill = ((through_ms, float(quantity)),)
        return TapeFillResult(
            filled_qty=float(quantity),
            fills=fill,
            first_fill_ms=through_ms,
            completed_ms=through_ms,
            through_ms=through_ms,
        )
    ahead = 0.0 if order.queue_ahead_qty is None else order.queue_ahead_qty * spec.queue_ahead_fraction
    target = np.clip(np.cumsum(np.where(eligible, wqty, 0.0)) - ahead, 0.0, quantity)
    done = np.flatnonzero((target > 0.0) & ((quantity - target) <= spec.qty_rtol * quantity))
    matured = int(done[0]) if done.size else None
    first_through = int(hit[0]) if hit.size else None
    if first_through is not None and (matured is None or first_through < matured):
        width = first_through + 1
        increments = np.empty(width, dtype="float64")
        increments[0] = target[0]
        if width > 1:
            increments[1:] = np.diff(target[:width])
        increments[width - 1] = quantity - (target[width - 2] if width > 1 else 0.0)
        completed = True
    else:
        width = (matured + 1) if matured is not None else (hi - lo)
        increments = np.empty(width, dtype="float64")
        increments[0] = target[0]
        if width > 1:
            increments[1:] = np.diff(target[:width])
        completed = matured is not None
    # Sorted tape timestamps allow linear grouping without sorting the window again.
    boundaries = np.r_[0, np.flatnonzero(wtime[1:width] != wtime[: width - 1]) + 1]
    distinct = wtime[boundaries]
    grouped = np.add.reduceat(increments, boundaries)
    kept = grouped > 0.0
    fill_times = [int(t) for t in distinct[kept]]
    fill_qtys = [float(q) for q in grouped[kept]]
    if completed:
        fill_qtys[-1] += quantity - math.fsum(fill_qtys)
        completed_ms = fill_times[-1]
        filled = float(quantity)
    else:
        completed_ms = None
        filled = float(math.fsum(fill_qtys))
    return TapeFillResult(
        filled_qty=filled,
        fills=tuple(zip(fill_times, fill_qtys, strict=True)),
        first_fill_ms=fill_times[0] if fill_times else None,
        completed_ms=completed_ms,
        through_ms=through_ms,
    )


def touch_proxy(tape: TapeTrades, at_ms: int, lookback_ms: int) -> tuple[float, float] | None:
    "Best bid/ask proxy at `at_ms` from the latest seller-/buyer-aggressor prints in `[at_ms - lookback_ms, at_ms)`; None when either side is missing or bid >= ask."
    if lookback_ms <= 0:
        raise ValueError(f"lookback_ms must be > 0, got {lookback_ms}")
    low = at_ms - lookback_ms
    start, end = tape.coverage_ms
    if low < start or at_ms > end:
        raise DataIntegrityError(f"lookback [{low}, {at_ms}) exceeds tape coverage {(start, end)}")
    lo, hi = np.searchsorted(tape.time_ms, (low, at_ms), side="left")
    segment = tape.buyer_maker[int(lo) : int(hi)]
    bids = np.flatnonzero(segment)
    asks = np.flatnonzero(~segment)
    if not bids.size or not asks.size:
        return None
    bid = float(tape.price[int(lo) + int(bids[-1])])
    ask = float(tape.price[int(lo) + int(asks[-1])])
    if bid >= ask:
        return None
    return (bid, ask)


def post_only_limit(side: Literal["BUY", "SELL"], anchor: float, touch: tuple[float, float]) -> float:
    "Least-improved post-only price for an anchored order: BUY min(anchor, bid), SELL max(anchor, ask)."
    # Using the touch itself is deliberately conservative (it never improves inside the spread).
    bid, ask = touch
    for value in (anchor, bid, ask):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"anchor and touch must be finite and > 0, got {(anchor, touch)}")
    if side == "BUY":
        return float(min(anchor, bid))
    if side == "SELL":
        return float(max(anchor, ask))
    raise ValueError(f"side must be BUY or SELL, got {side!r}")
