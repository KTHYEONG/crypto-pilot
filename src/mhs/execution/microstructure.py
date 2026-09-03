"""Microstructure primitives — pure fill schedules and spread estimators."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from src.mhs.types import ExecutionSpec

from . import SPREAD_ESTIMATE_CEILING_BPS


def passive_fill_shortfall_bps(
    decision_price: float,
    adverse_path: np.ndarray,
    timeout_price: float,
    side: int,
    spec: ExecutionSpec,
    taker_cost_bps: float | None = None,
) -> float:
    """Implementation shortfall of one passive order against its decision price.

    ``side=+1`` is a buy and ``adverse_path`` carries the window's lows;
    ``side=-1`` is a sell and ``adverse_path`` carries the window's highs. A
    fill costs exactly the maker fee; a no-fill crosses at the timeout price
    and pays the all-in taker cost, so fee and adverse selection are always
    accounted together. ``taker_cost_bps`` overrides the flat slippage term
    when the caller supplies a liquidity-aware crossing cost; the default
    reproduces the frozen all-in taker cost bit-identically.
    """
    if decision_price <= 0 or timeout_price <= 0:
        raise ValueError("decision_price and timeout_price must be > 0")
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    if adverse_path.size == 0:
        raise ValueError("adverse_path must not be empty")

    extreme = float(np.min(adverse_path)) if side == 1 else float(np.max(adverse_path))
    if side == 1:
        filled = (
            extreme < decision_price if spec.require_trade_through else extreme <= decision_price
        )
    else:
        filled = (
            extreme > decision_price if spec.require_trade_through else extreme >= decision_price
        )
    if filled:
        return float(spec.maker_fee_bps)
    move_bps = side * (timeout_price / decision_price - 1.0) * 1e4
    slippage = float(spec.taker_slippage_bps) if taker_cost_bps is None else float(taker_cost_bps)
    return float(move_bps + spec.taker_fee_bps + slippage)


def notional_weighted_shortfall_bps(
    shortfalls: Iterable[float],
    notionals: Iterable[float],
) -> float:
    """Notional-weighted mean of per-fill implementation shortfalls in bps.

    The unweighted ``np.mean`` of per-fill shortfalls over-weights the many
    small fills that dominate the count but carry little capital. The
    economically correct aggregate weights each fill's shortfall by its
    absolute notional ``abs(qty) * fill_price``:
    ``sum(shortfall_i * notional_i) / sum(notional_i)``. Returns ``nan``
    (never ``0.0``, which would read as free execution, and never a
    ``ZeroDivisionError``) when no fills occurred or the total notional is
    zero.
    """
    shortfall_arr = np.asarray(list(shortfalls), dtype="float64")
    notional_arr = np.asarray(list(notionals), dtype="float64")
    if shortfall_arr.size == 0 or notional_arr.size == 0:
        return float("nan")
    total_notional = float(notional_arr.sum())
    if total_notional <= 0.0 or not np.isfinite(total_notional):
        return float("nan")
    return float(np.sum(shortfall_arr * notional_arr) / total_notional)


def laddered_fill_schedule(
    decision_price: float,
    side: int,
    adverse: np.ndarray,
    closes: np.ndarray,
    tranche_count: int,
    spec: ExecutionSpec,
    require_strict: bool,
) -> list[tuple[int, float, float, float]]:
    """Split one order into an escalating ladder of ``tranche_count`` limit sub-windows.

    The OHLCV execution window ``[0, len(adverse))`` is split into
    ``tranche_count`` equal-width sub-windows (the last absorbs any remainder
    bars); each sub-window reuses the existing binary trade-through predicate
    (``require_strict`` selects the strict ``<``/``>`` vs the touch ``<=``/``>=``
    comparison operators used at the inline STRICT/TOUCH branches) against its
    own limit price. Tranche 1 rests at ``decision_price``; tranche ``k > 1``
    reprices linearly toward the market by ``side * (k-1)/tranche_count`` of the
    gap to the previous sub-window's closing anchor ``closes[sub_end_{k-1}]``
    (the boundary bar's close, matching the codebase's timeout-close
    convention). A tranche whose sub-window trades through fills its accumulated
    carried share at its own limit price with the maker fee; only the final
    tranche, if it never trades through, converts its remaining share to an
    immediate market fill at the final sub-window's close with the all-in taker
    cost. Non-final tranches that fail carry their share forward without a
    market fallback.

    ``closes`` must be at least as long as ``adverse`` (the production callers
    pass one extra boundary close at index ``len(adverse)``); the market
    fallback uses that boundary close when present so ``tranche_count == 1``
    reproduces the pre-existing STRICT/TOUCH single-fill fallback exactly.
    Returns ``(relative_fill_position, fill_price, fee_bps, qty_fraction)``
    tuples in fill order with ``qty_fraction`` summing to 1.0.
    """
    if tranche_count < 1:
        raise ValueError(f"tranche_count must be >= 1, got {tranche_count}")
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    if decision_price <= 0:
        raise ValueError("decision_price must be > 0")
    n = len(adverse)
    if n == 0:
        raise ValueError("adverse must not be empty")
    if not np.isfinite(adverse).all():
        raise ValueError("adverse must be finite")
    if len(closes) < n:
        raise ValueError("closes must be at least as long as adverse")
    if not np.isfinite(closes[: n + 1]).all():
        raise ValueError("closes must be finite")

    schedule: list[tuple[int, float, float, float]] = []
    carried = 0.0
    own_share = 1.0 / tranche_count
    for k in range(1, tranche_count + 1):
        sub_start = (k - 1) * n // tranche_count
        sub_end = k * n // tranche_count if k < tranche_count else n
        if k == 1:
            limit_price = decision_price
        else:
            anchor = float(closes[sub_start])
            limit_price = decision_price + side * (k - 1) / tranche_count * (anchor - decision_price)
        sub = adverse[sub_start:sub_end]
        if side == 1:
            crossed = (sub < limit_price) if require_strict else (sub <= limit_price)
        else:
            crossed = (sub > limit_price) if require_strict else (sub >= limit_price)
        if crossed.any():
            hit = int(np.argmax(crossed))
            schedule.append(
                (sub_start + hit, float(limit_price), float(spec.maker_fee_bps), carried + own_share)
            )
            carried = 0.0
        else:
            carried += own_share
    if carried > 0.0:
        fallback_close = float(closes[min(n, len(closes) - 1)])
        schedule.append(
            (n, fallback_close, float(spec.taker_fee_bps + spec.taker_slippage_bps), carried)
        )
    return schedule


def peg_chase_fill_schedule(
    anchor_price: float,
    side: int,
    adverse: np.ndarray,
    closes: np.ndarray,
    spec: ExecutionSpec,
    taker_cost_bps: float | None = None,
) -> tuple[int, float, float, str] | None:
    """Repricing peg-chase schedule for one order over its execution window.

    Window bar ``w = 0..N-1`` maps to grid position ``spos + w``. The limit
    price re-pegs each bar to the previous bar's close (``peg[0] =
    anchor_price``, ``peg[w] = closes[w-1]``), reproducing the live
    own-touch re-peg at bar granularity. Only the adverse direction is
    clamped: ``cap = anchor_price * (1 + side * peg_chase_band_bps / 1e4)``
    bounds a buy's peg from above and a sell's from below, while a favourable
    excursion carries the peg freely (a maker fill below the band is the
    point of chasing).

    During the passive phase ``w < P`` with
    ``P = min(N, max(1, ceil(peg_passive_fraction * N)))`` a strict
    trade-through fills as maker (buy: ``adverse[w] < peg[w]``). The taker
    backstop crosses UNCONDITIONALLY at the first bar ``w >= min(P, N-1)``
    with a finite close: the band caps the maker peg only and is never a
    refusal to trade, and the final bar stays backstop-eligible so the
    schedule completes even when the passive fraction consumes the whole
    window. ``None`` is returned exactly when the window holds no finite
    close (a data gap), never as a pricing decision.

    ``taker_cost_bps`` overrides the all-in backstop fee term when the caller
    supplies a liquidity-aware cost; otherwise it defaults to
    ``taker_fee_bps + taker_slippage_bps``.

    Returns ``(relative_fill_position, fill_price, fee_bps, reason)`` with
    ``reason in {"maker_fill", "backstop_taker"}``, or ``None``.
    """
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    if anchor_price <= 0:
        raise ValueError(f"anchor_price must be > 0, got {anchor_price}")
    n = len(adverse)
    if n == 0:
        raise ValueError("adverse must not be empty")
    if not np.isfinite(adverse).all():
        raise ValueError("adverse must be finite")
    if len(closes) < n:
        raise ValueError("closes must be at least as long as adverse")

    finite_close = np.isfinite(closes[:n])
    if not finite_close.any():
        return None

    peg = np.empty(n, dtype="float64")
    peg[0] = anchor_price
    if n > 1:
        peg[1:] = closes[: n - 1]
    cap = anchor_price * (1.0 + side * spec.peg_chase_band_bps / 1e4)
    if side == 1:
        peg = np.minimum(peg, cap)
        crossed = adverse < peg
    else:
        peg = np.maximum(peg, cap)
        crossed = adverse > peg
    passive_len = min(n, max(1, math.ceil(spec.peg_passive_fraction * n)))
    passive_bars = np.arange(n) < passive_len
    maker_hits = crossed & passive_bars
    if maker_hits.any():
        hit = int(np.argmax(maker_hits))
        return (hit, float(peg[hit]), float(spec.maker_fee_bps), "maker_fill")
    backstop_from = min(passive_len, n - 1)
    backstop_hits = (np.arange(n) >= backstop_from) & finite_close
    if backstop_hits.any():
        hit = int(np.argmax(backstop_hits))
        fee_bps = (
            float(taker_cost_bps)
            if taker_cost_bps is not None
            else float(spec.taker_fee_bps + spec.taker_slippage_bps)
        )
        return (
            hit,
            float(closes[hit]),
            fee_bps,
            "backstop_taker",
        )
    return None

def peg_chase_partial_schedule(
    anchor_price: float,
    side: int,
    adverse: np.ndarray,
    closes: np.ndarray,
    spec: ExecutionSpec,
    taker_cost_bps: float | None = None,
) -> list[tuple[int, float, float, float, str]]:
    """Tranched peg-chase schedule: equal sub-windows with carried quantity.

    The window splits into ``spec.peg_chase_tranches`` equal sub-windows; each
    tranche re-pegs from ``anchor_price`` and follows the same previous-close
    chain (band-capped on the adverse side only) as ``peg_chase_fill_schedule``.
    A tranche whose sub-window trades through fills its carried share as maker
    at its peg; a failed non-final tranche carries its share forward; the final
    residual crosses unconditionally at the deadline bar exactly like the
    single-window backstop (passive fraction, then first finite close). With
    ``peg_chase_tranches == 1`` the output is exactly the single-element
    equivalent of ``peg_chase_fill_schedule``, so the default reproduces the
    all-or-nothing fill bit-identically.

    Returns ``(relative_fill_position, price, fee_bps, qty_fraction, reason)``
    tuples in fill order; the ``qty_fraction`` values sum to 1.0 within double
    precision whenever a finite close exists. An empty list is returned
    exactly when the window holds no usable finite close for completion (a
    data gap), never as a pricing decision.

    ``taker_cost_bps`` overrides the all-in backstop fee term when the caller
    supplies a liquidity-aware cost; otherwise it defaults to
    ``taker_fee_bps + taker_slippage_bps``.
    """
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    if anchor_price <= 0:
        raise ValueError(f"anchor_price must be > 0, got {anchor_price}")
    tranches = int(spec.peg_chase_tranches)
    if tranches < 1:
        raise ValueError(f"peg_chase_tranches must be >= 1, got {spec.peg_chase_tranches}")
    n = len(adverse)
    if n == 0:
        raise ValueError("adverse must not be empty")
    if not np.isfinite(adverse).all():
        raise ValueError("adverse must be finite")
    if len(closes) < n:
        raise ValueError("closes must be at least as long as adverse")

    finite_close = np.isfinite(closes[:n])
    if not finite_close.any():
        return []

    cap = anchor_price * (1.0 + side * spec.peg_chase_band_bps / 1e4)
    backstop_fee = (
        float(taker_cost_bps)
        if taker_cost_bps is not None
        else float(spec.taker_fee_bps + spec.taker_slippage_bps)
    )
    boundaries = [int(b) for b in np.linspace(0, n, tranches + 1)]

    def _subwindow_fill(lo: int, hi: int, final: bool) -> tuple[int, float, float, str] | None:
        length = hi - lo
        if length <= 0:
            return None
        peg = np.empty(length, dtype="float64")
        peg[0] = anchor_price
        if length > 1:
            peg[1:] = closes[lo : lo + length - 1]
        if side == 1:
            local_peg = np.minimum(peg, cap)
            crossed = adverse[lo:hi] < local_peg
        else:
            local_peg = np.maximum(peg, cap)
            crossed = adverse[lo:hi] > local_peg
        passive_len = min(length, max(1, math.ceil(spec.peg_passive_fraction * length)))
        if final:
            crossed = crossed & (np.arange(length) < passive_len)
        maker_hits = np.flatnonzero(crossed)
        if maker_hits.size > 0:
            hit = int(maker_hits[0])
            return (hit, float(local_peg[hit]), float(spec.maker_fee_bps), "maker_fill")
        if not final:
            return None
        backstop_from = min(passive_len, length - 1)
        backstop_hits = np.flatnonzero(
            (np.arange(length) >= backstop_from) & finite_close[lo:hi]
        )
        if backstop_hits.size > 0:
            hit = int(backstop_hits[0])
            return (hit, float(closes[lo + hit]), backstop_fee, "backstop_taker")
        return None

    schedule: list[tuple[int, float, float, float, str]] = []
    filled_fraction = 0.0
    carried = 0.0
    for index in range(tranches):
        lo = boundaries[index]
        hi = boundaries[index + 1]
        final = index == tranches - 1
        share = 1.0 / tranches + carried
        carried = 0.0
        filled = _subwindow_fill(lo, hi, final)
        if filled is None:
            if final:
                # No completable crossing (data gap): mirror the single-window
                # None semantics by discarding the schedule to the residual path.
                return []
            carried = share
            continue
        rel_pos, price, fee_bps, reason = filled
        qty_fraction = (1.0 - filled_fraction) if final else share
        schedule.append((lo + rel_pos, price, fee_bps, qty_fraction, reason))
        filled_fraction += qty_fraction
    return schedule


def corwin_schultz_half_spread_bps(
    highs: np.ndarray,
    lows: np.ndarray,
) -> np.ndarray:
    """Column-wise Corwin-Schultz (2012) effective-spread half-spread in bps.

    Estimates one half-spread per column from consecutive 2-bar high/low
    pairs: ``b`` sums the squared per-bar log ranges, ``g`` squares the
    2-bar log range of the union, and the spread follows
    ``S = 2*(e^a - 1)/(e^a + 1)`` with
    ``a = (sqrt(2b) - sqrt(b))/k - sqrt(g/k)`` and ``k = 3 - 2*sqrt(2)``.
    Negative 2-bar estimates -- the estimator's known degeneracy on quiet
    pairs -- are floored at 0 before averaging; the column mean is halved to
    a half-spread, converted to bps, and clipped to
    ``[0, SPREAD_ESTIMATE_CEILING_BPS]`` so a degenerate sequence can never
    price an unbounded crossing.

    Bars whose high/low are non-finite, non-positive, or inverted
    (``high < low``) are masked out, and non-adjacent valid bars are never
    paired. A column with fewer than 3 valid bars yields ``nan`` and the
    caller falls back to the flat slippage. Input shape ``(n_bars, n_cols)``
    with matching highs/lows and at least 2 rows (both fail closed with
    ``ValueError``); output shape ``(n_cols,)``. Fully vectorised over bars.
    """
    high_arr = np.asarray(highs, dtype="float64")
    low_arr = np.asarray(lows, dtype="float64")
    if high_arr.shape != low_arr.shape:
        raise ValueError(
            f"highs and lows must share one shape, got {high_arr.shape} vs {low_arr.shape}"
        )
    if high_arr.ndim != 2 or high_arr.shape[0] < 2:
        raise ValueError(f"highs/lows must be 2-D with >= 2 rows, got shape {high_arr.shape}")

    valid = (
        np.isfinite(high_arr)
        & np.isfinite(low_arr)
        & (high_arr > 0.0)
        & (low_arr > 0.0)
        & (high_arr >= low_arr)
    )
    pair_ok = valid[:-1] & valid[1:]

    h0 = np.where(pair_ok, high_arr[:-1], np.nan)
    l0 = np.where(pair_ok, low_arr[:-1], np.nan)
    h1 = np.where(pair_ok, high_arr[1:], np.nan)
    l1 = np.where(pair_ok, low_arr[1:], np.nan)

    b = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
    g = np.log(np.maximum(h0, h1) / np.minimum(l0, l1)) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    a = (np.sqrt(2.0 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    # tanh(a/2) == (e^a - 1)/(e^a + 1) without the overflow at large |a|.
    spread = 2.0 * np.tanh(a / 2.0)
    floored = np.maximum(spread, 0.0)

    pair_counts = np.sum(np.isfinite(floored), axis=0)
    pair_sums = np.nansum(floored, axis=0)
    mean_spread = np.where(
        pair_counts > 0, pair_sums / np.where(pair_counts > 0, pair_counts, 1), np.nan,
    )
    half_bps = mean_spread / 2.0 * 1e4
    bar_counts = np.sum(valid, axis=0)
    half_bps = np.where(bar_counts >= 3, half_bps, np.nan)
    return np.clip(half_bps, 0.0, SPREAD_ESTIMATE_CEILING_BPS)
