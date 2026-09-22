"""Account-scale replay of a unit-exposure target book as one real USDT account."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import VenueRuleSnapshot
from src.mhs.account_policy import (
    ExposurePolicy,
    bayesian_unit_moments,
    build_venue_ladders,
    choose_exposure,
    maintenance_and_initial_margin,
)


@dataclass(frozen=True, slots=True)
class AccountMarkPanels:
    """Aligned 3m mark panels, built once and reused across replays."""

    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame


@dataclass(frozen=True, slots=True)
class AccountLedgerResult:
    """Account-scale replay outcome in USDT; daily series indexed by UTC date.

    intraday_max_drawdown: running-peak drawdown magnitude (>= 0) of the 3m close-marked equity
        path, the same definition as the canonical ledger's base_max_drawdown; 1.0 after liquidation.
    maker_fill_fraction: share of sent notional filled passively (0.0 under taker execution).
    """

    capital: float
    daily_equity: pd.Series
    daily_exposure: pd.Series
    liquidated_at: pd.Timestamp | None
    skipped_orders: int
    untraded_fraction: float
    initial_margin_breaches: int
    fee_paid: float
    impact_paid: float
    funding_paid: float
    fallback_ladder_symbols: tuple[str, ...]
    missing_filter_symbols: tuple[str, ...]
    intraday_max_drawdown: float
    maker_fill_fraction: float = 0.0


def replay_account(
    unit_weights: pd.DataFrame,
    marks: AccountMarkPanels,
    funding_cum: pd.DataFrame,
    adv: pd.DataFrame,
    daily_sigma: pd.DataFrame,
    rules: VenueRuleSnapshot,
    policy: ExposurePolicy,
    *,
    anchor_times: pd.DatetimeIndex,
    capital: float,
    taker_fee_bps: float,
    apply_order_filters: bool = True,
    unit_equity: pd.Series | None = None,
    execution: Literal["taker", "maker"] = "taker",
    maker_fee_bps: float | None = None,
    passive_window_bars: int | None = None,
) -> AccountLedgerResult:
    """Replay a unit-exposure target book as one real USDT account.

    At each entry the policy picks exposure from equity marked at the entry's
    anchor bar close (``anchor_times``: the last close observable at submission, the canonical
    ``submit_bar`` anchor); orders are sized and priced there, and quantities are held until the
    next entry's anchor. Between entries quantities are fixed
    and marked on 3m closes; maintenance margin is evaluated on each bar's adverse low/high, and
    the first bar whose adverse equity falls below total maintenance margin liquidates the account
    (equity 0, replay stops).

    ``growth`` policies size each entry from posterior moments of the unit-exposure book:
    ``unit_equity`` is that book's daily equity (exposure 1, no filters, no impact) on the
    same entry index, and the entry at date d uses only its returns realized at entries
    strictly before d (returns 1..d-1), so today's move never informs today's size.

    ``execution="maker"`` replaces the immediate taker fill with the canonical strict passive
    rule: each order rests at the anchor close for the next ``passive_window_bars``
    3m bars after the anchor of the same holding segment; a buy fills at the
    anchor only if a finite bar low trades strictly below it (a sell: a high strictly above),
    paying ``maker_fee_bps``; otherwise it crosses at the last finite close of that window
    paying ``taker_fee_bps``. Order sizing, step rounding, minimum-notional skips and
    square-root impact are decided at the anchor exactly as under taker execution, so impact
    stays a capacity charge on every sent notional regardless of liquidity. Margin and
    liquidation are evaluated on post-trade quantities for the whole segment.

    ``intraday_max_drawdown`` is measured on the 3m close-marked equity path across all
    segments (running peak from ``capital``), so it is directly comparable with the canonical
    ledger's drawdown; ``daily_equity`` keeps the post-trade anchor equity per entry label.

    Raises:
        DataIntegrityError: Misaligned indexes/columns, non-positive capital, or no entry row.
        DataIntegrityError: growth policy without unit_equity, or unit_equity misaligned
            with unit_weights, non-finite, or non-positive.
        DataIntegrityError: maker execution without maker_fee_bps/passive_window_bars, a
            negative or non-finite maker fee, a maker fee above the taker fee, or passive_window_bars < 1.
        DataIntegrityError: ``anchor_times`` not aligned 1:1 with ``unit_weights``, not
            strictly increasing, or containing a bar absent from the mark panels.
    """
    if not np.isfinite(capital) or capital <= 0:
        raise DataIntegrityError(f"capital must be positive, got {capital}")
    if execution not in ("taker", "maker"):
        raise DataIntegrityError(f"execution must be 'taker' or 'maker', got {execution}")
    is_maker = execution == "maker"
    maker_fee_rate = 0.0
    window_bars = 0
    if is_maker:
        if maker_fee_bps is None:
            raise DataIntegrityError("maker execution requires maker_fee_bps")
        if not np.isfinite(maker_fee_bps) or maker_fee_bps < 0 or maker_fee_bps > taker_fee_bps:
            raise DataIntegrityError(f"invalid maker_fee_bps {maker_fee_bps} for taker_fee_bps {taker_fee_bps}")
        if isinstance(passive_window_bars, bool) or not isinstance(passive_window_bars, (int, np.integer)):
            raise DataIntegrityError(f"passive_window_bars must be an int >= 1, got {passive_window_bars}")
        if int(passive_window_bars) < 1:
            raise DataIntegrityError(f"passive_window_bars must be >= 1, got {passive_window_bars}")
        maker_fee_rate = float(maker_fee_bps) / 1e4
        window_bars = int(passive_window_bars)
    dates = unit_weights.index
    if len(dates) == 0:
        raise DataIntegrityError("unit_weights carries no entry row")
    reference_columns = list(unit_weights.columns)
    for name, frame in (("funding_cum", funding_cum), ("adv", adv), ("daily_sigma", daily_sigma)):
        if list(frame.columns) != reference_columns:
            raise DataIntegrityError(f"{name} columns misaligned with unit_weights")
        if not frame.index.equals(dates):
            raise DataIntegrityError(f"{name} index misaligned with unit_weights")
    if not (marks.high.index.equals(marks.close.index) and marks.low.index.equals(marks.close.index)):
        raise DataIntegrityError("mark panels carry misaligned indexes")
    if not (list(marks.high.columns) == list(marks.close.columns) and list(marks.low.columns) == list(marks.close.columns)):
        raise DataIntegrityError("mark panels carry misaligned columns")
    symbols = list(reference_columns)
    absent = [symbol for symbol in symbols if symbol not in marks.close.columns]
    if absent:
        raise DataIntegrityError(f"mark panels miss weight symbols {absent}")
    unit_values: np.ndarray | None = None
    unit_cum: np.ndarray | None = None
    unit_cum_sq: np.ndarray | None = None
    if policy.kind == "growth":
        if unit_equity is None:
            raise DataIntegrityError("growth policy requires unit_equity")
        if not unit_equity.index.equals(dates):
            raise DataIntegrityError("unit_equity index misaligned with unit_weights")
        unit_values = unit_equity.to_numpy(dtype="float64")
        if not bool(np.isfinite(unit_values).all()) or bool((unit_values <= 0).any()):
            raise DataIntegrityError("unit_equity must be finite and positive")
        unit_returns = unit_values[1:] / unit_values[:-1] - 1.0
        unit_cum = np.cumsum(unit_returns)
        unit_cum_sq = np.cumsum(unit_returns * unit_returns)
    ladders, fallback_symbols = build_venue_ladders(symbols, rules)
    steps = np.full(len(symbols), np.nan)
    mins = np.full(len(symbols), np.nan)
    missing_filters: list[str] = []
    for position, symbol in enumerate(symbols):
        entry = rules.symbols.get(symbol)
        step = entry.step_size if entry is not None else None
        minimum = entry.min_notional if entry is not None else None
        if step is not None:
            steps[position] = step
        if minimum is not None:
            mins[position] = minimum
        if step is None or minimum is None:
            missing_filters.append(symbol)
    bar_index = marks.close.index
    if len(anchor_times) != len(dates):
        raise DataIntegrityError("anchor_times not aligned 1:1 with unit_weights")
    anchor_pos = bar_index.get_indexer(anchor_times)
    if bool((anchor_pos < 0).any()):
        raise DataIntegrityError("anchor bar absent from the mark panels")
    if len(anchor_pos) > 1 and bool((anchor_pos[1:] <= anchor_pos[:-1]).any()):
        raise DataIntegrityError("anchor_times must be strictly increasing")
    n_bars = len(bar_index)
    # unit_weights carries every symbol EVER held across the full window, but a symbol not yet
    # listed (or already delisted) has no price for the bars outside its life -- forward-fill in
    # _read_held_marks only covers gaps AFTER a symbol's first bar, so leading/trailing NaN marks
    # remain. Weight there is always exactly 0.0 (the PIT roster gates it), but IEEE754 makes
    # 0.0 * NaN = NaN, which would otherwise poison quantities and equity from day 0 onward.
    # Panels stay float32 (one copy of the three planes); only the current segment or maker
    # window is converted to float64 with a neutral finite placeholder, so peak additional
    # memory stays at or below one float32 copy of the three panels.
    close_f32 = marks.close[symbols].to_numpy(dtype="float32")
    high_f32 = marks.high[symbols].to_numpy(dtype="float32")
    low_f32 = marks.low[symbols].to_numpy(dtype="float32")
    weight_values = unit_weights.to_numpy(dtype="float64")
    adv_values = adv.to_numpy(dtype="float64")
    sigma_values = daily_sigma.to_numpy(dtype="float64")
    funding_step = np.vstack([np.zeros((1, len(symbols))), np.diff(funding_cum.to_numpy(dtype="float64"), axis=0)])
    days = len(dates)
    count = len(symbols)
    daily_equity = np.full(days, np.nan)
    daily_exposure = np.zeros(days)
    cash = float(capital)
    quantities = np.zeros(count)
    skipped_orders = 0
    skipped_notional = 0.0
    intended_notional = 0.0
    fee_paid = 0.0
    impact_paid = 0.0
    funding_paid = 0.0
    margin_breaches = 0
    liquidated_at: pd.Timestamp | None = None
    fee_rate = taker_fee_bps / 1e4
    maker_anchor_sent = 0.0
    total_anchor_sent = 0.0
    running_peak = float(capital)
    max_drawdown = 0.0
    for day in range(days):
        anchor = int(anchor_pos[day])
        price = np.nan_to_num(close_f32[anchor].astype(np.float64), nan=1.0)
        held = quantities * price
        marked = cash + float(held.sum())
        charge = float((held * funding_step[day]).sum())
        cash -= charge
        funding_paid += charge
        equity = marked - charge
        moments = None
        if policy.kind == "growth":
            assert unit_cum is not None
            assert unit_cum_sq is not None
            n = max(0, day - 1)
            total = float(unit_cum[n - 1]) if n > 0 else 0.0
            total_sq = float(unit_cum_sq[n - 1]) if n > 0 else 0.0
            moments = bayesian_unit_moments(
                n, total, total_sq,
                prior_days=policy.prior_days, min_moment_days=policy.min_moment_days,
            )
        exposure = choose_exposure(
            weight_values[day], equity, held, adv_values[day], sigma_values[day], ladders, policy,
            moments,
        )
        target = exposure * equity * weight_values[day]
        _, initial_margin = maintenance_and_initial_margin(np.abs(target), ladders)
        if float(initial_margin.sum()) > policy.initial_margin_cap * equity:
            margin_breaches += 1
        raw_quantity = target / price
        if apply_order_filters:
            roundable = np.isfinite(steps) & (steps > 0)
            safe_step = np.where(roundable, steps, 1.0)
            rounded = np.trunc(raw_quantity / safe_step) * safe_step
            intended_delta = np.abs(target - held)
            full_close = (target == 0.0) & (quantities != 0.0)
            enforceable = np.isfinite(mins) & (mins > 0)
            skip = (~full_close) & enforceable & (intended_delta > 0.0) & (intended_delta < mins)
            skipped_orders += int(skip.sum())
            skipped_notional += float(np.where(skip, intended_delta, 0.0).sum())
            intended_notional += float(intended_delta.sum())
            rounded = np.where(skip, quantities, rounded)
            delta = rounded - quantities
            sent = np.abs(delta) * price
        else:
            delta = raw_quantity - quantities
            sent = np.abs(delta) * price
            intended_notional += float(np.abs(target - held).sum())
        fee = fee_rate * float(sent.sum())
        fill_price = price
        maker_fill = np.zeros(count, dtype=bool)
        if is_maker:
            window_start = anchor + 1
            window_limit = int(anchor_pos[day + 1]) if day + 1 < days else n_bars
            window_end = min(window_start + window_bars, window_limit)
            fill_price = price.copy()
            if window_end > window_start:
                window_close = close_f32[window_start:window_end].astype(np.float64)
                window_low = low_f32[window_start:window_end].astype(np.float64)
                window_high = high_f32[window_start:window_end].astype(np.float64)
                anchor_raw = close_f32[anchor].astype(np.float64)
                active = np.abs(delta) > 0.0
                for position in range(count):
                    if not bool(active[position]):
                        continue
                    anchor_level = float(anchor_raw[position])
                    closes = window_close[:, position]
                    finite_closes = closes[np.isfinite(closes)]
                    if not np.isfinite(anchor_level):
                        if finite_closes.size > 0:
                            fill_price[position] = float(finite_closes[-1])
                        continue
                    if delta[position] > 0.0:
                        adverse_col = window_low[:, position]
                        filled = bool(np.any(np.isfinite(adverse_col) & (adverse_col < anchor_level)))
                    else:
                        adverse_col = window_high[:, position]
                        filled = bool(np.any(np.isfinite(adverse_col) & (adverse_col > anchor_level)))
                    if filled:
                        maker_fill[position] = True
                        fill_price[position] = price[position]
                    elif finite_closes.size > 0:
                        fill_price[position] = float(finite_closes[-1])
                maker_notional = np.abs(delta) * fill_price * maker_fill
                taker_notional = np.abs(delta) * fill_price * (~maker_fill)
                fee = maker_fee_rate * float(maker_notional.sum()) + fee_rate * float(taker_notional.sum())
            anchor_sent = np.abs(delta) * price
            total_anchor_sent += float(anchor_sent.sum())
            maker_anchor_sent += float(anchor_sent[maker_fill].sum())
        # ADV·σ가 없는 종목은 충격을 추정할 근거가 없으므로 0으로 두고, NaN이 현금을 오염시키지 않게 한다.
        valid_impact = np.isfinite(adv_values[day]) & (adv_values[day] > 0) & np.isfinite(sigma_values[day])
        safe_adv = np.where(valid_impact, adv_values[day], np.inf)
        safe_sigma = np.where(valid_impact, sigma_values[day], 0.0)
        impact = float((sent * policy.impact_y * safe_sigma * np.sqrt(sent / safe_adv)).sum())
        cash -= float((delta * fill_price).sum()) + fee + impact
        fee_paid += fee
        impact_paid += impact
        quantities = quantities + delta
        daily_equity[day] = cash + float((quantities * price).sum())
        daily_exposure[day] = exposure
        stop = int(anchor_pos[day + 1]) if day + 1 < days else n_bars
        seg_close = np.nan_to_num(close_f32[anchor:stop].astype(np.float64), nan=1.0)
        seg_high = np.nan_to_num(high_f32[anchor:stop].astype(np.float64), nan=1.0)
        seg_low = np.nan_to_num(low_f32[anchor:stop].astype(np.float64), nan=1.0)
        adverse = np.where(quantities > 0, seg_low, seg_high)
        adverse = np.where(np.isfinite(adverse), adverse, seg_close)
        adverse_equity = cash + (quantities[None, :] * adverse).sum(axis=1)
        adverse_notional = np.abs(quantities[None, :] * adverse)
        required = np.zeros(adverse_equity.shape[0])
        for position, ladder in enumerate(ladders):
            tier = np.searchsorted(ladder.floors, adverse_notional[:, position], side="right") - 1
            tier = np.clip(tier, 0, ladder.floors.shape[0] - 1)
            required += np.maximum(adverse_notional[:, position] * ladder.ratios[tier] - ladder.amounts[tier], 0.0)
        hits = np.flatnonzero(adverse_equity < required)
        if hits.shape[0] > 0:
            liquidated_at = pd.Timestamp(bar_index[anchor + int(hits[0])])
            cash = 0.0
            quantities = np.zeros(count)
            daily_equity[day:] = 0.0
            daily_exposure[day:] = 0.0
            max_drawdown = 1.0
            break
        seg_equity = cash + (quantities[None, :] * seg_close).sum(axis=1)
        peaks = np.maximum(running_peak, np.maximum.accumulate(seg_equity))
        with np.errstate(divide="ignore", invalid="ignore"):
            seg_dd = np.where(peaks > 0, 1.0 - seg_equity / peaks, 0.0)
        seg_dd = np.maximum(seg_dd, 0.0)
        if seg_dd.size:
            max_drawdown = max(max_drawdown, float(seg_dd.max()))
            running_peak = float(peaks[-1])
    untraded = skipped_notional / intended_notional if intended_notional > 0 else 0.0
    maker_fill_fraction = maker_anchor_sent / total_anchor_sent if total_anchor_sent > 0 else 0.0
    return AccountLedgerResult(
        capital=float(capital),
        daily_equity=pd.Series(daily_equity, index=dates),
        daily_exposure=pd.Series(daily_exposure, index=dates),
        liquidated_at=liquidated_at,
        skipped_orders=skipped_orders,
        untraded_fraction=float(untraded),
        initial_margin_breaches=margin_breaches,
        fee_paid=fee_paid,
        impact_paid=impact_paid,
        funding_paid=funding_paid,
        fallback_ladder_symbols=fallback_symbols,
        missing_filter_symbols=tuple(sorted(missing_filters)),
        intraday_max_drawdown=float(max_drawdown),
        maker_fill_fraction=float(maker_fill_fraction),
    )
