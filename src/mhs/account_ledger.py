"""Account-scale replay of a unit-exposure target book as one real USDT account."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import VenueRuleSnapshot
from src.mhs.account_policy import (
    ExposurePolicy,
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
    """Account-scale replay outcome in USDT; daily series indexed by UTC date."""

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


def replay_account(
    unit_weights: pd.DataFrame,
    marks: AccountMarkPanels,
    funding_cum: pd.DataFrame,
    adv: pd.DataFrame,
    daily_sigma: pd.DataFrame,
    rules: VenueRuleSnapshot,
    policy: ExposurePolicy,
    *,
    capital: float,
    taker_fee_bps: float,
    apply_order_filters: bool = True,
) -> AccountLedgerResult:
    """Replay a unit-exposure target book as one real USDT account.

    At each daily entry the policy picks exposure from current equity; target quantities are
    rounded to the symbol's step size, orders below the symbol's minimum notional are not sent
    (full closes are reduce-only and always sent), fees and square-root impact are charged on
    sent notional, and funding accrues on held notional. Between entries quantities are fixed
    and marked on 3m closes; maintenance margin is evaluated on each bar's adverse low/high, and
    the first bar whose adverse equity falls below total maintenance margin liquidates the account
    (equity 0, replay stops).

    Raises:
        DataIntegrityError: Misaligned indexes/columns, non-positive capital, or no entry row.
    """
    if not np.isfinite(capital) or capital <= 0:
        raise DataIntegrityError(f"capital must be positive, got {capital}")
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
    # unit_weights carries every symbol EVER held across the full window, but a symbol not yet
    # listed (or already delisted) has no price for the bars outside its life -- forward-fill in
    # _read_held_marks only covers gaps AFTER a symbol's first bar, so leading/trailing NaN marks
    # remain. Weight there is always exactly 0.0 (the PIT roster gates it), but IEEE754 makes
    # 0.0 * NaN = NaN, which would otherwise poison quantities and equity from day 0 onward.
    # Replacing non-finite marks with a neutral finite placeholder keeps every such symbol's
    # contribution exactly zero without ever being economically load-bearing.
    close_values = np.nan_to_num(marks.close[symbols].to_numpy(dtype="float64"), nan=1.0)
    high_values = np.nan_to_num(marks.high[symbols].to_numpy(dtype="float64"), nan=1.0)
    low_values = np.nan_to_num(marks.low[symbols].to_numpy(dtype="float64"), nan=1.0)
    entry_pos = bar_index.searchsorted(dates, side="right") - 1
    if bool((entry_pos < 0).any()):
        raise DataIntegrityError("entry date precedes mark panels")
    entry_close = close_values[entry_pos]
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
    for day in range(days):
        price = entry_close[day]
        held = quantities * price
        marked = cash + float(held.sum())
        charge = float((held * funding_step[day]).sum())
        cash -= charge
        funding_paid += charge
        equity = marked - charge
        exposure = choose_exposure(
            weight_values[day], equity, held, adv_values[day], sigma_values[day], ladders, policy
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
        # ADV·σ가 없는 종목은 충격을 추정할 근거가 없으므로 0으로 두고, NaN이 현금을 오염시키지 않게 한다.
        valid_impact = np.isfinite(adv_values[day]) & (adv_values[day] > 0) & np.isfinite(sigma_values[day])
        safe_adv = np.where(valid_impact, adv_values[day], np.inf)
        safe_sigma = np.where(valid_impact, sigma_values[day], 0.0)
        impact = float((sent * policy.impact_y * safe_sigma * np.sqrt(sent / safe_adv)).sum())
        cash -= float((delta * price).sum()) + fee + impact
        fee_paid += fee
        impact_paid += impact
        quantities = quantities + delta
        daily_equity[day] = cash + float((quantities * price).sum())
        daily_exposure[day] = exposure
        stop = int(entry_pos[day + 1]) if day + 1 < days else close_values.shape[0]
        segment = slice(int(entry_pos[day]), stop)
        adverse = np.where(quantities > 0, low_values[segment], high_values[segment])
        adverse = np.where(np.isfinite(adverse), adverse, close_values[segment])
        adverse_equity = cash + (quantities[None, :] * adverse).sum(axis=1)
        adverse_notional = np.abs(quantities[None, :] * adverse)
        required = np.zeros(adverse_equity.shape[0])
        for position, ladder in enumerate(ladders):
            tier = np.searchsorted(ladder.floors, adverse_notional[:, position], side="right") - 1
            tier = np.clip(tier, 0, ladder.floors.shape[0] - 1)
            required += np.maximum(adverse_notional[:, position] * ladder.ratios[tier] - ladder.amounts[tier], 0.0)
        hits = np.flatnonzero(adverse_equity < required)
        if hits.shape[0] > 0:
            liquidated_at = pd.Timestamp(bar_index[int(entry_pos[day]) + int(hits[0])])
            cash = 0.0
            quantities = np.zeros(count)
            daily_equity[day:] = 0.0
            daily_exposure[day:] = 0.0
            break
    untraded = skipped_notional / intended_notional if intended_notional > 0 else 0.0
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
    )
