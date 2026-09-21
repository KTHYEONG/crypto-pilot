"""Account-scale exposure policy under venue margin ladders and size-dependent impact."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import VenueRuleSnapshot


@dataclass(frozen=True, slots=True)
class VenueLadder:
    """Per-symbol margin ladder as parallel numpy arrays over notional tiers."""

    floors: np.ndarray
    ratios: np.ndarray
    amounts: np.ndarray
    leverages: np.ndarray


@dataclass(frozen=True, slots=True)
class ExposurePolicy:
    """Daily exposure rule. ``fixed`` applies ``exposure_max`` unchanged (diagnostic only);
    ``growth`` maximizes haircut expected log growth net of impact inside the margin cap."""

    kind: Literal["fixed", "growth"]
    exposure_max: float
    exposure_step: float
    unit_daily_mean: float
    unit_daily_sigma: float
    mean_haircut: float
    shock_per_unit: float
    margin_reserve: float
    initial_margin_cap: float
    impact_y: float


def build_venue_ladders(
    symbols: Sequence[str], rules: VenueRuleSnapshot
) -> tuple[tuple[VenueLadder, ...], tuple[str, ...]]:
    """Build one ladder per symbol; symbols without a ladder use the snapshot's most punitive tier-1 ratio ladder."""
    firsts = [
        (
            entry.brackets[0].maint_margin_ratio,
            entry.brackets[0].maint_amount,
            entry.brackets[0].initial_leverage,
        )
        for entry in rules.symbols.values()
        if entry.brackets
    ]
    if not firsts:
        raise DataIntegrityError("venue snapshot carries no margin ladders")
    ratio, amount, leverage = max(firsts, key=lambda item: item[0])
    fallback = VenueLadder(
        floors=np.array([0.0]),
        ratios=np.array([ratio]),
        amounts=np.array([amount]),
        leverages=np.array([float(leverage)]),
    )
    ladders: list[VenueLadder] = []
    fallback_symbols: list[str] = []
    for symbol in symbols:
        entry = rules.symbols.get(symbol)
        if entry is None or not entry.brackets:
            ladders.append(fallback)
            fallback_symbols.append(symbol)
        else:
            ladders.append(
                VenueLadder(
                    floors=np.array([tier.notional_floor for tier in entry.brackets]),
                    ratios=np.array([tier.maint_margin_ratio for tier in entry.brackets]),
                    amounts=np.array([tier.maint_amount for tier in entry.brackets]),
                    leverages=np.array([float(tier.initial_leverage) for tier in entry.brackets]),
                )
            )
    return tuple(ladders), tuple(fallback_symbols)


def maintenance_and_initial_margin(
    notional_abs: np.ndarray, ladders: Sequence[VenueLadder],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-symbol maintenance margin (notional * ratio - maint_amount, floored at 0) and initial
    margin (notional / tier initial leverage), tier chosen by |notional| on each symbol's ladder."""
    notionals = np.asarray(notional_abs, dtype=float)
    maintenance = np.empty_like(notionals)
    initial = np.empty_like(notionals)
    for index, ladder in enumerate(ladders):
        tier = int(np.searchsorted(ladder.floors, notionals[index], side="right")) - 1
        tier = min(max(tier, 0), ladder.floors.shape[0] - 1)
        maintenance[index] = max(notionals[index] * ladder.ratios[tier] - ladder.amounts[tier], 0.0)
        initial[index] = notionals[index] / ladder.leverages[tier]
    return maintenance, initial


def _policy_grid(policy: ExposurePolicy) -> np.ndarray:
    count = int(policy.exposure_max / policy.exposure_step)
    return policy.exposure_step * np.arange(1, count + 1)


def margin_exposure_cap(
    weights: np.ndarray, equity: float, ladders: Sequence[VenueLadder], policy: ExposurePolicy,
) -> float:
    """Largest exposure on the policy grid (<= exposure_max) whose maintenance margin plus a
    ``shock_per_unit`` loss per unit of exposure leaves ``margin_reserve`` of equity, and whose
    initial margin fits within ``initial_margin_cap`` of equity; 0.0 when none does.

    Margin tiers are dollar-denominated, so this cap falls as equity compounds; it is the
    mechanism that removes the self-inflicted liquidation of fixed exposure."""
    if not equity > 0:
        return 0.0
    gross = np.abs(np.asarray(weights, dtype=float))
    for rung in _policy_grid(policy)[::-1]:
        notionals = rung * equity * gross
        maintenance, initial = maintenance_and_initial_margin(notionals, ladders)
        if maintenance.sum() + policy.shock_per_unit * rung * equity <= equity * (
            1.0 - policy.margin_reserve
        ) and initial.sum() <= policy.initial_margin_cap * equity:
            return float(rung)
    return 0.0


def choose_exposure(
    weights: np.ndarray, equity: float, held_notional: np.ndarray, adv: np.ndarray,
    daily_sigma: np.ndarray, ladders: Sequence[VenueLadder], policy: ExposurePolicy,
) -> float:
    """Exposure for today's entry using only information available at the entry.

    ``growth``: among grid rungs up to ``margin_exposure_cap``, maximize
    mu*L - 0.5*(sigma*L)**2 - impact(L)/E, with mu = unit_daily_mean*(1 - mean_haircut),
    sigma = unit_daily_sigma and impact(L) = sum |dnotional_i| * impact_y * daily_sigma_i *
    sqrt(|dnotional_i| / adv_i) for the orders this rung would send from ``held_notional``.
    ``fixed``: ``exposure_max``. Ties resolve to the smaller rung."""
    if policy.kind == "fixed":
        return float(policy.exposure_max)
    if not equity > 0:
        return 0.0
    cap = margin_exposure_cap(weights, equity, ladders, policy)
    rungs = _policy_grid(policy)
    rungs = rungs[rungs <= cap + 1e-9]
    unit = np.asarray(weights, dtype=float)
    held = np.asarray(held_notional, dtype=float)
    adv_values = np.asarray(adv, dtype=float)
    sigma_values = np.asarray(daily_sigma, dtype=float)
    mu = policy.unit_daily_mean * (1.0 - policy.mean_haircut)
    base = mu * rungs - 0.5 * (policy.unit_daily_sigma * rungs) ** 2
    deltas = np.abs(rungs[:, None] * equity * unit[None, :] - held[None, :])
    # unit_weights carries every symbol ever held across the window; a symbol not yet (or no
    # longer) eligible has weight 0 and thus delta 0 here, but its ADV/sigma can be NaN (no
    # trading history at this date). 0.0 * NaN is NaN in IEEE754, so both inputs need a finite
    # guard or a zero-impact rung would otherwise be scored NaN and silently lose the argmax.
    safe_adv = np.where(np.isfinite(adv_values) & (adv_values > 0), adv_values, np.inf)
    safe_sigma = np.where(np.isfinite(sigma_values), sigma_values, 0.0)
    impact = (deltas * policy.impact_y * safe_sigma[None, :] * np.sqrt(deltas / safe_adv[None, :])).sum(axis=1)
    scores = base - impact / equity
    return float(rungs[int(np.argmax(scores))]) if rungs.shape[0] > 0 else 0.0
