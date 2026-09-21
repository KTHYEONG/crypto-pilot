"""Account-scale exposure policy under venue margin ladders and size-dependent impact."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import VenueRuleSnapshot
from src.mhs.params import (
    ACCOUNT_EXPOSURE_MAX,
    ACCOUNT_EXPOSURE_STEP,
    ACCOUNT_IMPACT_Y,
    ACCOUNT_INITIAL_MARGIN_CAP,
    ACCOUNT_MARGIN_RESERVE,
    ACCOUNT_MEAN_HAIRCUT,
    ACCOUNT_MIN_MOMENT_DAYS,
    ACCOUNT_PRIOR_DAYS,
    ACCOUNT_SHOCK_PER_UNIT,
)


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
    ``growth`` maximizes fractional-Kelly expected log growth on causally updated posterior
    unit-book moments, net of impact, inside the margin cap. No field carries a moment
    estimated on the evaluation window."""

    kind: Literal["fixed", "growth"]
    exposure_max: float
    exposure_step: float
    mean_haircut: float
    prior_days: float
    min_moment_days: int
    shock_per_unit: float
    margin_reserve: float
    initial_margin_cap: float
    impact_y: float


@dataclass(frozen=True, slots=True)
class UnitMoments:
    """Posterior daily moments of the unit-exposure book from returns observed strictly
    before the current entry."""

    mean: float
    sigma: float
    observations: int


def account_growth_policy(*, impact_y: float = ACCOUNT_IMPACT_Y) -> ExposurePolicy:
    """Registered growth exposure policy shared by the account backtest and the live frozen step."""
    return ExposurePolicy(
        kind="growth",
        exposure_max=ACCOUNT_EXPOSURE_MAX,
        exposure_step=ACCOUNT_EXPOSURE_STEP,
        mean_haircut=ACCOUNT_MEAN_HAIRCUT,
        prior_days=ACCOUNT_PRIOR_DAYS,
        min_moment_days=ACCOUNT_MIN_MOMENT_DAYS,
        shock_per_unit=ACCOUNT_SHOCK_PER_UNIT,
        margin_reserve=ACCOUNT_MARGIN_RESERVE,
        initial_margin_cap=ACCOUNT_INITIAL_MARGIN_CAP,
        impact_y=impact_y,
    )


def bayesian_unit_moments(
    observations: int, total: float, total_sq: float, *, prior_days: float, min_moment_days: int,
) -> UnitMoments | None:
    """Posterior unit-book moments under a zero-edge prior, from sufficient statistics.

    The prior mean is 0 with the weight of ``prior_days`` observations, so the posterior
    mean shrinks the sample mean by n / (n + prior_days); the posterior sigma inflates the
    sample (population) standard deviation by sqrt(1 + 1 / (n + prior_days)) for mean
    uncertainty. Sufficient statistics keep the daily update O(1).

    Args:
        observations: Count n of unit daily returns observed before the entry.
        total: Sum of those returns.
        total_sq: Sum of their squares.
        prior_days: Prior weight in days; must be positive.
        min_moment_days: Minimum n for a sample variance to be used.

    Returns:
        Posterior moments, or None when n < min_moment_days or the sample variance is not
        strictly positive and finite.

    Raises:
        DataIntegrityError: prior_days not positive/finite, min_moment_days < 2, or
            non-finite total/total_sq.
    """
    if not math.isfinite(prior_days) or not prior_days > 0:
        raise DataIntegrityError(f"prior_days must be positive finite, got {prior_days}")
    if min_moment_days < 2:
        raise DataIntegrityError(f"min_moment_days must be >= 2, got {min_moment_days}")
    if not math.isfinite(total) or not math.isfinite(total_sq):
        raise DataIntegrityError(f"total/total_sq must be finite, got {total} {total_sq}")
    n = observations
    if n < min_moment_days:
        return None
    sample_mean = total / n
    variance = total_sq / n - sample_mean * sample_mean
    if not math.isfinite(variance) or not variance > 0:
        return None
    mean = total / (n + prior_days)
    sigma = math.sqrt(variance * (1.0 + 1.0 / (n + prior_days)))
    return UnitMoments(mean=mean, sigma=sigma, observations=n)


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
    moments: UnitMoments | None,
) -> float:
    """Exposure for today's entry using only information available at the entry.

    ``growth``: among grid rungs up to ``margin_exposure_cap``, maximize
    mu*L - 0.5*(sigma*L)**2 - impact(L)/E with mu = moments.mean*(1 - mean_haircut) and
    sigma = moments.sigma (posterior, causal); impact as before. With ``moments`` None
    (too little evidence) the smallest grid rung within the cap is used, never a larger
    one. ``fixed``: ``exposure_max``; ``moments`` is ignored. Ties resolve to the smaller rung."""
    if policy.kind == "fixed":
        return float(policy.exposure_max)
    if not equity > 0:
        return 0.0
    cap = margin_exposure_cap(weights, equity, ladders, policy)
    rungs = _policy_grid(policy)
    rungs = rungs[rungs <= cap + 1e-9]
    if rungs.shape[0] == 0:
        return 0.0
    if moments is None:
        return float(rungs[0])
    unit = np.asarray(weights, dtype=float)
    held = np.asarray(held_notional, dtype=float)
    adv_values = np.asarray(adv, dtype=float)
    sigma_values = np.asarray(daily_sigma, dtype=float)
    mu = moments.mean * (1.0 - policy.mean_haircut)
    base = mu * rungs - 0.5 * (moments.sigma * rungs) ** 2
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
