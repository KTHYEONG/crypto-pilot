"""Committee combination and wealth-objective measurement primitives.

The wealth committee combines selected economic-family signals with long-only equal-risk
weights scaled to an annualized volatility target using train-window statistics only.
This module provides cost decomposition, wealth metrics, volatility scaling, and walk-forward harnesses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from src.mhs.params import (
    COMMITTEE_GROWTH_BARS_PER_YEAR,
    COMMITTEE_GROWTH_HORIZON_YEARS,
    COMMITTEE_GROWTH_MAX_DRAWDOWN,
    COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
    COMMITTEE_GROWTH_MAX_RUIN_PROB,
    COMMITTEE_GROWTH_N_PATHS,
    COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS,
    COMMITTEE_GROWTH_RUIN_FRACTION,
    PNL_TARGET_ANNUAL_VOL,
    GrowthRiskEnvelope,
)
from src.mhs.params import (
    PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H,
)
from src.mhs.types import COMMITTEE_TARGET_VOL


def decompose_cost(
    net_low: pd.DataFrame,
    net_high: pd.DataFrame,
    bps_low: float,
    bps_high: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover ``(gross, turnover_cost)`` from two net-of-cost PnL panels.

    At two cost tiers the per-bar net PnL is ``gross - turnover_cost * bps``
    (both terms linear in the weight magnitude -- cost is a drag for every
    sign), so ``tc = (net_low - net_high) / (bps_high - bps_low)`` and
    ``gross = net_low + tc * bps_low``. ``tc`` is clipped at 0.0 because a cost
    can never be negative. This is the sign-safe accounting the spec's §0
    requires: without it a negative combiner weight turns cost drag into
    profit. Raises ``ValueError`` when ``bps_high <= bps_low``, when either bps
    is negative, or when the two panels are not identically indexed and
    columned.
    """
    if bps_low < 0 or bps_high < 0:
        raise ValueError(f"bps must be non-negative, got {bps_low}, {bps_high}")
    if bps_high <= bps_low:
        raise ValueError(f"bps_high must be > bps_low, got {bps_high} <= {bps_low}")
    if not net_low.index.equals(net_high.index) or list(net_low.columns) != list(
        net_high.columns
    ):
        raise ValueError("net_low and net_high must be identically indexed and columned")
    tc = (net_low - net_high) / (bps_high - bps_low)
    tc = tc.clip(lower=0.0)
    gross = net_low + tc * bps_low
    return gross, tc


def score_weighted_net(
    weights: pd.Series,
    gross: pd.DataFrame,
    turnover_cost: pd.DataFrame,
    cost_bps: float,
) -> pd.Series:
    """Per-bar net PnL of a weighted committee.

    ``(gross * weights).sum(axis=1) - (turnover_cost * weights.abs()).sum(axis=1)
    * cost_bps`` -- the absolute value of every weight makes cost a drag for
    BOTH signs, the regression guarding the v1 defect where inverting a
    strategy refunded its cost. Raises ``ValueError`` on a weights/gross column
    mismatch, on non-identical ``gross``/``turnover_cost`` index or columns, or
    on ``cost_bps < 0``.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if not gross.index.equals(turnover_cost.index) or list(gross.columns) != list(
        turnover_cost.columns
    ):
        raise ValueError("gross and turnover_cost must be identically indexed and columned")
    if list(weights.index) != list(gross.columns):
        raise ValueError("weights index must match gross columns")
    w = weights.reindex(gross.columns)
    gross_part = gross.multiply(w, axis=1).sum(axis=1)
    cost_part = turnover_cost.multiply(w.abs(), axis=1).sum(axis=1) * cost_bps
    return gross_part - cost_part


def train_evidence_weights(
    member_proxy_returns: Mapping[str, pd.Series],
    train_mask: pd.Series,
    min_train_rows: int = 30,
) -> dict[str, float]:
    """Train-only P&L-aligned evidence weights for committee members.

    Skill-aware counterpart to ``long_only_equal_risk_weights`` (equal-RISK,
    rejected in RC-4): weights each member by its realized proxy-return
    t-statistic computed exclusively on train-window rows, never on rank IC
    (rank ordering and dollar P&L disagree under fat-tailed cross-sectional
    crypto returns).  Weights are non-negative and sum to 1 so a convex
    combination of dollar-neutral member books stays dollar-neutral with
    gross <= 1.0.  Fails closed to exact equal weights when no member has
    positive train evidence, reproducing today's behaviour.
    """
    if not member_proxy_returns:
        raise ValueError("member_proxy_returns must not be empty")
    if min_train_rows < 1:
        raise ValueError(f"min_train_rows must be >= 1, got {min_train_rows}")

    raw: dict[str, float] = {}
    for name, series in member_proxy_returns.items():
        train_series = series[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
        if len(train_series) < min_train_rows:
            raw[name] = 0.0
            continue
        std = float(train_series.std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            raw[name] = 0.0
            continue
        t = float(train_series.mean()) / (std / np.sqrt(float(len(train_series))))
        raw[name] = max(0.0, t)

    total = sum(raw.values())
    if total <= 0.0 or len(member_proxy_returns) < 2:
        n = len(member_proxy_returns)
        return dict.fromkeys(member_proxy_returns, 1.0 / n)
    return {name: raw[name] / total for name in member_proxy_returns}


def long_only_equal_risk_weights(train_net: pd.DataFrame) -> pd.Series:
    """Inverse-volatility committee weights from a TRAIN-window net panel.

    Weights are inverse-vol, normalized to sum to 1.0, and every weight is
    non-negative. Long-only is mandatory: the measured long-short variants
    score -0.209 (Sharpe-weighted) and -0.568 (shrinkage MV, -1.813 at the
    stress tier) because shorting a strategy book still pays its turnover cost.
    A column with zero or non-finite training volatility receives weight 0.0
    rather than raising. Raises ``ValueError`` on an empty panel or when every
    column is degenerate.
    """
    if train_net.empty:
        raise ValueError("train_net must not be empty")
    vol = train_net.std(ddof=1)
    inv = 1.0 / vol
    inv = inv.where(np.isfinite(vol) & (vol > 0), 0.0)
    total = float(inv.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("train_net must have at least one column with positive finite volatility")
    return inv / total


def wealth_metrics(
    returns: pd.Series,
    periods_per_year: float = _PERIODS_PER_YEAR_1H,
) -> dict[str, float]:
    """Compounded-growth metrics for the wealth objective.

    ``cagr`` from ``(1 + r).cumprod()``, ``mdd`` as the minimum of
    ``equity / equity.cummax() - 1.0``, ``logret`` as ``sum(log1p(r))`` with
    ``r`` floored at -0.99 so the log stays finite, and the annualized
    ``sharpe``. An empty or all-NaN series yields ``nan`` values rather than
    raising; a degenerate variance yields ``nan`` Sharpe. Raises ``ValueError``
    on ``periods_per_year <= 0``.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    nan: float = float("nan")
    r = returns.dropna()
    if r.empty:
        return {"cagr": nan, "mdd": nan, "logret": nan, "sharpe": nan}
    equity = (1.0 + r).cumprod()
    cagr = float(equity.iloc[-1] ** (periods_per_year / len(r)) - 1.0)
    mdd = float((equity / equity.cummax() - 1.0).min())
    logret = float(np.log1p(r.clip(lower=-0.99)).sum())
    sd = float(r.std(ddof=1))
    sharpe = (
        nan
        if not np.isfinite(sd) or sd <= 0
        else float(r.mean() / sd * np.sqrt(periods_per_year))
    )
    return {"cagr": cagr, "mdd": mdd, "logret": logret, "sharpe": sharpe}


def volatility_target_scale(
    train_returns: pd.Series,
    target_vol: float,
    periods_per_year: float = _PERIODS_PER_YEAR_1H,
) -> float:
    """Scalar making the TRAIN window's annualized realized vol equal to ``target_vol``.

    ``target_vol / (train_returns.std(ddof=1) * sqrt(periods_per_year))`` --
    computed from the supplied train series only, never from test data. Returns
    exactly ``0.0`` when the training volatility is zero or non-finite (fail
    closed to no exposure, never inf). Raises ``ValueError`` on
    ``target_vol <= 0`` or ``periods_per_year <= 0``.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    r = train_returns.dropna()
    sd = float(r.std(ddof=1)) if len(r) >= 2 else float("nan")
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    return float(target_vol / (sd * np.sqrt(periods_per_year)))


def kelly_lcb_scale(
    train_returns: pd.Series,
    fraction: float = 0.25,
    z: float = 1.0,
    cap: float = 1.5,
) -> float:
    """Fractional-Kelly total-exposure scale with a one-SE lower-confidence-bound shrinkage.

    The train-window optimal Kelly leverage is ``mean / variance`` (per-period
    native units, never annualized). To shrink the estimate, ``fraction`` (a
    quarter-Kelly default) is applied to the LCB of the mean --
    ``mean - z * std / sqrt(n)`` -- rather than to the raw mean; when that LCB
    is <= 0 the estimate cannot justify exposure and the result is ``0.0``,
    the same fail-closed discipline ``volatility_target_scale`` uses. The
    final value is clipped to ``[0.0, cap]``. Returns exactly ``0.0`` when the
    train series has fewer than 2 observations or a zero/non-finite std. Raises
    ``ValueError`` on ``fraction <= 0``, ``cap <= 0``, or ``z < 0``.
    """
    if fraction <= 0:
        raise ValueError(f"fraction must be > 0, got {fraction}")
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z}")
    r = train_returns.dropna()
    if len(r) < 2:
        return 0.0
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        return 0.0
    variance = sd * sd
    # A constant series leaves only float round-off in the std (e.g. repeated
    # 0.001 yields sd ~ 2e-19); such a variance cannot justify any leverage.
    # Fail closed when it is indistinguishable from the mean's float noise.
    if variance <= np.finfo(np.float64).eps * float(mean * mean):
        return 0.0
    lcb_mean = mean - z * sd / np.sqrt(float(len(r)))
    if lcb_mean <= 0:
        return 0.0
    return float(np.clip(fraction * lcb_mean / variance, 0.0, cap))


def committee_block_edges_from(
    start: pd.Timestamp,
    oos_start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """6-month OOS block starts covering [max(start, oos_start), end).

    Anchored at ``max(start, oos_start)`` instead of the raw diagnostic start so
    a purged walk-forward cannot evaluate pre-OOS blocks as pseudo-OOS via
    ``min_train_bars`` alone. Raises ``ValueError`` when ``end <= max(start, oos_start)``.
    """
    anchored = max(start, oos_start)
    if end <= anchored:
        raise ValueError(
            f"end {end} must be after max(start, oos_start) = {anchored}"
        )
    edges: list[pd.Timestamp] = []
    cursor = anchored
    while cursor < end:
        edges.append(cursor)
        cursor = cursor + pd.DateOffset(months=6)
    return edges


def purged_walk_forward(
    gross: pd.DataFrame,
    turnover_cost: pd.DataFrame,
    cost_bps: float,
    block_edges: Sequence[pd.Timestamp],
    purge: pd.Timedelta,
    target_vol: float = COMMITTEE_TARGET_VOL,
    min_train_bars: int = 2000,
    periods_per_year: float = _PERIODS_PER_YEAR_1H,
    sizing_mode: Literal["vol_target", "kelly_blend"] = "vol_target",
    kelly_fraction: float = 0.25,
    kelly_z: float = 1.0,
    kelly_cap: float = 1.5,
) -> pd.Series:
    """Expanding-train purged walk-forward net PnL of the committee.

    For each test block starting at ``t0`` the weights come from
    ``long_only_equal_risk_weights`` fitted on the net panel restricted to bars
    strictly before ``t0 - purge``, and the volatility scale comes from
    ``volatility_target_scale`` on the train-window weighted combination -- so
    no test-window statistic ever feeds a weight or a scale. With
    ``sizing_mode='kelly_blend'`` the block scale is instead the 50/50 average
    of the vol-target scale and ``kelly_lcb_scale`` on the same train
    combination (an opt-in quarter-Kelly LCB overlay; the default
    ``sizing_mode='vol_target'`` keeps every pre-existing call byte-identical).
    Blocks whose train window has fewer than ``min_train_bars`` bars, and
    blocks with no test bars, are skipped rather than raising. The returned
    series covers only test-block timestamps, is sorted, and has no duplicate
    index entries. Raises ``ValueError`` on empty or non-monotonic
    ``block_edges``, a non-positive ``purge``, ``cost_bps < 0``, a
    ``gross``/``turnover_cost`` shape mismatch, or an unknown ``sizing_mode``.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if purge <= pd.Timedelta(0):
        raise ValueError(f"purge must be positive, got {purge}")
    if sizing_mode not in ("vol_target", "kelly_blend"):
        raise ValueError(f"unknown sizing_mode {sizing_mode!r}")
    if not block_edges:
        raise ValueError("block_edges must not be empty")
    if not gross.index.equals(turnover_cost.index) or list(gross.columns) != list(
        turnover_cost.columns
    ):
        raise ValueError("gross and turnover_cost must be identically indexed and columned")
    edges = [pd.Timestamp(e) for e in block_edges]
    for i in range(1, len(edges)):
        if not edges[i - 1] < edges[i]:
            raise ValueError("block_edges must be strictly ascending")

    net = gross - turnover_cost * cost_bps
    blocks: list[pd.Series] = []
    for i, t0 in enumerate(edges):
        next_edge = edges[i + 1] if i + 1 < len(edges) else gross.index[-1] + pd.Timedelta(hours=1)
        train_rows = gross.index < (t0 - purge)
        if int(train_rows.sum()) < min_train_bars:
            continue
        test_rows = (gross.index >= t0) & (gross.index < next_edge)
        if not bool(test_rows.any()):
            continue
        weights = long_only_equal_risk_weights(net.loc[train_rows])
        combined_train = score_weighted_net(
            weights, gross.loc[train_rows], turnover_cost.loc[train_rows], cost_bps,
        )
        scale_vol = volatility_target_scale(combined_train, target_vol, periods_per_year)
        scale = (
            scale_vol
            if sizing_mode == "vol_target"
            else 0.5 * scale_vol + 0.5 * kelly_lcb_scale(
                combined_train, kelly_fraction, kelly_z, kelly_cap,
            )
        )
        combined_test = score_weighted_net(
            weights, gross.loc[test_rows], turnover_cost.loc[test_rows], cost_bps,
        )
        blocks.append((combined_test * scale).astype(float))
    if not blocks:
        return pd.Series(dtype=float)
    result = pd.concat(blocks)
    return result.sort_index()


def growth_budget_annual_vol(
    train_returns: pd.Series,
    *,
    envelope: GrowthRiskEnvelope | None = None,
    bars_per_year: float = 365.0,
    floor: float = 0.05,
    cap: float = 1.0,
    fallback: float = PNL_TARGET_ANNUAL_VOL,
) -> float:
    """Annualized target volatility derived from the registered drawdown budget.

    Builds the risk grid as ``reference_risk * COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS``
    where ``reference_risk = train_returns.std(ddof=1)``; calls
    ``solve_growth_optimal_risk`` with ``use_drawdown_overlay=False``; returns
    ``clip(selected_risk * sqrt(bars_per_year), floor, cap)``. On infeasible or
    degenerate input returns ``fallback`` (PNL_TARGET_ANNUAL_VOL). The
    ``use_drawdown_overlay=False`` is mandatory: with the overlay on every grid
    point reports ``mdd_breach_prob=0.000`` and the solver selects a risk the live
    path cannot honour.
    """
    from src.mhs.params import GrowthRiskEnvelope
    from src.quant.risk.growth_sizing import GrowthSizingConfig, solve_growth_optimal_risk

    if envelope is None:
        envelope = GrowthRiskEnvelope(
            name="conservative",
            max_drawdown=COMMITTEE_GROWTH_MAX_DRAWDOWN,
            max_drawdown_prob=COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
            ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
            max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
            horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
            leverage_ceiling=1.0,
        )

    r = train_returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return fallback
    reference_risk = float(r.std(ddof=1))
    if not np.isfinite(reference_risk) or reference_risk <= 0:
        return fallback
    risk_grid = tuple(sorted(reference_risk * m for m in COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS))
    config = GrowthSizingConfig(
        risk_grid=risk_grid,
        reference_risk=reference_risk,
        max_drawdown=envelope.max_drawdown,
        max_drawdown_prob=envelope.max_drawdown_prob,
        ruin_fraction=envelope.ruin_fraction,
        max_ruin_prob=envelope.max_ruin_prob,
        horizon_years=envelope.horizon_years,
        n_paths=COMMITTEE_GROWTH_N_PATHS,
        bars_per_year=COMMITTEE_GROWTH_BARS_PER_YEAR,
    )
    result = solve_growth_optimal_risk(r.to_numpy(), config, use_drawdown_overlay=False)
    if result.selected_risk is None:
        return fallback
    annual_vol = float(result.selected_risk * np.sqrt(bars_per_year))
    return float(np.clip(annual_vol, floor, cap))
