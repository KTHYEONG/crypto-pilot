"""Signal EMA / deadband / regime-cash / pnl-vol-target / kelly / exposure scales (I4 seam)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.common.errors import DataIntegrityError
from src.mhs.horizons import efficiency_ratio
from src.mhs.params import (
    COMMITTEE_OOS_START,
    CONSTANT_RISK_CAP_BINDING_QUANTILE,
    CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
    CONSTANT_RISK_MIN_PERIODS_DAYS,
    CONSTANT_RISK_TARGET_ANNUAL_VOL,
    PNL_TARGET_ANNUAL_VOL,
    PNL_VOL_TARGET_BURN_IN_DAYS,
    PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
    PNL_VOL_TARGET_MAX_SCALE,
    PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS,
    PNL_VOL_TARGET_SCALE_FLOOR,
    PNL_VOL_TARGET_WINDOW_DAYS,
    REBALANCE_DEADBAND_POSITION_FRACTION,
    REGIME_CASH_MEDIAN_WINDOW_HOURS,
    REGIME_CASH_SCALE_FLOOR,
    GrowthRiskEnvelope,
)
from src.mhs.regime import trend_efficiency_scale


def _smooth_signal_ema(signal: pd.DataFrame, span_steps: int) -> pd.DataFrame:
    """Apply an exponential moving average to a step-grid signal.

    The EMA is the spec's ``Autocorr Smoothing`` (§3.2): it removes the
    high-frequency noise that drives negative return autocorrelation (whipsaw)
    while preserving the trend polarity. ``span_steps`` is one full horizon
    cycle in decision steps; ``adjust=False`` so the span is the constant
    half-life ``span - 1`` and the filtered series is fully causal.
    """
    if span_steps < 1:
        raise ValueError(f"span_steps must be >= 1, got {span_steps}")
    return signal.ewm(span=span_steps, adjust=False).mean()


def _apply_rebalance_deadband(
    target: pd.DataFrame,
    position_fraction: float = REBALANCE_DEADBAND_POSITION_FRACTION,
) -> pd.DataFrame:
    """Suppress per-symbol rebalances smaller than a scale-relative deadband.

    A target-weight change below ``position_fraction * scale_t`` (where
    ``scale_t`` is the per-decision per-symbol position scale) carries the last
    decided (held) target forward instead of retrading, so the executor never
    churns on sub-threshold signal deltas; the hold is stateful, so a slow
    drift cannot creep through one small step at a time. A target of exactly
    ``0.0`` is a liquidation instruction, never a resize to be carried (the
    exit-always invariant). The first observation is always a decision, NaN
    targets remain NaN (a delisting is never silently re-expressed), and a held
    NaN resets the deadband so a re-listed symbol trades from its own first
    finite target.
    """
    if position_fraction < 0:
        raise ValueError(f"position_fraction must be >= 0, got {position_fraction}")
    if target.empty:
        return target.copy()
    values = target.to_numpy(dtype="float64")
    out = values.copy()
    held = out[0].copy()
    finite = np.isfinite(values)
    for i in range(1, len(values)):
        row = values[i]
        active = np.count_nonzero(np.abs(row) > 0.0)
        min_delta = (
            position_fraction * float(np.abs(row[np.isfinite(row)]).sum()) / active
            if active
            else 0.0
        )
        carry = (np.abs(row - held) < min_delta) & finite[i] & np.isfinite(held) & (row != 0.0)
        out[i] = np.where(carry, held, row)
        held = out[i]
    # Invariant H (fail-closed): holdings can never exceed the roster that
    # produced them; a violation is a systemic misconfiguration.
    holdings_in = np.count_nonzero(np.abs(values) > 0.0, axis=1)
    holdings_out = np.count_nonzero(np.abs(out) > 0.0, axis=1)
    for i in range(len(values)):
        if holdings_out[i] > holdings_in[i]:
            raise DataIntegrityError(
                f"holdings boundedness violated at {target.index[i]}: "
                f"holdings_out={int(holdings_out[i])} > holdings_in={int(holdings_in[i])}"
            )
    return pd.DataFrame(out, index=target.index, columns=target.columns).fillna(0.0)


def _trend_efficiency_overlay_scale(
    log_close: pd.DataFrame,
    execution_mask: pd.DataFrame,
    fast_horizon_hours: int,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Execution-roster mean efficiency_ratio at the fast band's horizon."""
    mean_er = efficiency_ratio(log_close, fast_horizon_hours).where(execution_mask).reindex(target_index).mean(axis=1)
    return trend_efficiency_scale(mean_er)


def _regime_cash_scale(
    vol_mean: pd.Series,
    median_window_hours: int = REGIME_CASH_MEDIAN_WINDOW_HOURS,
    floor: float = REGIME_CASH_SCALE_FLOOR,
) -> pd.Series:
    """Per-decision gross-exposure scale that raises cash in high-vol regimes.

    Exposure is ``median(vol) / vol`` clipped to ``[floor, 1.0]``: a calm regime
    keeps full gross, a high-vol regime scales toward the cash floor, and a
    flat/insufficient-history window carries full exposure (never 0/0). This is
    the spec's ``Dynamic Band Weighting`` (§3.2) expressed as cash weighting.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if median_window_hours < 1:
        raise ValueError(f"median_window_hours must be >= 1, got {median_window_hours}")
    if vol_mean.empty:
        return pd.Series(1.0, index=vol_mean.index)
    median = vol_mean.rolling(
        median_window_hours, min_periods=min(48, median_window_hours),
    ).median()
    scale = median.div(vol_mean.clip(lower=1e-12))
    scale = scale.clip(lower=floor, upper=1.0)
    return scale.fillna(1.0)


def _pnl_vol_target_scale(
    reference_daily_returns: pd.Series,
    window_days: int = PNL_VOL_TARGET_WINDOW_DAYS,
    median_window_days: int = PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS,
    floor: float = PNL_VOL_TARGET_SCALE_FLOOR,
) -> pd.Series:
    """Strategy-own-P&L realized-vol targeting scale (Barroso & Santa-Clara).

    ``scale_t = clip(rolling_median(trailing_vol, window=365d)_{t-1} /
    trailing_vol_{t-1}, floor, 1.0)``: the strategy de-risks when its own
    daily P&L becomes more volatile than its recent historical median
    (momentum-crash protection), never levering up and never scaling on an
    under-sampled estimate. Causality is strict: both the trailing-vol window
    AND the rolling-median target are ``shift(1)`` before use (two independent
    shifts, not one combined), so ``scale_t`` depends only on realized returns
    strictly before ``t``.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    if median_window_days < PNL_VOL_TARGET_BURN_IN_DAYS:
        raise ValueError(
            f"median_window_days must be >= PNL_VOL_TARGET_BURN_IN_DAYS "
            f"({PNL_VOL_TARGET_BURN_IN_DAYS}), got {median_window_days}"
        )
    if reference_daily_returns.empty:
        return pd.Series(1.0, index=reference_daily_returns.index)
    trailing_vol = reference_daily_returns.rolling(
        window_days, min_periods=max(5, window_days // 2),
    ).std().shift(1)
    rolling_target = trailing_vol.rolling(
        median_window_days, min_periods=PNL_VOL_TARGET_BURN_IN_DAYS,
    ).median().shift(1)
    scale = rolling_target.div(trailing_vol.where(trailing_vol > 0))
    return scale.clip(lower=floor, upper=1.0).fillna(1.0)


def _committee_kelly_scale(
    reference_daily_returns: pd.Series,
    window_days: int = PNL_VOL_TARGET_WINDOW_DAYS,
    fraction: float = 0.25,
    z: float = 1.0,
    floor: float = PNL_VOL_TARGET_SCALE_FLOOR,
    cap: float = 1.0,
) -> pd.Series:
    """Strategy-own-P&L trailing quarter-Kelly LCB exposure scale.

    ``scale_t = clip(fraction * lcb_mean_{t-1} / var_{t-1}, floor, cap)`` where
    ``lcb_mean = trailing_mean - z * trailing_std / sqrt(n)`` (Wald-style
    lower-confidence-bound mean), mirroring ``_pnl_vol_target_scale``'s
    shift(1)-before-use causality and floor clip exactly. The upper bound is
    supplied by the caller from the resolved growth envelope's
    ``leverage_ceiling`` (via ``resolved_exposure_cap``) and defaults to 1.0
    for byte-identical legacy behaviour -- the former hardcoded 1.0 cap turned
    the 50/50 blend into a pure de-leverager. A weak or negative LCB edge
    shrinks the scale to ``floor``, same as the P&L-vol-target scale's
    momentum-crash de-risking.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    if fraction <= 0:
        raise ValueError(f"fraction must be > 0, got {fraction}")
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z}")
    if cap < 1.0:
        raise ValueError(f"cap must be >= 1.0, got {cap}")
    if reference_daily_returns.empty:
        return pd.Series(1.0, index=reference_daily_returns.index)
    min_periods = max(5, window_days // 2)
    trailing_mean = reference_daily_returns.rolling(window_days, min_periods=min_periods).mean().shift(1)
    trailing_std = reference_daily_returns.rolling(window_days, min_periods=min_periods).std().shift(1)
    trailing_n = reference_daily_returns.rolling(window_days, min_periods=min_periods).count().shift(1)
    se = trailing_std.div(np.sqrt(trailing_n))
    lcb_mean = trailing_mean - z * se
    var = trailing_std.pow(2)
    raw_scale = fraction * lcb_mean.div(var.where(var > 0))
    return raw_scale.clip(lower=floor, upper=cap).fillna(1.0)


def _committee_capital_replay_scale(
    pnl_vol_target_scale: pd.Series,
    reference_daily_returns: pd.Series,
    committee_capital: bool,
    committee_kelly_sizing: bool,
    cap: float = 1.0,
) -> pd.Series:
    """50/50 blend of the P&L-vol-target scale with the committee Kelly-LCB scale.

    Only active when both ``committee_capital`` and ``committee_kelly_sizing``
    are set (opt-in on top of an opt-in); otherwise returns
    ``pnl_vol_target_scale`` unchanged so every other run stays byte-identical.
    Both blended terms must share the SAME ``cap`` (I2): differing caps make
    the blend change the exposure LEVEL rather than its SHAPE.
    """
    if not (committee_capital and committee_kelly_sizing):
        return pnl_vol_target_scale
    kelly_scale = _committee_kelly_scale(reference_daily_returns, cap=cap).reindex(
        pnl_vol_target_scale.index,
    ).fillna(1.0)
    return 0.5 * pnl_vol_target_scale + 0.5 * kelly_scale


def _exante_vol_target_scale(reference_daily_returns: pd.Series, target_vol: float = PNL_TARGET_ANNUAL_VOL, halflife_days: int = PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS, min_days: int = PNL_VOL_TARGET_BURN_IN_DAYS, floor: float = PNL_VOL_TARGET_SCALE_FLOOR, cap: float = 1.0) -> pd.Series:
    """절대 ex-ante 변동성 타겟팅: 목표 변동성 대비 실현 변동성 비율로 스케일링.

    ``sigma_t = ewm(std, halflife=20d).shift(1) * sqrt(365)``
    ``scale_t = clip(target_vol / sigma_t, floor, cap)``

    _pnl_vol_target_scale와 달리 자가 trailing vol의 롤링 중앙값이 아닌
    절대 위험 기준이므로 저위험 연도(2023)에서도 충분한 노출을 유지한다.
    측정: 2023 vol 0.172 -> mean scale 0.991 vs _pnl_vol_target_scale 0.880.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")
    if halflife_days < 1:
        raise ValueError(f"halflife_days must be >= 1, got {halflife_days}")
    if min_days < 1:
        raise ValueError(f"min_days must be >= 1, got {min_days}")
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if cap < 1.0:
        raise ValueError(f"cap must be >= 1.0, got {cap}")
    if reference_daily_returns.empty:
        return pd.Series(1.0, index=reference_daily_returns.index)
    sigma = (
        reference_daily_returns
        .ewm(halflife=halflife_days, min_periods=min_days)
        .std()
        .shift(1)
        * np.sqrt(365.0)
    )
    scale = target_vol / sigma.where(sigma > 0)
    return scale.clip(lower=floor, upper=cap).fillna(1.0)


def _growth_budget_target_vol(
    reference_daily_returns: pd.Series,
    envelope: GrowthRiskEnvelope | None = None,
    oos_start: pd.Timestamp = COMMITTEE_OOS_START,
    *,
    fail_closed: bool = False,
) -> float:
    """Leak-free wrapper: slices to index < oos_start, delegates to growth_budget_annual_vol.

    Returns PNL_TARGET_ANNUAL_VOL when fewer than
    PNL_VOL_TARGET_BURN_IN_DAYS train rows exist. ``fail_closed=True``
    promotes that unresolvable case to ``DataIntegrityError`` instead of
    silently re-resolving the policy to a different target vol.
    """
    from src.mhs.committee import growth_budget_annual_vol

    train = reference_daily_returns.loc[reference_daily_returns.index < oos_start]
    train = train.dropna()
    if len(train) < PNL_VOL_TARGET_BURN_IN_DAYS:
        if fail_closed:
            raise DataIntegrityError(
                f"growth_budget target vol unresolved: {len(train)} finite train "
                f"rows before {oos_start}, require >= {PNL_VOL_TARGET_BURN_IN_DAYS}"
            )
        return PNL_TARGET_ANNUAL_VOL
    return growth_budget_annual_vol(train, envelope=envelope)


def _growth_budget_target_vol_by_boundary(
    reference_daily_returns: pd.Series,
    envelope: GrowthRiskEnvelope,
    train_ends: Mapping[str, pd.Timestamp],
) -> dict[str, float]:
    """Boundary-resolved growth-budget target vol (I2/I3).

    Each boundary is fit strictly on rows with index < its own ``train_end``,
    so a fold never sees its own validation window inside its scale fit and
    every path resolves the same exposure policy as the top-level blend.
    Mirrors ``_committee_evidence_weights_by_boundary``. Fail-closed (I4): a
    boundary with fewer than PNL_VOL_TARGET_BURN_IN_DAYS finite train rows
    raises ``DataIntegrityError`` naming the boundary -- PNL_TARGET_ANNUAL_VOL
    is never silently substituted.
    """
    from src.mhs.committee import growth_budget_annual_vol

    resolved: dict[str, float] = {}
    for label in sorted(train_ends):
        boundary_train = reference_daily_returns.loc[
            reference_daily_returns.index < train_ends[label]
        ].dropna()
        if len(boundary_train) < PNL_VOL_TARGET_BURN_IN_DAYS:
            raise DataIntegrityError(
                f"growth_budget target vol unresolved for boundary '{label}': "
                f"{len(boundary_train)} finite train rows before {train_ends[label]}, "
                f"require >= {PNL_VOL_TARGET_BURN_IN_DAYS}"
            )
        resolved[label] = growth_budget_annual_vol(boundary_train, envelope=envelope)
    return resolved


def _feasible_constant_risk_target(
    reference_daily_returns: pd.Series,
    envelope: GrowthRiskEnvelope,
    budget_target_vol: float,
    *,
    halflife_days: int = CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
    min_periods_days: int = CONSTANT_RISK_MIN_PERIODS_DAYS,
    quantile: float = CONSTANT_RISK_CAP_BINDING_QUANTILE,
) -> float:
    """Clamp a budget-solved target vol to the leverage-feasible risk level.

    Returns ``min(budget_target_vol, leverage_ceiling * q_p(sigma_book))``
    where ``sigma_book`` is the reference book's own causal EWMA annualized
    vol -- so cap saturation on the fit slice is structurally bounded by
    probability ``p``. Fewer than ``min_periods_days`` finite sigma rows
    raises ``DataIntegrityError``; PNL_TARGET_ANNUAL_VOL is never silently
    substituted.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if halflife_days < 1:
        raise ValueError(f"halflife_days must be >= 1, got {halflife_days}")
    if min_periods_days < 1:
        raise ValueError(f"min_periods_days must be >= 1, got {min_periods_days}")
    sigma_book = (
        reference_daily_returns
        .ewm(halflife=halflife_days, min_periods=min_periods_days)
        .std()
        .shift(1)
        * np.sqrt(365.0)
    )
    finite_rows = int(np.isfinite(sigma_book.to_numpy(dtype="float64")).sum())
    if finite_rows < min_periods_days:
        raise DataIntegrityError(
            f"constant_risk target vol unresolved: {finite_rows} finite sigma_book "
            f"rows, require >= {min_periods_days}"
        )
    feasible_cap = float(envelope.leverage_ceiling) * float(sigma_book.quantile(quantile))
    return min(float(budget_target_vol), feasible_cap)


def _constant_risk_scale(
    reference_daily_returns: pd.Series,
    target_vol: float,
    cap: float,
    *,
    halflife_days: int = CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
    min_periods_days: int = CONSTANT_RISK_MIN_PERIODS_DAYS,
    floor: float = PNL_VOL_TARGET_SCALE_FLOOR,
    warmup_returns: pd.Series | None = None,
) -> pd.Series:
    """Realized-risk-constant exposure scale with optional warm-up support.

    ``scale_t = clip(target_vol / sigma_hat_{t-1}, floor, cap)`` where
    ``sigma_hat`` is the EWMA(halflife) daily std, ``shift(1)``-ed strictly
    causal estimate annualized by sqrt(365). With ``warmup_returns``, the
    EWMA warms up on ``concat(warmup, reference)`` and is reindexed back to
    ``reference.index`` (removing the min-periods dead zone at a fold seam);
    warmup rows at or after ``reference.index[0]`` raise ``ValueError``.
    Only rows whose sigma stays un-interpreted fall back to 1.0.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")
    if halflife_days < 1:
        raise ValueError(f"halflife_days must be >= 1, got {halflife_days}")
    if min_periods_days < 1:
        raise ValueError(f"min_periods_days must be >= 1, got {min_periods_days}")
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if cap < 1.0:
        raise ValueError(f"cap must be >= 1.0, got {cap}")
    if reference_daily_returns.empty:
        return pd.Series(1.0, index=reference_daily_returns.index)
    if warmup_returns is not None:
        overlapping = warmup_returns.index[
            warmup_returns.index >= reference_daily_returns.index[0]
        ]
        if len(overlapping) > 0:
            # 엄격 인과(I-WARM): 워밍업은 참조 첫 행 이전만 허용한다.
            raise ValueError(
                f"warmup_returns must precede reference_daily_returns "
                f"({len(overlapping)} rows at or after "
                f"{reference_daily_returns.index[0]}, first offender {overlapping[0]})"
            )
        combined = pd.concat([warmup_returns, reference_daily_returns])
    else:
        combined = reference_daily_returns
    sigma = (
        combined
        .ewm(halflife=halflife_days, min_periods=min_periods_days)
        .std()
        .shift(1)
        * np.sqrt(365.0)
    ).reindex(reference_daily_returns.index)
    scale = target_vol / sigma.where(sigma > 0)
    return scale.clip(lower=floor, upper=cap).fillna(1.0)


def _constant_risk_target_vol_by_boundary(
    reference_daily_returns: pd.Series,
    envelope: GrowthRiskEnvelope,
    train_ends: Mapping[str, pd.Timestamp],
) -> dict[str, float]:
    """Boundary-resolved constant-risk target vol (leak-free, fail-closed).

    Mirrors ``_growth_budget_target_vol_by_boundary``'s slicing discipline:
    each boundary fits strictly on rows with index < its own train_end. Unlike
    ``growth_budget`` mode, the base target is the single registered
    ``CONSTANT_RISK_TARGET_ANNUAL_VOL`` constant -- never a per-boundary
    ``growth_budget_annual_vol`` re-solve, which is sample-specific and
    diverges across boundaries (measured: fold0-2 realized vol 0.14-0.16 vs
    fold3 0.29 when re-solved per boundary, reproducing
    FOLD_GROWTH_CONCENTRATION). Only the feasibility clamp
    (``_feasible_constant_risk_target``) stays boundary-local; insufficient
    history raises ``DataIntegrityError`` naming the boundary label.
    """
    resolved: dict[str, float] = {}
    for label in sorted(train_ends):
        boundary_train = reference_daily_returns.loc[
            reference_daily_returns.index < train_ends[label]
        ].dropna()
        if len(boundary_train) < PNL_VOL_TARGET_BURN_IN_DAYS:
            raise DataIntegrityError(
                f"constant_risk target vol unresolved for boundary '{label}': "
                f"{len(boundary_train)} finite train rows before {train_ends[label]}, "
                f"require >= {PNL_VOL_TARGET_BURN_IN_DAYS}"
            )
        resolved[label] = _feasible_constant_risk_target(
            boundary_train, envelope, CONSTANT_RISK_TARGET_ANNUAL_VOL,
        )
    return resolved


def _constant_risk_target_vol(
    reference_daily_returns: pd.Series,
    envelope: GrowthRiskEnvelope,
    oos_start: pd.Timestamp = COMMITTEE_OOS_START,
) -> float:
    """Leak-free top-level wrapper: mirrors ``_constant_risk_target_vol_by_boundary``'s
    ``"top_level"`` label exactly (same oos_start slice, same fixed base
    constant, same feasibility clamp) so the two entry points can never
    diverge (I2). Slicing to ``index < oos_start`` before the feasibility
    clamp is required -- fitting ``sigma_book`` on the unsliced full-history
    series would see post-OOS rows the fold paths never see, producing
    FOLD_BLEND_PATH_DIVERGENCE (I3).
    """
    train = reference_daily_returns.loc[reference_daily_returns.index < oos_start].dropna()
    if len(train) < PNL_VOL_TARGET_BURN_IN_DAYS:
        return CONSTANT_RISK_TARGET_ANNUAL_VOL
    return _feasible_constant_risk_target(train, envelope, CONSTANT_RISK_TARGET_ANNUAL_VOL)


def _envelope_exposure_cap(
    envelope: GrowthRiskEnvelope,
    target_gross: float | None,
    reference_daily_returns: pd.Series,
) -> float:
    """Verification-first, policy-constant exposure cap.

    Verifies the envelope's registered ``leverage_ceiling`` against the
    bootstrap ruin frontier computed on ``reference_daily_returns`` itself
    (the actual strategy P&L the cap will be applied to, never a synthetic
    stand-in) with ``reference_risk = std(ddof=1)``, then returns THE
    REGISTERED POLICY CONSTANT -- the cap is never refit from the returns it
    is applied to (I1). Raises ``ValueError`` (fail-closed) when the series
    has fewer than 2 finite observations, a zero/non-finite std, the solver
    is infeasible at that reference risk, or ``leverage_ceiling`` exceeds
    the verified frontier -- a ceiling beyond the bootstrap ruin frontier,
    or one that cannot be verified against real data, must never be wired.
    (``target_gross`` stays in the signature for call-site compatibility; the
    cap is budget-derived and no longer scales with nominal gross.)
    """
    del target_gross
    from src.mhs.params import COMMITTEE_GROWTH_BARS_PER_YEAR, COMMITTEE_GROWTH_N_PATHS, COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS  # noqa: I001
    from src.research.risk.growth_sizing import GrowthSizingConfig, solve_growth_optimal_risk

    r = reference_daily_returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    reference_risk = float(r.std(ddof=1)) if len(r) >= 2 else float("nan")
    if not np.isfinite(reference_risk) or reference_risk <= 0:
        raise ValueError(
            f"envelope '{envelope.name}' has leverage_ceiling={envelope.leverage_ceiling} "
            f"but reference_daily_returns has too little history/variance "
            f"({len(r)} finite rows) to verify the bootstrap ruin frontier"
        )
    config = GrowthSizingConfig(
        risk_grid=tuple(sorted(
            reference_risk * m for m in COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS
        )),
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
        raise ValueError(
            f"envelope '{envelope.name}' has leverage_ceiling={envelope.leverage_ceiling} "
            f"but the bootstrap ruin frontier is infeasible on reference_daily_returns; "
            f"ceiling must not exceed the frontier"
        )
    frontier_multiple = result.selected_risk / reference_risk
    if envelope.leverage_ceiling > frontier_multiple:
        raise ValueError(
            f"envelope '{envelope.name}' has leverage_ceiling={envelope.leverage_ceiling} "
            f"but the bootstrap ruin frontier on reference_daily_returns allows only "
            f"{frontier_multiple:.6f}x reference risk; ceiling must not exceed the frontier"
        )
    return float(envelope.leverage_ceiling)


def _assert_envelope_leverage_ceiling_verified(
    envelope: GrowthRiskEnvelope,
    reference_daily_returns: pd.Series,
    oos_start: pd.Timestamp = COMMITTEE_OOS_START,
) -> None:
    """Leak-free once-per-run audit of the registered leverage ceiling (I3).

    Mirrors ``_growth_budget_target_vol``'s slicing discipline: the bootstrap
    verification runs strictly on rows with ``index < oos_start``, so it never
    sees evaluation data. A fold-local reference window legitimately contains
    no pre-OOS rows -- fewer than PNL_VOL_TARGET_BURN_IN_DAYS finite train
    rows returns without invoking the solver (I6). Otherwise delegates to
    ``_envelope_exposure_cap`` purely for verification and discards the
    result; its fail-closed ``ValueError`` propagates unchanged (I4).
    """
    train = reference_daily_returns.loc[reference_daily_returns.index < oos_start].dropna()
    if len(train) < PNL_VOL_TARGET_BURN_IN_DAYS:
        return None
    _envelope_exposure_cap(envelope, None, train)


def resolved_exposure_cap(request: MhsDiagnosticRequest) -> float:
    """단일 소유 노출 상한(I2): 클립 상한을 결정하는 순수·데이터 비의존 함수.

    Series를 읽지 않으므로 top-level과 fold 경로가 동일 값을 반환하고 캡 축의
    FOLD_BLEND_PATH_DIVERGENCE는 구조적으로 불가능하다. 함수 내부에 리터럴
    상한(1.0 등)을 박아둘 수 없다(I1).
    """
    if not request.exposure_scale_two_sided:
        return 1.0
    if request.pnl_vol_target_mode == "median_relative":
        # median_relative는 자체 1.0 클립인 _pnl_vol_target_scale로 라우트된다.
        return 1.0
    from src.application.research.mhs.research_go import _resolved_growth_envelope

    envelope = _resolved_growth_envelope(request)
    if request.pnl_vol_target_mode == "exante_target" and envelope.name == "conservative":
        # 등록 기본 엔벨로프의 byte-identical 특수 케이스를 보존한다.
        return float(PNL_VOL_TARGET_MAX_SCALE)
    return float(envelope.leverage_ceiling)


def _replay_exposure_scale(
    reference_daily_returns: pd.Series,
    request: MhsDiagnosticRequest,
    growth_budget_target_vol: float | None = None,
    *,
    warmup_returns: pd.Series | None = None,
) -> pd.Series:
    """단일 디스패처: 노출 스케일 모드 선택 + committee_capital 합성.

    fold 경로와 top-level 경로 모두에서 동일 함수를 사용하여
    FOLD_BLEND_PATH_DIVERGENCE를 회피한다(I2). ``growth_budget_target_vol``이
    None이 아니면 growth_budget 및 non-conservative exante_target 모드에서
    fold-local 재적합 대신 그 경계별 사전 적합값을 쓴다 -- fold 참조 수익률은
    validation 윈도우만 담으므로 자기 적합은 leak이거나 fallback이다(I3/I4).
    constant_risk는 Kelly 블렌드를 경유하지 않고 즉시 반환하며
    ``warmup_returns``로 fold 검증 시작 이전 EWMA 워밍업을 받는다(I-WARM).
    """
    from src.application.research.mhs.research_go import _resolved_growth_envelope

    def _resolve_target_vol(envelope: GrowthRiskEnvelope) -> float:
        if growth_budget_target_vol is not None:
            return growth_budget_target_vol
        return _growth_budget_target_vol(reference_daily_returns, envelope=envelope)

    if request.pnl_vol_target_mode == "constant_risk":
        envelope = _resolved_growth_envelope(request)
        if growth_budget_target_vol is not None:
            target_vol = growth_budget_target_vol
        else:
            # I2/I3: leak-free oos_start slice, mirroring the fold path's
            # "top_level" boundary -- never fit sigma_book on the full
            # unsliced series (FOLD_BLEND_PATH_DIVERGENCE root cause).
            target_vol = _constant_risk_target_vol(reference_daily_returns, envelope)
        # I-NO-PROCYCLIC: 경기순응 Kelly 항은 위험 상수성을 깨뜨린다.
        return _constant_risk_scale(
            reference_daily_returns, target_vol=target_vol,
            cap=resolved_exposure_cap(request),
            warmup_returns=warmup_returns,
        )
    if request.pnl_vol_target_mode == "median_relative":
        scale = _pnl_vol_target_scale(reference_daily_returns)
    elif request.pnl_vol_target_mode == "exante_target":
        envelope = _resolved_growth_envelope(request)
        if envelope.name == "conservative":
            # I4: the registered default envelope reproduces every
            # pre-existing exante_target call byte-for-byte -- it must never
            # route through the solver-fitted target_vol below, which would
            # silently change the production default's exposure.
            scale = _exante_vol_target_scale(reference_daily_returns, cap=resolved_exposure_cap(request))
        else:
            target_vol = _resolve_target_vol(envelope)
            scale = _exante_vol_target_scale(
                reference_daily_returns, target_vol=target_vol,
                cap=resolved_exposure_cap(request),
            )
    elif request.pnl_vol_target_mode == "growth_budget":
        envelope = _resolved_growth_envelope(request)
        target_vol = _resolve_target_vol(envelope)
        scale = _exante_vol_target_scale(
            reference_daily_returns, target_vol=target_vol,
            cap=resolved_exposure_cap(request),
        )
    else:
        raise ValueError(f"unknown pnl_vol_target_mode '{request.pnl_vol_target_mode}'")
    return _committee_capital_replay_scale(
        scale, reference_daily_returns,
        request.committee_capital, request.committee_kelly_sizing,
        cap=resolved_exposure_cap(request),
    )


def is_streaming_scale_mode(request: MhsDiagnosticRequest) -> bool:
    """True only when the resolved exposure scale is causal + prefix-deterministic.

    A streaming (coupled one-pass) coordinator may recompute the scale from the
    reference-return PREFIX alone iff every scale value at day ``d`` depends
    only on realized returns strictly before ``d``. Verified for:

    * ``median_relative`` -- rolling trailing vol and rolling median, both
      ``shift(1)``-ed;
    * ``exante_target`` under the registered conservative envelope --
      ``clip(PNL_TARGET_ANNUAL_VOL / (ewm(std, halflife=20d,
      min_periods=20).shift(1) * sqrt(365)), floor, cap)``: a constant target
      with a one-day shift.

    Returns False for ``growth_budget`` and non-conservative envelopes
    (``_growth_budget_target_vol`` slices the whole pre-OOS train set -- not a
    prefix computation), and when the Kelly blend is actually active (the
    blend changes the resolved formula). ``constant_risk`` also returns False:
    its feasible target needs the train-slice sigma_book quantile fit.
    """
    if request.committee_capital and request.committee_kelly_sizing:
        # _committee_capital_replay_scale만이 실제 Kelly 블렌드를 발동하는
        # 조건이다(둘 다 True). committee_kelly_sizing 단독으로는 무영향이므로
        # committee_capital=False일 때 스트리밍 경로를 부당하게 차단해서는 안 된다.
        return False
    if request.pnl_vol_target_mode == "median_relative":
        return True
    if request.pnl_vol_target_mode == "exante_target":
        from src.application.research.mhs.research_go import _resolved_growth_envelope

        return _resolved_growth_envelope(request).name == "conservative"
    return False
