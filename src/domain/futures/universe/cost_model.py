"""Stage 4: execution cost model filters (funding excluded from execution cost)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from .config import Stage4Config

HalfSpreadFallback = Callable[[pd.DataFrame, Stage4Config, date], pd.Series]


def _to_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _resolve_as_of_date(frame: pd.DataFrame, as_of: str | date | None) -> date:
    if as_of is not None:
        return _to_date(as_of)
    if "date" in frame.columns:
        parsed = pd.to_datetime(frame["date"], utc=True, errors="coerce").dropna()
        if not parsed.empty:
            latest_iso = str(parsed.max())
            return _to_date(latest_iso[:10])
    return date.today()


def _default_half_spread_fallback(
    frame: pd.DataFrame,
    config: Stage4Config,
    as_of_date: date,
) -> pd.Series:
    switch_date = _to_date(config.spread_source_switch_date)
    fallback_bps = (
        config.pre2020_half_spread_bps
        if as_of_date < switch_date
        else config.post2020_half_spread_bps
    )
    return pd.Series(fallback_bps, index=frame.index, dtype="float64")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _corwin_schultz_half_spread_bps(frame: pd.DataFrame) -> pd.Series:
    """Return half-spread estimate (bps) from OHLC using a vectorized CS-style formula."""
    high = _numeric_series(frame, "high")
    low = _numeric_series(frame, "low")
    close = _numeric_series(frame, "close")

    valid_hl = (high > 0.0) & (low > 0.0) & high.ge(low)
    hl_log = pd.Series(np.nan, index=frame.index, dtype="float64")
    hl_log.loc[valid_hl] = np.log(high.loc[valid_hl] / low.loc[valid_hl])

    # If lagged OHLC exists, use canonical 2-day CS inputs; otherwise degrade to same-day proxy.
    high_prev = _numeric_series(frame, "high_prev")
    low_prev = _numeric_series(frame, "low_prev")
    has_lag = high_prev.notna() & low_prev.notna() & (high_prev > 0.0) & (low_prev > 0.0)
    gamma = pd.Series(np.nan, index=frame.index, dtype="float64")
    gamma.loc[has_lag] = np.log(
        np.maximum(high.loc[has_lag], high_prev.loc[has_lag])
        / np.minimum(low.loc[has_lag], low_prev.loc[has_lag])
    ) ** 2
    gamma = gamma.where(gamma.notna(), hl_log.pow(2))

    beta = hl_log.pow(2).where(hl_log.notna(), np.nan)
    const_k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / const_k - np.sqrt(gamma / const_k)
    alpha = alpha.clip(lower=0.0)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    half_spread_bps = spread * 5_000.0

    # Last-resort proxy from intrabar range and close when CS terms are not available.
    close_valid = close > 0.0
    range_proxy_bps = (
        (np.log(high.where(valid_hl) / low.where(valid_hl)).abs() * 5_000.0)
        .where(close_valid)
        .replace([np.inf, -np.inf], np.nan)
    )
    return half_spread_bps.where(half_spread_bps.notna(), range_proxy_bps)


def apply_cost_model_stage(
    frame: pd.DataFrame,
    *,
    config: Stage4Config | None = None,
    as_of: str | date | None = None,
    half_spread_fallback: HalfSpreadFallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate round-trip execution cost and gate by threshold."""
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])
    cfg = config or Stage4Config()

    out = frame.copy()
    as_of_date = _resolve_as_of_date(out, as_of)
    taker_bps = out.get(
        "taker_fee_bps",
        pd.Series(cfg.default_taker_fee_bps, index=out.index),
    ).fillna(cfg.default_taker_fee_bps)
    switch_date = _to_date(cfg.spread_source_switch_date)
    is_post2020 = as_of_date >= switch_date
    bookdepth_half_spread_input = _numeric_series(out, "bookdepth_half_spread_bps")
    half_spread_input = _numeric_series(out, "half_spread_bps")
    fallback_fn = half_spread_fallback or _default_half_spread_fallback
    half_spread_fallback_values = pd.to_numeric(
        fallback_fn(out, cfg, as_of_date).reindex(out.index),
        errors="coerce",
    )
    cs_half_spread = _corwin_schultz_half_spread_bps(out)

    if is_post2020:
        half_spread_bps = bookdepth_half_spread_input.where(
            bookdepth_half_spread_input.notna(), half_spread_input
        )
        half_spread_source = pd.Series("bookdepth_half_spread_bps", index=out.index, dtype="string")
        half_spread_source = half_spread_source.where(
            bookdepth_half_spread_input.notna(), "half_spread_bps"
        )
        half_spread_bps = half_spread_bps.where(
            half_spread_bps.notna(), half_spread_fallback_values
        ).fillna(cfg.default_half_spread_bps)
        half_spread_source = half_spread_source.where(
            (bookdepth_half_spread_input.notna() | half_spread_input.notna()),
            "fallback_default",
        )
    else:
        half_spread_bps = cs_half_spread.where(
            cs_half_spread.notna(), half_spread_fallback_values
        ).fillna(cfg.default_half_spread_bps)
        half_spread_source = pd.Series("corwin_schultz", index=out.index, dtype="string")
        half_spread_source = half_spread_source.where(cs_half_spread.notna(), "fallback_default")

    half_spread_bps = half_spread_bps.replace([np.inf, -np.inf], np.nan).fillna(
        cfg.default_half_spread_bps
    )
    clip_usdt = out.get("screening_clip_usdt", pd.Series(10_000.0, index=out.index)).fillna(
        10_000.0
    )
    adv = out.get("adv_usdt_median", pd.Series(np.nan, index=out.index)).replace(0.0, np.nan)
    impact_coef_bps = out.get(
        "impact_coef_bps", pd.Series(cfg.default_impact_coef_bps, index=out.index)
    ).fillna(cfg.default_impact_coef_bps)
    impact_bps_raw = out.get("impact_bps", pd.Series(np.nan, index=out.index))
    # Use precomputed impact when provided; otherwise use sqrt-impact model.
    impact_bps = impact_bps_raw.where(
        impact_bps_raw.notna(),
        impact_coef_bps * np.sqrt((clip_usdt / adv).clip(lower=0.0)),
    ).fillna(np.inf)
    tick_cost_bps = out.get("tick_cost_bps", pd.Series(np.nan, index=out.index))
    tick_cost_bps = tick_cost_bps.where(
        tick_cost_bps.notna(),
        (
            0.5
            * 1e4
            * (
                pd.to_numeric(
                    out.get("tick_size", pd.Series(np.nan, index=out.index)),
                    errors="coerce",
                )
                / pd.to_numeric(
                    out.get("mark_price", pd.Series(np.nan, index=out.index)),
                    errors="coerce",
                )
            )
        ),
    )
    tick_cost_bps = tick_cost_bps.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Funding is intentionally not included in execution cost by design.
    execution_cost_bps = (2.0 * taker_bps) + (2.0 * half_spread_bps) + impact_bps + tick_cost_bps
    pass_mask = execution_cost_bps <= cfg.max_execution_cost_bps
    reasons = pd.Series(
        np.where(pass_mask, "pass", "execution_cost_too_high"),
        index=out.index,
        dtype="string",
    )

    out["execution_cost_bps"] = execution_cost_bps.astype(float)
    out["half_spread_source"] = half_spread_source.astype("string")
    report = pd.DataFrame(
        {
            "symbol": out["symbol"].astype("string"),
            "stage": "stage4_cost_model",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
            "impact_bps": impact_bps.astype(float),
            "tick_cost_bps": tick_cost_bps.astype(float),
            "half_spread_bps": half_spread_bps.astype(float),
            "half_spread_source": half_spread_source.astype("string"),
            "execution_cost_bps": execution_cost_bps.astype(float),
        }
    )
    return out.loc[pass_mask].copy(), report
