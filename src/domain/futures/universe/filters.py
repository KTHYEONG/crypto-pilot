"""Universe filtering stages for futures universe selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Stage3Config:
    """Liquidity and execution feasibility gates."""

    min_adv_usdt_median: float = 50_000_000.0
    max_amihud_30d: float = 1.00e-9
    max_clip_to_adv: float = 0.0025
    enable_oi_adv_crowding_gate: bool = True
    max_oi_to_adv: float = 12.0
    screening_tier: str = "mid"
    screening_clip_usdt_by_tier: dict[str, float] = field(
        default_factory=lambda: {
            "seed": 1_000.0,
            "small": 5_000.0,
            "mid": 10_000.0,
            "large": 25_000.0,
            "xlarge": 50_000.0,
        }
    )
    capacity_clip_usdt_list: tuple[float, ...] = (50_000.0, 100_000.0)


@dataclass(frozen=True, slots=True)
class Stage4Config:
    """Execution-cost model gates."""

    max_execution_cost_bps: float = 35.0
    default_taker_fee_bps: float = 5.0
    default_half_spread_bps: float = 1.0
    spread_source_switch_date: str = "2020-01-01"
    pre2020_half_spread_bps: float = 2.5
    post2020_half_spread_bps: float = 1.0
    default_impact_coef_bps: float = 18.0


@dataclass(frozen=True, slots=True)
class Stage5Config:
    """Risk-event and anomaly gates.

    Notes:
        vol_30d는 4h 바 기준 연율화 변동성 (std * sqrt(6*365)).
        median ~0.75, p90 ~1.81 수준의 스케일.
        min=0.05(5% 연율, 사실상 거래 없는 코인 제거),
        max=4.0(400% 연율, 극단적 meme 제거).

        funding_sign_flip_min_abs: 부호 반전이 이상치로 인정받으려면
        양쪽 모두 절대값이 이 임계값 이상이어야 한다.
        +0.001% → -0.001% 수준의 중립 진동은 이상치에서 제외.
    """

    min_listing_age_days: int = 180
    min_vol_30d: float = 0.05  # 5% annualized — 거래 없는 죽은 코인 제거
    max_vol_30d: float = 4.0  # 400% annualized — 극단적 meme/junk 제거
    max_abs_funding_z: float = 2.5
    enable_funding_sign_flip: bool = True
    funding_sign_flip_min_abs: float = 0.001  # |funding| > threshold 양쪽 모두여야 flip 이상치
    funding_sign_flip_columns: tuple[str, ...] = (
        "funding_sign_flip_1d",
        "funding_sign_reversal_1d",
        "funding_sign_change_1d",
    )
    funding_prev_rate_column: str = "funding_rate_8h_prev"


HalfSpreadFallback = Callable[[pd.DataFrame, Stage4Config, date], pd.Series]
_OI_SOURCE_COLUMNS: tuple[str, ...] = (
    "oi_usdt_median",
    "sum_open_interest_value",
    "open_interest_usdt",
    "open_interest",
)


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


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _first_available_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom_safe = denom.replace(0, np.nan)
    return numer / denom_safe


# --- Stage 3: Liquidity ---


def apply_liquidity_stage(
    frame: pd.DataFrame,
    config: Stage3Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter symbols by ADV/Amihud/clip capacity."""
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])
    cfg = config or Stage3Config()

    adv = frame.get("adv_usdt_median", pd.Series(0.0, index=frame.index)).fillna(0.0)
    amihud = frame.get("amihud_30d", pd.Series(np.nan, index=frame.index))
    if amihud.isna().all() and "vol_30d" in frame.columns:
        vol = frame["vol_30d"].fillna(0.0)
        amihud = _safe_div(vol.abs(), adv).fillna(np.inf)
    default_clip = cfg.screening_clip_usdt_by_tier.get(cfg.screening_tier, 10_000.0)
    clip_usdt = frame.get("screening_clip_usdt", pd.Series(default_clip, index=frame.index)).fillna(default_clip)
    clip_to_adv = _safe_div(clip_usdt, adv).fillna(np.inf)
    oi_col = _first_available_column(frame, _OI_SOURCE_COLUMNS)
    oi_series = _numeric_series(frame, oi_col) if oi_col is not None else pd.Series(np.nan, index=frame.index)
    oi_to_adv = _safe_div(oi_series, adv)
    oi_gate_enabled = bool(cfg.enable_oi_adv_crowding_gate) and oi_col is not None
    if oi_gate_enabled:
        oi_pass = oi_to_adv.le(cfg.max_oi_to_adv) | oi_to_adv.isna()
    else:
        oi_pass = pd.Series(True, index=frame.index)

    pass_mask = (
        (adv >= cfg.min_adv_usdt_median)
        & (amihud <= cfg.max_amihud_30d)
        & (clip_to_adv <= cfg.max_clip_to_adv)
        & oi_pass
    )
    reasons = np.where(adv < cfg.min_adv_usdt_median, "adv_too_low", "")
    reasons = np.where((reasons == "") & (amihud > cfg.max_amihud_30d), "amihud_too_high", reasons)
    reasons = np.where(
        (reasons == "") & (clip_to_adv > cfg.max_clip_to_adv),
        "clip_too_large_vs_adv",
        reasons,
    )
    reasons = np.where(
        (reasons == "") & oi_gate_enabled & oi_to_adv.gt(cfg.max_oi_to_adv),
        "oi_adv_crowded",
        reasons,
    )
    reasons = pd.Series(np.where(reasons == "", "pass", reasons), index=frame.index, dtype="string")

    out = frame.copy()
    out["clip_to_adv"] = clip_to_adv.astype(float)
    out["oi_to_adv"] = oi_to_adv.astype(float)
    report = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype("string"),
            "stage": "stage3_liquidity",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
            "screening_clip_usdt": clip_usdt.astype(float),
            "clip_to_adv": clip_to_adv.astype(float),
            "oi_to_adv": oi_to_adv.astype(float),
        }
    )
    return out.loc[pass_mask].copy(), report


# --- Stage 4: Cost Model ---


def _default_half_spread_fallback(
    frame: pd.DataFrame,
    config: Stage4Config,
    as_of_date: date,
) -> pd.Series:
    switch_date = _to_date(config.spread_source_switch_date)
    fallback_bps = config.pre2020_half_spread_bps if as_of_date < switch_date else config.post2020_half_spread_bps
    return pd.Series(fallback_bps, index=frame.index, dtype="float64")


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
    gamma.loc[has_lag] = (
        np.log(
            np.maximum(high.loc[has_lag], high_prev.loc[has_lag]) / np.minimum(low.loc[has_lag], low_prev.loc[has_lag])
        )
        ** 2
    )
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
        half_spread_bps = bookdepth_half_spread_input.where(bookdepth_half_spread_input.notna(), half_spread_input)
        half_spread_source = pd.Series("bookdepth_half_spread_bps", index=out.index, dtype="string")
        half_spread_source = half_spread_source.where(bookdepth_half_spread_input.notna(), "half_spread_bps")
        half_spread_bps = half_spread_bps.where(half_spread_bps.notna(), half_spread_fallback_values).fillna(
            cfg.default_half_spread_bps
        )
        half_spread_source = half_spread_source.where(
            (bookdepth_half_spread_input.notna() | half_spread_input.notna()),
            "fallback_default",
        )
    else:
        half_spread_bps = cs_half_spread.where(cs_half_spread.notna(), half_spread_fallback_values).fillna(
            cfg.default_half_spread_bps
        )
        half_spread_source = pd.Series("corwin_schultz", index=out.index, dtype="string")
        half_spread_source = half_spread_source.where(cs_half_spread.notna(), "fallback_default")

    half_spread_bps = half_spread_bps.replace([np.inf, -np.inf], np.nan).fillna(cfg.default_half_spread_bps)
    clip_usdt = out.get("screening_clip_usdt", pd.Series(10_000.0, index=out.index)).fillna(10_000.0)
    adv = out.get("adv_usdt_median", pd.Series(np.nan, index=out.index)).replace(0.0, np.nan)
    impact_coef_bps = out.get("impact_coef_bps", pd.Series(cfg.default_impact_coef_bps, index=out.index)).fillna(
        cfg.default_impact_coef_bps
    )
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


# --- Stage 5: Risk Events ---


def _resolve_funding_sign_flip(
    frame: pd.DataFrame,
    *,
    config: Stage5Config,
    funding_rate_8h: pd.Series,
) -> pd.Series:
    """Return boolean sign-flip anomaly signal from optional columns and fallback inputs."""
    flip_signal = pd.Series(False, index=frame.index)
    if not config.enable_funding_sign_flip:
        return flip_signal

    for column in config.funding_sign_flip_columns:
        if column not in frame.columns:
            continue
        raw = frame[column]
        if pd.api.types.is_bool_dtype(raw):
            candidate = raw.fillna(False).astype(bool)
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            candidate = numeric.fillna(0.0).abs() > 0.0
        flip_signal = flip_signal | candidate

    prev_column = config.funding_prev_rate_column
    if prev_column in frame.columns:
        prev_rate = pd.to_numeric(frame[prev_column], errors="coerce")
        # 양쪽 모두 |funding| > flip_threshold 인 경우의 부호 반전만 유의미한 이상치로 처리.
        # +0.001% → -0.001% 수준의 중립 진동(노이즈)을 이상치 제외에서 제거.
        flip_threshold = float(config.funding_sign_flip_min_abs)
        significant_flip = (
            (funding_rate_8h.abs() > flip_threshold)
            & (prev_rate.abs() > flip_threshold)
            & ((funding_rate_8h * prev_rate) < 0.0)
        )
        flip_signal = flip_signal | significant_flip.fillna(False)
    return flip_signal


def apply_risk_events_stage(
    frame: pd.DataFrame,
    *,
    config: Stage5Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter symbols using event and anomaly guards."""
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])
    cfg = config or Stage5Config()

    age = frame.get("listing_age_days", pd.Series(0, index=frame.index)).fillna(0)
    funding = pd.to_numeric(
        frame.get("funding_rate_8h", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    funding_z_input = pd.to_numeric(
        frame.get("funding_zscore", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    funding_std = float(funding.std(ddof=0))
    funding_z_fallback = (
        ((funding - float(funding.mean())) / funding_std) if funding_std > 0.0 else pd.Series(0.0, index=frame.index)
    )
    funding_z = funding_z_input.where(funding_z_input.notna(), funding_z_fallback).abs().fillna(0.0)
    funding_sign_flip = _resolve_funding_sign_flip(frame, config=cfg, funding_rate_8h=funding)
    funding_anomaly = (funding_z > cfg.max_abs_funding_z) | funding_sign_flip
    vol_30d = frame.get("vol_30d", pd.Series(0.0, index=frame.index)).fillna(0.0).abs()
    risk_override = frame.get("risk_event_override", pd.Series("", index=frame.index)).fillna("").astype("string")
    override_knowledge_date = frame.get("risk_event_knowledge_date", pd.Series(pd.NaT, index=frame.index))
    override_active = risk_override != ""
    override_missing_knowledge = (
        override_active & pd.to_datetime(override_knowledge_date, utc=True, errors="coerce").isna()
    )

    pass_mask = (
        (age >= cfg.min_listing_age_days)
        & (~funding_anomaly)
        & (vol_30d >= cfg.min_vol_30d)
        & (vol_30d <= cfg.max_vol_30d)
        & (~override_active)
        & (~override_missing_knowledge)
    )

    reasons = np.where(age < cfg.min_listing_age_days, "listing_age_too_young", "")
    reasons = np.where(
        (reasons == "") & funding_anomaly,
        "funding_anomaly",
        reasons,
    )
    reasons = np.where((reasons == "") & (vol_30d < cfg.min_vol_30d), "vol_too_low", reasons)
    reasons = np.where((reasons == "") & (vol_30d > cfg.max_vol_30d), "vol_too_high", reasons)
    reasons = np.where(
        (reasons == "") & override_missing_knowledge,
        "manual_override_fail_closed_missing_knowledge_date",
        reasons,
    )
    reasons = np.where((reasons == "") & (risk_override != ""), "manual_risk_override", reasons)
    reasons = pd.Series(np.where(reasons == "", "pass", reasons), index=frame.index, dtype="string")

    out = frame.copy()
    report = pd.DataFrame(
        {
            "symbol": out["symbol"].astype("string"),
            "stage": "stage5_risk_events",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
            "funding_z_abs": funding_z.astype(float),
            "funding_sign_flip": funding_sign_flip.astype(bool),
        }
    )
    return out.loc[pass_mask].copy(), report
