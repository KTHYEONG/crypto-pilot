"""Stage 5: risk/event-based filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage5Config

MAX_OI_TO_ADV = 12.0


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
        sign_flip = (funding_rate_8h * prev_rate) < 0.0
        flip_signal = flip_signal | sign_flip.fillna(False)
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
        ((funding - float(funding.mean())) / funding_std)
        if funding_std > 0.0
        else pd.Series(0.0, index=frame.index)
    )
    funding_z = (
        funding_z_input.where(funding_z_input.notna(), funding_z_fallback)
        .abs()
        .fillna(0.0)
    )
    funding_sign_flip = _resolve_funding_sign_flip(frame, config=cfg, funding_rate_8h=funding)
    funding_anomaly = (funding_z > cfg.max_abs_funding_z) | funding_sign_flip
    basis_z = pd.to_numeric(
        frame.get("basis_z_score", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0).abs()
    oi = frame.get("oi_usdt_median", pd.Series(0.0, index=frame.index)).fillna(0.0)
    vol_30d = frame.get("vol_30d", pd.Series(0.0, index=frame.index)).fillna(0.0).abs()
    adv = frame.get("adv_usdt_median", pd.Series(np.nan, index=frame.index)).replace(0, np.nan)
    risk_override = frame.get(
        "risk_event_override", pd.Series("", index=frame.index)
    ).fillna("").astype("string")
    override_knowledge_date = frame.get(
        "risk_event_knowledge_date", pd.Series(pd.NaT, index=frame.index)
    )
    override_active = risk_override != ""
    override_missing_knowledge = override_active & pd.to_datetime(
        override_knowledge_date, utc=True, errors="coerce"
    ).isna()
    oi_to_adv = (oi / adv).replace([np.inf, -np.inf], np.nan).fillna(np.inf)

    pass_mask = (
        (age >= cfg.min_listing_age_days)
        & (~funding_anomaly)
        & (basis_z <= cfg.max_abs_basis_z)
        & (oi_to_adv <= MAX_OI_TO_ADV)
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
    reasons = np.where((reasons == "") & (basis_z > cfg.max_abs_basis_z), "basis_anomaly", reasons)
    reasons = np.where((reasons == "") & (oi_to_adv > MAX_OI_TO_ADV), "oi_to_adv_too_high", reasons)
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
    out["oi_to_adv"] = oi_to_adv.astype(float)
    report = pd.DataFrame(
        {
            "symbol": out["symbol"].astype("string"),
            "stage": "stage5_risk_events",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
            "oi_to_adv": oi_to_adv.astype(float),
            "funding_z_abs": funding_z.astype(float),
            "funding_sign_flip": funding_sign_flip.astype(bool),
            "basis_z_abs": basis_z.astype(float),
        }
    )
    return out.loc[pass_mask].copy(), report
