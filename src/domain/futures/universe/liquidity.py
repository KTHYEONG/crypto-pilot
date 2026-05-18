"""Stage 3: liquidity and execution capacity filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage3Config


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom_safe = denom.replace(0, np.nan)
    return numer / denom_safe


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
    clip_usdt = frame.get("screening_clip_usdt", pd.Series(default_clip, index=frame.index)).fillna(
        default_clip
    )
    clip_to_adv = _safe_div(clip_usdt, adv).fillna(np.inf)

    pass_mask = (
        (adv >= cfg.min_adv_usdt_median)
        & (amihud <= cfg.max_amihud_30d)
        & (clip_to_adv <= cfg.max_clip_to_adv)
    )
    reasons = np.where(adv < cfg.min_adv_usdt_median, "adv_too_low", "")
    reasons = np.where((reasons == "") & (amihud > cfg.max_amihud_30d), "amihud_too_high", reasons)
    reasons = np.where(
        (reasons == "") & (clip_to_adv > cfg.max_clip_to_adv),
        "clip_too_large_vs_adv",
        reasons,
    )
    reasons = pd.Series(np.where(reasons == "", "pass", reasons), index=frame.index, dtype="string")

    out = frame.copy()
    out["clip_to_adv"] = clip_to_adv.astype(float)
    report = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype("string"),
            "stage": "stage3_liquidity",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
            "screening_clip_usdt": clip_usdt.astype(float),
            "clip_to_adv": clip_to_adv.astype(float),
        }
    )
    return out.loc[pass_mask].copy(), report
