"""Stage 1: structural eligibility filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVERAGED_TOKEN_PATTERNS = ("UP", "DOWN", "BULL", "BEAR")


def apply_structure_stage(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply structure filters in vectorized form."""
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])

    symbol = frame["symbol"].astype("string")
    upper = symbol.str.upper()
    is_perp = frame.get("contract_type", pd.Series("", index=frame.index)).eq("PERPETUAL")
    is_usdt_quote = frame.get("quote_asset", pd.Series("", index=frame.index)).eq("USDT")
    is_usdt_margin = frame.get("margin_asset", pd.Series("", index=frame.index)).eq("USDT")
    is_trading = frame.get("status", pd.Series("", index=frame.index)).eq("TRADING")
    multiplier = pd.to_numeric(
        frame.get("contract_multiplier", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    multiplier_valid = multiplier.notna() & np.isfinite(multiplier) & (multiplier > 0.0)
    leveraged = np.logical_or.reduce(
        [upper.str.contains(p, regex=False) for p in LEVERAGED_TOKEN_PATTERNS]
    )

    usdt_quote_or_margin = is_usdt_quote | is_usdt_margin
    pass_mask = is_perp & usdt_quote_or_margin & is_trading & multiplier_valid & (~leveraged)
    reasons = np.where(~is_perp, "not_perpetual", "")
    reasons = np.where(
        (reasons == "") & (~usdt_quote_or_margin),
        "not_usdt_quote_or_margin",
        reasons,
    )
    reasons = np.where((reasons == "") & (~is_trading), "not_trading", reasons)
    reasons = np.where(
        (reasons == "") & (~multiplier_valid),
        "invalid_contract_multiplier",
        reasons,
    )
    reasons = np.where((reasons == "") & leveraged, "leveraged_token_pattern", reasons)
    reasons = pd.Series(np.where(reasons == "", "pass", reasons), index=frame.index, dtype="string")

    report = pd.DataFrame(
        {
            "symbol": symbol,
            "stage": "stage1_structure",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
        }
    )
    return frame.loc[pass_mask].copy(), report
