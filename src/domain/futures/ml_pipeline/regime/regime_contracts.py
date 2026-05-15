"""Regime probability contracts and compatibility helpers.

Primary contract is canonical 4-state regime posterior.
Legacy 5-column hmm_prob_* outputs are derived for backward compatibility.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REGIME_STATE_COLUMNS: tuple[str, ...] = (
    "regime_prob_risk_on_calm",
    "regime_prob_risk_on_volatile",
    "regime_prob_risk_off_trend",
    "regime_prob_chop_liquidity_thin",
)

LEGACY_HMM_PROB_COLUMNS: tuple[str, ...] = (
    "hmm_prob_bull_calm",
    "hmm_prob_bull_vol_up",
    "hmm_prob_bear_trend",
    "hmm_prob_chop",
    "hmm_prob_crisis",
)

# Keep historical symbol for existing imports.
REGIME_PROB_COLUMNS: tuple[str, ...] = LEGACY_HMM_PROB_COLUMNS

REGIME_STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "regime_prob_risk_on_calm": (
        "regime_prob_risk_on_calm",
        "hmm_prob_bull_calm",
    ),
    "regime_prob_risk_on_volatile": (
        "regime_prob_risk_on_volatile",
        "hmm_prob_bull_vol_up",
    ),
    "regime_prob_risk_off_trend": (
        "regime_prob_risk_off_trend",
        "hmm_prob_bear_trend",
        "hmm_prob_bear",
    ),
    "regime_prob_chop_liquidity_thin": (
        "regime_prob_chop_liquidity_thin",
        "hmm_prob_chop",
    ),
}

REGIME_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(REGIME_PROB_COLUMNS)}


def _safe_row_normalize(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    row_sum = arr.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1e-12
    if np.any(bad):
        arr[bad] = 1.0 / float(arr.shape[1])
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / np.maximum(row_sum, 1e-12)


def canonical_regime_prob_columns() -> tuple[str, ...]:
    """Return legacy hmm_prob_* canonical order (backward-compatible API)."""
    return REGIME_PROB_COLUMNS


def canonical_regime_state_columns() -> tuple[str, ...]:
    """Return primary canonical 4-state regime posterior order."""
    return REGIME_STATE_COLUMNS


def normalize_regime_state_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with canonical 4-state columns and row-normalized probs."""
    out = pd.DataFrame(index=df.index)
    for canonical in REGIME_STATE_COLUMNS:
        val = None
        for alias in REGIME_STATE_ALIASES[canonical]:
            if alias in df.columns:
                val = pd.to_numeric(df[alias], errors="coerce")
                break
        out[canonical] = 0.0 if val is None else val.fillna(0.0).astype(np.float64)
    mat = _safe_row_normalize(out.loc[:, REGIME_STATE_COLUMNS].to_numpy(dtype=np.float64, copy=False))
    out.loc[:, REGIME_STATE_COLUMNS] = mat
    return out


def derive_legacy_hmm_prob_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derive legacy 5-column hmm_prob_* frame from canonical 4-state probabilities."""
    state_df = normalize_regime_state_frame(df)
    out = pd.DataFrame(index=state_df.index)
    out["hmm_prob_bull_calm"] = state_df["regime_prob_risk_on_calm"]
    out["hmm_prob_bull_vol_up"] = state_df["regime_prob_risk_on_volatile"]
    out["hmm_prob_bear_trend"] = state_df["regime_prob_risk_off_trend"]
    out["hmm_prob_chop"] = state_df["regime_prob_chop_liquidity_thin"]
    out["hmm_prob_crisis"] = np.clip(
        0.65 * out["hmm_prob_bear_trend"] + 0.35 * out["hmm_prob_chop"],
        0.0,
        1.0,
    )
    return out


def normalize_regime_prob_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return backward-compatible legacy hmm_prob_* columns."""
    return derive_legacy_hmm_prob_frame(df)


def regime_prob_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract legacy hmm_prob_* matrix for existing consumers."""
    ordered = normalize_regime_prob_frame(df)
    return ordered.loc[:, REGIME_PROB_COLUMNS].to_numpy(dtype=np.float64, copy=False)


def semantic_probs_from_vector(probs: Sequence[float]) -> dict[str, float]:
    """Map a legacy probability vector to semantic keys."""
    if len(probs) < len(REGIME_PROB_COLUMNS):
        raise ValueError("Expected legacy 5-state probability vector.")
    p = np.asarray(probs, dtype=np.float64)
    bull_calm = float(p[REGIME_INDEX["hmm_prob_bull_calm"]])
    bull_vol_up = float(p[REGIME_INDEX["hmm_prob_bull_vol_up"]])
    bear = float(p[REGIME_INDEX["hmm_prob_bear_trend"]])
    chop = float(p[REGIME_INDEX["hmm_prob_chop"]])
    crisis = float(p[REGIME_INDEX["hmm_prob_crisis"]])
    return {
        "bull": bull_calm + bull_vol_up,
        "bull_calm": bull_calm,
        "bull_vol_up": bull_vol_up,
        "bear": bear,
        "chop": chop,
        "crisis": crisis,
    }


def semantic_probs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return bull/bear/chop/crisis semantics from any compatible frame."""
    ordered = normalize_regime_prob_frame(df)
    out = pd.DataFrame(index=ordered.index)
    out["bull"] = ordered["hmm_prob_bull_calm"] + ordered["hmm_prob_bull_vol_up"]
    out["bull_calm"] = ordered["hmm_prob_bull_calm"]
    out["bull_vol_up"] = ordered["hmm_prob_bull_vol_up"]
    out["bear"] = ordered["hmm_prob_bear_trend"]
    out["chop"] = ordered["hmm_prob_chop"]
    out["crisis"] = ordered["hmm_prob_crisis"]
    return out
