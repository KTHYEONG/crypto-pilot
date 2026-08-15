"""Per-symbol beta scaling + idiosyncratic risk overlay for HMM modulator.

BTC-anchored HMM posterior를 심볼별 β 스케일로 변환하고,
고유변동성(idiosyncratic volatility) 기반 리스크 오버레이를 적용한다.

Mathematical references:
    Phase A: Rolling beta shrinkage (Vasicek-style) with asymmetric modulator scaling.
    Phase B: Idiosyncratic vol z-score + rolling drawdown stress composite.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers (pure vectorised — no Python time-loops)
# ---------------------------------------------------------------------------

_SIGMOID_CLIP = 20.0  # exp overflow guard


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clip = np.clip(x, -_SIGMOID_CLIP, _SIGMOID_CLIP)
    result: np.ndarray = 1.0 / (1.0 + np.exp(-x_clip))
    return result


def _log_returns(close: pd.Series) -> pd.Series:
    """1-bar log returns; first bar = NaN."""
    shifted = close.shift(1)
    ratio: pd.Series = close / shifted
    log_ret: pd.Series = ratio.apply(np.log)
    return log_ret


# ---------------------------------------------------------------------------
# Phase A: Rolling Beta
# ---------------------------------------------------------------------------

def _compute_rolling_beta(
    r_sym: pd.Series,
    r_anchor: pd.Series,
    window: int,
    beta_min: float,
    beta_max: float,
    beta_prior: float,
) -> pd.Series:
    """Shrunk, clipped rolling beta of sym vs anchor.

    Algorithm:
        β_raw = rolling_cov(r_i, r_b, W) / (rolling_var(r_b, W) + ε)
        λ     = clip(n_valid / W, 0, 1)
        β_s   = λ * β_raw + (1 - λ) * prior
        β̃     = clip(β_s, min, max)

    Args:
        r_sym: 1h log returns of the symbol (aligned to anchor index).
        r_anchor: 1h log returns of the anchor (BTC).
        window: Rolling window length W.
        beta_min: Hard floor for β̃.
        beta_max: Hard ceiling for β̃.
        beta_prior: Shrinkage target (usually 1.0).

    Returns:
        pd.Series of beta-tilde aligned to r_anchor's index. shift(1) already applied.

    Note:
        Time complexity: O(N) per pandas rolling -- amortised.

    """
    eps = 1e-12
    aligned_sym, aligned_anc = r_sym.align(r_anchor, join="right", fill_value=np.nan)

    rolling_cov = aligned_sym.rolling(window, min_periods=2).cov(aligned_anc)
    rolling_var = aligned_anc.rolling(window, min_periods=2).var()

    beta_raw = rolling_cov / (rolling_var + eps)

    # Vasicek shrinkage: λ proportional to available valid observations
    n_valid = aligned_sym.rolling(window, min_periods=1).count()
    lam = np.clip(n_valid / window, 0.0, 1.0)

    beta_shrunk = lam * beta_raw + (1.0 - lam) * beta_prior
    beta_clipped = np.clip(beta_shrunk, beta_min, beta_max)

    # Time-shift: use beta estimated up to t-1 → shift(1)
    result: pd.Series = beta_clipped.shift(1).astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# Phase B: Idiosyncratic overlay
# ---------------------------------------------------------------------------

def _compute_idio_stress(
    r_sym: pd.Series,
    r_anchor: pd.Series,
    beta_lagged: pd.Series,
    close: pd.Series,
    w2: int,
    w3: int,
    w4: int,
    max_cut: float,
) -> tuple[pd.Series, pd.Series]:
    """Compute idiosyncratic stress and multiplier.

    Algorithm::

        eps_t     = r_i,t - beta_{t-1} * r_b,t      (beta already shifted)
        sigma_t   = rolling_std(eps, W2).shift(1)
        z_idio    = (sigma_t - mean(sigma, W3)) / (std(sigma, W3) + eps)
        dd_t      = close / rolling_max(close, W4).shift(1) - 1  (<=0)
        stress    = clip(0.6*sigmoid((z-1.5)*2) + 0.4*clip(-dd/0.25, 0, 1), 0, 1)
        mult      = 1.0 - stress * max_cut    in [1-max_cut, 1.0]

    Args:
        r_sym: Symbol 1h log returns.
        r_anchor: Anchor 1h log returns.
        beta_lagged: β̃ already shifted by 1 (from Phase A).
        close: Symbol closing prices.
        w2: Short vol estimation window.
        w3: Long z-score normalisation window.
        w4: Rolling max window for drawdown.
        max_cut: Maximum idio damping factor (0.5 → mult ∈ [0.5, 1.0]).

    Returns:
        Tuple of (idio_stress Series, idio_mult Series).

    Note:
        Space complexity: O(N) -- no intermediate array accumulation.

    """
    eps = 1e-12

    # Residual (β already lagged, so no look-ahead)
    residual = r_sym - beta_lagged * r_anchor

    # Rolling idio vol — shift(1) to prevent look-ahead
    sigma_eps = residual.rolling(w2, min_periods=2).std().shift(1)

    # Z-score normalisation over long window
    mu_sigma = sigma_eps.rolling(w3, min_periods=2).mean()
    sd_sigma = sigma_eps.rolling(w3, min_periods=2).std()
    z_idio = (sigma_eps - mu_sigma) / (sd_sigma + eps)

    # Drawdown from rolling max — shift(1) to avoid look-ahead
    rolling_peak = close.rolling(w4, min_periods=1).max().shift(1)
    dd = close / (rolling_peak + eps) - 1.0  # ≤ 0

    # Composite stress
    vol_term = 0.6 * _sigmoid((z_idio.to_numpy(dtype=np.float64) - 1.5) * 2.0)
    dd_arr = dd.to_numpy(dtype=np.float64)
    dd_term = 0.4 * np.clip(-dd_arr / 0.25, 0.0, 1.0)

    stress_arr = np.clip(vol_term + dd_term, 0.0, 1.0)
    stress_arr = np.nan_to_num(stress_arr, nan=0.0)

    mult_arr = 1.0 - stress_arr * max_cut  # ∈ [1-max_cut, 1.0]
    mult_arr = np.clip(mult_arr, 1.0 - max_cut, 1.0)

    idx = r_sym.index
    return (
        pd.Series(stress_arr, index=idx, dtype=np.float64),
        pd.Series(mult_arr, index=idx, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_symbol_overlay(
    sym_1h: pd.DataFrame,
    anchor_1h: pd.DataFrame,
    dt_grid: pd.Series,
    cfg: dict[str, Any],
    is_anchor: bool = False,
) -> pd.DataFrame:
    """Compute per-symbol beta + idiosyncratic overlay aligned to dt_grid.

    Args:
        sym_1h: Symbol 1h OHLCV (must have 'datetime' and 'close' columns).
        anchor_1h: Anchor (BTC) 1h OHLCV.
        dt_grid: market_probs["datetime"] Series — output alignment target.
        cfg: Configuration dict (uses FUTURES_BETA_* and FUTURES_IDIO_* keys).
        is_anchor: If True, return neutral overlay immediately (beta=1, stress=0, mult=1).

    Returns:
        DataFrame with columns [beta, idio_stress, idio_mult], len == len(dt_grid),
        index reset to RangeIndex.

    Note:
        All intermediate values are NaN-guarded. Insufficient history triggers
        Vasicek shrinkage to beta_prior (1.0). Alignment uses merge_asof
        (backward fill) to map sym_1h data onto dt_grid without look-ahead.

    """
    n = len(dt_grid)

    # --- neutral sentinel (anchor or trivial cases) -------------------------
    def _neutral() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "beta": np.ones(n, dtype=np.float64),
                "idio_stress": np.zeros(n, dtype=np.float64),
                "idio_mult": np.ones(n, dtype=np.float64),
            }
        )

    if is_anchor:
        return _neutral()

    # --- config extraction --------------------------------------------------
    W: int = int(cfg.get("FUTURES_BETA_WINDOW", 240))
    beta_min: float = float(cfg.get("FUTURES_BETA_MIN", 0.3))
    beta_max: float = float(cfg.get("FUTURES_BETA_MAX", 4.0))
    beta_prior: float = float(cfg.get("FUTURES_BETA_PRIOR", 1.0))
    W2: int = int(cfg.get("FUTURES_IDIO_VOL_WINDOW", 72))
    W3: int = int(cfg.get("FUTURES_IDIO_ZHIST_WINDOW", 480))
    W4: int = int(cfg.get("FUTURES_IDIO_DD_WINDOW", 168))
    max_cut: float = float(cfg.get("FUTURES_IDIO_MAX_CUT", 0.5))

    # --- validate inputs ----------------------------------------------------
    required_cols = {"datetime", "close"}
    if sym_1h is None or not required_cols.issubset(set(sym_1h.columns)):
        _logger.warning(
            "compute_symbol_overlay: sym_1h missing required columns; returning neutral."
        )
        return _neutral()
    if anchor_1h is None or not required_cols.issubset(set(anchor_1h.columns)):
        _logger.warning(
            "compute_symbol_overlay: anchor_1h missing required columns; returning neutral."
        )
        return _neutral()
    if len(sym_1h) < 4:
        return _neutral()

    # --- prepare symbol data ------------------------------------------------
    sym_data = sym_1h[["datetime", "close"]].copy()
    sym_data["datetime"] = pd.to_datetime(sym_data["datetime"], utc=True)
    sym_data = sym_data.sort_values("datetime").drop_duplicates("datetime")

    anc_data = anchor_1h[["datetime", "close"]].copy()
    anc_data = anc_data.rename(columns={"close": "close_anc"})
    anc_data["datetime"] = pd.to_datetime(anc_data["datetime"], utc=True)
    anc_data = anc_data.sort_values("datetime").drop_duplicates("datetime")

    # Merge sym + anchor on sym's timeline (inner-like: sym is left)
    merged = pd.merge_asof(
        sym_data,
        anc_data,
        on="datetime",
        direction="backward",
    )
    merged = merged.set_index("datetime")

    close_sym = merged["close"].astype(np.float64)
    close_anc = merged["close_anc"].astype(np.float64)

    r_sym = _log_returns(close_sym)
    r_anc = _log_returns(close_anc)

    # --- Phase A: rolling beta ----------------------------------------------
    beta_lagged = _compute_rolling_beta(r_sym, r_anc, W, beta_min, beta_max, beta_prior)
    beta_lagged = beta_lagged.reindex(merged.index)
    beta_lagged = beta_lagged.fillna(beta_prior)

    # --- Phase B: idiosyncratic stress -------------------------------------
    idio_stress, idio_mult = _compute_idio_stress(
        r_sym, r_anc, beta_lagged, close_sym, W2, W3, W4, max_cut
    )

    # --- Build per-symbol raw result ----------------------------------------
    raw = pd.DataFrame(
        {
            "datetime": merged.index,
            "beta": beta_lagged.to_numpy(dtype=np.float64),
            "idio_stress": idio_stress.to_numpy(dtype=np.float64),
            "idio_mult": idio_mult.to_numpy(dtype=np.float64),
        }
    ).reset_index(drop=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)

    # --- Align to dt_grid using merge_asof (backward fill → no look-ahead) --
    grid_df = pd.DataFrame({"datetime": pd.to_datetime(dt_grid.values, utc=True)})
    grid_df = grid_df.sort_values("datetime").reset_index(drop=True)

    aligned = pd.merge_asof(
        grid_df,
        raw.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )

    # Fill NaN with neutral values
    aligned["beta"] = aligned["beta"].fillna(beta_prior)
    aligned["idio_stress"] = aligned["idio_stress"].fillna(0.0)
    aligned["idio_mult"] = aligned["idio_mult"].fillna(1.0)

    # Final guard: nan_to_num + clip
    result = pd.DataFrame(
        {
            "beta": np.clip(
                np.nan_to_num(aligned["beta"].to_numpy(dtype=np.float64), nan=beta_prior),
                beta_min,
                beta_max,
            ),
            "idio_stress": np.clip(
                np.nan_to_num(aligned["idio_stress"].to_numpy(dtype=np.float64), nan=0.0),
                0.0,
                1.0,
            ),
            "idio_mult": np.clip(
                np.nan_to_num(aligned["idio_mult"].to_numpy(dtype=np.float64), nan=1.0),
                1.0 - max_cut,
                1.0,
            ),
        }
    )

    if len(result) != n:
        _logger.warning(
            "compute_symbol_overlay: alignment mismatch (expected %d, got %d); returning neutral.",
            n,
            len(result),
        )
        return _neutral()

    return result.reset_index(drop=True)


def apply_symbol_overlay(
    base_mod: pd.DataFrame,
    overlay: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Apply beta scaling + idio_mult to base HMM modulator.

    Asymmetric beta scaling (Phase A):
        excess = mod_btc - 1.0
        if excess < 0:  mod_i = 1.0 + excess * β̃   (risk-off: high-β cut deeper)
        if excess >= 0: mod_i = 1.0 + excess / β̃   (risk-on: high-β sizing damped)
        clipped to [0.0, 2.5]

    Idiosyncratic overlay (Phase B):
        final_long  = beta_scaled_long * idio_mult   (long only — overlay tightens)
        final_short = beta_scaled_short              (short unchanged)

    Args:
        base_mod: Output of _hmm_modulator_kelly_values. Must contain
                  'hmm_modulator_long' and 'hmm_modulator_short'.
        overlay: Output of compute_symbol_overlay with columns [beta, idio_stress, idio_mult].
        cfg: Configuration dict (currently unused; reserved for future tuning knobs).

    Returns:
        DataFrame with same columns as base_mod plus 'idio_stress'.
        hmm_modulator_long and hmm_modulator_short are updated in-place semantics
        (a copy is returned; base_mod is not mutated).

    Note:
        beta=1 anchor case → mod_i = mod_btc (invariant), because
        excess < 0:  1 + excess * 1 = mod_btc
        excess >= 0: 1 + excess / 1 = mod_btc

    """
    result = base_mod.copy()
    n = len(result)

    if overlay is None or len(overlay) != n:
        _logger.warning(
            "apply_symbol_overlay: overlay length mismatch (base=%d, overlay=%d); skipping.",
            n,
            0 if overlay is None else len(overlay),
        )
        result["idio_stress"] = 0.0
        return result

    beta_arr = np.nan_to_num(overlay["beta"].to_numpy(dtype=np.float64), nan=1.0)
    beta_arr = np.clip(beta_arr, 1e-6, None)  # guard division by zero in risk-on branch

    idio_mult_arr = np.nan_to_num(overlay["idio_mult"].to_numpy(dtype=np.float64), nan=1.0)
    idio_stress_arr = np.nan_to_num(overlay["idio_stress"].to_numpy(dtype=np.float64), nan=0.0)

    mod_long_btc = result["hmm_modulator_long"].to_numpy(dtype=np.float64)
    mod_short_btc = result["hmm_modulator_short"].to_numpy(dtype=np.float64)

    # --- Phase A: asymmetric beta scaling -----------------------------------
    excess_long = mod_long_btc - 1.0
    excess_short = mod_short_btc - 1.0

    # Long scaling
    beta_long = np.where(
        excess_long < 0.0,
        1.0 + excess_long * beta_arr,       # de-risk → amplify cut
        1.0 + excess_long / beta_arr,       # risk-on → dampen
    )
    beta_long = np.clip(beta_long, 0.0, 2.5)

    # Short scaling
    beta_short = np.where(
        excess_short < 0.0,
        1.0 + excess_short * beta_arr,
        1.0 + excess_short / beta_arr,
    )
    beta_short = np.clip(beta_short, 0.0, 2.5)

    # --- Phase B: idio_mult on long only ------------------------------------
    final_long = np.clip(beta_long * idio_mult_arr, 0.0, 2.5)
    final_short = beta_short  # short side unchanged

    result["hmm_modulator_long"] = final_long
    result["hmm_modulator_short"] = final_short
    result["idio_stress"] = idio_stress_arr
    # Preserve beta for downstream per-symbol metric computation (Step 5).
    result["beta"] = beta_arr

    return result
