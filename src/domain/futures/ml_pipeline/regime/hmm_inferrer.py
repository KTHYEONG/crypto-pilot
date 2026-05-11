"""3-Layer Hierarchical Regime Classifier (v9.0) — Orchestrator / Facade.

Architecture:
    Layer 1: MS-GARCH Vol Regime  (3 states, 12h — vol_regime.py)
    Layer 2: Direction HMM        (3 states, 6h  — dir_regime.py)
    Layer 3: Crisis Detector      (rules-based   — crisis_detector.py)

External output: 5 semantic probability columns (HMM_SEMANTIC_PROB_COLUMNS)
    + auxiliary columns (hmm_entropy, hmm_expected_duration, hmm_current_duration,
                         hmm_hard_state, hmm_prob_bull_trend, datetime).

Public interface (backward-compatible):
    HMMStateInferrer.fit_predict_systemic(features_df, returns_ser, is_end_idx,
                                          symbol, tf) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from src.domain.futures.ml_pipeline.features.engineering import (
    HMM_SEMANTIC_PROB_COLUMNS,
)
from src.domain.futures.ml_pipeline.regime.crisis_detector import detect_crisis
from src.domain.futures.ml_pipeline.regime.dir_regime import DirRegimeModel
from src.domain.futures.ml_pipeline.regime.vol_regime import VolRegimeModel

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_logger = logging.getLogger(__name__)

_SEMANTIC_ORDER: list[str] = list(HMM_SEMANTIC_PROB_COLUMNS)

# ---------------------------------------------------------------------------
# Utility functions (re-used from legacy, kept for output compatibility)
# ---------------------------------------------------------------------------


def _normalized_entropy_k(probs: np.ndarray, k: int) -> np.ndarray:
    """Normalised Shannon entropy for a probability matrix.

    Args:
        probs: (T, K) probability array (rows sum to ~1).
        k:     Number of states (for normalisation).

    Returns:
        (T,) array with values in [0, 1].

    Time complexity: O(T * K).

    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    result: np.ndarray = (-np.sum(p * np.log(p), axis=1) / np.log(float(k))).astype(np.float64)
    return result


@njit(cache=True, fastmath=True)
def _numba_sticky_labels(labels: np.ndarray, min_durations: np.ndarray) -> np.ndarray:
    """Asymmetric min-duration constraint to reduce regime flip noise.

    Time complexity: O(T).
    """
    n = len(labels)
    if n < 2:
        return labels
    result = labels.copy()
    i = 0
    while i < n:
        curr = result[i]
        j = i + 1
        while j < n and result[j] == curr:
            j += 1
        run_len = j - i
        m_dur = min_durations[int(curr)]
        if run_len < m_dur and i > 0:
            prev = result[i - 1]
            for kk in range(i, j):
                result[kk] = prev
        i = j
    return result


@njit(cache=True, fastmath=True)
def _numba_current_duration(hard_states: np.ndarray) -> np.ndarray:
    """Compute run-length (current duration) for each bar.

    Time complexity: O(T).
    """
    n = len(hard_states)
    dur_arr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return dur_arr
    c = 1.0
    dur_arr[0] = c
    for i in range(1, n):
        if hard_states[i] == hard_states[i - 1]:
            c += 1.0
        else:
            c = 1.0
        dur_arr[i] = c
    return dur_arr


# ---------------------------------------------------------------------------
# 5-State output mapping
# ---------------------------------------------------------------------------

def _build_five_state_probs(
    vol_probs: pd.DataFrame,
    dir_probs: pd.DataFrame,
    crisis_prob: pd.Series,
    returns_ser: pd.Series | None = None,
    liq_proxy: pd.Series | None = None,
) -> pd.DataFrame:
    """Combine Layer 1+2+3 into 5 semantic probability columns.

    Mapping:
        bull_calm   = p_low  * p_bull
        bull_vol_up = (p_mid + p_high) * p_bull
        bear_trend  = (p_mid + p_high) * p_bear
        chop        = p_low * p_range + p_mid * p_range
        crisis_base = p_high * p_bear

    Layer 3 blend:
        crisis_final = crisis_base * (1 - blend) + blend
        scale others down so Σ = 1.

    Args:
        vol_probs:   (T, 3) — vol_low, vol_mid, vol_high (Layer 1).
        dir_probs:   (T, 3) — dir_bull, dir_range, dir_bear (Layer 2).
        crisis_prob: (T,)   — [0, 1] soft crisis score (Layer 3).
        returns_ser: Raw returns (for Capitulation Bypass).
        liq_proxy:   Liquidity proxy (for Capitulation Bypass).

    Returns:
        DataFrame with columns = HMM_SEMANTIC_PROB_COLUMNS.

    Time complexity: O(T).

    """
    p_low = vol_probs["vol_low"]
    p_mid = vol_probs["vol_mid"]
    p_high = vol_probs["vol_high"]
    p_bull = dir_probs["dir_bull"]
    p_range = dir_probs["dir_range"]
    p_bear = dir_probs["dir_bear"]

    bull_calm: pd.Series = p_low * p_bull
    bull_vol_up: pd.Series = (p_mid + p_high) * p_bull
    bear_trend: pd.Series = p_mid * p_bear
    chop: pd.Series = p_low * p_range + p_mid * p_range
    crisis_base: pd.Series = (p_high * p_bear * 1.8 + p_mid * p_bear * 0.3).clip(0.0, 0.50)

    # Layer 3 override: blend crisis_prob into crisis_final
    blend_factor = crisis_prob.clip(0.0, 1.0)
    crisis_final: pd.Series = crisis_base * (1.0 - blend_factor) + blend_factor

    # ── Capitulation Bypass (NEW for Step 2) ─────────────────────────────────
    if returns_ser is not None and liq_proxy is not None:
        idx = vol_probs.index
        ret = returns_ser.reindex(idx).fillna(0.0)
        liq = liq_proxy.reindex(idx).fillna(0.0)
        roll_std = ret.rolling(168, min_periods=24).std().fillna(float(ret.std()))
        p995 = float(liq.quantile(0.995))

        bypass_mask = (ret < -3.5 * roll_std) & (liq > p995)
        if bypass_mask.any():
            _logger.debug("⚡ Capitulation bypass | %d bars", int(bypass_mask.sum()))
            # Cap crisis at 0.1, distribute delta proportionally to chop and bear_trend
            old_crisis = crisis_final.copy()
            crisis_final = pd.Series(
                np.where(bypass_mask, np.minimum(crisis_final, 0.1), crisis_final),
                index=idx
            )
            delta = (old_crisis - crisis_final).clip(0.0, 1.0)
            
            total_def_probs = chop + bear_trend + 1e-8
            chop = chop + (delta * (chop / total_def_probs))
            bear_trend = bear_trend + (delta * (bear_trend / total_def_probs))

    # Re-scale non-crisis channels so the 5 probs sum to 1
    raw_sum = bull_calm + bull_vol_up + bear_trend + chop + 1e-8
    target_noncr = (1.0 - crisis_final).clip(0.0, 1.0)
    scale = target_noncr / raw_sum

    bull_calm = (bull_calm * scale).clip(0.0, 1.0)
    bull_vol_up = (bull_vol_up * scale).clip(0.0, 1.0)
    bear_trend = (bear_trend * scale).clip(0.0, 1.0)
    chop = (chop * scale).clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "hmm_prob_bull_calm": bull_calm.values,
            "hmm_prob_bull_vol_up": bull_vol_up.values,
            "hmm_prob_bear_trend": bear_trend.values,
            "hmm_prob_chop": chop.values,
            "hmm_prob_crisis": crisis_final.values,
        },
        index=vol_probs.index,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class HMMStateInferrer:
    """3-Layer Hierarchical Regime Classifier (v9.0).

    Layer 1: MS-GARCH Vol Regime (3 states, 12h)
    Layer 2: Direction HMM (3 states, 6h)
    Layer 3: Crisis Detector (rules-based)

    Output: 5-state semantic mapping aligned to HMM_SEMANTIC_PROB_COLUMNS.

    Attributes:
        n_states: Kept for interface compatibility (internally ignored).
        n_iter:   Max optimisation iterations for Layer 1 and 2.
        tol:      Convergence tolerance for early stopping.

    """

    n_states: int = 5
    n_iter: int = 1500
    tol: float = 1e-4
    _vol_model: VolRegimeModel = field(init=False, repr=False)
    _dir_model: DirRegimeModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise Layer 1 and Layer 2 sub-models."""
        self._vol_model = VolRegimeModel(n_iter=self.n_iter, tol=self.tol)
        self._dir_model = DirRegimeModel(n_iter=min(self.n_iter, 800), tol=self.tol)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "4h",
    ) -> pd.DataFrame:
        """Run 3-Layer inference and produce backward-compatible 5-state output.

        Args:
            features_df: DatetimeIndex DataFrame, cols = SYSTEMIC_HMM_FEATURE_COLUMNS.
            returns_ser: Returns series aligned to features_df.index.
            is_end_idx:  Last in-sample bar index (unused internally; kept for API compat).
            symbol:      Log annotation only.
            tf:          Original timeframe string ("4h", "1h", etc.).

        Returns:
            DataFrame with:
              - hmm_prob_bull_calm, hmm_prob_bull_vol_up, hmm_prob_bear_trend,
                hmm_prob_chop, hmm_prob_crisis
              - hmm_entropy, hmm_expected_duration, hmm_current_duration
              - hmm_hard_state, hmm_prob_bull_trend
              - datetime  (reset-indexed column)

        Time complexity: O(n_windows_vol * n_iter * T/12 * 9) + O(n_windows_dir * n_iter * T/6 * 9).

        """
        _logger.info("🧠 HMM | %s %s | %d bars", symbol, tf, len(features_df))

        n_orig = len(features_df)
        if n_orig < 200:
            _logger.warning("Too few bars (%d < 200), returning uniform priors.", n_orig)
            return self._zeros_semantic(features_df)

        orig_features_df = features_df.copy()

        # ── Normalise TF: resample sub-4h inputs to 4h for stability ────────
        is_fast_tf = tf in ("1h", "15m", "5m", "1m")
        if is_fast_tf:
            _logger.debug("⏩ TF %s → 4h resample", tf)
            features_df = features_df.resample("4h").last().ffill()
            returns_ser = returns_ser.resample("4h").apply(
                lambda x: (1.0 + x).prod() - 1.0
            )

        # ── Layer 1: Vol Regime (12h) ────────────────────────────────────────
        features_12h = features_df.resample("12h").last().ffill()
        returns_12h = returns_ser.resample("12h").apply(
            lambda x: (1.0 + x).prod() - 1.0
        )
        _logger.debug("  L1 MS-GARCH | %d 12h bars", len(features_12h))
        vol_probs_12h = self._vol_model.fit_predict(features_12h, returns_12h, is_end_idx)

        # Reindex vol probs back to original (4h or input) TF
        vol_probs_orig = vol_probs_12h.reindex(features_df.index).ffill().bfill()

        # ── Layer 2: Direction HMM (6h) ──────────────────────────────────────
        features_6h = features_df.resample("6h").last().ffill()
        returns_6h = returns_ser.resample("6h").apply(
            lambda x: (1.0 + x).prod() - 1.0
        )
        # Vol-normalise returns with Layer 1 posterior-weighted sigma
        vol_probs_6h = vol_probs_12h.reindex(features_6h.index).ffill().bfill()
        norm_ret_6h = self._compute_norm_returns(returns_6h, vol_probs_6h)

        _logger.debug("  L2 Dir HMM | %d 6h bars", len(features_6h))
        dir_probs_6h = self._dir_model.fit_predict(norm_ret_6h, vol_probs_6h)

        dir_probs_orig = dir_probs_6h.reindex(features_df.index).ffill().bfill()

        # ── Layer 3: Crisis Detector ─────────────────────────────────────────
        _logger.debug("  L3 Crisis | %d bars", len(features_df))
        crisis_orig = detect_crisis(features_df, returns_ser, vol_probs_orig, dir_probs_orig)

        # ── 5-State Mapping ──────────────────────────────────────────────────
        liq_proxy = features_df["macro_liq_proxy_24h"] if "macro_liq_proxy_24h" in features_df.columns else None
        result = _build_five_state_probs(
            vol_probs_orig,
            dir_probs_orig,
            crisis_orig,
            returns_ser=returns_ser,
            liq_proxy=liq_proxy,
        )

        # ── Auxiliary columns ────────────────────────────────────────────────
        result["hmm_prob_bull_trend"] = (
            result["hmm_prob_bull_calm"] + result["hmm_prob_bull_vol_up"]
        )

        prob_mat = result[_SEMANTIC_ORDER].to_numpy(dtype=np.float64)
        result["hmm_entropy"] = _normalized_entropy_k(prob_mat, len(_SEMANTIC_ORDER))

        # Hard state: argmax of 5-state probs
        hard_states_raw = np.argmax(prob_mat, axis=1).astype(np.int32)

        # Apply sticky min-duration (BULL_CALM=24, BULL_VOL_UP=12, BEAR=6, CHOP=8, CRISIS=4)
        _DUR_CFG = np.array([24, 12, 8, 10, 4], dtype=np.int32)
        sticky_hard = _numba_sticky_labels(hard_states_raw, _DUR_CFG)
        result["hmm_hard_state"] = sticky_hard.astype(np.float64)
        result["hmm_current_duration"] = _numba_current_duration(sticky_hard)

        # Expected duration from Layer 1 vol transition diagonal (as proxy)
        result["hmm_expected_duration"] = self._compute_expected_duration(
            vol_probs_orig, sticky_hard
        )

        # ── Reindex to original TF if sub-4h was pre-resampled ───────────────
        if is_fast_tf:
            result = result.reindex(orig_features_df.index).ffill().bfill()

        # ── Output format: reset_index + datetime column ─────────────────────
        result = result.ffill().bfill().fillna(0.0)
        result = result.reset_index().rename(
            columns={result.index.name or "index": "datetime"}
        )

        _logger.info(
            "✅ HMM done | crisis=%.1f%%",
            100.0 * float(result["hmm_prob_crisis"].mean()),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_norm_returns(
        returns: pd.Series, vol_probs: pd.DataFrame
    ) -> pd.Series:
        """Compute vol-normalised returns using Layer 1 posterior-weighted sigma.

        sigma_t = sqrt(sum_k p_k * sigma2_k)
        where sigma2_LOW ~= 0.01^2, sigma2_MID ~= 0.02^2, sigma2_HIGH ~= 0.04^2
        (rough order-of-magnitude; GARCH actual values vary per training window).

        Args:
            returns:   Raw returns aligned to vol_probs.index.
            vol_probs: Layer 1 posteriors (vol_low, vol_mid, vol_high).

        Returns:
            Vol-normalised returns series (same index).

        """
        _SIGMA_REF = np.array([0.01, 0.02, 0.04], dtype=np.float64)  # LOW, MID, HIGH
        vol_arr = vol_probs[["vol_low", "vol_mid", "vol_high"]].to_numpy(dtype=np.float64)
        sigma_sq = vol_arr @ (_SIGMA_REF ** 2)
        sigma = np.sqrt(np.maximum(sigma_sq, 1e-8))
        ret_arr = returns.reindex(vol_probs.index).fillna(0.0).to_numpy(dtype=np.float64)
        norm = ret_arr / sigma
        return pd.Series(np.clip(norm, -10.0, 10.0), index=vol_probs.index)

    @staticmethod
    def _compute_expected_duration(
        vol_probs: pd.DataFrame,
        sticky_hard: np.ndarray,
    ) -> pd.Series:
        """Approximate expected duration from vol-regime persistence.

        Uses: E[D_s] = 1 / (1 - p_stay) where p_stay ≈ max posterior of dominant vol state.

        Args:
            vol_probs:   Layer 1 posteriors (vol_low, vol_mid, vol_high).
            sticky_hard: (T,) sticky hard state labels (5-state external).

        Returns:
            pd.Series of expected durations, same index as vol_probs.

        """
        vol_arr = vol_probs[["vol_low", "vol_mid", "vol_high"]].to_numpy(dtype=np.float64)
        # Use max posterior as rough self-transition probability proxy
        p_stay = np.max(vol_arr, axis=1)
        p_stay = np.clip(p_stay, 0.05, 0.95)
        expected_dur = 1.0 / (1.0 - p_stay)
        return pd.Series(expected_dur, index=vol_probs.index)

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return uniform-prior output when there is insufficient data."""
        u = 1.0 / float(len(_SEMANTIC_ORDER))
        cols: list[str] = [
            *_SEMANTIC_ORDER,
            "hmm_entropy",
            "hmm_expected_duration",
            "hmm_current_duration",
            "hmm_hard_state",
            "hmm_prob_bull_trend",
        ]
        out = pd.DataFrame(
            np.zeros((len(df), len(cols))), index=df.index, columns=cols
        )
        for c in _SEMANTIC_ORDER:
            out[c] = u
        out["hmm_prob_bull_trend"] = out["hmm_prob_bull_calm"] + out["hmm_prob_bull_vol_up"]
        out["hmm_entropy"] = 1.0
        return out.reset_index().rename(
            columns={out.index.name or "index": "datetime"}
        )
