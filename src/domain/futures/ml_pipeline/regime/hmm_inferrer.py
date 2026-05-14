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
from scipy.special import expit

from config.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.ml_pipeline.regime.crisis_detector import detect_crisis_components
from src.domain.futures.ml_pipeline.regime.jax_hmm import JAXMultivariateHMM
from src.core.indicators.indicators import global_ind

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_logger = logging.getLogger(__name__)

HMM_SEMANTIC_PROB_COLUMNS: tuple[str, ...] = (
    "hmm_prob_bull_calm",
    "hmm_prob_bull_vol_up",
    "hmm_prob_bear_trend",
    "hmm_prob_chop",
    "hmm_prob_crisis",
)
_SEMANTIC_ORDER: list[str] = list(HMM_SEMANTIC_PROB_COLUMNS)
HMM_AUX_CRISIS_PROB_COLUMNS: tuple[str, str] = (
    "hmm_prob_pre_crisis",
    "hmm_prob_realized_crisis",
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _normalized_entropy_k(probs: np.ndarray, k: int) -> np.ndarray:
    """Normalised Shannon entropy for a probability matrix."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    result: np.ndarray = (-np.sum(p * np.log(p), axis=1) / np.log(float(k))).astype(np.float64)
    return result


@njit(cache=True, fastmath=True)
def _numba_sticky_labels(labels: np.ndarray, min_durations: np.ndarray) -> np.ndarray:
    """Asymmetric min-duration constraint to reduce regime flip noise."""
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
    """Compute run-length (current duration) for each bar."""
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
# Track A + B Mapping
# ---------------------------------------------------------------------------

def _map_jax_to_five_state(
    jax_probs: pd.DataFrame,
    crisis_comps: pd.DataFrame,
    f2_vol_z: pd.Series,
) -> pd.DataFrame:
    """Map 4-state JAX HMM + Crisis Detector into 5 semantic columns.

    Track A states: bull_trend, bear_trend, chop_high, chop_low
    Track B: crisis_score
    """
    idx = jax_probs.index
    p_bull = jax_probs["bull_trend"]
    p_bear = jax_probs["bear_trend"]
    p_chop = jax_probs["chop_high"] + jax_probs["chop_low"]
    
    # Track B: Crisis Score
    p_crisis = crisis_comps["crisis_score"].reindex(idx).fillna(0.0)
    
    # Split BULL_TREND using Volatility Z-score (f2)
    # Sigmoid(f2) gives weight to bull_vol_up, Sigmoid(-f2) to bull_calm
    vol_weight = pd.Series(expit(f2_vol_z.values), index=idx)
    p_bull_vol_up = p_bull * vol_weight
    p_bull_calm = p_bull * (1.0 - vol_weight)
    
    # Blend Track B (Crisis)
    # We follow the same logic as before: Track B is an overlay.
    # Sigma (bull_calm, bull_vol_up, bear_trend, chop) = 1 - p_crisis
    target_noncr = (1.0 - p_crisis).clip(0.0, 1.0)
    current_sum = p_bull_calm + p_bull_vol_up + p_bear + p_chop + 1e-12
    scale = target_noncr / current_sum
    
    return pd.DataFrame({
        "hmm_prob_bull_calm": (p_bull_calm * scale).clip(0.0, 1.0),
        "hmm_prob_bull_vol_up": (p_bull_vol_up * scale).clip(0.0, 1.0),
        "hmm_prob_bear_trend": (p_bear * scale).clip(0.0, 1.0),
        "hmm_prob_chop": (p_chop * scale).clip(0.0, 1.0),
        "hmm_prob_crisis": p_crisis.clip(0.0, 1.0),
    }, index=idx)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class HMMStateInferrer:
    """Next-Gen Regime Model Refactoring (v10.0 - JAX Native).

    Track A: 4-State JAX Multivariate HMM
    Track B: Rule-based Crisis Detector Overlay
    """

    n_states: int = 5
    n_iter: int = 1500
    tol: float = 1e-4
    smoothing_method: str = "EMA"
    smoothing_span: int = 8
    sticky_min_duration: tuple[int, int, int, int, int] = (24, 12, 8, 10, 4)
    crisis_attack_span: int | None = None
    crisis_decay_span: int | None = None
    _jax_model: JAXMultivariateHMM = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.smoothing_method = str(self.smoothing_method or "EMA").upper()
        self.smoothing_span = int(max(1, self.smoothing_span))
        self.sticky_min_duration = tuple(max(1, int(v)) for v in self.sticky_min_duration)
        self._jax_model = JAXMultivariateHMM(n_iter=self.n_iter, tol=self.tol)

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "4h",
    ) -> pd.DataFrame:
        """v10.0 implementation of regime inference."""
        _logger.info("🧠 HMM v10.0 | %s %s | %d bars", symbol, tf, len(features_df))

        n_orig = len(features_df)
        if n_orig < 200:
            return self._zeros_semantic(features_df)

        orig_idx = features_df.index
        # ── 1. Feature Engineering (f1, f2) ──────────────────────────────────
        # f1: Trend (Rolling Z-score of EMA(12)/EMA(144)-1)
        close = features_df["close"] if "close" in features_df.columns else returns_ser.cumsum() # Fallback
        ema12 = global_ind.calculate_ema(close, 12)
        ema144 = global_ind.calculate_ema(close, 144)
        f1_raw = (ema12 / ema144) - 1.0
        f1_z = (f1_raw - f1_raw.rolling(168).mean()) / f1_raw.rolling(168).std().replace(0, 0.001)
        
        # f2: Volatility (Rolling Z-score of Log(ATR(14)/Close))
        if "high" in features_df.columns and "low" in features_df.columns:
            atr14 = global_ind.calculate_atr(features_df, 14)
        else:
            # Fallback for OHLC lack
            atr14 = returns_ser.abs().rolling(14).mean() * close
            
        f2_raw = np.log(atr14 / close.replace(0, 0.001)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        f2_z = (f2_raw - f2_raw.rolling(168).mean()) / f2_raw.rolling(168).std().replace(0, 0.001)
        
        obs_df = pd.DataFrame({"f1": f1_z, "f2": f2_z}, index=orig_idx).fillna(0.0)

        # ── 2. Track A: JAX Multivariate HMM ─────────────────────────────────
        jax_probs = self._jax_model.fit_predict(obs_df)

        # ── 3. Track B: Crisis Detector Overlay ──────────────────────────────
        # We need vol_probs and dir_probs for legacy crisis detector, 
        # so we map JAX states to temporary vol/dir proxies.
        vol_proxies = pd.DataFrame({
            "vol_low": jax_probs["bull_trend"] + jax_probs["bear_trend"] + jax_probs["chop_low"],
            "vol_mid": jax_probs["chop_high"] * 0.5,
            "vol_high": jax_probs["chop_high"] * 0.5,
        }, index=orig_idx).fillna(0.33)
        dir_proxies = pd.DataFrame({
            "dir_bull": jax_probs["bull_trend"],
            "dir_range": jax_probs["chop_high"] + jax_probs["chop_low"],
            "dir_bear": jax_probs["bear_trend"],
        }, index=orig_idx).fillna(0.33)

        crisis_comps = detect_crisis_components(
            features_df, returns_ser, vol_proxies, dir_proxies
        )

        # ── 4. Final Mapping ─────────────────────────────────────────────────
        result = _map_jax_to_five_state(jax_probs, crisis_comps, f2_z)
        
        # Skip post-hoc smoothing if configured (Specification: "Remove post-hoc EMA probability smoothing")
        # But we keep it if smoothing_span > 1 for backward compatibility unless explicitly disabled.
        # Given the spec, I'll set a flag or just respect the config.
        # Actually, let's just use what's in the result.
        
        # ── Auxiliary columns ────────────────────────────────────────────────
        result["hmm_prob_pre_crisis"] = crisis_comps["pre_crisis_score"].reindex(result.index).fillna(0.0)
        result["hmm_prob_realized_crisis"] = crisis_comps["realized_crisis_score"].reindex(result.index).fillna(0.0)
        result["hmm_prob_bull_trend"] = jax_probs["bull_trend"]
        
        prob_mat = result[_SEMANTIC_ORDER].to_numpy(dtype=np.float64)
        result["hmm_entropy"] = _normalized_entropy_k(prob_mat, len(_SEMANTIC_ORDER))

        hard_states_raw = np.argmax(prob_mat, axis=1).astype(np.int32)
        _DUR_CFG = np.array(self.sticky_min_duration, dtype=np.int32)
        sticky_hard = _numba_sticky_labels(hard_states_raw, _DUR_CFG)
        result["hmm_hard_state"] = sticky_hard.astype(np.float64)
        result["hmm_current_duration"] = _numba_current_duration(sticky_hard)
        result["hmm_expected_duration"] = 1.0 / (1.0 - 0.96) # Default sticky prior

        result = result.ffill().bfill().fillna(0.0)
        result = result.reset_index().rename(columns={result.index.name or "index": "datetime"})

        _logger.info("✅ HMM v10.0 done | crisis=%.1f%%", 100.0 * float(result["hmm_prob_crisis"].mean()))
        return result

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return uniform-prior output when there is insufficient data."""
        u = 1.0 / float(len(_SEMANTIC_ORDER))
        cols: list[str] = [
            *_SEMANTIC_ORDER,
            *HMM_AUX_CRISIS_PROB_COLUMNS,
            "hmm_entropy",
            "hmm_expected_duration",
            "hmm_current_duration",
            "hmm_hard_state",
            "hmm_prob_bull_trend",
        ]
        out = pd.DataFrame(np.zeros((len(df), len(cols))), index=df.index, columns=cols)
        for c in _SEMANTIC_ORDER: out[c] = u
        out["hmm_prob_bull_trend"] = 2*u
        out["hmm_entropy"] = 1.0
        return out.reset_index().rename(columns={out.index.name or "index": "datetime"})

def build_hmm_inferrer_from_config(cfg: dict[str, object] | None = None, **kwargs) -> HMMStateInferrer:
    conf = OPT_FUTURES_CONFIG if cfg is None else cfg
    sticky_raw = conf.get("FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION", [24, 12, 8, 10, 4])
    try:
        sticky = tuple(int(v) for v in sticky_raw)
    except Exception:
        sticky = (24, 12, 8, 10, 4)
    return HMMStateInferrer(
        n_iter=int(conf.get("FUTURES_HMM_N_ITER", 1500)),
        sticky_min_duration=sticky,
    )

