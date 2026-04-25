"""Gaussian HMM regime probabilities with walk-forward refit and stable semantic labels."""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from numba import njit

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_CACHE_DIR
from src.domain.futures.ml_pipeline.feature_engineering import (
    HMM_SEMANTIC_PROB_COLUMNS,
    SYSTEMIC_HMM_FEATURE_COLUMNS,
)

warnings.filterwarnings("ignore", message=".*overwritten during initialization.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")

_logger = logging.getLogger(__name__)

_SEMANTIC_ORDER = list(HMM_SEMANTIC_PROB_COLUMNS)


def _bull_semantic_score(means: np.ndarray, state_i: int) -> float:
    """Composite bull score: trend strength (24h/168h) + volume energy (index 5)."""
    t24 = float(means[state_i, 0])
    t168 = float(means[state_i, 1])
    energy = float(means[state_i, 5])  # volume_momentum_24h
    br = float(means[state_i, 6])      # market_breadth
    
    # Structural stability (168h) prioritized over short-term (24h)
    composite_trend = 0.3 * t24 + 0.5 * t168 + 0.2 * energy
    secondary = float(np.clip(0.1 * br, -0.2, 0.2))
    return composite_trend + secondary


def _crisis_semantic_scores(means: np.ndarray) -> np.ndarray:
    """Crisis score: high vol + high downside risk + extreme negative distance (index 4)."""
    k = int(means.shape[0])
    out: np.ndarray = np.empty(k, dtype=np.float64)
    for i in range(k):
        t24 = float(means[i, 0])
        t168 = float(means[i, 1])
        vol = float(means[i, 2])
        down_ratio = float(means[i, 3])
        dist = float(means[i, 4])  # btc_ma_dist_168h (capitulation bottom marker)
        
        # Crisis = (Vol & Downside risk high) AND (Price far below long-term MA)
        out[i] = (vol + down_ratio) - (t24 + t168 + dist)
    return out


@njit  # type: ignore
def _winsorize_cols_numba(X: np.ndarray, pct: float = 0.01) -> np.ndarray:
    """Numba-accelerated winsorization for feature matrix (per-symbol HMM path)."""
    n_rows, n_cols = X.shape
    out = X.copy()
    for j in range(n_cols):
        col = X[:, j]
        lo_idx = int(n_rows * pct)
        hi_idx = int(n_rows * (1.0 - pct))
        if hi_idx >= n_rows:
            hi_idx = n_rows - 1
        tmp = np.sort(col)
        lo = tmp[lo_idx]
        hi = tmp[hi_idx]
        for i in range(n_rows):
            if out[i, j] < lo:
                out[i, j] = lo
            elif out[i, j] > hi:
                out[i, j] = hi
    return out


@njit  # type: ignore
def _calculate_ordered_probs_numba(
    state_seq: np.ndarray,
    lr: np.ndarray,
    last_p: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Sharpe-like ordering for legacy per-symbol HMM path."""
    scores: np.ndarray = np.zeros(n_states, dtype=np.float64)
    for s in range(n_states):
        count = 0
        total = 0.0
        for i in range(len(state_seq)):
            if state_seq[i] == s:
                total += lr[i]
                count += 1
        if count > 5:
            mean = total / count
            var = 0.0
            for i in range(len(state_seq)):
                if state_seq[i] == s:
                    var += (lr[i] - mean) ** 2
            var /= count
            std = np.sqrt(var) + 1e-12
            scores[s] = mean / std
        else:
            scores[s] = -1e9
    order = np.argsort(scores)
    reordered: np.ndarray = np.zeros(n_states, dtype=np.float64)
    for i in range(n_states):
        reordered[i] = last_p[order[i]]
    return reordered


def _robust_fit_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform robust scaling using Median and MAD.
    
    Returns:
        tuple: (scaled_matrix, median, mad)

    """
    med = np.asarray(np.median(X, axis=0), dtype=np.float64)
    mad = np.asarray(np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-12, dtype=np.float64)
    Xs = np.asarray((X - med) / mad, dtype=np.float64)
    Xs = np.clip(Xs, -10.0, 10.0)  # [FIX] Prevent FP underflow from extreme black swans
    return Xs, med, mad


def _robust_transform(X: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Apply robust scaling to new data using pre-computed median and MAD."""
    mad_safe = np.asarray(np.where(mad < 1e-12, 1.0, mad), dtype=np.float64)
    out = np.asarray((X - med) / mad_safe, dtype=np.float64)
    out = np.clip(out, -10.0, 10.0)  # [FIX] Prevent FP underflow
    return cast(np.ndarray, out)


def _sticky_transmat_prior(n_states: int, self_p: float = 0.9) -> np.ndarray:
    """Create a transition matrix prior with high self-transition probability."""
    off = (1.0 - self_p) / max(n_states - 1, 1)
    t_mat: np.ndarray = np.full((n_states, n_states), off, dtype=np.float64)
    np.fill_diagonal(t_mat, self_p)
    row_sums = np.asarray(t_mat.sum(axis=1)[:, np.newaxis], dtype=np.float64)
    return cast(np.ndarray, t_mat / row_sums)


def _blend_transition(
    T_fit: np.ndarray, n_states: int, alpha: float, self_p: float = 0.9
) -> np.ndarray:
    """Blend fitted transition matrix with a sticky prior."""
    T_prior = _sticky_transmat_prior(n_states, self_p=self_p)
    T_fit_safe = np.nan_to_num(T_fit, nan=1.0 / n_states)
    T_new = np.asarray((1.0 - alpha) * T_fit_safe + alpha * T_prior, dtype=np.float64)
    row_sums = np.asarray(T_new.sum(axis=1)[:, np.newaxis], dtype=np.float64)
    # Ensure no division by zero
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    return cast(np.ndarray, T_new / row_sums)


def _assign_state_semantic_labels(means: np.ndarray, X_ref: np.ndarray) -> dict[int, str]:
    """Map raw HMM state index → semantic label using structural trend and risk markers."""
    k = int(means.shape[0])
    crisis_scores = _crisis_semantic_scores(means)
    crisis_s = int(np.argmax(crisis_scores))
    
    rem = [i for i in range(k) if i != crisis_s]
    
    # Primary sort by long-term 168h trend (index 1)
    if len(rem) == 3:
        # Sort by composite bull score (which considers 24h and 168h)
        sorted_rem = sorted(rem, key=lambda x: _bull_semantic_score(means, x))
        bear_s = int(sorted_rem[0])
        chop_s = int(sorted_rem[1])
        bull_s = int(sorted_rem[2])
    else:
        # Fallback for k != 4
        sorted_rem = sorted(rem, key=lambda x: float(means[x, 1]))
        bear_s = int(sorted_rem[0])
        bull_s = int(sorted_rem[-1])
        chop_s = int(sorted_rem[len(sorted_rem) // 2]) if len(sorted_rem) > 1 else bear_s

    # [Safety Check] If "Bear" state has a strong positive long-term trend, warn.
    if float(means[bear_s, 1]) > 0.5:
        _logger.warning("HMM Bear state assignment anomaly: state %d has positive 168h trend %.4f", 
                        bear_s, float(means[bear_s, 1]))

    return {crisis_s: "crisis", bear_s: "bear_trend", bull_s: "bull_trend", chop_s: "chop"}


def _raw_posterior_to_semantic(
    raw_p: np.ndarray, state_to_label: dict[int, str]
) -> np.ndarray:
    """raw_p shape (k,); output shape (4,) in _SEMANTIC_ORDER."""
    out: np.ndarray = np.zeros(len(_SEMANTIC_ORDER), dtype=np.float64)
    for si, p_s in enumerate(raw_p):
        lab = state_to_label.get(si, "chop")
        prob_col = f"hmm_prob_{lab}"
        if prob_col in _SEMANTIC_ORDER:
            j = _SEMANTIC_ORDER.index(prob_col)
            out[j] += float(p_s)
    return out


def _regime_entropy(p: np.ndarray) -> float:
    """Calculate Shannon entropy of regime probabilities."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


LABEL_VERSION = "v5"


def _load_or_build_state_labels(
    label_path: Path,
    means: np.ndarray,
    X_ref_scaled: np.ndarray,
    k: int,
) -> dict[int, str]:
    """Load semantic labels from disk or build them if missing/invalid."""
    if label_path.exists():
        try:
            with open(label_path, "rb") as f:
                blob = pickle.load(f)  # noqa: S301
            mapping = blob.get("state_to_label", blob)
            n_feat = blob.get("n_features")
            if isinstance(mapping, dict) and len(mapping) == k:
                need_rebuild = (
                    n_feat is None
                    or int(n_feat) != int(means.shape[1])
                    or blob.get("label_version") != LABEL_VERSION
                )
                if not need_rebuild:
                    _logger.info("Loaded HMM state label mapping from %s", label_path)
                    return {int(a): str(b) for a, b in mapping.items()}
                _logger.info(
                    "HMM state labels invalidated (n_features=%s vs %s); rebuilding.",
                    n_feat,
                    int(means.shape[1]),
                )
        except Exception as e:
            _logger.warning("Failed to load state labels: %s", e)
    m = _assign_state_semantic_labels(means, X_ref_scaled)
    try:
        with open(label_path, "wb") as f:
            pickle.dump(
                {
                    "state_to_label": m,
                    "k_states": k,
                    "n_features": int(means.shape[1]),
                    "label_version": LABEL_VERSION,
                },
                f,
            )
        _logger.info("Saved HMM state label mapping to %s", label_path)
    except Exception as e:
        _logger.warning("Failed to save state labels: %s", e)
    return m


def _centroid_label_drift_warn(
    means: np.ndarray, X_ref_scaled: np.ndarray, state_to_label: dict[int, str]
) -> None:
    """Check if HMM semantic labels still match the underlying centroid characteristics."""
    try:
        c_scores = _crisis_semantic_scores(means)
        median_trend = float(np.median([float(means[i, 0]) for i in range(means.shape[0])]))
    except Exception:
        return

    for si, lab in state_to_label.items():
        if lab == "crisis" and float(means[si, 0]) > median_trend:
            _logger.warning(
                "HMM label drift: state %d crisis label trend=%.4f > median=%.4f "
                "(crisis_score=%.4f)",
                si,
                float(means[si, 0]),
                median_trend,
                float(c_scores[si]),
            )
        if lab == "bull_trend":
            b_score = _bull_semantic_score(means, si)
            if b_score < -0.2:
                _logger.warning(
                    "HMM label drift: state %d bull_trend but low composite score=%.4f",
                    si,
                    b_score,
                )
        if lab == "bear_trend" and float(means[si, 0]) > 0.5:
            _logger.warning(
                "HMM label drift: state %d bear_trend but positive trend=%.4f",
                si,
                float(means[si, 0]),
            )


@dataclass
class HMMStateInferrer:
    """Infers market regimes using Gaussian Hidden Markov Models.
    
    Provides both systemic (market-wide) and per-symbol regime detection
    with stable semantic labels (crisis, bear_trend, bull_trend, chop).
    """

    n_states: int = 4
    max_states: int = 4
    covariance_type: str = "diag"
    n_iter: int = 200
    predict_step: int = 24
    fit_step: int = 168
    min_covar: float = 1e-3

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "1h",
    ) -> pd.DataFrame:
        """Expanding-window systemic HMM with stable semantic posteriors."""
        _ = returns_ser  # GP-independent; ordering uses centroid labels only.

        k_cfg = int(OPT_FUTURES_CONFIG.get("FUTURES_HMM_K_STATES", self.n_states))
        self.n_states = max(2, k_cfg)
        tr_alpha = float(OPT_FUTURES_CONFIG.get("FUTURES_HMM_TRANSITION_PRIOR_ALPHA", 0.2))

        cache_fname = (
            f"HMM_systemic_{symbol}_{tf}_is{is_end_idx}_n{self.n_states}_v4multihz.parquet"
        )
        cache_path = FUTURES_CACHE_DIR / cache_fname
        label_path = FUTURES_CACHE_DIR / f"{symbol}_{tf}_state_labels.pkl"

        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                _logger.info("[%s] HMM Systemic probabilities loaded from cache.", symbol)
                return cached_df
            except Exception as e:
                _logger.debug("Failed to load HMM cache: %s", e)

        feat_cols = list(SYSTEMIC_HMM_FEATURE_COLUMNS)
        for c in feat_cols:
            if c not in features_df.columns:
                raise ValueError(f"Missing systemic HMM column: {c}")
        X_frame = features_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        n = len(X_frame)
        if n < 200:
            return self._zeros_semantic(features_df)

        X_raw = X_frame.to_numpy(dtype=np.float64)
        probs_sem: np.ndarray = np.full((n, len(_SEMANTIC_ORDER)), np.nan, dtype=np.float64)
        min_train = 500
        max_window = 8760
        model: GaussianHMM | None = None
        consecutive_fails = 0
        degraded = False
        state_to_label: dict[int, str] | None = None
        k_st = int(self.n_states)
        n_feat_expected = len(feat_cols)
        if label_path.exists():
            try:
                with open(label_path, "rb") as f:
                    blob = pickle.load(f)  # noqa: S301
                loaded = blob.get("state_to_label", blob)
                n_feat = blob.get("n_features")
                if (
                    isinstance(loaded, dict)
                    and len(loaded) == k_st
                    and n_feat is not None
                    and int(n_feat) == n_feat_expected
                    and blob.get("label_version") == LABEL_VERSION
                ):
                    state_to_label = {int(a): str(b) for a, b in loaded.items()}
                    _logger.info("Using existing HMM state label mapping from %s", label_path)
            except Exception as e:
                _logger.debug("No usable label pickle: %s", e)

        for t in range(self.predict_step, n, self.predict_step):
            if t < min_train:
                continue

            win_end_idx = max(1, t - 1)
            X_win_raw = X_raw[max(0, t - max_window) : win_end_idx]
            X_train, med, mad = _robust_fit_scale(X_win_raw)

            ent_window = probs_sem[max(0, t - 24) : t, :]
            force_entropy = False
            if t <= is_end_idx and np.isfinite(ent_window).all() and ent_window.size > 0:
                ent_rows = [_regime_entropy(ent_window[i]) for i in range(len(ent_window))]
                mean_h = float(np.mean(ent_rows))
                h_max = float(np.log(max(self.n_states, 2)))
                if mean_h > 0.95 * h_max:
                    force_entropy = True

            is_fit_cycle = (
                ((t % self.fit_step == 0) and t <= is_end_idx)
                or (model is None)
                or (force_entropy and t <= is_end_idx)
            )

            if is_fit_cycle:
                prev_model = model
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter if prev_model is None else 10,
                    random_state=42,
                    init_params="" if prev_model is not None else "stmc",
                    min_covar=self.min_covar,
                )
                if prev_model is not None:
                    try:
                        model.n_features = prev_model.n_features
                        model.startprob_ = np.nan_to_num(
                            prev_model.startprob_, nan=1.0 / self.n_states
                        )
                        tm = prev_model.transmat_.copy()
                        tm = np.nan_to_num(tm, nan=1.0 / self.n_states)
                        tm = np.clip(tm, 1e-3, None)
                        model.transmat_ = tm / tm.sum(axis=1)[:, np.newaxis]
                        model.means_ = prev_model.means_
                        cv = prev_model.covars_.copy()
                        if self.covariance_type == "diag":
                            # [FIX] hmmlearn 0.3.3 returns 3D even for 'diag', but setter expects 2D
                            if cv.ndim == 3:
                                cv = np.array([np.diag(c) for c in cv])
                            cv = np.clip(cv, self.min_covar, None)
                        elif self.covariance_type == "full":
                            for i in range(self.n_states):
                                cv[i] = (cv[i] + cv[i].T) / 2.0
                                cv[i] += np.eye(cv[i].shape[0]) * 1e-9
                        model.covars_ = cv
                    except Exception as e:
                        _logger.debug("HMM param injection failed: %s", e)
                        model.init_params = "stmc"

                fit_ok = False
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*overwritten during initialization.*"
                        )
                        model.fit(X_train)
                    tm = model.transmat_.copy()
                    tm = _blend_transition(tm, self.n_states, tr_alpha, self_p=0.9)
                    model.transmat_ = tm
                    if self.covariance_type == "diag":
                        # [FIX] hmmlearn 0.3.3 returns 3D even for 'diag', but setter expects 2D
                        cv_after = model.covars_
                        if cv_after.ndim == 3:
                            cv_after = np.array([np.diag(c) for c in cv_after])
                        model.covars_ = np.clip(cv_after, self.min_covar, None)
                    
                    # [P0] Prevent Degenerate startprob_ from absorbing inference
                    sp = model.startprob_.copy()
                    sp = np.nan_to_num(sp, nan=1.0 / self.n_states)
                    sp = sp + 1e-4
                    s_sum = sp.sum()
                    if s_sum > 1e-12:
                        model.startprob_ = sp / s_sum
                    else:
                        model.startprob_ = np.full(self.n_states, 1.0 / self.n_states)

                    # [P0] Occupancy floor check & State Re-initialization
                    try:
                        hard_states = model.predict(X_train)
                        occ = np.bincount(hard_states, minlength=self.n_states) / len(hard_states)
                        # Relax floor to 2% for better flexibility in shorter windows
                        if np.any(occ < 0.02):
                            _logger.info(
                                "[%s] HMM State Collapse detected (min occ %.4f). "
                                "Re-initializing...",
                                symbol,
                                np.min(occ),
                            )
                            dominant_s = int(np.argmax(occ))
                            collapsed_indices = np.where(occ < 0.02)[0]
                            
                            new_means = model.means_.copy()
                            for ci in collapsed_indices:
                                # Slightly lower noise (0.3 instead of 0.5) for stability
                                noise = np.random.normal(0, 0.3, size=model.n_features)
                                new_means[ci] = model.means_[dominant_s] + noise
                            
                            model.means_ = new_means
                            # Reset core params with strict normalization
                            model.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
                            model.transmat_ = _sticky_transmat_prior(self.n_states, self_p=0.8)
                            if self.covariance_type == "diag":
                                model.covars_ = np.ones((self.n_states, model.n_features))
                            
                            # Force re-labeling because centroids moved
                            state_to_label = None
                            if label_path.exists():
                                try:
                                    label_path.unlink()
                                except Exception as e:
                                    _logger.debug("Failed to delete label file: %s", e)
                    except Exception as e:
                        _logger.debug("Occupancy check failed: %s", e)
                    
                    fit_ok = True
                    consecutive_fails = 0
                except Exception as e:
                    consecutive_fails += 1
                    _logger.warning(
                        "HMM systemic fit failed at t=%d (%s); consecutive_fails=%d",
                        t,
                        e,
                        consecutive_fails,
                    )
                    if prev_model is not None:
                        model = prev_model
                        degraded = True
                        _logger.warning("HMM degraded mode: retaining previous model at t=%d", t)
                    else:
                        model = None
                    if consecutive_fails >= 2 and model is None:
                        _logger.error(
                            "HMM systemic: repeated fit failure - "
                            "uniform posteriors; check features."
                        )

                if fit_ok and model is not None:
                    try:
                        means_f = model.means_.copy()
                        if t <= is_end_idx:
                            state_to_label = _assign_state_semantic_labels(means_f, X_train)
                        elif state_to_label is None:
                            state_to_label = _assign_state_semantic_labels(means_f, X_train)
                        else:
                            _centroid_label_drift_warn(means_f, X_train, state_to_label)
                    except Exception as e:
                        _logger.warning("HMM label assignment warning: %s", e)

            if model is None or state_to_label is None:
                continue

            try:
                win_start = max(0, t - max_window)
                win_end = max(1, t - 1)
                x_seq = X_raw[win_start : win_end, :]
                x_seq_s = _robust_transform(x_seq, med, mad)
                p_seq = model.predict_proba(x_seq_s)
                last_p_raw = p_seq[-1]
                ps = _raw_posterior_to_semantic(last_p_raw, state_to_label)
                probs_sem[t - 1, :] = ps
            except Exception as e:
                _logger.debug("HMM systemic inference failed at t=%d: %s", t, e)

        probs_df = pd.DataFrame(probs_sem, index=features_df.index, columns=_SEMANTIC_ORDER)
        probs_df = probs_df.ffill().bfill().fillna(1.0 / float(len(_SEMANTIC_ORDER)))
        if degraded:
            _logger.warning(
                "[%s] HMM systemic completed with degraded=True (some refits used prior model).",
                symbol,
            )

        out = probs_df.copy()
        out = out.reset_index()
        if "datetime" not in out.columns:
            out = out.rename(columns={out.columns[0]: "datetime"})

        try:
            out.to_parquet(cache_path)
        except Exception as e:
            _logger.debug("Failed to cache HMM: %s", e)
        return out

    def fit_predict(
        self, df: pd.DataFrame, is_end_idx: int, symbol: str = "Unknown", tf: str = "1h"
    ) -> pd.DataFrame:
        """Per-symbol HMM (legacy Sharpe ordering); uses winsorize + Z-scale."""
        cache_sym = symbol.replace("/", "_")
        cache_fname = f"HMM_probs_{cache_sym}_{tf}_is{is_end_idx}_n{self.n_states}.parquet"
        cache_path = FUTURES_CACHE_DIR / cache_fname

        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                if len(cached_df) == len(df):
                    _logger.info("[%s] HMM probabilities loaded from cache: %s", symbol, cache_path)
                    return cached_df
                _logger.warning(
                    "[%s] Cache length mismatch (%d vs %d). Re-calculating...",
                    symbol,
                    len(cached_df),
                    len(df),
                )
            except Exception as e:
                _logger.error("[%s] Failed to load HMM cache: %s", symbol, e)

        n = len(df)
        if n < 200:
            return self._zeros(df)

        from src.domain.futures.ml_pipeline.feature_engineering import build_hmm_input_features

        hmm_feat = build_hmm_input_features(df.iloc[:n])
        X_raw = hmm_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_raw = X_raw.to_numpy(dtype=np.float64)

        probs: np.ndarray = np.full((len(df), self.n_states), np.nan, dtype=np.float64)
        min_train = max(self.predict_step * 5, 500)
        max_window = 8760

        model: GaussianHMM | None = None
        curr_step = 0
        tr_alpha = float(OPT_FUTURES_CONFIG.get("FUTURES_HMM_TRANSITION_PRIOR_ALPHA", 0.2))

        for t in range(self.predict_step, n, self.predict_step):
            curr_step += 1
            if t < min_train:
                continue

            win_end_idx = max(1, t - 1)
            X_win_raw = X_raw[max(0, t - max_window) : win_end_idx]
            X_win_win = _winsorize_cols_numba(X_win_raw, 0.01)
            w_mean = X_win_win.mean(axis=0)
            w_std = X_win_win.std(axis=0)
            w_std[w_std < 1e-8] = 1.0
            X_train = (X_win_win - w_mean) / w_std

            is_fit_cycle = ((t % self.fit_step == 0) and t <= is_end_idx) or (model is None)

            if is_fit_cycle:
                if curr_step % 200 == 0 or model is None:
                    _logger.info(
                        "[%s] HMM fit at t=%d (Warm-start: %s)...",
                        symbol,
                        t,
                        "Yes" if model is not None else "No",
                    )

                prev_model = model
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter if prev_model is None else 10,
                    tol=1e-3,
                    random_state=42,
                    init_params="" if prev_model is not None else "stmc",
                    min_covar=self.min_covar,
                )

                if prev_model is not None:
                    model.init_params = ""
                    try:
                        model.startprob_ = prev_model.startprob_
                        tm = prev_model.transmat_.copy()
                        tm = np.clip(tm, 1e-3, None)
                        model.transmat_ = tm / tm.sum(axis=1)[:, np.newaxis]
                        model.means_ = prev_model.means_
                        cv = prev_model.covars_.copy()
                        if self.covariance_type == "diag":
                            # [FIX] hmmlearn 0.3.3 returns 3D even for 'diag', but setter expects 2D
                            if cv.ndim == 3:
                                cv = np.array([np.diag(c) for c in cv])
                            cv = np.clip(cv, self.min_covar, None)
                        elif self.covariance_type == "full":
                            for i in range(self.n_states):
                                cv[i] = (cv[i] + cv[i].T) / 2.0
                                cv[i] += np.eye(cv[i].shape[0]) * 1e-9
                        model.covars_ = cv
                    except Exception as e:
                        _logger.debug("[%s] HMM warm-start param injection failed: %s", symbol, e)
                        model.init_params = "stmc"

                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*overwritten during initialization.*"
                        )
                        model.fit(X_train)
                    tm = model.transmat_.copy()
                    tm = _blend_transition(tm, self.n_states, tr_alpha, self_p=0.9)
                    model.transmat_ = tm
                    if self.covariance_type == "diag":
                        # [FIX] hmmlearn 0.3.3 returns 3D even for 'diag', but setter expects 2D
                        cv_after = model.covars_
                        if cv_after.ndim == 3:
                            cv_after = np.array([np.diag(c) for c in cv_after])
                        model.covars_ = np.clip(cv_after, self.min_covar, None)
                    try:
                        tm2 = model.transmat_.copy()
                        tm_sum = tm2.sum(axis=1)
                        zero_rows = np.where(tm_sum < 1e-10)[0]
                        if len(zero_rows) > 0:
                            tm2[zero_rows, :] = 1.0 / self.n_states
                        model.transmat_ = tm2 / tm2.sum(axis=1)[:, np.newaxis]
                        sp = model.startprob_.copy()
                        if sp.sum() < 1e-10:
                            sp[:] = 1.0 / self.n_states
                        model.startprob_ = sp / sp.sum()
                    except (AttributeError, ValueError):
                        pass
                except Exception as e:
                    _logger.debug("[%s] HMM fit failed at %d: %s", symbol, t, e)
                    if prev_model is not None:
                        model = prev_model
                    else:
                        continue

            if model is None:
                continue

            try:
                last_p_raw = model.predict_proba(X_train[-1:].reshape(1, -1))[0]
                state_seq = model.predict(X_train)
                win_start = max(0, t - max_window)
                win_end_idx = max(1, t - 1)
                c_win = df["close"].astype(np.float64).iloc[win_start:win_end_idx].to_numpy()
                w_len = len(X_train)
                lr_win: np.ndarray = np.zeros(w_len, dtype=np.float64)
                lr_win[1:] = np.log(np.clip(c_win[1:] / np.maximum(c_win[:-1], 1e-12), 1e-12, None))
                probs[t - 1, :] = _calculate_ordered_probs_numba(
                    state_seq, lr_win, last_p_raw, self.n_states
                )
            except Exception as e:
                _logger.debug("[%s] HMM inference failed at %d: %s", symbol, t, e)

        probs_df = pd.DataFrame(probs, index=df.index).ffill().bfill().fillna(1.0 / self.n_states)
        cols = [f"hmm_prob_{i}" for i in range(self.n_states)]
        out = pd.DataFrame({c: probs_df.iloc[:, i] for i, c in enumerate(cols)})
        out.index = df.index

        try:
            out.to_parquet(cache_path)
            _logger.info("[%s] HMM probabilities cached to %s", symbol, cache_path)
        except Exception as e:
            _logger.warning("[%s] Failed to cache HMM: %s", symbol, e)

        return out

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of uniform semantic probabilities."""
        u = 1.0 / float(len(_SEMANTIC_ORDER))
        out = pd.DataFrame(
            np.full((len(df), len(_SEMANTIC_ORDER)), u),
            index=df.index,
            columns=_SEMANTIC_ORDER,
        )
        out = out.reset_index()
        if "datetime" not in out.columns:
            out = out.rename(columns={out.columns[0]: "datetime"})
        return out

    def _zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of uniform numeric probabilities."""
        k = int(self.n_states)
        cols = [f"hmm_prob_{i}" for i in range(k)]
        return pd.DataFrame(
            np.full((len(df), k), 1.0 / float(k)),
            index=df.index,
            columns=cols,
        )
