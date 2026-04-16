"""Gaussian HMM regime probabilities with expanding-window refit and state reordering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from numba import njit

from config.settings import FUTURES_CACHE_DIR

_logger = logging.getLogger(__name__)


@njit  # type: ignore
def _winsorize_cols_numba(X: np.ndarray, pct: float = 0.01) -> np.ndarray:
    """Numba-accelerated winsorization for feature matrix."""
    n_rows, n_cols = X.shape
    out = X.copy()
    for j in range(n_cols):
        col = X[:, j]
        # Calculate indices for lo and hi
        lo_idx = int(n_rows * pct)
        hi_idx = int(n_rows * (1.0 - pct))
        if hi_idx >= n_rows:
            hi_idx = n_rows - 1
        
        # Use np.sort instead of unsupported .partition()
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
    n_states: int
) -> np.ndarray:
    """Calculate state means and reorder last-bar probabilities by mean returns."""
    means = np.zeros(n_states, dtype=np.float64)
    for s in range(n_states):
        count = 0
        total = 0.0
        for i in range(len(state_seq)):
            if state_seq[i] == s:
                total += lr[i]
                count += 1
        if count > 0:
            means[s] = total / count
        else:
            means[s] = 0.0
    
    order = np.argsort(means)
    return last_p[order]


@dataclass
class HMMStateInferrer:
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    predict_step: int = 24
    fit_step: int = 168

    def fit_predict(
        self, df: pd.DataFrame, is_end_idx: int, symbol: str = "Unknown", tf: str = "1h"
    ) -> pd.DataFrame:
        """
        Expanding-window inference with decoupled fit/predict cycles and warm-start.
        - fit (model update): every `fit_step` (e.g., 168 bars = 1 week)
        - predict (state inference): every `predict_step` (e.g., 24 bars = 1 day)
        - Disk caching: returns cached results if available for (symbol, tf).
        """
        # 0. Disk Caching Check
        cache_sym = symbol.replace("/", "_")
        cache_path = FUTURES_CACHE_DIR / f"HMM_probs_{cache_sym}_{tf}.parquet"
        
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                if len(cached_df) == len(df):
                    _logger.info("[%s] HMM probabilities loaded from cache: %s", symbol, cache_path)
                    return cached_df
                _logger.warning("[%s] Cache length mismatch (%d vs %d). Re-calculating...", 
                               symbol, len(cached_df), len(df))
            except Exception as e:
                _logger.error("[%s] Failed to load HMM cache: %s", symbol, e)

        n = min(len(df), is_end_idx)
        if n < 200:
            return self._zeros(df)

        from src.domain.futures.ml_pipeline.feature_engineering import build_hmm_input_features

        hmm_feat = build_hmm_input_features(df.iloc[:n])
        X = hmm_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        
        # 1. Accelerated Winsorization
        X = _winsorize_cols_numba(X, 0.01)
        
        # Z-score standardization
        col_std = X.std(axis=0)
        col_std[col_std < 1e-8] = 1.0
        X = (X - X.mean(axis=0)) / col_std

        probs = np.zeros((len(df), self.n_states), dtype=np.float64)
        min_train = max(self.predict_step * 5, 500)
        max_window = 8760
        
        model: GaussianHMM | None = None
        curr_step = 0

        for t in range(self.predict_step, n, self.predict_step):
            curr_step += 1
            if t < min_train:
                continue

            # A. Decoupled Fit logic (only if fit_step interval or first time)
            is_fit_cycle = (t % self.fit_step == 0) or (model is None)
            X_train = X[max(0, t - max_window):t]

            if is_fit_cycle:
                if curr_step % 200 == 0 or model is None:
                    _logger.info("[%s] HMM fit at t=%d (Warm-start: %s)...", 
                                symbol, t, "Yes" if model is not None else "No")

                # B. Parameter Warm-Start / Online Learning
                prev_model = model
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter if prev_model is None else 10,  # Rapid convergence
                    random_state=42,
                    init_params="" if prev_model is not None else "stmc"
                )
                
                if prev_model is not None:
                    try:
                        model.startprob_ = prev_model.startprob_
                        model.transmat_ = prev_model.transmat_
                        model.means_ = prev_model.means_
                        
                        # Ensure numerical stability for 'full' covariance
                        cv = prev_model.covars_.copy()
                        if self.covariance_type == "full":
                            for i in range(self.n_states):
                                cv[i] = (cv[i] + cv[i].T) / 2.0
                                cv[i] += np.eye(cv[i].shape[0]) * 1e-9
                        model.covars_ = cv
                    except Exception as e:
                        _logger.debug("[%s] HMM warm-start param injection failed: %s", symbol, e)
                        model.init_params = "stmc" # Fallback to random init
                
                try:
                    model.fit(X_train)
                    # C. Sanitize parameters to avoid transmat zero sum warnings
                    try:
                        tm = model.transmat_.copy()
                        tm_sum = tm.sum(axis=1)
                        zero_rows = np.where(tm_sum < 1e-10)[0]
                        if len(zero_rows) > 0:
                            tm[zero_rows, :] = 1.0 / self.n_states
                        model.transmat_ = tm / tm.sum(axis=1)[:, np.newaxis]
                        
                        sp = model.startprob_.copy()
                        if sp.sum() < 1e-10:
                            sp[:] = 1.0 / self.n_states
                        model.startprob_ = sp / sp.sum()
                    except (AttributeError, ValueError):
                        pass

                except Exception as e:
                    _logger.debug("[%s] HMM fit failed at %d: %s", symbol, t, e)
                    if prev_model is not None:
                        # Continue using previous model's parameters for inference if fit fails
                        model = prev_model 
                    else:
                        continue

            # C. Inference (always every predict_step)
            if model is None:
                continue

            try:
                last_p_raw = model.predict_proba(X_train[-1:].reshape(1, -1))[0]
                state_seq = model.predict(X_train)
                
                win_start = max(0, t - max_window)
                c_win = df["close"].astype(np.float64).iloc[win_start:t].to_numpy()
                w_len = len(X_train)
                lr_win = np.zeros(w_len, dtype=np.float64)
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
        
        # Save to Cache
        try:
            out.to_parquet(cache_path)
            _logger.info("[%s] HMM probabilities cached to %s", symbol, cache_path)
        except Exception as e:
            _logger.warning("[%s] Failed to cache HMM: %s", symbol, e)

        return out

    def _zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [f"hmm_prob_{i}" for i in range(self.n_states)]
        return pd.DataFrame(
            np.full((len(df), self.n_states), 1.0 / self.n_states),
            index=df.index,
            columns=cols,
        )
