"""Gaussian HMM regime probabilities with expanding-window refit and state reordering."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from numba import njit

from config.settings import FUTURES_CACHE_DIR

warnings.filterwarnings("ignore", message=".*overwritten during initialization.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")

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
    """
    Calculate state Risk-Adjusted Returns (Sharpe-like) and reorder probabilities.
    Returns: [prob_worst, ..., prob_best] such that the last element corresponds to the best state.
    """
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
    
    # argsort returns indices from smallest to largest score
    order = np.argsort(scores)
    
    # Map the raw probabilities (last_p) to their ranked positions
    reordered: np.ndarray = np.zeros(n_states, dtype=np.float64)
    for i in range(n_states):
        # i=0: index of worst state -> reordered[0] is prob of worst state
        # i=2: index of best state  -> reordered[2] is prob of best state
        reordered[i] = last_p[order[i]]
    return reordered


@dataclass
class HMMStateInferrer:
    n_states: int = 3
    max_states: int = 4          # BIC auto-select upper bound (2 ~ max_states)
    covariance_type: str = "diag"
    n_iter: int = 200
    predict_step: int = 24
    fit_step: int = 168
    _selected_n_states: int = field(default=0, repr=False)  # 0 = not yet auto-selected

    def _bic_pick_n_states(self, X_train: np.ndarray, symbol: str) -> int:
        """Select n_states in [2, max_states] by BIC on standardized training rows."""
        best_bic = np.inf
        best_n = 2
        n_feats = int(X_train.shape[1])
        row_n = max(len(X_train), 2)
        for n_try in range(2, self.max_states + 1):
            try:
                m_try = GaussianHMM(
                    n_components=n_try,
                    covariance_type=self.covariance_type,
                    n_iter=50,
                    tol=1e-2,
                    random_state=42,
                )
                m_try.fit(X_train)
                log_l = m_try.score(X_train) * len(X_train)
                k_params = n_try * (n_try - 1) + n_try * n_feats * 2
                bic = -2.0 * log_l + k_params * np.log(row_n)
                if bic < best_bic:
                    best_bic, best_n = bic, n_try
                    _logger.info(
                        "[%s] BIC n_states=%d: %.2f (best so far)", symbol, n_try, bic
                    )
            except Exception as e:
                _logger.debug("[%s] BIC n_try=%d failed: %s", symbol, n_try, e)
        return int(best_n)

    def fit_predict_systemic(
        self, 
        features_df: pd.DataFrame, 
        returns_ser: pd.Series, 
        is_end_idx: int, 
        symbol: str = "Market", 
        tf: str = "1h"
    ) -> pd.DataFrame:
        """
        Expanding-window inference for Systemic Market HMM.
        - features_df: Pre-computed market-wide features.
        - returns_ser: Performance proxy (e.g. GP LS Spread) for state ordering.
        """
        bic_auto = self.n_states <= 0
        cache_tag = "BIC" if bic_auto else str(int(self.n_states))
        cache_fname = f"HMM_systemic_{symbol}_{tf}_is{is_end_idx}_n{cache_tag}.parquet"
        cache_path = FUTURES_CACHE_DIR / cache_fname

        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                _logger.info("[%s] HMM Systemic probabilities loaded from cache.", symbol)
                return cached_df
            except Exception as e:
                _logger.debug("Failed to load HMM cache: %s", e)

        n = len(features_df)
        if n < 200:
            return self._zeros(features_df)

        X_raw = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_raw = X_raw.to_numpy(dtype=np.float64)

        alloc_w = self.max_states if bic_auto else int(self.n_states)
        probs: np.ndarray = np.full((len(features_df), alloc_w), np.nan, dtype=np.float64)
        min_train = 500
        max_window = 8760
        model: GaussianHMM | None = None
        lr_full = returns_ser.to_numpy(dtype=np.float64)

        for t in range(self.predict_step, n, self.predict_step):
            if t < min_train:
                continue
            
            # Dynamic scaling: exclude current bar row (use data up to t-1 only)
            win_end_idx = max(1, t - 1)
            X_win_raw = X_raw[max(0, t - max_window):win_end_idx]
            X_win_win = _winsorize_cols_numba(X_win_raw, 0.01)
            
            w_mean = X_win_win.mean(axis=0)
            w_std = X_win_win.std(axis=0)
            w_std[w_std < 1e-8] = 1.0
            X_train = (X_win_win - w_mean) / w_std

            is_fit_cycle = ((t % self.fit_step == 0) and t <= is_end_idx) or (model is None)

            if is_fit_cycle:
                if model is None and bic_auto and self._selected_n_states == 0:
                    best_n = self._bic_pick_n_states(X_train, symbol)
                    self._selected_n_states = best_n
                    self.n_states = best_n
                    probs = np.zeros((n, self.n_states), dtype=np.float64)

                prev_model = model
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter if prev_model is None else 10,
                    random_state=42,
                    init_params="" if prev_model is not None else "stmc"
                )
                if prev_model is not None:
                    try:
                        # Synchronize dimensions to avoid ValueError before fit
                        model.n_features = prev_model.n_features
                        model.startprob_ = prev_model.startprob_
                        
                        tm = prev_model.transmat_.copy()
                        tm = np.clip(tm, 1e-3, None)
                        model.transmat_ = tm / tm.sum(axis=1)[:, np.newaxis]
                        
                        model.means_ = prev_model.means_
                        
                        # Sanitize and inject covars
                        cv = prev_model.covars_.copy()
                        if self.covariance_type == "full":
                            for i in range(self.n_states):
                                cv[i] = (cv[i] + cv[i].T) / 2.0
                                cv[i] += np.eye(cv[i].shape[0]) * 1e-9
                        model.covars_ = cv
                    except Exception as e:
                        _logger.debug("HMM param injection failed: %s", e)
                        model.init_params = "stmc"  # Fallback
                
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*overwritten during initialization.*"
                        )
                        model.fit(X_train)
                    
                    # Sanitize after fit to ensure sum(axis=1) == 1.0
                    tm = model.transmat_.copy()
                    tm_sum = tm.sum(axis=1)
                    zero_rows = np.where(tm_sum < 1e-10)[0]
                    if len(zero_rows) > 0:
                        tm[zero_rows, :] = 1.0 / self.n_states
                    model.transmat_ = tm / tm.sum(axis=1)[:, np.newaxis]
                except Exception as e:
                    _logger.debug("HMM fit failed at t=%d: %s", t, e)
                    if prev_model is not None:
                        model = prev_model
                    else:
                        continue

            if model is not None:
                try:
                    last_p_raw = model.predict_proba(X_train[-1:].reshape(1, -1))[0]
                    state_seq = model.predict(X_train)
                    win_end_idx = max(1, t - 1)
                    lr_win = lr_full[max(0, t - max_window):win_end_idx]
                    
                    probs[t - 1, :] = _calculate_ordered_probs_numba(
                        state_seq, lr_win, last_p_raw, self.n_states
                    )
                except Exception as e:
                    _logger.debug("HMM inference failed at t=%d: %s", t, e)

        probs_df = pd.DataFrame(
            probs, index=features_df.index
        ).ffill().bfill().fillna(1.0 / self.n_states)
        cols = [f"hmm_prob_{i}" for i in range(self.n_states)]
        out = pd.DataFrame(
            {c: probs_df.iloc[:, i] for i, c in enumerate(cols)},
            index=features_df.index
        )
        
        try:
            out.to_parquet(cache_path)
        except Exception as e:
            _logger.debug("Failed to cache HMM: %s", e)
        return out

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
        # Cache key includes is_end_idx to prevent stale cache when IS/OOS boundary changes.
        cache_sym = symbol.replace("/", "_")
        bic_auto = self.n_states <= 0
        if self._selected_n_states > 0:
            cache_n_key = str(int(self._selected_n_states))
        elif bic_auto:
            cache_n_key = "BIC"
        else:
            cache_n_key = str(int(self.n_states))
        cache_fname = f"HMM_probs_{cache_sym}_{tf}_is{is_end_idx}_n{cache_n_key}.parquet"
        cache_path = FUTURES_CACHE_DIR / cache_fname

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

        n = len(df)
        if n < 200:
            return self._zeros(df)

        from src.domain.futures.ml_pipeline.feature_engineering import build_hmm_input_features

        hmm_feat = build_hmm_input_features(df.iloc[:n])
        X_raw = hmm_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_raw = X_raw.to_numpy(dtype=np.float64)

        alloc_w = self.max_states if bic_auto else int(self.n_states)
        probs: np.ndarray = np.full((len(df), alloc_w), np.nan, dtype=np.float64)
        min_train = max(self.predict_step * 5, 500)
        max_window = 8760
        
        model: GaussianHMM | None = None
        curr_step = 0

        for t in range(self.predict_step, n, self.predict_step):
            curr_step += 1
            if t < min_train:
                continue

            # Dynamic scaling: exclude current bar row (use data up to t-1 only)
            win_end_idx = max(1, t - 1)
            X_win_raw = X_raw[max(0, t - max_window):win_end_idx]
            X_win_win = _winsorize_cols_numba(X_win_raw, 0.01)
            
            w_mean = X_win_win.mean(axis=0)
            w_std = X_win_win.std(axis=0)
            w_std[w_std < 1e-8] = 1.0
            X_train = (X_win_win - w_mean) / w_std

            # A. Decoupled Fit logic (only if fit_step interval or first time)
            is_fit_cycle = ((t % self.fit_step == 0) and t <= is_end_idx) or (model is None)
            
            if is_fit_cycle:
                if curr_step % 200 == 0 or model is None:
                    _logger.info("[%s] HMM fit at t=%d (Warm-start: %s)...",
                                symbol, t, "Yes" if model is not None else "No")

                if model is None and bic_auto and self._selected_n_states == 0:
                    best_n = self._bic_pick_n_states(X_train, symbol)
                    self._selected_n_states = best_n
                    self.n_states = best_n
                    _logger.info("[%s] BIC auto-selected n_states=%d", symbol, best_n)
                    probs = np.zeros((len(df), self.n_states), dtype=np.float64)

                # B. Parameter Warm-Start / Online Learning
                prev_model = model
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter if prev_model is None else 10,  # Rapid convergence
                    tol=1e-3,
                    random_state=42,
                    init_params="" if prev_model is not None else "stmc"
                )
                
                if prev_model is not None:
                    # Explicitly clear init_params before setting attributes to suppress warnings
                    model.init_params = ""
                    try:
                        model.startprob_ = prev_model.startprob_
                        
                        tm = prev_model.transmat_.copy()
                        tm = np.clip(tm, 1e-3, None)  # Ensure minimum transition probability
                        model.transmat_ = tm / tm.sum(axis=1)[:, np.newaxis]
                        
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
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*overwritten during initialization.*"
                        )
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
        
        # Save to Cache
        try:
            out.to_parquet(cache_path)
            _logger.info("[%s] HMM probabilities cached to %s", symbol, cache_path)
        except Exception as e:
            _logger.warning("[%s] Failed to cache HMM: %s", symbol, e)

        return out

    def _zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        k = int(self.n_states) if self.n_states > 0 else max(2, int(self.max_states))
        if self._selected_n_states > 0:
            k = int(self._selected_n_states)
        cols = [f"hmm_prob_{i}" for i in range(k)]
        return pd.DataFrame(
            np.full((len(df), k), 1.0 / float(k)),
            index=df.index,
            columns=cols,
        )
