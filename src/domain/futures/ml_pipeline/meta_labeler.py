"""LightGBM meta-labeler with Platt (default) or isotonic calibration and purged CV.

Dual-classifier design (Q1 decision):
  - MetaLabeler trains two independent LGBMClassifiers:
      _lgb_long:  y = (tbm_label == +1) → calib_prob_long (Long edge probability)
      _lgb_short: y = (tbm_label == -1) → calib_prob_short (Short edge probability)
  - Phase 2: Platt scaling when calibration positives < min_pos_isotonic;
    IsotonicRegression when calibration positives ≥ min_pos_isotonic.
  - Concurrent-uniqueness weights on train rows (inverse sqrt rolling label density).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def _uniq_weights_binary(y: np.ndarray, horizon: int) -> np.ndarray:
    """Soft concurrent-uniqueness: down-weight dense label clusters."""
    if len(y) == 0:
        return np.ones(0, dtype=np.float64)
    win = max(2, min(int(horizon), len(y)))
    roll = pd.Series(y).rolling(win, min_periods=1).sum()
    out = (1.0 / np.sqrt(1.0 + roll.to_numpy(dtype=np.float64))).astype(np.float64)
    return cast(np.ndarray, out)


@dataclass
class MetaLabeler:
    """
    Directional meta-labeling model using LightGBM and probability calibration.
    
    Trains independent classifiers for Long and Short directions, providing
    calibrated probabilities for trade execution.
    """
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.05
    reg_lambda: float = 5.0
    min_child_samples: int = 30
    threshold_mode: str = "f1_optimal"   # "fixed_0.5" | "f1_optimal"
    vertical_barrier_bars: int = 24  # purge horizon in label rows (~24h @ 1h vs 1440x1m)
    min_pos_isotonic: int = 200

    # Long classifier internals
    _iso_long: IsotonicRegression | None = field(default=None, repr=False)
    _platt_long: LogisticRegression | None = field(default=None, repr=False)
    _lgb_long: lgb.LGBMClassifier | None = field(default=None, repr=False)
    _thr_long: float = field(default=0.5, repr=False)

    # Short classifier internals
    _iso_short: IsotonicRegression | None = field(default=None, repr=False)
    _platt_short: LogisticRegression | None = field(default=None, repr=False)
    _lgb_short: lgb.LGBMClassifier | None = field(default=None, repr=False)
    _thr_short: float = field(default=0.5, repr=False)

    @staticmethod
    def purged_train_calib_bounds(
        n: int,
        vertical_barrier_bars: int,
    ) -> tuple[int, int, int]:
        """
        Calculate (train_end, calib_split, calib_start) row counts for time-ordered split.

        Labels must not leak across [train_end, calib_start).
        """
        purge = max(0, int(vertical_barrier_bars))
        embargo = max(1, int(0.01 * n))
        calib_split = max(int(n * 0.8), n - 200)
        calib_split = max(40, min(calib_split, n - 20))
        train_end = calib_split - purge
        calib_start = calib_split + embargo
        if train_end < 40 or calib_start >= n - 10:
            purge = max(0, calib_split - 41)
            train_end = calib_split - purge
            calib_start = calib_split + embargo
        if calib_start >= n or train_end < 20:
            train_end = max(20, calib_split - 1)
            calib_start = min(calib_split + max(1, embargo // 2), n - 5)
        return int(train_end), int(calib_split), int(calib_start)

    def _fit_one(
        self,
        X_is: pd.DataFrame,
        y_is: pd.Series,
    ) -> tuple[
        lgb.LGBMClassifier | None,
        IsotonicRegression | None,
        LogisticRegression | None,
        float,
    ]:
        """Fit a single directional classifier. Returns (lgb, iso, platt, threshold)."""
        m = y_is.notna() & np.isfinite(X_is.to_numpy()).all(axis=1)
        X_clean = X_is[m]
        y_clean = y_is[m]

        _fallback_iso = IsotonicRegression(out_of_bounds="clip")
        _fallback_iso.fit(np.array([0.5]), np.array([0.5]))
        _fallback_lr = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
        _fallback_lr.fit(np.array([[0.5]]), np.array([0]))

        if len(X_clean) < 80:
            return None, _fallback_iso, None, 0.5

        n = len(X_clean)
        train_end, _, calib_start = MetaLabeler.purged_train_calib_bounds(
            n, self.vertical_barrier_bars
        )

        X_train = X_clean.iloc[:train_end]
        y_train = y_clean.iloc[:train_end]
        X_calib = X_clean.iloc[calib_start:]
        y_calib = y_clean.iloc[calib_start:]
        if len(X_train) < 30 or len(X_calib) < 10:
            return None, _fallback_iso, None, 0.5

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos = float(neg) / max(pos, 1)

        horizon = max(3, min(int(self.vertical_barrier_bars), len(y_train)))
        sample_w = _uniq_weights_binary(y_train.to_numpy(dtype=np.float64), horizon)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            scale_pos_weight=scale_pos,
            verbosity=-1,
            random_state=42,
        )
        lgb_model.fit(X_train, y_train, sample_weight=sample_w)

        raw_calib = lgb_model.predict_proba(X_calib)[:, 1]
        y_calib_np = y_calib.to_numpy()
        n_pos_calib = int(np.sum(y_calib_np))

        use_isotonic = n_pos_calib >= int(self.min_pos_isotonic)
        iso: IsotonicRegression | None = None
        platt: LogisticRegression | None = None

        if use_isotonic:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_calib, y_calib_np)
            calibrated = iso.predict(raw_calib)
        else:
            platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
            platt.fit(raw_calib.reshape(-1, 1), y_calib_np)
            calibrated = platt.predict_proba(raw_calib.reshape(-1, 1))[:, 1]

        threshold = 0.5
        if self.threshold_mode == "f1_optimal" and len(y_calib) >= 10:
            best_f1 = 0.0
            for thr in np.arange(0.20, 0.81, 0.05):
                preds = (calibrated >= thr).astype(int)
                f1 = float(f1_score(y_calib_np, preds, zero_division=0.0))
                if f1 > best_f1:
                    best_f1, threshold = f1, float(thr)

        return lgb_model, iso, platt, threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_end_idx: int,
    ) -> MetaLabeler:
        """Fit dual directional classifiers on IS data.

        Args:
            X: Feature matrix (gp_alpha + hmm_prob columns).
            y: TBM labels: +1 (Long TP), -1 (Short TP), 0 (neutral), 0.5 (fallback).
            is_end_idx: IS/OOS split index.

        """

        X_is = X.iloc[:is_end_idx].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_is = y.iloc[:is_end_idx].astype(np.float64)

        y_long = (y_is == 1.0).astype(np.float64)
        y_short = (y_is == -1.0).astype(np.float64)

        self._lgb_long, self._iso_long, self._platt_long, self._thr_long = self._fit_one(
            X_is, y_long
        )
        self._lgb_short, self._iso_short, self._platt_short, self._thr_short = (
            self._fit_one(X_is, y_short)
        )

        return self

    def predict_proba_calibrated(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (calib_prob_long, calib_prob_short) arrays of shape (N,)."""
        Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(Xv)

        def _predict(
            lgb_m: lgb.LGBMClassifier | None,
            iso: IsotonicRegression | None,
            platt_m: LogisticRegression | None,
        ) -> np.ndarray:
            if lgb_m is None:
                return np.full(n, 0.5, dtype=np.float64)
            raw = lgb_m.predict_proba(Xv)[:, 1]
            if platt_m is not None:
                cal = platt_m.predict_proba(raw.reshape(-1, 1))[:, 1]
            elif iso is not None:
                cal = iso.predict(raw)
            else:
                return np.full(n, 0.5, dtype=np.float64)
            return np.asarray(cal, dtype=np.float64)

        prob_long = _predict(self._lgb_long, self._iso_long, self._platt_long)
        prob_short = _predict(self._lgb_short, self._iso_short, self._platt_short)
        return prob_long, prob_short

    @property
    def threshold_long(self) -> float:
        """Optimal probability threshold for Long signals."""
        return self._thr_long

    @property
    def threshold_short(self) -> float:
        """Optimal probability threshold for Short signals."""
        return self._thr_short
