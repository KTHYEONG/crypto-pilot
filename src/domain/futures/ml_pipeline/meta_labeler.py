"""LightGBM meta-labeler with isotonic calibration and purged time-series CV.

Dual-classifier design (Q1 decision):
  - MetaLabeler trains two independent LGBMClassifiers:
      _lgb_long:  y = (tbm_label == +1) → calib_prob_long (Long edge probability)
      _lgb_short: y = (tbm_label == -1) → calib_prob_short (Short edge probability)
  - scale_pos_weight auto-set from class frequencies (mitigates 9~10% positive ratio).
  - F1-optimal threshold replaces fixed 0.5 (threshold_mode='f1_optimal').
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import f1_score


@dataclass
class MetaLabeler:
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.05
    reg_lambda: float = 5.0
    min_child_samples: int = 30
    threshold_mode: str = "f1_optimal"   # "fixed_0.5" | "f1_optimal"
    vertical_barrier_bars: int = 48  # TBM horizon in label rows (48 @ 1h ≈ 48h)

    # Long classifier internals
    _iso_long: IsotonicRegression | None = field(default=None, repr=False)
    _lgb_long: lgb.LGBMClassifier | None = field(default=None, repr=False)
    _thr_long: float = field(default=0.5, repr=False)

    # Short classifier internals
    _iso_short: IsotonicRegression | None = field(default=None, repr=False)
    _lgb_short: lgb.LGBMClassifier | None = field(default=None, repr=False)
    _thr_short: float = field(default=0.5, repr=False)

    @staticmethod
    def purged_train_calib_bounds(
        n: int,
        vertical_barrier_bars: int,
    ) -> tuple[int, int, int]:
        """
        Returns (train_end, calib_split, calib_start) row counts for time-ordered split.
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
    ) -> tuple[lgb.LGBMClassifier | None, IsotonicRegression, float]:
        """Fit a single directional classifier. Returns (lgb, iso, threshold)."""
        m = y_is.notna() & np.isfinite(X_is.to_numpy()).all(axis=1)
        X_clean = X_is[m]
        y_clean = y_is[m]

        # Fallback: degenerate calibrator
        _fallback_iso = IsotonicRegression(out_of_bounds="clip")
        _fallback_iso.fit(np.array([0.5]), np.array([0.5]))
        if len(X_clean) < 80:
            return None, _fallback_iso, 0.5

        n = len(X_clean)
        train_end, _, calib_start = MetaLabeler.purged_train_calib_bounds(
            n, self.vertical_barrier_bars
        )

        X_train = X_clean.iloc[:train_end]
        y_train = y_clean.iloc[:train_end]
        X_calib = X_clean.iloc[calib_start:]
        y_calib = y_clean.iloc[calib_start:]
        if len(X_train) < 30 or len(X_calib) < 10:
            return None, _fallback_iso, 0.5

        # Imbalance correction: scale_pos_weight balances minority class
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos = float(neg) / max(pos, 1)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            scale_pos_weight=scale_pos,  # Imbalance correction
            verbosity=-1,
            random_state=42,
        )
        lgb_model.fit(X_train, y_train)

        raw_calib = lgb_model.predict_proba(X_calib)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_calib, y_calib.to_numpy())

        # --- F1-optimal threshold on calibration set ---
        calibrated = iso.predict(raw_calib)
        threshold = 0.5
        if self.threshold_mode == "f1_optimal" and len(y_calib) >= 10:
            best_f1 = 0.0
            for thr in np.arange(0.20, 0.81, 0.05):
                preds = (calibrated >= thr).astype(int)
                f1 = float(f1_score(y_calib.to_numpy(), preds, zero_division=0.0))
                if f1 > best_f1:
                    best_f1, threshold = f1, float(thr)

        return lgb_model, iso, threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_end_idx: int,
    ) -> "MetaLabeler":
        """
        Fit dual directional classifiers on IS data.

        Args:
            X: Feature matrix (gp_alpha + hmm_prob columns).
            y: TBM labels: +1 (Long TP), -1 (Short TP), 0 (neutral), 0.5 (fallback).
            is_end_idx: IS/OOS split index.
        """
        X_is = X.iloc[:is_end_idx].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_is = y.iloc[:is_end_idx].astype(np.float64)

        # Long: y=1 when Long TP hit
        y_long = (y_is == 1.0).astype(np.float64)
        # Short: y=1 when Short TP hit (= Long SL / price went down)
        y_short = (y_is == -1.0).astype(np.float64)

        self._lgb_long, self._iso_long, self._thr_long = self._fit_one(X_is, y_long)
        self._lgb_short, self._iso_short, self._thr_short = self._fit_one(X_is, y_short)

        return self

    def predict_proba_calibrated(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (calib_prob_long, calib_prob_short) arrays of shape (N,).

        Values represent the probability of each directional edge being present.
        Use these independently for long_entry and short_entry signals.
        """
        Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(Xv)

        def _predict(
            lgb_m: lgb.LGBMClassifier | None,
            iso: IsotonicRegression | None,
        ) -> np.ndarray:
            if lgb_m is None or iso is None:
                return np.full(n, 0.5, dtype=np.float64)
            raw = lgb_m.predict_proba(Xv)[:, 1]
            cal = iso.predict(raw)
            out = np.asarray(cal, dtype=np.float64)
            return cast(np.ndarray, out)

        prob_long = _predict(self._lgb_long, self._iso_long)
        prob_short = _predict(self._lgb_short, self._iso_short)
        return prob_long, prob_short

    @property
    def threshold_long(self) -> float:
        return self._thr_long

    @property
    def threshold_short(self) -> float:
        return self._thr_short
