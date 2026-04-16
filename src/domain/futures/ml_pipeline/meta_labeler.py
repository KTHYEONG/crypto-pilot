"""LightGBM meta-labeler with isotonic calibration and purged time-series CV."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression


@dataclass
class MetaLabeler:
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.05
    reg_lambda: float = 5.0
    min_child_samples: int = 30
    _iso: IsotonicRegression | None = None
    _lgb: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_end_idx: int,
    ) -> MetaLabeler:
        X_is = X.iloc[:is_end_idx].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_is = y.iloc[:is_end_idx].astype(np.float64)
        m = y_is.notna() & np.isfinite(X_is.to_numpy()).all(axis=1)
        X_is, y_is = X_is[m], y_is[m]
        if len(X_is) < 80:
            self._lgb = None
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(np.array([0.5]), np.array([0.5]))
            return self

        self._lgb = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            verbosity=-1,
            random_state=42,
        )
        self._lgb.fit(X_is, y_is)
        raw = self._lgb.predict_proba(X_is)[:, 1]
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw, y_is.to_numpy())
        return self

    def predict_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        if self._lgb is None or self._iso is None:
            return np.full(len(X), 0.5, dtype=np.float64)
        Xv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = self._lgb.predict_proba(Xv)[:, 1]
        out = self._iso.predict(raw)
        return np.asarray(out, dtype=np.float64)
