from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(slots=True, frozen=True)
class CandidateGateModel:
    model: LGBMClassifier
    calibrator: CalibratedClassifierCV | None
    feature_names: tuple[str, ...]
    train_window: tuple[int, int]
    valid_window: tuple[int, int]


def _seed_from_cfg(cfg: Any) -> int:
    return int(getattr(cfg, "seed", getattr(getattr(cfg, "ml", cfg), "seed", 42)))


def fit_candidate_gate(
    *,
    train: CandidateDataset,
    valid: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateGateModel:
    """Fit and calibrate candidate trade/no-trade classifier.

    Follows spec hyperparameters for strict regime/overfit reduction.
    """
    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=100,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=2.0,
        reg_lambda=20.0,
        random_state=_seed_from_cfg(cfg),
        n_jobs=1,
        verbose=-1,
    )
    model.fit(
        train.X,
        train.y_gate.astype(np.int32, copy=False),
        sample_weight=train.sample_weight,
    )

    calibrator: CalibratedClassifierCV | None = None
    if valid.X.shape[0] > 0 and np.unique(valid.y_gate).size >= 2:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method="sigmoid",
            cv=None,
        )
        calibrator.fit(
            valid.X,
            valid.y_gate.astype(np.int32, copy=False),
            sample_weight=valid.sample_weight,
        )

    return CandidateGateModel(
        model=model,
        calibrator=calibrator,
        feature_names=tuple(train.feature_names),
        train_window=(0, int(train.X.shape[0])),
        valid_window=(0, int(valid.X.shape[0])),
    )


_logger = logging.getLogger(__name__)


def predict_candidate_gate(
    *,
    model: CandidateGateModel | None,
    dataset: CandidateDataset,
) -> NDArray[np.float64]:
    """Return calibrated pass probability for candidate events."""
    if model is None or dataset.X.shape[0] == 0:
        return np.zeros(dataset.X.shape[0], dtype=np.float64)

    predictor = model.calibrator if model.calibrator is not None else model.model
    probs = cast(NDArray[np.float64], predictor.predict_proba(dataset.X)[:, 1])
    clipped = np.clip(probs.astype(np.float64, copy=False), 0.0, 1.0)

    _logger.debug(
        "[DIAG][GATE] n=%d mean_p=%.4f median_p=%.4f max_p=%.4f "
        "pct_ge55=%.3f pct_ge50=%.3f pct_ge45=%.3f calibrated=%s",
        len(clipped),
        float(clipped.mean()) if len(clipped) > 0 else float("nan"),
        float(np.median(clipped)) if len(clipped) > 0 else float("nan"),
        float(clipped.max()) if len(clipped) > 0 else float("nan"),
        float((clipped >= 0.55).mean()) if len(clipped) > 0 else 0.0,
        float((clipped >= 0.50).mean()) if len(clipped) > 0 else 0.0,
        float((clipped >= 0.45).mean()) if len(clipped) > 0 else 0.0,
        model.calibrator is not None,
    )
    return clipped

