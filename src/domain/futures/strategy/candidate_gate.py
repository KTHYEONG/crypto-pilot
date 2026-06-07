from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_dataset import CandidateDataset
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(slots=True, frozen=True)
class GateValidationReport:
    enabled: bool
    threshold: float
    raw_brier: float
    calibrated_brier: float
    base_brier: float
    brier_skill: float
    roc_auc: float
    average_precision: float
    decile_lift: float
    incremental_log_growth_lcb: float
    reason: str


@dataclass(slots=True, frozen=True)
class CandidateGateModel:
    model: None
    calibrator: None
    feature_names: tuple[str, ...]
    train_window: tuple[int, int]
    valid_window: tuple[int, int]
    calibration_method: str
    calibration_used: bool
    calibration_reason: str
    validation: GateValidationReport


_logger = logging.getLogger(__name__)


def fit_candidate_gate(
    *,
    train: CandidateDataset,
    early_stop: CandidateDataset,
    calibration: CandidateDataset,
    calibration_eval: CandidateDataset | None = None,
    cfg: CandidateStrategyConfig,
) -> CandidateGateModel:
    """Return a placeholder gate model.

    The gate classifier was removed. Effective gate probabilities are now derived
    downstream from edge quantiles via the q10/upside ratio.
    """
    _ = early_stop
    eval_set = calibration_eval if calibration_eval is not None else calibration
    valid_n = int(eval_set.X.shape[0])
    valid_pos = int(eval_set.y_gate.sum()) if valid_n > 0 else 0
    valid_neg = valid_n - valid_pos
    validation = GateValidationReport(
        enabled=True,
        threshold=0.35,
        raw_brier=0.0,
        calibrated_brier=0.0,
        base_brier=0.0,
        brier_skill=0.0,
        roc_auc=0.0,
        average_precision=0.0,
        decile_lift=0.0,
        incremental_log_growth_lcb=0.0,
        reason="gate_replaced_by_q10_mu_ratio",
    )
    _logger.info(
        "[DIAG][GATE_FIT] train_n=%d valid_n=%d valid_pos=%d valid_neg=%d reason=%s",
        int(train.X.shape[0]),
        valid_n,
        valid_pos,
        valid_neg,
        validation.reason,
    )
    return CandidateGateModel(
        model=None,
        calibrator=None,
        feature_names=tuple(train.feature_names),
        train_window=(0, int(train.X.shape[0])),
        valid_window=(0, int(calibration.X.shape[0])),
        calibration_method="none",
        calibration_used=False,
        calibration_reason="gate_model_removed_ratio_discount",
        validation=validation,
    )


def predict_candidate_gate(
    *,
    model: CandidateGateModel | None,
    dataset: CandidateDataset,
    cfg: CandidateStrategyConfig | None = None,
) -> NDArray[np.float64]:
    """Return neutral pass probabilities.

    The returned array preserves the previous API shape; downstream edge logic
    replaces it with the q10/upside ratio.
    """
    _ = (model, cfg)
    if dataset.X.shape[0] == 0:
        return np.zeros(dataset.X.shape[0], dtype=np.float64)
    probs = np.ones(dataset.X.shape[0], dtype=np.float64)
    _logger.debug(
        (
            "[DIAG][GATE] n=%d mean_p=%.4f median_p=%.4f max_p=%.4f "
            "pct_ge55=%.3f pct_ge50=%.3f pct_ge45=%.3f calibrated=%s"
        ),
        len(probs),
        float(probs.mean()) if len(probs) > 0 else float("nan"),
        float(np.median(probs)) if len(probs) > 0 else float("nan"),
        float(probs.max()) if len(probs) > 0 else float("nan"),
        float((probs >= 0.55).mean()) if len(probs) > 0 else 0.0,
        float((probs >= 0.50).mean()) if len(probs) > 0 else 0.0,
        float((probs >= 0.45).mean()) if len(probs) > 0 else 0.0,
        False,
    )
    return probs
