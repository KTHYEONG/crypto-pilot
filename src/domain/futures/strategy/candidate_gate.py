from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from lightgbm import LGBMClassifier
from lightgbm import early_stopping as lgbm_early_stopping
from lightgbm import log_evaluation as lgbm_log_evaluation
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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
    model: LGBMClassifier
    calibrator: CalibratedClassifierCV | None
    feature_names: tuple[str, ...]
    train_window: tuple[int, int]
    valid_window: tuple[int, int]
    calibration_method: str
    calibration_used: bool
    calibration_reason: str
    validation: GateValidationReport


def _seed_from_cfg(cfg: Any) -> int:
    return int(getattr(cfg, "seed", getattr(getattr(cfg, "ml", cfg), "seed", 42)))


def fit_candidate_gate(
    *,
    train: CandidateDataset,
    early_stop: CandidateDataset,
    calibration: CandidateDataset,
    calibration_eval: CandidateDataset | None = None,
    cfg: CandidateStrategyConfig,
) -> CandidateGateModel:
    """Fit and calibrate candidate trade/no-trade classifier.

    Follows spec hyperparameters for strict regime/overfit reduction.
    """
    max_depth = int(getattr(cfg, "gate_lgbm_max_depth", 4))
    reg_lambda = float(getattr(cfg, "gate_lgbm_reg_lambda", 20.0))
    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=2**max_depth - 1 if max_depth > 0 else 15,
        max_depth=max_depth,
        min_child_samples=100,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=2.0,
        reg_lambda=reg_lambda,
        random_state=_seed_from_cfg(cfg),
        n_jobs=1,
        verbose=-1,
    )
    _es_cb = [lgbm_early_stopping(stopping_rounds=30, verbose=False), lgbm_log_evaluation(period=-1)]
    _gate_eval: list[tuple[Any, Any]] = (
        [(early_stop.X, early_stop.y_gate.astype(np.int32, copy=False))] if early_stop.X.shape[0] > 0 else []
    )
    model.fit(
        train.X,
        train.y_gate.astype(np.int32, copy=False),
        sample_weight=train.gate_weight,
        eval_set=_gate_eval if _gate_eval else None,
        callbacks=_es_cb if _gate_eval else None,  # type: ignore[arg-type]
    )

    calibration_method = str(cfg.gate_calibration_method)
    calibrator: CalibratedClassifierCV | None = None
    calibration_used = False
    calibration_reason = "calibration_disabled"
    eval_set = calibration_eval if calibration_eval is not None else calibration
    valid_n = int(eval_set.X.shape[0])
    valid_pos = int(eval_set.y_gate.sum()) if valid_n > 0 else 0
    valid_neg = valid_n - valid_pos
    raw_brier = float("nan")
    raw_std = float("nan")
    cal_brier = float("nan")
    cal_std = float("nan")
    validation_report = GateValidationReport(
        enabled=False,
        threshold=float(getattr(cfg, "selection_min_gate_probability_floor", 0.35)),
        raw_brier=float("nan"),
        calibrated_brier=float("nan"),
        base_brier=float("nan"),
        brier_skill=float("nan"),
        roc_auc=float("nan"),
        average_precision=float("nan"),
        decile_lift=float("nan"),
        incremental_log_growth_lcb=0.0,
        reason="not_evaluated",
    )
    if calibration_method == "none":
        calibration_reason = "calibration_method_none"
    elif valid_n < cfg.min_gate_calibration_obs:
        calibration_reason = "insufficient_calibration_rows"
    elif valid_pos < cfg.min_gate_calibration_pos or valid_neg < cfg.min_gate_calibration_pos:
        calibration_reason = "insufficient_calibration_class_balance"
    elif np.unique(calibration.y_gate).size < 2:
        calibration_reason = "single_class_calibration_target"
    else:
        raw_prob = cast(NDArray[np.float64], model.predict_proba(eval_set.X)[:, 1]).astype(
            np.float64,
            copy=False,
        )
        raw_std = float(np.std(raw_prob))
        raw_brier = float(brier_score_loss(eval_set.y_gate, raw_prob))
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=calibration_method,
            cv=None,
        )
        calibrator.fit(
            calibration.X,
            calibration.y_gate.astype(np.int32, copy=False),
            sample_weight=calibration.gate_weight,
        )
        cal_prob = cast(NDArray[np.float64], calibrator.predict_proba(eval_set.X)[:, 1]).astype(
            np.float64,
            copy=False,
        )
        cal_std = float(np.std(cal_prob))
        cal_brier = float(brier_score_loss(eval_set.y_gate, cal_prob))
        if not np.all(np.isfinite(cal_prob)):
            calibration_reason = "non_finite_calibration_probabilities"
            calibrator = None
        elif cal_std < cfg.min_gate_probability_std:
            calibration_reason = "calibration_probability_collapse"
            calibrator = None
        elif cal_brier > raw_brier + 1e-6:
            calibration_reason = "calibration_brier_regression"
            calibrator = None
        else:
            calibration_used = True
            calibration_reason = "calibration_accepted"
        base_rate = float(np.mean(eval_set.y_gate)) if valid_n > 0 else 0.0
        base_prob = np.full(valid_n, base_rate, dtype=np.float64)
        base_brier = float(brier_score_loss(eval_set.y_gate, base_prob)) if valid_n > 0 else float("nan")
        chosen_prob = cal_prob if calibration_used and calibrator is not None else raw_prob
        chosen_brier = float(brier_score_loss(eval_set.y_gate, chosen_prob)) if valid_n > 0 else float("nan")
        brier_skill = (
            float((base_brier - chosen_brier) / max(base_brier, 1e-12))
            if valid_n > 0 and np.isfinite(base_brier)
            else float("nan")
        )
        roc_auc = (
            float(roc_auc_score(eval_set.y_gate, chosen_prob))
            if valid_n > 0 and np.unique(eval_set.y_gate).size > 1
            else float("nan")
        )
        avg_precision = (
            float(average_precision_score(eval_set.y_gate, chosen_prob))
            if valid_n > 0 and np.unique(eval_set.y_gate).size > 1
            else float("nan")
        )
        decile_cut = float(np.quantile(chosen_prob, 0.9)) if chosen_prob.size > 0 else 1.0
        top_decile_mask = chosen_prob >= decile_cut
        top_decile_rate = float(np.mean(eval_set.y_gate[top_decile_mask])) if bool(top_decile_mask.any()) else base_rate
        decile_lift = float(top_decile_rate - base_rate)
        # Layer 1 simplification: gate is always active (catastrophic veto only).
        # brier_skill/decile_lift retained as diagnostics only — not used for gate enable/disable.
        enabled = True
        validation_report = GateValidationReport(
            enabled=enabled,
            threshold=float(getattr(cfg, "selection_min_gate_probability_floor", 0.35)),
            raw_brier=raw_brier,
            calibrated_brier=cal_brier if np.isfinite(cal_brier) else raw_brier,
            base_brier=base_brier,
            brier_skill=brier_skill,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            decile_lift=decile_lift,
            incremental_log_growth_lcb=0.0,
            reason="gate_always_active_catastrophic_veto",
        )

    _logger.info(
        (
            "[DIAG][GATE_FIT] train_n=%d valid_n=%d valid_pos=%d valid_neg=%d "
            "calibration_method=%s used=%s reason=%s "
            "raw_brier=%.6f raw_std=%.4f cal_brier=%.6f cal_std=%.4f"
        ),
        int(train.X.shape[0]),
        valid_n,
        valid_pos,
        valid_neg,
        calibration_method,
        calibration_used,
        calibration_reason,
        raw_brier,
        raw_std,
        cal_brier,
        cal_std,
    )

    return CandidateGateModel(
        model=model,
        calibrator=calibrator,
        feature_names=tuple(train.feature_names),
        train_window=(0, int(train.X.shape[0])),
        valid_window=(0, int(calibration.X.shape[0])),
        calibration_method=calibration_method,
        calibration_used=calibration_used,
        calibration_reason=calibration_reason,
        validation=validation_report,
    )


_logger = logging.getLogger(__name__)


def predict_candidate_gate(
    *,
    model: CandidateGateModel | None,
    dataset: CandidateDataset,
    cfg: CandidateStrategyConfig | None = None,
) -> NDArray[np.float64]:
    """Return calibrated pass probability for candidate events."""
    if model is None or dataset.X.shape[0] == 0:
        return np.zeros(dataset.X.shape[0], dtype=np.float64)
    # Gate is always active (catastrophic veto only). Always return calibrated p_pass.
    # calibrator is used when available; otherwise fall back to raw model probability.
    predictor = model.calibrator if model.calibration_used and model.calibrator is not None else model.model
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
        model.calibration_used,
    )
    return clipped
