"""Typed forecast contracts for alpha, cost, and risk layers."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class AlphaArtifactHash:
    """Granular artifact hash for IS/HO/OOS consistency validation."""

    alpha_config_hash: str
    feature_config_hash: str
    label_config_hash: str
    train_window_hash: str
    fold_spec_hash: str
    model_family: str
    selected_horizon: int

    def combined(self) -> str:
        """Return a 16-char hex digest of all hash fields combined.

        Returns:
            Hex-string (16 chars) of SHA-256 over sorted JSON payload.

        """
        payload: dict[str, Any] = {
            "alpha_config_hash": self.alpha_config_hash,
            "feature_config_hash": self.feature_config_hash,
            "label_config_hash": self.label_config_hash,
            "train_window_hash": self.train_window_hash,
            "fold_spec_hash": self.fold_spec_hash,
            "model_family": self.model_family,
            "selected_horizon": self.selected_horizon,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def structural_hash(self) -> str:
        """Return a 16-char hex digest of config-only fields (stable across IS/HO/OOS splits).

        Excludes train_window_hash and fold_spec_hash which differ per data window.
        Use this for cross-split consistency validation in final_evaluator.

        Returns:
            Hex-string (16 chars) of SHA-256 over config-only sorted JSON payload.

        """
        payload: dict[str, Any] = {
            "alpha_config_hash": self.alpha_config_hash,
            "feature_config_hash": self.feature_config_hash,
            "label_config_hash": self.label_config_hash,
            "model_family": self.model_family,
            "selected_horizon": self.selected_horizon,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(slots=True, frozen=True)
class AlphaForecast:
    """Typed alpha forecast contract — gross EV prediction per symbol per bar."""

    datetimes: np.ndarray
    symbols: tuple[str, ...]
    alpha_long_2d: np.ndarray
    alpha_short_2d: np.ndarray
    q10_long_2d: np.ndarray | None
    q50_long_2d: np.ndarray | None
    q90_long_2d: np.ndarray | None
    q10_short_2d: np.ndarray | None
    q50_short_2d: np.ndarray | None
    q90_short_2d: np.ndarray | None
    confidence_long_2d: np.ndarray | None
    confidence_short_2d: np.ndarray | None
    eligible_mask: np.ndarray
    source: str
    artifact_hash: AlphaArtifactHash


@dataclass(slots=True, frozen=True)
class CostForecast:
    """Typed cost forecast contract — execution friction per symbol per bar."""

    execution_cost_bps_2d: np.ndarray
    execution_cost_fraction_2d: np.ndarray
    uncertainty_bps_2d: np.ndarray
    capacity_notional_2d: np.ndarray | None
    source: str
    components: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RiskForecast:
    """Typed risk forecast contract — covariance, beta, residual variance, vol."""

    covariance_3d: np.ndarray
    beta_2d: np.ndarray | None
    residual_var_2d: np.ndarray | None
    forecast_vol_2d: np.ndarray
    beta_source: str  # raw_column | trailing_btc | trailing_equal_weight_market | unavailable
    source: str
