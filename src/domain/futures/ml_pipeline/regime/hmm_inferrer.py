"""Regime inference orchestrator with canonical 4-state contract + overlays."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import expit

from config.opt_config import OPT_FUTURES_CONFIG
from src.core.indicators.indicators import global_ind
from src.domain.futures.ml_pipeline.regime.jax_hmm import JAXMultivariateHMM
from src.domain.futures.ml_pipeline.regime.policy_mapper import map_policy_controls
from src.domain.futures.ml_pipeline.regime.regime_contracts import (
    REGIME_STATE_COLUMNS,
    derive_legacy_hmm_prob_frame,
    normalize_regime_state_frame,
)
from src.domain.futures.ml_pipeline.regime.tail_overlay import compute_tail_hazard_overlay

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_logger = logging.getLogger(__name__)

HMM_SEMANTIC_PROB_COLUMNS: tuple[str, ...] = (
    "hmm_prob_bull_calm",
    "hmm_prob_bull_vol_up",
    "hmm_prob_bear_trend",
    "hmm_prob_chop",
    "hmm_prob_crisis",
)


@njit(cache=True, fastmath=True)
def _numba_sticky_labels(labels: np.ndarray, min_durations: np.ndarray) -> np.ndarray:
    n = len(labels)
    if n < 2:
        return labels
    result = labels.copy()
    i = 0
    while i < n:
        curr = result[i]
        j = i + 1
        while j < n and result[j] == curr:
            j += 1
        run_len = j - i
        m_dur = min_durations[int(curr)]
        if run_len < m_dur and i > 0:
            prev = result[i - 1]
            for kk in range(i, j):
                result[kk] = prev
        i = j
    return result


@njit(cache=True, fastmath=True)
def _numba_current_duration(hard_states: np.ndarray) -> np.ndarray:
    n = len(hard_states)
    dur_arr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return dur_arr
    c = 1.0
    dur_arr[0] = c
    for i in range(1, n):
        if hard_states[i] == hard_states[i - 1]:
            c += 1.0
        else:
            c = 1.0
        dur_arr[i] = c
    return dur_arr


def _safe_probs(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    row_sum = arr.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 1e-12
    if np.any(bad):
        arr[bad] = 1.0 / float(arr.shape[1])
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / np.maximum(row_sum, 1e-12)


def _entropy4(prob4: np.ndarray) -> np.ndarray:
    p = np.clip(prob4, 1e-12, 1.0)
    return (-np.sum(p * np.log(p), axis=1) / np.log(4.0)).astype(np.float64)


@dataclass
class HMMStateInferrer:
    n_states: int = 4
    n_iter: int = 1500
    tol: float = 1e-4
    sticky_min_duration: tuple[int, int, int, int] = (24, 12, 10, 8)
    backend: str = "jax_gaussian"
    _model: Any = field(init=False, repr=False)
    _backend_name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sticky_min_duration = tuple(max(1, int(v)) for v in self.sticky_min_duration)
        backend_req = str(self.backend or "jax_gaussian").strip().lower()
        self._model = JAXMultivariateHMM(n_iter=self.n_iter, tol=self.tol)
        self._backend_name = "jax_gaussian"
        if backend_req == "student_t":
            student_model = self._build_student_t_model()
            if student_model is not None:
                self._model = student_model
                self._backend_name = "student_t"
            else:
                _logger.warning("HMM backend=student_t requested but unavailable; fallback=jax_gaussian")

    def _build_student_t_model(self) -> Any | None:
        candidates: tuple[tuple[str, str], ...] = (
            ("src.domain.futures.ml_pipeline.regime.student_t_hmm", "StudentTRegimeHMM"),
            ("src.domain.futures.ml_pipeline.regime.student_t_hmm", "StudentTMultivariateHMM"),
            ("src.domain.futures.ml_pipeline.regime.student_t_hmm", "StudentTHMM"),
        )
        for module_name, cls_name in candidates:
            try:
                mod = __import__(module_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls(n_iter=self.n_iter, tol=self.tol)
            except Exception:
                continue
        return None

    def _fit_filter_probs(self, obs_df: pd.DataFrame, is_end_idx: int) -> pd.DataFrame:
        if hasattr(self._model, "fit_filter_train_oos"):
            return self._model.fit_filter_train_oos(obs_df, is_end_idx=is_end_idx)
        if hasattr(self._model, "fit_predict"):
            _logger.warning("Selected backend lacks fit_filter_train_oos; using fit_predict fallback")
            return self._model.fit_predict(obs_df)
        raise AttributeError(f"HMM backend '{self._backend_name}' does not provide inference API")

    def _build_obs_df(self, features_df: pd.DataFrame, returns_ser: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        idx = features_df.index
        close = (
            pd.to_numeric(features_df["close"], errors="coerce").astype(np.float64)
            if "close" in features_df.columns
            else returns_ser.cumsum().astype(np.float64)
        )
        ema12 = global_ind.calculate_ema(close, 12)
        ema144 = global_ind.calculate_ema(close, 144).replace(0.0, np.nan)
        f1_raw = (ema12 / ema144) - 1.0
        f1_z = (f1_raw - f1_raw.rolling(168, min_periods=24).mean()) / f1_raw.rolling(168, min_periods=24).std().replace(0.0, np.nan)

        if "high" in features_df.columns and "low" in features_df.columns:
            atr14 = global_ind.calculate_atr(features_df, 14)
        else:
            atr14 = returns_ser.abs().rolling(14, min_periods=4).mean() * close

        f2_raw = np.log((atr14 / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan))
        f2_z = (f2_raw - f2_raw.rolling(168, min_periods=24).mean()) / f2_raw.rolling(168, min_periods=24).std().replace(0.0, np.nan)

        obs_df = pd.DataFrame({"f1": f1_z, "f2": f2_z}, index=idx)
        obs_df = obs_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return obs_df, f2_z.reindex(idx).fillna(0.0)

    def _map_to_canonical_states(self, jax_probs: pd.DataFrame, vol_z: pd.Series) -> pd.DataFrame:
        idx = jax_probs.index
        p_bull = pd.to_numeric(jax_probs.get("bull_trend", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
        p_bear = pd.to_numeric(jax_probs.get("bear_trend", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
        p_chop_h = pd.to_numeric(jax_probs.get("chop_high", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
        p_chop_l = pd.to_numeric(jax_probs.get("chop_low", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)

        split = expit(np.clip(vol_z.to_numpy(dtype=np.float64), -8.0, 8.0))
        p_volatile = p_bull * split
        p_calm = p_bull * (1.0 - split)
        p_off = p_bear
        p_chop = p_chop_h + p_chop_l

        prob4 = np.column_stack([p_calm, p_volatile, p_off, p_chop])
        prob4 = _safe_probs(prob4)

        out = pd.DataFrame(prob4, index=idx, columns=REGIME_STATE_COLUMNS)
        out["regime_entropy"] = _entropy4(prob4)

        hard = np.argmax(prob4, axis=1).astype(np.int32)
        sticky = _numba_sticky_labels(hard, np.array(self.sticky_min_duration, dtype=np.int32))
        out["regime_hard_state"] = sticky.astype(np.float64)
        out["regime_current_duration"] = _numba_current_duration(sticky)
        return out

    def fit_predict_systemic(
        self,
        features_df: pd.DataFrame,
        returns_ser: pd.Series,
        is_end_idx: int,
        symbol: str = "Market",
        tf: str = "4h",
    ) -> pd.DataFrame:
        _logger.info("HMM regime inference | %s %s | n=%d | backend=%s", symbol, tf, len(features_df), self._backend_name)
        if len(features_df) < 50:
            return self._zeros_semantic(features_df)

        idx = features_df.index
        obs_df, vol_z = self._build_obs_df(features_df, returns_ser.reindex(idx).fillna(0.0))
        jax_probs = self._fit_filter_probs(obs_df, is_end_idx=int(max(1, min(is_end_idx, len(obs_df)))))

        regime_df = self._map_to_canonical_states(jax_probs.reindex(idx).ffill().bfill().fillna(0.0), vol_z)
        regime_df = normalize_regime_state_frame(regime_df).join(regime_df[["regime_entropy", "regime_hard_state", "regime_current_duration"]])

        legacy_df = derive_legacy_hmm_prob_frame(regime_df)

        hazard_res = compute_tail_hazard_overlay(
            features_df=features_df,
            regime_df=regime_df,
            returns_ser=returns_ser,
            cfg=OPT_FUTURES_CONFIG,
        )
        hazard_df = hazard_res.hazards
        policy_df = map_policy_controls(regime_df=regime_df, hazard_df=hazard_df, cfg=OPT_FUTURES_CONFIG)

        out = pd.concat([regime_df, legacy_df, hazard_df, policy_df], axis=1)
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        out = out.reset_index().rename(columns={out.index.name or "index": "datetime"})
        return out

    def _zeros_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        n = len(df)
        prob4 = np.full((n, 4), 0.25, dtype=np.float64)
        out = pd.DataFrame(prob4, index=idx, columns=REGIME_STATE_COLUMNS)
        out["regime_entropy"] = 1.0
        out["regime_hard_state"] = 0.0
        out["regime_current_duration"] = 1.0
        legacy = derive_legacy_hmm_prob_frame(out)
        hazard = pd.DataFrame(
            {
                "pre_crisis_hazard": np.zeros(n, dtype=np.float64),
                "realized_crisis_hazard": np.zeros(n, dtype=np.float64),
                "tail_hazard_4h": np.zeros(n, dtype=np.float64),
                "tail_hazard_8h": np.zeros(n, dtype=np.float64),
                "tail_hazard_24h": np.zeros(n, dtype=np.float64),
            },
            index=idx,
        )
        policy = map_policy_controls(out, hazard, cfg=OPT_FUTURES_CONFIG)
        merged = pd.concat([out, legacy, hazard, policy], axis=1)
        return merged.reset_index().rename(columns={merged.index.name or "index": "datetime"})


def build_hmm_inferrer_from_config(cfg: dict[str, object] | None = None, **kwargs) -> HMMStateInferrer:
    conf = OPT_FUTURES_CONFIG if cfg is None else cfg
    sticky_raw = conf.get("FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION", [24, 12, 10, 8])
    try:
        sticky_tuple = tuple(int(v) for v in sticky_raw)
    except Exception:
        sticky_tuple = (24, 12, 10, 8)
    if len(sticky_tuple) < 4:
        sticky_tuple = (24, 12, 10, 8)
    else:
        sticky_tuple = sticky_tuple[:4]
    backend = str(conf.get("FUTURES_HMM_BACKEND", "jax_gaussian") or "jax_gaussian").strip().lower()
    if backend not in {"jax_gaussian", "student_t"}:
        backend = "jax_gaussian"
    return HMMStateInferrer(
        n_iter=int(conf.get("FUTURES_HMM_N_ITER", 1500)),
        tol=float(conf.get("FUTURES_HMM_TOL", 1e-4)),
        sticky_min_duration=sticky_tuple,
        backend=backend,
    )
