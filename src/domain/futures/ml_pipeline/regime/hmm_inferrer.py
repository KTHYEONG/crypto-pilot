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
from src.domain.futures.ml_pipeline.regime.crisis_detector import CrisisDetector
from src.domain.futures.ml_pipeline.regime.jax_hmm import JAXMultivariateHMM
from src.domain.futures.ml_pipeline.regime.policy_mapper import map_policy_controls
from src.domain.futures.ml_pipeline.regime.regime_contracts import (
    LEGACY_HMM_PROB_COLUMNS,
    REGIME_STATE_COLUMNS,
    derive_legacy_hmm_prob_frame,
    normalize_regime_state_frame,
)
from src.domain.futures.ml_pipeline.regime.tail_overlay import compute_tail_hazard_overlay

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_logger = logging.getLogger(__name__)


def _tail_zscore(ser: pd.Series, window: int = 168) -> np.ndarray:
    """Standard rolling z-score for tail feature engineering (optimized).

    Args:
        ser: Input series.
        window: Rolling window size.

    Returns:
        Clipped z-score array of shape (n,), dtype float64.

    """
    # Use standard mean/std for performance as requested. 
    # pd.to_numeric and fillna are kept for robustness but minimized.
    x = ser.to_numpy(dtype=np.float64) if isinstance(ser, pd.Series) else np.asarray(ser, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    
    s = pd.Series(x)
    min_p = max(12, window // 8)
    roll = s.rolling(window, min_periods=min_p)
    mu = roll.mean()
    sigma = roll.std(ddof=0)
    
    z = (s - mu) / sigma.replace(0.0, np.nan)
    return z.fillna(0.0).clip(-4.0, 4.0).to_numpy(dtype=np.float64)

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
def _numba_decay_max(signal: np.ndarray, initial_boost: float, decay_factor: float) -> np.ndarray:
    n = len(signal)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    curr = 0.0
    for i in range(n):
        if signal[i] > 0:
            curr = initial_boost
        else:
            curr *= decay_factor
        out[i] = curr
    return out


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
    sticky_min_duration: tuple[int, int, int, int] = (32, 16, 14, 12)
    backend: str = "jax_gaussian"
    tvtp_config: dict[str, float] = field(default_factory=dict)
    _model: Any = field(init=False, repr=False)
    _backend_name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sticky_min_duration = tuple(max(1, int(v)) for v in self.sticky_min_duration)
        backend_req = str(self.backend or "jax_gaussian").strip().lower()
        
        # Default fallback
        self._model = JAXMultivariateHMM(
            n_iter=self.n_iter,
            tol=self.tol,
            tvtp_config=self.tvtp_config,
        )
        self._backend_name = "jax_gaussian"
        
        if backend_req == "student_t":
            student_model = self._build_student_t_model()
            if student_model is not None:
                self._model = student_model
                self._backend_name = "student_t"
            else:
                _logger.warning("HMM backend=student_t requested but unavailable; fallback=jax_gaussian")
        elif backend_req == "skewed_t":
            skewed_model = self._build_skewed_t_model()
            if skewed_model is not None:
                self._model = skewed_model
                self._backend_name = "skewed_t"
            else:
                _logger.warning("HMM backend=skewed_t requested but unavailable; fallback=jax_gaussian")

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

    def _build_skewed_t_model(self) -> Any | None:
        try:
            from src.domain.futures.ml_pipeline.regime.skewed_t_hmm import (
                SkewedTMultivariateHMM,
            )
            return SkewedTMultivariateHMM(n_iter=self.n_iter, tol=self.tol)
        except Exception as exc:
            _logger.error("Failed to load SkewedTMultivariateHMM: %s", exc)
            return None

    def _fit_filter_probs(self, obs_df: pd.DataFrame, is_end_idx: int) -> pd.DataFrame:
        if hasattr(self._model, "fit_filter_train_oos"):
            return self._model.fit_filter_train_oos(obs_df, is_end_idx=is_end_idx)
        if hasattr(self._model, "fit_predict"):
            _logger.warning("Selected backend lacks fit_filter_train_oos; using fit_predict fallback")
            return self._model.fit_predict(obs_df)
        raise AttributeError(f"HMM backend '{self._backend_name}' does not provide inference API")

    def _build_obs_df(self, features_df: pd.DataFrame, returns_ser: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Build 4-feature observation frame for HMM (Action 2).

        Features:
            f1: Trend           (macro_trend_168h)
            f2: Volatility      (macro_vol_24h)
            f3: Downside Vol    (macro_downside_vol_24h)
            f4: CS Dispersion   (macro_cs_dispersion_24h)
            f5: OI Delta        (macro_oi_delta_24h)
            f6: Funding Mom     (macro_funding_mom_24h)
        """
        idx = features_df.index
        
        # Check if systemic features are already in features_df (standard pipeline path)
        needed_base = [
            "macro_trend_168h", "macro_vol_24h", "macro_downside_vol_24h",
            "macro_cs_dispersion_24h", "macro_oi_delta_24h", "macro_funding_mom_24h",
        ]
        # Phase C: universe breadth features — always attempt to include; pad 0 if absent.
        breadth_extras = ["macro_breadth_ma20", "macro_alt_btc_rs_24h"]
        if all(c in features_df.columns for c in needed_base):
            obs_data: dict[str, pd.Series] = {
                f"f{i+1}": features_df[c] for i, c in enumerate(needed_base)
            }
            # f7, f8: Phase C breadth extras (0.0 if not yet computed)
            obs_data["f7"] = features_df[breadth_extras[0]] if breadth_extras[0] in features_df.columns else pd.Series(0.0, index=idx)
            obs_data["f8"] = features_df[breadth_extras[1]] if breadth_extras[1] in features_df.columns else pd.Series(0.0, index=idx)
            obs_df = pd.DataFrame(obs_data, index=idx)
            vol_z = obs_df["f2"]
        else:
            # Fallback for direct usage without full systemic feature builder
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

            # f3: Downside vol proxy
            ret_1h = close.pct_change().fillna(0.0)
            down_vol = ret_1h.clip(upper=0.0).rolling(24).std().fillna(0.0)
            f3_z = (down_vol - down_vol.rolling(168, min_periods=24).mean()) / down_vol.rolling(168, min_periods=24).std().replace(0.0, np.nan)

            # f4: Sell-off Pressure or Negative Acceleration
            if "volume" in features_df.columns:
                # Sell-off Pressure: abs(ret) * volume when ret < 0
                vol_ser = pd.to_numeric(features_df["volume"], errors="coerce").fillna(0.0)
                sell_off = (ret_1h.clip(upper=0.0).abs() * vol_ser).rolling(24).sum().fillna(0.0)
                f4_z = pd.Series(_tail_zscore(sell_off), index=idx)
            else:
                # Negative Acceleration: (ret_t - ret_t-1) when accelerating downwards
                accel = ret_1h.diff().clip(upper=0.0).abs().rolling(24).mean().fillna(0.0)
                f4_z = pd.Series(_tail_zscore(accel), index=idx)

            # f5~f8 fallback: use zeros if not provided in direct usage
            zeros = pd.Series(0.0, index=idx)
            obs_df = pd.DataFrame(
                {"f1": f1_z, "f2": f2_z, "f3": f3_z, "f4": f4_z,
                 "f5": zeros, "f6": zeros, "f7": zeros, "f8": zeros},
                index=idx,
            )
            vol_z = f2_z

        obs_df = obs_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return obs_df, vol_z.reindex(idx).fillna(0.0)

    def _map_to_canonical_states(self, jax_probs: pd.DataFrame, vol_z: pd.Series) -> pd.DataFrame:
        idx = jax_probs.index

        if "bull_calm" in jax_probs.columns:
            # 5-state 직접 매핑 (HMM이 이미 bull_calm / bull_volatile 분리)
            p_calm = pd.to_numeric(jax_probs.get("bull_calm", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            p_vol = pd.to_numeric(jax_probs.get("bull_volatile", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            p_off = pd.to_numeric(jax_probs.get("bear_trend", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            p_chop = pd.to_numeric(jax_probs.get("chop", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            p_pre = pd.to_numeric(jax_probs.get("pre_crisis", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            # pre_crisis를 bear_trend에 합산 → canonical 4-state 유지
            p_off = p_off + p_pre
            # Vol-Scale(Calm) FAIL② 보정:
            # p_calm에 vol_z 역가중치(inverse vol weight)를 직접 곱해서
            # 고변동성 구간에서 p_calm이 dominant가 되지 않도록 억제
            vol_z_np = vol_z.to_numpy(dtype=np.float64)
            vol_z_smooth = pd.Series(vol_z_np).ewm(span=24, adjust=False).mean().to_numpy(dtype=np.float64)
            # inv_vol_weight: vol_z=-2→~1.0, vol_z=0→~0.5, vol_z=1→~0.27, vol_z=2→~0.12
            # 이를 통해 고변동성 시 p_calm이 자연스럽게 작아져 dominant 되지 않음
            inv_vol_weight = expit(-vol_z_smooth * 2.5)  # soft: vol_z=0 → 0.5 weight
            p_calm_weighted = p_calm * inv_vol_weight
            # 줄어든 calm의 절반을 volatile로, 절반을 chop으로 분산
            leaked = p_calm - p_calm_weighted
            p_calm = p_calm_weighted
            p_vol = p_vol + leaked * 0.7
            p_chop = p_chop + leaked * 0.3
        else:
            # 기존 4-state fallback (하위 호환: bull_trend / bear_trend / chop_high / chop_low)
            p_bull = pd.to_numeric(jax_probs.get("bull_trend", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            p_bear = pd.to_numeric(jax_probs.get("bear_trend", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            p_chop_h = pd.to_numeric(jax_probs.get("chop_high", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            p_chop_l = pd.to_numeric(jax_probs.get("chop_low", 0.25), errors="coerce").fillna(0.25).to_numpy(dtype=np.float64)
            vol_z_np = vol_z.to_numpy(dtype=np.float64)
            vol_z_smooth = pd.Series(vol_z_np).ewm(span=24, adjust=False).mean().to_numpy(dtype=np.float64)
            split = expit(np.clip(vol_z_smooth, -8.0, 8.0))
            p_calm = p_bull * (1.0 - split)
            p_vol = p_bull * split
            p_off = p_bear
            p_chop = p_chop_h + p_chop_l

        prob4 = np.column_stack([p_calm, p_vol, p_off, p_chop])
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

        # ── Step 3: supervised score 계산 ─────────────────────────────────────
        try:
            supervised_scores = self._compute_supervised_tail_score(
                legacy_df=legacy_df,
                returns_ser=returns_ser,
                features_df=features_df,
                idx=idx,
                is_end_idx=is_end_idx,
            )
        except Exception as _sup_exc:
            _logger.warning("Supervised tail score computation raised: %s", _sup_exc)
            supervised_scores = None

        # ── Step 3.5: Supervised Boost (Vectorized) & Crisis Overlay ──────────
        overlay_supervised_score: dict[str, np.ndarray] | None = None
        if supervised_scores is not None:
            sup = np.nan_to_num(
                supervised_scores.get(
                    "sup_score_q10_h8",
                    supervised_scores.get("sup_score_soft", np.zeros(len(idx), dtype=np.float64)),
                )
            ).astype(np.float64)
            
            # Rank-based top-13% mask
            n_sup = len(sup)
            target_n = max(10, int(n_sup * 0.13))
            sorted_desc = np.sort(sup)[::-1]
            rank_threshold = float(sorted_desc[min(target_n - 1, n_sup - 1)])
            crisis_mask = (sup >= rank_threshold)

            # Optimized Numba decay-max boost
            boost_series = _numba_decay_max(crisis_mask.astype(np.int32), 0.95, 0.985)
            boosted_sup = np.clip(np.maximum(sup, boost_series), 0.0, 1.0)
            
            overlay_supervised_score = dict(supervised_scores)
            overlay_supervised_score["sup_score_q10_h8"] = boosted_sup
            overlay_supervised_score["sup_score_soft"] = boosted_sup
            
            _logger.info(
                "Step3.5 hazard-overlay boost | Market %s | extended_pct=%.3f",
                tf, float((boost_series > 0.0).mean())
            )

        # ── Crisis Detector overlay ────────────────────────────────────────────
        try:
            detector = CrisisDetector()
            det_score = detector.score(features_df.reindex(idx).fillna(0.0), returns_ser.reindex(idx).fillna(0.0))
            raw_crisis = legacy_df["hmm_prob_crisis"].to_numpy(dtype=np.float64)
            merged_crisis = np.clip(np.maximum(raw_crisis, det_score), 0.0, 1.0)
            delta = merged_crisis - raw_crisis
            bear_adj = np.clip(legacy_df["hmm_prob_bear_trend"].to_numpy(dtype=np.float64) - delta, 0.0, 1.0)

            legacy_df = legacy_df.copy()
            legacy_df["hmm_prob_crisis"] = merged_crisis
            legacy_df["hmm_prob_bear_trend"] = bear_adj

            prob_mat = legacy_df[list(LEGACY_HMM_PROB_COLUMNS)].to_numpy(dtype=np.float64)
            legacy_df[list(LEGACY_HMM_PROB_COLUMNS)] = prob_mat / np.maximum(prob_mat.sum(axis=1, keepdims=True), 1e-12)
        except Exception as exc:
            _logger.warning("CrisisDetector overlay failed: %s", exc)

        # ── Step 4: Final Hazard Overlay (Single Call) ────────────────────────
        hazard_res = compute_tail_hazard_overlay(
            features_df=features_df,
            regime_df=regime_df,
            returns_ser=returns_ser,
            cfg=OPT_FUTURES_CONFIG,
            supervised_score=overlay_supervised_score,
        )
        hazard_df = hazard_res.hazards

        # ── Step 5: Adaptive Policy Thresholds ────────────────────────────────
        try:
            realized_haz = hazard_df["realized_crisis_hazard"]
            # Optimized rolling quantile for thresholds (kept for policy accuracy)
            roll_haz = realized_haz.rolling(168, min_periods=48)
            dynamic_cfg = dict(OPT_FUTURES_CONFIG)
            dynamic_cfg["FUTURES_POLICY_DEFENSE_SOFT_THR"] = roll_haz.quantile(0.65).fillna(0.48).to_numpy()
            dynamic_cfg["FUTURES_POLICY_DEFENSE_HARD_THR"] = roll_haz.quantile(0.85).fillna(0.66).to_numpy()
            dynamic_cfg["FUTURES_POLICY_DEFENSE_NEAR_FLAT_THR"] = roll_haz.quantile(0.95).fillna(0.84).to_numpy()
        except Exception as _thr_exc:
            _logger.warning("Adaptive thresholds failed: %s", _thr_exc)
            dynamic_cfg = OPT_FUTURES_CONFIG

        policy_df = map_policy_controls(regime_df=regime_df, hazard_df=hazard_df, cfg=dynamic_cfg)

        out = pd.concat([regime_df, legacy_df, hazard_df, policy_df], axis=1)
        out["hmm_tail_risk_8bar"] = hazard_df["tail_hazard_8h"].to_numpy(dtype=np.float64)
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        out = out.reset_index().rename(columns={out.index.name or "index": "datetime"})
        return out

    @staticmethod
    def _compute_supervised_tail_score(
        legacy_df: pd.DataFrame,
        returns_ser: pd.Series,
        features_df: pd.DataFrame,
        idx: pd.Index,
        is_end_idx: int,
    ) -> dict[str, np.ndarray] | None:
        """IS supervised logistic+isotonic score for tail events.

        Causal: trains on [:is_end_idx] only, applies to full timeline.

        Args:
            legacy_df: Legacy HMM probability frame (hmm_prob_bear_trend, etc.).
            returns_ser: Return series.
            features_df: Feature dataframe (macro_liq_proxy_24h, etc.).
            idx: Full timeline index.
            is_end_idx: Exclusive end of in-sample window.

        Returns:
            Calibrated multi-score dict or None on failure.

        """
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            _logger.warning("scikit-learn not available; skipping supervised tail score")
            return None

        try:
            n = len(idx)
            cfg = OPT_FUTURES_CONFIG
            if is_end_idx < 50:
                return None

            p_calm = pd.to_numeric(
                legacy_df.get("hmm_prob_bull_calm", pd.Series(0.25, index=idx)),
                errors="coerce",
            ).fillna(0.25).to_numpy(dtype=np.float64)
            p_off = pd.to_numeric(
                legacy_df.get("hmm_prob_bear_trend", pd.Series(0.25, index=idx)),
                errors="coerce",
            ).fillna(0.25).to_numpy(dtype=np.float64)
            p_chop = pd.to_numeric(
                legacy_df.get("hmm_prob_chop", pd.Series(0.25, index=idx)),
                errors="coerce",
            ).fillna(0.25).to_numpy(dtype=np.float64)
            ent = pd.to_numeric(
                legacy_df.get("regime_entropy", pd.Series(0.5, index=idx)),
                errors="coerce",
            ).fillna(0.5).to_numpy(dtype=np.float64)

            r_ser = returns_ser.reindex(idx).fillna(0.0).astype(np.float64)
            # Use r_ser (Series) directly to avoid repeated Series creation with index
            rv24_z = _tail_zscore(r_ser.abs().rolling(24, min_periods=4).std().fillna(0.0))
            down24_z = _tail_zscore(r_ser.clip(upper=0.0).abs().rolling(24, min_periods=4).std().fillna(0.0))
            mom6_z = _tail_zscore(r_ser.rolling(6, min_periods=2).mean().fillna(0.0))
            mom24_z = _tail_zscore(r_ser.rolling(24, min_periods=4).mean().fillna(0.0))
            
            vol24 = r_ser.rolling(24, min_periods=4).std().fillna(0.0)
            vol96 = r_ser.rolling(96, min_periods=12).std().replace(0.0, np.nan).fillna(vol24 + 1e-9)
            vol_ratio_z = _tail_zscore((vol24 / vol96).replace([np.inf, -np.inf], np.nan).fillna(0.0))
            
            px = (1.0 + r_ser).cumprod()
            dd96 = (px / px.rolling(96, min_periods=8).max() - 1.0).fillna(0.0)
            dd96_z = _tail_zscore(dd96)
            
            spread_off_calm = _tail_zscore(pd.Series(p_off - p_calm, index=idx))
            spread_off_chop = _tail_zscore(pd.Series(p_off - p_chop, index=idx))

            r_np = r_ser.to_numpy(dtype=np.float64)

            f = features_df.reindex(idx)
            if "macro_liq_proxy_24h" in f.columns:
                liq24_z = _tail_zscore(pd.to_numeric(f["macro_liq_proxy_24h"], errors="coerce").fillna(0.0))
            else:
                liq24_z = np.zeros(n, dtype=np.float64)
            
            # Micro-structure features activation
            if "macro_oi_delta_24h" in f.columns:
                oi24_z = _tail_zscore(pd.to_numeric(f["macro_oi_delta_24h"], errors="coerce").fillna(0.0))
            else:
                oi24_z = np.zeros(n, dtype=np.float64)
            
            if "macro_funding_mom_24h" in f.columns:
                funding24_z = _tail_zscore(pd.to_numeric(f["macro_funding_mom_24h"], errors="coerce").fillna(0.0))
            else:
                funding24_z = np.zeros(n, dtype=np.float64)

            jump_z = _tail_zscore(pd.Series(np.clip(-r_np, 0.0, None), index=idx))

            X = np.column_stack([
                p_off, p_chop, ent, rv24_z, down24_z, liq24_z, jump_z,
                mom6_z, mom24_z, vol_ratio_z, dd96_z, spread_off_calm, spread_off_chop,
                oi24_z, funding24_z
            ]).astype(np.float64)

            horizons_raw = cfg.get("FUTURES_HMM_SUP_HORIZONS", [4, 8, 16])
            try:
                horizons = tuple(int(h) for h in horizons_raw)
            except Exception:
                horizons = (4, 8, 16)
            horizons = tuple(sorted({h for h in horizons if h > 1})) or (4, 8, 16)
            quantiles = {
                "q10": float(np.clip(cfg.get("FUTURES_HMM_SUP_LABEL_Q10", 0.10), 0.001, 0.40)),
                "q05": float(np.clip(cfg.get("FUTURES_HMM_SUP_LABEL_Q05", 0.05), 0.001, 0.30)),
                "q03": float(np.clip(cfg.get("FUTURES_HMM_SUP_LABEL_Q03", 0.03), 0.001, 0.20)),
            }
            min_pos = int(max(5, cfg.get("FUTURES_HMM_SUP_MIN_POS", 12)))
            rank_blend_w = float(np.clip(cfg.get("FUTURES_HMM_SUP_RANK_BLEND_W", 0.30), 0.0, 0.95))
            rank_blend_pow = float(np.clip(cfg.get("FUTURES_HMM_SUP_RANK_BLEND_POW", 1.20), 0.5, 3.0))

            fwd_map: dict[int, pd.Series] = {}
            for h in horizons:
                # Optimized vectorized look-forward minimum (worst) return
                fwd_worst = r_ser.shift(-h).rolling(window=h, min_periods=1).min()
                fwd_map[h] = fwd_worst.replace([np.inf, -np.inf], np.nan)

            X_is = X[:is_end_idx]
            scores: dict[str, np.ndarray] = {}
            for h in horizons:
                fwd_ser = fwd_map[h]
                valid_is = fwd_ser.iloc[:is_end_idx].dropna()
                if len(valid_is) < 20:
                    continue
                for q_name, q_v in quantiles.items():
                    q_thr = float(valid_is.quantile(q_v))
                    if not np.isfinite(q_thr):
                        continue
                    y_is = (fwd_ser.iloc[:is_end_idx].fillna(0.0).to_numpy() <= q_thr).astype(int)
                    if int(y_is.sum()) < min_pos:
                        continue
                    try:
                        import lightgbm as lgb
                        lgb_params = {
                            "objective": "binary",
                            "metric": "auc",
                            "n_estimators": 200,
                            "learning_rate": 0.05,
                            "num_leaves": 15,
                            "min_child_samples": max(5, int(y_is.sum() * 0.1)),
                            "scale_pos_weight": max(1.0, (len(y_is) - int(y_is.sum())) / max(1, int(y_is.sum()))),
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "reg_alpha": 0.1,
                            "reg_lambda": 1.0,
                            "random_state": 42,
                            "verbose": -1,
                            "n_jobs": 1,
                        }
                        clf = lgb.LGBMClassifier(**lgb_params)
                        clf.fit(X_is, y_is)
                        prob_is = clf.predict_proba(X_is)[:, 1]
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(prob_is, y_is.astype(np.float64))
                        prob_all = clf.predict_proba(X)[:, 1]
                        score_iso = np.clip(iso.transform(prob_all).astype(np.float64), 0.0, 1.0)
                    except Exception as _lgb_exc:
                        _logger.warning("LightGBM supervised tail failed (%s); fallback to LogisticRegression", _lgb_exc)
                        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)
                        clf.fit(X_is, y_is)
                        prob_is = clf.predict_proba(X_is)[:, 1]
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(prob_is, y_is.astype(np.float64))
                        prob_all = clf.predict_proba(X)[:, 1]
                        score_iso = np.clip(iso.transform(prob_all).astype(np.float64), 0.0, 1.0)
                    score_rank = pd.Series(score_iso).rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=np.float64)
                    score_rank = np.clip(score_rank**rank_blend_pow, 0.0, 1.0)
                    score = np.clip((1.0 - rank_blend_w) * score_iso + rank_blend_w * score_rank, 0.0, 1.0)
                    scores[f"sup_score_{q_name}_h{h}"] = score

            if "sup_score_q10_h4" in scores:
                scores["sup_score_soft"] = scores["sup_score_q10_h4"]
            elif "sup_score_q10_h8" in scores:
                scores["sup_score_soft"] = scores["sup_score_q10_h8"]
            if "sup_score_q05_h8" in scores:
                scores["sup_score_hard"] = scores["sup_score_q05_h8"]
            elif "sup_score_q05_h16" in scores:
                scores["sup_score_hard"] = scores["sup_score_q05_h16"]
            if "sup_score_q03_h16" in scores:
                scores["sup_score_near_flat"] = scores["sup_score_q03_h16"]
            elif "sup_score_q03_h8" in scores:
                scores["sup_score_near_flat"] = scores["sup_score_q03_h8"]
            if "sup_score_q10_h8" not in scores and "sup_score_soft" in scores:
                scores["sup_score_q10_h8"] = scores["sup_score_soft"]
            if not scores:
                _logger.warning("Supervised tail: no valid multi-label model")
                return None
            _logger.info(
                "Step3 supervised tail score | is_end=%d | n=%d | multi_scores=%d | soft_mean=%.3f",
                is_end_idx,
                n,
                len(scores),
                float(np.mean(scores.get("sup_score_soft", scores.get("sup_score_q10_h8", np.zeros(n, dtype=np.float64))))),
            )
            return scores
        except Exception as exc:
            _logger.warning("_compute_supervised_tail_score failed: %s; no supervised score", exc)
            return None

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
    sticky_raw = conf.get("FUTURES_HMM_OUTPUT_STICKY_MIN_DURATION", [132, 56, 28, 16])
    try:
        sticky_tuple = tuple(int(v) for v in sticky_raw)
    except Exception:
        sticky_tuple = (132, 56, 28, 16)
    if len(sticky_tuple) < 4:
        sticky_tuple = (132, 56, 28, 16)
    else:
        sticky_tuple = sticky_tuple[:4]
    backend = str(conf.get("FUTURES_HMM_BACKEND", "jax_gaussian") or "jax_gaussian").strip().lower()
    if backend not in {"jax_gaussian", "student_t", "skewed_t"}:
        backend = "jax_gaussian"
    tvtp_config = {
        "enabled": float(bool(conf.get("FUTURES_HMM_TVTP_ENABLED", True))),
        "vol_center": float(conf.get("FUTURES_HMM_TVTP_VOL_CENTER", 0.0)),
        "vol_scale": float(conf.get("FUTURES_HMM_TVTP_VOL_SCALE", 1.0)),
        "diag_slope": float(conf.get("FUTURES_HMM_TVTP_DIAG_SLOPE", -0.24)),
        "diag_bias": float(conf.get("FUTURES_HMM_TVTP_DIAG_BIAS", 0.0)),
        "diag_clip": float(conf.get("FUTURES_HMM_TVTP_DIAG_CLIP", 0.34)),
        "sticky_prior_base": float(conf.get("FUTURES_HMM_STICKY_PENALTY_WEIGHT", 1100.0)),
        "sticky_prior_vol_slope": float(conf.get("FUTURES_HMM_TVTP_STICKY_PRIOR_VOL_SLOPE", -0.30)),
        "sticky_prior_min_mult": float(conf.get("FUTURES_HMM_TVTP_STICKY_PRIOR_MIN_MULT", 0.90)),
        "sticky_prior_max_mult": float(conf.get("FUTURES_HMM_TVTP_STICKY_PRIOR_MAX_MULT", 1.28)),
    }
    return HMMStateInferrer(
        n_iter=int(conf.get("FUTURES_HMM_N_ITER", 1500)),
        tol=float(conf.get("FUTURES_HMM_TOL", 1e-4)),
        sticky_min_duration=sticky_tuple,
        backend=backend,
        tvtp_config=tvtp_config,
    )
