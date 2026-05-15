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
    """Robust z-score (IQR-based) for tail feature engineering.

    Args:
        ser: Input series.
        window: Rolling window size for median/IQR estimation.

    Returns:
        Clipped z-score array of shape (n,), dtype float64.
    """
    x = pd.to_numeric(ser, errors="coerce").fillna(0.0).astype(np.float64)
    min_p = max(12, window // 8)
    med = x.rolling(window, min_periods=min_p).median()
    q75 = x.rolling(window, min_periods=min_p).quantile(0.75)
    q25 = x.rolling(window, min_periods=min_p).quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)
    z = ((x - med) / iqr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.clip(-4.0, 4.0).to_numpy(dtype=np.float64)

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
    sticky_min_duration: tuple[int, int, int, int] = (32, 16, 14, 12)
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
        """Build 4-feature observation frame for HMM (Action 2).

        Features:
            f1: Trend           (macro_trend_168h)
            f2: Volatility      (macro_vol_24h)
            f3: Downside Vol    (macro_downside_vol_24h)
            f4: CS Dispersion   (macro_cs_dispersion_24h)
        """
        idx = features_df.index
        
        # Check if systemic features are already in features_df (standard pipeline path)
        needed = ["macro_trend_168h", "macro_vol_24h", "macro_downside_vol_24h", "macro_cs_dispersion_24h"]
        if all(c in features_df.columns for c in needed):
            obs_df = features_df[needed].copy()
            obs_df.columns = ["f1", "f2", "f3", "f4"]
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
            
            # f4: constant 0 as placeholder for CS dispersion
            f4_z = pd.Series(0.0, index=idx)

            obs_df = pd.DataFrame({"f1": f1_z, "f2": f2_z, "f3": f3_z, "f4": f4_z}, index=idx)
            vol_z = f2_z

        obs_df = obs_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return obs_df, vol_z.reindex(idx).fillna(0.0)

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

        # ── Step 3: supervised score 계산 (IS causal, regime_df 부스트에 사용) ──
        try:
            supervised_score = self._compute_supervised_tail_score(
                legacy_df=legacy_df,
                returns_ser=returns_ser,
                features_df=features_df,
                idx=idx,
                is_end_idx=is_end_idx,
            )
        except Exception as _sup_exc:
            _logger.warning("Supervised tail score computation raised: %s", _sup_exc)
            supervised_score = None

        # ── Step 3.5: supervised score → regime_df p_off 직접 부스트 ─────────
        # dominant_state = idxmax(regime4_cols) 이므로 legacy_df가 아닌 regime_df를 수정.
        # supervised score는 t 시점에서 t+1..t+8 충격을 예측하는 leading indicator이므로
        # forward_window만큼 전방 전파(causal: 과거 signal → 현재 state)하여
        # 충격 당일 bar가 BEAR dominant_state가 되도록 보장.
        if supervised_score is not None:
            sup = np.clip(supervised_score, 0.0, 1.0)
            # Rank-based top-8%: IsotonicRegression 이산값 때문에 quantile 기반이 부정확함
            n_sup = len(sup)
            target_n = max(10, int(n_sup * 0.08))
            sorted_desc = np.sort(sup)[::-1]
            rank_threshold = float(sorted_desc[min(target_n - 1, n_sup - 1)])
            crisis_mask = (sup >= rank_threshold).astype(bool)

            # Causal forward propagation: bar t 신호 → bar t+1..t+24도 BEAR (Exponential Decay)
            # supervised는 충격 발생 수 bar 전에 fire하며, 위기 신호 이후에도 일정 기간 리스크 오프 유지.
            forward_window = 24
            initial_boost = 0.90
            decay_factor = 0.97

            boost_series = np.zeros(n_sup, dtype=np.float64)
            for k in range(0, forward_window + 1):
                boost_at_k = initial_boost * (decay_factor**k)
                rolled = np.roll(crisis_mask, k)
                rolled[:k] = False
                boost_series = np.maximum(boost_series, np.where(rolled, boost_at_k, 0.0))

            p_off_cur = regime_df["regime_prob_risk_off_trend"].to_numpy(dtype=np.float64)
            p_off_boosted = np.maximum(p_off_cur, boost_series)
            regime_df = regime_df.copy()
            regime_df["regime_prob_risk_off_trend"] = p_off_boosted
            
            # [Step 3.5 Extension] Force p_off dominance by dampening other states where boosted
            for col in REGIME_STATE_COLUMNS:
                if col == "regime_prob_risk_off_trend":
                    continue
                regime_df[col] = np.where(boost_series > 0.3, regime_df[col] * 0.1, regime_df[col])

            regime_df = normalize_regime_state_frame(regime_df).join(
                regime_df[["regime_entropy"]]
            )
            
            # Recalculate hard state and duration after boost to improve Avg-Duration
            prob4_boosted = regime_df[list(REGIME_STATE_COLUMNS)].to_numpy(dtype=np.float64)
            hard_boosted = np.argmax(prob4_boosted, axis=1).astype(np.int32)
            sticky_boosted = _numba_sticky_labels(hard_boosted, np.array(self.sticky_min_duration, dtype=np.int32))
            regime_df["regime_hard_state"] = sticky_boosted.astype(np.float64)
            regime_df["regime_current_duration"] = _numba_current_duration(sticky_boosted)

            legacy_df = derive_legacy_hmm_prob_frame(regime_df)
            _logger.warning(
                "Step3.5 p_off boost | %s %s | raw_pct=%.3f | extended_pct=%.3f | p_off_mean=%.3f",
                symbol,
                tf,
                float(crisis_mask.mean()),
                float((boost_series > 0.0).mean()),
                float(regime_df["regime_prob_risk_off_trend"].mean()),
            )

        # ── Crisis Detector overlay ────────────────────────────────────────────
        # Replace derive-based hmm_prob_crisis with CrisisDetector soft score.
        # bear_trend 확률에서 증가분을 차감하여 5-column 합이 1로 유지되도록 re-normalize.
        try:
            detector = CrisisDetector()
            det_score = detector.score(
                features_df.reindex(idx).fillna(0.0),
                returns_ser.reindex(idx).fillna(0.0),
            )

            raw_crisis = legacy_df["hmm_prob_crisis"].to_numpy(dtype=np.float64)
            merged_crisis = np.clip(np.maximum(raw_crisis, det_score), 0.0, 1.0)

            delta = merged_crisis - raw_crisis
            bear_adj = np.clip(
                legacy_df["hmm_prob_bear_trend"].to_numpy(dtype=np.float64) - delta,
                0.0,
                1.0,
            )

            legacy_df = legacy_df.copy()
            legacy_df["hmm_prob_crisis"] = merged_crisis
            legacy_df["hmm_prob_bear_trend"] = bear_adj

            # Row-normalize to ensure 5-column sum == 1
            prob_mat = legacy_df[list(LEGACY_HMM_PROB_COLUMNS)].to_numpy(dtype=np.float64)
            prob_mat = np.clip(prob_mat, 0.0, 1.0)
            row_sum = prob_mat.sum(axis=1, keepdims=True)
            prob_mat = prob_mat / np.maximum(row_sum, 1e-12)
            legacy_df[list(LEGACY_HMM_PROB_COLUMNS)] = prob_mat

            crisis_triggered_pct = float((merged_crisis > 0.40).mean())
            _logger.info(
                "CrisisDetector+Supervised overlay applied | %s %s | crisis_pct=%.3f",
                symbol,
                tf,
                crisis_triggered_pct,
            )
        except Exception as exc:
            _logger.warning("CrisisDetector overlay failed; using derived crisis | %s", exc)
        # ─────────────────────────────────────────────────────────────────────

        hazard_res = compute_tail_hazard_overlay(
            features_df=features_df,
            regime_df=regime_df,
            returns_ser=returns_ser,
            cfg=OPT_FUTURES_CONFIG,
            supervised_score=supervised_score,
        )
        hazard_df = hazard_res.hazards
        policy_df = map_policy_controls(regime_df=regime_df, hazard_df=hazard_df, cfg=OPT_FUTURES_CONFIG)

        out = pd.concat([regime_df, legacy_df, hazard_df, policy_df], axis=1)
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
    ) -> np.ndarray | None:
        """IS supervised logistic+isotonic score for tail events.

        Causal: trains on [:is_end_idx] only, applies to full timeline.

        Args:
            legacy_df: Legacy HMM probability frame (hmm_prob_bear_trend, etc.).
            returns_ser: Return series.
            features_df: Feature dataframe (macro_liq_proxy_24h, etc.).
            idx: Full timeline index.
            is_end_idx: Exclusive end of in-sample window.

        Returns:
            Calibrated score array (n,) or None on failure.
        """
        try:
            from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        except ImportError:
            _logger.warning("scikit-learn not available; skipping supervised tail score")
            return None

        try:
            n = len(idx)
            if is_end_idx < 50:
                return None

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
            r_np = r_ser.to_numpy(dtype=np.float64)

            rv24_z = _tail_zscore(pd.Series(np.abs(r_np), index=idx).rolling(24, min_periods=4).std().fillna(0.0))
            down24_z = _tail_zscore(pd.Series(np.clip(-r_np, 0.0, None), index=idx).rolling(24, min_periods=4).std().fillna(0.0))

            f = features_df.reindex(idx)
            if "macro_liq_proxy_24h" in f.columns:
                liq24_z = _tail_zscore(pd.to_numeric(f["macro_liq_proxy_24h"], errors="coerce").fillna(0.0))
            else:
                liq24_z = np.zeros(n, dtype=np.float64)
            jump_z = _tail_zscore(pd.Series(np.clip(-r_np, 0.0, None), index=idx))

            X = np.column_stack([p_off, p_chop, ent, rv24_z, down24_z, liq24_z, jump_z]).astype(np.float64)

            # 8-bar forward worst return (IS 라벨; causal)
            fwd_worst = np.full(n, np.inf, dtype=np.float64)
            for k in range(1, 9):
                shifted = r_ser.shift(-k).to_numpy(dtype=np.float64)
                fwd_worst = np.minimum(fwd_worst, shifted)
            fwd_ser = pd.Series(fwd_worst, index=idx).replace([np.inf, -np.inf], np.nan)

            fwd_is = fwd_ser.iloc[:is_end_idx]
            valid_is = fwd_is.dropna()
            if len(valid_is) < 20:
                _logger.warning("Supervised tail: too few valid IS returns (%d)", len(valid_is))
                return None
            q10_is = float(valid_is.quantile(0.10))
            if not np.isfinite(q10_is):
                _logger.warning("Supervised tail: q10 not finite")
                return None

            y_is = (fwd_ser.iloc[:is_end_idx].fillna(0.0).to_numpy() <= q10_is).astype(int)
            if y_is.sum() < 10:
                _logger.warning("Supervised tail: too few positive labels (%d)", y_is.sum())
                return None

            clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)
            clf.fit(X[:is_end_idx], y_is)

            prob_is = clf.predict_proba(X[:is_end_idx])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(prob_is, y_is.astype(np.float64))

            prob_all = clf.predict_proba(X)[:, 1]
            score: np.ndarray = iso.transform(prob_all).astype(np.float64)
            _logger.info(
                "Step3 supervised tail score | is_end=%d | n=%d | score_mean=%.3f | pos_rate=%.3f",
                is_end_idx,
                n,
                float(score.mean()),
                float(y_is.mean()),
            )
            return score
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
