"""Phase D: CAWF-R ML cross-sectional rank portfolio optimization (Optuna TPE).

Trial objective minimizes the negative robust anchored-WF scalar
(median(log leg-TW) − λ·MAD − ψ·max leg-DD) with λ, ψ fixed in OPT_FUTURES_CONFIG.
Phase D futures: anchored walk-forward legs, robust AWF scalar objective.
Trial user_attrs set both AWF-native keys (``awf_*`` / ``awf_path_*``) and legacy
``cpcv_*`` / ``ml_*_cpcv`` aliases for older JSON readers.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import (
    FUTURES_INITIAL_BALANCE,
    MAKER_FEE_RATE,
    SLIPPAGE_RATE,
    TAKER_FEE_RATE,
)
from src.core.indicators.numpy_ops_futures import compute_atr_numpy
from src.domain.futures.backtest_engine import (
    backtest_target_weights_numba,
    hours_per_bar_from_timeframe,
    max_hold_bars_from_time_barrier,
)
from src.domain.futures.ml_pipeline.features.engineering import HMM_SEMANTIC_PROB_COLUMNS
from src.domain.futures.optimization.data_aligner import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
)
from src.domain.futures.optimization.evaluator import (
    _log_tw_from_ret_pct,
    calc_gate1_dsr_from_path_log_tw,
    calc_mdd_from_equity,
    compute_awf_robust_objective_score,
)
from src.domain.futures.optimization.validation import build_anchored_wf_legs
from src.domain.futures.portfolio.portfolio_constructor import (
    cov_lookback_bars,
    portfolio_weight_params_from_optuna,
    precompute_rebalance_weights,
    precompute_rolling_covariances,
)
from src.domain.futures.portfolio.portfolio_optimizer import (
    finalize_strategy_portfolio_params,
    load_portfolio_policy_config,
)
from src.domain.futures.portfolio.signal_composer import (
    composer_sigma_lookback_bars,
    rolling_per_bar_return_std,
)
from src.domain.futures.strategy_ml import FuturesMLStrategy

_logger = logging.getLogger(__name__)
_PRECOMPUTE_LOCK = threading.Lock()

# Cross-sectional momentum lookback periods precomputed for all trials.
_CS_MOM_LOOKBACKS: list[int] = [12, 24, 36, 48, 60, 72]


def inject_cs_momentum_ranks(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    lookbacks: list[int] | None = None,
) -> None:
    """Inject cross-sectional momentum rank columns into data_maps[sym][tf].

    Ranking at time t uses only pct_change(periods=lb) which is backward-looking
    — no temporal look-ahead bias.  Safe to call on IS-only or full (IS+OOS) data.
    """
    if lookbacks is None:
        lookbacks = _CS_MOM_LOOKBACKS
    if len(symbols) < 2:
        return

    rets_series: dict[str, pd.Series] = {}
    for sym in symbols:
        df = data_maps.get(sym, {}).get(tf)
        if df is not None and not df.empty:
            rets_series[sym] = df["close"]

    if len(rets_series) < 2:
        return

    for lb in lookbacks:
        all_rets = {s: r.pct_change(periods=lb) for s, r in rets_series.items()}
        ranks_df = pd.DataFrame(all_rets).rank(axis=1, pct=True)
        for sym in rets_series:
            if sym in ranks_df.columns:
                col_name = f"cs_mom_rank_{lb}"
                data_maps[sym][tf][col_name] = ranks_df[sym].astype(np.float64)


def resolve_embargo_bars_for_tf(cfg: dict[str, Any], tf: str, longest_indicator_period: int = 150) -> int:
    """Prefer EMBARGO_BARS_BY_TF; fallback to horizon ratio heuristic."""
    by_tf = cfg.get("EMBARGO_BARS_BY_TF")
    if isinstance(by_tf, dict) and tf in by_tf:
        return max(0, int(by_tf[str(tf)]))
    fixed_min: dict[str, int] = {"1h": 24}
    ratio_map: dict[str, float] = {"1h": 0.08}
    ratio: float = ratio_map.get(tf, 0.05)
    return max(fixed_min.get(tf, 12), int(longest_indicator_period * ratio))


EMBARGO_BARS: dict[str, int] = {
    tf: resolve_embargo_bars_for_tf(OPT_FUTURES_CONFIG, tf) for tf in ("1h", "4h", "1d")
}


class SignalCalibrator:
    """Platt scaling on alpha vs forward returns.

    Default AWF precompute fits only on **out-of-sample** bars (see
    ``_fit_oos_platt_calibrators_from_maps``). Legacy tail-window or in-sample fits are not used
    there.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(penalty=None, solver="lbfgs")
        self.is_fitted = False
        self.mean_b = 1.05  # Estimated win/loss ratio (Profit Factor)

    def fit(self, alphas: np.ndarray, returns: np.ndarray) -> None:
        """Fit calibration model and estimate average win/loss ratio."""
        # y=1 if forward return is positive
        y = (returns > 0.0001).astype(int)
        X = alphas.reshape(-1, 1)

        if len(np.unique(y)) > 1:
            try:
                self.model.fit(X, y)
                self.is_fitted = True
            except Exception as e:
                _logger.warning("[SignalCalibrator] Logistic fit failed: %s", e)

        # Estimate average win/loss ratio (b) for Kelly
        pos_rets = returns[returns > 0]
        neg_rets = np.abs(returns[returns < 0])
        if pos_rets.size > 0 and neg_rets.size > 0:
            raw_b = float(np.mean(pos_rets) / np.mean(neg_rets))
            self.mean_b = float(np.clip(raw_b, 0.7, 2.0))
            if self.is_fitted:
                _logger.debug(
                    "📐 Platt coef=%.4f b=%.4f n=%d/%d",
                    self.model.coef_[0][0],
                    self.mean_b,
                    int(pos_rets.size),
                    int(neg_rets.size),
                )
            else:
                _logger.debug(
                    "📐 Platt skip (no fit) b=%.4f n=%d/%d",
                    self.mean_b,
                    int(pos_rets.size),
                    int(neg_rets.size),
                )
        else:
            self.mean_b = 1.05
            _logger.debug("📐 Platt skip (insufficient samples) b=%.4f default", self.mean_b)

    def predict_prob(self, alphas: np.ndarray) -> np.ndarray:
        """Predict win probability p."""
        if not self.is_fitted:
            # Fallback: sigmoid-like mapping centered at 0.5
            z = (alphas - 0.5) * 8.0
            return 1.0 / (1.0 + np.exp(-z))
        X = alphas.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class DynamicKellySizer:
    """Fractional Kelly sizing with HMM and Crisis modulation."""

    @staticmethod
    def calculate(
        p: np.ndarray,
        b: float,
        hmm_crisis: np.ndarray,
        hmm_mod: np.ndarray,
        lam: float = 0.5,
        crisis_gamma: float = 1.0,
    ) -> np.ndarray:
        """Compute optimal fractional Kelly size."""
        # f* = p - (1-p)/b
        b = max(b, 0.5)
        f_star = p - (1.0 - p) / b
        f_star = np.clip(f_star, 0.0, 1.0)

        # HMM Crisis Scale
        crisis_scale = (1.0 - hmm_crisis) ** crisis_gamma

        # Final Size = Kelly * Scale (lam) * Modulator * Crisis
        return f_star * lam * hmm_mod * crisis_scale


@dataclass
class MLPhaseDContext:
    """Shared precomputed context passed to each Optuna trial in Phase D."""

    data_maps: dict[str, dict[str, Any]]
    symbols: list[str]
    tf: str
    seed: int = 42
    awf_leg_slices: list[dict[str, Any]] | None = None
    is_slice: dict[str, Any] | None = None
    holdout_slice: dict[str, np.ndarray] | None = None
    multi_alignment_info: dict[str, Any] | None = None
    # SOTA: Mathematical Policy Components
    calibrator: SignalCalibrator | None = None
    calibrator_short: SignalCalibrator | None = None
    estimated_b: float = 1.05
    kelly_ic_upper: float = 0.5  # T3-B: IC EWMA-based Kelly upper bound
    # Effective Bonferroni count = n_seeds × trials_per_seed (multi-seed studies).
    effective_total_trials: int | None = None
    # Coordinate ascent: "A"/"B"/"C"; frozen holds completed phases' Optuna-param dict slices.
    coordinate_phase: str | None = None
    coordinate_frozen_params: dict[str, Any] | None = None
    # When ``FUTURES_WF_HMM_LEG_REFIT`` is True, anchored-WF precompute reruns the full
    # universe ML pipeline (cross-sectional alpha + systemic HMM + fusion) per leg anchor.
    ml_pipeline_fetch_start: str | None = None
    ml_pipeline_end: str | None = None
    ml_pipeline_is_start: str | None = None
    ml_pipeline_workers: int | None = None


def _fit_oos_platt_calibrators_from_maps(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    info: dict[str, Any],
    *,
    window_lo: int,
    window_hi_excl: int,
    oos_pool_start: int | None = None,
) -> tuple[SignalCalibrator | None, SignalCalibrator | None, float]:
    """Platt scaling fit **only** on OOS bars (aligned eff_ref_len indices).

    Uses bars ``k ∈ [window_lo, window_hi_excl)`` per symbol (relative to each symbol's
    aligned ``[start_idx : start_idx + eff_ref_len)`` strip). Typically:

    - **Per AWF leg (with ML refit):** ``[anchor, test_s)`` — post-train embargo gap, strictly
      before that leg's test window (no leakage into the leg being backtested).
    - **Single global ML snapshot:** ``[first_anchor, eff_len)`` — entire AWF OOS-pool region,
      excluding the initial IS-pool ``[0, first_anchor)``.

    If too few valid (alpha, forward-return) pairs, optionally widens the start index back to
    ``oos_pool_start`` (still excludes IS-pool bars before the AWF OOS region).
    """
    eff_len = int(info["eff_ref_len"])
    lo = int(np.clip(int(window_lo), 0, eff_len))
    hi = int(np.clip(int(window_hi_excl), 0, eff_len))
    min_bars = int(OPT_FUTURES_CONFIG.get("FUTURES_CALIB_PLATT_MIN_OOS_BARS", 80))
    widen = bool(OPT_FUTURES_CONFIG.get("FUTURES_CALIB_PLATT_OOS_WIDEN_TO_POOL", True))
    pool = int(oos_pool_start) if oos_pool_start is not None else lo

    def _collect(a0: int, a1: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        al, rl, ash, rsh = [], [], [], []
        for sym in symbols:
            if sym not in data_maps or tf not in data_maps[sym]:
                continue
            start_idx = int(info["alignment_offsets"][sym])
            sym_df = data_maps[sym][tf]
            raw = sym_df.iloc[start_idx : start_idx + eff_len]
            fwd_ret = raw["close"].pct_change(12).shift(-12).to_numpy(dtype=np.float64)
            i0, i1 = int(np.clip(a0, 0, eff_len)), int(np.clip(a1, 0, eff_len))
            if i1 <= i0:
                continue
            if "ml_alpha_long" in raw.columns:
                alpha_long = raw["ml_alpha_long"].to_numpy(dtype=np.float64)[i0:i1]
                r_t = fwd_ret[i0:i1]
                mask = ~np.isnan(alpha_long) & ~np.isnan(r_t)
                if mask.any():
                    al.append(alpha_long[mask])
                    rl.append(r_t[mask])
            if "ml_alpha_short" in raw.columns:
                alpha_short = raw["ml_alpha_short"].to_numpy(dtype=np.float64)[i0:i1]
                r_ts = fwd_ret[i0:i1]
                mask_s = ~np.isnan(alpha_short) & ~np.isnan(r_ts)
                if mask_s.any():
                    ash.append(alpha_short[mask_s])
                    rsh.append(r_ts[mask_s])
        return al, rl, ash, rsh

    all_alphas, all_returns, all_alphas_short, all_returns_short = _collect(lo, hi)
    n_l = int(sum(a.size for a in all_alphas))
    n_s = int(sum(a.size for a in all_alphas_short))

    if widen and oos_pool_start is not None and pool < lo and (
        n_l < min_bars or n_s < min_bars
    ):
        _logger.info(
            "[SignalCalibrator] OOS Platt widening window start %d → %d (pool start)",
            lo,
            pool,
        )
        lo = int(np.clip(pool, 0, eff_len))
        all_alphas, all_returns, all_alphas_short, all_returns_short = _collect(lo, hi)
        n_l = int(sum(a.size for a in all_alphas))
        n_s = int(sum(a.size for a in all_alphas_short))

    if n_l < min_bars and n_s < min_bars:
        _logger.warning(
            "[SignalCalibrator] OOS Platt skipped: only %d long / %d short samples in [%d,%d) "
            "(min %d). Using uncalibrated scores in prebuilt.",
            n_l,
            n_s,
            lo,
            hi,
            min_bars,
        )
        return None, None, 1.05

    calib: SignalCalibrator | None = None
    calib_s: SignalCalibrator | None = None
    est_b = 1.05
    if all_alphas and n_l >= min(30, min_bars):
        calib = SignalCalibrator()
        calib.fit(np.concatenate(all_alphas), np.concatenate(all_returns))
        est_b = calib.mean_b
    if all_alphas_short and n_s >= min(30, min_bars):
        calib_s = SignalCalibrator()
        calib_s.fit(
            np.concatenate(all_alphas_short), np.concatenate(all_returns_short)
        )
    return calib, calib_s, est_b


def _fit_tail_platt_calibrators_from_maps(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    info: dict[str, Any],
    *,
    tail_frac: float | None = None,
) -> tuple[SignalCalibrator | None, SignalCalibrator | None, float]:
    """**Deprecated path:** tail of full aligned window (IS+OOS). Prefer
    :func:`_fit_oos_platt_calibrators_from_maps` with an explicit OOS index range.
    """
    eff_len = int(info["eff_ref_len"])
    if tail_frac is None:
        tail_frac = float(OPT_FUTURES_CONFIG.get("FUTURES_CALIB_PLATT_TAIL_FRAC", 0.30))
    calib_start = max(0, int(round(eff_len * (1.0 - float(tail_frac)))))
    return _fit_oos_platt_calibrators_from_maps(
        data_maps,
        symbols,
        tf,
        info,
        window_lo=calib_start,
        window_hi_excl=eff_len,
        oos_pool_start=None,
    )


def _build_prebuilt_full_arrays(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    info: dict[str, Any],
    *,
    calibrator: SignalCalibrator | None,
    calibrator_short: SignalCalibrator | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Vector bundle builders for ``_build_aligned_2d_from_prebuilt`` (one row = eff_ref_len bar)."""
    prebuilt_full: dict[str, dict[str, np.ndarray]] = {}
    eff_len = int(info["eff_ref_len"])
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        start_idx = int(info["alignment_offsets"][sym])
        raw_full = data_maps[sym][tf].iloc[start_idx : start_idx + eff_len]

        trimmed_sig = pd.DataFrame(index=raw_full.index)
        trimmed_sig["close"] = raw_full["close"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["high"] = raw_full["high"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["low"] = raw_full["low"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["open"] = raw_full["open"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["volume"] = raw_full["volume"].to_numpy(dtype=np.float64, copy=False) if "volume" in raw_full.columns else np.ones(len(raw_full))
        _atr_col = raw_full["atr"].to_numpy(dtype=np.float64, copy=False) if "atr" in raw_full.columns else None
        if _atr_col is None or not np.any(_atr_col > 0):
            _atr_period_fb = int(OPT_FUTURES_CONFIG.get("FUTURES_ATR_PERIOD_FIXED", 30))
            _atr_col = compute_atr_numpy(
                raw_full["high"].to_numpy(dtype=np.float64),
                raw_full["low"].to_numpy(dtype=np.float64),
                raw_full["close"].to_numpy(dtype=np.float64),
                _atr_period_fb,
            )
        trimmed_sig["atr"] = np.where(np.isfinite(_atr_col) & (_atr_col > 0), _atr_col,
                                       raw_full["close"].to_numpy(dtype=np.float64) * 0.01)
        trimmed_sig["garch_kelly_f"] = (
            raw_full["garch_kelly_f"].to_numpy(dtype=np.float64, copy=False)
            if "garch_kelly_f" in raw_full.columns
            else np.ones(len(raw_full))
        )
        trimmed_sig["slot_rank_score"] = (
            raw_full["slot_rank_score"].to_numpy(dtype=np.float64, copy=False)
            if "slot_rank_score" in raw_full.columns
            else np.zeros(len(raw_full))
        )

        if "funding_rate_sum" in raw_full.columns:
            trimmed_sig["funding_rate_sum"] = raw_full["funding_rate_sum"].to_numpy(
                dtype=np.float64, copy=False
            )

        gp_base = (
            raw_full["ml_alpha_00"].to_numpy(dtype=np.float64, copy=False)
            if "ml_alpha_00" in raw_full.columns
            else np.zeros(len(raw_full), dtype=np.float64)
        )
        if calibrator:
            p_base = calibrator.predict_prob(gp_base)
            trimmed_sig["ml_calib_prob"] = p_base

            if "ml_alpha_long" in raw_full.columns:
                p_l = calibrator.predict_prob(
                    raw_full["ml_alpha_long"].to_numpy(dtype=np.float64, copy=False)
                )
                trimmed_sig["ml_calib_prob_long"] = p_l
            else:
                trimmed_sig["ml_calib_prob_long"] = trimmed_sig["ml_calib_prob"]

            if "ml_alpha_short" in raw_full.columns:
                calib_s = calibrator_short or calibrator
                p_s = calib_s.predict_prob(
                    raw_full["ml_alpha_short"].to_numpy(dtype=np.float64, copy=False)
                )
                trimmed_sig["ml_calib_prob_short"] = p_s
            else:
                trimmed_sig["ml_calib_prob_short"] = trimmed_sig["ml_calib_prob"]
        else:
            trimmed_sig["ml_calib_prob"] = raw_full.get("ml_calib_prob", 0.5)
            trimmed_sig["ml_calib_prob_long"] = raw_full.get("ml_calib_prob_long", 0.5)
            trimmed_sig["ml_calib_prob_short"] = raw_full.get("ml_calib_prob_short", 0.5)

        gp_centered = gp_base - 0.5
        trimmed_sig["trend_direction"] = np.where(
            np.abs(gp_centered) > 0.01, np.sign(gp_centered), 0.0
        ).astype(np.float64)
        trimmed_sig["entry_upper"] = 0.0
        trimmed_sig["entry_lower"] = 999999.0
        _xs_cols = (
            "xs_score_long",
            "xs_score_short",
            "hmm_prob_bull_calm",
            "hmm_prob_bull_vol_up",
            "hmm_prob_bear_trend",
            "hmm_prob_chop",
            "hmm_prob_crisis",
            "hmm_hard_state",
            "hmm_modulator_long",
            "hmm_modulator_short",
            "hmm_modulator_base_long",
            "hmm_modulator_base_short",
            "expected_variance",
            "target_variance",
            "btc_trend_vol_adj_24h",
        )
        for col in _xs_cols:
            if col in raw_full.columns:
                trimmed_sig[col] = raw_full[col].to_numpy(dtype=np.float64, copy=False)

        _sig_win = composer_sigma_lookback_bars(tf, OPT_FUTURES_CONFIG)
        trimmed_sig["composer_sigma_bar"] = rolling_per_bar_return_std(
            raw_full["close"].to_numpy(dtype=np.float64, copy=False),
            _sig_win,
        )

        _inject_dyn_leverage_trimmed(trimmed_sig, raw_full)

        prebuilt_full[sym] = _dataframe_to_symbol_arrays(trimmed_sig)

    return prebuilt_full


def _hmm_columns_for_dyn_leverage(df: pd.DataFrame) -> list[str]:
    sem = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in df.columns]
    if sem:
        return sem
    leg = sorted(
        (c for c in df.columns if str(c).startswith("hmm_prob_")),
        key=lambda x: int(str(x).split("_")[-1]),
    )
    return leg


def _inject_dyn_leverage_trimmed(trimmed_sig: pd.DataFrame, raw_full: pd.DataFrame) -> None:
    """Apply Kelly quality x entropy discount + crisis de-risk to trimmed signal."""
    cfg = OPT_FUTURES_CONFIG
    hmm_cols = _hmm_columns_for_dyn_leverage(raw_full)
    base_lev = float(cfg.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    crisis_thr = float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.6))
    if not hmm_cols:
        trimmed_sig["dyn_leverage"] = float(base_lev)
        return
    k = len(hmm_cols)
    try:
        p_mat = (
            raw_full[hmm_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0 / float(k))
            .to_numpy(dtype=np.float64)
        )
    except Exception as e:
        _logger.error(
            "Error creating p_mat: %s. hmm_cols=%s, columns=%s",
            e,
            hmm_cols,
            list(raw_full.columns),
        )
        raise

    if p_mat.shape[1] != k:
         _logger.error("SHAPE MISMATCH: p_mat.shape[1]=%s, k=%s. hmm_cols=%s. ALL COLS=%s",
                      p_mat.shape[1], k, hmm_cols, list(raw_full.columns))

    close = raw_full["close"].astype(np.float64)
    r = np.log(close / close.shift(1).clip(lower=1e-12)).fillna(0.0).to_numpy(dtype=np.float64)
    g = (
        raw_full["ml_alpha_00"].fillna(0.0).to_numpy(dtype=np.float64)
        if "ml_alpha_00" in raw_full.columns
        else np.zeros(len(raw_full), dtype=np.float64)
    )
    factor_ret = g * r
    state_hard = np.argmax(p_mat, axis=1).astype(np.int64)
    k_vec: np.ndarray = np.zeros(k, dtype=np.float64)
    for s in range(k):
        sel = state_hard == s
        rr = factor_ret[sel]
        if np.sum(sel) > 30:
            k_vec[s] = float(np.clip(np.mean(rr) / (np.var(rr, ddof=1) + 1e-12), -1.0, 1.0))
    k_min, k_max = float(np.min(k_vec)), float(np.max(k_vec))
    k_qual = (k_vec - k_min) / (k_max - k_min + 1e-12)
    log_k = float(np.log(max(k, 2)))
    ent = -np.sum(p_mat * np.log(np.clip(p_mat, 1e-12, 1.0)), axis=1)
    ent_disc = np.clip(1.0 - (ent / log_k), 0.0, 1.0)
    exp_q = p_mat @ k_qual
    lev_min = 2.0
    lev_max = max(base_lev, lev_min)
    levs = lev_min + (lev_max - lev_min) * exp_q * ent_disc
    levs = np.clip(levs, lev_min, lev_max)
    if "hmm_prob_crisis" in raw_full.columns:
        pc = raw_full["hmm_prob_crisis"].fillna(0.0).to_numpy(dtype=np.float64)
        levs = np.where(pc > crisis_thr, 1.0, levs)
    trimmed_sig["dyn_leverage"] = levs.astype(np.float64, copy=False)


def compute_multi_alignment_info(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    embargo: int,
) -> dict[str, Any] | None:
    """Precompute alignment, fingerprints, and CSM ranks to avoid per-trial overhead."""
    is_start_dts_per_sym: dict[str, Any] = {}

    for sym in symbols:
        sym_df = data_maps[sym].get(tf)
        if sym_df is None or sym_df.empty:
            continue

        is_off = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
        if len(sym_df) > is_off and "datetime" in sym_df.columns:
            is_start_dts_per_sym[sym] = sym_df["datetime"].iloc[is_off]

    if not is_start_dts_per_sym:
        return None

    common_is_start_dt = max(is_start_dts_per_sym.values())

    alignment_offsets: dict[str, int] = {}
    eff_ref_lens: list[int] = []

    for sym in symbols:
        sym_df = data_maps[sym].get(tf)
        if sym_df is None or sym_df.empty or "datetime" not in sym_df.columns:
            continue
        start_idx = sym_df["datetime"].searchsorted(common_is_start_dt)
        alignment_offsets[sym] = int(start_idx)
        eff_ref_lens.append(len(sym_df) - int(start_idx))

    if not eff_ref_lens:
        return None

    eff_ref_len = min(eff_ref_lens)
    if eff_ref_len < 200:
        return None

    return {
        "common_is_start_dt": common_is_start_dt,
        "alignment_offsets": alignment_offsets,
        "eff_ref_len": eff_ref_len,
    }


def precompute_ml_optimization_context(ctx: MLPhaseDContext) -> None:
    """Pre-align and pre-slice all data before Optuna starts to eliminate trial overhead."""
    with _PRECOMPUTE_LOCK:
        if ctx.awf_leg_slices is not None:
            return

        # 1. Alignment & Baseline Signals
        fc_pre = OPT_FUTURES_CONFIG
        atm0 = float(fc_pre.get("FUTURES_ATR_STOP_MULT", 2.5))
        pre_ml: dict[str, Any] = {
            "TRAILING_ACTIVATION_ATR": 1.0,
            "BAYESIAN_C": 10.0,
            "KELLY_SHRINKAGE": 0.3,
            "K_LONG": 2,
            "K_SHORT": 2,
            "REBALANCE_BARS": 6,
            "MIN_SCORE_PERCENTILE": 0.55,
            "CRISIS_GAMMA": 1.0,
            "ATR_PERIOD": 30,
            "ATR_MULT": atm0,
            "TRAIL_MULT": atm0,
            "PORTFOLIO_KAPPA": float(fc_pre.get("FUTURES_PORTFOLIO_KAPPA", 0.35)),
        }
        params = _base_engine_params(pre_ml, ctx.tf)
        FuturesMLStrategy(name="Precompute", params=params)

        emb = int(EMBARGO_BARS.get(ctx.tf, 12))
        info = compute_multi_alignment_info(ctx.data_maps, ctx.symbols, ctx.tf, emb)
        if info is None:
            return
        ctx.multi_alignment_info = info

        eff_len = int(info["eff_ref_len"])
        embargo = int(EMBARGO_BARS.get(ctx.tf, 12))
        k_legs = int(OPT_FUTURES_CONFIG.get("FUTURES_AWF_K_LEGS", 6))
        is_pool = float(OPT_FUTURES_CONFIG.get("FUTURES_AWF_IS_POOL_FRAC", 0.70))
        awf_legs = build_anchored_wf_legs(
            eff_len, k=k_legs, embargo=embargo, is_pool_frac=is_pool
        )
        
        # SOTA: Precompute Global Rolling Covariance (once per optimization run)
        # Eliminates O(T * N^3) sklearn LW calls from the inner Trial loops.
        lookback = cov_lookback_bars(ctx.tf, OPT_FUTURES_CONFIG)
        close_2d_full = np.zeros((eff_len, len(ctx.symbols)), dtype=np.float64)
        for s_idx, sym in enumerate(ctx.symbols):
            if sym in ctx.data_maps and ctx.tf in ctx.data_maps[sym]:
                start_idx = info["alignment_offsets"][sym]
                close_2d_full[:, s_idx] = ctx.data_maps[sym][ctx.tf]["close"].iloc[
                    start_idx : start_idx + eff_len
                ].to_numpy(dtype=np.float64)
        sigma_3d_full = precompute_rolling_covariances(close_2d_full, lookback)
        
        first_awf_anchor = int(awf_legs[0][1]) if awf_legs else max(1, int(eff_len * is_pool))

        dates_ok = bool(ctx.ml_pipeline_fetch_start and ctx.ml_pipeline_end)
        want_leg_refit = bool(OPT_FUTURES_CONFIG.get("FUTURES_WF_HMM_LEG_REFIT", False))
        use_full_leg_ml = want_leg_refit and dates_ok and bool(awf_legs)
        if want_leg_refit and not dates_ok:
            _logger.warning(
                "[ML_OPT] FUTURES_WF_HMM_LEG_REFIT=True but ml_pipeline_fetch_start/ml_pipeline_end "
                "not set on MLPhaseDContext; using one merged ML snapshot for every AWF leg."
            )

        wrk = ctx.ml_pipeline_workers or max(1, min(8, len(ctx.symbols)))

        leg_refit_slices: list[dict[str, Any]] | None = None
        last_calib: SignalCalibrator | None = None
        last_calib_short: SignalCalibrator | None = None
        last_est_b = 1.05

        if use_full_leg_ml:
            from src.domain.futures.ml_pipeline.pipeline_runner import (
                copy_data_maps_tf_clone,
                merge_ml_output_into_data_maps,
                run_ml_pipeline_for_universe,
            )

            _logger.debug(
                "[ML_OPT] AWF full ML leg refit — %d× universe pipeline "
                "(cross-sectional alpha + systemic HMM + fusion). Expect long precompute.",
                len(awf_legs),
            )
            ref_sym = next(
                (s for s in ctx.symbols if s in ctx.data_maps and ctx.tf in ctx.data_maps[s]),
                None,
            )
            if ref_sym is None:
                use_full_leg_ml = False
            else:
                sym_df_ref = ctx.data_maps[ref_sym][ctx.tf]
                start_idx_ref = int(info["alignment_offsets"][ref_sym])
                tmp_slices: list[dict[str, Any]] = []
                failed = False
                for leg_i, (_train_s, anchor, test_s, test_e) in enumerate(awf_legs):
                    idx_row = start_idx_ref + int(anchor)
                    if idx_row >= len(sym_df_ref):
                        _logger.error(
                            "[ML_OPT] AWF leg %d anchor idx %d out of range (len=%d).",
                            leg_i,
                            idx_row,
                            len(sym_df_ref),
                        )
                        failed = True
                        break
                    cutoff_dt = pd.to_datetime(sym_df_ref["datetime"].iloc[idx_row], utc=True)
                    is_end_str = cutoff_dt.isoformat()

                    _logger.debug(
                        "[ML_OPT] AWF leg %d/%d: is_end=%s train=[0,%d) test=[%d,%d)",
                        leg_i + 1,
                        len(awf_legs),
                        is_end_str,
                        int(anchor),
                        int(test_s),
                        int(test_e),
                    )

                    ml_out = run_ml_pipeline_for_universe(
                        list(ctx.symbols),
                        ctx.tf,
                        ctx.ml_pipeline_fetch_start,
                        ctx.ml_pipeline_end,
                        dict(OPT_FUTURES_CONFIG),
                        workers=wrk,
                        n_jobs=wrk,
                        is_end_date=is_end_str,
                        is_start_date=ctx.ml_pipeline_is_start,
                        gp_only=False,
                        hmm_only=False,
                        preloaded_data_maps=ctx.data_maps,
                        seed=ctx.seed,
                    )
                    if not ml_out.meta_feature_frame_by_symbol:
                        _logger.error("[ML_OPT] AWF leg %d ML pipeline returned empty output.", leg_i)
                        failed = True
                        break

                    leg_maps = copy_data_maps_tf_clone(ctx.data_maps, ctx.symbols, ctx.tf)
                    merge_ml_output_into_data_maps(
                        ml_out,
                        leg_maps,
                        ctx.symbols,
                        ctx.tf,
                        log_tag=f" leg{leg_i}_AWF",
                    )

                    calib_leg, calib_s_leg, est_b_leg = _fit_oos_platt_calibrators_from_maps(
                        leg_maps,
                        ctx.symbols,
                        ctx.tf,
                        info,
                        window_lo=int(anchor),
                        window_hi_excl=int(test_s),
                        oos_pool_start=first_awf_anchor,
                    )
                    last_calib, last_calib_short, last_est_b = calib_leg, calib_s_leg, est_b_leg

                    prebuilt_leg = _build_prebuilt_full_arrays(
                        leg_maps,
                        ctx.symbols,
                        ctx.tf,
                        info,
                        calibrator=calib_leg,
                        calibrator_short=calib_s_leg,
                    )
                    aligned_leg = _build_aligned_2d_from_prebuilt(
                        prebuilt_leg, ctx.symbols, test_s, test_e,
                        sigma_3d_full=sigma_3d_full
                    )
                    tmp_slices.append({"leg_range": (test_s, test_e), "data": aligned_leg})

                if failed or len(tmp_slices) != len(awf_legs):
                    _logger.debug(
                        "[ML_OPT] Per-leg ML refit incomplete (%d/%d legs); "
                        "falling back to single global ML merge for AWF.",
                        len(tmp_slices),
                        len(awf_legs),
                    )
                else:
                    leg_refit_slices = tmp_slices

        if leg_refit_slices is not None:
            ctx.awf_leg_slices = leg_refit_slices
            ctx.calibrator = last_calib
            ctx.calibrator_short = last_calib_short
            ctx.estimated_b = last_est_b
            
            # [Optimization] Populate is_slice for decoupled IS optimization
            prebuilt_full_is = _build_prebuilt_full_arrays(
                ctx.data_maps,
                ctx.symbols,
                ctx.tf,
                info,
                calibrator=ctx.calibrator,
                calibrator_short=ctx.calibrator_short,
            )
            ctx.is_slice = _build_aligned_2d_from_prebuilt(
                prebuilt_full_is, ctx.symbols, 0, eff_len,
                sigma_3d_full=sigma_3d_full
            )
        else:
            # [LEAKAGE FIX] Calibrator must be trained on IS bars only.
            # Range: [0, first_awf_anchor - embargo) — no leakage into OOS test legs.
            calib_min_bars = int(OPT_FUTURES_CONFIG.get("FUTURES_CALIB_PLATT_MIN_OOS_BARS", 80))
            calib_hi = max(0, first_awf_anchor - embargo)
            if calib_hi < calib_min_bars:
                _logger.warning(
                    "[CALIB] IS-only calibration window too small (%d bars < min %d); "
                    "falling back to OOS-pool window [%d, %d) to avoid data starvation.",
                    calib_hi,
                    calib_min_bars,
                    first_awf_anchor,
                    eff_len,
                )
                c0, c0s, eb0 = _fit_oos_platt_calibrators_from_maps(
                    ctx.data_maps,
                    ctx.symbols,
                    ctx.tf,
                    info,
                    window_lo=first_awf_anchor,
                    window_hi_excl=eff_len,
                    oos_pool_start=first_awf_anchor,
                )
            else:
                c0, c0s, eb0 = _fit_oos_platt_calibrators_from_maps(
                    ctx.data_maps,
                    ctx.symbols,
                    ctx.tf,
                    info,
                    window_lo=0,
                    window_hi_excl=calib_hi,
                    oos_pool_start=None,  # IS 구간이므로 pool_start 불필요
                )
            ctx.calibrator = c0
            ctx.calibrator_short = c0s
            ctx.estimated_b = eb0

            prebuilt_full = _build_prebuilt_full_arrays(
                ctx.data_maps,
                ctx.symbols,
                ctx.tf,
                info,
                calibrator=ctx.calibrator,
                calibrator_short=ctx.calibrator_short,
            )

            # [Optimization] Populate is_slice for decoupled IS optimization
            ctx.is_slice = _build_aligned_2d_from_prebuilt(
                prebuilt_full, ctx.symbols, 0, eff_len,
                sigma_3d_full=sigma_3d_full
            )

            ctx.awf_leg_slices = []
            for _train_s, _train_e, test_s, test_e in awf_legs:
                aligned = _build_aligned_2d_from_prebuilt(
                    prebuilt_full, ctx.symbols, test_s, test_e,
                    sigma_3d_full=sigma_3d_full
                )
                ctx.awf_leg_slices.append({"leg_range": (test_s, test_e), "data": aligned})

        ctx.holdout_slice = None  # last AWF leg covers holdout zone

        # T3-B: IS xs_score_long vs forward return의 Spearman IC → Kelly upper bound
        if ctx.is_slice is not None:
            xl_is = ctx.is_slice.get("xs_score_long")
            cl_is = ctx.is_slice.get("close")
            if xl_is is not None and cl_is is not None:
                xl_arr = np.asarray(xl_is, dtype=np.float64)
                cl_arr = np.asarray(cl_is, dtype=np.float64)
                if xl_arr.ndim == 2:
                    xl_arr = np.nanmean(xl_arr, axis=1)
                if cl_arr.ndim == 2:
                    cl_arr = np.nanmean(cl_arr, axis=1)
                fwd = np.log(
                    np.clip(cl_arr[1:], 1e-12, None) / np.clip(cl_arr[:-1], 1e-12, None)
                )
                sig = xl_arr[:-1]
                mask = np.isfinite(sig) & np.isfinite(fwd)
                if int(np.sum(mask)) > 30:
                    from scipy.stats import spearmanr as _spearmanr
                    _ic_raw, _ = _spearmanr(sig[mask], fwd[mask])
                    ctx.kelly_ic_upper = float(np.clip(abs(_ic_raw) * 10.0, 0.05, 0.5))
                    _logger.debug(
                        "[T3-B] IC→Kelly upper: %.4f (Spearman IC=%.4f)",
                        ctx.kelly_ic_upper,
                        _ic_raw,
                    )

        ctx.multi_alignment_info["awf_legs"] = awf_legs

    # Diagnostic on AWF leg 0
    if ctx.awf_leg_slices:
        aligned0 = ctx.awf_leg_slices[0].get("data") or {}
        xl0 = aligned0.get("xs_score_long")
        if xl0 is not None and getattr(xl0, "size", 0) > 0:
            xs_std = float(np.nanstd(np.asarray(xl0, dtype=np.float64)))
            ctx.multi_alignment_info["xs_score_aligned_std"] = xs_std
            if xs_std < 0.05:
                _logger.debug(
                    "[ML_OPT] xs_score dispersion low (std=%.6f); check GP/CS merge path.",
                    xs_std,
                )

        _log_precompute_xs_dispersion(
            aligned0,
            k_long=int(pre_ml["K_LONG"]),
        )


def _log_precompute_xs_dispersion(aligned0: dict[str, Any], *, k_long: int) -> None:
    """Lightweight precompute diagnostic (linear xs scores; no CS rank engine)."""
    _ = k_long
    xl = aligned0.get("xs_score_long")
    if xl is None or getattr(xl, "size", 0) == 0:
        return
    arr_l = np.ascontiguousarray(xl, dtype=np.float64)
    n_b = arr_l.shape[0]
    prev_i = min(n_b - 1, max(0, n_b // 2))
    flat_l = arr_l[prev_i, :].ravel()
    q1, q50, q99 = (
        float(np.nanpercentile(flat_l, 1)),
        float(np.nanpercentile(flat_l, 50)),
        float(np.nanpercentile(flat_l, 99)),
    )
    _logger.debug(
        "[ML_OPT][precompute] xs_score_long row quantiles p01/p50/p99=%.6f/%.6f/%.6f",
        q1,
        q50,
        q99,
    )


def _fixed_ml_phase_d_params() -> dict[str, Any]:
    """Constants that must stay aligned between optimization and final evaluation."""
    return {
        "MIN_SCORE_PERCENTILE": 0.55,
        "RISK_PER_TRADE": 0.05,
    }


def infer_kelly_shrinkage_bayesian_c_for_enqueue(
    fk_target: float, *, shield: bool
) -> tuple[float, float]:
    """Grid BAYESIAN_C so fk from _base_engine_params matches deploy KELLY_FRACTION."""
    fk_t = float(np.clip(float(fk_target), 0.05, 0.6))
    ks_lo, ks_hi = (0.52, 1.02) if shield else (0.45, 1.20)
    bc_lo, bc_hi = (5.0, 14.0) if shield else (5.0, 15.0)
    best_err = 1e9
    best_bc, best_ks = 10.0, float(np.clip(fk_t / (0.35 * (1.0 + 0.1)), ks_lo, ks_hi))
    for i in range(2001):
        bc = bc_lo + (bc_hi - bc_lo) * (i / 2000.0)
        denom = 0.35 * (1.0 + 1.0 / bc)
        if denom < 1e-12:
            continue
        raw_ks = fk_t / denom
        ks = float(np.clip(raw_ks, ks_lo, ks_hi))
        pred = float(np.clip(0.35 * ks * (1.0 + 1.0 / bc), 0.05, 0.6))
        err = abs(pred - fk_t)
        if err < best_err:
            best_err = err
            best_bc, best_ks = float(bc), ks
    return best_bc, best_ks


def _snap_int_list(val: int, choices: list[int]) -> int:
    return int(min(choices, key=lambda c: abs(c - val)))


def _snap_float_list(val: float, choices: list[float]) -> float:
    return float(min(choices, key=lambda c: abs(c - val)))


def build_phase_d_enqueue_params_from_deploy_json(
    deploy: dict[str, Any],
) -> dict[str, Any] | None:
    """Map deploy JSON to Optuna enqueue_trial param dict (single-objective Phase-D)."""
    shield = bool(OPT_FUTURES_CONFIG.get("FUTURES_TIER1_SHIELD_MODE", False))
    policy = load_portfolio_policy_config(OPT_FUTURES_CONFIG)
    fk_raw = deploy.get("KELLY_FRACTION", deploy.get("FK_FRACTION"))
    if fk_raw is None:
        return None
    fixed = _fixed_ml_phase_d_params()
    atr_stop = float(OPT_FUTURES_CONFIG.get("FUTURES_ATR_STOP_MULT", 2.5))
    atr_choices_i = [30]
    atr_m_choices = [
        round(max(0.5, atr_stop - 0.25), 3),
        atr_stop,
        round(atr_stop + 0.25, 3),
    ]
    trail_choices = [2.5, 3.0]
    stp_choices = [1.0, 1.5]
    lsc_choices = [2.0, 2.5]
    stress_choices = [2.5, 3.0]
    pfk_choices = [40, 48, 60]
    try:
        bc, ks = infer_kelly_shrinkage_bayesian_c_for_enqueue(float(fk_raw), shield=shield)
        reb = int(deploy["REBALANCE_BARS"])
        k_long = int(deploy["K_LONG"])
        crisis = float(deploy.get("CRISIS_GAMMA", deploy.get("CRISIS_GATE_PROB", 1.3)))
        atr_p = int(_snap_int_list(int(deploy.get("ATR_PERIOD", 30)), atr_choices_i))

        atr_m = _snap_float_list(
            float(deploy.get("ATR_MULT", deploy.get("LONG_ATR_MULT", atr_stop))),
            atr_m_choices,
        )
        trail_m = _snap_float_list(
            float(deploy.get("TRAIL_MULT", deploy.get("LONG_TRAIL_MULT", 3.0))), trail_choices
        )

        s_tp = _snap_float_list(float(deploy["SHORT_TP_MULT"]), stp_choices)
        l_scale = _snap_float_list(float(deploy["LONG_SCALE_ATR_MULT"]), lsc_choices)
        max_exp = float(deploy.get("MAX_EXPOSURE_PER_COIN", 1.0))
        max_gross = float(deploy.get("MAX_EXPOSURE", 1.0))
        dd_thr = float(deploy["DD_SCALING_THRESHOLD"])
        cs_z_thr = float(deploy.get("CS_Z_SCORE_THRESHOLD", 1.0))
        long_cs_z = float(deploy.get("LONG_CS_Z_ENTRY", cs_z_thr))
        short_cs_z = float(deploy.get("SHORT_CS_Z_ENTRY", cs_z_thr))
        hyst_gap_enq = float(deploy.get("HYSTERESIS_GAP", 0.3))
        crisis_lzb_enq = float(deploy.get("CRISIS_LONG_Z_BOOST", 0.0))
        crisis_lms_enq = float(
            deploy.get(
                "CRISIS_LONG_MAG_SUPPRESS",
                OPT_FUTURES_CONFIG.get("FUTURES_CRISIS_LONG_MAG_SUPPRESS", 1.0),
            )
        )
        pfk_win = _snap_int_list(int(deploy.get("PFK_WINDOW", 40)), pfk_choices)
        stress = _snap_float_list(float(deploy.get("STRESS_VOL_Z", 2.5)), stress_choices)
        rpt = float(deploy.get("RISK_PER_TRADE", fixed["RISK_PER_TRADE"]))
        min_score = float(deploy.get("MIN_SCORE_PERCENTILE", 0.55))
    except (KeyError, TypeError, ValueError):
        return None
    if rpt != float(fixed["RISK_PER_TRADE"]):
        return None
    out = finalize_strategy_portfolio_params(
        {
            "SIZING_METHOD": "profit_factor_kelly",
            "BAYESIAN_C": bc,
            "KELLY_SHRINKAGE": ks,
            "K_RANK": k_long,
            "K_LONG": k_long,
            "K_SHORT": k_long,
            "REBALANCE_BARS": reb,
            "CRISIS_GAMMA": crisis,
            "ATR_PERIOD": atr_p,
            "ATR_MULT": atr_m,
            "TRAIL_MULT": trail_m,
            "SHORT_TP_MULT": s_tp,
            "LONG_SCALE_ATR_MULT": l_scale,
            "MAX_EXPOSURE_PER_COIN": max_exp,
            "MAX_EXPOSURE": max_gross,
            "DD_SCALING_THRESHOLD": dd_thr,
            "CS_Z_SCORE_THRESHOLD": cs_z_thr,
            "LONG_CS_Z_ENTRY": long_cs_z,
            "SHORT_CS_Z_ENTRY": short_cs_z,
            "HYSTERESIS_GAP": hyst_gap_enq,
            "CRISIS_LONG_Z_BOOST": crisis_lzb_enq,
            "CRISIS_LONG_MAG_SUPPRESS": crisis_lms_enq,
            "PFK_WINDOW": pfk_win,
            "STRESS_VOL_Z": stress,
            "RISK_PER_TRADE": rpt,
            "MIN_SCORE_PERCENTILE": min_score,
            "USE_CS_RANK_ENGINE": False,
        },
        policy,
    )
    return out


def _baseline_ml_out_dict_for_coordinate(policy: Any) -> dict[str, Any]:
    """Center point for coordinate-ascent phases (before phase-specific suggests)."""
    gh = min(
        float(policy.gross_exposure_cap),
        float(OPT_FUTURES_CONFIG.get("FUTURES_PHASE_A_MAX_GROSS_EXPOSURE", 1.5)),
    )
    ann = 0.25
    fc = OPT_FUTURES_CONFIG
    atm = float(fc.get("FUTURES_ATR_STOP_MULT", 2.5))
    kappa0 = float(fc.get("FUTURES_PORTFOLIO_KAPPA", 0.35))
    return {
        "SIZING_METHOD": "profit_factor_kelly",
        "TARGET_ANN_VOL": ann,
        "PORTFOLIO_KAPPA": kappa0,
        "KELLY_LAMBDA": kappa0,
        "CRISIS_GAMMA": 2.0,
        "ATR_PERIOD": 30,
        "ATR_MULT": atm,
        "TRAIL_MULT": atm,
        "SHORT_TP_MULT": 2.0,
        "LONG_SCALE_ATR_MULT": 3.0,
        "MAX_EXPOSURE_PER_COIN": float(policy.per_symbol_cap),
        "MAX_EXPOSURE": min(1.2, gh),
        "RISK_PER_TRADE": kappa0,
        "REBALANCE_BARS": 6,
        "K_LONG": int(policy.top_k_long),
        "K_SHORT": int(policy.top_k_short),
        "DD_SCALING_THRESHOLD": 0.0,
        "MIN_SCORE_PERCENTILE": 0.55,
        "DYNAMIC_RA_CRISIS_COEF": 3.0,
        "DYNAMIC_RA_BEAR_COEF": 1.5,
        "NORM_VAR_CONSTANT": 0.5,
        "CRISIS_LONG_Z_BOOST": 0.0,
        "CRISIS_LONG_MAG_SUPPRESS": float(
            OPT_FUTURES_CONFIG.get("FUTURES_CRISIS_LONG_MAG_SUPPRESS", 1.0)
        ),
        "BETA_ALPHA": float(fc.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0)),
        "BETA_REGIME_BULL": float(fc.get("FUTURES_DEFAULT_BETA_REGIME_BULL", 1.0)),
        "BETA_REGIME_BEAR": float(fc.get("FUTURES_DEFAULT_BETA_REGIME_BEAR", 0.25)),
        "BETA_REGIME_CRISIS": float(fc.get("FUTURES_DEFAULT_BETA_REGIME_CRISIS", 0.5)),
        "BETA_REGIME_CHOP": float(fc.get("FUTURES_DEFAULT_BETA_REGIME_CHOP", 0.25)),
        "EV_HURDLE_BPS": float(fc.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 5.0)),
        "SLIPPAGE_BPS_BUFFER_MULT": float(fc.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0)),
        "TIME_BARRIER_H": float(fc.get("FUTURES_DEFAULT_TIME_BARRIER_H", 0.0)),
    }


def _suggest_ml_joint_nsga2(trial: optuna.Trial, ctx: MLPhaseDContext) -> dict[str, Any]:
    """Joint NSGA-II parameter space (15-20 parameters)."""
    policy = load_portfolio_policy_config(OPT_FUTURES_CONFIG)

    # 1. Capacity & Risk
    ann = float(trial.suggest_float("TARGET_ANN_VOL", 0.15, 0.45))
    gh = min(float(policy.gross_exposure_cap), 1.5)
    mx = float(trial.suggest_float("MAX_EXPOSURE", 0.8, gh))
    pc = float(trial.suggest_float("MAX_EXPOSURE_PER_COIN", 0.15, min(0.5, mx)))
    kappa = float(trial.suggest_float("PORTFOLIO_KAPPA", 0.2, 0.5))

    # 2. Alpha & Regime Betas
    beta_a = float(trial.suggest_float("BETA_ALPHA", 0.5, 1.5))
    b_bull = float(trial.suggest_float("BETA_REGIME_BULL", 0.5, 2.0))
    b_bear = float(trial.suggest_float("BETA_REGIME_BEAR", 0.0, 1.0))
    b_chop = float(trial.suggest_float("BETA_REGIME_CHOP", 0.0, 1.0))
    b_crisis = float(trial.suggest_float("BETA_REGIME_CRISIS", -0.5, 0.5))
    ev_h = float(trial.suggest_float("EV_HURDLE_BPS", 0.0, 20.0))

    # 3. Execution & Friction
    reb = int(trial.suggest_categorical("REBALANCE_BARS", [1, 3, 6, 12]))
    slip_buf = float(trial.suggest_float("SLIPPAGE_BPS_BUFFER_MULT", 0.8, 1.5))
    tb_h = float(trial.suggest_categorical("TIME_BARRIER_H", [0.0, 24.0, 48.0, 168.0]))

    # 4. HMM-conditioned Dynamic Kelly
    crisis_thr = float(trial.suggest_float("CRISIS_OVERRIDE_THRESHOLD", 0.3, 0.7))
    gamma = float(trial.suggest_float("CRISIS_GAMMA", 0.5, 3.0))

    base = {
        "TARGET_ANN_VOL": ann,
        "MAX_EXPOSURE": mx,
        "MAX_EXPOSURE_PER_COIN": pc,
        "PORTFOLIO_KAPPA": kappa,
        "KELLY_LAMBDA": kappa,
        "RISK_PER_TRADE": kappa,
        "BETA_ALPHA": beta_a,
        "BETA_REGIME_BULL": b_bull,
        "BETA_REGIME_BEAR": b_bear,
        "BETA_REGIME_CHOP": b_chop,
        "BETA_REGIME_CRISIS": b_crisis,
        "EV_HURDLE_BPS": ev_h,
        "REBALANCE_BARS": reb,
        "SLIPPAGE_BPS_BUFFER_MULT": slip_buf,
        "TIME_BARRIER_H": tb_h,
        "CRISIS_OVERRIDE_THRESHOLD": crisis_thr,
        "CRISIS_GAMMA": gamma,
        "SIZING_METHOD": "profit_factor_kelly",
        "MIN_SCORE_PERCENTILE": 0.55,
    }

    return finalize_strategy_portfolio_params(base, policy)


def build_ml_phase_d_params(trial_params: dict[str, Any], tf: str) -> dict[str, Any]:
    merged = dict(_fixed_ml_phase_d_params())
    merged.update(trial_params)
    return _base_engine_params(merged, tf)



def _pf_and_ev_cost_from_trades(all_trades: np.ndarray) -> tuple[float, float]:
    """PF = gross_win / |gross_loss|; EV/cost = |sum(pnl)| / sum(entry_fee + funding_fee)."""
    if all_trades.size == 0:
        return 1.0, 0.0
    pnl: np.ndarray = all_trades[:, 6].astype(np.float64, copy=False)
    gross_win = float(np.sum(pnl[pnl > 0.0]))
    gross_loss = float(np.sum(np.abs(pnl[pnl < 0.0])))
    avg_pf = gross_win / max(abs(gross_loss), 1e-9) if gross_loss != 0.0 else 1.0
    net_pnl = float(np.sum(pnl))
    fees = all_trades[:, 8].astype(np.float64, copy=False) + all_trades[:, 9].astype(
        np.float64, copy=False
    )
    total_fee = float(np.sum(fees))
    ev_cost_ratio = abs(net_pnl) / max(total_fee, 1e-9)
    return avg_pf, ev_cost_ratio


def _base_engine_params(ml: dict[str, Any], tf: str) -> dict[str, Any]:
    # In SOTA mode, KELLY_LAMBDA is used as the Shrinkage (lam)
    # RISK_PER_TRADE in engine is used as Kelly Lambda.
    ann_vol = float(ml.get("TARGET_ANN_VOL", 0.20))
    kelly_lambda = float(ml.get("KELLY_LAMBDA", 0.20))
    lev = float(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    cfg = OPT_FUTURES_CONFIG
    atm = float(ml.get("ATR_MULT", cfg.get("FUTURES_ATR_STOP_MULT", 2.5)))
    atr_period_fixed = int(cfg.get("FUTURES_ATR_PERIOD_FIXED", 30))

    return {
        "TIMEFRAME": tf,
        "SIGNAL_TYPE": "ML_CALIB_PROB",
        "REGIME_TYPE": "EMA_ATR",
        "SIZING_METHOD": "profit_factor_kelly",
        "USE_CS_RANK_ENGINE": False,
        "K_LONG": int(ml.get("K_LONG", 2)),
        "K_SHORT": int(ml.get("K_SHORT", 2)),
        "REBALANCE_BARS": int(ml.get("REBALANCE_BARS", 1)),
        "REBALANCE_TURNOVER_THRESHOLD": float(ml.get("REBALANCE_TURNOVER_THRESHOLD", 0.15)),
        "MIN_SCORE_PERCENTILE": float(ml.get("MIN_SCORE_PERCENTILE", 0.50)),
        "CRISIS_GAMMA": float(ml.get("CRISIS_GAMMA", 1.0)),
        "TRAIL_MULT": float(ml.get("TRAIL_MULT", atm)),
        "ATR_MULT": atm,
        "ATR_PERIOD": int(ml.get("ATR_PERIOD", atr_period_fixed)),
        "SHORT_TP_MULT": float(ml.get("SHORT_TP_MULT", 2.0)),
        "LONG_SCALE_ATR_MULT": float(ml.get("LONG_SCALE_ATR_MULT", 3.0)),
        "RISK_PER_TRADE": kelly_lambda,  # Used as Kelly Lambda in calculate_position_size
        "MAX_EXPOSURE_PER_COIN": float(ml.get("MAX_EXPOSURE_PER_COIN", 1.0)),
        "MAX_EXPOSURE": float(ml.get("MAX_EXPOSURE", 1.0)),
        "DD_SCALING_THRESHOLD": float(ml.get("DD_SCALING_THRESHOLD", 0.0)),
        "CS_Z_SCORE_THRESHOLD": float(ml.get("CS_Z_SCORE_THRESHOLD", 1.0)),
        "LONG_CS_Z_ENTRY": float(ml.get("LONG_CS_Z_ENTRY", ml.get("CS_Z_SCORE_THRESHOLD", 1.0))),
        "SHORT_CS_Z_ENTRY": float(ml.get("SHORT_CS_Z_ENTRY", ml.get("CS_Z_SCORE_THRESHOLD", 1.0))),
        "HYSTERESIS_GAP": float(ml.get("HYSTERESIS_GAP", 0.3)),
        "DYNAMIC_RA_CRISIS_COEF": float(ml.get("DYNAMIC_RA_CRISIS_COEF", 3.0)),
        "DYNAMIC_RA_BEAR_COEF": float(ml.get("DYNAMIC_RA_BEAR_COEF", 1.5)),
        "NORM_VAR_CONSTANT": float(ml.get("NORM_VAR_CONSTANT", 0.5)),
        "CRISIS_LONG_Z_BOOST": float(ml.get("CRISIS_LONG_Z_BOOST", 0.0)),
        "CRISIS_LONG_MAG_SUPPRESS": float(
            ml.get(
                "CRISIS_LONG_MAG_SUPPRESS",
                cfg.get("FUTURES_CRISIS_LONG_MAG_SUPPRESS", 1.0),
            )
        ),
        "TARGET_ANN_VOL": ann_vol,
        "KELLY_LAMBDA": kelly_lambda,
        "USE_COMPOUNDING": True,
        "LEVERAGE": int(lev),
        "BETA_ALPHA": float(ml.get("BETA_ALPHA", cfg.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0))),
        "BETA_REGIME_BULL": float(
            ml.get("BETA_REGIME_BULL", cfg.get("FUTURES_DEFAULT_BETA_REGIME_BULL", 1.0))
        ),
        "BETA_REGIME_BEAR": float(
            ml.get("BETA_REGIME_BEAR", cfg.get("FUTURES_DEFAULT_BETA_REGIME_BEAR", 0.25))
        ),
        "BETA_REGIME_CRISIS": float(
            ml.get("BETA_REGIME_CRISIS", cfg.get("FUTURES_DEFAULT_BETA_REGIME_CRISIS", -0.5))
        ),
        "BETA_REGIME_CHOP": float(
            ml.get("BETA_REGIME_CHOP", cfg.get("FUTURES_DEFAULT_BETA_REGIME_CHOP", 0.25))
        ),
        "EV_HURDLE_BPS": float(
            ml.get("EV_HURDLE_BPS", cfg.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 5.0))
        ),
        "SLIPPAGE_BPS_BUFFER_MULT": float(
            ml.get("SLIPPAGE_BPS_BUFFER_MULT", cfg.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0))
        ),
        "TIME_BARRIER_H": float(
            ml.get("TIME_BARRIER_H", cfg.get("FUTURES_DEFAULT_TIME_BARRIER_H", 0.0))
        ),
        "PORTFOLIO_KAPPA": float(
            ml.get("PORTFOLIO_KAPPA", cfg.get("FUTURES_PORTFOLIO_KAPPA", 0.35))
        ),
    }


def _cached_kill_fund_lev(
    aligned: dict[str, Any], params: dict[str, Any]
) -> tuple[Any, Any, Any]:
    if "kill_signal_cached" not in aligned:
        zkill = aligned.get("kill_signal")
        if zkill is None:
            zkill = np.zeros_like(aligned["close"])
        aligned["kill_signal_cached"] = zkill
    zkill = aligned["kill_signal_cached"]
    if "funding_rate_sum_cached" not in aligned:
        zfund = aligned.get("funding_rate_sum")
        if zfund is None:
            zfund = np.zeros_like(aligned["close"])
        aligned["funding_rate_sum_cached"] = zfund
    zfund = aligned["funding_rate_sum_cached"]
    if "dyn_leverage_cached" not in aligned:
        lev_blk = aligned.get("dyn_leverage")
        if lev_blk is None or lev_blk.shape != aligned["close"].shape:
            lev_blk = np.full_like(aligned["close"], float(params["LEVERAGE"]), dtype=np.float64)
        else:
            lev_blk = np.maximum(lev_blk.astype(np.float64, copy=False), 1.0)
        aligned["dyn_leverage_cached"] = lev_blk
    lev_blk = aligned["dyn_leverage_cached"]
    return zkill, zfund, lev_blk


def _run_portfolio_numba_block(
    params: dict[str, Any],
    aligned: dict[str, Any],
    zkill: Any,
    zfund: Any,
    lev_blk: Any,
    estimated_b: float = 1.05,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    _ = estimated_b
    cfg_block = OPT_FUTURES_CONFIG
    reb_b = max(1, int(params["REBALANCE_BARS"]))
    pwp = portfolio_weight_params_from_optuna(params, cfg_block)

    # T3-A: Kelly × HMM Entropy 동적 스케일링
    _hmm_cols_t3 = [
        "hmm_prob_bull_calm", "hmm_prob_bull_vol_up", "hmm_prob_bear_trend",
        "hmm_prob_chop", "hmm_prob_crisis",
    ]
    _hmm_t3 = [aligned.get(c) for c in _hmm_cols_t3]
    if all(a is not None for a in _hmm_t3):
        def _to_1d(a: Any) -> np.ndarray:
            arr = np.asarray(a, dtype=np.float64)
            return arr[:, 0] if arr.ndim == 2 else arr
        _p5 = np.stack([_to_1d(a) for a in _hmm_t3], axis=1)  # (n_bars, 5)
        _log5 = np.log(5.0)
        _ent = -np.sum(_p5 * np.log(np.clip(_p5, 1e-12, 1.0)), axis=1)
        _h_norm = float(np.mean(_ent) / _log5)
        _mean_crisis = float(np.mean(np.clip(_p5[:, 4], 0.0, 1.0)))
        _kelly_disc = max(0.1, (1.0 - _h_norm) * (1.0 - _mean_crisis))
        pwp["f_kelly_max"] = float(pwp["f_kelly_max"]) * _kelly_disc
    # T3-B: IC EWMA → Kelly upper bound
    pwp["f_kelly_max"] = min(float(pwp["f_kelly_max"]), float(params.get("KELLY_IC_UPPER", 0.5)))

    hpb = hours_per_bar_from_timeframe(str(params.get("TIMEFRAME", "4h")))
    bars_py = (365.0 * 24.0) / max(hpb, 1e-9)
    close_np = np.asarray(aligned["close"], dtype=np.float64)
    xl = np.asarray(
        aligned.get("xs_score_long", np.zeros_like(close_np)), dtype=np.float64
    )
    xs = np.asarray(
        aligned.get("xs_score_short", np.zeros_like(close_np)), dtype=np.float64
    )
    sigma_3d = aligned.get("sigma_3d")
    tw_blk = np.asarray(
        precompute_rebalance_weights(
            close_np,
            xl,
            xs,
            rebalance_bars=reb_b,
            lookback=int(pwp["lookback"]),
            bars_per_year=bars_py,
            kappa=float(pwp["kappa"]),
            f_kelly_max=float(pwp["f_kelly_max"]),
            sigma_target_ann=float(pwp["sigma_target_ann"]),
            gross_cap=float(pwp["gross_cap"]),
            per_symbol_cap=float(pwp["per_symbol_cap"]),
            current_dd=0.0,
            composer_sigma_2d=(
                np.asarray(aligned["composer_sigma_bar"], dtype=np.float64)
                if aligned.get("composer_sigma_bar") is not None
                else None
            ),
            sigma_3d=sigma_3d,
        ),
        dtype=np.float64,
    )
    aligned["target_weights"] = tw_blk
    use_simple_atr_i = (
        1 if bool(cfg_block.get("FUTURES_SIMPLE_ATR_STOP", True)) else 0
    )
    mx_hold = max_hold_bars_from_time_barrier(params)
    sborr = float(cfg_block.get("FUTURES_SHORT_BORROW_DAILY", 0.0))
    buf_mult = float(
        params.get(
            "SLIPPAGE_BPS_BUFFER_MULT",
            OPT_FUTURES_CONFIG.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0),
        )
    )
    slip_eff = float(SLIPPAGE_RATE) * max(buf_mult, 1e-9)
    atr_m = float(params["ATR_MULT"])
    trail_m = float(params["TRAIL_MULT"])

    out_tw = backtest_target_weights_numba(
        aligned["close"],
        aligned["high"],
        aligned["low"],
        aligned["open"],
        zfund,
        zkill,
        tw_blk,
        float(FUTURES_INITIAL_BALANCE),
        lev_blk,
        MAKER_FEE_RATE,
        TAKER_FEE_RATE,
        slip_eff,
        reb_b,
        mx_hold,
        sborr,
        aligned["atr"],
        atr_m,
        trail_m,
        use_simple_atr_i,
        int(params.get("MAX_CONCURRENT_POSITIONS", 2)),
        float(params.get("MAX_EXPOSURE", 0.8)),
        float(params["MAX_EXPOSURE_PER_COIN"]),
        float(params["DD_SCALING_THRESHOLD"]),
        volume_2d=aligned.get("volume"),
    )
    return cast(tuple[np.ndarray, float, np.ndarray, np.ndarray], out_tw)


def _awf_gate_stat_ref_bars(awf_slices: list[dict[str, Any]]) -> int:
    tot = 0
    for leg in awf_slices:
        lr = leg.get("leg_range")
        if isinstance(lr, (tuple, list)) and len(lr) >= 2:
            tot += max(0, int(lr[1]) - int(lr[0]))
    return max(tot, 1)



def rerun_precompute_for_ctx(ctx: MLPhaseDContext) -> None:
    """Force regeneration of anchored AWF caches (different seed/calibration path)."""
    ctx.awf_leg_slices = None
    ctx.multi_alignment_info = None
    ctx.calibrator = None
    ctx.calibrator_short = None
    precompute_ml_optimization_context(ctx)


def replay_robust_awf_for_trial_params(
    ctx: MLPhaseDContext, raw_optuna_params: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Replay AWF legs with fixed tuned Optuna param dict (no Optuna Trial); for multi-seed checks."""
    if ctx.awf_leg_slices is None:
        rerun_precompute_for_ctx(ctx)
    merged_full = build_ml_phase_d_params(raw_optuna_params, ctx.tf)
    return _evaluate_awf_phase_d_aggregate(ctx, merged_full, trial=None)


def _evaluate_awf_phase_d_aggregate(
    ctx: MLPhaseDContext,
    ml_bundle: dict[str, Any],
    trial: optuna.Trial | None,
) -> tuple[float | tuple[float, float], dict[str, Any]]:
    """Core AWF leg loop + robust objective."""
    cfg = OPT_FUTURES_CONFIG
    awf_slices = ctx.awf_leg_slices or []
    mai = ctx.multi_alignment_info
    if not awf_slices or mai is None:
        diag = {"empty": True, "robust_val": (-1e9)}
        fail = 1e9
        ns = bool(cfg.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False))
        return ((fail, fail) if ns else fail), diag

    if ml_bundle.get("TIMEFRAME"):
        params = dict(ml_bundle)
    else:
        params = _base_engine_params(ml_bundle, ctx.tf)
    params["ESTIMATED_B"] = ctx.estimated_b
    params["KELLY_IC_UPPER"] = ctx.kelly_ic_upper  # T3-B

    n_trials_eff = int(cfg.get("total_trials", 400))
    if ctx.effective_total_trials is not None:
        n_trials_eff = max(int(ctx.effective_total_trials), 1)
    gate_stat_ref = _awf_gate_stat_ref_bars(awf_slices)

    if trial is not None and trial.number < 10 and awf_slices:
        ad0 = awf_slices[0].get("data") or {}
        xl = ad0.get("xs_score_long")
        hy = ad0.get("hmm_prob_crisis")
        if xl is not None and hy is not None and getattr(xl, "size", 0) > 0:
            disp = float(np.nanstd(np.asarray(xl, dtype=np.float64)))
            cclip = np.clip(np.asarray(hy, dtype=np.float64), 0.0, 1.0)
            gamma = float(params.get("CRISIS_GAMMA", params.get("CRISIS_GATE_PROB", 1.0)))
            soft_m = float(np.mean((1.0 - cclip) ** gamma))
            thr = float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.6))
            rej_r = float(np.mean(np.max(hy, axis=1) > thr))
            trial.set_user_attr("xs_score_dispersion_mean", disp)
            trial.set_user_attr("crisis_soft_weight_mean", soft_m)
            trial.set_user_attr("crisis_gate_rejection_rate", rej_r)

    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))
    n_syms_ctx = max(1, len(ctx.symbols))

    leg_log_tw: list[float] = []
    leg_mdds: list[float] = []
    all_trades_chunks: list[np.ndarray] = []
    leg_trade_counts: list[float] = []
    leg_long_counts: list[int] = []
    leg_short_counts: list[int] = []
    leg_l_pf: list[float] = []
    leg_s_pf: list[float] = []
    leg_exposures: list[float] = []
    leg_crisis_mean: list[float] = []
    first_leg_done = False

    for leg_idx, leg in enumerate(awf_slices):
        aligned = leg.get("data")
        leg_range: tuple[int, int] = leg["leg_range"]
        if not aligned:
            leg_log_tw.append(-10.0)
            leg_mdds.append(100.0)
            leg_trade_counts.append(0.0)
            leg_exposures.append(0.0)
            leg_crisis_mean.append(0.0)
            continue

        zkill, zfund, lev_leg = _cached_kill_fund_lev(aligned, params)
        b_trades_raw, b_bal, b_equity, _b_diag = _run_portfolio_numba_block(
            params, aligned, zkill, zfund, lev_leg, ctx.estimated_b
        )

        n_tr = int(b_trades_raw.shape[0])
        all_trades_chunks.append(b_trades_raw)

        if not first_leg_done:
            first_leg_done = True
            if n_tr == 0:
                _logger.debug(
                    "⚠️  Zero trades | trial=%s diag=%s",
                    trial.number if trial is not None else "replay",
                    _b_diag,
                )
                if trial is not None:
                    raise optuna.TrialPruned()
                diag = {"pruned": True, "robust_val": (-1e9)}
                return 1e9, diag
            
        mdd = float(calc_mdd_from_equity(b_equity)) if b_equity.size > 0 else 100.0
        log_ret = _log_tw_from_ret_pct(float((b_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0))

        # [Speed Optimization] Early Pruning after first AWF leg
        if leg_idx == 0 and trial is not None:
            if log_ret < -0.1:
                raise optuna.TrialPruned()


        if mdd >= liq_mdd_thr:
            log_ret -= (mdd - liq_mdd_thr) * 3.0

        b_bars = max(1, leg_range[1] - leg_range[0])
        b_exposure = 0.0
        n_long, n_short = 0, 0
        if n_tr > 0:
            holding_bars = float(np.sum(b_trades_raw[:, 2] - b_trades_raw[:, 1]))
            b_exposure = holding_bars / float(b_bars * n_syms_ctx)
            n_long = int(np.sum(b_trades_raw[:, 3] == 1.0))
            n_short = int(np.sum(b_trades_raw[:, 3] == -1.0))

        if n_tr > 0 and b_trades_raw.size > 0:
            _pnl_arr = b_trades_raw[:, 6].astype(np.float64, copy=False)
            _dir_arr = b_trades_raw[:, 3]
            _l_pnl = _pnl_arr[_dir_arr == 1.0]
            _s_pnl = _pnl_arr[_dir_arr == -1.0]
            _l_win = float(np.sum(_l_pnl[_l_pnl > 0.0]))
            _l_loss = float(np.sum(np.abs(_l_pnl[_l_pnl < 0.0])))
            _s_win = float(np.sum(_s_pnl[_s_pnl > 0.0]))
            _s_loss = float(np.sum(np.abs(_s_pnl[_s_pnl < 0.0])))
            _lpf = _l_win / max(_l_loss, 1e-9) if _l_loss > 0 else (1.5 if _l_win > 0 else 1.0)
            _spf = _s_win / max(_s_loss, 1e-9) if _s_loss > 0 else (1.5 if _s_win > 0 else 1.0)
        else:
            _lpf, _spf = 1.0, 1.0
        leg_l_pf.append(_lpf)
        leg_s_pf.append(_spf)

        _hy_arr = aligned.get("hmm_prob_crisis") if aligned else None
        if _hy_arr is not None:
            try:
                _hy_np = np.asarray(_hy_arr, dtype=np.float64)
                if _hy_np.ndim > 1:
                    _hy_np = _hy_np[:, 0]
                leg_crisis_mean.append(float(np.nanmean(_hy_np)))
            except Exception:
                leg_crisis_mean.append(0.0)
        else:
            leg_crisis_mean.append(0.0)

        leg_log_tw.append(log_ret)
        leg_mdds.append(mdd)
        leg_trade_counts.append(float(n_tr))
        leg_long_counts.append(n_long)
        leg_short_counts.append(n_short)
        leg_exposures.append(b_exposure)

        if leg_idx >= 1 and trial is not None:
            cum_log_tw = float(np.sum(leg_log_tw))
            max_leg_mdd = float(np.max(leg_mdds))
            if (not np.isfinite(cum_log_tw)) or (not np.isfinite(max_leg_mdd)):
                break
            if abs(cum_log_tw) > 100.0:
                break
            if cum_log_tw < -0.25 or max_leg_mdd > liq_mdd_thr:
                break
            if leg_idx >= 2:
                prune_min_pos = float(cfg.get("FUTURES_PHASE_D_PRUNE_MIN_POS_RATIO", 0.25))
                part = np.asarray(leg_log_tw, dtype=np.float64)
                pos_r = float(np.sum(part > 0.0)) / float(max(part.size, 1))
                if pos_r < prune_min_pos:
                    pass

            if trial is not None and len(trial.study.directions) == 1:
                trial.report(float(np.mean(leg_log_tw)), step=leg_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    leg_arr = np.asarray(leg_log_tw, dtype=np.float64)
    n_legs_done = leg_arr.size
    all_trades = (
        np.vstack(all_trades_chunks) if all_trades_chunks
        else np.zeros((0, 10), dtype=np.float64)
    )

    avg_trades_agg = float(np.mean(leg_trade_counts)) if leg_trade_counts else 0.0
    worst_mdd_legs = float(max(leg_mdds, default=100.0))
    avg_exposure = float(np.mean(leg_exposures)) if leg_exposures else 0.0
    total_long = sum(leg_long_counts)
    total_short = sum(leg_short_counts)
    total_dir = total_long + total_short
    minority = float(min(total_long, total_short) / total_dir) if total_dir > 0 else 0.0

    l_pf_agg, s_pf_agg = 1.0, 1.0
    if all_trades.size > 0:
        pnl_arr = all_trades[:, 6].astype(np.float64, copy=False)
        dir_arr = all_trades[:, 3]
        l_mask = dir_arr == 1.0
        s_mask = dir_arr == -1.0
        l_pnl = pnl_arr[l_mask]
        l_win = float(np.sum(l_pnl[l_pnl > 0.0]))
        l_loss = float(np.sum(np.abs(l_pnl[l_pnl < 0.0])))
        l_pf_agg = l_win / max(l_loss, 1e-9) if l_loss > 0 else 1.0
        s_pnl = pnl_arr[s_mask]
        s_win = float(np.sum(s_pnl[s_pnl > 0.0]))
        s_loss = float(np.sum(np.abs(s_pnl[s_pnl < 0.0])))
        s_pf_agg = s_win / max(s_loss, 1e-9) if s_loss > 0 else 1.0

    lam_m = float(cfg.get("FUTURES_AWF_OBJ_LAMBDA_MAD", 1.0))
    psi_d = float(cfg.get("FUTURES_AWF_OBJ_PSI_DD", 0.5))
    robust_val = compute_awf_robust_objective_score(
        leg_arr,
        worst_mdd_legs,
        lambda_mad=lam_m,
        psi_dd=psi_d,
    )

    obj = -robust_val

    k_cfg = int(cfg.get("FUTURES_AWF_K_LEGS", 6))
    if n_legs_done < k_cfg:
        obj += 20.0 * (k_cfg - n_legs_done)

    k_legs_n = float(max(n_legs_done, 1))
    mu_log = float(np.mean(leg_arr)) if leg_arr.size > 0 else -10.0
    worst_leg = float(np.min(leg_arr)) if leg_arr.size > 0 else -10.0
    awf_pos_frac = float(np.sum(leg_arr > 0.0)) / k_legs_n
    dsr_awf = calc_gate1_dsr_from_path_log_tw(
        leg_arr,
        ctx.tf,
        float(gate_stat_ref),
        float(n_trials_eff),
    )

    sig_awf_diag = float(np.std(leg_arr, ddof=1)) if leg_arr.size >= 2 else 0.0

    diag: dict[str, Any] = {
        "objective": float(obj),
        "robust_val": float(robust_val),
        "awf_robust_score": float(robust_val),
        "awf_contract_reward": float(robust_val),
        "awf_plgd": float(robust_val),
        "awf_plgd_n_trials": int(n_trials_eff),
        "gate1_eff_ref_len": int(gate_stat_ref),
        "awf_path_leg_log_tw": [float(x) for x in leg_log_tw],
        "cpcv_path_oos_log_tw": [float(x) for x in leg_log_tw],
        "awf_leg_log_tw": [float(x) for x in leg_log_tw],
        "awf_leg_trade_counts": [float(x) for x in leg_trade_counts],
        "awf_pos_frac": awf_pos_frac,
        "gate1_dsr": dsr_awf,
        "dsr_awf": dsr_awf,
        "n_negative_legs": int(np.sum(leg_arr < 0.0)),
        "leg_l_pf": [round(x, 4) for x in leg_l_pf],
        "leg_s_pf": [round(x, 4) for x in leg_s_pf],
        "leg_long_counts": leg_long_counts,
        "leg_short_counts": leg_short_counts,
        "awf_mu_log": mu_log,
        "mu_log": mu_log,
        "awf_sigma_log": sig_awf_diag,
        "sig_awf_diag": sig_awf_diag,
        "awf_mean_log_tw": float(mu_log),
        "ml_mean_log_growth_cpcv": mu_log,
        "awf_worst_leg_log_tw": float(worst_leg),
        "worst_leg": worst_leg,
        "ml_p10_log_growth_cpcv": worst_leg,
        "awf_worst_mdd_pct": float(worst_mdd_legs),
        "worst_mdd_legs": worst_mdd_legs,
        "ml_worst_mdd_cpcv": worst_mdd_legs,
        "avg_trades": avg_trades_agg,
        "avg_exposure": avg_exposure,
        "long_short_ratio": minority,
        "leg_crisis_mean": [round(x, 4) for x in leg_crisis_mean],
        "n_valid_paths": int(n_legs_done),
        "l_pf_agg": l_pf_agg,
        "s_pf_agg": s_pf_agg,
    }

    if trial is not None:
        # Scalar-only storage to prevent DB bottleneck
        trial.set_user_attr("awf_mu_log", mu_log)
        trial.set_user_attr("awf_sigma_log", sig_awf_diag)
        trial.set_user_attr("awf_robust_score", float(robust_val))
        trial.set_user_attr("awf_pos_frac", awf_pos_frac)
        trial.set_user_attr("gate1_dsr", dsr_awf)
        trial.set_user_attr("awf_worst_leg_log_tw", float(worst_leg))
        trial.set_user_attr("awf_worst_mdd_pct", float(worst_mdd_legs))
        trial.set_user_attr("avg_trades", avg_trades_agg)

    ns2 = bool(cfg.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False))
    if ns2:
        # Obj1: compound growth 최대화 (Kelly criterion)
        obj1 = -float(np.mean(leg_arr)) if leg_arr.size > 0 else 1e9
        # Obj2: worst leg 최대화 (tail risk 방어)
        obj2 = -float(np.min(leg_arr)) if leg_arr.size > 0 else 1e9
        return (obj1, obj2), diag
    return float(obj), diag


def _evaluate_is_phase_d(
    ctx: MLPhaseDContext,
    ml_bundle: dict[str, Any],
    trial: optuna.Trial | None,
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Single-IS backtest for decoupled optimization."""
    cfg = OPT_FUTURES_CONFIG
    aligned = ctx.is_slice
    mai = ctx.multi_alignment_info

    if aligned is None or mai is None:
        fail = 1e9
        return (fail, fail), {"empty": True}

    params = ml_bundle if ml_bundle.get("TIMEFRAME") else _base_engine_params(ml_bundle, ctx.tf)
    params["ESTIMATED_B"] = ctx.estimated_b

    # Run IS backtest
    zkill, zfund, lev_blk = _cached_kill_fund_lev(aligned, params)
    b_trades_raw, b_bal, b_equity, _b_diag = _run_portfolio_numba_block(
        params, aligned, zkill, zfund, lev_blk, ctx.estimated_b
    )

    n_tr = int(b_trades_raw.shape[0])
    # Early Pruning: zero trades
    if n_tr == 0:
        if trial is not None:
            raise optuna.TrialPruned()
        return (1e9, 1e9), {"pruned": True}

    # Early Pruning: excessive loss (> 20%)
    is_ret_pct = (b_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0
    if trial is not None and is_ret_pct < -20.0:
        raise optuna.TrialPruned()

    is_mdd = float(calc_mdd_from_equity(b_equity))

    # Split equity into 10 chunks to calculate robust score and DSR
    k_chunks = 10
    n_bars = b_equity.size
    chunk_size = max(1, n_bars // k_chunks)
    leg_log_tw = []
    for i in range(k_chunks):
        s = i * chunk_size
        e = (i + 1) * chunk_size if i < k_chunks - 1 else n_bars
        if e > s + 1:
            chunk_ret = (b_equity[e-1] / b_equity[s] - 1.0) * 100.0
            leg_log_tw.append(_log_tw_from_ret_pct(chunk_ret))

    leg_arr = np.asarray(leg_log_tw, dtype=np.float64)
    robust_val = compute_awf_robust_objective_score(
        leg_arr, is_mdd,
        lambda_mad=float(cfg.get("FUTURES_AWF_OBJ_LAMBDA_MAD", 1.0)),
        psi_dd=float(cfg.get("FUTURES_AWF_OBJ_PSI_DD", 0.5))
    )

    # DSR calculation for constraints
    n_trials_eff = int(cfg.get("total_trials", 1500))
    if ctx.effective_total_trials is not None:
        n_trials_eff = max(int(ctx.effective_total_trials), 1)

    is_dsr = calc_gate1_dsr_from_path_log_tw(
        leg_arr, ctx.tf, float(n_bars), float(n_trials_eff)
    )

    obj1 = -float(robust_val)
    obj2 = -float(np.min(leg_arr)) if leg_arr.size > 0 else 1e9

    if trial is not None:
        # Scalar-only storage
        trial.set_user_attr("IS_MDD", is_mdd)
        trial.set_user_attr("IS_DSR", is_dsr)
        trial.set_user_attr("IS_RET_PCT", float(is_ret_pct))
        trial.set_user_attr("IS_ROBUST_SCORE", float(robust_val))
        trial.set_user_attr("avg_trades", float(n_tr))

    diag = {
        "is_ret_pct": is_ret_pct,
        "is_mdd": is_mdd,
        "is_dsr": is_dsr,
        "robust_val": robust_val,
        "obj1": obj1,
        "obj2": obj2,
    }
    return (obj1, obj2), diag


def objective_ml_phase_d(trial: optuna.Trial, ctx: MLPhaseDContext) -> tuple[float, float]:
    """Joint NSGA-II Portfolio Optimization — AWF-based objectives (T2).

    T2: AWF leg log-TW를 직접 목적함수로 사용 (IS-only 탈피).
    Obj1 = -mean(leg_log_tw),  Obj2 = -min(leg_log_tw)  [both minimized].
    """
    if ctx.awf_leg_slices is None:
        precompute_ml_optimization_context(ctx)
    merged = _suggest_ml_joint_nsga2(trial, ctx)
    # T2: AWF leg log-TW 직접 목적함수 (NSGA-II 분기 강제 활성)
    result, _ = _evaluate_awf_phase_d_aggregate(ctx, merged, trial=trial)
    if isinstance(result, tuple):
        return result
    # FUTURES_ML_ALPHA_NSGA2_ENABLED=False 환경에서의 fallback — tuple로 통일
    scalar = float(result)
    return (scalar, scalar)


def select_best_trial_by_holdout_log_ret(trials: list[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    if not trials:
        raise ValueError("empty trials")

    def _score(t: optuna.trial.FrozenTrial) -> tuple[float, float, float, float, float, float, float]:
        holdout = float(np.clip(t.user_attrs.get("ml_holdout_log_ret", 0.0), -2.0, 2.0))
        robust = float(t.user_attrs.get("awf_robust_score", t.user_attrs.get("awf_contract_reward", -1e9)))
        is_cpcv = float(
            np.clip(
                t.user_attrs.get("awf_mean_log_tw", t.user_attrs.get("ml_mean_log_growth_cpcv", -2.0)),
                -2.0,
                2.0,
            )
        )
        p10_cpcv = float(
            np.clip(
                t.user_attrs.get(
                    "awf_worst_leg_log_tw", t.user_attrs.get("ml_p10_log_growth_cpcv", -2.0)
                ),
                -2.0,
                2.0,
            )
        )
        worst_mdd = float(
            t.user_attrs.get("awf_worst_mdd_pct", t.user_attrs.get("ml_worst_mdd_cpcv", 999.0))
        )
        dsr = float(t.user_attrs.get("gate1_dsr", 0.0))
        path_std = float(np.clip(t.user_attrs.get("ml_std_log_growth_cpcv", 1.0), 0.0, 2.0))
        if is_cpcv < 0:
            holdout = holdout - abs(is_cpcv) * 2.0
        return (robust, dsr, is_cpcv, p10_cpcv, -path_std, -worst_mdd, holdout)

    return max(trials, key=_score)


def topsis_select_best(pareto_trials: list[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    if not pareto_trials:
        raise ValueError("empty pareto_trials")
    if len(pareto_trials) == 1:
        return pareto_trials[0]
    n_dim = int(max(len(t.values) for t in pareto_trials))
    vals_list: list[list[float]] = []
    for t in pareto_trials:
        row = [float(x) for x in t.values]
        while len(row) < n_dim:
            row.append(0.0)
        vals_list.append(row[:n_dim])
    vals = np.asarray(vals_list, dtype=np.float64)
    vmin, vmax = vals.min(axis=0), vals.max(axis=0)
    norm = (vals - vmin) / np.where(vmax - vmin < 1e-12, 1.0, vmax - vmin)
    
    # Asymmetric Weights: [0.7 (Growth/Obj1), 0.3 (Tail Risk/Obj2)]
    weights = np.array([0.7, 0.3]) if n_dim == 2 else np.ones(n_dim) / n_dim
    
    ideal: np.ndarray = np.zeros(n_dim, dtype=np.float64)
    nadir: np.ndarray = np.ones(n_dim, dtype=np.float64)
    
    # Weighted Euclidean distance
    d_pos = np.sqrt(np.sum(weights * (norm - ideal)**2, axis=1))
    d_neg = np.sqrt(np.sum(weights * (norm - nadir)**2, axis=1))
    
    return pareto_trials[int(np.argmax(d_neg / (d_pos + d_neg + 1e-12)))]


def check_hard_gates_ml(
    oos_result: dict[str, Any],
    pbo_val: float,
    dsr_val: float,
    is_precision: float,
    *,
    pbo_max_override: float | None = None,
    dsr_min_override: float | None = None,
) -> bool:
    from config.opt_config import OPT_FUTURES_CONFIG
    cfg = OPT_FUTURES_CONFIG
    pbo_lim = float(pbo_max_override if pbo_max_override is not None else cfg.get("FUTURES_PBO_MAX", 0.45))
    pbo_ok = pbo_val < pbo_lim
    dsr_floor = float(dsr_min_override if dsr_min_override is not None else cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.20))
    dsr_ok = dsr_val >= dsr_floor
    wr_pct = float(oos_result.get("win_rate_pct", oos_result.get("win_rate", 0.0)))
    wr_frac = wr_pct / 100.0 if wr_pct > 1.0 else wr_pct
    wr_ok = wr_frac >= is_precision * 0.85
    mdd_v = float(oos_result.get("mdd_pct", oos_result.get("mdd", 100.0)))
    mdd_ok = abs(mdd_v) < float(cfg.get("FUTURES_MAX_MDD", 25.0))
    l_pf = float(oos_result.get("long_profit_factor", oos_result.get("oos_long_pf", 1.0)))
    s_pf = float(oos_result.get("short_profit_factor", oos_result.get("oos_short_pf", 1.0)))
    combined_pf = float(oos_result.get("profit_factor", (l_pf + s_pf) / 2.0))
    dir_ok = combined_pf >= 1.05
    return bool(pbo_ok and dsr_ok and wr_ok and mdd_ok and dir_ok)
