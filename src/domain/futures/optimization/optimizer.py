"""Phase D: CAWF-R ML cross-sectional rank portfolio optimization (Optuna TPE).

Walk-forward legs use a composite objective (positive-leg ratio, worst-leg log-TW,
directional PF, IS log-alpha) with light PLGD variance regularization; legacy PLGD
components remain in user_attrs for diagnostics.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FrozenTrial
from sklearn.linear_model import LogisticRegression

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_INITIAL_BALANCE, SLIPPAGE_RATE, TRADING_FEE_RATE
from src.domain.futures.backtest_engine import (
    _recompute_cs_dirs_numba,
    backtest_portfolio_numba,
)
from src.domain.futures.ml_pipeline.features.engineering import HMM_SEMANTIC_PROB_COLUMNS
from src.domain.futures.optimization.data_aligner import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
)
from src.domain.futures.optimization.evaluator import (
    _log_tw_from_ret_pct,
    calc_mdd_from_equity,
)
from src.domain.futures.optimization.validation import (
    build_anchored_wf_legs,
    build_cpcv_test_paths_with_fallback,
    list_cpcv_block_ranges,
)
from src.domain.futures.portfolio.policy_engine import (
    apply_policy_constraints,
    load_portfolio_policy_config,
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


def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: dict[str, int] = {"1h": 24}
    ratio_map: dict[str, float] = {"1h": 0.08}
    ratio: float = ratio_map.get(tf, 0.05)
    return max(fixed_min.get(tf, 12), int(longest_indicator_period * ratio))


EMBARGO_BARS: dict[str, int] = {
    "1h": compute_embargo_bars("1h"),
}


class SignalCalibrator:
    """Platt Scaling: Map Alpha scores (0-1) to true Win Probabilities using IS data."""

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
                _logger.info(
                    "[SignalCalibrator] Fitted Platt Scaling: coef=%.4f intercept=%.4f",
                    self.model.coef_[0][0],
                    self.model.intercept_[0],
                )
            except Exception as e:
                _logger.warning("[SignalCalibrator] Logistic fit failed: %s", e)

        # Estimate average win/loss ratio (b) for Kelly
        pos_rets = returns[returns > 0]
        neg_rets = np.abs(returns[returns < 0])
        if pos_rets.size > 0 and neg_rets.size > 0:
            self.mean_b = float(np.mean(pos_rets) / np.mean(neg_rets))
            _logger.info("[SignalCalibrator] Estimated b (Avg Win/Loss)=%.4f", self.mean_b)

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
    # CAWF-R: K chronological AWF test-leg slices (replaces cpcv_block_slices).
    awf_leg_slices: list[dict[str, Any]] | None = None
    # Legacy fields kept for backward-compat; cleared in precompute under CAWF-R.
    cpcv_block_slices: list[dict[str, Any]] | None = None
    holdout_slice: dict[str, np.ndarray] | None = None
    multi_alignment_info: dict[str, Any] | None = None
    # SOTA: Mathematical Policy Components
    calibrator: SignalCalibrator | None = None
    estimated_b: float = 1.05


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
        if ctx.cpcv_block_slices is not None:
            return

        # 1. Alignment & Baseline Signals
        pre_ml = {
            "TRAILING_ACTIVATION_ATR": 1.0,
            "BAYESIAN_C": 10.0,
            "KELLY_SHRINKAGE": 0.3,
            "K_LONG": 2,
            "K_SHORT": 2,
            "REBALANCE_BARS": 6,
            "MIN_SCORE_PERCENTILE": 0.55,
            "CRISIS_GAMMA": 1.0,
            "USE_CS_RANK_ENGINE": True,
        }
        params = _base_engine_params(pre_ml, ctx.tf)
        FuturesMLStrategy(name="Precompute", params=params)

        emb = int(EMBARGO_BARS.get(ctx.tf, 12))
        info = compute_multi_alignment_info(ctx.data_maps, ctx.symbols, ctx.tf, emb)
        if info is None:
            return
        ctx.multi_alignment_info = info

        # 1.5 SOTA: Signal Calibration (Platt Scaling)
        all_alphas = []
        all_returns = []
        eff_len = int(info["eff_ref_len"])
        for sym in ctx.symbols:
            start_idx = int(info["alignment_offsets"][sym])
            sym_df = ctx.data_maps[sym][ctx.tf]
            # Calibration must happen on the aligned IS zone (eff_len)
            raw_is = sym_df.iloc[start_idx : start_idx + eff_len]
            if "ml_alpha_00" in raw_is.columns:
                alpha_arr = raw_is["ml_alpha_00"].to_numpy(dtype=np.float64)
                # Forward return (12-bar) for probability calibration to capture longer edge
                fwd_ret = raw_is["close"].pct_change(12).shift(-12).to_numpy(dtype=np.float64)
                mask = ~np.isnan(alpha_arr) & ~np.isnan(fwd_ret)
                if mask.any():
                    all_alphas.append(alpha_arr[mask])
                    all_returns.append(fwd_ret[mask])

        if all_alphas:
            calib = SignalCalibrator()
            calib.fit(np.concatenate(all_alphas), np.concatenate(all_returns))
            ctx.calibrator = calib
            ctx.estimated_b = calib.mean_b

        # 2. Build Global Aligned Arrays
        prebuilt_full: dict[str, dict[str, np.ndarray]] = {}
        eff_len = int(info["eff_ref_len"])
        for sym in ctx.symbols:
            start_idx = int(info["alignment_offsets"][sym])
            raw_full = ctx.data_maps[sym][ctx.tf].iloc[start_idx:start_idx+eff_len]
            
            # Create a trimmed signal dataframe from raw_full
            trimmed_sig = pd.DataFrame(index=raw_full.index)
            trimmed_sig["close"] = raw_full["close"].to_numpy(dtype=np.float64, copy=False)
            trimmed_sig["high"] = raw_full["high"].to_numpy(dtype=np.float64, copy=False)
            trimmed_sig["low"] = raw_full["low"].to_numpy(dtype=np.float64, copy=False)
            trimmed_sig["open"] = raw_full["open"].to_numpy(dtype=np.float64, copy=False)
            trimmed_sig["atr"] = (
                raw_full["atr"].to_numpy(dtype=np.float64, copy=False)
                if "atr" in raw_full.columns
                else np.zeros(len(raw_full))
            )
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

            # SOTA: Predict win probabilities p using calibrated model and compute Kelly f*
            gp_base = (
                raw_full["ml_alpha_00"].to_numpy(dtype=np.float64, copy=False)
                if "ml_alpha_00" in raw_full.columns
                else np.zeros(len(raw_full), dtype=np.float64)
            )
            if ctx.calibrator:
                p_base = ctx.calibrator.predict_prob(gp_base)
                b = ctx.estimated_b
                # f* = p - (1-p)/b
                f_base = p_base - (1.0 - p_base) / b
                trimmed_sig["ml_calib_prob"] = np.clip(f_base, 0.0, 1.0)

                if "ml_alpha_long" in raw_full.columns:
                    p_l = ctx.calibrator.predict_prob(
                        raw_full["ml_alpha_long"].to_numpy(dtype=np.float64, copy=False)
                    )
                    f_l = p_l - (1.0 - p_l) / b
                    trimmed_sig["ml_calib_prob_long"] = np.clip(f_l, 0.0, 1.0)
                else:
                    trimmed_sig["ml_calib_prob_long"] = trimmed_sig["ml_calib_prob"]

                if "ml_alpha_short" in raw_full.columns:
                    p_s = ctx.calibrator.predict_prob(
                        raw_full["ml_alpha_short"].to_numpy(dtype=np.float64, copy=False)
                    )
                    f_s = p_s - (1.0 - p_s) / b
                    trimmed_sig["ml_calib_prob_short"] = np.clip(f_s, 0.0, 1.0)
                else:
                    trimmed_sig["ml_calib_prob_short"] = trimmed_sig["ml_calib_prob"]
            else:
                trimmed_sig["ml_calib_prob"] = raw_full.get("ml_calib_prob", 1.0)
                trimmed_sig["ml_calib_prob_long"] = raw_full.get("ml_calib_prob_long", 1.0)
                trimmed_sig["ml_calib_prob_short"] = raw_full.get("ml_calib_prob_short", 1.0)

            gp_centered = gp_base - 0.5
            trimmed_sig["trend_direction"] = np.where(
                np.abs(gp_centered) > 0.01, np.sign(gp_centered), 0.0
            ).astype(np.float64)
            trimmed_sig["entry_upper"] = 0.0
            trimmed_sig["entry_lower"] = 999999.0
            _xs_cols = (
                "xs_score_long",
                "xs_score_short",
                "hmm_prob_crisis",
                "hmm_modulator_long",
                "hmm_modulator_short",
                "hmm_modulator_base_long",
                "btc_trend_vol_adj_24h",
            )
            for col in _xs_cols:
                if col in raw_full.columns:
                    trimmed_sig[col] = raw_full[col].to_numpy(dtype=np.float64, copy=False)

            _inject_dyn_leverage_trimmed(trimmed_sig, raw_full)

            prebuilt_full[sym] = _dataframe_to_symbol_arrays(trimmed_sig)

        # 3. CAWF-R: Build K chronological AWF test-leg slices.
        embargo = int(EMBARGO_BARS.get(ctx.tf, 12))
        k_legs = int(OPT_FUTURES_CONFIG.get("FUTURES_AWF_K_LEGS", 5))
        min_train_frac = float(OPT_FUTURES_CONFIG.get("FUTURES_AWF_MIN_TRAIN_FRAC", 0.40))
        awf_legs = build_anchored_wf_legs(
            eff_len, k=k_legs, min_train_frac=min_train_frac, embargo=embargo
        )

        ctx.awf_leg_slices = []
        for _train_s, _train_e, test_s, test_e in awf_legs:
            aligned = _build_aligned_2d_from_prebuilt(prebuilt_full, ctx.symbols, test_s, test_e)
            ctx.awf_leg_slices.append({"leg_range": (test_s, test_e), "data": aligned})

        # Legacy CPCV paths preserved for downstream PBO-check compat only.
        cpcv_zone_len = max(200, int(eff_len * 0.80))
        cpcv_bundle = build_cpcv_test_paths_with_fallback(cpcv_zone_len, embargo=embargo)
        unique_blocks = list_cpcv_block_ranges(cpcv_zone_len, cpcv_bundle[1], embargo=embargo)
        ctx.cpcv_block_slices = []  # not used in AWF objective
        ctx.holdout_slice = None    # last AWF leg covers holdout zone
        ctx.multi_alignment_info["cpcv_paths"] = cpcv_bundle[0]
        ctx.multi_alignment_info["cpcv_all_block_ranges"] = unique_blocks
        ctx.multi_alignment_info["awf_legs"] = awf_legs

    # Diagnostic on AWF leg 0
    if ctx.awf_leg_slices:
        aligned0 = ctx.awf_leg_slices[0].get("data") or {}
        xl0 = aligned0.get("xs_score_long")
        if xl0 is not None and getattr(xl0, "size", 0) > 0:
            xs_std = float(np.nanstd(np.asarray(xl0, dtype=np.float64)))
            ctx.multi_alignment_info["xs_score_aligned_std"] = xs_std
            if xs_std < 0.05:
                _logger.warning(
                    "[ML_OPT] xs_score dispersion low (std=%.6f); check GP/CS merge path.",
                    xs_std,
                )

        _log_precompute_computed_dir_sample(
            aligned0,
            int(pre_ml["K_LONG"]),
            int(pre_ml["K_SHORT"]),
            float(pre_ml["CRISIS_GAMMA"]),
        )


def _log_precompute_computed_dir_sample(
    aligned0: dict[str, Any],
    k_long: int,
    k_short: int,
    crisis_gamma: float,
) -> None:
    """Single-bar sample of |computed_dir| dispersion (block 0, mid time index)."""
    xl = aligned0.get("xs_score_long")
    xs = aligned0.get("xs_score_short")
    hy = aligned0.get("hmm_prob_crisis")
    ml_l = aligned0.get("hmm_modulator_long")
    ml_s = aligned0.get("hmm_modulator_short")
    if (
        xl is None
        or xs is None
        or hy is None
        or ml_l is None
        or ml_s is None
        or getattr(xl, "size", 0) == 0
    ):
        return
    arr_l = np.ascontiguousarray(xl, dtype=np.float64)
    arr_s = np.ascontiguousarray(xs, dtype=np.float64)
    arr_h = np.ascontiguousarray(hy, dtype=np.float64)
    arr_ml_l = np.ascontiguousarray(ml_l, dtype=np.float64)
    arr_ml_s = np.ascontiguousarray(ml_s, dtype=np.float64)

    n_b, n_sy = arr_l.shape[0], arr_l.shape[1]
    prev_i = min(n_b - 1, max(0, n_b // 2))
    out = np.zeros(n_sy, dtype=np.float64)
    _recompute_cs_dirs_numba(
        prev_i,
        n_sy,
        arr_l,
        arr_s,
        arr_h,
        arr_ml_l,
        arr_ml_s,
        crisis_gamma,
        k_long,
        1.0,    # cs_z_threshold
        0.7,    # cs_z_exit_threshold
        out,
        np.zeros_like(out),
    )
    mags = np.abs(out)
    _logger.debug(
        "[ML_OPT][precompute] computed_dir |.| mid_bar=%d "
        "std=%.6f mean=%.6f max=%.6f nonzero_frac=%.4f",
        prev_i,
        float(np.nanstd(mags)),
        float(np.nanmean(mags)),
        float(np.nanmax(mags)),
        float(np.mean(mags > 1e-12)),
    )
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
    atr_choices_i = [28, 30, 32]
    atr_m_choices = [2.25, 2.5, 2.75]
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
        atr_p = _snap_int_list(int(deploy["ATR_PERIOD"]), atr_choices_i)

        atr_m = _snap_float_list(
            float(deploy.get("ATR_MULT", deploy.get("LONG_ATR_MULT", 2.5))), atr_m_choices
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
        pfk_win = _snap_int_list(int(deploy.get("PFK_WINDOW", 40)), pfk_choices)
        stress = _snap_float_list(float(deploy.get("STRESS_VOL_Z", 2.5)), stress_choices)
        rpt = float(deploy.get("RISK_PER_TRADE", fixed["RISK_PER_TRADE"]))
        min_score = float(deploy.get("MIN_SCORE_PERCENTILE", 0.55))
    except (KeyError, TypeError, ValueError):
        return None
    if rpt != float(fixed["RISK_PER_TRADE"]):
        return None
    out = apply_policy_constraints(
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
            "PFK_WINDOW": pfk_win,
            "STRESS_VOL_Z": stress,
            "RISK_PER_TRADE": rpt,
            "MIN_SCORE_PERCENTILE": min_score,
            "USE_CS_RANK_ENGINE": True,
        },
        policy,
    )
    return out


def _suggest_ml_phase_d(trial: optuna.Trial) -> dict[str, Any]:
    """SOTA: Narrow Optuna to Tail-Risk/Defense and Entry/Exit timing parameters.
    
    1. vol_target_annual (TARGET_ANN_VOL)
    2. hmm_crisis_penalty (CRISIS_GAMMA)
    3. atr_stop_loss_mult (ATR_MULT)
    4. rebalance_frequency (REBALANCE_BARS)
    5. entry_thresholds (CS_Z_SCORE_THRESHOLD, MIN_SCORE_PERCENTILE)
    """
    policy = load_portfolio_policy_config(OPT_FUTURES_CONFIG)

    # 1. Target Volatility (Annualized)
    ann_vol = float(trial.suggest_float("TARGET_ANN_VOL", 0.15, 0.45, step=0.05))

    # 2. HMM Crisis Penalty (Gamma)
    crisis_gamma = float(trial.suggest_float("CRISIS_GAMMA", 1.0, 5.0, step=0.5))

    # 3. ATR Stop Multiplier
    atr_m = float(trial.suggest_float("ATR_MULT", 3.0, 7.0, step=0.5))

    # 4. Rebalance Bars (Crucial for fee control)
    # Allowing smaller bars but Optuna must balance with higher thresholds
    reb = int(trial.suggest_categorical("REBALANCE_BARS", [1, 3, 6, 12, 24]))
    reb_turnover = float(trial.suggest_float("REBALANCE_TURNOVER_THRESHOLD", 0.05, 0.30, step=0.05))

    # 5. Entry Thresholds (Selective Entry)
    # Higher Z-score means fewer but higher quality trades, reducing churn
    # Realistic Optuna Bounds: Limit CS_Z search range to [0.5, 1.5]
    cs_z = float(trial.suggest_float("CS_Z_SCORE_THRESHOLD", 0.5, 1.5, step=0.25))
    min_p = float(trial.suggest_float("MIN_SCORE_PERCENTILE", 0.50, 0.90, step=0.10))

    # Risk per trade - adjust heuristic to be more robust
    # Using 1/N scaling with annual target vol
    rpt = ann_vol / math.sqrt(252)

    atr_p = 30
    trail_m = atr_m
    s_tp = 2.0
    l_scale = 3.0
    
    max_exp_sym = policy.per_symbol_cap
    max_exp_gross = policy.gross_exposure_cap
    
    sizing_m = "profit_factor_kelly"

    out = {
        "SIZING_METHOD": sizing_m,
        "TARGET_ANN_VOL": ann_vol,
        "CRISIS_GAMMA": crisis_gamma,
        "ATR_PERIOD": atr_p,
        "ATR_MULT": atr_m,
        "TRAIL_MULT": trail_m,
        "SHORT_TP_MULT": s_tp,
        "LONG_SCALE_ATR_MULT": l_scale,
        "MAX_EXPOSURE_PER_COIN": max_exp_sym,
        "MAX_EXPOSURE": max_exp_gross,
        "RISK_PER_TRADE": rpt,
        "REBALANCE_BARS": reb,
        "REBALANCE_TURNOVER_THRESHOLD": reb_turnover,
        "K_LONG": policy.top_k_long,
        "K_SHORT": policy.top_k_short,
        "DD_SCALING_THRESHOLD": 0.20,
        "CS_Z_SCORE_THRESHOLD": cs_z,
        "MIN_SCORE_PERCENTILE": min_p,
        "USE_CS_RANK_ENGINE": True,
    }
    return apply_policy_constraints(out, policy)


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
    # In SOTA mode, RISK_PER_TRADE is used as the Kelly Lambda (lam)
    # Target Annual Vol is used to scale position sizes.
    ann_vol = float(ml.get("TARGET_ANN_VOL", 0.20))
    lev = float(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5))

    return {
        "TIMEFRAME": tf,
        "SIGNAL_TYPE": "ML_CALIB_PROB",
        "REGIME_TYPE": "EMA_ATR",
        "SIZING_METHOD": "profit_factor_kelly",
        "USE_CS_RANK_ENGINE": True,
        "K_LONG": int(ml.get("K_LONG", 2)),
        "K_SHORT": int(ml.get("K_SHORT", 2)),
        "REBALANCE_BARS": int(ml.get("REBALANCE_BARS", 1)),
        "REBALANCE_TURNOVER_THRESHOLD": float(ml.get("REBALANCE_TURNOVER_THRESHOLD", 0.15)),
        "MIN_SCORE_PERCENTILE": float(ml.get("MIN_SCORE_PERCENTILE", 0.50)),
        "CRISIS_GAMMA": float(ml.get("CRISIS_GAMMA", 1.0)),
        "TRAIL_MULT": float(ml.get("TRAIL_MULT", 3.0)),
        "ATR_MULT": float(ml.get("ATR_MULT", 3.0)),
        "ATR_PERIOD": int(ml.get("ATR_PERIOD", 30)),
        "SHORT_TP_MULT": float(ml.get("SHORT_TP_MULT", 2.0)),
        "LONG_SCALE_ATR_MULT": float(ml.get("LONG_SCALE_ATR_MULT", 3.0)),
        "RISK_PER_TRADE": ann_vol,  # lambda
        "MAX_EXPOSURE_PER_COIN": float(ml.get("MAX_EXPOSURE_PER_COIN", 1.0)),
        "MAX_EXPOSURE": float(ml.get("MAX_EXPOSURE", 1.0)),
        "DD_SCALING_THRESHOLD": float(ml.get("DD_SCALING_THRESHOLD", 0.20)),
        "TARGET_ANN_VOL": ann_vol,
        "USE_COMPOUNDING": True,
        "LEVERAGE": int(lev),
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
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    strength_g = aligned["ml_calib_prob"]
    use_cs = 1 if bool(params.get("USE_CS_RANK_ENGINE", True)) else 0
    
    atr_m = float(params["ATR_MULT"])
    trail_m = float(params["TRAIL_MULT"])
    
    out_bt = backtest_portfolio_numba(
        aligned["close"],
        aligned["high"],
        aligned["low"],
        aligned["open"],
        aligned["entry_upper"],
        aligned["entry_lower"],
        aligned["trend_direction"],
        strength_g,
        aligned["atr"],
        aligned["garch_kelly_f"],
        zkill,
        zfund,
        aligned["slot_rank_score"],
        aligned["xs_score_long"],
        aligned["xs_score_short"],
        aligned["hmm_prob_crisis"],
        aligned["hmm_modulator_long"],
        aligned["hmm_modulator_short"],
        float(FUTURES_INITIAL_BALANCE),
        lev_blk,
        TRADING_FEE_RATE,
        SLIPPAGE_RATE,
        float(params["RISK_PER_TRADE"]),
        atr_m,
        trail_m,
        atr_m,
        float(params["SHORT_TP_MULT"]),
        float(params["LONG_SCALE_ATR_MULT"]),
        trail_m,
        int(params.get("MAX_CONCURRENT_POSITIONS", 2)),
        float(params.get("MAX_EXPOSURE", 0.8)),
        float(params["MAX_EXPOSURE_PER_COIN"]),
        float(params["DD_SCALING_THRESHOLD"]),
        int(params["K_LONG"]),
        int(params["K_SHORT"]),
        float(params.get("CS_Z_SCORE_THRESHOLD", 1.0)),
        max(1, int(params["REBALANCE_BARS"])),
        float(params.get("REBALANCE_TURNOVER_THRESHOLD", 0.15)),
        float(params.get("CRISIS_GAMMA", params.get("CRISIS_GATE_PROB", 1.0))),
        use_cs,
    )
    return cast(tuple[np.ndarray, float, np.ndarray, np.ndarray], out_bt)


def objective_ml_phase_d(trial: optuna.Trial, ctx: MLPhaseDContext) -> float | tuple[float, float]:
    """CAWF-R Phase-D: K AWF legs → validation-contract composite (minimize -reward + penalties)."""
    if ctx.awf_leg_slices is None:
        precompute_ml_optimization_context(ctx)
    awf_slices = ctx.awf_leg_slices
    mai = ctx.multi_alignment_info
    if not awf_slices or mai is None:
        nsga2 = bool(OPT_FUTURES_CONFIG.get("FUTURES_ML_ALPHA_NSGA2_ENABLED", False))
        return (1e9, 1e9) if nsga2 else 1e9

    ml = _suggest_ml_phase_d(trial)
    params = _base_engine_params(ml, ctx.tf)
    cfg = OPT_FUTURES_CONFIG

    if trial.number < 10 and awf_slices:
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
    min_trades_target = int(cfg.get("FUTURES_MIN_TRADES_TARGET", 30))
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
            params, aligned, zkill, zfund, lev_leg
        )

        n_tr = int(b_trades_raw.shape[0])
        all_trades_chunks.append(b_trades_raw)

        if not first_leg_done:
            first_leg_done = True
            if n_tr == 0:
                raise optuna.TrialPruned()

        mdd = float(calc_mdd_from_equity(b_equity)) if b_equity.size > 0 else 100.0
        log_ret = _log_tw_from_ret_pct(float((b_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0))

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

        if leg_idx >= 1:
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
                    # We still want to set attrs even if we prune, but Optuna handles Pruned differently
                    # For now, let's continue to set attrs and then raise
                    pass

            trial.report(float(np.mean(leg_log_tw)), step=leg_idx)
            # if trial.should_prune():
            #     break

    # --- AGGREGATE RESULTS ---
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

    weak_side_pf = float(min(l_pf_agg, s_pf_agg))
    dir_pf_score = float(np.log(max(weak_side_pf, 1.001)))

    import math as _math
    k_legs_n = float(max(n_legs_done, 1))
    mu_log = float(np.mean(leg_arr)) if leg_arr.size > 0 else -10.0
    sigma_log = float(np.std(leg_arr, ddof=1)) if leg_arr.size > 1 else 0.0
    
    # --- COST-AWARE OBJECTIVE ---
    est_rt_cost = 0.0020
    trade_cost_drag = (avg_trades_agg * est_rt_cost) / k_legs_n
    net_mu_log = mu_log - trade_cost_drag

    n_trials_cfg = float(cfg.get("total_trials", 400))
    lambda_def = float(cfg.get("FUTURES_PLGD_LAMBDA_DEF", 0.5))
    lambda_tail = float(cfg.get("FUTURES_PLGD_LAMBDA_TAIL", 2.0))

    variance_drag = 0.5 * sigma_log**2
    sr_bench = _math.sqrt(2.0 * _math.log(max(n_trials_cfg, 2.0)))
    deflation = lambda_def * sr_bench * sigma_log / _math.sqrt(max(k_legs_n, 1.0))
    worst_leg = float(np.min(leg_arr)) if leg_arr.size > 0 else -10.0

    awf_pos_frac = float(np.sum(leg_arr > 0.0)) / k_legs_n
    dsr_awf = float(min(0.99, max(0.0, awf_pos_frac)))

    w_pos = float(cfg.get("FUTURES_PHASE_D_W_POS_LEGS", 2.5))
    w_worst = float(cfg.get("FUTURES_PHASE_D_W_WORST_LEG_LOGTW", 2.0))
    w_pf = float(cfg.get("FUTURES_PHASE_D_W_DIR_PF", 1.0))
    w_is = float(cfg.get("FUTURES_PHASE_D_W_IS_ALPHA_LOG", 1.2))
    w_mu = float(cfg.get("FUTURES_PHASE_D_W_MU_LOG", 5.0))
    plgd_reg_w = float(cfg.get("FUTURES_PHASE_D_PLGD_REG_WEIGHT", 0.12))

    reward = (
        w_mu * net_mu_log
        + w_pos * awf_pos_frac
        + w_worst * worst_leg
        + w_pf * dir_pf_score
        + w_is * mu_log
    )
    plgd_reg = plgd_reg_w * (variance_drag + deflation)

    obj = -reward + plgd_reg
    if n_legs_done < int(cfg.get("FUTURES_AWF_K_LEGS", 5)):
        obj += 10.0 * (int(cfg.get("FUTURES_AWF_K_LEGS", 5)) - n_legs_done)
    if mu_log < -0.25:
        obj += 100.0

    # --- FINAL USER ATTRS ---
    trial.set_user_attr("cpcv_path_oos_log_tw", [float(x) for x in leg_log_tw])
    trial.set_user_attr("awf_pos_frac", awf_pos_frac)
    trial.set_user_attr("gate1_dsr", dsr_awf)
    trial.set_user_attr("n_negative_legs", int(np.sum(leg_arr < 0.0)))
    trial.set_user_attr("leg_l_pf", [round(x, 4) for x in leg_l_pf])
    trial.set_user_attr("leg_s_pf", [round(x, 4) for x in leg_s_pf])
    trial.set_user_attr("leg_long_counts", leg_long_counts)
    trial.set_user_attr("leg_short_counts", leg_short_counts)
    trial.set_user_attr("ml_mean_log_growth_cpcv", mu_log)
    trial.set_user_attr("ml_p10_log_growth_cpcv", worst_leg)
    trial.set_user_attr("ml_worst_mdd_cpcv", worst_mdd_legs)
    trial.set_user_attr("avg_trades", avg_trades_agg)
    trial.set_user_attr("avg_exposure", avg_exposure)
    trial.set_user_attr("long_short_ratio", minority)
    trial.set_user_attr("leg_crisis_mean", [round(x, 4) for x in leg_crisis_mean])
    trial.set_user_attr("n_valid_paths", int(n_legs_done))

    return float(obj)


def select_best_trial_by_holdout_log_ret(trials: list[optuna.trial.FrozenTrial]) -> optuna.trial.FrozenTrial:
    if not trials:
        raise ValueError("empty trials")

    def _score(t: optuna.trial.FrozenTrial) -> tuple[float, float, float, float, float, float]:
        holdout = float(np.clip(t.user_attrs.get("ml_holdout_log_ret", 0.0), -2.0, 2.0))
        is_cpcv = float(np.clip(t.user_attrs.get("ml_mean_log_growth_cpcv", -2.0), -2.0, 2.0))
        p10_cpcv = float(np.clip(t.user_attrs.get("ml_p10_log_growth_cpcv", -2.0), -2.0, 2.0))
        dsr = float(t.user_attrs.get("gate1_dsr", 0.0))
        worst_mdd = float(t.user_attrs.get("ml_worst_mdd_cpcv", 999.0))
        path_std = float(np.clip(t.user_attrs.get("ml_std_log_growth_cpcv", 1.0), 0.0, 2.0))
        if is_cpcv < 0:
            holdout = holdout - abs(is_cpcv) * 2.0
        return (dsr, is_cpcv, p10_cpcv, -path_std, -worst_mdd, holdout)

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
    ideal: np.ndarray = np.zeros(n_dim, dtype=np.float64)
    nadir: np.ndarray = np.ones(n_dim, dtype=np.float64)
    d_pos = np.linalg.norm(norm - ideal, axis=1)
    d_neg = np.linalg.norm(norm - nadir, axis=1)
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



