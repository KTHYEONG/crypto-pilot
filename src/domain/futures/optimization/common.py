from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression

import pandas as pd
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

_logger = logging.getLogger(__name__)

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


def _array_stats(name: str, arr: Any) -> str:
    """Return compact finite/non-zero stats for debug logging."""
    if arr is None:
        return f"{name}=None"
    np_arr = np.asarray(arr, dtype=np.float64)
    if np_arr.size == 0:
        return f"{name}=empty"
    finite_mask = np.isfinite(np_arr)
    finite = np_arr[finite_mask]
    nnz = int(np.count_nonzero(np.abs(np_arr) > 1e-12))
    if finite.size == 0:
        return f"{name}=size:{np_arr.size} finite:0 nnz:{nnz}"
    return (
        f"{name}=size:{np_arr.size} finite:{finite.size} nnz:{nnz} "
        f"mean:{float(np.mean(finite)):.6g} std:{float(np.std(finite)):.6g} "
        f"min:{float(np.min(finite)):.6g} max:{float(np.max(finite)):.6g}"
    )


def _diag_to_dict(diag: Any) -> dict[str, float]:
    """Convert Numba diag array to named dictionary when possible."""
    arr = np.asarray(diag)
    if arr.size < 5:
        return {"diag_size": float(arr.size)}
    # [dust_skip_cnt, margin_fail_cnt, spare_a, spare_b, mode_flag]
    return {
        "dust_skip_cnt": float(arr[0]),
        "margin_fail_cnt": float(arr[1]),
        "spare_a": float(arr[2]),
        "spare_b": float(arr[3]),
        "mode_flag": float(arr[4]),
    }


def _safe_pct(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.nanpercentile(finite, q))


def _nonzero_ratio(arr: np.ndarray, eps: float = 1e-12) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(arr) > eps) / arr.size)


def _trial_diag_sampled(trial: optuna.Trial | None, *, n_trades: int) -> bool:
    if n_trades == 0:
        return True
    if trial is None:
        return False
    try:
        return int(trial.number) < 5
    except Exception:
        return False


def _weight_stage_diag(
    target_weights: np.ndarray | None,
    *,
    per_symbol_cap: float | None,
) -> dict[str, float | str]:
    if target_weights is None:
        return {
            "tw_row_nz_ratio": 0.0,
            "gross_mean": 0.0,
            "gross_p95": 0.0,
            "cap_hit_proxy_ratio": "not_available",
        }
    tw = np.asarray(target_weights, dtype=np.float64)
    if tw.ndim != 2 or tw.size == 0:
        return {
            "tw_row_nz_ratio": 0.0,
            "gross_mean": 0.0,
            "gross_p95": 0.0,
            "cap_hit_proxy_ratio": "not_available",
        }
    row_nz = np.any(np.abs(tw) > 1e-12, axis=1)
    gross = np.sum(np.abs(tw), axis=1)
    cap_hit_proxy: float | str = "not_available"
    if per_symbol_cap is not None and float(per_symbol_cap) > 0.0:
        cap = float(per_symbol_cap)
        active = np.abs(tw) > 1e-12
        hits = np.abs(tw) >= (0.99 * cap)
        den = int(np.count_nonzero(active))
        cap_hit_proxy = float(np.count_nonzero(hits & active) / max(den, 1))
    return {
        "tw_row_nz_ratio": float(np.mean(row_nz)) if row_nz.size > 0 else 0.0,
        "gross_mean": float(np.mean(gross)) if gross.size > 0 else 0.0,
        "gross_p95": _safe_pct(gross, 95) if gross.size > 0 else 0.0,
        "cap_hit_proxy_ratio": cap_hit_proxy,
    }


def _safe_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def resolve_embargo_bars_for_tf(
    cfg: dict[str, Any], tf: str, longest_indicator_period: int = 150
) -> int:
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
        """Initialize the signal calibrator with a logistic regression model."""
        self.model = LogisticRegression(penalty=None, solver="lbfgs")
        self.is_fitted = False
        self.mean_b = 1.05  # Estimated win/loss ratio (Profit Factor)

    def fit(self, alphas: np.ndarray, returns: np.ndarray) -> None:
        """Fit calibration model and estimate average win/loss ratio."""
        # y=1 if forward return is positive
        y = (returns > 0.0001).astype(int)
        x_input = alphas.reshape(-1, 1)

        if len(np.unique(y)) > 1:
            try:
                self.model.fit(x_input, y)
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
        x_input = alphas.reshape(-1, 1)
        return self.model.predict_proba(x_input)[:, 1]


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
