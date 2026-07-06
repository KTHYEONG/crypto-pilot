from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

_logger = logging.getLogger(__name__)

# Cross-sectional momentum lookback periods precomputed for all trials.
_CS_MOM_LOOKBACKS: list[int] = [12, 24, 36, 48, 60, 72]


class SignalCalibrator:
    """Deterministic Platt-style calibrator for alpha scores."""

    def __init__(self) -> None:
        self.mean_b: float = 1.05
        self._coef: float = 1.0
        self._intercept: float = 0.0

    def fit(self, alpha: np.ndarray, returns: np.ndarray) -> SignalCalibrator:
        """Fit a simple logistic mapping from alpha to positive-return probability."""
        x = np.asarray(alpha, dtype=np.float64).reshape(-1)
        y_raw = np.asarray(returns, dtype=np.float64).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y_raw)
        if not np.any(mask):
            self.mean_b = 1.05
            self._coef = 1.0
            self._intercept = 0.0
            return self

        x = x[mask]
        y = (y_raw[mask] > 0.0).astype(np.float64)
        x_centered = x - float(np.mean(x))
        design = np.column_stack([x_centered, np.ones_like(x_centered)])
        ridge = np.eye(2, dtype=np.float64) * 1e-6
        ridge[1, 1] = 0.0
        beta, *_ = np.linalg.lstsq(design.T @ design + ridge, design.T @ y, rcond=None)
        coef = float(beta[0])
        intercept = float(beta[1])
        if not np.isfinite(coef):
            coef = 1.0
        if not np.isfinite(intercept):
            intercept = 0.0
        self._coef = float(np.clip(coef, -50.0, 50.0))
        self._intercept = float(np.clip(intercept, -50.0, 50.0))
        self.mean_b = float(max(0.1, abs(self._coef)))
        return self

    def predict_prob(self, alpha: np.ndarray) -> np.ndarray:
        """Return calibrated probability scores."""
        x = np.asarray(alpha, dtype=np.float64)
        logits = np.clip(self._coef * x + self._intercept, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))


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
        if df is not None and not df.empty and "datetime" in df.columns:
            # [Fix] Use datetime as index to ensure correct cross-sectional alignment
            ser = df.set_index("datetime")["close"]
            rets_series[sym] = ser

    if len(rets_series) < 2:
        return

    for lb in lookbacks:
        # pd.DataFrame(rets_series) will now correctly align by datetime index
        all_rets = {s: r.pct_change(periods=lb) for s, r in rets_series.items()}
        ranks_df = pd.DataFrame(all_rets).rank(axis=1, pct=True)
        for sym in rets_series:
            if sym in ranks_df.columns:
                col_name = f"cs_mom_rank_{lb}"
                # Map back to original dataframe using datetime alignment
                target_df = data_maps[sym][tf]
                ranks_ser = ranks_df[sym]
                # Reindex to match target_df datetime and convert to numpy for fast assignment
                data_maps[sym][tf][col_name] = ranks_ser.reindex(target_df["datetime"]).to_numpy(dtype=np.float64)


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


def resolve_embargo_bars_for_tf(cfg: dict[str, Any], tf: str, longest_indicator_period: int = 150) -> int:
    """Prefer EMBARGO_BARS_BY_TF; fallback to horizon ratio heuristic."""
    by_tf = cfg.get("EMBARGO_BARS_BY_TF")
    if isinstance(by_tf, dict) and tf in by_tf:
        return max(0, int(by_tf[str(tf)]))
    fixed_min: dict[str, int] = {"1h": 24}
    ratio_map: dict[str, float] = {"1h": 0.08}
    ratio: float = ratio_map.get(tf, 0.05)
    return max(fixed_min.get(tf, 12), int(longest_indicator_period * ratio))


EMBARGO_BARS: dict[str, int] = {tf: resolve_embargo_bars_for_tf(OPT_FUTURES_CONFIG, tf) for tf in ("1h", "4h", "1d")}
