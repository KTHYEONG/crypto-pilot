"""Multi-layer champion checks (AWF-centric + lightweight OOS PSR)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from config.settings import SLIPPAGE_RATE, TAKER_FEE_RATE
except ImportError:  # pragma: no cover — tests/unit without full pkg path
    TAKER_FEE_RATE = 0.0005  # type: ignore[misc]
    SLIPPAGE_RATE = 2e-4  # type: ignore[misc]

try:
    from scipy.stats import kurtosis, norm, skew
except ImportError:  # pragma: no cover
    norm = None  # type: ignore[assignment]
    skew = None  # type: ignore[assignment]
    kurtosis = None  # type: ignore[assignment]


def probabilistic_sharpe_ratio_observed(
    sharpe_ann: float, n: int, skewness: float, kurtosis_excess: float
) -> float:
    """PSR(SR*) vs 0 adjustment (Probabilistic Sharpe Ratio). Returns in [0,1]."""
    if norm is None or n < 4:
        return 0.5
    denom = (
        1.0 - skewness * sharpe_ann + (kurtosis_excess / 4.0) * sharpe_ann**2
    )
    denom = math.sqrt(max(1e-12, denom))
    z_stat = sharpe_ann * math.sqrt(max(n - 1, 1)) / denom
    return float(norm.cdf(z_stat))


def friction_round_trip_frac_for_tmp_layer2(cfg: dict[str, Any]) -> float:
    """Rough round-trip cost fraction aligned with futures opt settings.

    Specification (Layer 2 friction stress): ``2 * TAKER_FEE + SLIPPAGE * BUFFER`` where
    ``BUFFER`` is ``SLIPPAGE_BPS_BUFFER_MULT`` from *cfg* (same key as OPT_FUTURES_CONFIG).

    This is a one-way-ish round-trip shorthand (two taker legs + buffer-scaled slippage term),
    not a full maker/taker lifecycle model.
    """
    buf = float(cfg.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0))
    return float(2.0 * float(TAKER_FEE_RATE) + float(SLIPPAGE_RATE) * buf)


def median_log_tw_under_friction_stress(
    log_tw_legs: list[float] | np.ndarray,
    stress_mult: float = 1.5,
    *,
    round_trip_cost_frac: float | None = None,
    cfg_for_friction: dict[str, Any] | None = None,
) -> float:
    """Median leg log total weight after a simple friction stress (Layer 2 gate helper).

    Approximation (**additive on log_TW**, consistent with dominant cost level for typical
    futures fee/slip): ``log_tw_eff_i = log_tw_i − stress_mult * r_rt`` with ``r_rt`` from
    :func:`friction_round_trip_frac_for_tmp_layer2`.

    When *round_trip_cost_frac* is omitted, uses *cfg_for_friction* if given, otherwise
    ``{"SLIPPAGE_BPS_BUFFER_MULT": 1.0}``.
    """
    rt = round_trip_cost_frac
    if rt is None:
        c = cfg_for_friction if cfg_for_friction is not None else {"SLIPPAGE_BPS_BUFFER_MULT": 1.0}
        rt = friction_round_trip_frac_for_tmp_layer2(c)
    penalty = float(stress_mult) * float(rt)
    arr = np.asarray(log_tw_legs, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("-inf")
    eff = arr - penalty
    return float(np.median(eff))


def tmp_md_layer1_gate_failures(
    user_attrs: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    raw = (
        user_attrs.get("awf_path_leg_log_tw")
        or user_attrs.get("awf_leg_log_tw")
        or user_attrs.get("cpcv_path_oos_log_tw")
        or []
    )
    logs = [float(x) for x in raw]
    if not logs:
        # scalar-only 저장 정책으로 list가 없을 때: 저장된 scalar로 동일 체크 수행
        awf_pos = float(user_attrs.get("awf_pos_frac", -1.0))
        if awf_pos < 0.0:
            failures.append("TMP_LAYER1_NO_AWF_LEGS")
            return failures
        thr = float(cfg.get("FUTURES_TMP_LAYER1_POS_LEG_RATIO_MIN", 4.0 / 6.0))
        if awf_pos + 1e-12 < thr:
            failures.append("TMP_LAYER1_POS_LEG_RATIO")
        awf_mu = float(user_attrs.get("awf_mu_log", -999.0))
        if awf_mu <= float(cfg.get("FUTURES_TMP_LAYER1_MEDIAN_LOG_TW_MIN", 0.0)):
            failures.append("TMP_LAYER1_MEDIAN_LOG_TW")
        mdd_pct = float(
            user_attrs.get("awf_worst_mdd_pct", user_attrs.get("ml_worst_mdd_cpcv", 0.0))
        )
        if mdd_pct > float(cfg.get("FUTURES_TMP_LAYER1_MAX_DD_PCT", 12.0)):
            failures.append("TMP_LAYER1_MAX_DD")
        return failures
    arr = np.asarray(logs, dtype=np.float64)
    med = float(np.median(arr))
    if med <= float(cfg.get("FUTURES_TMP_LAYER1_MEDIAN_LOG_TW_MIN", 0.0)):
        failures.append("TMP_LAYER1_MEDIAN_LOG_TW")
    pos_frac = float(np.mean(arr > 0.0))
    thr = float(cfg.get("FUTURES_TMP_LAYER1_POS_LEG_RATIO_MIN", 4.0 / 6.0))
    if pos_frac + 1e-12 < thr:
        failures.append("TMP_LAYER1_POS_LEG_RATIO")
    mdd_pct = float(
        user_attrs.get("awf_worst_mdd_pct", user_attrs.get("ml_worst_mdd_cpcv", 0.0))
    )
    if mdd_pct > float(cfg.get("FUTURES_TMP_LAYER1_MAX_DD_PCT", 12.0)):
        failures.append("TMP_LAYER1_MAX_DD")
    return failures


def tmp_md_layer2_gate_failures(
    user_attrs: dict[str, Any],
    *,
    oos_bar_rets: np.ndarray,
    ann_factor: float,
    cfg: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    floor_psr = float(cfg.get("FUTURES_TMP_LAYER2_PSR_MIN", 0.95))
    r = np.asarray(oos_bar_rets, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size >= 8:
        sigma = float(np.std(r, ddof=1))
        mu_br = float(np.mean(r))
        if sigma > 1e-12 and skew is not None and kurtosis is not None:
            sharpe_ann = mu_br / sigma * math.sqrt(max(ann_factor, 1e-9))
            sk = float(skew(r, bias=False)) if r.size >= 6 else 0.0
            kt = (
                float(kurtosis(r, fisher=True, bias=False)) if r.size >= 8 else 0.0
            )
            psr = probabilistic_sharpe_ratio_observed(
                sharpe_ann, int(r.size), sk, kt
            )
            if psr + 1e-12 < floor_psr:
                failures.append("TMP_LAYER2_PSR")
    min_trades = int(cfg.get("FUTURES_TMP_LAYER2_MIN_TRADES_PER_LEG", 30))
    leg_counts = user_attrs.get("awf_leg_trade_counts")
    if leg_counts is not None and len(leg_counts) > 0:
        if min(float(x) for x in leg_counts) + 1e-9 < float(min_trades):
            failures.append("TMP_LAYER2_LEG_TRADES")

    if bool(cfg.get("FUTURES_TMP_LAYER2_FRICTION_STRESS_ENABLED", False)):
        raw = (
            user_attrs.get("awf_path_leg_log_tw")
            or user_attrs.get("awf_leg_log_tw")
            or user_attrs.get("cpcv_path_oos_log_tw")
            or []
        )
        logs = [float(x) for x in raw]
        if logs:
            stress = float(cfg.get("FUTURES_TMP_LAYER2_FRICTION_STRESS_MULT", 1.5))
            rt = friction_round_trip_frac_for_tmp_layer2(cfg)
            med_eff = median_log_tw_under_friction_stress(
                logs, stress, round_trip_cost_frac=rt
            )
            if med_eff <= 0.0:
                failures.append("TMP_LAYER2_FRICTION_STRESS")

    return failures


def tmp_md_layer1_failures_from_awf_diag(diag: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    if diag.get("empty") or diag.get("pruned"):
        return ["TMP_LAYER1_NO_AWF_DIAG"]
    ua = {
        "awf_path_leg_log_tw": diag.get("leg_log_tw"),
        "awf_worst_mdd_pct": float(diag.get("worst_mdd_legs", 0.0)),
    }
    return tmp_md_layer1_gate_failures(ua, cfg)


def collect_tmp_md_champion_gate_failures(
    user_attrs: dict[str, Any],
    *,
    oos_bar_rets: np.ndarray,
    ann_factor: float,
    cfg: dict[str, Any],
) -> list[str]:
    out = tmp_md_layer1_gate_failures(user_attrs, cfg)
    out.extend(
        tmp_md_layer2_gate_failures(
            user_attrs, oos_bar_rets=oos_bar_rets, ann_factor=ann_factor, cfg=cfg
        )
    )
    return out
