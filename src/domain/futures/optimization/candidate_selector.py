from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import optuna

from config.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.optimization.dashboard import safe_float
from src.domain.futures.optimization.optimizer import (
    MLPhaseDContext,
    replay_robust_awf_for_trial_params,
)

_logger: logging.Logger = logging.getLogger("candidate_selector")


def sanitize_metric_map(m: dict[str, Any]) -> dict[str, float]:
    """Clamp metrics to reasonable bounds for stability in comparisons."""
    limits = {
        "pbo": 1e3,
        "p10": 100.0,
        "dsr": 1e3,
        "tw": 1e6,
        "cagr": 1e5,
        "mdd": 1e3,
        "time_2x": 1e6,
        "cvar": 1e3,
        "net_alpha": 1e5,
        "avg_pnl": 1e5,
        "pf": 1e3,
        "is_cagr": 1e5,
        "ho_cagr": 1e5,
        "awf_pos_frac": 10.0,
        "mu_awf": 100.0,
        "sig_awf": 100.0,
        "plgd": 100.0,
        "erg_dev": 1e4,
        "oos_long_pf": 1e3,
        "oos_short_pf": 1e3,
        "oos_win_rate": 100.0,
        "oos_pf": 1e3,
        "oos_cagr": 1e5,
        "oos_mdd": 1e3,
        "oos_sharpe": 100.0,
        "oos_ulcer": 100.0,
        "awf_worst_leg_log_tw": 100.0,
        "awf_worst_mdd_pct": 100.0,
        "awf_robust_score": 100.0,
        "awf_trade_count_mean": 1e5,
        "awf_long_pf_mean": 1e3,
        "awf_short_pf_mean": 1e3,
        "awf_leg_worst_log_tw": 100.0,
        "awf_leg_pos_ratio": 10.0,
        "awf_chop_loss_share": 10.0,
        "awf_chop_trade_share": 10.0,
        "awf_flip_rate_proxy": 10.0,
        "awf_turnover_cost_ratio": 1e3,
    }
    out: dict[str, float] = {}
    for k, v in m.items():
        out[k] = safe_float(v, default=0.0, clip=limits.get(k, 1e6))
    return out


def list_mean(vals: Any, default: float) -> float:
    """Safely compute mean of a list or return default."""
    if not isinstance(vals, (list, np.ndarray)) or len(vals) == 0:
        return default
    return float(np.mean(vals))


def cand_metric(cand: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Retrieve a metric from a candidate dictionary, checking user_attrs and diag."""
    tr = cand.get("trial")
    ua = tr.user_attrs if tr is not None else {}
    diag = cand.get("awf_diag", {}) or {}
    if key == "awf_worst_leg_log_tw":
        return safe_float(
            ua.get(
                "awf_worst_leg_log_tw",
                ua.get("ml_p10_log_growth_cpcv", diag.get("awf_worst_leg_log_tw", default)),
            ),
            default,
        )
    if key == "awf_worst_mdd_pct":
        return safe_float(
            ua.get(
                "awf_worst_mdd_pct",
                ua.get("ml_worst_mdd_cpcv", diag.get("awf_worst_mdd_pct", default)),
            ),
            default,
        )
    if key == "awf_trade_count_mean":
        return safe_float(
            ua.get(
                "awf_trade_count_mean",
                ua.get(
                    "avg_trades",
                    diag.get("awf_trade_count_mean", diag.get("avg_trades", default)),
                ),
            ),
            default,
        )
    if key == "awf_long_pf_mean":
        long_pf = ua.get("awf_long_pf_mean", diag.get("awf_long_pf_mean"))
        if long_pf is not None:
            return max(0.0, safe_float(long_pf, default))
        fallback_l_pf = list_mean(diag.get("leg_l_pf", []), default)
        return max(0.0, safe_float(diag.get("l_pf_agg", fallback_l_pf), default))
    if key == "awf_short_pf_mean":
        short_pf = ua.get("awf_short_pf_mean", diag.get("awf_short_pf_mean"))
        if short_pf is not None:
            return max(0.0, safe_float(short_pf, default))
        fallback_s_pf = list_mean(diag.get("leg_s_pf", []), default)
        return max(0.0, safe_float(diag.get("s_pf_agg", fallback_s_pf), default))
    return safe_float(ua.get(key, diag.get(key, default)), default)


def objective_key(cand: dict[str, Any]) -> tuple[float, float, float, float, int]:
    """Sort key for candidates based on Optuna objectives and robustness."""
    tr = cand["trial"]
    vals = list(tr.values or [])
    v0 = (
        safe_float(vals[0], 1e9)
        if len(vals) >= 1
        else -cand_metric(cand, "awf_robust_score", -1e9)
    )
    v1 = (
        safe_float(vals[1], 1e9)
        if len(vals) >= 2
        else -cand_metric(cand, "awf_leg_worst_log_tw", -1e9)
    )
    robust = cand_metric(cand, "awf_robust_score", -1e9)
    worst = cand_metric(cand, "awf_leg_worst_log_tw", -1e9)
    return (v0, v1, -robust, -worst, -int(tr.number))


def bounded_center_score(x: float, center: float, scale: float) -> float:
    """Compute a score (0 to 1) based on distance from a threshold/center."""
    if scale <= 0:
        return 1.0 if x >= center else 0.0
    return float(np.clip((x - center) / scale + 0.5, 0.0, 1.0))


def deploy_score(cand: dict[str, Any]) -> float:
    """Unified score for deployment suitability across multiple metrics."""
    cfg = dict(OPT_FUTURES_CONFIG)
    step2_enabled = bool(cfg.get("FUTURES_CHOP_REGIME_GATE_ENABLED", False))
    
    robust = cand_metric(cand, "awf_robust_score", -1.0)
    worst_leg_sc = cand_metric(cand, "awf_leg_worst_log_tw", -0.5)
    pos_ratio = cand_metric(cand, "awf_leg_pos_ratio", 0.0)
    trades = cand_metric(cand, "awf_trade_count_mean", 0.0)
    long_pf = cand_metric(cand, "awf_long_pf_mean", 1.0)
    short_pf = cand_metric(cand, "awf_short_pf_mean", 1.0)
    disp = cand_metric(cand, "awf_leg_dispersion", 0.5)

    deploy_min_robust = float(cfg.get("FUTURES_DEPLOY_MIN_ROBUST", 0.10))
    deploy_min_worst_leg = float(cfg.get("FUTURES_DEPLOY_MIN_WORST_LEG", -0.05))
    deploy_min_pos_ratio = float(cfg.get("FUTURES_DEPLOY_MIN_POS_RATIO", 0.35))
    deploy_min_trades = float(cfg.get("FUTURES_DEPLOY_MIN_TRADES", 20))
    deploy_min_side_pf = float(cfg.get("FUTURES_DEPLOY_MIN_SIDE_PF", 1.05))
    deploy_leg_disp_ref = float(cfg.get("FUTURES_DEPLOY_LEG_DISP_REF", 0.25))

    score = (
        0.30 * bounded_center_score(robust, deploy_min_robust, 0.10)
        + 0.16 * bounded_center_score(worst_leg_sc, deploy_min_worst_leg, 0.08)
        + 0.14 * bounded_center_score(pos_ratio, deploy_min_pos_ratio, 0.08)
        + 0.10 * bounded_center_score(trades, deploy_min_trades, max(deploy_min_trades * 0.4, 5.0))
        + 0.07 * bounded_center_score(long_pf, deploy_min_side_pf, 0.20)
        + 0.07 * bounded_center_score(short_pf, deploy_min_side_pf, 0.20)
        + 0.06 * bounded_center_score(-disp, -deploy_leg_disp_ref, max(deploy_leg_disp_ref, 0.05))
    )
    if step2_enabled:
        chop_loss_share = float(np.clip(cand_metric(cand, "awf_chop_loss_share", 0.0), 0.0, 1.0))
        chop_trade_share = float(np.clip(cand_metric(cand, "awf_chop_trade_share", 0.0), 0.0, 1.0))
        score += (
            0.04 * (1.0 - chop_loss_share)
            + 0.03 * (1.0 - chop_trade_share)
            + 0.03 * (1.0 - float(np.clip(cand_metric(cand, "awf_flip_rate_proxy", 0.0), 0.0, 1.0)))
        )
    return score


def deploy_reject_reasons(cand: dict[str, Any]) -> list[str]:
    """Check why a candidate failed deployment gates."""
    cfg = dict(OPT_FUTURES_CONFIG)
    step2_enabled = bool(cfg.get("FUTURES_CHOP_REGIME_GATE_ENABLED", False))
    step4_enabled = bool(cfg.get("FUTURES_AWF_POST_OPT_GATE_STEP4_ENABLED", False))
    
    robust = cand_metric(cand, "awf_robust_score", -1.0)
    worst_leg = cand_metric(cand, "awf_leg_worst_log_tw", -0.5)
    pos_ratio = cand_metric(cand, "awf_leg_pos_ratio", 0.0)
    trades = cand_metric(cand, "awf_trade_count_mean", 0.0)
    long_pf = cand_metric(cand, "awf_long_pf_mean", 1.0)
    short_pf = cand_metric(cand, "awf_short_pf_mean", 1.0)

    deploy_min_robust = float(cfg.get("FUTURES_DEPLOY_MIN_ROBUST", 0.05))
    deploy_min_worst_leg = float(cfg.get("FUTURES_DEPLOY_MIN_WORST_LEG", -0.12))
    deploy_min_pos_ratio = float(cfg.get("FUTURES_DEPLOY_MIN_POS_RATIO", 0.30))
    deploy_min_trades = float(cfg.get("FUTURES_DEPLOY_MIN_TRADES", 15))
    deploy_min_side_pf = float(cfg.get("FUTURES_DEPLOY_MIN_SIDE_PF", 1.01))

    reasons = []
    if robust < deploy_min_robust:
        reasons.append("LOW_ROBUST_SCORE")
    if worst_leg < deploy_min_worst_leg:
        reasons.append("CRITICAL_LEG_LOSS")
    if pos_ratio < deploy_min_pos_ratio:
        reasons.append("LOW_POS_RATIO")
    if trades < deploy_min_trades:
        reasons.append("LOW_TRADE_COUNT")
    if long_pf < deploy_min_side_pf and short_pf < deploy_min_side_pf:
        reasons.append("WEAK_BOTH_SIDE_PF")
    if step2_enabled:
        chop_loss_share = float(np.clip(cand_metric(cand, "awf_chop_loss_share", 0.0), 0.0, 1.0))
        chop_trade_share = float(np.clip(cand_metric(cand, "awf_chop_trade_share", 0.0), 0.0, 1.0))
        step2_chop_loss_max = float(cfg.get("FUTURES_CHOP_LOSS_SHARE_MAX", 0.45))
        step2_chop_trade_max = float(cfg.get("FUTURES_CHOP_TRADE_SHARE_MAX", 0.50))
        if chop_loss_share > step2_chop_loss_max:
            reasons.append("HIGH_CHOP_LOSS_SHARE")
        if chop_trade_share > step2_chop_trade_max:
            reasons.append("HIGH_CHOP_TRADE_SHARE")
    if step4_enabled:
        ua = cand["trial"].user_attrs
        diag = cand.get("awf_diag", {}) or {}
        chop_trade_share = float(np.clip(cand_metric(cand, "awf_chop_trade_share", 0.0), 0.0, 1.0))
        turnover_cost_ratio = max(0.0, cand_metric(cand, "awf_turnover_cost_ratio", 0.0))
        chop_pf_raw = ua.get("awf_chop_pf", diag.get("awf_chop_pf"))
        chop_pf = safe_float(chop_pf_raw, 0.0) if chop_pf_raw is not None else 1.0
        
        step4_chop_trade_max = float(cfg.get("FUTURES_AWF_CHOP_TRADE_SHARE_MAX", 0.35))
        step4_turnover_cost_max = float(cfg.get("FUTURES_AWF_TURNOVER_COST_RATIO_MAX", 0.12))
        step4_chop_pf_min = float(cfg.get("FUTURES_AWF_CHOP_PF_MIN", 0.90))
        
        if chop_trade_share > step4_chop_trade_max:
            reasons.append("STEP4_HIGH_CHOP_TRADE")
        if turnover_cost_ratio > step4_turnover_cost_max:
            reasons.append("STEP4_HIGH_TURNOVER_COST")
        if chop_pf < step4_chop_pf_min:
            reasons.append("STEP4_LOW_CHOP_PF")
    return reasons


def clear_stability_runtime_cache(sctx: MLPhaseDContext) -> None:
    """Clear Numba-cached signals from context to force re-evaluation in stability tests."""
    awf = sctx.awf_leg_slices or []
    for _leg in awf:
        _aligned = _leg.get("data")
        if isinstance(_aligned, dict):
            _aligned.pop("kill_signal_cached", None)
            _aligned.pop("funding_cached", None)
            _aligned.pop("lev_cached", None)


def release_stability_ctx(sctx: MLPhaseDContext) -> None:
    """Release memory held by a stability context."""
    sctx.data_maps = {}
    sctx.symbols = []
    sctx.awf_leg_slices = []
    gc.collect()


def replay_stability_candidate(
    params: dict[str, Any],
    tf: str,
    base_ctx: MLPhaseDContext,
    seed: int,
) -> tuple[float, bool]:
    """Replay a candidate with a specific seed to check stability (Layer 3)."""
    from src.domain.futures.optimization.optimizer import rerun_precompute_for_ctx
    from src.domain.futures.validation.tmp_md_champion import tmp_md_layer1_failures_from_awf_diag
    
    sctx = rerun_precompute_for_ctx(base_ctx, seed)
    try:
        clear_stability_runtime_cache(sctx)
        obj, diag = replay_robust_awf_for_trial_params(sctx, params)
        l1_fail = bool(tmp_md_layer1_failures_from_awf_diag(diag))
        return float(obj), l1_fail
    finally:
        release_stability_ctx(sctx)


def stability_cv(label: str, objs: list[float], l3_fail: bool) -> float | None:
    """Compute Coefficient of Variation for stability objectives."""
    if not objs:
        return None
    arr = np.asarray(objs, dtype=np.float64)
    mu = float(np.mean(arr))
    sig = float(np.std(arr))
    cv = sig / abs(mu) if abs(mu) > 1e-9 else 0.0
    _logger.info(
        " [STABILITY] %s: mu=%.4f sig=%.4f cv=%.4f l3_fail=%s",
        label, mu, sig, cv, l3_fail
    )
    return cv


def select_and_rank_candidates(
    study_ml: optuna.Study,
    base_ctx: MLPhaseDContext,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Step 4 Refinement: Augment Pareto trials, filter, and rank candidates.

    Returns:
        tuple[dict, dict]: (best_cand, selection_summary)

    """
    all_trials = study_ml.get_trials()
    pareto_trials = list(study_ml.best_trials)
    if not pareto_trials:
        return {}, {}

    from optuna.trial import TrialState

    from src.domain.futures.optimization.validation import awf_pos_frac_to_pseudo_pbo

    _study_complete = [t for t in all_trials if t.state == TrialState.COMPLETE]
    _k_aug = int(cfg.get("FUTURES_AWF_AUGMENT_TOPK", 5))
    _top_aug = sorted(
        _study_complete,
        key=lambda t: (float(t.user_attrs.get("awf_pos_frac", 0.0)),
                       float(t.user_attrs.get("awf_mu_log", -9.0))),
        reverse=True
    )[:_k_aug]
    _pareto_nums = {t.number for t in pareto_trials}
    for _t in _top_aug:
        if _t.number not in _pareto_nums:
            pareto_trials.append(_t)
            _pareto_nums.add(_t.number)

    validated_candidates = []
    pbo_limit = float(cfg.get("FUTURES_PBO_MAX", 0.45))
    mu_log_min = float(cfg.get("FUTURES_AWF_MU_LOG_MIN", 0.0))

    for t in pareto_trials:
        _, val_diag = replay_robust_awf_for_trial_params(base_ctx, t.params)
        awf_pos = float(t.user_attrs.get("awf_pos_frac", val_diag.get("awf_pos_frac", 0.0)))
        awf_mu = float(t.user_attrs.get("awf_mu_log", val_diag.get("awf_mu_log", -9.0)))
        if awf_pos_frac_to_pseudo_pbo(awf_pos) < pbo_limit and awf_mu >= mu_log_min:
            validated_candidates.append({
                "trial": t, "awf_diag": val_diag, "params": t.params, "values": t.values
            })

    if not validated_candidates:
        if bool(cfg.get("FUTURES_AWF_FAIL_OPEN_ENABLED", False)):
            for t in pareto_trials:
                _, val_diag = replay_robust_awf_for_trial_params(base_ctx, t.params)
                validated_candidates.append({
                    "trial": t, "awf_diag": val_diag, "params": t.params, "values": t.values
                })
        else:
            return {}, {}

    deploy_candidates = []
    reject_reason_count = {}
    for cand in validated_candidates:
        reasons = deploy_reject_reasons(cand)
        if reasons:
            for r in reasons:
                reject_reason_count[r] = reject_reason_count.get(r, 0) + 1
            continue
        cand["deploy_score"] = deploy_score(cand)
        deploy_candidates.append(cand)

    if deploy_candidates:
        ranked = sorted(
            deploy_candidates,
            key=lambda c: (safe_float(c.get("deploy_score"), -1e9),
                           cand_metric(c, "awf_robust_score", -1e9),
                           -int(c["trial"].number)),
            reverse=True
        )
        best_cand = ranked[0]
        selected_by = "deploy_score"
    else:
        best_cand = sorted(validated_candidates, key=objective_key)[0]
        selected_by = "objective"

    selection_summary = {
        "selected_by": selected_by,
        "selected_trial_number": int(best_cand["trial"].number),
        "deploy_score": float(safe_float(best_cand.get("deploy_score"), deploy_score(best_cand))),
        "selection_reject_reason_count": reject_reason_count
    }
    return best_cand, selection_summary


def check_stability_layer3(
    best_cand: dict[str, Any],
    base_ctx: MLPhaseDContext,
    cfg: dict[str, Any],
) -> tuple[float | None, bool]:
    """Execute Layer 3 multi-seed stability check for the best candidate.

    Returns:
        tuple[float|None, bool]: (champ_stab_cv, champ_l3_fail)

    """
    import dataclasses

    from src.domain.futures.optimization.optimizer import rerun_precompute_for_ctx
    from src.domain.futures.validation.tmp_md_champion import tmp_md_layer1_failures_from_awf_diag

    stab_seeds = [int(x) for x in (cfg.get("FUTURES_STABILITY_SEEDS") or [])]
    if not stab_seeds or not best_cand.get("params"):
        return None, False

    champion_raw_params = dict(best_cand["params"])
    champ_objs, champ_l3_fail = [], False
    seed_ctxs = []
    try:
        for sx in stab_seeds:
            sctx = rerun_precompute_for_ctx(dataclasses.replace(base_ctx, seed=int(sx)), sx)
            seed_ctxs.append(sctx)
            clear_stability_runtime_cache(sctx)
            obj, diag = replay_robust_awf_for_trial_params(sctx, champion_raw_params)
            champ_objs.append(obj)
            champ_l3_fail = champ_l3_fail or bool(tmp_md_layer1_failures_from_awf_diag(diag))
    finally:
        for sctx in seed_ctxs:
            release_stability_ctx(sctx)

    champ_stab_cv = stability_cv("champion", champ_objs, champ_l3_fail)
    return champ_stab_cv, champ_l3_fail
