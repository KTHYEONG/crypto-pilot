from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from src.domain.futures.optimization.opt_config import (
    OPT_FUTURES_CONFIG,
    default_ev_hurdle_bps,
)
from src.domain.futures.optimization.workflow import (
    FIXED_DEFAULTS,
    suggest_joint_params,
    suggest_risk_params,
    suggest_signal_params,
)
from src.domain.futures.portfolio.portfolio_optimizer import (
    load_portfolio_policy_config,
)

if TYPE_CHECKING:
    from src.domain.futures.optimization.ml_context import MLPhaseDContext


def _fixed_ml_phase_d_params() -> dict[str, Any]:
    """Constants that must stay aligned between optimization and final evaluation."""
    return {
        "MIN_SCORE_PERCENTILE": 0.55,
        "RISK_PER_TRADE": 0.05,
    }


def infer_kelly_shrinkage_bayesian_c_for_enqueue(fk_target: float, *, shield: bool) -> tuple[float, float]:
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
    from src.domain.futures.portfolio.portfolio_optimizer import finalize_strategy_portfolio_params

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
        trail_m = _snap_float_list(float(deploy.get("TRAIL_MULT", deploy.get("LONG_TRAIL_MULT", 3.0))), trail_choices)

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
        "CRISIS_LONG_MAG_SUPPRESS": float(OPT_FUTURES_CONFIG.get("FUTURES_CRISIS_LONG_MAG_SUPPRESS", 1.0)),
        "BETA_ALPHA": float(fc.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0)),
        "EV_HURDLE_BPS": float(default_ev_hurdle_bps(fc)),
        "SLIPPAGE_BPS_BUFFER_MULT": float(fc.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0)),
        "TIME_BARRIER_H": float(fc.get("FUTURES_DEFAULT_TIME_BARRIER_H", 0.0)),
    }


def _suggest_ml_joint_nsga2(trial: optuna.Trial, ctx: MLPhaseDContext) -> dict[str, Any]:
    """Joint parameter space for main objective path (V4.3 core + fixed defaults)."""
    from src.domain.futures.portfolio.portfolio_optimizer import finalize_strategy_portfolio_params

    policy = load_portfolio_policy_config(OPT_FUTURES_CONFIG)
    default_ranges = {"MAX_EXPOSURE": (0.50, min(float(policy.gross_exposure_cap), 3.00))}
    phase = str(getattr(ctx, "coordinate_phase", "") or "").lower()
    frozen = dict(getattr(ctx, "coordinate_frozen_params", None) or {})
    phase_ranges = dict(default_ranges)
    shrunk_ranges = dict(getattr(ctx, "coordinate_shrunk_ranges", None) or {})
    phase_ranges.update(shrunk_ranges)
    phase_ranges.update(dict(getattr(ctx, "phase_ranges", None) or {}))

    baseline = _baseline_ml_out_dict_for_coordinate(policy)
    baseline_core = {
        "BETA_ALPHA": float(baseline.get("BETA_ALPHA", 1.0)),
        "K_LONG": int(baseline.get("K_LONG", 2)),
        "K_SHORT": int(baseline.get("K_SHORT", 2)),
        "REBALANCE_BARS": int(baseline.get("REBALANCE_BARS", 6)),
        "EV_HURDLE_BPS": float(baseline.get("EV_HURDLE_BPS", default_ev_hurdle_bps(OPT_FUTURES_CONFIG))),
        "PORTFOLIO_KAPPA": float(baseline.get("PORTFOLIO_KAPPA", 0.35)),
        "TARGET_ANN_VOL": float(baseline.get("TARGET_ANN_VOL", 0.25)),
        "MAX_EXPOSURE": float(baseline.get("MAX_EXPOSURE", 1.2)),
        "MAX_EXPOSURE_PER_COIN": float(baseline.get("MAX_EXPOSURE_PER_COIN", 0.25)),
    }

    base: dict[str, Any]
    if phase in {"phase_a1", "a1"}:
        phase_suggested = suggest_signal_params(trial, ranges=phase_ranges, fixed=frozen)
        base = dict(baseline_core)
        base.update(frozen)
        base.update(phase_suggested)
    elif phase in {"phase_a2", "a2"}:
        phase_suggested = suggest_risk_params(trial, ranges=phase_ranges, fixed=frozen)
        base = dict(baseline_core)
        base.update(frozen)
        base.update(phase_suggested)
    else:
        base = dict(suggest_joint_params(trial, ranges=phase_ranges, fixed=frozen))

    base.update(FIXED_DEFAULTS)
    kappa = float(base["PORTFOLIO_KAPPA"])
    base["KELLY_LAMBDA"] = kappa
    base["RISK_PER_TRADE"] = kappa
    base["SIZING_METHOD"] = "profit_factor_kelly"

    return finalize_strategy_portfolio_params(base, policy)


def build_ml_phase_d_params(trial_params: dict[str, Any], tf: str) -> dict[str, Any]:
    from src.domain.futures.optimization.objectives import _base_engine_params

    merged = dict(_fixed_ml_phase_d_params())
    merged.update(trial_params)
    return _base_engine_params(merged, tf)
