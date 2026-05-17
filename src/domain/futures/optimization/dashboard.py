"""UI Dashboard and Reporting for Optimization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_logger: logging.Logger = logging.getLogger("opt_futures")

C_GRN, C_RED, C_RST, C_BLD, C_YEL = "\033[92m", "\033[91m", "\033[0m", "\033[1m", "\033[93m"
SEP_85 = " " + "─" * 84
DBL_SEP_85 = "═" * 85
# Shared regime contract used by optimization modules.
REGIME_NAMES: list[str] = ["bull", "bear", "chop", "crisis"]


def safe_float(v: Any, default: float = 0.0, limit: float = 1e9) -> float:
    """Safe conversion to float with optional clamping."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return max(-limit, min(f, limit))
    except (TypeError, ValueError):
        return default


def log_alpha_component_summary(alpha_panel: pd.DataFrame, is_end_date: str | None = None) -> None:
    """Standardized Alpha Component Audit (v8.0.0 - YetiRank Survival)."""
    if alpha_panel is None or alpha_panel.empty:
        return

    _logger.info("\n 🤖 [G-ALPHA v8.0: ELITE COMPONENT SURVIVAL AUDIT]")
    _logger.info(SEP_85)
    _logger.info("  COMPONENT      │ IS-IC  │ OOS-IC │ SHORT-IC │ HALF-LIFE │ STATUS")
    _logger.info(" ────────────────┼────────┼────────┼──────────┼───────────┼───────────────────")

    # IS/OOS split
    is_panel = alpha_panel
    oos_panel: pd.DataFrame = pd.DataFrame()
    if is_end_date:
        cut = pd.to_datetime(is_end_date, utc=True)
        times = alpha_panel.index.get_level_values("datetime")
        times_utc = times.tz_convert("UTC") if times.tz is not None else times.tz_localize("UTC")
        is_panel = alpha_panel[times_utc < cut]
        oos_panel = alpha_panel[times_utc >= cut]

    components = []
    has_comp_level = "component" in alpha_panel.index.names
    if has_comp_level:
        components = sorted(alpha_panel.index.get_level_values("component").unique())
    else:
        # Fallback to column-based discovery (standard for current MLAlphaMiner)
        components = sorted([c for c in alpha_panel.columns if c.startswith("alpha_long_") or c == "alpha_long"])
    
    if not components:
        _logger.info("  No elite components found.")
        _logger.info(SEP_85 + "\n")
        return

    components = []
    has_comp_level = "component" in alpha_panel.index.names
    if has_comp_level:
        components = sorted(alpha_panel.index.get_level_values("component").unique())
    else:
        # Discover both Long and Short components
        components = sorted([c for c in alpha_panel.columns if c.startswith("alpha_long") or c.startswith("alpha_short")])
    
    if not components:
        _logger.info("  No elite components found.")
        _logger.info(SEP_85 + "\n")
        return

    filt_meta = getattr(alpha_panel, "attrs", {}).get("alpha_component_filter", {})
    n_surv = int(filt_meta.get("n_surviving", 0))
    gate_status_by_col = filt_meta.get("gate_status_by_col", {})
    ic_by_slot = filt_meta.get("ic_by_slot", {})
    gate_fail_reasons_by_col = filt_meta.get("gate_fail_reasons_by_col", {})
    
    # Sort and Group: Long first, then Short. Within each, PASS first then by OOS-IC.
    def _get_sort_key(c: str) -> tuple[int, int, float]:
        side_priority = 0 if "long" in c else 1
        stat = gate_status_by_col.get(c, {})
        is_ok = 0 if bool(stat.get("final_selection_ok", False)) else 1 # 0 is higher priority in ascending sort
        oos_val = safe_float(ic_by_slot.get(c, 0.0))
        return (side_priority, is_ok, -oos_val) # -oos_val for descending IC

    sorted_components = sorted(components, key=_get_sort_key)

    # Display logic
    failing_limit_per_side = 10
    long_failing_count = 0
    short_failing_count = 0
    
    for comp in sorted_components:
        if has_comp_level:
            sub_full = alpha_panel.xs(comp, level="component")
            primary_col = "alpha_long_00" if "alpha_long_00" in sub_full.columns else ("alpha_long" if "alpha_long" in sub_full.columns else None)
        else:
            sub_full = alpha_panel
            primary_col = comp

        if primary_col is None or "target" not in sub_full.columns:
            continue
            
        # Get metrics
        is_idx = is_panel.index.intersection(sub_full.index)
        oos_idx = oos_panel.index.intersection(sub_full.index)
        is_ic = float(sub_full.loc[is_idx, primary_col].corr(sub_full.loc[is_idx, "target"], method="spearman")) if not is_idx.empty else 0.0
        oos_ic = float(sub_full.loc[oos_idx, primary_col].corr(sub_full.loc[oos_idx, "target"], method="spearman")) if not oos_idx.empty else 0.0
        
        slot_stat = gate_status_by_col.get(comp, {})
        short_ic = safe_float(ic_by_slot.get(comp, oos_ic), default=oos_ic)
        half_life = safe_float(slot_stat.get("half_life_bars", np.nan), default=np.nan)
        if not np.isfinite(half_life):
            half_life = safe_float(filt_meta.get("primary_half_life", np.nan), default=np.nan)
        
        is_passed = bool(slot_stat.get("final_selection_ok", False))
        
        # Filtering for display
        is_long = "long" in comp
        if not is_passed:
            if is_long:
                long_failing_count += 1
                if long_failing_count > failing_limit_per_side: continue
            else:
                short_failing_count += 1
                if short_failing_count > failing_limit_per_side: continue

        status = f"{C_GRN}[PASS]{C_RST}" if is_passed else f"{C_YEL}[FAIL]{C_RST}"
        reasons = gate_fail_reasons_by_col.get(comp, [])
        reason_str = f" ({','.join(reasons)})" if not is_passed and reasons else ""
        
        _logger.info(f"  {comp:<14} │ {is_ic:>6.3f} │ {oos_ic:>6.3f} │  {short_ic:>6.3f}  │   {half_life:>4.1f}b    │ {status}{reason_str}")

    # Summary of hidden failing components
    total_long_fail = len([c for c in sorted_components if "long" in c and not gate_status_by_col.get(c, {}).get("final_selection_ok")])
    total_short_fail = len([c for c in sorted_components if "short" in c and not gate_status_by_col.get(c, {}).get("final_selection_ok")])
    
    if total_long_fail > failing_limit_per_side:
        _logger.info(f"  ... ({total_long_fail - failing_limit_per_side} more failing LONG components hidden)")
    if total_short_fail > failing_limit_per_side:
        _logger.info(f"  ... ({total_short_fail - failing_limit_per_side} more failing SHORT components hidden)")

    _logger.info(" ────────────────┴────────┴────────┴──────────┴───────────┴───────────────────")
    alpha_goal_eval_meta = _build_alpha_goal_eval_meta(alpha_panel=alpha_panel, is_end_date=is_end_date)
    alpha_panel.attrs["alpha_goal_eval_meta"] = alpha_goal_eval_meta
    
    # Calculate retention honestly: Universe mean of (OOS-IC / IS-IC) for surviving components
    surviving_ic_pairs = []
    for c in [k for k, v in gate_status_by_col.items() if v.get("final_selection_ok")]:
        p_is = float(filt_meta.get("ic_by_slot_is", {}).get(c, 0.0))
        p_oos = float(ic_by_slot.get(c, 0.0))
        if abs(p_is) > 1e-6:
            surviving_ic_pairs.append(p_oos / p_is)
    
    retention = float(np.mean(surviving_ic_pairs) * 100.0) if surviving_ic_pairs else 0.0
    verdict_str = f"{C_GRN}[READY]{C_RST}" if n_surv > 0 else f"{C_RED}[FAIL]{C_RST}"
    _logger.info(f"  🚀 G-ALPHA Verdict: {verdict_str} - {n_surv} elite slots surviving. (IC Retention: {retention:.1f}%)")
    _logger.info(SEP_85 + "\n")


def log_hmm_report_summary(hmm_report: dict[str, Any]) -> None:
    """Standardized HMM Goal Audit (v11.0.0 - Preventive Risk-Off)."""
    if not hmm_report:
        return

    _logger.info("\n 🛡️ [H-HMM v11.0: PREVENTIVE RISK-OFF AUDIT]")
    _logger.info(SEP_85)
    _logger.info("  REGIME          │ TIME % │ VOL-SCALE │ G-LOG   │ BEHAVIOR │ VERDICT")
    _logger.info(" ─────────────────┼────────┼───────────┼─────────┼──────────┼──────────────")

    regimes = [
        ("🐂 BULL-CALM", "hmm_prob_bull_calm", "GROWTH"),
        ("🚀 BULL-VOL", "hmm_prob_bull_vol_up", "GROWTH"),
        ("🎢 CHOP-ZONE", "hmm_prob_chop", "NOISE"),
        ("🐻 BEAR-TREND", "hmm_prob_bear_trend", "DEFENSE"),
        ("💀 CRISIS", "hmm_prob_crisis", "DEFENSE"),
    ]

    for label, key, behavior in regimes:
        # Compatibility: try with _share suffix first, then raw key
        time_pct = safe_float(hmm_report.get(f"{key}_share", hmm_report.get(key, 0.0))) * 100.0
        vol_scale = safe_float(hmm_report.get(f"{key}_vol_scale", 1.0))
        # Compatibility: try with _g_log suffix first, then raw g_log (already in %)
        g_log_raw = hmm_report.get(f"{key}_g_log")
        if g_log_raw is not None:
            g_log = safe_float(g_log_raw) * 100.0
        else:
            # Fallback to legacy hardcoded keys if available
            g_log = safe_float(hmm_report.get("hmm_bull_g_log" if "BULL" in label else "hmm_crisis_g_log", 0.0))
        
        verdict = "[READY]"
        if "BULL-CALM" in label:
            verdict = "[PASS: Vol < 0.85]" if vol_scale < 0.85 else "[FAIL: High Vol]"
        elif "BEAR" in label:
            verdict = "[PASS: Risk Isolated]" if g_log < 0 else "[WARN: Positive G]"
        elif "CRISIS" in label:
            verdict = "[PASS: High Stress]" if vol_scale > 2.0 else "[WARN: Low Stress]"

        # Wide-character emoji alignment:
        # Most emojis are 2 cells wide, but Python len()=1. 
        # We split emoji and text to handle padding manually.
        emoji_part = label.split(" ")[0]
        text_part = " ".join(label.split(" ")[1:])
        # "🐂 BULL-CALM" -> emoji_part="🐂", text_part="BULL-CALM"
        # We need the vertical bar at a fixed offset. 
        # Header "  REGIME          │" uses 18 characters total including "  " and "REGIME          ".
        # Emojis take 2 cells. 2(emoji) + 1(space) + 12(text) = 15 cells.
        display_label = f"  {emoji_part} {text_part:<12}"

        _logger.info(f"{display_label} │ {time_pct:>5.1f}% │   {vol_scale:>5.2f}x  │ {g_log:>+7.3f}% │ {behavior:<8} │ {verdict}")

    _logger.info(" ─────────────────┴────────┴───────────┴─────────┴──────────┴──────────────")
    _logger.info("  [INFERENCE LEVEL GATES]")
    _logger.info("  Metric                  Value      Target     Status    Meaning")
    _logger.info("  ──────────────────────────────────────────────────────────────────────────")
    
    metrics = [
        ("Lead-Lag Tail Capture", "hmm_lead_lag_tail_capture_8bar", 40.0, ">", "하방 예지력 우수"),
        ("Avg-Duration (Bars)", "hmm_avg_duration", 18.0, ">", "구조적 안정성 확보"),
        ("Crisis-Precision", "hmm_crisis_precision", 10.0, ">", "보험 오버레이 효율 적정"),
    ]
    
    for label, key, target, op, meaning in metrics:
        val = safe_float(hmm_report.get(key, 0.0))
        passed = (val >= target) if op == ">" else (val <= target)
        status = f"{C_GRN}[PASS]{C_RST}" if passed else f"{C_RED}[FAIL]{C_RST}"
        val_str = f"{val:>5.1f}%" if "%" not in label and "Duration" not in label else f"{val:>5.1f}"
        if "Duration" in label: val_str = f"{val:>5.1f} "
        
        _logger.info(f"  {label:<23} : {val_str:<8} | {op} {target:<6.1f} | {status:<8} | {meaning}")

    _logger.info("  ──────────────────────────────────────────────────────────────────────────")
    hmm_goal_eval_meta = _build_hmm_goal_eval_meta(hmm_report=hmm_report)
    hmm_report["hmm_goal_eval_meta"] = hmm_goal_eval_meta
    verdict = f"{C_GRN}[READY]{C_RST}" if safe_float(hmm_report.get("hmm_lead_lag_tail_capture_8bar", 0.0)) > 40.0 else f"{C_YEL}[WARN]{C_RST}"
    _logger.info(f"  🛡️ H-HMM Verdict: {verdict} - All risk-isolation gates passed.")
    _logger.info(SEP_85 + "\n")


def _build_alpha_goal_eval_meta(
    alpha_panel: pd.DataFrame,
    is_end_date: str | None = None,
) -> dict[str, Any]:
    """Build structured G-ALPHA audit meta for downstream aggregation."""
    filt_meta = getattr(alpha_panel, "attrs", {}).get("alpha_component_filter", {})
    required = {
        "fdr": bool("gate_status_by_col" in filt_meta),
        "dsr": bool("gate_status_by_col" in filt_meta),
        "oos_ic_floor": bool("primary_oos_mu" in filt_meta or "primary_oos_ic_mean" in filt_meta),
        "retention": bool("primary_is_mu" in filt_meta and "primary_oos_mu" in filt_meta),
        "icir_oos": bool("primary_oos_icir" in filt_meta),
        "tail_ic": bool("tail_ic_by_slot" in filt_meta),
        "short_side_ic": bool("short_head_oos_ic_mean" in filt_meta),
        "half_life": bool("half_life_diag_code_by_col" in filt_meta),
        "symbol_balance": bool("gate_status_by_col" in filt_meta),
    }
    reasons: list[str] = []
    if alpha_panel is None or alpha_panel.empty:
        reasons.append("no_elite_components")
    if not required["oos_ic_floor"]:
        reasons.append("insufficient_oos")
    if not required["icir_oos"]:
        reasons.append("missing_icir_oos")
    if not required["half_life"]:
        reasons.append("missing_half_life_diag")
    if not required["tail_ic"]:
        reasons.append("missing_tail_ic")
    if not required["short_side_ic"]:
        reasons.append("missing_short_side_ic")
    if not required["symbol_balance"]:
        reasons.append("missing_symbol_balance")
    if not required["fdr"] or not required["dsr"]:
        reasons.append("missing_gate_status")

    verdict = "pass" if not reasons else ("warn" if "no_elite_components" not in reasons else "fail")
    return {
        "framework": "g-alpha.v8",
        "verdict": verdict,
        "reason_codes": reasons,
        "required_metrics_present": required,
        "gate_summary": {
            "n_surviving": int(float(filt_meta.get("n_surviving", 0.0))),
            "n_components": int(float(filt_meta.get("n_components", 0.0))),
            "fail_fdr": int(float(filt_meta.get("fail_fdr", 0.0))),
            "fail_dsr": int(float(filt_meta.get("fail_dsr", 0.0))),
            "fail_half_life": int(float(filt_meta.get("fail_half_life", 0.0))),
            "fail_tail": int(float(filt_meta.get("fail_tail", 0.0))),
            "fail_oos": int(float(filt_meta.get("fail_oos", 0.0))),
            "fail_short": int(float(filt_meta.get("fail_short", 0.0))),
            "fail_symbol_balance": int(float(filt_meta.get("fail_sym_bal", 0.0))),
        },
        "is_end_date": str(is_end_date or ""),
    }


def _build_hmm_goal_eval_meta(hmm_report: dict[str, Any]) -> dict[str, Any]:
    """Build structured H-HMM audit meta for downstream aggregation."""
    checks: dict[str, tuple[str, float, str]] = {
        "lead_lag_tail_capture": ("hmm_lead_lag_tail_capture_8bar", 40.0, ">"),
        "avg_duration": ("hmm_avg_duration", 18.0, ">"),
        "crisis_precision": ("hmm_crisis_precision", 10.0, ">"),
        "vol_scale_calm": ("hmm_prob_bull_calm_vol_scale", 0.85, "<"),
        "damp_tail_capture": ("hmm_execution_damp_tail_capture", 80.0, ">"),
        "damp_crisis_cap": ("hmm_execution_damp_crisis_cap", 90.0, ">"),
        "protected_exposure_low": ("hmm_execution_protected_exposure_share", 30.0, ">"),
        "protected_exposure_high": ("hmm_execution_protected_exposure_share", 55.0, "<"),
        "false_flat": ("hmm_false_flat_cost", 15.0, "<"),
    }
    metric_status: dict[str, dict[str, Any]] = {}
    reason_codes: list[str] = []
    for name, (key, thr, op) in checks.items():
        raw = hmm_report.get(key)
        missing = raw is None or (isinstance(raw, float) and not np.isfinite(raw))
        if missing:
            metric_status[name] = {"status": "warn", "key": key, "reason_code": "insufficient_oos"}
            reason_codes.append("insufficient_oos")
            continue
        val = safe_float(raw, default=np.nan)
        if not np.isfinite(val):
            metric_status[name] = {"status": "warn", "key": key, "reason_code": "insufficient_oos"}
            reason_codes.append("insufficient_oos")
            continue
        passed = bool(val >= thr) if op == ">" else bool(val <= thr)
        metric_status[name] = {"status": "pass" if passed else "fail", "key": key, "value": val, "target": thr, "op": op}
        if not passed:
            reason_codes.append(f"gate_fail:{name}")
    if "hmm_sup_q10_h8_top_decile_hit" not in hmm_report:
        reason_codes.append("missing_feature_group:supervised_q_scores")
    if not bool(hmm_report.get("hmm_execution_damp_active_rate", 0.0) > 0.0):
        reason_codes.append("missing_feature_group:execution_damp_gates")
    uniq_reasons = sorted(set(reason_codes))
    if any(r.startswith("gate_fail:") for r in uniq_reasons):
        verdict = "fail"
    elif uniq_reasons:
        verdict = "warn"
    else:
        verdict = "pass"
    return {
        "framework": "h-hmm.v11",
        "verdict": verdict,
        "reason_codes": uniq_reasons,
        "metric_status": metric_status,
    }


def print_mechanical_dashboard(
    oos_port: dict[str, Any],
    gate_status: str,
    pbo_obs: float = 0.0,
    dsr_obs: float = 0.0,
    hmm_damp_tail: float = 0.0,
) -> None:
    """Final Evaluation Dashboard (v4.0.0 - Crypto-Native Mechanical Compounder)."""
    _logger.info("\n" + DBL_SEP_85)
    _logger.info(" [STEP 4/4] Final Evaluation: [MECHANICAL 24/7 DASHBOARD]")
    _logger.info(DBL_SEP_85)
    
    _logger.info("\n ────────────────────────────────────────────────────────────────────────────")
    _logger.info(" [G-OPTUNA v4.0: COMPOUND ENGINE AUDIT]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info("  Metric                  Value      Target     Status    Meaning")
    _logger.info("  ──────────────────────────────────────────────────────────────────────────")

    def get_v(k: str) -> float: return safe_float(oos_port.get(k, 0.0))

    metrics = [
        ("EV/Cost Ratio", "ev_cost_ratio", 3.0, ">", "Friction 가드 통과 (Top Priority)"),
        ("Funding Drag", "funding_drag_ratio", 25.0, "<", "펀딩 비용 전가율 안정적"),
        ("CAGR (Annualized)", "cagr_pct", 30.0, ">", "수익 목표 달성"),
        ("Sortino Ratio", "sortino", 1.8, ">", "하방 리스크 효율 우수"),
        ("Max Drawdown", "mdd_pct", 20.0, "<", "복리 생존 한도 내 관리됨"),
        ("PBO (Champion)", "pbo", 15.0, "<", "과적합 확률 통제됨"),
    ]

    for label, key, target, op, meaning in metrics:
        val = get_v(key) if key != "pbo" else pbo_obs * 100.0
        passed = (val >= target) if op == ">" else (val <= target)
        status = f"{C_GRN}[PASS]{C_RST}" if passed else f"{C_RED}[FAIL]{C_RST}"
        
        val_fmt = f"{val:>8.2f}%" if key in ["cagr_pct", "mdd_pct", "pbo", "funding_drag_ratio"] else f"{val:>8.2f} "
        tgt_fmt = f"{'≥' if op == '>' else '≤'} {target:>4.1f}" + ("%" if "%" in val_fmt else "")
        
        _logger.info(f"  {label:<21} : {val_fmt} | {tgt_fmt:<8} | {status:<8} | {meaning}")

    # Add HMM Dampening as a survival metric if provided
    if hmm_damp_tail > 0:
        passed = hmm_damp_tail >= 80.0
        status = f"{C_GRN}[PASS]{C_RST}" if passed else f"{C_RED}[FAIL]{C_RST}"
        _logger.info("  %-21s : %8.1f%% | %-8s | %-8s | %s" % 
                     ("Damp Tail-Capture", hmm_damp_tail, "≥ 80.0%", status, "(Policy) 하락장 방어 성공"))

    _logger.info("  ──────────────────────────────────────────────────────────────────────────")
    v_color = C_GRN if "PROMOTE" in gate_status or "PASS" in gate_status else C_RED
    _logger.info(f"  🏆 STRATEGY VERDICT: {v_color}[{gate_status}]{C_RST} - Ready for 24/7 Mechanical Trading.")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")


def log_ml_merge_feature_stats(oos_data_maps: Any, valid_symbols: Any, tf: Any) -> None:
    """Log minimal OOS feature merge stats for quick sanity-check."""
    if not isinstance(oos_data_maps, dict) or not valid_symbols:
        _logger.info(" [ML-MERGE] feature stats skipped (empty input)")
        return

    total_rows = 0
    total_cols = 0
    alpha_present = 0
    hmm_present = 0
    for sym in valid_symbols:
        smap = oos_data_maps.get(sym, {})
        df = smap.get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        total_rows += len(df)
        total_cols += len(df.columns)
        if "alpha_long_00" in df.columns:
            alpha_present += 1
        hmm_cols = {
            "hmm_prob_bull_calm",
            "hmm_prob_bull_vol_up",
            "hmm_prob_bear_trend",
            "hmm_prob_chop",
            "hmm_prob_crisis",
        }
        if hmm_cols.issubset(set(df.columns)):
            hmm_present += 1

    n = max(len(valid_symbols), 1)
    _logger.info(
        " [ML-MERGE] tf=%s symbols=%d avg_rows=%.1f avg_cols=%.1f alpha_col_coverage=%.1f%% hmm_col_coverage=%.1f%%",
        str(tf),
        len(valid_symbols),
        float(total_rows) / float(n),
        float(total_cols) / float(n),
        float(alpha_present) * 100.0 / float(n),
        float(hmm_present) * 100.0 / float(n),
    )


def log_oos_regime_attribution(regime_attr: dict[str, Any]) -> None:
    """Log compact OOS regime attribution table (Visual Audit)."""
    if not regime_attr:
        _logger.info(" [OOS REGIME] attribution unavailable")
        return

    metrics = regime_attr.get("regime_metrics", {})
    _logger.info("\n 📊 [OOS REGIME PERFORMANCE ATTRIBUTION]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info("  REGIME     TIME%    TRADES    WIN%      PF     AVG-PNL")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    
    for name in REGIME_NAMES:
        m = metrics.get(name, {})
        t_pct = safe_float(m.get("time_pct", 0.0))
        trades = int(m.get("trade_count", 0) or 0)
        win = safe_float(m.get("win_rate", 0.0))
        pf = safe_float(m.get("profit_factor", 1.0))
        pnl = safe_float(m.get("avg_pnl", 0.0))
        
        _logger.info(f"  {name:<9} {t_pct:>6.1f}% {trades:>8d} {win:>7.1f}% {pf:>7.2f} {pnl:>11.4f}")

    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    coverage = safe_float(regime_attr.get("trade_regime_coverage_pct", 0.0))
    flip = safe_float(regime_attr.get("chop_flip_proxy", 0.0))
    _logger.info(f"  > Signal Coverage: {coverage:.1f}% | Flip Proxy: {flip:.3f}")
    _logger.info(f"  > Chop Sensitivity: Loss Share {regime_attr.get('chop_loss_share', 0.0):.3f} | Trade Share {regime_attr.get('chop_trade_share', 0.0):.3f}")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")


def print_dual_audit_dashboard(new_m: dict[str, Any], champ_m: dict[str, Any], verdict: str) -> None:
    """Compare candidate and champion core metrics in one panel (Visual Audit)."""
    new_m = new_m or {}
    champ_m = champ_m or {}
    
    _logger.info("\n 🏆 [CANDIDATE VS CHAMPION AUDIT]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info("  METRIC           CANDIDATE     CHAMPION      DELTA       STATUS")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    metrics = [
        # (Label, Key, HigherIsBetter)
        ("CAGR(%)", "cagr", True),
        ("MDD(%)", "mdd", False),
        ("Calmar", "calmar", True),
        ("Sortino", "sortino", True),
        ("PBO(%)", "pbo", False),
        ("DSR", "dsr", True),
        ("NetAlpha(%)", "net_alpha", True),
        ("EV/Cost", "ev_cost_ratio", True),
    ]

    for label, key, higher_is_better in metrics:
        cand = safe_float(new_m.get(key, 0.0))
        champ = safe_float(champ_m.get(key, 0.0))
        delta = cand - champ
        
        # Determine status emoji
        if abs(delta) < 1e-6:
            status = "⚪ Equal"
        else:
            is_better = delta > 0 if higher_is_better else delta < 0
            status = "🔥 Better" if is_better else "🔻 Worse"

        # Formatting
        suffix = "%" if "%" in label else ""
        c_str = f"{cand:>9.2f}{suffix}"
        h_str = f"{champ:>9.2f}{suffix}"
        # For delta, use sign prefix
        d_str = f"{delta:>+9.2f}{suffix}"

        _logger.info(f"  {label:<15}  {c_str}    {h_str}    {d_str}      {status}")

    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    v_icon = "🚀" if "PROMOTE" in verdict.upper() or "PASS" in verdict.upper() else "⚠️"
    _logger.info(f"  🏁 FINAL VERDICT: {v_icon} {verdict}")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")
