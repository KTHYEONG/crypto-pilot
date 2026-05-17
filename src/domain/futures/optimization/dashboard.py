"""UI Dashboard and Reporting for Optimization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_logger: logging.Logger = logging.getLogger("opt_futures")

REGIME_NAMES = ["bull", "bear", "chop", "crisis"]


def safe_float(v: Any, default: float = 0.0, limit: float = 1e9) -> float:
    """Safe conversion to float with optional clamping."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return max(-limit, min(f, limit))
    except (TypeError, ValueError):
        return default


def _per_bar_cs_ic_series(
    df: pd.DataFrame,
    signal_col: str,
    target_col: str,
    *,
    by_level: str = "datetime",
    min_obs: int = 5,
) -> np.ndarray:
    """Return per-bar cross-sectional Spearman IC series."""
    if df is None or df.empty or signal_col not in df.columns or target_col not in df.columns:
        return np.array([], dtype=float)

    def _ic_on_group(g: pd.DataFrame) -> float:
        gg = g[[signal_col, target_col]].dropna()
        if len(gg) < min_obs:
            return np.nan
        return gg[signal_col].corr(gg[target_col], method="spearman")

    if isinstance(df.index, pd.MultiIndex) and by_level in df.index.names:
        grouped = df.groupby(level=by_level, sort=False)
        vals = grouped.apply(_ic_on_group).to_numpy(dtype=float)
    elif by_level in df.columns:
        grouped = df.groupby(by_level, sort=False)
        vals = grouped.apply(_ic_on_group).to_numpy(dtype=float)
    else:
        vals = np.array([_ic_on_group(df)], dtype=float)

    vals = vals[np.isfinite(vals)]
    return vals if vals.size else np.array([], dtype=float)


def print_human_dashboard(
    is_port: dict[str, Any],
    ho_port: dict[str, Any],
    oos_port: dict[str, Any],
    gate_status: str,
    benchmark_is: float = 0.0,
    benchmark_oos: float = 0.0,
    meta_port: dict[str, Any] | None = None,
) -> None:
    """Unified Human Dashboard: COMPOUND & SURVIVAL SUMMARY (V3.1 Mechanical)."""
    c_grn, c_red, c_rst, c_bld, c_yel = "\033[92m", "\033[91m", "\033[0m", "\033[1m", "\033[93m"

    _logger.info("\n 🤖 [MECHANICAL 24/7 DASHBOARD: V3.1 STANDARDS]")
    _logger.info(" ───────────────────────────────────────────────────────────────────────────────────")
    _logger.info("  METRIC           │      IS      │    OOS (Meta)   │  TARGET (Hurdle)  │  RESULT")
    _logger.info(" ──────────────────┼──────────────┼─────────────────┼───────────────────┼───────────")

    def get_v(p: dict[str, Any], k: str) -> float:
        return safe_float(p.get(k, 0.0))

    # [Hurdle Definitions]
    hurdles = {
        "cagr_pct": (30.0, ">"),
        "calmar": (2.5, ">"),
        "sortino": (2.0, ">"),
        "mdd_pct": (20.0, "<"),
        "mdd_duration_days": (180.0, "<"),
        "avg_trade_pnl_pct": (0.40, ">"),
        "pbo_reliability": (15.0, "<"),
    }

    def format_row(label, key, is_pct=False, is_days=False):
        is_v = get_v(is_port, key)
        oos_v = get_v(oos_port, key)
        if meta_port:
            oos_v = get_v(meta_port, key)
        
        target_val, op = hurdles.get(key, (0.0, ">"))
        
        # Color coding result
        passed = False
        if op == ">":
            passed = oos_v >= target_val
        else:
            passed = abs(oos_v) <= target_val
            
        res_color = c_grn if passed else c_red
        res_str = f"{res_color}{'PASS' if passed else 'FAIL'}{c_rst}"
        
        if is_pct:
            is_str = f"{is_v:>8.2f}%"
            oos_str = f"{oos_v:>9.2f}%"
            tgt_str = f"{op} {target_val:>6.1f}%"
        elif is_days:
            is_str = f"{is_v:>9.0f} d"
            oos_str = f"{oos_v:>10.0f} d"
            tgt_str = f"{op} {target_val:>6.0f} d"
        else:
            is_str = f"{is_v:>9.2f} "
            oos_str = f"{oos_v:>10.2f} "
            tgt_str = f"{op} {target_val:>6.2f} "
            
        _logger.info(f"  {label:<16} │  {is_str}  │  {oos_str}  │  {tgt_str:<17} │  {res_str}")

    _logger.info(" [Compound Engine]")
    format_row("CAGR", "cagr_pct", is_pct=True)
    format_row("Calmar Ratio", "calmar")
    format_row("Sortino Ratio", "sortino")
    
    _logger.info(" [Tail Risk]")
    format_row("Max Drawdown", "mdd_pct", is_pct=True)
    format_row("MDD Duration", "mdd_duration_days", is_days=True)
    
    _logger.info(" [ML Robustness]")
    format_row("Avg Trade PnL", "avg_trade_pnl_pct", is_pct=True)
    format_row("PBO Prob.", "pbo_reliability", is_pct=True)

    _logger.info(" ──────────────────┼──────────────┼─────────────────┼───────────────────┼───────────")

    is_cagr = get_v(is_port, "cagr_pct")
    oos_cagr = get_v(oos_port, "cagr_pct")
    retention = (oos_cagr / is_cagr * 100.0) if is_cagr > 1e-6 else 0.0

    ret_info = f"{retention:.1f}% of IS Performance"
    if meta_port:
        meta_cagr = get_v(meta_port, "cagr_pct")
        meta_ret = (meta_cagr / is_cagr * 100.0) if is_cagr > 1e-6 else 0.0
        ret_info = (
            f"{meta_ret:.1f}% of IS Performance (Meta-Gain: {meta_ret - retention:>+0.1f}%)"
        )

    v_icon = "⚠️ " if "HOLD" in gate_status else "✅ "
    v_color = c_grn if "PROMOTE" in gate_status else c_red
    persisted = "" if "PROMOTE" in gate_status else " - Parameters NOT persisted"

    _logger.info(f"  > OOS Retention  : {ret_info}")
    _logger.info(f"  > FINAL VERDICT  : {v_icon} {v_color}{c_bld}{gate_status}{c_rst}{persisted}")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")


def print_dual_audit_dashboard(
    new_m: dict[str, Any],
    champ_m: dict[str, Any],
    gate_status: str,
) -> None:
    """SOTA Dashboard for Strategy Promotion Audit (V3)."""
    c_grn, c_red, c_rst, c_bld, c_yel = "\033[92m", "\033[91m", "\033[0m", "\033[1m", "\033[93m"

    _logger.info("\n 🛡️ [STRATEGY AUDIT: CANDIDATE vs CHAMPION]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info("  CRITICAL (OOS)   │  CHAMPION    │  CANDIDATE   │      DELTA (Δ)")
    _logger.info(" ──────────────────┼──────────────┼──────────────┼───────────────────────────")

    def log_row(met, c_val, n_val, is_pct=False, low_better=False, is_days=False):
        c_v = safe_float(c_val)
        n_v = safe_float(n_val)
        diff = n_v - c_v

        good = (diff < 0) if low_better else (diff > 0)
        mark = f"{c_grn}▲{c_rst}" if good else f"{c_red}▼{c_rst}"
        if abs(diff) < 1e-7:
            mark = "─"

        if is_pct:
            c_str = f"{c_v:>8.2f}%"
            n_str = f"{n_v:>8.2f}%"
            d_str = f"{diff:>+8.2f}%"
        elif is_days:
            c_str = f"{c_v:>9.0f} d"
            n_str = f"{n_v:>9.0f} d"
            d_str = f"{diff:>+9.0f} d"
        else:
            c_str = f"{c_v:>9.2f} "
            n_str = f"{n_v:>9.2f} "
            d_str = f"{diff:>+9.2f} "

        _logger.info(f"  {met:<16} │  {c_str}  │  {n_str}  │  {d_str} ({mark})")

    _logger.info(" [Growth & Efficiency]")
    log_row("CAGR", champ_m.get("cagr", 0.0), new_m.get("cagr", 0.0), is_pct=True)
    _logger.info(f"   {c_yel}└ Hurdle: > 30.0%{c_rst}")
    log_row("Calmar Ratio", champ_m.get("calmar", 0.0), new_m.get("calmar", 0.0))
    _logger.info(f"   {c_yel}└ Hurdle: > 2.50{c_rst}")
    log_row("Sortino Ratio", champ_m.get("sortino", 0.0), new_m.get("sortino", 0.0))
    _logger.info(f"   {c_yel}└ Hurdle: > 2.00{c_rst}")
    
    _logger.info(" [Survival & Risk]")
    log_row("Max Drawdown", champ_m.get("mdd", 0.0), new_m.get("mdd", 0.0), is_pct=True, low_better=True)
    _logger.info(f"   {c_yel}└ Hurdle: < 20.0%{c_rst}")
    log_row("MDD Duration", champ_m.get("mdd_duration", 0.0), new_m.get("mdd_duration", 0.0), is_days=True, low_better=True)
    _logger.info(f"   {c_yel}└ Hurdle: < 180 d{c_rst}")
    
    _logger.info(" [ML Robustness]")
    log_row("Avg Trade PnL", champ_m.get("avg_pnl", 0.0), new_m.get("avg_pnl", 0.0), is_pct=True)
    _logger.info(f"   {c_yel}└ Hurdle: > 0.40%{c_rst}")
    log_row("PBO Prob.", champ_m.get("pbo", 50.0), new_m.get("pbo", 50.0), is_pct=True, low_better=True)
    _logger.info(f"   {c_yel}└ Hurdle: < 15.0%{c_rst}")

    _logger.info(" ──────────────────┼──────────────┼──────────────┼───────────────────────────")


def log_alpha_component_summary(alpha_panel: pd.DataFrame, is_end_date: str | None = None) -> None:
    """Standardized Alpha Component Audit (V7.1.0 - IS/OOS 분리 버전)."""
    if alpha_panel is None or alpha_panel.empty:
        return

    c_grn, c_red, c_rst, c_bld, c_yel = "\033[92m", "\033[91m", "\033[0m", "\033[1m", "\033[93m"

    _logger.info("\n 🤖 [ALPHA MINER AUDIT: V7.1.0 (IS/OOS SEPARATED)]")
    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────────────")
    _logger.info("  COMPONENT         │  IS-IC   │  OOS-IC  │  Tail-OOS │  ICIR(OOS) │  STATUS")
    _logger.info(" ───────────────────┼──────────┼──────────┼───────────┼────────────┼───────────────────")

    # IS/OOS 분리 슬라이스 준비
    is_panel = alpha_panel
    oos_panel: pd.DataFrame = pd.DataFrame()
    if is_end_date:
        cut = pd.to_datetime(is_end_date, utc=True)
        times = alpha_panel.index.get_level_values("datetime")
        if getattr(times, "tz", None) is None:
            times_utc = times.tz_localize("UTC")
        else:
            times_utc = times.tz_convert("UTC")
        is_panel = alpha_panel[times_utc < cut]
        oos_panel = alpha_panel[times_utc >= cut]

    components = alpha_panel.index.get_level_values("component").unique() if "component" in alpha_panel.index.names else []

    # Hurdles from g-alpha.md
    HURDLE_IC = 0.015
    HURDLE_SHORT = 0.010
    HURDLE_TAIL = 0.000
    HURDLE_ICIR = 0.50

    def _compute_col_metrics(
        panel_is: pd.DataFrame,
        panel_oos: pd.DataFrame,
        col: str,
    ) -> tuple[float, float, float, float, float]:
        """(is_ic, oos_ic, tail_oos_ic, icir_oos, short_oos_ic) 반환."""
        if "target" not in panel_is.columns:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # IS-IC
        is_ic = float(panel_is[col].corr(panel_is["target"], method="spearman")) if not panel_is.empty else 0.0

        # OOS-IC
        if not panel_oos.empty and "target" in panel_oos.columns and col in panel_oos.columns:
            oos_ic = float(panel_oos[col].corr(panel_oos["target"], method="spearman"))
            # Tail-OOS IC
            q10_low = panel_oos["target"].quantile(0.10)
            q10_high = panel_oos["target"].quantile(0.90)
            tail_mask = (panel_oos["target"] <= q10_low) | (panel_oos["target"] >= q10_high)
            tail_oos_ic = float(
                panel_oos.loc[tail_mask, col].corr(panel_oos.loc[tail_mask, "target"], method="spearman")
            ) if tail_mask.sum() > 20 else 0.0
            # ICIR(OOS) — per-bar cross-sectional IC series on OOS slice
            ic_series = _per_bar_cs_ic_series(panel_oos, col, "target")
            ic_mean = float(np.nanmean(ic_series)) if ic_series.size else 0.0
            ic_std = float(np.nanstd(ic_series, ddof=1)) if ic_series.size > 1 else 0.0
            icir_oos = (ic_mean / ic_std) if ic_std > 1e-12 else 0.0
            # Short IC(OOS)
            short_signal_col = "alpha_short" if "alpha_short" in panel_oos.columns else None
            if short_signal_col:
                short_sig = panel_oos[short_signal_col]
            else:
                short_sig = 1.0 - panel_oos[col]
            short_target = -panel_oos["target"]
            s_mask = short_sig.notna() & short_target.notna()
            short_oos_ic = float(
                short_sig.loc[s_mask].corr(short_target.loc[s_mask], method="spearman")
            ) if s_mask.sum() > 20 else 0.0
        else:
            # OOS 없으면 IS 수치로 fallback (표시만)
            oos_ic = is_ic
            q10_low = panel_is["target"].quantile(0.10)
            q10_high = panel_is["target"].quantile(0.90)
            tail_mask = (panel_is["target"] <= q10_low) | (panel_is["target"] >= q10_high)
            tail_oos_ic = float(
                panel_is.loc[tail_mask, col].corr(panel_is.loc[tail_mask, "target"], method="spearman")
            ) if tail_mask.sum() > 20 else 0.0
            ic_series = _per_bar_cs_ic_series(panel_is, col, "target")
            ic_mean = float(np.nanmean(ic_series)) if ic_series.size else 0.0
            ic_std = float(np.nanstd(ic_series, ddof=1)) if ic_series.size > 1 else 0.0
            icir_oos = (ic_mean / ic_std) if ic_std > 1e-12 else 0.0
            short_oos_ic = 0.0

        return is_ic, oos_ic, tail_oos_ic, icir_oos, short_oos_ic

    if not components:
        cols = [c for c in alpha_panel.columns if (c.startswith("alpha_long_") and c[-2:].isdigit()) or c == "alpha_long"]
        if not cols:
            _logger.info("  No alpha components found in panel.")
            return

        if "target" not in alpha_panel.columns:
            for col in sorted(cols):
                _logger.info(f"  {col:<17} │    --    │    --    │    --     │     --     │   [READY]")
            _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────────────\n")
            return

        for col in sorted(cols):
            is_ic, oos_ic, tail_oos_ic, icir_oos, _short_oos_ic = _compute_col_metrics(
                is_panel, oos_panel, col
            )
            # Status 판정: OOS-IC 기준
            eval_ic = oos_ic if not oos_panel.empty else is_ic
            ok_ic = eval_ic >= HURDLE_IC
            ok_tail = tail_oos_ic >= HURDLE_TAIL
            ok_icir = icir_oos >= HURDLE_ICIR
            all_pass = ok_ic and ok_tail and ok_icir

            if not ok_tail:
                res_str = f"{c_red}[REJECTED]{c_rst}"
            elif all_pass:
                res_str = f"{c_grn}[PASS]{c_rst}"
            else:
                res_str = f"{c_yel}[FAIL]{c_rst}"

            _logger.info(
                f"  {col:<17} │ {is_ic:>7.3f}  │ {oos_ic:>7.3f}  │ {tail_oos_ic:>8.3f}  │ {icir_oos:>9.2f}  │  {res_str}"
            )
        _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────────────\n")
        return

    for comp in sorted(components):
        sub_full = alpha_panel.xs(comp, level="component")
        primary_col = "alpha_long_00" if "alpha_long_00" in sub_full.columns else ("alpha_long" if "alpha_long" in sub_full.columns else None)
        if "target" not in sub_full.columns or primary_col is None:
            continue

        # component-level IS/OOS 분리
        if not is_panel.empty and "component" in is_panel.index.names:
            sub_is = is_panel.xs(comp, level="component") if comp in is_panel.index.get_level_values("component") else pd.DataFrame()
            sub_oos = oos_panel.xs(comp, level="component") if (not oos_panel.empty and comp in oos_panel.index.get_level_values("component")) else pd.DataFrame()
        else:
            sub_is = sub_full
            sub_oos = pd.DataFrame()

        is_ic, oos_ic, tail_oos_ic, icir_oos, _short_oos_ic = _compute_col_metrics(
            sub_is, sub_oos, primary_col
        )

        # Status 판정: OOS-IC 기준
        eval_ic = oos_ic if not sub_oos.empty else is_ic
        ok_ic = eval_ic >= HURDLE_IC
        ok_tail = tail_oos_ic >= HURDLE_TAIL
        ok_icir = icir_oos >= HURDLE_ICIR
        all_pass = ok_ic and ok_tail and ok_icir

        if not ok_tail:
            res_str = f"{c_red}[REJECTED]{c_rst}"
        elif all_pass:
            res_str = f"{c_grn}[PASS]{c_rst}"
        else:
            res_str = f"{c_yel}[FAIL]{c_rst}"

        _logger.info(
            f"  {comp:<17} │ {is_ic:>7.3f}  │ {oos_ic:>7.3f}  │ {tail_oos_ic:>8.3f}  │ {icir_oos:>9.2f}  │  {res_str}"
        )

    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────────────")
    oos_label = f"OOS >= {is_end_date}" if is_end_date else "OOS (IS tail fallback)"
    _logger.info(f"  [HURDLES] OOS-IC > {HURDLE_IC} | Tail-OOS > {HURDLE_TAIL} | ICIR(OOS) > {HURDLE_ICIR}  [{oos_label}]")
    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────────────\n")


def log_hmm_report_summary(hmm_report: dict[str, Any]) -> None:
    """Standardized HMM Audit Report (V11.0.0)."""
    if not hmm_report:
        return

    c_grn, c_red, c_rst, c_bld, c_yel = "\033[92m", "\033[91m", "\033[0m", "\033[1m", "\033[93m"

    _logger.info("\n 🧠 [HMM RISK OVERLAY AUDIT: V11.0.0]")
    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────")
    _logger.info("  METRIC            │      VALUE      │   TARGET (Hurdle)   │  RESULT")
    _logger.info(" ───────────────────┼─────────────────┼─────────────────────┼─────────────────────────")

    def get_v(k: str) -> float:
        return safe_float(hmm_report.get(k, 0.0))

    # [Hurdle Definitions]
    # (Label, Key, Target, Operator, Is_Pct)
    inference_metrics = [
        ("Lead-Lag Tail Cap", "hmm_lead_lag_tail_capture_8bar", 40.0, ">", True),
        ("Avg-Duration", "hmm_avg_duration", 18.0, ">", False),
        ("Crisis-Prec", "hmm_crisis_precision", 10.0, ">", True),
        ("Vol-Scale (Calm)", "regime_prob_risk_on_calm_vol_scale", 0.85, "<", False),
    ]

    policy_metrics = [
        ("Damp Tail-Capture", "hmm_execution_damp_tail_capture", 80.0, ">", True),
        ("Damp Crisis-Cap", "hmm_execution_damp_crisis_cap", 90.0, ">", True),
        ("Protected Exp.", "hmm_execution_protected_exposure_share", (30.0, 55.0), "range", True),
        ("False-Flat", "hmm_false_flat_cost", 15.0, "<", True),
    ]

    def print_rows(metrics, header):
        _logger.info(f" [{header}]")
        for label, key, target, op, is_pct in metrics:
            val = get_v(key)
            
            passed = False
            if op == ">":
                passed = val >= target
            elif op == "<":
                passed = val <= target
            elif op == "range":
                passed = target[0] <= val <= target[1]
            
            res_color = c_grn if passed else c_red
            res_str = f"{res_color}{'[PASS]' if passed else '[FAIL]'}{c_rst}"
            
            val_fmt = f"{val:>14.2f}%" if is_pct else f"{val:>14.2f} "
            if not is_pct and "Duration" in label:
                val_fmt = f"{val:>14.1f} b"
            elif not is_pct and "Scale" in label:
                val_fmt = f"{val:>14.2f} x"

            if op == "range":
                tgt_fmt = f"{target[0]:.0f}% ~ {target[1]:.0f}%"
            else:
                tgt_fmt = f"{op} {target:.1f}" + ("%" if is_pct else "")
            
            _logger.info(f"  {label:<17} │ {val_fmt}  │  {tgt_fmt:<18} │   {res_str}")

    print_rows(inference_metrics, "Inference Level")
    print_rows(policy_metrics, "Policy Level")

    # Per-symbol beta/idio overlay metrics (Step 5)
    _logger.info(" [Per-Symbol Overlay]")
    _ps_beta_exp = safe_float(hmm_report.get("per_sym_beta_adj_protected_exp", 0.0))
    _ps_mono = safe_float(hmm_report.get("per_sym_beta_monotonicity_corr", 0.0))
    _ps_exp_passed = 30.0 <= _ps_beta_exp <= 55.0
    _ps_mono_passed = _ps_mono > 0.0
    _logger.info(
        "  %-17s │ %14.2f%%  │  %-18s │   %s",
        "β-adj Prot. Exp.",
        _ps_beta_exp,
        "30.0% ~ 55.0%",
        f"{c_grn}[PASS]{c_rst}" if _ps_exp_passed else f"{c_red}[FAIL]{c_rst}",
    )
    _logger.info(
        "  %-17s │ %14.4f   │  %-18s │   %s",
        "β-Protect Monot.",
        _ps_mono,
        "> 0",
        f"{c_grn}[PASS]{c_rst}" if _ps_mono_passed else f"{c_red}[FAIL]{c_rst}",
    )
    _logger.info(
        "  %-17s │ %14s   │  %-18s │   %s",
        "Idio Crash Cap.",
        "N/A",
        "> 50.0%",
        "[N/A]",
    )

    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────")
    backend = hmm_report.get("hmm_backend", "unknown")
    _logger.info(f"  > HMM Backend    : {backend}")
    
    all_passed = True
    for _, k, t, o, _ in inference_metrics + policy_metrics:
        v = get_v(k)
        p = False
        if o == ">": p = v >= t
        elif o == "<": p = v <= t
        elif o == "range": p = t[0] <= v <= t[1]
        if not p:
            all_passed = False
            break
            
    verdict = f"{c_grn}[READY]{c_rst}" if all_passed else f"{c_yel}[WARN]{c_rst}"
    _logger.info(f"  > Final Verdict  : {verdict} - Validated for Optuna Search Space")
    _logger.info(" ─────────────────────────────────────────────────────────────────────────────────────\n")


def log_ml_merge_feature_stats(oos_data_maps: Any, valid_symbols: Any, tf: Any) -> None:
    pass


def log_oos_regime_attribution(regime_attr: dict[str, Any]) -> None:
    pass
