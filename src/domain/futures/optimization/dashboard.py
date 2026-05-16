"""UI Dashboard and Reporting for Optimization."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_logger: logging.Logger = logging.getLogger("opt_futures")


def safe_float(v: Any, default: float = 0.0, limit: float = 1e9) -> float:
    """Safe conversion to float with optional clamping."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return max(-limit, min(f, limit))
    except (TypeError, ValueError):
        return default


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
    c_grn, c_red, c_rst, c_bld = "\033[92m", "\033[91m", "\033[0m", "\033[1m"

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


def log_alpha_component_summary(alpha_panel: Any) -> None:
    pass


def log_hmm_report_summary(hmm_report: dict[str, Any]) -> None:
    pass


def log_ml_merge_feature_stats(oos_data_maps: Any, valid_symbols: Any, tf: Any) -> None:
    pass


def log_oos_regime_attribution(regime_attr: dict[str, Any]) -> None:
    pass
