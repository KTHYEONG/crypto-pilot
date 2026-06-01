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
    _logger.info("  COMPONENT      │ IS-IC  │ OOS-IC │ CSIC-OOS │ HALF-LIFE │ STATUS")
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

    components: list[str] = []
    has_comp_level = "component" in alpha_panel.index.names
    if not has_comp_level:
        # Prefer gate-status keys to avoid duplicate alias rows.
        filt_meta_probe = getattr(alpha_panel, "attrs", {}).get("alpha_component_filter", {})
        gate_probe = (
            filt_meta_probe.get("gate_status_by_col", {})
            if isinstance(filt_meta_probe, dict)
            else {}
        )
        if isinstance(gate_probe, dict) and gate_probe:
            components = sorted(str(k) for k in gate_probe)
    if has_comp_level:
        components = sorted(alpha_panel.index.get_level_values("component").unique())
    elif not components:
        # Discover both Long and Short components
        components = sorted(
            [
                c
                for c in alpha_panel.columns
                if c.startswith("alpha_long_")
                or c.startswith("alpha_short_")
                or c in {"alpha_long_signal", "alpha_short_signal"}
            ]
        )

    if not components:
        _logger.info("  No elite components found.")
        _logger.info(SEP_85 + "\n")
        return

    filt_meta = getattr(alpha_panel, "attrs", {}).get("alpha_component_filter", {})
    n_surv = int(filt_meta.get("n_surviving", 0))
    gate_status_by_col = filt_meta.get("gate_status_by_col", {})
    ic_by_slot = filt_meta.get("ic_by_slot", {})
    gate_fail_reasons_by_col = filt_meta.get("gate_fail_reasons_by_col", {})

    def _resolve_comp_series_col(comp: str) -> str | None:
        if comp in alpha_panel.columns:
            return comp
        if comp == "alpha_long_signal":
            for c in ("alpha_long_signal", "alpha_long"):
                if c in alpha_panel.columns:
                    return c
        if comp == "alpha_short_signal":
            for c in ("alpha_short_signal", "alpha_short"):
                if c in alpha_panel.columns:
                    return c
        return None

    def _csic_mean(panel: pd.DataFrame, pred_col: str, is_short: bool) -> float:
        if panel.empty or pred_col not in panel.columns or "target" not in panel.columns:
            return 0.0
        wide_pred = panel[pred_col].unstack(level="symbol")
        wide_tgt = panel["target"].unstack(level="symbol")
        if is_short:
            wide_tgt = 1.0 - wide_tgt
        pred_rank = wide_pred.rank(axis=1)
        tgt_rank = wide_tgt.rank(axis=1)
        ics = pred_rank.corrwith(tgt_rank, axis=1).dropna()
        return float(ics.mean()) if len(ics) > 0 else 0.0

    is_ic_fallback: dict[str, float] = {}
    oos_ic_fallback: dict[str, float] = {}
    for comp in components:
        pred_col = _resolve_comp_series_col(comp)
        if pred_col is None:
            continue
        is_short = "short" in comp
        is_ic_fallback[comp] = _csic_mean(is_panel, pred_col, is_short=is_short)
        oos_ic_fallback[comp] = _csic_mean(oos_panel, pred_col, is_short=is_short)

    # Sort and Group: Long first, then Short. Within each, PASS first then by OOS-IC.
    def _get_sort_key(c: str) -> tuple[int, int, float]:
        side_priority = 0 if "long" in c else 1
        stat = gate_status_by_col.get(c, {})
        is_ok = (
            0 if bool(stat.get("final_selection_ok", False)) else 1
        )  # 0 is higher priority in ascending sort
        oos_val = safe_float(
            filt_meta.get("ic_oos_by_slot", {}).get(c, oos_ic_fallback.get(c, 0.0))
        )
        return (side_priority, is_ok, -oos_val)  # -oos_val for descending IC

    sorted_components = sorted(components, key=_get_sort_key)

    # Display logic
    failing_limit_per_side = 10
    long_failing_count = 0
    short_failing_count = 0

    for comp in sorted_components:
        if has_comp_level:
            sub_full = alpha_panel.xs(comp, level="component")
            primary_col = (
                "alpha_long_signal"
                if "alpha_long_signal" in sub_full.columns
                else (
                    "alpha_long" if "alpha_long" in sub_full.columns else None
                )
            )
        else:
            sub_full = alpha_panel
            primary_col = _resolve_comp_series_col(comp)

        if primary_col is None or "target" not in sub_full.columns:
            continue

        # Get metrics
        is_ic = safe_float(ic_by_slot.get(comp, is_ic_fallback.get(comp, 0.0)))
        oos_ic = safe_float(
            filt_meta.get("ic_oos_by_slot", {}).get(comp, oos_ic_fallback.get(comp, 0.0))
        )

        slot_stat = gate_status_by_col.get(comp, {})
        ic_oos_slot_map = filt_meta.get("ic_oos_by_slot", {})  # Fix 3B: OOS CS-IC per slot
        short_ic = safe_float(ic_oos_slot_map.get(comp, oos_ic), default=oos_ic)
        half_life = safe_float(slot_stat.get("half_life_bars", np.nan), default=np.nan)
        if not np.isfinite(half_life):
            half_life = safe_float(filt_meta.get("primary_half_life", np.nan), default=np.nan)

        is_passed = bool(slot_stat.get("final_selection_ok", False))

        # Filtering for display
        is_long = "long" in comp
        if not is_passed:
            if is_long:
                long_failing_count += 1
                if long_failing_count > failing_limit_per_side:
                    continue
            else:
                short_failing_count += 1
                if short_failing_count > failing_limit_per_side:
                    continue

        status = f"{C_GRN}[PASS]{C_RST}" if is_passed else f"{C_YEL}[FAIL]{C_RST}"
        reasons = gate_fail_reasons_by_col.get(comp, [])
        reason_str = f" ({','.join(reasons)})" if not is_passed and reasons else ""

        disp_comp = (
            "long_signal"
            if comp == "alpha_long_signal"
            else ("short_signal" if comp == "alpha_short_signal" else comp)
        )
        _logger.info(
            "  %s │ %6.3f │ %6.3f │  %6.3f  │   %4.1fb    │ %s%s",
            f"{disp_comp:<14}",
            is_ic,
            oos_ic,
            short_ic,
            half_life,
            status,
            reason_str,
        )

    # Summary of hidden failing components
    total_long_fail = len(
        [
            c
            for c in sorted_components
            if "long" in c and not gate_status_by_col.get(c, {}).get("final_selection_ok")
        ]
    )
    total_short_fail = len(
        [
            c
            for c in sorted_components
            if "short" in c and not gate_status_by_col.get(c, {}).get("final_selection_ok")
        ]
    )

    if total_long_fail > failing_limit_per_side:
        _logger.info(
            "  ... (%d more failing LONG components hidden)",
            total_long_fail - failing_limit_per_side,
        )
    if total_short_fail > failing_limit_per_side:
        _logger.info(
            "  ... (%d more failing SHORT components hidden)",
            total_short_fail - failing_limit_per_side,
        )

    _logger.info(" ────────────────┴────────┴────────┴──────────┴───────────┴───────────────────")
    root_diag = filt_meta.get("root_cause_diag", {}) if isinstance(filt_meta, dict) else {}
    if isinstance(root_diag, dict) and root_diag:
        raw_is = safe_float(root_diag.get("raw_alpha_is_csic_mean", 0.0))
        raw_oos = safe_float(root_diag.get("raw_alpha_oos_csic_mean", 0.0))
        adj_is = safe_float(root_diag.get("adjusted_alpha_is_csic_mean", 0.0))
        adj_oos = safe_float(root_diag.get("adjusted_alpha_oos_csic_mean", 0.0))
        component_oos = (
            root_diag.get("component_oos_csic_mean", {})
            if isinstance(root_diag.get("component_oos_csic_mean", {}), dict)
            else {}
        )
        sign_ok = bool(root_diag.get("signal_sign_ok", False))
        _logger.info("  [ALPHA ROOT-CAUSE]")
        _logger.info(
            "  raw_is=%+.4f, raw_oos=%+.4f, adj_is=%+.4f, adj_oos=%+.4f",
            raw_is,
            raw_oos,
            adj_is,
            adj_oos,
        )
        if component_oos:
            formatted = ", ".join(
                f"{key!s}={safe_float(value, default=0.0):+.4f}"
                for key, value in sorted(component_oos.items())
            )
            _logger.info("  component_oos: %s", formatted)
        _logger.info(f"  sign_ok={sign_ok}")

    alpha_goal_eval_meta = _build_alpha_goal_eval_meta(
        alpha_panel=alpha_panel, is_end_date=is_end_date
    )
    alpha_panel.attrs["alpha_goal_eval_meta"] = alpha_goal_eval_meta

    # Fix 3A: IC Retention — IS CS-IC (ic_by_slot) vs OOS CS-IC (ic_oos_by_slot)
    surviving_ic_pairs = []
    ic_oos_by_slot = filt_meta.get("ic_oos_by_slot", {})
    for c in [k for k, v in gate_status_by_col.items() if v.get("final_selection_ok")]:
        p_is = float(ic_by_slot.get(c, is_ic_fallback.get(c, 0.0)))
        p_oos = float(ic_oos_by_slot.get(c, oos_ic_fallback.get(c, 0.0)))
        if abs(p_is) > 1e-6:
            surviving_ic_pairs.append(p_oos / p_is)

    retention = float(np.mean(surviving_ic_pairs) * 100.0) if surviving_ic_pairs else 0.0

    # Fix 3C: Verdict with OOS CS-IC quality gate
    surviving_cs = [k for k, v in gate_status_by_col.items() if v.get("final_selection_ok")]
    mean_survivor_oos_cs = (
        float(
            np.mean(
                [
                    safe_float(ic_oos_by_slot.get(c, oos_ic_fallback.get(c, 0.0)))
                    for c in surviving_cs
                ]
            )
        )
        if surviving_cs
        else 0.0
    )
    is_ready = (
        n_surv > 0
        and mean_survivor_oos_cs >= 0.02
        and (retention >= 50.0 or len(surviving_ic_pairs) == 0)
    )
    verdict_str = (
        f"{C_GRN}[READY]{C_RST}"
        if is_ready
        else (f"{C_YEL}[MARGINAL]{C_RST}" if n_surv > 0 else f"{C_RED}[FAIL]{C_RST}")
    )
    _logger.info(
        "  🚀 G-ALPHA Verdict: %s - %d elite slots surviving. "
        "(IC Retention: %.1f%% | OOS-CS-IC: %.4f)",
        verdict_str,
        n_surv,
        retention,
        mean_survivor_oos_cs,
    )
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

    verdict = (
        "pass" if not reasons else ("warn" if "no_elite_components" not in reasons else "fail")
    )
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


def print_mechanical_dashboard(
    oos_port: dict[str, Any],
    gate_status: str,
    pbo_obs: float = 0.0,
    dsr_obs: float = 0.0,
) -> None:
    """Final Evaluation Dashboard (v4.0.0 - Crypto-Native Mechanical Compounder)."""
    _logger.info("\n" + DBL_SEP_85)
    _logger.info(" [STEP 4/4] Final Evaluation: [MECHANICAL 24/7 DASHBOARD]")
    _logger.info(DBL_SEP_85)

    # Brief Summary of Core Metrics
    trades = int(oos_port.get("total_trades", oos_port.get("trade_count", 0)))
    win_rate = safe_float(oos_port.get("win_rate_pct", 0.0))
    pf = safe_float(oos_port.get("profit_factor", 1.0))
    avg_pnl = safe_float(oos_port.get("avg_trade_pnl_pct", 0.0))

    _logger.info(
        "\n [OOS_SUMMARY] trades=%d win_rate=%.1f%% profit_factor=%.2f avg_pnl=%.4f%%",
        trades,
        win_rate,
        pf,
        avg_pnl,
    )

    _logger.info("\n ────────────────────────────────────────────────────────────────────────────")
    _logger.info(" [STRATEGY PERFORMANCE AUDIT]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info("  Metric                  Value      Target     Status    Meaning")
    _logger.info("  ──────────────────────────────────────────────────────────────────────────")

    def get_v(k: str) -> float:
        return safe_float(oos_port.get(k, 0.0))

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

        val_fmt = (
            f"{val:>8.2f}%"
            if key in ["cagr_pct", "mdd_pct", "pbo", "funding_drag_ratio"]
            else f"{val:>8.2f} "
        )
        tgt_fmt = f"{'≥' if op == '>' else '≤'} {target:>4.1f}" + ("%" if "%" in val_fmt else "")

        _logger.info(f"  {label:<21} : {val_fmt} | {tgt_fmt:<8} | {status:<8} | {meaning}")

    _logger.info("  ──────────────────────────────────────────────────────────────────────────")
    v_color = C_GRN if "PROMOTE" in gate_status or "PASS" in gate_status else C_RED
    _logger.info(
        "  🏆 STRATEGY VERDICT: %s[%s]%s - Ready for 24/7 Mechanical Trading.",
        v_color,
        gate_status,
        C_RST,
    )
    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")


def log_ml_merge_feature_stats(oos_data_maps: Any, valid_symbols: Any, tf: Any) -> None:
    """Log minimal OOS feature merge stats for quick sanity-check."""
    if not isinstance(oos_data_maps, dict) or not valid_symbols:
        _logger.info(" [ML-MERGE] feature stats skipped (empty input)")
        return

    total_rows = 0
    total_cols = 0
    alpha_present = 0
    for sym in valid_symbols:
        smap = oos_data_maps.get(sym, {})
        df = smap.get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        total_rows += len(df)
        total_cols += len(df.columns)
        has_alpha_cols = (
            "alpha_long" in df.columns or "alpha_short" in df.columns
        )
        if has_alpha_cols:
            alpha_present += 1

    n = max(len(valid_symbols), 1)
    _logger.info(
        " [ML-MERGE] tf=%s symbols=%d avg_rows=%.1f avg_cols=%.1f alpha_col_coverage=%.1f%%",
        str(tf),
        len(valid_symbols),
        float(total_rows) / float(n),
        float(total_cols) / float(n),
        float(alpha_present) * 100.0 / float(n),
    )


def log_oos_regime_attribution(regime_attr: dict[str, Any]) -> None:
    """Log high-level signal metrics (Visual Audit)."""
    if not regime_attr:
        return

    coverage = safe_float(regime_attr.get("trade_regime_coverage_pct", 0.0))
    flip = safe_float(regime_attr.get("chop_flip_proxy", 0.0))
    _logger.info(
        " [OOS_SIGNAL] coverage=%.1f%% flip_proxy=%.3f chop_loss_share=%.3f chop_trade_share=%.3f",
        coverage,
        flip,
        regime_attr.get("chop_loss_share", 0.0),
        regime_attr.get("chop_trade_share", 0.0),
    )


def log_oos_alpha_attribution(report: dict[str, Any]) -> None:
    """Log OOS alpha attribution diagnostics report."""
    if not report:
        return
    pnl = report.get("pnl_pct", {})
    share = report.get("share_pct", {})
    obs = int(safe_float(report.get("n_obs", 0.0), 0.0))
    _logger.info(
        " [OOS_ALPHA_ATTR] n_obs=%d total=%.4f market=%.4f factor_proxy=%.4f residual=%.4f",
        obs,
        safe_float(pnl.get("total", 0.0)),
        safe_float(pnl.get("market", 0.0)),
        safe_float(pnl.get("factor_proxy", 0.0)),
        safe_float(pnl.get("residual", 0.0)),
    )
    _logger.info(
        " [OOS_ALPHA_ATTR_SHARE] market=%.1f%% factor_proxy=%.1f%% residual=%.1f%%",
        safe_float(share.get("market", 0.0)),
        safe_float(share.get("factor_proxy", 0.0)),
        safe_float(share.get("residual", 0.0)),
    )


def print_dual_audit_dashboard(
    new_m: dict[str, Any], champ_m: dict[str, Any], verdict: str
) -> None:
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
