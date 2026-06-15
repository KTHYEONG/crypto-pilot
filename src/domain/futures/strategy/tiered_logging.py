"""Tiered pipeline pipe-table log formatters (§9 Logging Contract).

All functions are pure (no side-effects). They return formatted pipe-table
strings intended to be passed to logging.info() by callers.

Time Complexity: O(n) where n is the length of optional detail lists.
Space Complexity: O(n) for output string construction.
"""
from __future__ import annotations

import math
from typing import Any

from src.domain.futures.optimization.opt_config import LayeredWindow

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gate(passed: bool) -> str:
    """Return 'PASS' or 'BLOCKED' based on gate status."""
    return "PASS" if passed else "BLOCKED"


def _pct(v: float) -> str:
    """Format float as percentage string (e.g. 0.85 → '85.0%')."""
    return f"{v:.1%}"


def _fmt_date(d: Any) -> str:
    """Format a date-like object as ISO string."""
    return str(d)


# ---------------------------------------------------------------------------
# §9.0 System Dashboard & Headers
# ---------------------------------------------------------------------------

def format_system_context_dashboard(
    *,
    window: Any,
    universe_report: dict[str, Any],
    data_quality: dict[str, Any],
    strategy_info: dict[str, Any],
) -> str:
    """Consolidate initialization metadata into a single box dashboard."""
    w = window
    u = universe_report
    dq = data_quality
    s = strategy_info

    def _box(content: list[str], width: int = 78) -> str:
        # width는 내부 텍스트가 들어갈 가용 공간 (padding 제외)
        # 전체 가로 길이는 ┌ + 공백 + width + 공백 + ┐ = width + 4
        top = "┌" + "─" * (width + 2) + "┐"
        bottom = "└" + "─" * (width + 2) + "┘"
        lines = [top]
        for line in content:
            if line == "SEP":
                # 구분선은 박스 내부 너비 전체를 채움
                lines.append("├" + "─" * (width + 2) + "┤")
            else:
                lines.append(f"│ {line:<{width}} │")
        lines.append(bottom)
        return "\n".join(lines)

    # Convert window to string
    window_str = f"Range: {w.fetch_start} ~ {w.end_date} (IS:{w.is_start}, OOS:{w.oos_start})"
    
    # Universe string
    u_str = (
        f"Discovered: {u.get('discovered', 0)} symbols | "
        f"Selected: {u.get('selected', 0)} | "
        f"Live Panel: {u.get('live_panel', 0)}"
    )

    # Data Quality string
    dq_str = (
        f"Loaded: {dq.get('loaded_ratio', '0%')} ({dq.get('loaded_count', 0)}/{dq.get('req_count', 0)}) | "
        f"Ready: {dq.get('ready_count', 0)} | "
        f"Dropped: {dq.get('fail_summary', '-')}"
    )

    # Strategy info
    engine_name = s.get("engine", "Alpha-Ensemble Engine")
    s_str = (
        f"Engine: {engine_name} | "
        f"Inf Panel: {s.get('inf_panel', 0)} | "
        f"Trade Scope: {s.get('trade_scope', 0)}"
    )

    content = [
        "● [SYSTEM CONTEXT: INFRASTRUCTURE & DATA PREPARATION]",
        "SEP",
        f"Window:   {window_str}",
        f"Universe: {u_str}",
        f"Quality:  {dq_str}",
        f"Strategy: {s_str}",
    ]
    return _box(content)


def format_layer_header(layer: int, title: str) -> str:
    """섹션 상/하단에 굵은 선을 배치한 레이어 헤더 스타일."""
    border = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    return f"\n{border}\n● [LAYER {layer}: {title.upper()}]\n{border}"


def format_data_integrity_summary(
    total: int,
    passed: int,
    bars: int,
    nan_pct: float,
    zero_pct: float,
) -> str:
    """하단 얇은 선과 트리 구조를 적용한 데이터 무결성 감사 결과."""
    sep = "──────────────────────────────────────────────────────────────────────────────"
    
    if total == passed:
        lines = [
            "",
            "● [DATA-INTEGRITY AUDIT]",
            sep,
            f"  STATUS  : ✅ ALL {total} SYMBOLS PASSED",
            f"  METRICS : Total Bars: {bars:,}",
            f"  DETAIL  : [NaN: {nan_pct:.1f}%] [Zero/Neg: {zero_pct:.1f}%] [Range: PASS]",
            sep,
            ""
        ]
        return "\n".join(lines)
        
    lines = [
        "",
        "● [DATA-INTEGRITY AUDIT]",
        sep,
        f"  STATUS  : ❌ {total - passed}/{total} SYMBOLS FAILED",
        f"  METRICS : Total Bars: {bars:,}",
        f"  DETAIL  : [NaN: {nan_pct:.1f}%] [Zero/Neg: {zero_pct:.1f}%]",
        sep,
        ""
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.1 Window table
# ---------------------------------------------------------------------------

def format_window_table(w: LayeredWindow) -> str:
    """Format LayeredWindow as a pipe-table string (§9.1).

    Args:
        w: LayeredWindow instance containing all segment boundary dates.

    Returns:
        Multi-line pipe-table string with segment start/end/duration info.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """
    lines: list[str] = [
        "[WINDOW: TIERED] ------------------------------------",
        "| Segment      | Start      | End        | Duration  |",
        "| ------------ | ---------- | ---------- | --------- |",
        f"| Regime Floor | {_fmt_date(w.regime_floor):<10} | {'—':<10} | (hard LB) |",
        f"| L1 (SWF)    | {_fmt_date(w.l1_start):<10} | {_fmt_date(w.l2_start):<10} | 18 months |",
        f"| L2 (AWF)    | {_fmt_date(w.l2_start):<10} | {_fmt_date(w.holdout_start):<10} | 12 months |",
        f"| Holdout     | {_fmt_date(w.holdout_start):<10} | {_fmt_date(w.holdout_end):<10} | 6 months  |",
        "------------------------------------------------------",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.2 Layer 1 table
# ---------------------------------------------------------------------------

def format_layer1_table(
    r: Any,
    *,
    fold_details: list[dict[str, Any]] | None = None,
    per_symbol_top10: list[dict[str, Any]] | None = None,
) -> str:
    """Format Layer 1 (SWF-K) result as pipe-table string (§9.2).

    Args:
        r: Layer1Result-compatible object with fields:
            cs_ic_mean, cs_ic_tstat, cs_ic_fold_pass_ratio, breadth,
            n_valid, n_total, n_valid_strategies, panel_diversity, gate_passed.
        fold_details: Optional list of dicts with keys:
            fold, ic, breadth, n_valid, n_events, pass.
        per_symbol_top10: Optional list of dicts with keys:
            symbol, raw_mu, vol, t_stat, ic, valid.

    Returns:
        Multi-line pipe-table string for Layer 1 diagnostics.

    Time Complexity: O(n) where n = len(fold_details) + len(per_symbol_top10).
    Space Complexity: O(n).
    """
    n_valid: int = getattr(r, "n_valid", 0)
    n_total: int = getattr(r, "n_total", 0)
    n_trade_scope: int = getattr(r, "n_trade_scope", n_total)
    gate_str: str = _gate(r.gate_passed)
    cs_ic_mean: float = float(getattr(r, "cs_ic_mean", 0.0))
    cs_ic_tstat: float = float(getattr(r, "cs_ic_tstat", 0.0))
    cs_ic_fold_pass_ratio: float = float(getattr(r, "cs_ic_fold_pass_ratio", 0.0))
    decile_lift_bps: float = float(getattr(r, "decile_lift_bps", 0.0))
    n_valid_strategies: int = int(getattr(r, "n_valid_strategies", 0))
    panel_diversity: float = float(getattr(r, "panel_diversity", 0.0))
    strategy_panel = getattr(r, "strategy_panel", ())
    strategy_panel_count = len(strategy_panel) if isinstance(strategy_panel, tuple) else 0

    def _row(metric: str, value: str, gate: str, status: str) -> str:
        return f"| {metric:<20} | {value:<7} | {gate:<5} | {status:<11} |"

    lines: list[str] = [
        "[LAYER 1: SWF SIGNAL VALIDATION] --------------------",
        "| Metric               | Value   | Gate  | Status      |",
        "| -------------------- | ------- | ----- | ----------- |",
        _row("CS IC Mean", f"{cs_ic_mean:.3f}", "—", "—"),
        _row("CS IC t-stat", f"{cs_ic_tstat:.2f}", "—", "—"),
        _row("CS Fold Pass%", _pct(cs_ic_fold_pass_ratio), "≥60%", "—"),
        _row("Strategy Panel", f"{n_valid_strategies}/{strategy_panel_count}", "≥5", "—"),
        _row("Panel Diversity", f"{panel_diversity:.3f}", "≥30%", "—"),
        _row("Decile Lift", f"{decile_lift_bps:.2f}bps", "—", "—"),
        _row("Symbol Breadth", f"{getattr(r, 'breadth', 0.0):.3f}", ">0.3", "—"),
        _row("Valid Symbols/N", f"{n_valid}/{n_trade_scope}", "—", "—"),
        _row("L1 Gate", "—", "—", gate_str),
        "------------------------------------------------------",
    ]

    if fold_details:
        lines.append("")
        lines.append("[SWF FOLD DETAILS] ----------------------------------")
        lines.append("| Fold | IC      | Breadth | N Valid | N Events | Pass |")
        lines.append("| ---- | ------- | ------- | ------- | -------- | ---- |")
        for fd in fold_details:
            pass_str: str = "PASS" if fd.get("pass") else "FAIL"
            raw_ic = fd.get("ic")
            ic_str: str = f"{raw_ic:.3f}" if raw_ic is not None else "n/a"
            lines.append(
                f"| {fd['fold']:<4} | {ic_str:<7} | {fd['breadth']:>7.3f} | "
                f"{fd['n_valid']:>7} | {fd['n_events']:>8} | {pass_str:<4} |"
            )
        lines.append("------------------------------------------------------")

    if per_symbol_top10:
        lines.append("")
        lines.append("[PER-SYMBOL AGGREGATE] ------------------------------")
        lines.append("| Symbol       | Raw Mu    | Vol       | t-stat   | IC(avg)   | Valid |")
        lines.append("| ------------ | --------- | --------- | -------- | --------- | ----- |")
        for ps in per_symbol_top10:
            valid_str: str = "Y" if ps.get("valid") else "N"
            lines.append(
                f"| {ps['symbol']!s:<12} | {ps['raw_mu']:>9.3f} | {ps['vol']:>9.4f} | "
                f"{ps['t_stat']:>8.2f} | {ps['ic']:>9.3f} | {valid_str:<5} |"
            )
        lines.append("------------------------------------------------------")

    return "\n".join(lines)


def format_layer1_gate_table(report: Any) -> str:
    """하드 게이트 체크 항목을 미니멀리스트 리스트 형태로 변경."""
    checks = tuple(getattr(report, "checks", ()) or ())
    passed = bool(getattr(report, "passed", False))
    
    display_gate_map = {
        "fold_cov": "Time-Coverage",
        "match_ratio": "Signal-Quality",
        "sym_count": "Symbol-Breadth",
        "fold_ratio": "Stable-Folds",
        "probe_lcb_bps": "Min-Profit",
    }
    
    n_checks = len(checks)
    n_passed = sum(1 for c in checks if bool(getattr(c, "passed", False)))
    
    status_icon = "✅" if passed else "❌"
    status_text = "PASSED" if passed else "BLOCKED"
    
    sep = "──────────────────────────────────────────────────────────────────────────────"
    lines = [
        "",
        "● [LAYER 1 HARD GATE CHECKS]",
        sep,
        f"  STATUS  : {status_icon} {status_text} ({n_passed}/{n_checks} Passed)",
        ""
    ]
    
    for check in checks:
        check_passed = bool(getattr(check, "passed", False))
        icon = "✅" if check_passed else "❌"
        
        key_str = getattr(check, "key", "")
        display_key = display_gate_map.get(key_str, key_str)
        
        value = float(getattr(check, "value", 0.0))
        comparator = ">=" if getattr(check, "comparator", "ge") == "ge" else ">"
        threshold = f"{comparator}{getattr(check, 'threshold', 0.0):.3f}"
        
        blocker_suffix = "  ← [BLOCKER]" if not check_passed else ""
        
        lines.append(
            f"  {icon} [{display_key:<15}] : {value:>8.3f} (Target {threshold:<8}){blocker_suffix}"
        )
        
    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def format_layer1_outer_fold_table(reports: tuple[Any, ...]) -> str:
    """폴드별 준비 상태 진단을 트리 뷰 구조로 변경."""
    n_folds = len(reports)
    n_passed = sum(1 for r in reports if bool(getattr(r, "passed", False)))
    
    status_icon = "✅" if n_passed > 0 else "❌"
    status_text = "READY" if n_passed > 0 else "BLOCKED"
    
    sep = "──────────────────────────────────────────────────────────────────────────────"
    lines = [
        "",
        "● [LAYER 1 OUTER FOLD READINESS]",
        sep,
        f"  STATUS  : {status_icon} {status_text} ({n_passed}/{n_folds} Folds Ready)",
        ""
    ]
    
    for r in reports:
        passed = bool(getattr(r, "passed", False))
        icon = "✅" if passed else "❌"
        fold_id = int(getattr(r, "fold_id", 0))
        fit_end = int(getattr(r, "registry_source_end_idx", 0))
        oos_start = int(getattr(r, "outer_oos_start_idx", 0))
        
        ready_count = len(tuple(getattr(r, "ready_symbols", ()) or ()))
        times = int(getattr(r, "valid_opportunity_timestamp_count", 0))
        probe = float(getattr(r, "probe_bps", 0.0))

        lines.append(f"  [{icon}] Fold #{fold_id} (Fit: {fit_end} → OOS: {oos_start})")
        lines.append(f"       ├─ Symbols : {ready_count} symbols loaded")
        lines.append(f"       ├─ Events  : {times} unique events")
        lines.append(f"       └─ Quality : Edge: {probe:.2f} bps")
        
        if not passed:
            blockers = tuple(getattr(r, "blockers", ()) or ())
            if blockers:
                blocker_str = ", ".join(blockers)
                lines.append(f"       └─ BLOCKERS: {blocker_str}")
        lines.append("")
        
    lines.append(sep)
    return "\n".join(lines)


def format_layer1_deployment_registry_table(registry: Any) -> str:
    """Format deployment registry entries with enhanced visualization."""
    by_symbol = getattr(registry, "by_symbol", {}) or {}
    
    # Flatten all evidence items to sort and rank them
    all_entries = []
    for symbol in by_symbol:
        evidence_items = tuple(by_symbol.get(symbol, ()) or ())
        for ev in evidence_items:
            key = getattr(ev, "key", None)
            strategy_id = getattr(key, "strategy_id", "")
            context = getattr(key, "activation_context", "all")
            all_entries.append({
                "symbol": symbol,
                "strategy_id": strategy_id,
                "context": context,
                "edge": float(getattr(ev, "mean_incremental_bps", 0.0)),
                "tstat": float(getattr(ev, "bootstrap_tstat_incremental", 0.0)),
                "q_value": float(getattr(ev, "q_value", 0.0)),
                "effective_n": float(getattr(ev, "effective_n", 0.0)),
            })
    
    # Sort by t-stat descending for ranking
    all_entries.sort(key=lambda x: x["tstat"], reverse=True)
    
    lines = [
        "",  # Leading newline for separation
        "[L1 FINAL PROMOTION SUMMARY] 🚀",
        "--------------------------------------------------------------------------------------------",
        " RANK | SYMBOL   | STRATEGY (Family)              | EDGE(bps) | SIG(t-stat) | CONF(q) | STATUS",
        "--------------------------------------------------------------------------------------------",
    ]
    
    for i, entry in enumerate(all_entries, 1):
        # Strategy family extraction
        strat_parts = entry["strategy_id"].split(":")
        family = strat_parts[0] if len(strat_parts) > 1 else entry["strategy_id"]
        variant = strat_parts[1] if len(strat_parts) > 1 else ""
        
        # Include context if not 'all'
        ctx_suffix = f" [{entry['context']}]" if entry['context'] != "all" else ""
        strat_display = f"{family} ({variant}){ctx_suffix}" if variant else f"{family}{ctx_suffix}"
        
        # Star rating for t-stat
        t = entry["tstat"]
        if t >= 4.0:
            stars = "★★★★★"
        elif t >= 3.0:
            stars = "★★★★☆"
        elif t >= 2.0:
            stars = "★★★☆☆"
        elif t >= 1.0:
            stars = "★★☆☆☆"
        else:
            stars = "★☆☆☆☆"
            
        # Status based on q-value
        q = entry["q_value"]
        if q <= 0.15:
            status = "PROMOTED (Best Q)"
        elif q <= 0.30:
            status = "PROMOTED"
        elif q <= 0.70:
            status = "WATCH"
        else:
            status = "REJECTED"
            
        lines.append(
            f"  #{i:<2} | {entry['symbol']:<8} | {strat_display[:28]:<30} | "
            f"{entry['edge']:>+9.1f} | {stars} {t:>4.2f} |  {q:>5.3f}  | {status}"
        )
        
    lines.append("--------------------------------------------------------------------------------------------")
    if not all_entries:
        lines.append("  (No variants promoted to Layer 2)")
        
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.3 Layer 2 table
# ---------------------------------------------------------------------------

def format_layer2_table(
    r: Any,
    *,
    awf_folds: list[dict[str, Any]] | None = None,
    topk_selection: list[dict[str, Any]] | None = None,
) -> str:
    """Layer2 AWF 결과 로그 테이블.

    Args:
        r: Layer2Result (sharpe_hybrid, sharpe_baseline, mdd_hybrid,
           mdd_baseline, cagr_hybrid, cagr_baseline, mar_hybrid, mar_baseline,
           fold_pass_ratio, turnover, friction_pass_pct, gate_passed,
           blocker_reason 필드 필요).
        awf_folds: Optional fold 상세 (fold, sharpe, mdd, pass 키).
        topk_selection: Optional Top-K 선택 (rank, symbol, score, selected 키).

    Returns:
        Multi-line 파이프 테이블 문자열.

    Time Complexity: O(n) where n = len(awf_folds) + len(topk_selection).
    Space Complexity: O(n).
    """
    sharpe_h: float = getattr(r, "sharpe_hybrid", 0.0)
    sharpe_b: float = getattr(r, "sharpe_baseline", 0.0)
    mdd_h: float = getattr(r, "mdd_hybrid", 0.0)
    mdd_b: float = getattr(r, "mdd_baseline", 0.0)
    cagr_h: float = getattr(r, "cagr_hybrid", float("nan"))
    mar_h: float = getattr(r, "mar_hybrid", float("nan"))
    fold_pass: float = getattr(r, "fold_pass_ratio", 0.0)
    turnover: float = getattr(r, "turnover", 0.0)
    friction_pct: float = getattr(r, "friction_pass_pct", 0.0)
    psr_val: float = getattr(r, "psr_hybrid", float("nan"))
    dsr_val: float = getattr(r, "dsr_hybrid", float("nan"))
    gate_passed: bool = getattr(r, "gate_passed", False)
    blocker: str = getattr(r, "blocker_reason", "")

    # 가산식 uplift 게이트 (부호 무관): base+0.20
    uplift_val = sharpe_h - sharpe_b
    uplift_gate_val = 0.20

    def _f(v: float, fmt: str = ".3f") -> str:
        return "nan" if not math.isfinite(v) else format(v, fmt)

    def _status(passed: bool) -> str:
        return "✅" if passed else "❌"

    def _mar_str(mar_val: float, cagr_val: float) -> str:
        """MAR 표기: 음수 CAGR이면 n/a(loss) 반환."""
        if not math.isfinite(mar_val) or cagr_val < 0.0:
            return "n/a(loss)"
        return format(mar_val, ".3f")

    # Status determination (보수적 임계값)
    cagr_ok = cagr_h >= 0.15
    sharpe_ok = sharpe_h >= 1.0
    mar_ok = mar_h >= 1.0 and cagr_h >= 0.0
    mdd_ok = (mdd_h <= 0.20) and (mdd_h <= mdd_b)
    fold_ok = fold_pass >= 0.6
    psr_ok = math.isfinite(psr_val) and psr_val >= 0.90
    dsr_ok = math.isfinite(dsr_val) and dsr_val >= 0.95
    friction_ok = friction_pct >= 0.50
    uplift_ok = uplift_val >= uplift_gate_val

    # Style 3: Minimalist Grouped Summary
    sep = "──────────────────────────────────────────────────────────────────────────────"
    
    # Overall Status line
    status_icon = "✅" if gate_passed else "❌"
    result_str = _gate(gate_passed)
    if blocker:
        result_str += f" ({blocker})"
        
    # Categorize status for group icons
    return_ok = cagr_ok and sharpe_ok and mar_ok
    risk_ok = mdd_ok
    
    lines: list[str] = [
        "● [LAYER 2 PORTFOLIO SCORECARD]",
        sep,
        f"  STATUS  : {status_icon} {result_str}",
        "",
        (
            f"  {_status(return_ok)} [Return    ] "
            f"CAGR: {_f(cagr_h, '+.1%')} | Sharpe: {_f(sharpe_h)} | MAR: {_mar_str(mar_h, cagr_h)}"
        ),
        f"  {_status(risk_ok)} [Risk      ] MDD: {_pct(mdd_h)} | Turnover: {turnover:.3f}",
        f"  {_status(uplift_ok)} [Uplift    ] Sharpe Uplift: {_f(uplift_val, '+.2f')} (Target: >= +0.20)",
        (
            f"  {_status(fold_ok and psr_ok and dsr_ok and friction_ok)} [Robustness] "
            f"DSR: {_f(dsr_val)} (>= 0.95) | PSR: {_f(psr_val)} | Fold Pass: {_pct(fold_pass)}"
        ),
        sep,
    ]

    if awf_folds:
        lines.append("")
        lines.append("  [ FOLD DETAIL BREAKDOWN ]")
        lines.append("  ──────────────────────────────────────────────────────────────────────────")
        for i, af in enumerate(awf_folds):
            is_last = (i == len(awf_folds) - 1)
            prefix = "└─" if is_last else "├─"
            pass_icon = "✅" if af.get("pass") else "❌"
            sharpe_v = af["sharpe"]
            sharpe_str = "nan" if not math.isfinite(sharpe_v) else f"{sharpe_v:.3f}"
            mdd_v = af["mdd"]
            mdd_str = "nan%" if not math.isfinite(mdd_v) else _pct(mdd_v)
            
            line = (
                f"  {prefix} Fold #{af['fold']} : {pass_icon} Sharpe: {sharpe_str:>6} | "
                f"MDD: {mdd_str:>7} | Status: {'PASS' if af.get('pass') else 'FAIL'}"
            )
            lines.append(line)

    if topk_selection:
        lines.append("")
        lines.append(f"  {'Rank':<5} {'Symbol':<12} {'Score':>7} {'Sel':>4}")
        lines.append(f"  {'-'*5} {'-'*12} {'-'*7} {'-'*4}")
        for ts in topk_selection:
            sel_str: str = "Y" if ts.get("selected") else "N"
            lines.append(
                f"  {ts['rank']:<5} {ts['symbol']!s:<12} {ts['score']:>7.3f} {sel_str:>4}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.4 Layer 3 table
# ---------------------------------------------------------------------------

def format_layer3_table(
    r: Any,
    *,
    ho_start: str | None = None,
    ho_end: str | None = None,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
) -> str:
    """Layer3 Holdout 최종 검증 결과 로그 테이블.

    Args:
        r: Layer3Result (cagr, mdd, sharpe, mar, cagr_baseline, mdd_baseline,
           sharpe_baseline, mar_baseline, gate_passed, blocker_reason 필드 필요).
        ho_start: Legacy hold-out start date string (ISO format).
        ho_end: Legacy hold-out end date string (ISO format).
        holdout_start: Preferred hold-out start date string (ISO format).
        holdout_end: Preferred hold-out end date string (ISO format).

    Returns:
        Multi-line scorecard string for Layer 3 hold-out diagnostics.
    """
    start = holdout_start or ho_start or "—"
    end = holdout_end or ho_end or "—"

    def _resolve_status(result: Any) -> tuple[str, str]:
        blocker = str(
            getattr(result, "blocker_reason", "")
            or getattr(result, "blocker", "")
            or ""
        )
        raw_status = str(getattr(result, "status", "") or "").upper()
        if raw_status == "L3_ERROR":
            return "ERROR", blocker or "layer3_execution_error"
        if raw_status in {"ERROR", "BLOCKED", "PASS"}:
            return raw_status, blocker
        if bool(getattr(result, "error", False) or getattr(result, "errored", False)):
            return "ERROR", blocker or "layer3_execution_error"
        if bool(getattr(result, "gate_passed", False)):
            return "PASS", blocker
        if blocker:
            return "BLOCKED", blocker
        return "BLOCKED", ""

    def _f(v: float, fmt: str = ".3f") -> str:
        return "nan" if not math.isfinite(v) else format(v, fmt)

    def _status(passed: bool) -> str:
        return "✅" if passed else "❌"

    def _mar_str(mar_val: float, cagr_val: float) -> str:
        if not math.isfinite(mar_val) or cagr_val < 0.0:
            return "n/a(loss)"
        return format(mar_val, ".3f")

    status, blocker = _resolve_status(r)
    
    # Header & Initial summary
    sep_main = "──────────────────────────────────────────────────────────────────────────────"
    sep_sub = "  ──────────────────────────────────────────────────────────────────────────"
    
    lines: list[str] = [
        f"● [LAYER 3: HOLDOUT VALIDATION SCORECARD] ({start} ~ {end})",
        sep_main,
        "",
        "  [ OUT-OF-SAMPLE PERFORMANCE ]",
        sep_sub,
        f"  {'Metric':<20} {'Strategy':>12} {'( EW Bench )':>15}  {'Gate':>14} {'Status':>8}",
        sep_sub,
    ]

    if status == "ERROR":
        lines.extend([
            f"  {'STATUS':<20} [ {'ERROR':>8} ] ({'—':>11} )  {'—':>14} {'❌':>7}",
            f"  Error Summary   {blocker or 'layer3_execution_error'}",
            "",
            f"  >> FINAL RESULT : ❌ ERROR ({blocker or 'layer3_execution_error'})"
        ])
        return "\n".join(lines)

    # Metrics
    cagr_h = float(getattr(r, "cagr", 0.0))
    cagr_b = float(getattr(r, "cagr_baseline", 0.0))
    mdd_h = float(getattr(r, "mdd", 0.0))
    mdd_b = float(getattr(r, "mdd_baseline", 0.0))
    sharpe_h = float(getattr(r, "sharpe", 0.0))
    sharpe_b = float(getattr(r, "sharpe_baseline", 0.0))
    mar_h = float(getattr(r, "mar", 0.0))
    mar_b = float(getattr(r, "mar_baseline", 0.0))

    lines.append(
        f"  {'CAGR':<20} [ {_f(cagr_h, '+.1%'):>8} ] ({_f(cagr_b, '+.1%'):>11} )  {'>= Bench':>14} "
        f"{_status(cagr_h >= cagr_b):>7}"
    )
    lines.append(
        f"  {'MDD':<20} [ {_pct(mdd_h):>8} ] ({_pct(mdd_b):>11} )  {'<= Bench':>14} "
        f"{_status(mdd_h <= mdd_b):>7}"
    )
    lines.append(
        f"  {'Sharpe':<20} [ {_f(sharpe_h):>8} ] ({_f(sharpe_b):>11} )  {'>= Bench':>14} "
        f"{_status(sharpe_h >= sharpe_b):>7}"
    )
    lines.append(
        f"  {'MAR (CAGR/MDD)':<20} [ {_mar_str(mar_h, cagr_h):>8} ] ({_mar_str(mar_b, cagr_b):>11} )  {'>= Bench':>14} "
        f"{_status(mar_h >= mar_b):>7}"
    )

    # Optional Metrics (if present)
    if hasattr(r, "growth_lcb") or hasattr(r, "growth_lcb_baseline"):
        val_h = float(getattr(r, "growth_lcb", 0.0))
        val_b = float(getattr(r, "growth_lcb_baseline", 0.0))
        lines.append(
            f"  {'Growth LCB':<20} [ {_pct(val_h):>8} ] ({_pct(val_b):>11} )  {'>= Bench':>14} "
            f"{_status(val_h >= val_b):>7}"
        )
    
    if hasattr(r, "total_cost_bps"):
        val_h = float(getattr(r, "total_cost_bps", 0.0))
        val_b = float(getattr(r, "total_cost_bps_baseline", 0.0))
        # lower is better for cost
        lines.append(
            f"  {'Cost Drag':<20} [ {val_h/100:>+8.1%} ] ({val_b/100:>+11.1%} )  {'<= Bench':>14} "
            f"{_status(val_h <= val_b):>7}"
        )

    lines.append(sep_sub)
    lines.append("")
    
    final_icon = "✅" if status == "PASS" else "❌"
    final_status = f"{status} (Reason: {blocker})" if blocker else status
    lines.append(f"  >> FINAL RESULT : {final_icon} {final_status}")
    
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# §9.5 System status table
# ---------------------------------------------------------------------------

def format_system_status(
    l1: Any,
    l2: Any | None,
    l3: Any | None,
) -> str:
    """Format overall pipeline system status as pipe-table string (§9.5).

    Args:
        l1: Layer1Result-compatible object with ``gate_passed`` attribute.
        l2: Layer2Result-compatible object or None (None → SKIP).
        l3: Layer3Result-compatible object or None (None → SKIP).

    Returns:
        Multi-line pipe-table string showing per-layer status.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """
    def _layer_status(r: Any | None, *, skip_if_none: bool = True) -> tuple[str, str]:
        """Return (status, blocker) tuple for a layer result."""
        if r is None and skip_if_none:
            return "SKIP", "—"
        if r is None:
            return "SKIP", "—"
        raw_status = str(getattr(r, "status", "") or "").upper()
        blocker: str = str(
            getattr(r, "blocker_reason", "")
            or getattr(r, "blocker", "")
            or "—"
        )
        if raw_status == "L3_ERROR" or bool(getattr(r, "error", False) or getattr(r, "errored", False)):
            return "L3_ERROR", blocker
        passed: bool = getattr(r, "gate_passed", False)
        if passed:
            return "PASS", "—"
        return "BLOCKED", blocker if blocker != "—" else "gate_passed=False"

    l1_status, l1_blocker = _layer_status(l1, skip_if_none=False)
    l2_status, l2_blocker = _layer_status(l2)
    l3_status, l3_blocker = _layer_status(l3)

    lines: list[str] = [
        "[SYSTEM STATUS] ------------------------------------",
        "| Layer   | Status  | Blocker (if any)            |",
        "| ------- | ------- | --------------------------- |",
        f"| Layer 1 | {l1_status:<7} | {l1_blocker:<27} |",
        f"| Layer 2 | {l2_status:<7} | {l2_blocker:<27} |",
        f"| Layer 3 | {l3_status:<7} | {l3_blocker:<27} |",
        "-----------------------------------------------------",
    ]
    return "\n".join(lines)
