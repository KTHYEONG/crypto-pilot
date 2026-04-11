from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class CheckRecord:
    check_id: str
    label: str
    observed: float
    threshold: float
    passed: bool


@dataclass
class GoNoGoResult:
    passed: bool
    details: Dict[str, bool]
    summary: str
    checks: List[CheckRecord] = field(default_factory=list)
    advisory: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolGateRow:
    symbol: str
    net_cagr_pct: float
    max_mdd_pct: float
    tail_ratio: float
    win_rate_pct: float
    trade_count: int


@dataclass(frozen=True)
class FinalDeploymentReportInput:
    """Structured inputs for the final 3-gate deployment log."""

    gate1_sqn: float
    gate1_path_sortino: float
    gate1_tail_ratio: float
    gate1_p10_gmgr: float
    gate1_max_ui: float
    gate1_psr: float
    gate1_dsr: float
    cpcv_mean_path_return_pct: float
    cpcv_worst_segment_mdd_pct: float
    sqn_target: float
    path_sortino_target: float
    tail_ratio_target: float
    psr_target: float
    dsr_target: float
    moic: float
    initial_capital_krw: float
    oos_net_cagr_pct: float
    oos_mdd_pct: float
    hw_recovery_days: float
    alpha_decay_pct: float
    oos_cagr_target_pct: float
    oos_mdd_limit_pct: float
    hw_recovery_max_days: float
    alpha_decay_floor_pct: float
    oos_cvar_pct: float
    cvar_limit_pct: float
    terminal_wealth_ratio: float
    tw_target: float
    oos_total_trades: int
    oos_pf: float
    pf_target: float
    oos_calmar: float
    calmar_target: float
    oos_win_rate_pct: float
    symbol_rows: Sequence[SymbolGateRow] = field(default_factory=list)
    loso_warning: str = ""
    hard_passed: int = 0
    hard_total: int = 0
    final_decision_go: bool = False
    pbo: float = float("nan")
    spearman_rho: float = float("nan")
    pbo_gate_passed: bool = True
    pbo_hard_gate: bool = False
    pbo_n_paths: int = 0
    multi_window_passed: bool = True
    multi_window_summary: str = ""
    regime_diagnostic_block: str = ""


def run_pbo_gate(
    *,
    pbo: float,
    pbo_max: float,
    hard: bool,
) -> GoNoGoResult:
    """CPCV IS vs OOS rank PBO proxy; advisory unless hard mode."""
    ok = float(pbo) <= float(pbo_max)
    passed = bool(ok or not hard)
    lines = [
        "[Gate 1.5 — CPCV PBO (train vs test path score consistency)]",
        f"  - PBO <= {pbo_max:.2f} | {'PASS' if ok else 'FAIL'} | obs={float(pbo):.4f} | mode={'HARD' if hard else 'ADVISORY'}",
        "-" * 55,
        f"  FINAL: {'GO' if passed else 'NO-GO'}",
    ]
    return GoNoGoResult(
        passed=passed,
        details={"pbo": ok},
        summary="\n".join(lines),
        checks=[
            CheckRecord(
                "pbo",
                "PBO vs max",
                float(pbo),
                float(pbo_max),
                ok,
            )
        ],
    )


def run_multi_window_oos_gate(
    *,
    window_results: List[Dict[str, Any]],
    min_positive_windows: int,
    min_median_cagr_pct: float,
    max_worst_mdd_pct: float,
) -> GoNoGoResult:
    """Anchored multi-window OOS consistency (hard gate when enabled from config)."""
    if not window_results:
        return GoNoGoResult(
            passed=False,
            details={"windows": False},
            summary="[Gate 3.5 — Multi-Window OOS]\n  FAIL: no window results",
            checks=[],
        )
    cagrs = [float(w.get("cagr_pct", -100.0)) for w in window_results]
    pos = int(sum(1 for c in cagrs if c > 0.0))
    med_c = float(median(cagrs)) if cagrs else -100.0
    worst_mdd = float(max(abs(float(w.get("mdd_pct", 0.0))) for w in window_results))
    ok_pos = pos >= int(min_positive_windows)
    ok_med = med_c >= float(min_median_cagr_pct)
    ok_mdd = worst_mdd <= float(max_worst_mdd_pct)
    passed = bool(ok_pos and ok_med and ok_mdd)
    lines = [
        "[Gate 3.5 — Multi-Window OOS Consistency]",
        f"  - Positive windows >= {min_positive_windows} | {'PASS' if ok_pos else 'FAIL'} | obs={pos}/{len(window_results)}",
        f"  - Median window CAGR >= {min_median_cagr_pct}% | {'PASS' if ok_med else 'FAIL'} | obs={med_c:.2f}%",
        f"  - Worst-window |MDD| <= {max_worst_mdd_pct}% | {'PASS' if ok_mdd else 'FAIL'} | obs={worst_mdd:.2f}%",
        "-" * 55,
        f"  FINAL: {'GO' if passed else 'NO-GO'}",
    ]
    return GoNoGoResult(
        passed=passed,
        details={"positive_windows": ok_pos, "median_cagr": ok_med, "worst_mdd": ok_mdd},
        summary="\n".join(lines),
        checks=[],
    )


def format_regime_oos_diagnostic_block(
    regime_metrics: Dict[str, Dict[str, float]],
    stress_mdd_warn_pct: float,
) -> str:
    """Advisory TIER 4 text block for deployment log."""
    lines: List[str] = [
        "=" * 71,
        " [TIER 4. REGIME ROBUSTNESS DIAGNOSTIC (advisory)]",
        "=" * 71,
    ]
    order = ("risk_on", "cautious", "stress")
    labels = {
        "risk_on": "Risk-On bars (mult > 0.5)",
        "cautious": "Cautious bars (0 < mult <= 0.5)",
        "stress": "Stress bars (mult <= 0)",
    }
    for key in order:
        m = regime_metrics.get(key, {})
        n = int(m.get("bar_count", 0.0))
        rp = float(m.get("return_pct", 0.0))
        mdd = float(m.get("mdd_pct", 0.0))
        lines.append(f"  - {labels[key]}: N={n} | return: {rp:.2f}% | MDD: {mdd:.2f}%")
    stress_mdd = float(regime_metrics.get("stress", {}).get("mdd_pct", 0.0))
    if stress_mdd > float(stress_mdd_warn_pct):
        lines.append(
            f"  ⚠ WARNING: Stress-regime MDD ({stress_mdd:.2f}%) exceeds advisory threshold ({stress_mdd_warn_pct}%)"
        )
    lines.append("=" * 71)
    return "\n".join(lines)


def run_go_nogo_check(
    cv_fold_scores: List[float],
    holdout_score: float,
    oos_romad_scores: List[float],
    max_mdd_pct: float,
    tail_ratio: float,
    long_count: int,
    tf: str = "4h",
    *,
    mdd_limit_pct: float = 45.0,
    tail_ratio_min: float = 2.0,
) -> GoNoGoResult:
    """Per-symbol OOS diagnostic gate (tail ratio floor, MDD cap, min trades)."""
    total_trades = long_count
    oos_cagr: float = float(oos_romad_scores[0]) if oos_romad_scores else -100.0
    growth_pass: bool = oos_cagr > 0.0
    mdd_pass: bool = abs(max_mdd_pct) <= mdd_limit_pct
    tr_pass: bool = tail_ratio >= tail_ratio_min
    min_trades_req = 5
    trades_pass: bool = total_trades >= min_trades_req

    details: Dict[str, bool] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": growth_pass,
        f"2. Volatility Drag (MDD <= {mdd_limit_pct}%)": mdd_pass,
        f"3. Tail Ratio (>= {tail_ratio_min})": tr_pass,
        f"4. Stat Edge (Trades >= {min_trades_req})": trades_pass,
    }
    all_passed = all(details.values())
    summary_lines: List[str] = ["[Spot Holdout Safety]"]
    metric_values: Dict[str, str] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": f"CAGR: {oos_cagr:.2f}%",
        f"2. Volatility Drag (MDD <= {mdd_limit_pct}%)": f"MDD: {abs(max_mdd_pct):.2f}%",
        f"3. Tail Ratio (>= {tail_ratio_min})": f"Tail: {tail_ratio:.2f}",
        f"4. Stat Edge (Trades >= {min_trades_req})": f"N: {total_trades}",
    }
    req_met = sum(1 for v in details.values() if v)
    total_req = len(details)
    for k, v in details.items():
        status = "PASS" if v else "FAIL"
        val_str = metric_values.get(k, "")
        summary_lines.append(f"  - {k:<40}: {status:<5} ({val_str})")
    summary_lines.append("-" * 55)
    final_status = "GO" if all_passed else f"NO-GO (Passed {req_met}/{total_req})"
    summary_lines.append(f"  FINAL: {final_status}")
    return GoNoGoResult(passed=all_passed, details=details, summary="\n".join(summary_lines))


def run_portfolio_discovery_veto(
    *,
    psr: float,
    dsr: float,
    p10_gmgr: float = 0.0,
    psr_min: float = 0.5,
    dsr_min: float = -1.0,
) -> GoNoGoResult:
    """
    Discovery veto: PSR hard; DSR soft floor; P10 GMGR >= -0.001 (noise tolerance).
    """
    psr_ok = psr >= psr_min
    dsr_ok = dsr >= dsr_min
    gmgr_ok = p10_gmgr >= -0.001
    details: Dict[str, bool] = {
        "psr_hard": psr_ok,
        "dsr_soft": dsr_ok,
        "p10_gmgr_positive": gmgr_ok,
    }
    passed = bool(psr_ok and dsr_ok and gmgr_ok)
    dsr_label = "soft floor" if dsr_min < 0.0 else "hard"
    summary_lines = [
        "[Portfolio Discovery Veto]",
        f"  PSR (hard): {psr:.4f} vs {psr_min} -> {'PASS' if psr_ok else 'FAIL'}",
        f"  DSR ({dsr_label}): {dsr:.4f} vs {dsr_min} -> {'PASS' if dsr_ok else 'FAIL'}",
        f"  P10 GMGR (>= -0.001): {p10_gmgr:.6f} -> {'PASS' if gmgr_ok else 'FAIL'}",
    ]
    return GoNoGoResult(
        passed=passed,
        details=details,
        summary="\n".join(summary_lines),
        checks=[],
        advisory={"dsr": dsr},
    )


def run_holdout_portfolio_trade_floor(
    *,
    portfolio_long_trades: int,
    min_portfolio_trades: int,
) -> GoNoGoResult:
    """Minimum shared-cash holdout trade count (blocking)."""
    ok = portfolio_long_trades >= min_portfolio_trades
    lines = [
        "[Spot Holdout Portfolio Trade Floor]",
        (
            f"  - Portfolio long trades >= {min_portfolio_trades} | "
            f"{'PASS' if ok else 'FAIL'} | observed={portfolio_long_trades} | need >= {min_portfolio_trades}"
        ),
        "-" * 55,
        f"  FINAL: {'GO' if ok else 'NO-GO'}",
    ]
    return GoNoGoResult(
        passed=ok,
        details={"trade_floor": ok},
        summary="\n".join(lines),
        checks=[
            CheckRecord(
                "trades",
                "Portfolio trades floor",
                float(portfolio_long_trades),
                float(min_portfolio_trades),
                ok,
            )
        ],
    )


def run_holdout_portfolio_shared_cash(
    *,
    portfolio_cagr_pct: float,
    portfolio_mdd_pct: float,
    portfolio_cvar_pct: float,
    portfolio_tail_ratio: float,
    min_path_terminal_wealth_ratio: float,
    portfolio_profit_factor: float,
    portfolio_calmar_ratio: float,
    max_cvar_pct: float,
    mdd_limit_pct: float = 20.0,
    tw_need: float = 1.0,
    tail_ratio_min: float = 1.20,
    cagr_min_pct: float = 35.0,
    oos_dd_days: float = 0.0,
    hw_recovery_days_max: float = 180.0,
    is_cagr_pct: float = 0.0,
    alpha_decay_floor_pct: float = -25.0,
    pf_min: float = 1.3,
    calmar_min: float = 1.5,
) -> GoNoGoResult:
    """
    Shared-cash holdout: terminal wealth, CAGR floor, MDD, CVaR, tail ratio,
    PF, Calmar, HWM recovery, alpha decay.
    """
    c_tw = min_path_terminal_wealth_ratio > tw_need
    c_cagr = portfolio_cagr_pct >= cagr_min_pct
    c_mdd = abs(portfolio_mdd_pct) <= mdd_limit_pct
    c_cvar = portfolio_cvar_pct <= max_cvar_pct
    c_tr = portfolio_tail_ratio >= tail_ratio_min
    c_hw = oos_dd_days <= hw_recovery_days_max
    c_pf = portfolio_profit_factor >= pf_min
    c_calmar = portfolio_calmar_ratio >= calmar_min

    # Alpha decay
    if is_cagr_pct <= 0.0:
        # IS period was losing or flat: ratio-based decay is undefined.
        # CAGR gate already enforces profitability; alpha decay is N/A.
        alpha_decay_pct = float("nan")
        c_alpha = True
    else:
        alpha_decay_pct = ((portfolio_cagr_pct / is_cagr_pct) - 1.0) * 100.0
        c_alpha = alpha_decay_pct >= alpha_decay_floor_pct

    passed = all([c_tw, c_cagr, c_mdd, c_cvar, c_tr, c_hw, c_alpha, c_pf, c_calmar])

    lines = [
        "[Spot Holdout Portfolio (shared-cash)]",
        f"  - Terminal wealth > {tw_need} | {'PASS' if c_tw else 'FAIL'} | obs={min_path_terminal_wealth_ratio:.4f}",
        f"  - CAGR >= {cagr_min_pct}% | {'PASS' if c_cagr else 'FAIL'} | obs={portfolio_cagr_pct:.2f}%",
        f"  - MDD <= {mdd_limit_pct}% | {'PASS' if c_mdd else 'FAIL'} | obs={abs(portfolio_mdd_pct):.2f}%",
        f"  - CVaR <= {max_cvar_pct}% | {'PASS' if c_cvar else 'FAIL'} | obs={portfolio_cvar_pct:.2f}%",
        f"  - Tail Ratio >= {tail_ratio_min} | {'PASS' if c_tr else 'FAIL'} | obs={portfolio_tail_ratio:.4f}",
        f"  - Profit Factor >= {pf_min} | {'PASS' if c_pf else 'FAIL'} | obs={portfolio_profit_factor:.4f}",
        f"  - Calmar Ratio >= {calmar_min} | {'PASS' if c_calmar else 'FAIL'} | obs={portfolio_calmar_ratio:.4f}",
        f"  - HWM recovery <= {hw_recovery_days_max}d | {'PASS' if c_hw else 'FAIL'} | obs={oos_dd_days:.1f}d",
        (
            f"  - Alpha decay >= {alpha_decay_floor_pct}% | {'PASS' if c_alpha else 'FAIL'} | "
            f"obs={('N/A (IS<=0)' if not math.isfinite(alpha_decay_pct) else f'{alpha_decay_pct:.1f}%')} "
            f"(ref CAGR={is_cagr_pct:.1f}% for decay ratio)"
        ),
        "-" * 55,
        f"  FINAL: {'GO' if passed else 'NO-GO'}",
    ]

    details = {
        "tw": c_tw,
        "cagr": c_cagr,
        "mdd": c_mdd,
        "cvar": c_cvar,
        "tail_ratio": c_tr,
        "hw_recovery": c_hw,
        "alpha_decay": c_alpha,
        "pf": c_pf,
        "calmar": c_calmar,
    }

    # We maintain CheckRecord for backward compatibility with existing reporting if needed
    checks = [
        CheckRecord("tw", "Wealth", min_path_terminal_wealth_ratio, tw_need, c_tw),
        CheckRecord("cagr", "CAGR", portfolio_cagr_pct, cagr_min_pct, c_cagr),
        CheckRecord("mdd", "MDD", abs(portfolio_mdd_pct), mdd_limit_pct, c_mdd),
        CheckRecord("pf", "ProfitFactor", portfolio_profit_factor, pf_min, c_pf),
        CheckRecord("calmar", "Calmar", portfolio_calmar_ratio, calmar_min, c_calmar),
    ]

    return GoNoGoResult(
        passed=passed,
        details=details,
        summary="\n".join(lines),
        checks=checks,
        advisory={"alpha_decay_pct": alpha_decay_pct},
    )


def run_go_nogo_holdout_portfolio_growth(
    *,
    portfolio_cagr_pct: float,
    portfolio_mdd_pct: float,
    portfolio_cvar_pct: float,
    portfolio_tail_ratio: float,
    portfolio_long_trades: int,
    min_path_terminal_wealth_ratio: float,
    portfolio_profit_factor: float,
    portfolio_calmar_ratio: float,
    min_portfolio_trades: int,
    max_cvar_pct: float,
    tail_ratio_min: float = 1.20,
    cagr_min_pct: float = 35.0,
    mdd_limit_pct: float = 20.0,
    oos_dd_days: float = 0.0,
    hw_recovery_days_max: float = 180.0,
    is_cagr_pct: float = 0.0,
    alpha_decay_floor_pct: float = -25.0,
    pf_min: float = 1.3,
    calmar_min: float = 1.5,
) -> GoNoGoResult:
    """
    Backward-compatible: trade floor AND shared-cash screen; all must pass.
    """
    tfloor = run_holdout_portfolio_trade_floor(
        portfolio_long_trades=portfolio_long_trades,
        min_portfolio_trades=min_portfolio_trades,
    )
    scash = run_holdout_portfolio_shared_cash(
        portfolio_cagr_pct=portfolio_cagr_pct,
        portfolio_mdd_pct=portfolio_mdd_pct,
        portfolio_cvar_pct=portfolio_cvar_pct,
        portfolio_tail_ratio=portfolio_tail_ratio,
        min_path_terminal_wealth_ratio=min_path_terminal_wealth_ratio,
        portfolio_profit_factor=portfolio_profit_factor,
        portfolio_calmar_ratio=portfolio_calmar_ratio,
        max_cvar_pct=max_cvar_pct,
        tail_ratio_min=tail_ratio_min,
        cagr_min_pct=cagr_min_pct,
        mdd_limit_pct=mdd_limit_pct,
        oos_dd_days=oos_dd_days,
        hw_recovery_days_max=hw_recovery_days_max,
        is_cagr_pct=is_cagr_pct,
        alpha_decay_floor_pct=alpha_decay_floor_pct,
        pf_min=pf_min,
        calmar_min=calmar_min,
    )
    passed = bool(tfloor.passed and scash.passed)
    summary = tfloor.summary + "\n\n" + scash.summary
    return GoNoGoResult(
        passed=passed,
        details={**tfloor.details, **scash.details},
        summary=summary,
        checks=[*tfloor.checks, *scash.checks],
    )


def _fmt_pass_info(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


# PART 3 markdown-style columns: must match separator dash counts exactly.
_PART3_COL_WIDTHS: Tuple[int, int, int, int, int] = (11, 18, 9, 10, 8)


def _part3_symbol_table_lines(rows: Sequence[SymbolGateRow]) -> List[str]:
    """Fixed-width symbol table: header, rule, and one row per symbol (aligned columns)."""
    w = _PART3_COL_WIDTHS
    header = (
        "  | "
        + f"{'Symbol':<{w[0]}} | "
        + f"{'PnL contrib ann%':^{w[1]}} | "
        + f"{'Max MDD':^{w[2]}} | "
        + f"{'Win Rate':^{w[3]}} | "
        + f"{'Trades':^{w[4]}} |"
    )
    rule = (
        "  | "
        + "-" * w[0]
        + " | "
        + "-" * w[1]
        + " | "
        + "-" * w[2]
        + " | "
        + "-" * w[3]
        + " | "
        + "-" * w[4]
        + " |"
    )
    out: List[str] = [header, rule]
    for row in rows:
        sym = row.symbol if len(row.symbol) <= w[0] else row.symbol[: w[0] - 2] + ".."
        pnl = f"{row.net_cagr_pct:+.1f}%"
        mdd = f"{row.max_mdd_pct:.1f}%"
        wr = f"{row.win_rate_pct:.1f}%"
        tr = f"{int(row.trade_count)} ⚠" if int(row.trade_count) < 30 else str(int(row.trade_count))
        out.append(
            "  | "
            + f"{sym:<{w[0]}} | "
            + f"{pnl:^{w[1]}} | "
            + f"{mdd:^{w[2]}} | "
            + f"{wr:^{w[3]}} | "
            + f"{tr:^{w[4]}} |"
        )
    return out


def run_final_deployment_report(ctx: FinalDeploymentReportInput) -> str:
    """Build the Spot Strategy deployment report: TIER 1 (IS), TIER 2 (OOS Risk), TIER 3 (OOS Profit)."""
    # TIER 1 Gates: Statistical Discovery Rigor
    sqn_ok = ctx.gate1_sqn >= ctx.sqn_target
    ps_ok = ctx.gate1_path_sortino >= ctx.path_sortino_target
    g1_tr_ok = ctx.gate1_tail_ratio >= ctx.tail_ratio_target
    gmgr_ok = ctx.gate1_p10_gmgr >= -0.001
    psr_ok = ctx.gate1_psr >= ctx.psr_target
    dsr_ok = ctx.gate1_dsr >= ctx.dsr_target

    # TIER 2 Gates: OOS Risk Management (Shared-Cash)
    oos_mdd_ok = abs(ctx.oos_mdd_pct) <= ctx.oos_mdd_limit_pct
    cvar_ok = ctx.oos_cvar_pct <= ctx.cvar_limit_pct
    hw_ok = ctx.hw_recovery_days <= ctx.hw_recovery_max_days
    calmar_ok = ctx.oos_calmar >= ctx.calmar_target

    # TIER 3 Gates: OOS Profitability & Statistical Quality
    oos_cagr_ok = ctx.oos_net_cagr_pct >= ctx.oos_cagr_target_pct
    pf_ok = ctx.oos_pf >= ctx.pf_target
    ad_ok = ctx.alpha_decay_pct >= ctx.alpha_decay_floor_pct
    tw_ok = ctx.terminal_wealth_ratio > ctx.tw_target

    # Business Impact
    final_capital = ctx.initial_capital_krw * ctx.moic
    profit_pct = (ctx.moic - 1.0) * 100.0

    pbo_disp = f"{ctx.pbo:.4f}" if math.isfinite(ctx.pbo) else "N/A"
    rho_disp = f"{ctx.spearman_rho:.4f}" if math.isfinite(ctx.spearman_rho) else "N/A"

    lines: List[str] = [
        "=" * 71,
        " [TIER 1. CPCV STATISTICAL EDGE RIGOR]",
        "=" * 71,
        f"  - System Quality Number (SQN) : {ctx.gate1_sqn:.2f}   {_fmt_pass_info(sqn_ok)} (Min: {ctx.sqn_target})",
        f"  - Path Sortino Ratio          : {ctx.gate1_path_sortino:.2f}   {_fmt_pass_info(ps_ok)} (Min: {ctx.path_sortino_target})",
        f"  - Path Tail Ratio (Discovery) : {ctx.gate1_tail_ratio:.2f}   {_fmt_pass_info(g1_tr_ok)} (Min: {ctx.tail_ratio_target})",
        f"  - Prob. Sharpe Ratio (PSR)    : {ctx.gate1_psr:.4f}   {_fmt_pass_info(psr_ok)} (Min: {ctx.psr_target})",
        f"  - Deflated Sharpe Ratio (DSR) : {ctx.gate1_dsr:.4f}   {_fmt_pass_info(dsr_ok)} (Min: {ctx.dsr_target})",
        f"  - P10 GMGR (Worst Path Grow)  : {ctx.gate1_p10_gmgr:.6f}   {_fmt_pass_info(gmgr_ok)} (Target: >= -0.001)",
        f"  - CPCV Mean Path Return       : {ctx.cpcv_mean_path_return_pct:.1f}%",
        f"  - CPCV Worst Segment MDD      : {ctx.cpcv_worst_segment_mdd_pct:.1f}%",
        "",
        f"  - PBO (IS vs OOS path ranks, n_paths={ctx.pbo_n_paths})  : {pbo_disp}   "
        f"{_fmt_pass_info(ctx.pbo_gate_passed)} "
        f"({'HARD' if ctx.pbo_hard_gate else 'ADVISORY'})",
        f"  - Spearman rho (IS vs OOS)    : {rho_disp}",
        "",
        "=" * 71,
        " [TIER 2. OOS ABSOLUTE RISK HARD GATES: 4H SPOT]",
        "=" * 71,
        f"  - Maximum Pain (MDD Limit)    : {ctx.oos_mdd_pct:.1f}%   {_fmt_pass_info(oos_mdd_ok)} (Limit: {ctx.oos_mdd_limit_pct}%)",
        f"  - Portfolio CVaR(5%) Loss     : {ctx.oos_cvar_pct:.2f}%   {_fmt_pass_info(cvar_ok)} (Limit: {ctx.cvar_limit_pct}%)",
        f"  - Recovery Time (Max UD)      : {ctx.hw_recovery_days:.1f}d   {_fmt_pass_info(hw_ok)} (Limit: {ctx.hw_recovery_max_days}d)",
        f"  - OOS Calmar Ratio (Grow/Risk): {ctx.oos_calmar:.2f}   {_fmt_pass_info(calmar_ok)} (Min: {ctx.calmar_target})",
        "",
        "=" * 71,
        " [TIER 3. OOS PROFITABILITY & ROBUSTNESS]",
        "=" * 71,
        f"  - Annualized Return (CAGR)    : {ctx.oos_net_cagr_pct:.1f}%   {_fmt_pass_info(oos_cagr_ok)} (Min: {ctx.oos_cagr_target_pct}%)",
        f"  - Trade Profit Factor         : {ctx.oos_pf:.2f}   {_fmt_pass_info(pf_ok)} (Min: {ctx.pf_target})",
        f"  - Alpha Decay (Stability)     : {ctx.alpha_decay_pct:.1f}%   {_fmt_pass_info(ad_ok)} (Limit: {ctx.alpha_decay_floor_pct}%)",
        f"  - Terminal Wealth Ratio       : {ctx.terminal_wealth_ratio:.3f}   {_fmt_pass_info(tw_ok)} (Min: {ctx.tw_target})",
        f"  - OOS Win Rate (INFO)         : {ctx.oos_win_rate_pct:.1f}%",
        "",
    ]
    if ctx.multi_window_summary:
        lines.append(ctx.multi_window_summary)
        lines.append("")
    lines.extend(
        [
            "=" * 71,
            " [PART 3. SYMBOL MICROSTRUCTURE & FINAL VERDICT]",
            "=" * 71,
            "▶ Portfolio Composition (Shared Cash)",
            f"  - Capital: ₩{ctx.initial_capital_krw:,.0f} -> ₩{final_capital:,.0f} ({profit_pct:+.1f}%)",
            f"  - Total Trades: {ctx.oos_total_trades} | Concentration: {ctx.loso_warning}",
            "",
        ]
    )
    lines.extend(_part3_symbol_table_lines(ctx.symbol_rows))

    lines.extend(
        [
            "  ※ Symbol PnL contrib ann%: shared-cash trade PnL vs initial, annualized (not standalone engine CAGR).",
            "",
            f"▶ Final Verdict : {'[GO - DEPLOYABLE]' if ctx.final_decision_go else '[NO-GO - REFINEMENT NEEDED]'}",
            f"  Compliance Score: {ctx.hard_passed}/{ctx.hard_total} Critical Gates Passed",
        ]
    )

    if ctx.regime_diagnostic_block:
        lines.append("")
        lines.append(ctx.regime_diagnostic_block)

    if not ctx.final_decision_go:
        lines.append("\n  ※ 주요 결격 사유 (Critical Failures):")
        if not psr_ok:
            lines.append(
                f"    - TIER1: PSR 점수({ctx.gate1_psr:.4f})가 기준({ctx.psr_target}) 미달"
            )
        if not dsr_ok:
            lines.append(
                f"    - TIER1: DSR 점수({ctx.gate1_dsr:.4f})가 기준({ctx.dsr_target}) 미달"
            )
        if not oos_mdd_ok:
            lines.append(
                f"    - TIER2: OOS MDD({abs(ctx.oos_mdd_pct):.1f}%)가 제한({ctx.oos_mdd_limit_pct}%) 초과"
            )
        if not calmar_ok:
            lines.append(
                f"    - TIER2: Calmar Ratio({ctx.oos_calmar:.2f})가 기준({ctx.calmar_target}) 미달"
            )
        if not oos_cagr_ok:
            lines.append(
                f"    - TIER3: OOS CAGR({ctx.oos_net_cagr_pct:.1f}%)이 목표({ctx.oos_cagr_target_pct}%) 미달"
            )
        if not pf_ok:
            lines.append(
                f"    - TIER3: Profit Factor({ctx.oos_pf:.2f})가 기준({ctx.pf_target}) 미달"
            )
        if not ad_ok:
            lines.append(
                f"    - TIER3: Alpha Decay({ctx.alpha_decay_pct:.1f}%)가 허용치({ctx.alpha_decay_floor_pct}%) 초과"
            )

    lines.append("=" * 71)
    return "\n".join(lines)
