from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


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
    Discovery veto: PSR hard; DSR soft floor; P10 GMGR must be > 0.
    """
    psr_ok = psr >= psr_min
    dsr_ok = dsr >= dsr_min
    gmgr_ok = p10_gmgr > 0.0
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
        f"  P10 GMGR (>0): {p10_gmgr:.6f} -> {'PASS' if gmgr_ok else 'FAIL'}",
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
    if abs(is_cagr_pct) < 1e-6:
        alpha_decay_pct = 0.0
        c_alpha = portfolio_cagr_pct >= alpha_decay_floor_pct # Fallback
    else:
        # Simple percentage decay relative to IS mean
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
        f"  - Alpha decay >= {alpha_decay_floor_pct}% | {'PASS' if c_alpha else 'FAIL'} | obs={alpha_decay_pct:.1f}% (IS={is_cagr_pct:.1f}%)",
        "-" * 55,
        f"  FINAL: {'GO' if passed else 'NO-GO'}",
    ]
    
    details = {
        "tw": c_tw, "cagr": c_cagr, "mdd": c_mdd, "cvar": c_cvar,
        "tail_ratio": c_tr, "hw_recovery": c_hw, "alpha_decay": c_alpha,
        "pf": c_pf, "calmar": c_calmar,
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


def run_final_deployment_report(ctx: FinalDeploymentReportInput) -> str:
    """Build the Spot Strategy deployment report: TIER 1 (IS), TIER 2 (OOS Risk), TIER 3 (OOS Profit)."""
    # TIER 1 Gates: Statistical Discovery Rigor
    sqn_ok = ctx.gate1_sqn >= ctx.sqn_target
    ps_ok = ctx.gate1_path_sortino >= ctx.path_sortino_target
    g1_tr_ok = ctx.gate1_tail_ratio >= ctx.tail_ratio_target
    gmgr_ok = ctx.gate1_p10_gmgr > 0.0
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

    lines: List[str] = [
        "=" * 71,
        " [TIER 1. CPCV STATISTICAL EDGE RIGOR]",
        "=" * 71,
        f"  - System Quality Number (SQN) : {ctx.gate1_sqn:.2f}   {_fmt_pass_info(sqn_ok)} (Min: {ctx.sqn_target})",
        f"  - Path Sortino Ratio          : {ctx.gate1_path_sortino:.2f}   {_fmt_pass_info(ps_ok)} (Min: {ctx.path_sortino_target})",
        f"  - Path Tail Ratio (Discovery) : {ctx.gate1_tail_ratio:.2f}   {_fmt_pass_info(g1_tr_ok)} (Min: {ctx.tail_ratio_target})",
        f"  - Prob. Sharpe Ratio (PSR)    : {ctx.gate1_psr:.4f}   {_fmt_pass_info(psr_ok)} (Min: {ctx.psr_target})",
        f"  - Deflated Sharpe Ratio (DSR) : {ctx.gate1_dsr:.4f}   {_fmt_pass_info(dsr_ok)} (Min: {ctx.dsr_target})",
        f"  - P10 GMGR (Worst Path Grow)  : {ctx.gate1_p10_gmgr:.6f}   {_fmt_pass_info(gmgr_ok)} (Target: > 0)",
        f"  - CPCV Mean Path Return       : {ctx.cpcv_mean_path_return_pct:.1f}%",
        f"  - CPCV Worst Segment MDD      : {ctx.cpcv_worst_segment_mdd_pct:.1f}%",
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
        "=" * 71,
        " [PART 3. SYMBOL MICROSTRUCTURE & FINAL VERDICT]",
        "=" * 71,
        "▶ Portfolio Composition (Shared Cash)",
        f"  - Capital: ₩{ctx.initial_capital_krw:,.0f} -> ₩{final_capital:,.0f} ({profit_pct:+.1f}%)",
        f"  - Total Trades: {ctx.oos_total_trades} | Concentration: {ctx.loso_warning}",
        "",
        "  | Symbol    | Net CAGR | Max MDD | Win Rate | Trades |",
        "  |-----------|----------|---------|----------|--------|",
    ]
    for row in ctx.symbol_rows:
        lines.append(
            f"  | {row.symbol:<9} | {row.net_cagr_pct:>+6.1f}% | {row.max_mdd_pct:>6.1f}% | "
            f"{row.win_rate_pct:>7.1f}% | {row.trade_count:>6} |"
        )

    lines.extend(
        [
            "",
            f"▶ Final Verdict : {'[GO - DEPLOYABLE]' if ctx.final_decision_go else '[NO-GO - REFINEMENT NEEDED]'}",
            f"  Compliance Score: {ctx.hard_passed}/{ctx.hard_total} Critical Gates Passed",
        ]
    )

    if not ctx.final_decision_go:
        lines.append("\n  ※ 주요 결격 사유 (Critical Failures):")
        if not psr_ok:
            lines.append(f"    - TIER1: PSR 점수({ctx.gate1_psr:.4f})가 기준({ctx.psr_target}) 미달")
        if not dsr_ok:
            lines.append(f"    - TIER1: DSR 점수({ctx.gate1_dsr:.4f})가 기준({ctx.dsr_target}) 미달")
        if not oos_mdd_ok:
            lines.append(f"    - TIER2: OOS MDD({abs(ctx.oos_mdd_pct):.1f}%)가 제한({ctx.oos_mdd_limit_pct}%) 초과")
        if not calmar_ok:
            lines.append(f"    - TIER2: Calmar Ratio({ctx.oos_calmar:.2f})가 기준({ctx.calmar_target}) 미달")
        if not oos_cagr_ok:
            lines.append(f"    - TIER3: OOS CAGR({ctx.oos_net_cagr_pct:.1f}%)이 목표({ctx.oos_cagr_target_pct}%) 미달")
        if not pf_ok:
            lines.append(f"    - TIER3: Profit Factor({ctx.oos_pf:.2f})가 기준({ctx.pf_target}) 미달")
        if not ad_ok:
            lines.append(f"    - TIER3: Alpha Decay({ctx.alpha_decay_pct:.1f}%)가 허용치({ctx.alpha_decay_floor_pct}%) 초과")

    lines.append("=" * 71)
    return "\n".join(lines)
