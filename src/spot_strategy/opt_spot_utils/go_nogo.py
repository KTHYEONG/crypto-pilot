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
    cpcv_mean_path_return_pct: float
    cpcv_worst_segment_mdd_pct: float
    sqn_target: float
    path_sortino_target: float
    tail_ratio_target: float
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
    symbol_rows: Sequence[SymbolGateRow]
    loso_warning: str
    hard_passed: int
    hard_total: int
    final_decision_go: bool


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
    psr_min: float = 0.5,
    dsr_min: float = -1.0,
) -> GoNoGoResult:
    """
    Discovery veto: PSR hard; DSR soft floor (fail if dsr < dsr_min, default -1.0).
    """
    psr_ok = psr >= psr_min
    dsr_ok = dsr >= dsr_min
    details: Dict[str, bool] = {"psr_hard": psr_ok, "dsr_soft": dsr_ok}
    passed = bool(psr_ok and dsr_ok)
    dsr_label = "soft floor" if dsr_min < 0.0 else "hard"
    summary_lines = [
        "[Portfolio Discovery Veto]",
        f"  PSR (hard): {psr:.4f} vs {psr_min} -> {'PASS' if psr_ok else 'FAIL'}",
        f"  DSR ({dsr_label}): {dsr:.4f} vs {dsr_min} -> {'PASS' if dsr_ok else 'FAIL'}",
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
    max_cvar_pct: float,
    mdd_limit_pct: float = 45.0,
    tw_need: float = 1.0,
    tail_ratio_min: float = 2.0,
    cagr_min_pct: float = 30.0,
    oos_dd_days: float = 0.0,
    hw_recovery_days_max: float = 300.0,
    is_cagr_pct: float = 0.0,
    alpha_decay_floor_pct: float = -50.0,
) -> GoNoGoResult:
    """
    Shared-cash holdout: terminal wealth, CAGR floor, MDD, CVaR, tail ratio,
    HWM recovery (days underwater), alpha decay vs IS mean CAGR.
    """
    c_tw = min_path_terminal_wealth_ratio > tw_need
    c_cagr = portfolio_cagr_pct > cagr_min_pct
    c_mdd = abs(portfolio_mdd_pct) <= mdd_limit_pct
    c_cvar = portfolio_cvar_pct <= max_cvar_pct
    c_tr = portfolio_tail_ratio >= tail_ratio_min
    c_hw = oos_dd_days <= hw_recovery_days_max
    if is_cagr_pct <= 0.0:
        c_alpha = False
        alpha_decay_pct = -100.0
    else:
        alpha_decay_pct = float((portfolio_cagr_pct - is_cagr_pct) / abs(is_cagr_pct) * 100.0)
        c_alpha = alpha_decay_pct >= alpha_decay_floor_pct

    passed = bool(c_tw and c_cagr and c_mdd and c_cvar and c_tr and c_hw and c_alpha)
    lines = [
        "[Spot Holdout Portfolio (shared-cash)]",
        (
            f"  - Holdout portfolio terminal wealth ratio > {tw_need} | "
            f"{'PASS' if c_tw else 'FAIL'} | observed={min_path_terminal_wealth_ratio:.5f} | need > {tw_need}"
        ),
        (
            f"  - Holdout portfolio CAGR > {cagr_min_pct}% | "
            f"{'PASS' if c_cagr else 'FAIL'} | observed={portfolio_cagr_pct:.4f} | need > {cagr_min_pct}"
        ),
        (
            f"  - Holdout portfolio MDD <= {mdd_limit_pct}% | "
            f"{'PASS' if c_mdd else 'FAIL'} | observed={abs(portfolio_mdd_pct):.5f} | need <= {mdd_limit_pct}"
        ),
        (
            f"  - Holdout portfolio CVaR <= {max_cvar_pct}% | "
            f"{'PASS' if c_cvar else 'FAIL'} | observed={portfolio_cvar_pct:.6f} | need <= {max_cvar_pct}"
        ),
        (
            f"  - Holdout portfolio Tail Ratio >= {tail_ratio_min} | "
            f"{'PASS' if c_tr else 'FAIL'} | observed={portfolio_tail_ratio:.4f} | need >= {tail_ratio_min}"
        ),
        (
            f"  - HWM recovery (underwater days) <= {hw_recovery_days_max} | "
            f"{'PASS' if c_hw else 'FAIL'} | observed={oos_dd_days:.1f} | need <= {hw_recovery_days_max}"
        ),
        (
            f"  - Alpha decay % >= {alpha_decay_floor_pct}% | "
            f"{'PASS' if c_alpha else 'FAIL'} | observed={alpha_decay_pct:.2f}% | IS_CAGR={is_cagr_pct:.2f}%"
        ),
        "-" * 55,
        f"  FINAL: {'GO' if passed else 'NO-GO'}",
    ]
    checks: List[CheckRecord] = [
        CheckRecord("tw", "terminal wealth ratio", min_path_terminal_wealth_ratio, tw_need, c_tw),
        CheckRecord("cagr", f"CAGR > {cagr_min_pct}", portfolio_cagr_pct, cagr_min_pct, c_cagr),
        CheckRecord("mdd", "MDD cap", abs(portfolio_mdd_pct), mdd_limit_pct, c_mdd),
        CheckRecord("cvar", "CVaR cap", portfolio_cvar_pct, max_cvar_pct, c_cvar),
        CheckRecord("tail_ratio", "Tail ratio floor", portfolio_tail_ratio, tail_ratio_min, c_tr),
        CheckRecord("hwm", "HWM recovery days", oos_dd_days, hw_recovery_days_max, c_hw),
        CheckRecord("alpha_decay", "Alpha decay floor %", alpha_decay_pct, alpha_decay_floor_pct, c_alpha),
    ]
    return GoNoGoResult(
        passed=passed,
        details={
            "tw": c_tw,
            "cagr": c_cagr,
            "mdd": c_mdd,
            "cvar": c_cvar,
            "tail_ratio": c_tr,
            "hw_recovery": c_hw,
            "alpha_decay": c_alpha,
        },
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
    min_portfolio_trades: int,
    max_cvar_pct: float,
    tail_ratio_min: float = 2.0,
    cagr_min_pct: float = 30.0,
    mdd_limit_pct: float = 45.0,
    oos_dd_days: float = 0.0,
    hw_recovery_days_max: float = 300.0,
    is_cagr_pct: float = 0.0,
    alpha_decay_floor_pct: float = -50.0,
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
        max_cvar_pct=max_cvar_pct,
        tail_ratio_min=tail_ratio_min,
        cagr_min_pct=cagr_min_pct,
        mdd_limit_pct=mdd_limit_pct,
        oos_dd_days=oos_dd_days,
        hw_recovery_days_max=hw_recovery_days_max,
        is_cagr_pct=is_cagr_pct,
        alpha_decay_floor_pct=alpha_decay_floor_pct,
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
    """Build the 3-gate Spot Strategy deployment report (logging-friendly)."""
    sqn_ok = ctx.gate1_sqn >= ctx.sqn_target
    ps_ok = ctx.gate1_path_sortino >= ctx.path_sortino_target
    g1_tr_ok = ctx.gate1_tail_ratio >= ctx.tail_ratio_target

    oos_cagr_ok = ctx.oos_net_cagr_pct >= ctx.oos_cagr_target_pct
    oos_mdd_ok = abs(ctx.oos_mdd_pct) <= ctx.oos_mdd_limit_pct
    hw_ok = ctx.hw_recovery_days <= ctx.hw_recovery_max_days
    ad_ok = ctx.alpha_decay_pct >= ctx.alpha_decay_floor_pct

    lines: List[str] = [
        "=" * 71,
        "[Spot Strategy Final Verification & Deployment Report]",
        "=" * 71,
        "",
        "▶ Gate 1. Statistical Edge (CPCV / path robustness)",
        f"  - System Quality Number (SQN) : {ctx.gate1_sqn:.2f}   {_fmt_pass_info(sqn_ok)} (Target: >= {ctx.sqn_target})",
        f"  - Path Sortino Ratio          : {ctx.gate1_path_sortino:.2f}   {_fmt_pass_info(ps_ok)} (Target: >= {ctx.path_sortino_target})",
        f"  - Tail Ratio (path asymmetry)  : {ctx.gate1_tail_ratio:.2f}   {_fmt_pass_info(g1_tr_ok)} (Target: >= {ctx.tail_ratio_target})",
        f"  - CPCV Mean path return       : {ctx.cpcv_mean_path_return_pct:.1f}%",
        f"  - CPCV Worst Segment MDD      : {ctx.cpcv_worst_segment_mdd_pct:.1f}%",
        "",
        "▶ Gate 2. Business Impact (OOS holdout)",
        f"  - MOIC (multiple on capital)   : {ctx.moic:.1f}x   [INFO] (₩{ctx.initial_capital_krw:,.0f} -> ₩{ctx.initial_capital_krw * ctx.moic:,.0f})",
        f"  - OOS Net CAGR                 : {ctx.oos_net_cagr_pct:.1f}%   {_fmt_pass_info(oos_cagr_ok)} (Target: >= {ctx.oos_cagr_target_pct}%)",
        f"  - OOS Max Drawdown             : {ctx.oos_mdd_pct:.1f}%   {_fmt_pass_info(oos_mdd_ok)} (Target: <= {ctx.oos_mdd_limit_pct}%)",
        f"  - HWM Recovery (max underwater): {ctx.hw_recovery_days:.0f} days   {_fmt_pass_info(hw_ok)} (Target: <= {ctx.hw_recovery_max_days:.0f} days)",
        f"  - Alpha Decay (IS vs OOS)      : {ctx.alpha_decay_pct:.1f}%   {_fmt_pass_info(ad_ok)} (Target: >= {ctx.alpha_decay_floor_pct}%)",
        "",
        "▶ Gate 3. Symbol-Level Microstructure (OOS)",
        "  | Symbol    | Net CAGR | Max MDD | Tail Ratio | Win Rate | Trade Count |",
        "  |-----------|----------|---------|------------|----------|-------------|",
    ]
    for row in ctx.symbol_rows:
        lines.append(
            f"  | {row.symbol:<9} | {row.net_cagr_pct:>+6.1f}% | {row.max_mdd_pct:>6.1f}% | "
            f"{row.tail_ratio:>10.2f} | {row.win_rate_pct:>7.1f}% | {row.trade_count:>11} |"
        )
    lines.extend(
        [
            "  ---------------------------------------------------------------------",
            f"  * LOSO / concentration: {ctx.loso_warning}",
            "",
            "=" * 71,
            f"  FINAL DECISION: [{'GO - DEPLOYABLE' if ctx.final_decision_go else 'NO-GO'}] "
            f"(Passed {ctx.hard_passed}/{ctx.hard_total} Hard Constraints)",
            "=" * 71,
        ]
    )
    return "\n".join(lines)
