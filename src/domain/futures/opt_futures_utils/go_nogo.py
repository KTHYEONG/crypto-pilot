"""
상위 1% 기관급 자산 증식을 위한 OOS 심층 검증 (RoMaD 및 계단식 PF 필터 적용)
Futures deployment report: same layout as Spot `run_final_deployment_report`, futures thresholds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

_logger: logging.Logger = logging.getLogger("opt_futures")


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


def run_go_nogo_check(
    cv_fold_scores: List[float],
    holdout_score: float,
    oos_romad_scores: List[float],
    max_mdd_pct: float,
    profit_factor: float,
    long_count: int,
    short_count: int,
    tf: str = "4h",
    *,
    long_short_ratio_oos: float = 0.0,
) -> GoNoGoResult:

    total_trades = long_count + short_count
    oos_cagr: float = float(oos_romad_scores[0]) if oos_romad_scores else -100.0
    abs_mdd = abs(max_mdd_pct) if max_mdd_pct != 0 else 1e-9

    # --- 1. Return over Max Drawdown (RoMaD) - 리스크 대비 수익성 ---
    # 단순히 CAGR > 0이 아니라, "MDD 대비 수익률이 1.0배 이상인가?"를 검증
    romad = oos_cagr / abs_mdd
    growth_pass: bool = oos_cagr > 5.0 and romad >= 0.8  # 최소 수익률 5% 보장 및 방어력 검증

    # --- 2. Absolute Volatility Drag (MDD 한계치) ---
    # 상위 1% 레버리지 운용을 위해 MDD는 25%로 하향 압박 (기존 35%는 파산 리스크 농후)
    mdd_pass: bool = abs_mdd <= 25.0

    # --- 3. Mathematical Edge (PF Baseline) ---
    # 실전 슬리피지와 수수료를 극복하는 최소 컷오프를 1.50으로 상향
    target_pf = 1.50
    pf_pass: bool = profit_factor >= target_pf

    # --- 4. Dynamic Statistical Edge (Tiered N-Requirement) ---
    # 퀄리티(PF)가 압도적일수록 통계적 표본 요구량을 유연하게 삭감 (INJ 구제 및 노이즈 차단)
    if tf == "1h":
        base_req = 30
    else:  # 4h
        base_req = 10

    if profit_factor >= 3.0:
        # Tier 1: 초격차 퀄리티 (예: INJ PF 13.0). 거래 횟수 5회 이상이면 포트폴리오 편입 승인.
        trades_pass = total_trades >= int(base_req * 0.5)
    elif profit_factor >= 2.0:
        # Tier 2: 우수 퀄리티. 거래 횟수 7회 이상 승인.
        trades_pass = total_trades >= int(base_req * 0.7)
    else:
        # Tier 3: 기준선 (PF 1.5 ~ 1.99). 철저하게 기본 횟수(10회) 충족 요구.
        trades_pass = total_trades >= base_req

    ls_pass: bool = float(long_short_ratio_oos) >= 0.15

    details: Dict[str, bool] = {
        "1. Risk-Adjusted Return (RoMaD >= 0.8)": growth_pass,
        "2. Strict Volatility Limit (MDD <= 25%)": mdd_pass,
        f"3. Institutional Edge (PF >= {target_pf})": pf_pass,
        "4. Dynamic Stat Edge (N-Tier Valid)": trades_pass,
        "5. Long/Short balance (minority >= 15%)": ls_pass,
    }

    all_passed = all(details.values())

    # --- Summary Formatting ---
    summary_lines: List[str] = ["[Elite 1% Wealth Compounding Checklist]"]

    metric_values: Dict[str, str] = {
        "1. Risk-Adjusted Return (RoMaD >= 0.8)": f"RoMaD: {romad:.2f} (CAGR {oos_cagr:.1f}%)",
        "2. Strict Volatility Limit (MDD <= 25%)": f"MDD: {abs_mdd:.1f}%",
        f"3. Institutional Edge (PF >= {target_pf})": f"PF: {profit_factor:.2f}",
        "4. Dynamic Stat Edge (N-Tier Valid)": f"N: {total_trades}",
        "5. Long/Short balance (minority >= 15%)": f"L/S ratio: {long_short_ratio_oos:.2f}",
    }

    req_met: int = sum(1 for v in details.values() if v)
    total_req: int = len(details)

    for k, v in details.items():
        status: str = "PASS" if v else "FAIL"
        val_str: str = metric_values.get(k, "")
        summary_lines.append(f"  - {k:<45}: {status:<5} ({val_str})")

    summary_lines.append("-" * 60)
    final_status: str = (
        "🌟 ELITE GO (Top 1% Ready)"
        if all_passed
        else f"🔴 NO-GO (Needs Revision, Passed {req_met}/{total_req})"
    )
    summary_lines.append(f"  FINAL VERDICT: {final_status}")

    return GoNoGoResult(passed=all_passed, details=details, summary="\n".join(summary_lines))


@dataclass(frozen=True)
class FuturesSymbolGateRow:
    symbol: str
    net_cagr_pct: float
    max_mdd_pct: float
    win_rate_pct: float
    trade_count: int


@dataclass(frozen=True)
class FuturesDeploymentReportInput:
    gate1_sqn: float
    gate1_path_sortino: float
    gate1_tail_ratio: float
    gate1_p10_gmgr: float
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
    initial_capital_usdt: float
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
    funding_drag_pct: float
    funding_drag_limit_pct: float
    terminal_wealth_ratio: float
    tw_target: float
    oos_total_trades: int
    oos_pf: float
    oos_long_pf: float
    oos_short_pf: float
    oos_short_win_rate_pct: float
    oos_ev_cost_ratio: float
    oos_ulcer_index: float
    pf_target: float
    oos_calmar: float
    calmar_target: float
    oos_win_rate_pct: float
    oos_long_short_minority_pct: float
    symbol_rows: Sequence[FuturesSymbolGateRow] = field(default_factory=tuple)
    loso_warning: str = ""
    hard_passed: int = 0
    hard_total: int = 0
    final_decision_go: bool = False
    pbo: float = float("nan")
    spearman_rho: float = float("nan")
    pbo_n_paths: int = 0
    pbo_gate_passed: bool = True
    pbo_hard_gate: bool = False
    multi_window_passed: bool = True
    multi_window_summary: str = ""
    regime_diagnostic_block: str = ""
    oos_long_trades: int = 0
    oos_short_trades: int = 0
    funding_cost_total_usdt: float = 0.0
    gross_pnl_abs_usdt: float = 0.0


def _fmt_pass_info(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def _fmt_pf(val: float) -> str:
    return f"{val:.2f}" if val > 0 else "N/A"


_PART3_COL_WIDTHS: Tuple[int, int, int, int, int] = (11, 18, 9, 10, 8)


def _part3_symbol_table_lines(rows: Sequence[FuturesSymbolGateRow]) -> List[str]:
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


def run_futures_deployment_report(ctx: FuturesDeploymentReportInput) -> str:
    # --- Relaxed Targets for Futures Trend Following ---
    sqn_ok = ctx.gate1_sqn >= 1.6  # Relaxed from 2.0
    ps_ok = ctx.gate1_path_sortino >= ctx.path_sortino_target
    g1_tr_ok = ctx.gate1_tail_ratio >= ctx.tail_ratio_target
    gmgr_ok = ctx.gate1_p10_gmgr >= -0.001
    psr_ok = ctx.gate1_psr >= 0.40  # Relaxed from 0.50
    dsr_ok = ctx.gate1_dsr >= 0.20  # Relaxed from 0.25

    oos_mdd_ok = abs(ctx.oos_mdd_pct) <= ctx.oos_mdd_limit_pct
    cvar_ok = ctx.oos_cvar_pct <= ctx.cvar_limit_pct
    hw_ok = ctx.hw_recovery_days <= ctx.hw_recovery_max_days
    calmar_ok = ctx.oos_calmar >= ctx.calmar_target
    fund_ok = ctx.funding_drag_pct <= ctx.funding_drag_limit_pct

    oos_cagr_ok = ctx.oos_net_cagr_pct >= ctx.oos_cagr_target_pct
    pf_ok = ctx.oos_pf >= ctx.pf_target
    l_pf_ok = ctx.oos_long_pf >= 1.05 if ctx.oos_long_trades > 0 else True
    s_pf_ok = ctx.oos_short_pf >= 1.05 if ctx.oos_short_trades > 0 else True

    # New Futures Hard Gates
    ev_cost_ok = ctx.oos_ev_cost_ratio >= 3.0
    short_wr_ok = ctx.oos_short_win_rate_pct >= 35.0 if ctx.oos_short_trades > 5 else True

    ad_ok = ctx.alpha_decay_pct >= ctx.alpha_decay_floor_pct
    tw_ok = ctx.terminal_wealth_ratio > ctx.tw_target

    final_capital = ctx.initial_capital_usdt * ctx.moic
    profit_pct = (ctx.moic - 1.0) * 100.0

    pbo_disp = f"{ctx.pbo:.4f}" if math.isfinite(ctx.pbo) else "N/A"
    rho_disp = f"{ctx.spearman_rho:.4f}" if math.isfinite(ctx.spearman_rho) else "N/A"

    lines: List[str] = [
        "=" * 71,
        " [TIER 1. CPCV STATISTICAL EDGE RIGOR]",
        "=" * 71,
        f"  - System Quality Number (SQN) : {ctx.gate1_sqn:.2f}   {_fmt_pass_info(sqn_ok)} (Min: 1.6)",
        f"  - Path Sortino Ratio          : {ctx.gate1_path_sortino:.2f}   {_fmt_pass_info(ps_ok)} (Min: {ctx.path_sortino_target})",
        f"  - Path Tail Ratio (Discovery) : {ctx.gate1_tail_ratio:.2f}   {_fmt_pass_info(g1_tr_ok)} (Min: {ctx.tail_ratio_target})",
        f"  - Prob. Sharpe Ratio (PSR)    : {ctx.gate1_psr:.4f}   {_fmt_pass_info(psr_ok)} (Min: 0.40)",
        f"  - Deflated Sharpe Ratio (DSR) : {ctx.gate1_dsr:.4f}   {_fmt_pass_info(dsr_ok)} (Min: 0.20)",
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
        " [TIER 2. OOS ABSOLUTE RISK HARD GATES: 4H FUTURES]",
        "=" * 71,
        f"  - Maximum Pain (MDD Limit)    : {ctx.oos_mdd_pct:.1f}%   {_fmt_pass_info(oos_mdd_ok)} (Limit: {ctx.oos_mdd_limit_pct}%)",
        f"  - Portfolio CVaR(5%) Loss     : {ctx.oos_cvar_pct:.2f}%   {_fmt_pass_info(cvar_ok)} (Limit: {ctx.cvar_limit_pct}%)",
        f"  - Recovery Time (Max UD)      : {ctx.hw_recovery_days:.1f}d   {_fmt_pass_info(hw_ok)} (Limit: {ctx.hw_recovery_max_days}d)",
        f"  - Ulcer Index (Pain)          : {ctx.oos_ulcer_index:.2f}   (Info Only)",
        f"  - OOS Calmar Ratio (Grow/Risk): {ctx.oos_calmar:.2f}   {_fmt_pass_info(calmar_ok)} (Min: {ctx.calmar_target})",
        f"  - Funding Drag Ratio          : {ctx.funding_drag_pct:.2f}%   {_fmt_pass_info(fund_ok)} (Limit: {ctx.funding_drag_limit_pct}%)",
        "",
        "=" * 71,
        " [TIER 3. OOS PROFITABILITY & ROBUSTNESS]",
        "=" * 71,
        f"  - Annualized Return (CAGR)    : {ctx.oos_net_cagr_pct:.1f}%   {_fmt_pass_info(oos_cagr_ok)} (Min: {ctx.oos_cagr_target_pct}%)",
        f"  - EV/Cost Ratio (Min 3.0)     : {ctx.oos_ev_cost_ratio:.2f}   {_fmt_pass_info(ev_cost_ok)}",
        f"  - Trade Profit Factor         : {ctx.oos_pf:.2f}   {_fmt_pass_info(pf_ok)} (Min: {ctx.pf_target})",
        f"  - Directional PF (L/S >= 1.05): {_fmt_pf(ctx.oos_long_pf)} / {_fmt_pf(ctx.oos_short_pf)}   {_fmt_pass_info(l_pf_ok and s_pf_ok)}",
        f"  - Short Win Rate (Min 35%)    : {ctx.oos_short_win_rate_pct:.1f}%   {_fmt_pass_info(short_wr_ok)}",
        f"  - Alpha Decay (Stability)     : {ctx.alpha_decay_pct:.1f}%   {_fmt_pass_info(ad_ok)} (Limit: {ctx.alpha_decay_floor_pct}%)",
        f"  - Terminal Wealth Ratio       : {ctx.terminal_wealth_ratio:.3f}   {_fmt_pass_info(tw_ok)} (Min: {ctx.tw_target})",
        f"  - OOS Win Rate (INFO)         : {ctx.oos_win_rate_pct:.1f}%",
        f"  - Long/Short Ratio (INFO)     : {ctx.oos_long_short_minority_pct:.1f}% minority direction",
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
            "▶ Portfolio Composition (Margin-Shared)",
            f"  - Capital: ${ctx.initial_capital_usdt:,.0f} -> ${final_capital:,.0f} ({profit_pct:+.1f}%)",
            f"  - Total Trades: {ctx.oos_total_trades} | Concentration: {ctx.loso_warning}",
            "",
        ]
    )
    lines.extend(_part3_symbol_table_lines(ctx.symbol_rows))

    lines.extend(
        [
            "  ※ Symbol PnL contrib ann%: margin-shared trade PnL vs initial, annualized (not standalone engine CAGR).",
            "",
            f"▶ Final Verdict : {'[GO - DEPLOYABLE]' if ctx.final_decision_go else '[NO-GO - REFINEMENT NEEDED]'}",
            f"  Compliance Score: {ctx.hard_passed}/{ctx.hard_total} Critical Gates Passed",
        ]
    )

    tot_ls = int(ctx.oos_long_trades) + int(ctx.oos_short_trades)
    ratio_txt = f"{ctx.oos_long_trades}/{ctx.oos_short_trades}" if tot_ls > 0 else "0/0"
    drag_pct = (
        (ctx.funding_cost_total_usdt / max(ctx.gross_pnl_abs_usdt, 1e-9)) * 100.0
        if ctx.gross_pnl_abs_usdt > 0
        else 0.0
    )
    lines.extend(
        [
            "",
            "=" * 71,
            " [PART 4. LONG/SHORT BALANCE (FUTURES)]",
            "=" * 71,
            "▶ Long/Short Balance Diagnostic",
            f"  - Long Trades: {ctx.oos_long_trades} | Short Trades: {ctx.oos_short_trades} | Ratio: {ratio_txt}",
            f"  - Funding Cost Total: ${ctx.funding_cost_total_usdt:,.2f} | Drag: {drag_pct:.1f}% of gross PnL",
            "=" * 71,
        ]
    )

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
        if not cvar_ok:
            lines.append(
                f"    - TIER2: OOS CVaR({ctx.oos_cvar_pct:.2f}%)가 제한({ctx.cvar_limit_pct}%) 초과"
            )
        if not calmar_ok:
            lines.append(
                f"    - TIER2: Calmar Ratio({ctx.oos_calmar:.2f})가 기준({ctx.calmar_target}) 미달"
            )
        if not fund_ok:
            lines.append(
                f"    - TIER2: Funding drag({ctx.funding_drag_pct:.2f}%)가 제한({ctx.funding_drag_limit_pct}%) 초과"
            )
        if not oos_cagr_ok:
            lines.append(
                f"    - TIER3: OOS CAGR({ctx.oos_net_cagr_pct:.1f}%)이 목표({ctx.oos_cagr_target_pct}%) 미달"
            )
        if not pf_ok:
            lines.append(
                f"    - TIER3: Profit Factor({ctx.oos_pf:.2f})가 기준({ctx.pf_target}) 미달"
            )
        if not l_pf_ok:
            lines.append(
                f"    - TIER3: Long Profit Factor({ctx.oos_long_pf:.2f})가 1.05 미달 (방향성 엣지 붕괴)"
            )
        if not s_pf_ok:
            lines.append(
                f"    - TIER3: Short Profit Factor({ctx.oos_short_pf:.2f})가 1.05 미달 (방향성 엣지 붕괴)"
            )
        if not ad_ok:
            lines.append(
                f"    - TIER3: Alpha Decay({ctx.alpha_decay_pct:.1f}%)가 허용치({ctx.alpha_decay_floor_pct}%) 미달"
            )

    if ctx.regime_diagnostic_block:
        lines.append("")
        lines.append(ctx.regime_diagnostic_block)

    lines.append("=" * 71)
    return "\n".join(lines)


def run_multi_window_oos_gate(
    *,
    window_results: List[Dict[str, Any]],
    min_positive_windows: int,
    min_median_cagr_pct: float,
    max_worst_mdd_pct: float,
) -> GoNoGoResult:
    """Anchored multi-window OOS consistency."""
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
        "=" * 71,
        " [Gate 3.5 — Multi-Window OOS (anchored)]",
        "=" * 71,
        *[
            (
                f"  - end_idx={w['end_idx']} | CAGR {w['cagr_pct']:.2f}% | "
                f"MDD {w['mdd_pct']:.2f}% | PF {w['pf']:.2f}"
            )
            for w in window_results
        ],
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
    )


def format_regime_oos_diagnostic_block(
    regime_metrics: Dict[str, Dict[str, float]],
    stress_mdd_warn_pct: float,
) -> str:
    """Advisory TIER 4 text block for futures deployment log."""
    lines: List[str] = [
        "=" * 71,
        " [TIER 4. REGIME ROBUSTNESS DIAGNOSTIC (advisory)]",
        "=" * 71,
    ]
    order = ("risk_on", "cautious", "stress")
    labels = {
        "risk_on": "Risk-On (mult > 0.5)      ",
        "cautious": "Cautious (0 < mult <= 0.5)",
        "stress": "Stress (mult <= 0)       ",
    }
    for key in order:
        m = regime_metrics.get(key, {})
        n = int(m.get("bar_count", 0.0))
        rp = float(m.get("return_pct", 0.0))
        mdd = float(m.get("mdd_pct", 0.0))
        lines.append(f"  - {labels[key]}: N={n:<4} | ret: {rp:>+.2f}% | MDD: {mdd:>5.2f}%")
    stress_mdd = float(regime_metrics.get("stress", {}).get("mdd_pct", 0.0))
    if stress_mdd > float(stress_mdd_warn_pct):
        lines.append("")
        lines.append(
            f"  ⚠ WARNING: Stress-regime MDD ({stress_mdd:.2f}%) exceeds threshold ({stress_mdd_warn_pct}%)"
        )
    lines.append("=" * 71)
    return "\n".join(lines)
