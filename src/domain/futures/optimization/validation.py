"""Anchored walk-forward leg builders and futures validation helpers.

Includes PBO/DSR gate adjustment and Go/No-Go checks.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import numpy as np

_logger: logging.Logger = logging.getLogger("opt_futures")

# Anchored WF leg geometry: (train_start, train_end, test_start, test_end)
AWFLeg = tuple[int, int, int, int]


def build_anchored_wf_legs(
    n_bars: int,
    k: int = 6,
    embargo: int = 0,
    *,
    is_pool_frac: float = 0.70,
    min_train_frac: float | None = None,
) -> list[AWFLeg]:
    """Anchored WF: cumulative train [:anchor_i); test only inside the trailing OOS-pool."""
    n = int(n_bars)
    k = max(2, int(k))
    e = max(0, int(embargo))
    frac = float(min_train_frac) if min_train_frac is not None else float(is_pool_frac)
    frac = float(np.clip(frac, 0.05, 0.95))
    anchor0 = max(1, int(n * frac))
    # Remaining timeline after earliest train cutoff still needs room for embargo + legs.
    remaining = n - anchor0 - e
    leg_width = remaining // k if k > 0 else 0
    if leg_width < 20 or remaining <= 0:
        return []

    legs: list[AWFLeg] = []
    for i in range(k):
        anchor = anchor0 + i * leg_width
        test_start = anchor + e
        test_end = test_start + leg_width if i < k - 1 else n
        if test_end <= test_start or anchor <= 0:
            continue
        legs.append((0, anchor, test_start, test_end))

    return legs


# --- MC Gate Adjustment (from mc_gate_adjust.py) ---


def trial_adjusted_pbo_ceiling(
    base: float,
    n_trials: int,
    *,
    step: float = 0.01,
    bucket: int = 100,
    clamp_min: float = 0.38,
) -> float:
    b = max(1, int(bucket))
    k = max(0, int(n_trials)) // b
    adj = float(base) - float(step) * float(k)
    return float(min(float(base), max(float(clamp_min), adj)))


def trial_adjusted_dsr_floor(
    base: float,
    n_trials: int,
    *,
    step: float = 0.02,
    bucket: int = 100,
    clamp_max: float = 0.95,
) -> float:
    b = max(1, int(bucket))
    k = max(0, int(n_trials)) // b
    adj = float(base) + float(step) * float(k)
    return float(min(float(clamp_max), adj))


def wf_path_ergodicity_deviation_pct(leg_tw: Sequence[float]) -> float:
    arr = np.asarray(list(leg_tw), dtype=np.float64)
    if arr.size < 2:
        return 0.0
    m = float(np.mean(arr))
    if m < 1e-12:
        return 0.0
    return float(np.max(np.abs(arr - m)) / m * 100.0)


def awf_pos_frac_to_pseudo_pbo(pos_frac: float) -> float:
    """Heuristic selection-pressure proxy in [0, 1], **not** Lopez-Prado PBO.

    Kept for backwards-compatible gate wiring; interpret as "failure pressure from
    low positive-leg fraction", not a calibrated probability of backtest overfitting.
    """
    return float(np.clip(1.0 - float(pos_frac), 0.0, 1.0))


def resolve_adjusted_gates(cfg: dict[str, Any], n_trials: int) -> tuple[float, float, float]:
    raw_pbo_max = float(cfg.get("FUTURES_PBO_MAX", 0.45))
    raw_dsr_min = float(cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.20))
    raw_champ = float(cfg.get("FUTURES_CHAMPION_PBO_STRICT_MAX", 0.40))
    if not bool(cfg.get("FUTURES_MC_GATE_TRIAL_ADJUST_ENABLED", False)):
        return raw_pbo_max, raw_dsr_min, raw_champ

    bucket = int(cfg.get("FUTURES_MC_GATE_BUCKET_TRIALS", 100))
    pbo_step = float(cfg.get("FUTURES_MC_PBO_STEP_PER_BUCKET", 0.01))
    pbo_clamp = float(cfg.get("FUTURES_MC_PBO_CEILING_CLAMP_MIN", 0.38))

    pbo_max = trial_adjusted_pbo_ceiling(raw_pbo_max, n_trials, step=pbo_step, bucket=bucket, clamp_min=pbo_clamp)
    champ = trial_adjusted_pbo_ceiling(raw_champ, n_trials, step=pbo_step, bucket=bucket, clamp_min=pbo_clamp)
    dsr_min = raw_dsr_min
    if bool(cfg.get("FUTURES_MC_DSR_TRIAL_ADJUST_ENABLED", False)):
        dsr_step = float(cfg.get("FUTURES_MC_DSR_STEP_PER_BUCKET", 0.02))
        dsr_cap = float(cfg.get("FUTURES_MC_DSR_FLOOR_CAP", 0.95))
        dsr_min = trial_adjusted_dsr_floor(raw_dsr_min, n_trials, step=dsr_step, bucket=bucket, clamp_max=dsr_cap)
    return pbo_max, dsr_min, champ


# --- Go/No-Go Check (from go_nogo.py) ---


@dataclass
class CheckRecord:
    """Record of a single gate check."""

    check_id: str
    label: str
    observed: float
    threshold: float
    passed: bool


@dataclass
class GoNoGoResult:
    """Result of a Go/No-Go check with details and summary."""

    passed: bool
    details: dict[str, bool]
    summary: str
    checks: list[CheckRecord] = field(default_factory=list)
    advisory: dict[str, Any] = field(default_factory=dict)


def run_go_nogo_check(
    cv_fold_scores: list[float],
    holdout_score: float,
    oos_romad_scores: list[float],
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

    romad = oos_cagr / abs_mdd
    growth_pass: bool = oos_cagr > 5.0 and romad >= 0.8
    mdd_pass: bool = abs_mdd <= 35.0
    target_pf = 1.50
    pf_pass: bool = profit_factor >= target_pf

    base_req = 30 if tf == "1h" else 10

    if profit_factor >= 3.0:
        trades_pass = total_trades >= int(base_req * 0.5)
    elif profit_factor >= 2.0:
        trades_pass = total_trades >= int(base_req * 0.7)
    else:
        trades_pass = total_trades >= base_req

    ls_pass: bool = float(long_short_ratio_oos) >= 0.15

    details: dict[str, bool] = {
        "1. Risk-Adjusted Return (RoMaD >= 0.8)": growth_pass,
        "2. Healthy Volatility Limit (MDD <= 35%)": mdd_pass,
        f"3. Institutional Edge (PF >= {target_pf})": pf_pass,
        "4. Dynamic Stat Edge (N-Tier Valid)": trades_pass,
        "5. Long/Short balance (minority >= 15%)": ls_pass,
    }

    all_passed = all(details.values())

    summary_lines: list[str] = ["[Elite 1% Wealth Compounding Checklist]"]
    metric_values: dict[str, str] = {
        "1. Risk-Adjusted Return (RoMaD >= 0.8)": f"RoMaD: {romad:.2f} (CAGR {oos_cagr:.1f}%)",
        "2. Healthy Volatility Limit (MDD <= 35%)": f"MDD: {abs_mdd:.1f}%",
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
        "🌟 ELITE GO (Top 1% Ready)" if all_passed else f"🔴 NO-GO (Needs Revision, Passed {req_met}/{total_req})"
    )
    summary_lines.append(f"  FINAL VERDICT: {final_status}")

    return GoNoGoResult(passed=all_passed, details=details, summary="\n".join(summary_lines))


@dataclass(frozen=True)
class FuturesSymbolGateRow:
    """Row for a single symbol in the futures deployment report."""

    symbol: str
    net_cagr_pct: float
    max_mdd_pct: float
    win_rate_pct: float
    trade_count: int


@dataclass(frozen=True)
class FuturesDeploymentReportInput:
    """Input parameters for the futures deployment report."""

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


_PART3_COL_WIDTHS: tuple[int, int, int, int, int] = (11, 18, 9, 10, 8)


def _part3_symbol_table_lines(rows: Sequence[FuturesSymbolGateRow]) -> list[str]:
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
        "  | " + "-" * w[0] + " | " + "-" * w[1] + " | " + "-" * w[2] + " | " + "-" * w[3] + " | " + "-" * w[4] + " |"
    )
    out: list[str] = [header, rule]
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
    sqn_ok = ctx.gate1_sqn >= 1.6
    ps_ok = ctx.gate1_path_sortino >= ctx.path_sortino_target
    g1_tr_ok = ctx.gate1_tail_ratio >= ctx.tail_ratio_target
    gmgr_ok = ctx.gate1_p10_gmgr >= -0.001
    psr_ok = ctx.gate1_psr >= 0.40
    dsr_ok = ctx.gate1_dsr >= 0.20

    oos_mdd_ok = abs(ctx.oos_mdd_pct) <= 35.0
    cvar_ok = ctx.oos_cvar_pct <= ctx.cvar_limit_pct
    hw_ok = ctx.hw_recovery_days <= 120.0
    ui_ok = ctx.oos_ulcer_index <= 15.0
    calmar_ok = ctx.oos_calmar >= ctx.calmar_target
    fund_ok = ctx.funding_drag_pct <= ctx.funding_drag_limit_pct

    oos_cagr_ok = ctx.oos_net_cagr_pct >= ctx.oos_cagr_target_pct
    pf_ok = ctx.oos_pf >= ctx.pf_target
    l_pf_ok = ctx.oos_long_pf >= 1.05 if ctx.oos_long_trades > 0 else True
    s_pf_ok = ctx.oos_short_pf >= 1.05 if ctx.oos_short_trades > 0 else True

    ev_cost_ok = ctx.oos_ev_cost_ratio >= 3.0
    short_wr_ok = ctx.oos_short_win_rate_pct >= 35.0 if ctx.oos_short_trades > 5 else True

    ad_ok = ctx.alpha_decay_pct >= ctx.alpha_decay_floor_pct
    tw_ok = ctx.terminal_wealth_ratio > ctx.tw_target

    final_capital = ctx.initial_capital_usdt * ctx.moic
    profit_pct = (ctx.moic - 1.0) * 100.0

    pbo_disp = f"{ctx.pbo:.4f}" if math.isfinite(ctx.pbo) else "N/A"
    rho_disp = f"{ctx.spearman_rho:.4f}" if math.isfinite(ctx.spearman_rho) else "N/A"

    lines: list[str] = [
        "=" * 71,
        " [TIER 1. AWF STATISTICAL EDGE RIGOR]",
        "=" * 71,
        f"  - System Quality Number (SQN) : {ctx.gate1_sqn:.2f}   {_fmt_pass_info(sqn_ok)} (Min: 1.6)",
        f"  - Path Sortino Ratio          : {ctx.gate1_path_sortino:.2f}   "
        f"{_fmt_pass_info(ps_ok)} (Min: {ctx.path_sortino_target})",
        f"  - Path Tail Ratio (Discovery) : {ctx.gate1_tail_ratio:.2f}   "
        f"{_fmt_pass_info(g1_tr_ok)} (Min: {ctx.tail_ratio_target})",
        f"  - Prob. Sharpe Ratio (PSR)    : {ctx.gate1_psr:.4f}   {_fmt_pass_info(psr_ok)} (Min: 0.40)",
        f"  - Deflated Sharpe Ratio (DSR) : {ctx.gate1_dsr:.4f}   {_fmt_pass_info(dsr_ok)} (Min: 0.20)",
        f"  - P10 GMGR (Worst Path Grow)  : {ctx.gate1_p10_gmgr:.6f}   {_fmt_pass_info(gmgr_ok)} (Target: >= -0.001)",
        f"  - AWF mean path return          : {ctx.cpcv_mean_path_return_pct:.1f}%",
        f"  - AWF worst segment MDD         : {ctx.cpcv_worst_segment_mdd_pct:.1f}%",
        "",
        f"  - PBO (IS vs OOS path ranks, n_paths={ctx.pbo_n_paths})  : {pbo_disp}   "
        f"{_fmt_pass_info(ctx.pbo_gate_passed)} "
        f"({'HARD' if ctx.pbo_hard_gate else 'ADVISORY'})",
        f"  - Spearman rho (IS vs OOS)    : {rho_disp}",
        "",
        "=" * 71,
        " [TIER 2. OOS ABSOLUTE RISK HARD GATES: 4H FUTURES]",
        "=" * 71,
        f"  - Maximum Pain (MDD Limit)    : {ctx.oos_mdd_pct:.1f}%   {_fmt_pass_info(oos_mdd_ok)} (Limit: 35.0%)",
        f"  - Portfolio CVaR(5%) Loss     : {ctx.oos_cvar_pct:.2f}%   "
        f"{_fmt_pass_info(cvar_ok)} (Limit: {ctx.cvar_limit_pct}%)",
        f"  - Recovery Time (Max UD)      : {ctx.hw_recovery_days:.1f}d   {_fmt_pass_info(hw_ok)} (Limit: 120.0d)",
        f"  - Ulcer Index (Pain)          : {ctx.oos_ulcer_index:.2f}   {_fmt_pass_info(ui_ok)} (Limit: 15.0)",
        f"  - OOS Calmar Ratio (Grow/Risk): {ctx.oos_calmar:.2f}   "
        f"{_fmt_pass_info(calmar_ok)} (Min: {ctx.calmar_target})",
        f"  - Funding Drag Ratio          : {ctx.funding_drag_pct:.2f}%   "
        f"{_fmt_pass_info(fund_ok)} (Limit: {ctx.funding_drag_limit_pct}%)",
        "",
        "=" * 71,
        " [TIER 3. OOS PROFITABILITY & ROBUSTNESS]",
        "=" * 71,
        f"  - Annualized Return (CAGR)    : {ctx.oos_net_cagr_pct:.1f}%   "
        f"{_fmt_pass_info(oos_cagr_ok)} (Min: {ctx.oos_cagr_target_pct}%)",
        f"  - EV/Cost Ratio (Min 3.0)     : {ctx.oos_ev_cost_ratio:.2f}   {_fmt_pass_info(ev_cost_ok)}",
        f"  - Trade Profit Factor         : {ctx.oos_pf:.2f}   {_fmt_pass_info(pf_ok)} (Min: {ctx.pf_target})",
        f"  - Directional PF (L/S >= 1.05): {_fmt_pf(ctx.oos_long_pf)} / "
        f"{_fmt_pf(ctx.oos_short_pf)}   {_fmt_pass_info(l_pf_ok and s_pf_ok)}",
        f"  - Short Win Rate (Min 35%)    : {ctx.oos_short_win_rate_pct:.1f}%   {_fmt_pass_info(short_wr_ok)}",
        f"  - Alpha Decay (Stability)     : {ctx.alpha_decay_pct:.1f}%   "
        f"{_fmt_pass_info(ad_ok)} (Limit: {ctx.alpha_decay_floor_pct}%)",
        f"  - Terminal Wealth Ratio       : {ctx.terminal_wealth_ratio:.3f}   "
        f"{_fmt_pass_info(tw_ok)} (Min: {ctx.tw_target})",
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
        (ctx.funding_cost_total_usdt / max(ctx.gross_pnl_abs_usdt, 1e-9)) * 100.0 if ctx.gross_pnl_abs_usdt > 0 else 0.0
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
        # (생략: 기존 리포트와 동일한 결격 사유 로직)
        if not sqn_ok:
            lines.append(f"    - TIER1: SQN 점수({ctx.gate1_sqn:.2f}) 미달")
        if not psr_ok:
            lines.append(f"    - TIER1: PSR 점수({ctx.gate1_psr:.4f}) 미달")
        if not dsr_ok:
            lines.append(f"    - TIER1: DSR 점수({ctx.gate1_dsr:.4f}) 미달")
        if not oos_mdd_ok:
            lines.append(f"    - TIER2: OOS MDD({abs(ctx.oos_mdd_pct):.1f}%) 초과")
        if not cvar_ok:
            lines.append(f"    - TIER2: OOS CVaR({ctx.oos_cvar_pct:.2f}%) 초과")
        if not calmar_ok:
            lines.append(f"    - TIER2: Calmar Ratio({ctx.oos_calmar:.2f}) 미달")
        if not fund_ok:
            lines.append(f"    - TIER2: Funding drag({ctx.funding_drag_pct:.2f}%) 초과")
        if not oos_cagr_ok:
            lines.append(f"    - TIER3: OOS CAGR({ctx.oos_net_cagr_pct:.1f}%) 미달")
        if not pf_ok:
            lines.append(f"    - TIER3: Profit Factor({ctx.oos_pf:.2f}) 미달")

    if ctx.regime_diagnostic_block:
        lines.append("")
        lines.append(ctx.regime_diagnostic_block)

    lines.append("=" * 71)
    return "\n".join(lines)


def run_multi_window_oos_gate(
    *,
    window_results: list[dict[str, Any]],
    min_positive_windows: int,
    min_median_cagr_pct: float,
    max_worst_mdd_pct: float,
) -> GoNoGoResult:
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
            (f"  - end_idx={w['end_idx']} | CAGR {w['cagr_pct']:.2f}% | MDD {w['mdd_pct']:.2f}% | PF {w['pf']:.2f}")
            for w in window_results
        ],
        f"  - Positive windows >= {min_positive_windows} | "
        f"{'PASS' if ok_pos else 'FAIL'} | obs={pos}/{len(window_results)}",
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
    regime_metrics: dict[str, dict[str, float]],
    stress_mdd_warn_pct: float,
) -> str:
    lines: list[str] = [
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
        lines.append(f"  ⚠ WARNING: Stress-regime MDD ({stress_mdd:.2f}%) exceeds threshold ({stress_mdd_warn_pct}%)")
    lines.append("=" * 71)
    return "\n".join(lines)
