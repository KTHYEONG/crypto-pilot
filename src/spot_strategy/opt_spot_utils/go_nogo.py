from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

_logger: logging.Logger = logging.getLogger("opt_spot")


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
    tf: str = "4h",
) -> GoNoGoResult:
    """Legacy per-symbol holdout gate (PF floor 1.6)."""
    total_trades = long_count
    oos_cagr: float = float(oos_romad_scores[0]) if oos_romad_scores else -100.0
    growth_pass: bool = oos_cagr > 0.0
    mdd_pass: bool = abs(max_mdd_pct) <= 35.0
    target_pf = 1.6
    pf_pass: bool = profit_factor >= target_pf
    min_trades_req = 5
    trades_pass: bool = total_trades >= min_trades_req

    details: Dict[str, bool] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": growth_pass,
        "2. Volatility Drag (MDD <= 35%)": mdd_pass,
        f"3. Mathematical Edge (PF >= {target_pf})": pf_pass,
        f"4. Stat Edge (Trades >= {min_trades_req})": trades_pass,
    }
    all_passed = all(details.values())
    summary_lines: List[str] = ["[Spot Holdout Safety]"]
    metric_values: Dict[str, str] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": f"CAGR: {oos_cagr:.2f}%",
        "2. Volatility Drag (MDD <= 35%)": f"MDD: {abs(max_mdd_pct):.2f}%",
        f"3. Mathematical Edge (PF >= {target_pf})": f"PF: {profit_factor:.2f}",
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
    portfolio_pf: float,
    min_path_terminal_wealth_ratio: float,
    max_cvar_pct: float,
    mdd_limit_pct: float = 35.0,
    tw_need: float = 1.0,
    pf_need: float = 1.0,
    cagr_min_pct: float = 25.0,
) -> GoNoGoResult:
    """
    Shared-cash holdout: terminal wealth, CAGR floor, MDD, CVaR, PF (no trade count here).
    """
    c_tw = min_path_terminal_wealth_ratio > tw_need
    c_cagr = portfolio_cagr_pct > cagr_min_pct
    c_mdd = abs(portfolio_mdd_pct) <= mdd_limit_pct
    c_cvar = portfolio_cvar_pct <= max_cvar_pct
    c_pf = portfolio_pf >= pf_need
    passed = bool(c_tw and c_cagr and c_mdd and c_cvar and c_pf)
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
            f"  - Holdout portfolio PF >= {pf_need} | "
            f"{'PASS' if c_pf else 'FAIL'} | observed={portfolio_pf:.4f} | need >= {pf_need}"
        ),
        "-" * 55,
        f"  FINAL: {'GO' if passed else 'NO-GO'}",
    ]
    checks: List[CheckRecord] = [
        CheckRecord("tw", "terminal wealth ratio", min_path_terminal_wealth_ratio, tw_need, c_tw),
        CheckRecord("cagr", f"CAGR > {cagr_min_pct}", portfolio_cagr_pct, cagr_min_pct, c_cagr),
        CheckRecord("mdd", "MDD cap", abs(portfolio_mdd_pct), mdd_limit_pct, c_mdd),
        CheckRecord("cvar", "CVaR cap", portfolio_cvar_pct, max_cvar_pct, c_cvar),
        CheckRecord("pf", "PF floor", portfolio_pf, pf_need, c_pf),
    ]
    return GoNoGoResult(
        passed=passed,
        details={
            "tw": c_tw,
            "cagr": c_cagr,
            "mdd": c_mdd,
            "cvar": c_cvar,
            "pf": c_pf,
        },
        summary="\n".join(lines),
        checks=checks,
    )


def run_go_nogo_holdout_portfolio_growth(
    *,
    portfolio_cagr_pct: float,
    portfolio_mdd_pct: float,
    portfolio_cvar_pct: float,
    portfolio_pf: float,
    portfolio_long_trades: int,
    min_path_terminal_wealth_ratio: float,
    min_portfolio_trades: int,
    max_cvar_pct: float,
    pf_need: float = 1.6,
    cagr_min_pct: float = 25.0,
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
        portfolio_pf=portfolio_pf,
        min_path_terminal_wealth_ratio=min_path_terminal_wealth_ratio,
        max_cvar_pct=max_cvar_pct,
        pf_need=pf_need,
        cagr_min_pct=cagr_min_pct,
    )
    passed = bool(tfloor.passed and scash.passed)
    summary = tfloor.summary + "\n\n" + scash.summary
    return GoNoGoResult(
        passed=passed,
        details={**tfloor.details, **scash.details},
        summary=summary,
        checks=[*tfloor.checks, *scash.checks],
    )
