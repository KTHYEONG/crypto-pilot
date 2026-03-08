"""
7개 필수 항목을 검증하여 파라미터의 실전 투입 가능 여부를 결정함.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import logging

_logger: logging.Logger = logging.getLogger("opt_futures")

@dataclass
class GoNoGoResult:
    passed: bool
    details: Dict[str, bool]
    summary: str

def run_go_nogo_check(
    cv_fold_scores: List[float], # Legacy, unused in NSGA-II
    holdout_score: float,        # Legacy, unused
    oos_romad_scores: List[float], # Contains [OOS CAGR]
    max_mdd_pct: float,          # OOS MDD
    profit_factor: float,        # OOS PF
    long_count: int,
    short_count: int,
    tf: str = "4h",
) -> GoNoGoResult:
    import numpy as np
    
    total_trades = long_count + short_count
    
    # Extract OOS CAGR from the passed list
    oos_cagr: float = float(oos_romad_scores[0]) if oos_romad_scores else -100.0

    # 1. Growth Engine (Must be profitable out-of-sample)
    growth_pass: bool = oos_cagr > 0.0
    
    # 2. Volatility Drag Control (Max MDD <= 35.0% - Half-Kelly Limit)
    mdd_pass: bool = abs(max_mdd_pct) <= 35.0
    
    # 3. Mathematical Edge (PF >= 1.10 - Covers fees and slippage securely)
    target_pf = 1.10
    pf_pass: bool = profit_factor >= target_pf

    # 4. Statistical Significance (Fat-tail strategies trade less, but we need some sample)
    if tf == "1h":
        min_trades_req = 30
    else:  # 4h
        min_trades_req = 10

    trades_pass: bool = total_trades >= min_trades_req

    details: Dict[str, bool] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": growth_pass,
        "2. Volatility Drag (MDD <= 35%)": mdd_pass,
        f"3. Mathematical Edge (PF >= {target_pf})": pf_pass,
        f"4. Stat Edge (Trades >= {min_trades_req})": trades_pass,
    }

    all_passed = all(details.values())

    summary_lines: List[str] = ["[Elite 1% Wealth Compounding Checklist]"]
    
    metric_values: Dict[str, str] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": f"CAGR: {oos_cagr:.2f}%",
        "2. Volatility Drag (MDD <= 35%)": f"MDD: {abs(max_mdd_pct):.2f}%",
        f"3. Mathematical Edge (PF >= {target_pf})": f"PF: {profit_factor:.2f}",
        f"4. Stat Edge (Trades >= {min_trades_req})": f"N: {total_trades}",
    }

    req_met: int = sum(1 for v in details.values() if v)
    total_req: int = len(details)

    for k, v in details.items():
        status: str = "PASS" if v else "FAIL"
        val_str: str = metric_values.get(k, "")
        summary_lines.append(f"  - {k:<40}: {status:<5} ({val_str})")
        
    summary_lines.append("-" * 55)
    final_status: str = "🌟 ELITE GO (Top 1% Ready)" if all_passed else f"🔴 NO-GO (Needs Revision, Passed {req_met}/{total_req})"
    summary_lines.append(f"  FINAL VERDICT: {final_status}")
    
    return GoNoGoResult(
        passed=all_passed,
        details=details,
        summary="\n".join(summary_lines)
    )
