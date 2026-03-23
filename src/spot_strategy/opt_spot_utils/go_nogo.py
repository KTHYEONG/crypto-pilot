from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import logging

_logger: logging.Logger = logging.getLogger("opt_spot")

@dataclass
class GoNoGoResult:
    passed: bool
    details: Dict[str, bool]
    summary: str

def run_go_nogo_check(
    cv_fold_scores: List[float], 
    holdout_score: float,        
    oos_romad_scores: List[float], 
    max_mdd_pct: float,          
    profit_factor: float,        
    long_count: int,
    tf: str = "4h",
) -> GoNoGoResult:
    
    total_trades = long_count
    
    oos_cagr: float = float(oos_romad_scores[0]) if oos_romad_scores else -100.0

    growth_pass: bool = oos_cagr > 0.0
    mdd_pass: bool = abs(max_mdd_pct) <= 35.0
    
    target_pf = 1.10
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

    summary_lines: List[str] = ["[Spot Wealth Compounding Checklist]"]
    
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
