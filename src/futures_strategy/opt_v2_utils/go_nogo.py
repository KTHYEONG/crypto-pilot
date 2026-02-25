"""
7개 필수 항목을 검증하여 파라미터의 실전 투입 가능 여부를 결정함.
"""
from dataclasses import dataclass
from typing import Dict, List
import logging

_logger: logging.Logger = logging.getLogger("opt_v2")

@dataclass
class GoNoGoResult:
    passed: bool
    details: Dict[str, bool]
    summary: str

def run_go_nogo_check(
    cv_fold_scores: List[float],
    holdout_score: float,
    oos_romad_scores: List[float],
    cross_sym_romad_scores: List[float],
    max_mdd_pct: float,
    profit_factor: float,
    long_count: int,
    short_count: int,
) -> GoNoGoResult:
    
    total_trades = long_count + short_count
    
    # 1. 모든 CV Fold RoMaD > 0
    all_cv_positive = all(s > 0 for s in cv_fold_scores) if cv_fold_scores else False
    
    # 2. Hold-out Score > 0
    holdout_positive = (holdout_score > 0)
    
    # 3. True OOS RoMaD > 0 (타겟 심볼 중 최소 1개 이상 양수)
    true_oos_positive = any(s > 0 for s in oos_romad_scores) if oos_romad_scores else False
    
    # 4. Cross-Symbol 50% 이상 양수
    if not cross_sym_romad_scores:
        cross_sym_pass = False
    else:
        positive_count = sum(1 for s in cross_sym_romad_scores if s > 0)
        cross_sym_pass = (positive_count / len(cross_sym_romad_scores)) >= 0.5

    # 5. Max MDD < 25%
    mdd_pass = (abs(max_mdd_pct) < 25.0)
    
    # 6. Profit Factor >= 1.3
    pf_pass = (profit_factor >= 1.3)
    
    # 7. Long/Short 균형 (각각 최소 10% 이상 대기)
    long_ratio = (long_count / total_trades) if total_trades > 0 else 0.0
    short_ratio = (short_count / total_trades) if total_trades > 0 else 0.0
    ls_balance_pass = (long_ratio >= 0.1) and (short_ratio >= 0.1)

    details = {
        "1. All CV Folds > 0": all_cv_positive,
        "2. Hold-out Score > 0": holdout_positive,
        "3. True OOS Score > 0": true_oos_positive,
        "4. Cross-Sym >= 50% Positive": cross_sym_pass,
        "5. Max MDD < 25%": mdd_pass,
        "6. Profit Factor >= 1.3": pf_pass,
        "7. Long/Short Balance >= 10%": ls_balance_pass,
    }

    all_passed = all(details.values())

    summary_lines = []
    summary_lines.append("[Go/No-Go Checklist]")
    for k, v in details.items():
        status = "PASS" if v else "FAIL"
        summary_lines.append(f"  - {k:<30}: {status}")
    summary_lines.append("--------------------------------------------------")
    final_status = "🟢 GO (Ready for Live)" if all_passed else "🔴 NO-GO (Needs Revision)"
    summary_lines.append(f"  FINAL VERDICT: {final_status}")
    
    return GoNoGoResult(
        passed=all_passed,
        details=details,
        summary="\n".join(summary_lines)
    )
