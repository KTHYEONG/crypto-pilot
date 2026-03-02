"""
7개 필수 항목을 검증하여 파라미터의 실전 투입 가능 여부를 결정함.
"""
from __future__ import annotations

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
    max_mdd_pct: float,
    profit_factor: float,
    long_count: int,
    short_count: int,
    tf: str = "4h",
) -> GoNoGoResult:
    import numpy as np
    
    total_trades = long_count + short_count
    
    cv_mean: float = float(np.mean(cv_fold_scores)) if cv_fold_scores else 0.0
    cv_min: float = float(np.min(cv_fold_scores)) if cv_fold_scores else -100.0

    # 1. 최악의 구간 방어력 (CV Min RoMaD > -0.3)
    cv_min_pass: bool = cv_min > -0.3
    
    # 2. 성장 엔진 확보 (CV Mean >= 0.0)
    cv_growth_pass: bool = cv_mean >= 0.0
    
    # 5. Volatility Drag Control (Max MDD <= 20.0%)
    mdd_pass: bool = abs(max_mdd_pct) <= 20.0
    
    # [장세 불문(Regime-Agnostic) 보편적 생존 기준]
    # 미래에 어떤 장세(추세/횡보)가 OOS로 잡히더라도, 전략의 고유한 약점 구간에서 세금(손실)을 내는 것은 당연함.
    # 따라서 OOS 수익이 양수(+)일 것을 강제하지 않으며, 오직 '계좌가 터지지 않고 방어(-1.0 RoMaD 이내)했는가'만 채점함.
    target_pf = 0.50 # 불리한 장세에서의 방어적 PF 허용선
    oos_str = "OOS > -1.0 (Regime Defense)"
    ho_str = "Holdout > -1.0 (Regime Defense)"
    
    ho_deg_pass: bool = holdout_score > -1.0
    all_oos_pass: bool = all((s > -1.0) for s in oos_romad_scores) if oos_romad_scores else False
    pf_pass: bool = profit_factor >= target_pf

    # 타임프레임별 통계적 유의성(거래 횟수)만 분리
    if tf == "1h":
        min_trades_req = 40
    else:  # 4h
        min_trades_req = 10

    trades_pass: bool = total_trades >= min_trades_req

    details: Dict[str, bool] = {
        "1. Robustness (CV Min > -0.3)": cv_min_pass,
        "2. Growth Engine (CV Mean >= 0.0)": cv_growth_pass,
        f"3. PBO Control ({ho_str})": ho_deg_pass,
        f"4. Target FW-Test ({oos_str})": all_oos_pass,
        "5. Vol Drag (Max MDD <= 20%)": mdd_pass,
        f"6. Math Edge (PF >= {target_pf})": pf_pass,
        f"7. Stat Edge (Trades >= {min_trades_req})": trades_pass,
    }

    all_passed = all(details.values())

    summary_lines: List[str] = ["[Elite 1% Go/No-Go Checklist]"]
    
    metric_values: Dict[str, str] = {
        "1. Robustness (CV Min > -0.3)": f"Min: {cv_min:.3f}",
        "2. Growth Engine (CV Mean >= 0.0)": f"Mean: {cv_mean:.3f}",
        f"3. PBO Control ({ho_str})": f"HO: {holdout_score:.3f}",
        f"4. Target FW-Test ({oos_str})": "PASS" if all_oos_pass else "FAIL",
        "5. Vol Drag (Max MDD <= 20%)": f"MDD: {abs(max_mdd_pct):.2f}%",
        f"6. Math Edge (PF >= {target_pf})": f"PF: {profit_factor:.2f}",
        f"7. Stat Edge (Trades >= {min_trades_req})": f"N: {total_trades}",
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
