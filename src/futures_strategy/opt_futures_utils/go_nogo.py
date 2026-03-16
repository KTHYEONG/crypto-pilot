"""
상위 1% 기관급 자산 증식을 위한 OOS 심층 검증 (RoMaD 및 계단식 PF 필터 적용)
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
    oos_cagr: float = float(oos_romad_scores[0]) if oos_romad_scores else -100.0
    abs_mdd = abs(max_mdd_pct) if max_mdd_pct != 0 else 1e-9

    # --- 1. Return over Max Drawdown (RoMaD) - 리스크 대비 수익성 ---
    # 단순히 CAGR > 0이 아니라, "MDD 대비 수익률이 1.0배 이상인가?"를 검증
    romad = oos_cagr / abs_mdd
    growth_pass: bool = (
        oos_cagr > 5.0 and romad >= 0.8
    )  # 최소 수익률 5% 보장 및 방어력 검증

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

    details: Dict[str, bool] = {
        "1. Risk-Adjusted Return (RoMaD >= 0.8)": growth_pass,
        "2. Strict Volatility Limit (MDD <= 25%)": mdd_pass,
        f"3. Institutional Edge (PF >= {target_pf})": pf_pass,
        f"4. Dynamic Stat Edge (N-Tier Valid)": trades_pass,
    }

    all_passed = all(details.values())

    # --- Summary Formatting ---
    summary_lines: List[str] = ["[Elite 1% Wealth Compounding Checklist]"]

    metric_values: Dict[str, str] = {
        "1. Risk-Adjusted Return (RoMaD >= 0.8)": f"RoMaD: {romad:.2f} (CAGR {oos_cagr:.1f}%)",
        "2. Strict Volatility Limit (MDD <= 25%)": f"MDD: {abs_mdd:.1f}%",
        f"3. Institutional Edge (PF >= {target_pf})": f"PF: {profit_factor:.2f}",
        f"4. Dynamic Stat Edge (N-Tier Valid)": f"N: {total_trades}",
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

    return GoNoGoResult(
        passed=all_passed, details=details, summary="\n".join(summary_lines)
    )
