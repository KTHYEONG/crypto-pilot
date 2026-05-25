"""Phase 2: compute_v3_score 함수 검증.

사양서 §4.3 — 고정 λ 기반 6항 score 공식.
"""

from __future__ import annotations

import inspect

import numpy as np

from src.domain.futures.optimization.evaluator import compute_v3_score

# 고정 λ 상수 (사양서 §4.1)
V3_LAMBDA = {
    "down": 0.50,
    "mdd": 1.00,
    "cvar": 0.30,
    "turnover": 0.20,
    "funding": 0.50,
    "capacity": 0.40,
}


class TestScoreV3:
    """compute_v3_score 함수 검증."""

    def test_known_input_score_value(self) -> None:
        """알려진 입력에 대한 score 검증 (수동 계산값과 비교, tolerance 1e-9)."""
        # 모든 leg +5%
        leg_log_tw = np.log(np.array([1.05] * 8, dtype=np.float64))
        mean_log_tw = float(np.mean(leg_log_tw))  # ≈ 0.04879

        # 다운사이드 세미편차 (음수 초과 수익률 없음 → 0)
        downside_vals = leg_log_tw[leg_log_tw < 0.0]
        semidev = float(np.std(downside_vals)) if len(downside_vals) > 1 else 0.0

        mdd = 0.10
        cvar = 0.05
        excess_turnover = 0.10
        funding_drag = 0.02
        aum_impact = 0.01

        # 수동 계산:
        # score = mean_log_tw
        #         - λ_down * semidev
        #         - λ_mdd * mdd
        #         - λ_cvar * cvar
        #         - λ_turnover * excess_turnover
        #         - λ_funding * funding_drag
        #         - λ_capacity * aum_impact
        expected = (
            mean_log_tw
            - V3_LAMBDA["down"] * semidev
            - V3_LAMBDA["mdd"] * mdd
            - V3_LAMBDA["cvar"] * cvar
            - V3_LAMBDA["turnover"] * excess_turnover
            - V3_LAMBDA["funding"] * funding_drag
            - V3_LAMBDA["capacity"] * aum_impact
        )

        actual = compute_v3_score(
            leg_log_tw=leg_log_tw,
            worst_mdd=mdd,
            cvar_5=cvar,
            excess_turnover=excess_turnover,
            funding_drag=funding_drag,
            aum_impact_penalty=aum_impact,
        )

        assert abs(actual - expected) < 1e-9, f"score 불일치: actual={actual}, expected={expected}"

    def test_lambda_is_fixed_not_injectable(self) -> None:
        """λ 파라미터가 외부 주입 불가 (함수 시그니처에 없어야 함)."""
        sig = inspect.signature(compute_v3_score)
        forbidden_params = [
            "lambda_down",
            "lambda_mdd",
            "lambda_cvar",
            "lambda_turnover",
            "lambda_funding",
            "lambda_capacity",
            "lambdas",
            "lam",
            "penalty_weights",
        ]
        for p in forbidden_params:
            assert p not in sig.parameters, f"λ 파라미터 '{p}'가 외부 주입 가능해서는 안 됨"

    def test_loss_strategy_lower_than_profit_strategy(self) -> None:
        """음의 score 전략 vs 양의 score 전략 순위."""
        # 손실 전략: 모든 leg 음수
        leg_loss = np.log(np.array([0.90] * 8, dtype=np.float64))
        score_loss = compute_v3_score(
            leg_log_tw=leg_loss,
            worst_mdd=0.40,
            cvar_5=0.20,
            excess_turnover=0.50,
            funding_drag=0.15,
            aum_impact_penalty=0.10,
        )

        # 수익 전략: 모든 leg 양수
        leg_profit = np.log(np.array([1.05] * 8, dtype=np.float64))
        score_profit = compute_v3_score(
            leg_log_tw=leg_profit,
            worst_mdd=0.05,
            cvar_5=0.02,
            excess_turnover=0.05,
            funding_drag=0.01,
            aum_impact_penalty=0.01,
        )

        assert score_loss < score_profit, (
            f"손실 전략({score_loss:.4f})이 수익 전략({score_profit:.4f})보다 낮아야 함"
        )

    def test_zero_penalties_score_equals_mean_log_tw(self) -> None:
        """모든 패널티 0이면 score = mean(leg_log_tw)."""
        leg_log_tw = np.array([0.02, 0.03, 0.04, 0.05, 0.03, 0.04, 0.02, 0.03])
        expected = float(np.mean(leg_log_tw))

        actual = compute_v3_score(
            leg_log_tw=leg_log_tw,
            worst_mdd=0.0,
            cvar_5=0.0,
            excess_turnover=0.0,
            funding_drag=0.0,
            aum_impact_penalty=0.0,
        )
        # 다운사이드 세미편차 항이 없으면 정확히 일치
        assert abs(actual - expected) < 1e-9
