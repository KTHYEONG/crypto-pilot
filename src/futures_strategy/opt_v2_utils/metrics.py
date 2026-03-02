"""
백테스트 결과인 손익 데이터를 바탕으로 수익률, MDD, RoMaD 등 정량적 성과 지표를 계산함.
전략의 우수성을 판단하기 위해 단순 수익률뿐만 아니라 리스크 대비 효율성을 점수화하는 역할을 수행함.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from config.settings import FUTURES_INITIAL_BALANCE

# Maximum annualization factor is no longer used for capping in V2.7 Time-Normalized logic.
_MAX_ANNUALIZATION_FACTOR: float = 1.0 

def calc_romad(pnl_series: pd.Series, n_trades: int, tf: str) -> Tuple[float, float, float]:
    """
    Calculate Return on Max Drawdown (RoMaD) as the primary risk-adjusted metric.
    Standardized to mirror the enhanced logic in calc_romad_from_metrics.
    """
    if pnl_series.empty or n_trades == 0:
        return -10.0, 0.0, 0.0

    equity: pd.Series = FUTURES_INITIAL_BALANCE + pnl_series.cumsum()
    end_equity: float = float(equity.iloc[-1])
    ret_pct: float = ((end_equity / FUTURES_INITIAL_BALANCE) - 1.0) * 100.0

    running_max: np.ndarray = np.maximum.accumulate(equity.values)
    running_max[running_max == 0] = 1e-9
    drawdown: np.ndarray = (equity.values - running_max) / running_max * 100.0
    mdd_pct: float = abs(float(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    span_days: float = float((pnl_series.index[-1] - pnl_series.index[0]).total_seconds() / 86400.0)
    win_rate: float = float((pnl_series > 0).mean() * 100.0) if not pnl_series.empty else 0.0

    if pnl_series.empty:
        pf: float = 1.0
    else:
        gains: float = float(pnl_series[pnl_series > 0].sum())
        losses: float = abs(float(pnl_series[pnl_series < 0].sum()))
        pf = float(gains / losses) if losses > 0 else (999.0 if gains > 0 else 1.0)

    return calc_romad_from_metrics(
        ret_pct=ret_pct,
        mdd_pct=mdd_pct,
        n_trades=n_trades,
        tf=tf,
        span_days=span_days,
        win_rate=win_rate,
        pf=pf,
        leverage=1.0,
    )

def calc_profit_factor_from_pnl(pnl_series: pd.Series) -> float:
    """Calculate Profit Factor from a pre-computed net PNL series (fee-deducted)."""
    if pnl_series.empty:
        return 1.0

    gross_profit: float = float(pnl_series[pnl_series > 0].sum())
    gross_loss: float = abs(float(pnl_series[pnl_series < 0].sum()))

    if gross_loss == 0.0:
        return 999.0 if gross_profit > 0 else 1.0

    return gross_profit / gross_loss


def calc_profit_factor(trades_df: pd.DataFrame) -> float:
    """Calculate Profit Factor (Gross Profit / Gross Loss) from raw trades_df."""
    if trades_df.empty:
        return 1.0
        
    gains: pd.Series = trades_df["pnl"][trades_df["pnl"] > 0]
    losses: pd.Series = trades_df["pnl"][trades_df["pnl"] < 0]
    
    gross_profit: float = float(gains.sum()) if not gains.empty else 0.0
    gross_loss: float = abs(float(losses.sum())) if not losses.empty else 0.0
    
    if gross_loss == 0.0:
        return 999.0 if gross_profit > 0 else 1.0
        
    return gross_profit / gross_loss

def calc_romad_from_metrics(
    ret_pct: float,
    mdd_pct: float,
    n_trades: int,
    tf: str,
    span_days: float,
    win_rate: float = 0.0,
    pf: float = 1.0,
    leverage: float = 1.0,
) -> Tuple[float, float, float]:
    """
    RoMaD score from precomputed return and MDD with strict financial engineering constraints.
    Returns: (score, return_pct, mdd_pct).
    
    Reflected Plan Details:
    1. 보편적 시간 정규화 (Time-Normalized Log-Return): 순수한 '일별 평균 복리 성장률' 기준.
    2. Log-Drawdown 기반 페널티: 위험과 수익의 수학적 일관성 유지.
    3. 표본 횟수 기반 통계적 페널티 (Standard Error Penalty): C/sqrt(N) 원리 차용.
    4. Profit Factor Sigmoid Credibility (Secondary Filter): PF 기반 신뢰도 점수.
    """
    # 1. 보편적 시간 정규화 (Time-Normalized Log-Return) — 복리 엔진 정합성 유지
    days: float = max(float(span_days), 90.0)
    mdd_abs: float = float(abs(mdd_pct))
    total_ret_ratio: float = 1.0 + (ret_pct / 100.0)
    # 1-1. TPE 하한선 그래디언트 소실 교정 (Log-Extension): total_ret_ratio < 0.01 구간에서 접선 기반 1차 선형 보간
    if total_ret_ratio >= 0.01:
        log_ret: float = float(np.log(total_ret_ratio))
    else:
        log_ret: float = float(np.log(0.01)) + 100.0 * (total_ret_ratio - 0.01)
    
    annualized_g: float = (log_ret / days) * 365.0

    # 2. Log-Drawdown 기반 페널티 (Scale alignment with Return)
    mdd_decimal: float = min(mdd_abs / 100.0, 0.999)
    risk_cost: float = -float(np.log(1.0 - mdd_decimal))
    
    # 2-1. MDD 20% 초과 시 비선형 페널티 (스케일 조정: 5.0 -> 1.0)
    if mdd_decimal > 0.20:
        risk_cost += ((mdd_decimal - 0.20) * 1.0) ** 2

    base_score: float = annualized_g - risk_cost

    # 3. 구간형 표본 횟수 기반 페널티 (Dynamic TF-based Penalty)
    # 타임프레임별 기대 거래 빈도(EDF), 일일 봉 개수, 최대 봉 비율(max_ratio), Sniper PF 기준
    tf_params_map: dict[str, tuple[float, float, float, float]] = {
        # 1H (Mean Reversion): 통계적 검증을 위해 최소 이틀에 1번 꼴(0.5) 거래 권장. 하루 최대 2.4번(0.10)까지 정상 허용.
        # 승률로 엣지를 내므로 PF 1.3 이상이면 훌륭한 전략으로 간주.
        "1h": (0.50, 24.0, 0.10, 1.3),
        
        # 4H (Trend Following): 매크로 대추세이므로 최소 25일에 1번(0.04) 꼴 권장. 잦은 진입은 휩쏘로 간주(0.03).
        # 손익비가 높아야 하므로 PF 1.5 이상일 때만 면제.
        "4h": (0.04, 6.0, 0.03, 1.5),
    }
    edf, bars_per_day, max_ratio, sniper_pf = tf_params_map.get(tf, (0.1, 24.0, 0.03, 1.5))
    
    # N_MIN: 통계적 유의성 하한선 (분기별 테스트 기준에 맞춰 절대 하한선을 3.0으로 완화)
    # 우연한 1~2번의 수익(Curve-fitting)은 걸러내되, 추세추종의 본질을 훼손하지 않는 타협점.
    N_MIN: float = max(3.0, float(span_days) * edf)
    # N_MAX: 노이즈 과적합(Turnover) 방어선 - 타임프레임 성격에 맞춘 상한선
    N_MAX: float = float(span_days) * bars_per_day * max_ratio

    # 3. Monotonic Penalty (Directional Pruning for TPE)
    if n_trades < N_MIN:
        dist_factor: float = (N_MIN - float(n_trades)) / N_MIN
        # Soft Penalty for Sniper Strategies (High PF) -> 적게 매매하되 훌륭한 엣지가 있는 전략 보존
        if pf >= sniper_pf:
            final_score: float = base_score - (dist_factor * 0.5)
        else:
            # TPE Gradient 보존을 위한 연속 선형 감점 (최대 -3.0)
            final_score: float = base_score - (dist_factor * 3.0)
        return float(final_score), ret_pct, mdd_abs
    
    if n_trades > N_MAX:
        # 과도한 매매 시 로그 스케일로 부드럽게 감점
        dist_factor: float = float(np.log(float(n_trades) / N_MAX))
        final_score: float = base_score - (dist_factor * 2.0)
        return float(final_score), ret_pct, mdd_abs
    
    # 4. Profit Factor 가산 보너스 (가산형 전환, 음수 구간 변별력 확보)
    pf_bonus: float = float(np.tanh(2.0 * (pf - 1.0))) * 0.1

    final_score: float = base_score + pf_bonus
    
    return float(final_score), ret_pct, mdd_abs
