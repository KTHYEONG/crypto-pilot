import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Tuple

def compute_vol_adj_forward_returns(df: pd.DataFrame, horizon: int = 6) -> np.ndarray:
    """
    Target_t = (Close_t+h - Close_t) / Close_t / ATR_t(14)
    변동성으로 정규화된 미래 수익률을 계산하여 시그널의 예측 강도를 표준화함.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    
    n = len(close)
    if n < 20:
        return np.full(n, np.nan)

    # 14-period ATR for normalization
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().to_numpy()
    atr = np.maximum(atr, 1e-9)
    
    # Forward returns (shifted back)
    # t 시점의 수익률은 t+horizon 시점의 가격 변화
    fwd_ret = np.full(n, np.nan)
    if n > horizon:
        fwd_ret[:-horizon] = (close[horizon:] - close[:-horizon]) / close[:-horizon]
    
    vol_adj_ret = fwd_ret / atr
    return vol_adj_ret

def calculate_spearman_ic(signal_scores: np.ndarray, target_returns: np.ndarray) -> float:
    """
    Spearman Rank Correlation (IC) 계산.
    연속형 시그널 점수와 미래 수익률 간의 상관관계를 측정함.
    """
    if len(signal_scores) != len(target_returns):
        return 0.0
        
    mask = ~np.isnan(signal_scores) & ~np.isnan(target_returns)
    if np.sum(mask) < 50: # 최소 샘플 사이즈 검증
        return 0.0
    
    # rank_score가 모두 동일한 값인 경우 (상관관계 계산 불가) 처리
    if np.unique(signal_scores[mask]).size < 2:
        return 0.0
        
    ic, _ = spearmanr(signal_scores[mask], target_returns[mask])
    return float(ic) if not np.isnan(ic) else 0.0

def calculate_conditional_ic(
    signal_scores: np.ndarray, 
    target_returns: np.ndarray, 
    regime_mask: np.ndarray
) -> Tuple[float, float]:
    """
    특정 Regime이 활성화된 구간에서의 조건부 IC(cIC) 및 커버리지 계산.
    """
    if len(signal_scores) != len(regime_mask):
        return 0.0, 0.0
        
    active_mask = (regime_mask > 0.5) # Boolean mask
    
    # Regime이 활성화된 구간의 데이터만 추출
    active_scores = signal_scores[active_mask]
    active_returns = target_returns[active_mask]
    
    ic = calculate_spearman_ic(active_scores, active_returns)
    coverage = float(np.mean(active_mask))
    return ic, coverage
