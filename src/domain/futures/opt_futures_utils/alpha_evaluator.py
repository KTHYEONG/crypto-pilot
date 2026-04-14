import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Tuple, Dict, List, Optional

def compute_vol_adj_forward_returns(
    df: pd.DataFrame, 
    horizons: List[int] = [2, 6, 12],
    market_returns: Optional[Dict[int, np.ndarray]] = None
) -> Dict[int, np.ndarray]:
    """
    여러 호흡(Horizons)에 대해 변동성으로 정규화된 미래 수익률을 계산함.
    market_returns가 제공되면 마켓 베타를 제거한 Idiosyncratic Return을 계산함 (Step 4).
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    
    n = len(close)
    if n < 20:
        return {h: np.full(n, np.nan) for h in horizons}

    # 14-period ATR for normalization
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().to_numpy()
    atr = np.maximum(atr, 1e-9)
    
    results = {}
    for h in horizons:
        fwd_ret = np.full(n, np.nan)
        if n > h:
            fwd_ret[:-h] = (close[h:] - close[:-h]) / close[:-h]
        
        vol_adj_ret = fwd_ret / atr
        
        # Step 4: Market Beta Scrubbing
        if market_returns and h in market_returns:
            m_ret = market_returns[h]
            mask = ~np.isnan(vol_adj_ret) & ~np.isnan(m_ret)
            if np.sum(mask) > 50:
                # y = alpha + beta * x
                beta = np.cov(vol_adj_ret[mask], m_ret[mask])[0, 1] / (np.var(m_ret[mask]) + 1e-9)
                vol_adj_ret = vol_adj_ret - beta * m_ret
                
        results[h] = vol_adj_ret
        
    return results

def calculate_spearman_ic(signal_scores: np.ndarray, target_returns: np.ndarray) -> float:
    """Spearman Rank Correlation (IC) 계산."""
    if len(signal_scores) != len(target_returns):
        return 0.0
        
    mask = ~np.isnan(signal_scores) & ~np.isnan(target_returns)
    if np.sum(mask) < 50:
        return 0.0
    
    if np.unique(signal_scores[mask]).size < 2:
        return 0.0
        
    ic, _ = spearmanr(signal_scores[mask], target_returns[mask])
    return float(ic) if not np.isnan(ic) else 0.0

def calculate_residual_score(candidate_scores: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    """Candidate 시그널에서 Base 시그널의 선형 성분을 제거한 잔차 점수 계산."""
    mask = ~np.isnan(candidate_scores) & ~np.isnan(base_scores)
    if np.sum(mask) < 50:
        return candidate_scores
        
    x = base_scores[mask]
    y = candidate_scores[mask]
    beta = np.cov(x, y)[0, 1] / (np.var(x) + 1e-9)
    residual = candidate_scores - beta * base_scores
    return residual

def calculate_conditional_ic(
    signal_scores: np.ndarray, 
    target_returns: np.ndarray, 
    regime_mask: np.ndarray
) -> Tuple[float, float]:
    """특정 Regime 구간에서의 조건부 IC 및 커버리지 계산."""
    if len(signal_scores) != len(regime_mask):
        return 0.0, 0.0
        
    active_mask = (regime_mask > 0.5)
    active_scores = signal_scores[active_mask]
    active_returns = target_returns[active_mask]
    
    ic = calculate_spearman_ic(active_scores, active_returns)
    coverage = float(np.mean(active_mask))
    return ic, coverage
