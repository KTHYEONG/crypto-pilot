
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365):
    """
    Calculate Sharpe Ratio
    Formula: (Mean Return - Risk Free Rate) / Std Dev of Return
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns, ddof=1)
    
    if std_dev == 0:
        return 0.0
        
    # Annualize based on periods (Crypto: 365 days)
    # If returns are daily, sqrt(365). If hourly, sqrt(365*24).
    # For simplicity, we assume the input 'returns' is already aligned to the period unit.
    # But usually Sharpe is reported annualized.
    
    # Assuming input 'returns' is a series of per-trade returns or daily returns.
    # If per-trade, annualization is tricky without trade frequency.
    # Standard approach: annualized_return / annualized_volatility
    
    daily_sharpe = mean_excess_return / std_dev
    annualized_sharpe = daily_sharpe * np.sqrt(periods_per_year)
    
    return annualized_sharpe

def calculate_sortino_ratio(returns, target_return=0.0, periods_per_year=365):
    """
    Calculate Sortino Ratio
    Formula: (Mean Return - Target Return) / Downside Deviation
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 100.0 # No downside risk
        
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 100.0
        
    annualized_sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)
    return annualized_sortino

def calculate_calmar_ratio(annualized_return_pct, max_drawdown_pct):
    """
    Calculate Calmar Ratio
    Formula: Annualized Return / Max Drawdown
    """
    if max_drawdown_pct == 0:
        return 100.0
    return annualized_return_pct / abs(max_drawdown_pct)

def calculate_var(returns, confidence=0.95):
    """
    Calculate Value at Risk (VaR)
    """
    if len(returns) == 0:
        return 0.0
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_cvar(returns, confidence=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) a.k.a Expected Shortfall
    """
    if len(returns) == 0:
        return 0.0
    var = calculate_var(returns, confidence)
    tail_losses = returns[returns <= var]
    
    if len(tail_losses) == 0:
        return var
        
    return np.mean(tail_losses)

def calculate_omega_ratio(returns, threshold=0.0):
    """
    Calculate Omega Ratio
    Formula: Probability weighted gains / Probability weighted losses
    """
    gains = returns[returns > threshold]
    losses = returns[returns <= threshold]
    
    sum_gains = np.sum(gains - threshold)
    sum_losses = np.sum(threshold - losses)
    
    if sum_losses == 0:
        return 100.0
        
    return sum_gains / sum_losses
