
import numpy as np
from scipy import stats

def perform_t_test(returns, population_mean=0.0):
    """
    Perform one-sample t-test
    H0: Mean of returns = population_mean (usually 0)
    Returns: t-statistic, p-value
    """
    if len(returns) < 2:
        return 0.0, 1.0
        
    t_stat, p_value = stats.ttest_1samp(returns, population_mean)
    return t_stat, p_value

def is_statistically_significant(returns, alpha=0.05):
    """
    Check if the returns are significantly different from 0
    """
    _, p_value = perform_t_test(returns)
    return p_value < alpha

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the mean
    Returns: (lower_bound, upper_bound)
    """
    if len(data) < 2:
        return np.mean(data), np.mean(data)
        
    mean = np.mean(data)
    sem = stats.sem(data) # Standard Error of Mean
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    
    return mean - margin, mean + margin

def calculate_outlier_thresholds(data, z_score_threshold=3.0):
    """
    Identify outlier thresholds using Z-Score
    """
    mean = np.mean(data)
    std = np.std(data)
    return mean - z_score_threshold * std, mean + z_score_threshold * std
