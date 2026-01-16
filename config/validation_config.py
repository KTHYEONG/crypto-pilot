
# Validation Thresholds & Standards

# 1. Walk-Forward Analysis Criteria
WFA_MIN_ROBUSTNESS_SCORE = 0.3
WFA_MIN_CONSISTENCY_RATIO = 0.6  # At least 60% of periods must be profitable

# 2. Monte Carlo Criteria
MC_MIN_PROB_PROFIT = 85.0  # 85% probability of profit
MC_MAX_WORST_CASE_MDD = -30.0  # Worst case MDD should not exceed -30%

# 3. Statistical Significance
STAT_SIGNIFICANCE_ALPHA = 0.05  # p-value < 0.05

# 4. Financial Metrics (Minimum Requirements)
MIN_SHARPE_RATIO = 1.0
MIN_SORTINO_RATIO = 1.5
MAX_MDD_PCT = -25.0
MIN_WIN_RATE = 40.0

# 5. Data Sufficiency
MIN_TRADES_COUNT = 30
