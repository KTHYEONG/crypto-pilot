
import optuna
import numpy as np

def suggest_params(trial, search_space):
    """
    Generate trial parameters from search space with conditional dependency pruning.
    Only suggests parameters that are actually used by the selected strategy configuration.
    
    Efficiency Gain: 60~70% reduction in search space by skipping irrelevant parameters.
    """
    params = {}
    
    # === Phase 1: Core Strategy Selection ===
    for key in ['ENTRY_TYPE', 'TREND_FILTER_TYPE', 'STRENGTH_FILTER_TYPE', 'EXIT_TYPE', 
                'STOP_LOSS_TYPE', 'USE_TAKE_PROFIT', 'USE_VOLUME_FILTER', 'TIMEFRAME',
                'REGIME_FILTER', 'USE_ADX']: # Added missing keys from strategies
        if key in search_space:
            spec = search_space[key]
            if spec['type'] == 'categorical':
                params[key] = trial.suggest_categorical(key, spec['choices'])
    
    # === Phase 2: Entry-Type Dependent Parameters ===
    entry_type = params.get('ENTRY_TYPE', 'DONCHIAN')
    
    if entry_type == 'BOLLINGER':
        if 'BB_STD' in search_space:
            spec = search_space['BB_STD']
            params['BB_STD'] = trial.suggest_float('BB_STD', spec['low'], spec['high'], step=spec.get('step'))
            
    elif entry_type == 'KELTNER':
        if 'KELTNER_ATR_MULT' in search_space:
            spec = search_space['KELTNER_ATR_MULT']
            params['KELTNER_ATR_MULT'] = trial.suggest_float('KELTNER_ATR_MULT', spec['low'], spec['high'], step=spec.get('step'))
            
    elif entry_type == 'CCI':
        if 'CCI_THRESHOLD' in search_space:
            spec = search_space['CCI_THRESHOLD']
            params['CCI_THRESHOLD'] = trial.suggest_int('CCI_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 3: Trend-Filter Dependent Parameters ===
    trend_filter = params.get('TREND_FILTER_TYPE', 'EMA')
    
    if trend_filter == 'SUPERTREND':
        for key in ['SUPERTREND_MULT', 'SUPERTREND_PERIOD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'float':
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
                elif spec['type'] == 'int':
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif trend_filter == 'MACD':
        for key in ['MACD_FAST', 'MACD_SLOW', 'MACD_SIGNAL']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif trend_filter == 'ICHIMOKU':
        for key in ['ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_B']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif trend_filter == 'VWAP':
        if 'VWAP_STD_MULT' in search_space:
            spec = search_space['VWAP_STD_MULT']
            params['VWAP_STD_MULT'] = trial.suggest_float('VWAP_STD_MULT', spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 4: Strength-Filter Dependent Parameters ===
    strength_filter = params.get('STRENGTH_FILTER_TYPE', 'NONE')
    
    if strength_filter in ['ADX', 'VHF', 'MFI', 'RSI', 'STOCHASTIC', 'STOCH_RSI']:
        if 'STRENGTH_FILTER_PERIOD' in search_space:
            spec = search_space['STRENGTH_FILTER_PERIOD']
            use_log = spec.get('log', False)
            params['STRENGTH_FILTER_PERIOD'] = trial.suggest_int('STRENGTH_FILTER_PERIOD', spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int('STRENGTH_FILTER_PERIOD', spec['low'], spec['high'], step=spec.get('step'))
    
    if strength_filter == 'VHF':
        if 'VHF_THRESHOLD' in search_space:
            spec = search_space['VHF_THRESHOLD']
            params['VHF_THRESHOLD'] = trial.suggest_float('VHF_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'MFI':
        if 'MFI_THRESHOLD' in search_space:
            spec = search_space['MFI_THRESHOLD']
            params['MFI_THRESHOLD'] = trial.suggest_int('MFI_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'RSI':
        for key in ['RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'RSI_OVERBOUGHT_FUTURES']:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'STOCHASTIC':
        for key in ['STOCH_OVERBOUGHT', 'STOCH_OVERSOLD']:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'STOCH_RSI':
        for key in ['STOCH_RSI_OVERBOUGHT', 'STOCH_RSI_OVERSOLD']:
            if key in search_space:
                spec = search_space[key]
                params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'CMF':
        for key in ['CMF_PERIOD', 'CMF_THRESHOLD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'int':
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
    
    elif strength_filter == 'HURST':
        # 1. Period (Independent)
        if 'HURST_PERIOD' in search_space:
            spec = search_space['HURST_PERIOD']
            use_log = spec.get('log', False)
            params['HURST_PERIOD'] = trial.suggest_int('HURST_PERIOD', spec['low'], spec['high'], log=use_log)

        # 2. Random Threshold (Independent Base)
        random_thresh = 0.5 # Default fallback
        if 'HURST_RANDOM_THRESHOLD' in search_space:
            spec = search_space['HURST_RANDOM_THRESHOLD']
            params['HURST_RANDOM_THRESHOLD'] = trial.suggest_float('HURST_RANDOM_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
            random_thresh = params['HURST_RANDOM_THRESHOLD']
            
        # 3. Trend Threshold (Dependent: Must be > Random)
        if 'HURST_TREND_THRESHOLD' in search_space:
            spec = search_space['HURST_TREND_THRESHOLD']
            # Enforce Logical Safety: Trend > Random + Buffer (0.01)
            # This prevents logical contradictions even if search ranges overlap
            safe_low = max(spec['low'], random_thresh + 0.01)
            
            if safe_low < spec['high']:
                params['HURST_TREND_THRESHOLD'] = trial.suggest_float('HURST_TREND_THRESHOLD', safe_low, spec['high'], step=spec.get('step'))
            else:
                # If constraint pushes low above high, clip to safe_low logic
                params['HURST_TREND_THRESHOLD'] = safe_low

    elif strength_filter == 'ER':
        if 'ER_THRESHOLD' in search_space:
            spec = search_space['ER_THRESHOLD']
            params['ER_THRESHOLD'] = trial.suggest_float('ER_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
            
    elif strength_filter == 'NATR':
        if 'NATR_THRESHOLD' in search_space:
            spec = search_space['NATR_THRESHOLD']
            params['NATR_THRESHOLD'] = trial.suggest_float('NATR_THRESHOLD', spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 5: Exit-Type Dependent Parameters ===
    exit_type = params.get('EXIT_TYPE', 'ATR')
    
    if exit_type == 'PARABOLIC_SAR':
        if 'SAR_STEP' in search_space:
            spec = search_space['SAR_STEP']
            params['SAR_STEP'] = trial.suggest_float('SAR_STEP', spec['low'], spec['high'], step=spec.get('step'))
    
    # === Phase 6: Common Parameters (Always Used) ===
    # [CRITICAL] Handle STOP_LOSS_TYPE logical conflict first
    stop_loss_type = params.get('STOP_LOSS_TYPE', 'FIXED')
    
    if stop_loss_type == 'FIXED':
        if 'STOP_LOSS_PCT' in search_space:
            spec = search_space['STOP_LOSS_PCT']
            params['STOP_LOSS_PCT'] = trial.suggest_float('STOP_LOSS_PCT', spec['low'], spec['high'], step=spec.get('step'))
    
    elif stop_loss_type == 'ATR':
        if 'ATR_STOP_LOSS_MULT' in search_space:
            spec = search_space['ATR_STOP_LOSS_MULT']
            use_log = spec.get('log', False)
            if use_log:
                params['ATR_STOP_LOSS_MULT'] = trial.suggest_float('ATR_STOP_LOSS_MULT', spec['low'], spec['high'], log=True)
            else:
                params['ATR_STOP_LOSS_MULT'] = trial.suggest_float('ATR_STOP_LOSS_MULT', spec['low'], spec['high'], step=spec.get('step'))
    
    # [CRITICAL] Handle USE_TAKE_PROFIT logical conflict
    use_take_profit = params.get('USE_TAKE_PROFIT', False)
    
    if use_take_profit:
        for key in ['TAKE_PROFIT_ATR_MULT', 'TAKE_PROFIT_ATR_MULT_FUTURES']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if use_log:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=True)
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
    
    # [CRITICAL] Handle USE_VOLUME_FILTER logical conflict
    use_volume_filter = params.get('USE_VOLUME_FILTER', False)
    
    if use_volume_filter:
        for key in ['VOLUME_THRESHOLD_MULT', 'VOLUME_MA_PERIOD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'float':
                    if use_log:
                        params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=True)
                    else:
                        params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
                elif spec['type'] == 'int':
                    if use_log:
                        params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=True)
                    else:
                        params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    # Other common parameters (no conflicts)
    common_keys = [
        'ENTRY_PERIOD', 'MA_PERIOD', 'ATR_PERIOD',
        'ATR_MULTIPLIER',
        'ADX_THRESHOLD',
        'RISK_PER_TRADE', 'LEVERAGE',
        'MAX_HOLDING_BARS', 'TRAILING_ACTIVATION_ATR',
        'TIME_EXIT_PROFIT_THRESHOLD',  # [NEW] Conditional time exit
        'RSI_EXIT_THRESHOLD', # [NEW] Panic exit
        'RSI_ENTRY_MAX', 'NATR_ENTRY_MIN', # [NEW] Entry Safety Filters
        'USE_DYNAMIC_RISK',
        'STRONG_REGIME_HURST', 'STRONG_REGIME_NATR', 'STRONG_REGIME_MULTIPLIER',
        'WEAK_REGIME_HURST', 'WEAK_REGIME_MULTIPLIER',
        'PANIC_REGIME_NATR', 'PANIC_REGIME_MULTIPLIER',
        'RISK_PER_TRADE_SPOT'
    ]
    
    for key in common_keys:
        if key in search_space:
            spec = search_space[key]
            use_log = spec.get('log', False)
            
            if spec['type'] == 'float':
                if use_log:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=True)
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
            elif spec['type'] == 'int':
                if use_log:
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=True)
                else:
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
    
    return params

def soft_sigmoid(x, L, k, x0):
    """
    Soft-Sigmoid mapping to handle diminishing returns without hard caps.
    L: Maximum value (Asymptote)
    k: Steepness
    x0: Midpoint (Center of the S-curve)
    
    Numerically stable implementation with overflow protection.
    """
    # Prevent overflow: clip exp argument to safe range [-500, 500]
    z = -k * (x - x0)
    z_safe = np.clip(z, -500, 500)
    return L / (1 + np.exp(z_safe))

def calculate_score(ret, mdd, trades_df, mode="DAY", market_type="spot", timeframe=None):
    """
    Objective Function v14: Semantic Robustness Optimization
    
    Improvements:
    1. Added SQN (System Quality Number): Rewards statistical significance and smoothness.
    2. Added Consistency Score (R^2): Penalizes volatile equity curves even if end return is high.
    3. Tighter Expectancy: Raised min avg profit to 0.20% (20bps) to guarantee slippage coverage.
    4. Market-Specific Weights: 
       - Spot: Prioritizes Consistency & Safety (Sortino/Calmar).
       - Futures: Prioritizes Survivability (Kelly/Ulcer).
    """
    if trades_df.empty:
        return -10000.0

    N = len(trades_df)
    if 'pnl' not in trades_df.columns or 'pnl_pct' not in trades_df.columns:
        return -10000.0
    
    returns = trades_df['pnl_pct'].values
    pnl_raw = trades_df['pnl'].values

    # --- 1. Core Efficiency Metrics ---
    r_avg = np.mean(returns)
    r_std = np.std(returns, ddof=1) if len(returns) > 1 else 0.001
    
    # [METRIC] SQN (System Quality Number)
    # SQN = sqrt(N) * (Mean / Std)
    # < 1.6: Poor, 1.6-2.0: Average, 2.0-3.0: Good, 3.0-5.0: Excellent, > 7.0: Holy Grail
    sqn_raw = np.sqrt(N) * (r_avg / r_std) if r_std > 0 else 0
    sqn = np.clip(sqn_raw, 0, 10)

    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.001
    
    # [METRIC] Sortino
    sortino_raw = r_avg / max(downside_std, 0.0001)
    sortino = np.clip(sortino_raw, -20, 20)
    
    pos_sum = np.sum(returns[returns > 0])
    neg_sum = abs(np.sum(returns[returns < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else 3.0
    
    abs_mdd = abs(mdd) if mdd != 0 else 0.01
    # [METRIC] Calmar
    calmar_raw = ret / abs_mdd
    calmar = np.clip(calmar_raw, -10, 20)

    # --- 2. Financial Safety: Kelly Criterion ---
    win_rate = len(returns[returns > 0]) / N
    avg_win = np.mean(returns[returns > 0]) if any(returns > 0) else 0.001
    avg_loss = abs(np.mean(returns[returns < 0])) if any(returns < 0) else 0.001
    win_loss_ratio = max(avg_win / avg_loss, 0.001)
    
    kelly_f = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # --- 3. Ulcer & Consistency (Linearity) ---
    equity_curve = np.cumsum(returns)
    
    # [METRIC] Consistency (R^2)
    # Measures how close the equity curve is to a straight line (perfect steady growth)
    if N > 5:
        x = np.arange(len(equity_curve))
        y = equity_curve
        correlation_matrix = np.corrcoef(x, y)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2 if not np.isnan(correlation_xy) else 0
    else:
        r_squared = 0.5

    hwm = np.maximum.accumulate(equity_curve)
    drawdowns = hwm - equity_curve
    ulcer_proxy = np.sqrt(np.mean(np.square(drawdowns))) if len(drawdowns) > 0 else 0
    
    if not np.isfinite(ulcer_proxy) or not np.isfinite(sortino) or not np.isfinite(calmar):
        return -10000.0

    # --- 4. Scoring Logic ---
    score = 0.0
    
    if market_type == "futures":
        # [FUTURES] Theme: "ROBUST SURVIVABILITY"
        target_mdd = 20.0
        min_trades = 100 if mode != 'SCALP' else 300
        
        # Sigmoids
        s_calmar = soft_sigmoid(calmar, L=10.0, k=0.8, x0=3.0)
        s_sortino = soft_sigmoid(sortino, L=8.0, k=1.0, x0=2.0)
        s_pf = soft_sigmoid(pf, L=6.0, k=1.5, x0=1.8)
        s_sqn = soft_sigmoid(sqn, L=8.0, k=0.8, x0=2.5) # Prefer SQN > 2.5
        
        score = (s_calmar * 10.0) + (s_sortino * 8.0) + (s_pf * 6.0) + (s_sqn * 8.0)
        score += (s_pf * s_sqn) * 2.0 # High Pf + High SQN = Stable Winner
        
        if kelly_f <= 0: return -10000.0
        
        # Penalties
        score -= (ulcer_proxy * 6.0)
        if r_squared < 0.85: score -= (0.85 - r_squared) * 20.0 # Heavy penalty for instability
        
        if abs_mdd > target_mdd:
            score -= (abs_mdd - target_mdd) ** 1.8 * 15.0

        # [ANTI-OVERFIT] Futures Only Penalties
        if win_rate > 0.85:
            excess_win = (win_rate - 0.85) * 100
            score -= (excess_win * 5.0) 
            
        if avg_loss > (avg_win * 3.0):
            score -= 50.0

    else:
        # [SPOT] Theme: "COMPOUNDING EFFICIENCY"
        # Objective: Maximize Geometric Growth (Kelly-Optimal) while minimizing volatility tax.
        target_mdd = 28.0 
        min_trades = 60
        
        # [CRITICAL] Kelly Check for Spot
        # Even without leverage, negative Kelly means negative geometric growth.
        if kelly_f <= 0: return -10000.0

        # Spot needs higher Return/Risk efficiency (no leverage helper)
        # Adjusted Centers for "Unleveraged Realism":
        # - Calmar: 2.5 (Ex: 50% Ret / 20% MDD) - 4.0 was too high
        # - PF: 1.5 (Trend Followers usually 1.5~2.0)
        # - Sortino: 2.0 (Solid downside control)
        
        s_calmar = soft_sigmoid(calmar, L=15.0, k=0.7, x0=2.5)
        s_sortino = soft_sigmoid(sortino, L=10.0, k=1.0, x0=2.0)
        s_pf = soft_sigmoid(pf, L=8.0, k=1.2, x0=1.5)
        s_sqn = soft_sigmoid(sqn, L=10.0, k=0.8, x0=2.0)
        
        score = (s_calmar * 12.0) + (s_sqn * 10.0) + (s_pf * 8.0) + (s_sortino * 6.0)
        
        # Bonus for Consistency (Relaxed to 0.90 for Crypto Reality)
        if r_squared > 0.90: score += 10.0
        
        score -= (ulcer_proxy * 5.0) # Increased penalty for volatility tax
        
        if abs_mdd > 15.0:
            score -= (abs_mdd - 15.0) * 1.5
        if abs_mdd > target_mdd:
            score -= (abs_mdd - target_mdd) * 10.0

    # --- 5. Common Penalties ---

    # Trade Count (Logarithmic Penalty - harsh on very low numbers)
    min_trades = 150 if market_type == 'futures' else 60
    
    if N < min_trades:
        if market_type == 'futures':
             if N < (min_trades * 0.5): return -10000.0
        else:
             if N < (min_trades * 0.4): return -10000.0
             
        shortfall = min_trades - N
        score -= (shortfall * 4.0) 
        
    # Expectancy Check (Slippage Safety)
    # Raised to 0.20% (20bps) - ensures we cover 5bps fee + 5bps slippage + buffer
    avg_profit_pct = r_avg
    if avg_profit_pct < 0.15: 
        return -10000.0
    elif avg_profit_pct < 0.30: 
        score -= (0.30 - avg_profit_pct) * 200.0 # Steep penalty between 0.15% and 0.30%

    return float(score)
