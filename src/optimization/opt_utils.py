
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
        for key in ['HURST_PERIOD', 'HURST_TREND_THRESHOLD', 'HURST_RANDOM_THRESHOLD']:
            if key in search_space:
                spec = search_space[key]
                use_log = spec.get('log', False)
                if spec['type'] == 'int':
                    params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=use_log) if use_log else trial.suggest_int(key, spec['low'], spec['high'], step=spec.get('step'))
                else:
                    params[key] = trial.suggest_float(key, spec['low'], spec['high'], step=spec.get('step'))
    
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

def calculate_score(ret, mdd, trades_df, mode="DAY"):
    """
    SQN Hybrid Objective v4 (Final) - Unified for Spot & Futures
    """
    if trades_df.empty:
        return -10000

    N = len(trades_df)
    
    # 1. Individual Trade Returns (%)
    if 'pnl_pct' not in trades_df.columns:
        # Try to calculate if pnl and initial_balance present? No simple way here.
        # Assume caller ensures pnl_pct
        if 'pnl' in trades_df.columns: 
             # Fallback if pnl is present but not pct (approximate? no, unsafe)
             return -10000
        raise ValueError("trades_df must contain 'pnl_pct' column for SQN calculation.")
    
    returns = trades_df['pnl_pct'].values

    r_avg = np.mean(returns)
    r_std = np.std(returns) if len(returns) > 1 else 100.0
    if r_std == 0: r_std = 0.001
    
    # --- Helper: Soft Sigmoid Normalization ---
    def soft_sigmoid(x, center, steepness):
        z = -steepness * (x - center)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(z))

    # --- Component 1: SQN ---
    sqn = (np.sqrt(N) * r_avg) / r_std
    sqn_score = soft_sigmoid(sqn, center=2.0, steepness=0.5)

    # --- Component 2: Calmar Ratio ---
    abs_mdd = abs(mdd) if mdd != 0 else 0.01
    calmar = ret / abs_mdd
    # Slightly different settings for Spot vs Futures?
    # Use Futures settings (more aggressive) as baseline, or Spot?
    # Spot used 3.5, Futures used 4.0. Let's use 3.5 for broad compatibility.
    calmar_score = soft_sigmoid(calmar, center=3.5, steepness=0.4)

    # --- Component 3: Profit Factor ---
    pos_sum = np.sum(returns[returns > 0])
    neg_sum = abs(np.sum(returns[returns < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else 3.0
    pf_score = soft_sigmoid(pf, center=1.5, steepness=1.0) # Using 1.5 (Spot) vs 1.8 (Futures). Low 1.5 is safer.

    # --- Component 4: Smooth MDD Penalty ---
    # Spot: -30, Futures: -35. Use -33? or depend on mode?
    mdd_center = -35.0 if mode in ['SCALP', 'DAY'] else -30.0
    mdd_penalty = soft_sigmoid(-abs_mdd, center=mdd_center, steepness=0.2)

    # --- Component 5: Soft Trade Count Penalty ---
    MIN_TRADES_MAP = {'SCALP': 500, 'DAY': 100, 'SWING': 30, 'ALL': 100}
    min_trades = MIN_TRADES_MAP.get(mode.upper(), 100)
    trade_penalty = soft_sigmoid(N, center=min_trades, steepness=0.1)
    
    # Hard floor
    if N < 10:
        return -10000

    # --- Final Score: Multiplicative ---
    final_score = sqn_score * calmar_score * pf_score * mdd_penalty * trade_penalty * 1000
    
    return final_score
