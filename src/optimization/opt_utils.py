
import os
import optuna
import numpy as np
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class ObjectiveConfig:
    # Core composition
    base_score_multiplier: float = 160.0
    gate_floor: float = 0.25
    gate_weight_activity: float = 0.35
    gate_weight_consistency: float = 0.25
    gate_weight_kelly: float = 0.20
    gate_weight_side: float = 0.30

    # Baselines
    min_trades_futures: int = 120
    min_trades_spot: int = 60
    target_mdd_futures: float = 38.0
    target_mdd_spot: float = 28.0
    min_side_ratio_futures: float = 0.15
    consistency_center_futures: float = 0.55
    consistency_center_spot: float = 0.65
    kelly_center_futures: float = 0.02
    kelly_center_spot: float = 0.01

    # Gates
    gate_scale_activity_ratio: float = 0.25
    gate_scale_consistency: float = 0.08
    gate_scale_kelly: float = 0.03
    gate_scale_side: float = 0.04

    # Penalties / bonuses
    expectancy_threshold_pct: float = 0.10
    expectancy_penalty_mult: float = 160.0
    negative_expectancy_penalty_mult: float = 120.0
    bonus_ret_weight: float = 18.0
    bonus_pf_weight: float = 8.0
    bonus_ret_scale: float = 40.0
    bonus_pf_scale: float = 0.6
    bonus_pf_center: float = 1.2

    # Composition weights
    w_growth_signal: float = 1.10
    w_quality_signal: float = 0.90
    w_risk_signal: float = 1.25
    w_tail_signal: float = 0.60
    w_quality_sortino: float = 0.35
    w_quality_sqn: float = 0.25
    w_quality_pf: float = 0.25
    w_quality_calmar: float = 0.15
    w_risk_mdd: float = 0.45
    w_risk_cvar: float = 0.35
    w_risk_ulcer: float = 0.20

    # ASINH transform scales (anti-saturation)
    asinh_growth_scale: float = 0.002
    asinh_sortino_scale: float = 2.0
    asinh_sqn_scale: float = 1.8
    asinh_pf_scale: float = 0.8
    asinh_calmar_scale: float = 2.0
    asinh_mdd_scale_ratio: float = 1.0
    asinh_cvar_scale: float = 1.0
    asinh_ulcer_scale: float = 4.0
    asinh_tail_scale_base: float = 6.0
    asinh_clip: float = 4.5


def _load_objective_config():
    return ObjectiveConfig(
        base_score_multiplier=_env_float("OBJ_BASE_SCORE_MULT", 160.0),
        gate_floor=_env_float("OBJ_GATE_FLOOR", 0.25),
        gate_weight_activity=_env_float("OBJ_GATE_W_ACTIVITY", 0.35),
        gate_weight_consistency=_env_float("OBJ_GATE_W_CONSISTENCY", 0.25),
        gate_weight_kelly=_env_float("OBJ_GATE_W_KELLY", 0.20),
        gate_weight_side=_env_float("OBJ_GATE_W_SIDE", 0.30),
        min_trades_futures=_env_int("OBJ_MIN_TRADES_FUTURES", 120),
        min_trades_spot=_env_int("OBJ_MIN_TRADES_SPOT", 60),
        target_mdd_futures=_env_float("OBJ_TARGET_MDD_FUTURES", 38.0),
        target_mdd_spot=_env_float("OBJ_TARGET_MDD_SPOT", 28.0),
        min_side_ratio_futures=_env_float("OBJ_MIN_SIDE_RATIO_FUTURES", 0.15),
        consistency_center_futures=_env_float("OBJ_CONSISTENCY_CENTER_FUTURES", 0.55),
        consistency_center_spot=_env_float("OBJ_CONSISTENCY_CENTER_SPOT", 0.65),
        kelly_center_futures=_env_float("OBJ_KELLY_CENTER_FUTURES", 0.02),
        kelly_center_spot=_env_float("OBJ_KELLY_CENTER_SPOT", 0.01),
        gate_scale_activity_ratio=_env_float("OBJ_GATE_SCALE_ACTIVITY_RATIO", 0.25),
        gate_scale_consistency=_env_float("OBJ_GATE_SCALE_CONSISTENCY", 0.08),
        gate_scale_kelly=_env_float("OBJ_GATE_SCALE_KELLY", 0.03),
        gate_scale_side=_env_float("OBJ_GATE_SCALE_SIDE", 0.04),
        expectancy_threshold_pct=_env_float("OBJ_EXPECTANCY_THRESHOLD_PCT", 0.10),
        expectancy_penalty_mult=_env_float("OBJ_EXPECTANCY_PENALTY_MULT", 160.0),
        negative_expectancy_penalty_mult=_env_float("OBJ_NEG_EXPECTANCY_PENALTY_MULT", 120.0),
        bonus_ret_weight=_env_float("OBJ_BONUS_RET_WEIGHT", 18.0),
        bonus_pf_weight=_env_float("OBJ_BONUS_PF_WEIGHT", 8.0),
        bonus_ret_scale=_env_float("OBJ_BONUS_RET_SCALE", 40.0),
        bonus_pf_scale=_env_float("OBJ_BONUS_PF_SCALE", 0.6),
        bonus_pf_center=_env_float("OBJ_BONUS_PF_CENTER", 1.2),
        w_growth_signal=_env_float("OBJ_W_GROWTH_SIGNAL", 1.10),
        w_quality_signal=_env_float("OBJ_W_QUALITY_SIGNAL", 0.90),
        w_risk_signal=_env_float("OBJ_W_RISK_SIGNAL", 1.25),
        w_tail_signal=_env_float("OBJ_W_TAIL_SIGNAL", 0.60),
        w_quality_sortino=_env_float("OBJ_W_QUALITY_SORTINO", 0.35),
        w_quality_sqn=_env_float("OBJ_W_QUALITY_SQN", 0.25),
        w_quality_pf=_env_float("OBJ_W_QUALITY_PF", 0.25),
        w_quality_calmar=_env_float("OBJ_W_QUALITY_CALMAR", 0.15),
        w_risk_mdd=_env_float("OBJ_W_RISK_MDD", 0.45),
        w_risk_cvar=_env_float("OBJ_W_RISK_CVAR", 0.35),
        w_risk_ulcer=_env_float("OBJ_W_RISK_ULCER", 0.20),
        asinh_growth_scale=_env_float("OBJ_ASINH_GROWTH_SCALE", 0.002),
        asinh_sortino_scale=_env_float("OBJ_ASINH_SORTINO_SCALE", 2.0),
        asinh_sqn_scale=_env_float("OBJ_ASINH_SQN_SCALE", 1.8),
        asinh_pf_scale=_env_float("OBJ_ASINH_PF_SCALE", 0.8),
        asinh_calmar_scale=_env_float("OBJ_ASINH_CALMAR_SCALE", 2.0),
        asinh_mdd_scale_ratio=_env_float("OBJ_ASINH_MDD_SCALE_RATIO", 1.0),
        asinh_cvar_scale=_env_float("OBJ_ASINH_CVAR_SCALE", 1.0),
        asinh_ulcer_scale=_env_float("OBJ_ASINH_ULCER_SCALE", 4.0),
        asinh_tail_scale_base=_env_float("OBJ_ASINH_TAIL_SCALE_BASE", 6.0),
        asinh_clip=_env_float("OBJ_ASINH_CLIP", 4.5),
    )


OBJECTIVE_CFG = _load_objective_config()

def suggest_params(trial, search_space):
    """
    Generate trial parameters from search space with conditional dependency pruning.
    Only suggests parameters that are actually used by the selected strategy configuration.
    
    Efficiency Gain: 60~70% reduction in search space by skipping irrelevant parameters.
    """
    params = {}

    def _sanitize_float_step_bounds(low, high, step):
        """Sanitize float-step bounds to avoid Optuna range/step divisibility warnings."""
        d_step = Decimal(str(step))
        if d_step <= 0:
            return float(low), float(high)
        places = max(0, -d_step.as_tuple().exponent)
        quant = Decimal(1).scaleb(-places)
        d_low = Decimal(str(low)).quantize(quant)
        d_high = Decimal(str(high)).quantize(quant)
        if d_high < d_low:
            d_high = d_low
        steps = ((d_high - d_low) / d_step).to_integral_value(rounding=ROUND_FLOOR)
        safe_high = (d_low + (steps * d_step)).quantize(quant)
        if safe_high < d_low:
            safe_high = d_low
        return float(d_low), float(safe_high)

    def _suggest_value(key, spec):
        typ = spec.get("type")
        use_log = bool(spec.get("log", False))

        if typ == "categorical":
            return trial.suggest_categorical(key, spec["choices"])
        if typ == "int":
            if use_log:
                return trial.suggest_int(key, spec["low"], spec["high"], log=True)
            return trial.suggest_int(key, spec["low"], spec["high"], step=spec.get("step"))
        if typ == "float":
            if use_log:
                return trial.suggest_float(key, spec["low"], spec["high"], log=True)
            step = spec.get("step")
            if step is None:
                return trial.suggest_float(key, spec["low"], spec["high"])
            low, high = _sanitize_float_step_bounds(spec["low"], spec["high"], step)
            if high <= low:
                return float(low)
            return trial.suggest_float(key, low, high, step=step)
        raise ValueError(f"Unsupported spec type: {typ} for {key}")

    def _suggest_timeframe_bounded_holding(spec):
        """
        Conditionally bound MAX_HOLDING_BARS by timeframe.
        This prevents structurally-low-frequency combinations from dominating spot search.
        """
        # Suggest from a single static distribution first (warning-safe),
        # then clamp by timeframe. Avoids per-trial distribution mismatch warnings.
        sampled = _suggest_value("MAX_HOLDING_BARS", spec)
        selected_tf = str(params.get("TIMEFRAME", "")).strip().lower()
        tf_bounds = {
            "15m": (36, 320),
            "30m": (30, 260),
            "1h": (24, 180),
            "2h": (18, 140),
            "4h": (12, 96),
            "1d": (8, 60),
            "1w": (4, 24),
        }
        if selected_tf not in tf_bounds:
            return int(sampled)

        tf_low, tf_high = tf_bounds[selected_tf]
        low = int(max(int(spec.get("low", tf_low)), int(tf_low)))
        high = int(min(int(spec.get("high", tf_high)), int(tf_high)))
        if high <= low:
            return int(low)
        return int(max(low, min(high, int(sampled))))

    def _set_inactive_default(key):
        if key not in search_space:
            return
        spec = search_space[key]
        typ = spec.get("type")
        if typ == "categorical":
            choices = list(spec.get("choices", []))
            if choices:
                params[key] = choices[0]
        elif typ == "int":
            params[key] = int(spec.get("low", 0))
        elif typ == "float":
            params[key] = float(spec.get("low", 0.0))
    
    # === Phase 1: Core Strategy Selection ===
    for key in ['ENTRY_TYPE', 'TREND_FILTER_TYPE', 'STRENGTH_FILTER_TYPE', 'EXIT_TYPE', 
                'STOP_LOSS_TYPE', 'USE_TAKE_PROFIT', 'USE_VOLUME_FILTER', 'TIMEFRAME',
                'REGIME_FILTER', 'USE_ADX']: # Added missing keys from strategies
        if key in search_space:
            spec = search_space[key]
            if spec['type'] == 'categorical':
                params[key] = _suggest_value(key, spec)
    
    # === Phase 2: Entry-Type Dependent Parameters ===
    entry_type = params.get('ENTRY_TYPE', 'DONCHIAN')
    
    if entry_type == 'BOLLINGER':
        if 'BB_STD' in search_space:
            params['BB_STD'] = _suggest_value('BB_STD', search_space['BB_STD'])
            
    elif entry_type == 'KELTNER':
        if 'KELTNER_ATR_MULT' in search_space:
            params['KELTNER_ATR_MULT'] = _suggest_value('KELTNER_ATR_MULT', search_space['KELTNER_ATR_MULT'])
            
    elif entry_type == 'CCI':
        if 'CCI_THRESHOLD' in search_space:
            params['CCI_THRESHOLD'] = _suggest_value('CCI_THRESHOLD', search_space['CCI_THRESHOLD'])
    
    # === Phase 3: Trend-Filter Dependent Parameters ===
    trend_filter = params.get('TREND_FILTER_TYPE', 'EMA')
    
    if trend_filter == 'SUPERTREND':
        for key in ['SUPERTREND_MULT', 'SUPERTREND_PERIOD']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    elif trend_filter == 'MACD':
        for key in ['MACD_FAST', 'MACD_SLOW', 'MACD_SIGNAL']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    elif trend_filter == 'ICHIMOKU':
        for key in ['ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_B']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    elif trend_filter == 'VWAP':
        if 'VWAP_STD_MULT' in search_space:
            params['VWAP_STD_MULT'] = _suggest_value('VWAP_STD_MULT', search_space['VWAP_STD_MULT'])
    
    # === Phase 4: Strength-Filter Dependent Parameters ===
    strength_filter = params.get('STRENGTH_FILTER_TYPE', 'NONE')
    
    if strength_filter in ['ADX', 'VHF', 'MFI', 'RSI', 'STOCHASTIC', 'STOCH_RSI']:
        if 'STRENGTH_FILTER_PERIOD' in search_space:
            params['STRENGTH_FILTER_PERIOD'] = _suggest_value('STRENGTH_FILTER_PERIOD', search_space['STRENGTH_FILTER_PERIOD'])
    
    if strength_filter == 'VHF':
        if 'VHF_THRESHOLD' in search_space:
            params['VHF_THRESHOLD'] = _suggest_value('VHF_THRESHOLD', search_space['VHF_THRESHOLD'])
    
    elif strength_filter == 'MFI':
        if 'MFI_THRESHOLD' in search_space:
            params['MFI_THRESHOLD'] = _suggest_value('MFI_THRESHOLD', search_space['MFI_THRESHOLD'])
    
    elif strength_filter == 'RSI':
        for key in ['RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'RSI_OVERBOUGHT_FUTURES']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    elif strength_filter == 'STOCHASTIC':
        for key in ['STOCH_OVERBOUGHT', 'STOCH_OVERSOLD']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    elif strength_filter == 'STOCH_RSI':
        for key in ['STOCH_RSI_OVERBOUGHT', 'STOCH_RSI_OVERSOLD']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    elif strength_filter == 'CMF':
        for key in ['CMF_PERIOD', 'CMF_THRESHOLD']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    elif strength_filter == 'HURST':
        # 1. Period (Independent)
        if 'HURST_PERIOD' in search_space:
            params['HURST_PERIOD'] = _suggest_value('HURST_PERIOD', search_space['HURST_PERIOD'])

        # 2. Random Threshold (Independent Base)
        random_thresh = 0.5 # Default fallback
        if 'HURST_RANDOM_THRESHOLD' in search_space:
            params['HURST_RANDOM_THRESHOLD'] = _suggest_value('HURST_RANDOM_THRESHOLD', search_space['HURST_RANDOM_THRESHOLD'])
            random_thresh = params['HURST_RANDOM_THRESHOLD']
            
        # 3. Trend Threshold (Dependent: Must be > Random)
        if 'HURST_TREND_THRESHOLD' in search_space:
            spec = search_space['HURST_TREND_THRESHOLD']
            # Enforce Logical Safety: Trend > Random + Buffer (0.01)
            # This prevents logical contradictions even if search ranges overlap
            safe_low = max(spec['low'], random_thresh + 0.01)
            raw_trend = _suggest_value('HURST_TREND_THRESHOLD', spec)
            step = spec.get("step")
            if safe_low < spec['high']:
                if step is None:
                    params['HURST_TREND_THRESHOLD'] = float(max(safe_low, min(spec['high'], raw_trend)))
                else:
                    low, high = _sanitize_float_step_bounds(safe_low, spec['high'], step)
                    if high <= low:
                        params['HURST_TREND_THRESHOLD'] = float(low)
                    else:
                        snapped = low + round((float(raw_trend) - low) / float(step)) * float(step)
                        params['HURST_TREND_THRESHOLD'] = float(max(low, min(high, snapped)))
            else:
                params['HURST_TREND_THRESHOLD'] = float(safe_low)

    elif strength_filter == 'ER':
        if 'ER_THRESHOLD' in search_space:
            params['ER_THRESHOLD'] = _suggest_value('ER_THRESHOLD', search_space['ER_THRESHOLD'])
            
    elif strength_filter == 'NATR':
        if 'NATR_THRESHOLD' in search_space:
            params['NATR_THRESHOLD'] = _suggest_value('NATR_THRESHOLD', search_space['NATR_THRESHOLD'])
    
    # === Phase 5: Exit-Type Dependent Parameters ===
    exit_type = params.get('EXIT_TYPE', 'ATR')
    
    if exit_type == 'PARABOLIC_SAR':
        if 'SAR_STEP' in search_space:
            params['SAR_STEP'] = _suggest_value('SAR_STEP', search_space['SAR_STEP'])
    
    # === Phase 6: Common Parameters (Always Used) ===
    # [CRITICAL] Handle STOP_LOSS_TYPE logical conflict first
    stop_loss_type = params.get('STOP_LOSS_TYPE', 'FIXED')
    
    if stop_loss_type == 'FIXED':
        if 'STOP_LOSS_PCT' in search_space:
            params['STOP_LOSS_PCT'] = _suggest_value('STOP_LOSS_PCT', search_space['STOP_LOSS_PCT'])
    
    elif stop_loss_type == 'ATR':
        if 'ATR_STOP_LOSS_MULT' in search_space:
            params['ATR_STOP_LOSS_MULT'] = _suggest_value('ATR_STOP_LOSS_MULT', search_space['ATR_STOP_LOSS_MULT'])
    
    # [CRITICAL] Handle USE_TAKE_PROFIT logical conflict
    use_take_profit = params.get('USE_TAKE_PROFIT', False)
    
    if use_take_profit:
        for key in ['TAKE_PROFIT_ATR_MULT', 'TAKE_PROFIT_ATR_MULT_FUTURES']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
    # [CRITICAL] Handle USE_VOLUME_FILTER logical conflict
    use_volume_filter = params.get('USE_VOLUME_FILTER', False)
    
    if use_volume_filter:
        for key in ['VOLUME_THRESHOLD_MULT', 'VOLUME_MA_PERIOD']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    
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
        'ENABLE_SCALE_OUT',
        'ENABLE_BREAKEVEN',
        'ENABLE_PYRAMIDING',
        'USE_DYNAMIC_RISK',
        'RISK_PER_TRADE_SPOT'
    ]
    
    for key in common_keys:
        if key in search_space:
            if key == 'MAX_HOLDING_BARS':
                params[key] = _suggest_timeframe_bounded_holding(search_space[key])
            else:
                params[key] = _suggest_value(key, search_space[key])

    # [CRITICAL] Active position management dependencies
    if params.get('ENABLE_SCALE_OUT', False):
        for key in ['SCALE_OUT_TRIGGER_ATR', 'SCALE_OUT_RATIO']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    else:
        _set_inactive_default('SCALE_OUT_TRIGGER_ATR')
        _set_inactive_default('SCALE_OUT_RATIO')

    if params.get('ENABLE_BREAKEVEN', False):
        if 'BREAKEVEN_BUFFER_PCT' in search_space:
            params['BREAKEVEN_BUFFER_PCT'] = _suggest_value('BREAKEVEN_BUFFER_PCT', search_space['BREAKEVEN_BUFFER_PCT'])
    else:
        _set_inactive_default('BREAKEVEN_BUFFER_PCT')

    if params.get('ENABLE_PYRAMIDING', False):
        for key in ['PYRAMID_TRIGGER_ATR', 'PYRAMID_STEP_ATR', 'PYRAMID_RISK_RATIO', 'PYRAMID_MAX_ADDS']:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    else:
        _set_inactive_default('PYRAMID_TRIGGER_ATR')
        _set_inactive_default('PYRAMID_STEP_ATR')
        _set_inactive_default('PYRAMID_RISK_RATIO')
        _set_inactive_default('PYRAMID_MAX_ADDS')

    # [CRITICAL] Dynamic-risk dependencies
    if params.get('USE_DYNAMIC_RISK', False):
        for key in [
            'STRONG_REGIME_HURST', 'STRONG_REGIME_NATR', 'STRONG_REGIME_MULTIPLIER',
            'WEAK_REGIME_HURST', 'WEAK_REGIME_MULTIPLIER',
            'PANIC_REGIME_NATR', 'PANIC_REGIME_MULTIPLIER',
        ]:
            if key in search_space:
                params[key] = _suggest_value(key, search_space[key])
    else:
        _set_inactive_default('STRONG_REGIME_HURST')
        _set_inactive_default('STRONG_REGIME_NATR')
        _set_inactive_default('STRONG_REGIME_MULTIPLIER')
        _set_inactive_default('WEAK_REGIME_HURST')
        _set_inactive_default('WEAK_REGIME_MULTIPLIER')
        _set_inactive_default('PANIC_REGIME_NATR')
        _set_inactive_default('PANIC_REGIME_MULTIPLIER')
    
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


def _smooth_gate(value, center, scale):
    """
    Smooth gate in [0, 1] to avoid hard discontinuities in optimization landscape.
    """
    safe_scale = max(float(scale), 1e-9)
    z = (float(value) - float(center)) / safe_scale
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _asinh_score(value, scale, clip_abs):
    """
    Anti-saturation transform: slower saturation than tanh, better rank resolution in tails.
    """
    safe_scale = max(float(scale), 1e-9)
    transformed = np.arcsinh(float(value) / safe_scale)
    return float(np.clip(transformed, -abs(float(clip_abs)), abs(float(clip_abs))))


def _blend_gates_with_floor(gates, weights, gate_floor):
    """
    Conservative-bias mitigation:
    Instead of multiplicative collapse, use floor + weighted average.
    """
    g = np.array(gates, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        weighted = float(np.mean(g)) if len(g) > 0 else 0.0
    else:
        weighted = float(np.sum(g * w) / w_sum)
    floor = float(np.clip(gate_floor, 0.0, 0.95))
    return floor + (1.0 - floor) * weighted


def calculate_score(ret, mdd, trades_df, mode="UNIFIED", market_type="spot", timeframe=None, min_trades_override=None):
    """
    Overfitting-resistant objective (continuous-form):
    score = C(activity, consistency, Kelly, side-coverage) * (Growth + Quality - Risk)

    Design goals:
    1) Avoid hard cliff (-10000) except for truly invalid inputs.
    2) Reward geometric growth and statistical quality.
    3) Penalize tail risk (MDD/CVaR/Ulcer) and one-sided futures behavior.
    4) Keep bonus sweep coefficients effective via growth/risk/tail multipliers.
    """
    if trades_df.empty:
        return -10000.0

    N = len(trades_df)
    if 'pnl' not in trades_df.columns or 'pnl_pct' not in trades_df.columns:
        return -10000.0

    returns = trades_df['pnl_pct'].values.astype(np.float64)

    # --- 1) Base statistics ---
    r_avg = float(np.mean(returns))
    r_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.001
    r_std = max(r_std, 1e-6)

    downside_returns = returns[returns < 0]
    downside_std = float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else 0.001
    downside_std = max(downside_std, 1e-6)

    sortino = np.clip(r_avg / downside_std, -20.0, 20.0)
    sqn_raw = np.sqrt(N) * (r_avg / r_std)
    sqn = np.clip(sqn_raw, -10.0, 10.0)

    pos_sum = float(np.sum(returns[returns > 0]))
    neg_sum = abs(float(np.sum(returns[returns < 0])))
    pf = pos_sum / neg_sum if neg_sum > 0 else (4.0 if pos_sum > 0 else 0.0)

    abs_mdd = max(abs(float(mdd)), 1e-6)
    calmar = np.clip(float(ret) / abs_mdd, -15.0, 25.0)

    win_rate = float(np.mean(returns > 0))
    avg_win = float(np.mean(returns[returns > 0])) if np.any(returns > 0) else 0.001
    avg_loss = abs(float(np.mean(returns[returns < 0]))) if np.any(returns < 0) else 0.001
    win_loss_ratio = max(avg_win / max(avg_loss, 1e-9), 1e-6)
    kelly_f = win_rate - ((1.0 - win_rate) / win_loss_ratio)

    # Equity linearity / ulcer
    equity_curve = np.cumsum(returns)
    if N > 5:
        x = np.arange(N)
        corr = np.corrcoef(x, equity_curve)[0, 1]
        r_squared = float(corr**2) if np.isfinite(corr) else 0.0
    else:
        r_squared = 0.4

    hwm = np.maximum.accumulate(equity_curve)
    drawdowns = hwm - equity_curve
    ulcer_proxy = float(np.sqrt(np.mean(np.square(drawdowns)))) if len(drawdowns) > 0 else 0.0

    # CVaR on tail losses (in pnl_pct units)
    loss_returns = returns[returns < 0]
    if len(loss_returns) > 0:
        alpha = 0.10 if N >= 30 else 0.20
        var_alpha = float(np.quantile(loss_returns, alpha))
        tail = loss_returns[loss_returns <= var_alpha]
        cvar_abs = abs(float(np.mean(tail))) if len(tail) > 0 else abs(var_alpha)
    else:
        cvar_abs = 0.0

    if not np.isfinite(ulcer_proxy) or not np.isfinite(sortino) or not np.isfinite(calmar):
        return -10000.0

    # Bonus coefficients (A/B/C sweep compatibility)
    fut_growth_coef = _env_float("FUT_GROWTH_BONUS_COEF", 30.0)
    fut_risk_drag_coef = _env_float("FUT_RISK_DRAG_COEF", 8.0)
    fut_tail_drag_coef = _env_float("FUT_TAIL_DRAG_COEF", 12.0)
    spot_growth_coef = _env_float("SPOT_GROWTH_BONUS_COEF", 18.0)
    spot_risk_drag_coef = _env_float("SPOT_RISK_DRAG_COEF", 10.0)
    spot_tail_drag_coef = _env_float("SPOT_TAIL_DRAG_COEF", 10.0)

    cfg = OBJECTIVE_CFG

    # --- 2) Market baselines ---
    if market_type == "futures":
        min_trades = cfg.min_trades_futures
        target_mdd = cfg.target_mdd_futures
        min_side_ratio = cfg.min_side_ratio_futures
        growth_scale = fut_growth_coef / 30.0
        risk_scale = fut_risk_drag_coef / 8.0
        tail_scale = fut_tail_drag_coef / 12.0
        consistency_center = cfg.consistency_center_futures
        kelly_center = cfg.kelly_center_futures
    else:
        min_trades = cfg.min_trades_spot
        target_mdd = cfg.target_mdd_spot
        min_side_ratio = 0.00
        growth_scale = spot_growth_coef / 18.0
        risk_scale = spot_risk_drag_coef / 10.0
        tail_scale = spot_tail_drag_coef / 10.0
        consistency_center = cfg.consistency_center_spot
        kelly_center = cfg.kelly_center_spot

    if min_trades_override is not None:
        min_trades = int(min_trades_override)
    else:
        min_trades = int(min_trades)

    # --- 3) Growth/Quality/Risk components ---
    mu = r_avg / 100.0
    sigma = r_std / 100.0
    geom_growth = mu - 0.5 * (sigma ** 2)

    growth_signal = _asinh_score(geom_growth, cfg.asinh_growth_scale, cfg.asinh_clip)

    q_sortino = _asinh_score(sortino, cfg.asinh_sortino_scale, cfg.asinh_clip)
    q_sqn = _asinh_score(sqn, cfg.asinh_sqn_scale, cfg.asinh_clip)
    q_pf = _asinh_score((pf - 1.0), cfg.asinh_pf_scale, cfg.asinh_clip)
    q_calmar = _asinh_score(calmar, cfg.asinh_calmar_scale, cfg.asinh_clip)
    quality_signal = (
        (cfg.w_quality_sortino * q_sortino)
        + (cfg.w_quality_sqn * q_sqn)
        + (cfg.w_quality_pf * q_pf)
        + (cfg.w_quality_calmar * q_calmar)
    )

    d_mdd = _asinh_score(abs_mdd, max(target_mdd * cfg.asinh_mdd_scale_ratio, 1e-6), cfg.asinh_clip)
    d_cvar = _asinh_score(cvar_abs, cfg.asinh_cvar_scale, cfg.asinh_clip)
    d_ulcer = _asinh_score(ulcer_proxy, cfg.asinh_ulcer_scale, cfg.asinh_clip)
    risk_signal = (
        (cfg.w_risk_mdd * d_mdd)
        + (cfg.w_risk_cvar * d_cvar)
        + (cfg.w_risk_ulcer * d_ulcer)
    )

    tail_excess = max(abs_mdd - target_mdd, 0.0)
    tail_signal = _asinh_score(
        tail_excess,
        max(cfg.asinh_tail_scale_base, target_mdd * 0.25),
        cfg.asinh_clip,
    )

    base_signal = (
        (cfg.w_growth_signal * growth_scale * growth_signal)
        + (cfg.w_quality_signal * quality_signal)
        - (cfg.w_risk_signal * risk_scale * risk_signal)
        - (cfg.w_tail_signal * tail_scale * tail_signal)
    )

    # --- 4) Continuous gates (activity / consistency / Kelly / side coverage) ---
    activity_gate = _smooth_gate(
        N,
        center=min_trades,
        scale=max(4.0, min_trades * cfg.gate_scale_activity_ratio),
    )
    consistency_gate = _smooth_gate(
        r_squared,
        center=consistency_center,
        scale=cfg.gate_scale_consistency,
    )
    kelly_gate = _smooth_gate(
        kelly_f,
        center=kelly_center,
        scale=cfg.gate_scale_kelly,
    )

    side_gate = 1.0
    short_ratio = 0.0
    if market_type == "futures":
        if "side" in trades_df.columns:
            side_vals = trades_df["side"].astype(str).str.upper().values
            long_count = int(np.sum(side_vals == "LONG"))
            short_count = int(np.sum(side_vals == "SHORT"))
            long_ratio = long_count / max(N, 1)
            short_ratio = short_count / max(N, 1)
            long_gate = _smooth_gate(long_ratio, center=min_side_ratio, scale=cfg.gate_scale_side)
            short_gate = _smooth_gate(short_ratio, center=min_side_ratio, scale=cfg.gate_scale_side)
            side_gate = 0.5 * (long_gate + short_gate)
        else:
            # Missing side column means direction-diversity evidence is unavailable.
            side_gate = cfg.gate_floor

    confidence = min(1.0, np.sqrt(N / 220.0))
    if market_type == "futures":
        gates = [activity_gate, consistency_gate, kelly_gate, side_gate]
        weights = [
            cfg.gate_weight_activity,
            cfg.gate_weight_consistency,
            cfg.gate_weight_kelly,
            cfg.gate_weight_side,
        ]
    else:
        gates = [activity_gate, consistency_gate, kelly_gate]
        weights = [
            cfg.gate_weight_activity,
            cfg.gate_weight_consistency,
            cfg.gate_weight_kelly,
        ]
    combined_gate = _blend_gates_with_floor(gates, weights, cfg.gate_floor)

    score = cfg.base_score_multiplier * confidence * combined_gate * base_signal

    # --- 5) Smooth penalties/bonuses ---
    expectancy_gap = max(cfg.expectancy_threshold_pct - r_avg, 0.0)
    score -= expectancy_gap * cfg.expectancy_penalty_mult
    score -= max(-r_avg, 0.0) * cfg.negative_expectancy_penalty_mult

    score += confidence * (
        cfg.bonus_ret_weight * _asinh_score(float(ret), cfg.bonus_ret_scale, cfg.asinh_clip)
        + cfg.bonus_pf_weight * _asinh_score((pf - cfg.bonus_pf_center), cfg.bonus_pf_scale, cfg.asinh_clip)
    )

    if not np.isfinite(score):
        return -10000.0
    return float(score)
