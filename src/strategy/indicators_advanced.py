import os

from src.common.indicators_advanced import IndicatorEngine


_SPOT_CACHE_MODE = os.getenv("SPOT_INDICATOR_CACHE_MODE", os.getenv("INDICATOR_CACHE_MODE", "disabled"))
_ENGINE = IndicatorEngine(cache_mode=_SPOT_CACHE_MODE)

calculate_sma = _ENGINE.calculate_sma
calculate_ema = _ENGINE.calculate_ema
calculate_wma = _ENGINE.calculate_wma
calculate_hma = _ENGINE.calculate_hma
calculate_dema = _ENGINE.calculate_dema
calculate_tema = _ENGINE.calculate_tema
calculate_atr = _ENGINE.calculate_atr
calculate_bollinger_bands = _ENGINE.calculate_bollinger_bands
calculate_keltner_channels = _ENGINE.calculate_keltner_channels
calculate_keltner_channel = _ENGINE.calculate_keltner_channel
calculate_supertrend = _ENGINE.calculate_supertrend
calculate_vhf = _ENGINE.calculate_vhf
calculate_adx = _ENGINE.calculate_adx
calculate_rsi = _ENGINE.calculate_rsi
calculate_stochastic = _ENGINE.calculate_stochastic
calculate_stoch_rsi = _ENGINE.calculate_stoch_rsi
calculate_macd = _ENGINE.calculate_macd
calculate_ichimoku = _ENGINE.calculate_ichimoku
calculate_cci = _ENGINE.calculate_cci
calculate_mfi = _ENGINE.calculate_mfi
calculate_parabolic_sar = _ENGINE.calculate_parabolic_sar
calculate_vwap = _ENGINE.calculate_vwap
calculate_cmf = _ENGINE.calculate_cmf
calculate_hurst_exponent = _ENGINE.calculate_hurst_exponent
calculate_natr = _ENGINE.calculate_natr
calculate_efficiency_ratio = _ENGINE.calculate_efficiency_ratio
calculate_garman_klass_vol = _ENGINE.calculate_garman_klass_vol
calculate_dmi = _ENGINE.calculate_dmi
calculate_aroon = _ENGINE.calculate_aroon
calculate_force_index = _ENGINE.calculate_force_index
calculate_williams_r = _ENGINE.calculate_williams_r
calculate_obv = _ENGINE.calculate_obv
clear_indicator_cache = _ENGINE.clear_indicator_cache
