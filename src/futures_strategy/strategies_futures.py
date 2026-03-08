from types import SimpleNamespace

from src.strategy.base import (
    MasterStrategyBase,
    StrategyBase,
    UltimateStrategyBase,
)

from .indicators_advanced_futures import (
    calculate_adx,
    calculate_aroon,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_cci,
    calculate_cmf,
    calculate_dema,
    calculate_dmi,
    calculate_efficiency_ratio,
    calculate_ema,
    calculate_force_index,
    calculate_garman_klass_vol,
    calculate_hma,
    calculate_hurst_exponent,
    calculate_ichimoku,
    calculate_keltner_channel,
    calculate_keltner_channels,
    calculate_macd,
    calculate_mfi,
    calculate_natr,
    calculate_obv,
    calculate_parabolic_sar,
    calculate_rsi,
    calculate_sma,
    calculate_stoch_rsi,
    calculate_stochastic,
    calculate_supertrend,
    calculate_tema,
    calculate_vhf,
    calculate_vwap,
    calculate_vwma,
    calculate_roc,
    calculate_williams_r,
)


class Strategy(StrategyBase):
    pass


class MasterStrategy(MasterStrategyBase):
    pass


_FUTURES_INDICATORS = SimpleNamespace(
    calculate_sma=calculate_sma,
    calculate_ema=calculate_ema,
    calculate_hma=calculate_hma,
    calculate_dema=calculate_dema,
    calculate_tema=calculate_tema,
    calculate_supertrend=calculate_supertrend,
    calculate_atr=calculate_atr,
    calculate_bollinger_bands=calculate_bollinger_bands,
    calculate_keltner_channel=calculate_keltner_channel,
    calculate_keltner_channels=calculate_keltner_channels,
    calculate_adx=calculate_adx,
    calculate_vhf=calculate_vhf,
    calculate_parabolic_sar=calculate_parabolic_sar,
    calculate_rsi=calculate_rsi,
    calculate_stochastic=calculate_stochastic,
    calculate_stoch_rsi=calculate_stoch_rsi,
    calculate_macd=calculate_macd,
    calculate_ichimoku=calculate_ichimoku,
    calculate_cci=calculate_cci,
    calculate_mfi=calculate_mfi,
    calculate_vwap=calculate_vwap,
    calculate_cmf=calculate_cmf,
    calculate_hurst_exponent=calculate_hurst_exponent,
    calculate_efficiency_ratio=calculate_efficiency_ratio,
    calculate_natr=calculate_natr,
    calculate_garman_klass_vol=calculate_garman_klass_vol,
    calculate_dmi=calculate_dmi,
    calculate_aroon=calculate_aroon,
    calculate_force_index=calculate_force_index,
    calculate_williams_r=calculate_williams_r,
    calculate_obv=calculate_obv,
    calculate_roc=calculate_roc,
    calculate_vwma=calculate_vwma,
)


class UltimateStrategy(UltimateStrategyBase):
    """
    Pure TSMOM Trend-Following Strategy for Futures.
    Uses ATR Chandelier trailing exit.
    ENTRY_SHIFT=True: signal at index i uses close[i-1] → engine enters at bar i open. No lookahead.
    """

    INDICATORS = _FUTURES_INDICATORS
    ENTRY_SHIFT = False
