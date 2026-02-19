"""
Optimization configuration for UNIFIED/ALL modes.
"""

from copy import deepcopy
from typing import Any, Dict

# =========================================================
# Spot Trade Gate Policy (single source of truth)
# =========================================================
SPOT_TRADE_GATE_POLICY: Dict[str, Any] = {
    "statistical": {
        "confidence": 0.80,
        "margin_error": 0.25,
        "reference_days": 120,
        "day_scale_exp": 0.5,
        "min_day_scale": 0.6,
    },
    "selection": {
        "use_stat_min_trades": True,
        "stat_min_trades_cap": 60,
        "total_trades_multiplier": 1.0,
        "pf_shrink_ref_trades": 30.0,
        "tf_min_gates": {
            "30m": {"core_min_trades": 12, "core_min_total_trades": 24},
            "1h": {"core_min_trades": 10, "core_min_total_trades": 20},
            "4h": {"core_min_trades": 8, "core_min_total_trades": 16},
        },
    },
    "holdout": {
        "use_stat_min_trades": True,
        "sanity_gates": {
            "core_min_return": 0.0,
            "core_min_excess_ret": 3.0,
            "core_dual_return_mode": True,
            "core_min_pf": 1.05,
            "core_min_reliability": 0.70,
            "core_min_trades": 8,
            "core_min_symbol_trades": 6,
            "core_min_total_trades": 14,
            "core_total_trades_gate_mult": 1.5,
            "low_activity_neutralize_ratio": 0.75,
            "low_activity_severe_min_return": -1.0,
            "low_activity_severe_min_pf": 0.90,
            "bear_only_min_abs_return": -2.0,
            "bear_only_min_pf": 0.95,
            "bear_only_excess_ret_exempt": 20.0,
            "core_max_avg_mdd_abs": 2.5,
            "min_symbol_coverage": 0.67,
        },
        "trades_per_30d": {
            "30m": 3.5,
            "1h": 2.5,
            "4h": 1.2,
            "default": 2.5,
        },
        "min_trades_floor": {
            "30m": 10,
            "1h": 8,
            "4h": 5,
            "default": 8,
        },
        "min_trades_cap": {
            "30m": 32,
            "1h": 20,
            "4h": 12,
            "default": 20,
        },
    },
}


def GET_SPOT_TRADE_GATE_POLICY() -> Dict[str, Any]:
    return deepcopy(SPOT_TRADE_GATE_POLICY)


# =========================================================
# 1. PERIOD & THRESHOLD CONSTANTS (User Configurable)
# =========================================================

# UNIFIED: All-in-one strategy (모든 범위 병합)
# 타임프레임: 1h, 4h, 1d 고정 (노이즈 제거)
# 파라미터: 각 지표의 최솟값 ~ 최댓값 사용
UNIFIED_CONFIG = {
    "ENTRY_PERIOD": {"low": 10, "high": 200, "log": True},  # 통합 탐색 범위
    "MA_PERIOD": {"low": 5, "high": 200, "log": True},  # 통합 탐색 범위
    "ATR_PERIOD": {
        "low": 5,
        "high": 60,
        "log": True,
    },  # [Optimized] 10~30 -> 5~60 (민감~둔감 다양한 변동성 대응)
    "SL_PCT": {
        "low": 0.005,
        "high": 0.05,
        "step": 0.005,
    },  # [Optimized] Max 20% -> 5% (고배율 선물에서 5% 이상 손절은 의미 없음)
    "TP_ATR_MULT": {
        "low": 1.5,
        "high": 15.0,
        "log": True,
    },  # [Optimized] Max 30->15 (Realistic Big Win)
    "ADX_THRESH": {"low": 15, "high": 45, "step": 1},  # 통합 탐색 범위
    "VOL_THRESHOLD": {
        "low": 1.1,
        "high": 3.0,
        "log": True,
    },  # [Optimized] Max 5.0 -> 3.0 (현실적인 거래량 돌파 기준)
    "MAX_HOLDING_BARS": {
        "low": 5,
        "high": 500,
        "log": True,
    },  # [Max Profit] 200->500 (Catch Monster Trend)
    "TRAILING_ACTIVATION_ATR": {
        "low": 0.5,
        "high": 8.0,
        "log": False,
        "step": 0.5,
    },  # [Optimized] Min 0.0->0.5 (Avoid immediate whipsaw)
}

# =========================================================
# 2. BASE SEARCH SPACE
# =========================================================
BASE_SEARCH_SPACE = {
    "ENTRY_TYPE": {
        "type": "categorical",
        "choices": ["DONCHIAN", "BOLLINGER", "KELTNER", "CCI"],
    },
    "TREND_FILTER_TYPE": {
        "type": "categorical",
        "choices": [
            "SMA",
            "EMA",
            "HMA",
            "DEMA",
            "TEMA",
            "SUPERTREND",
            "MACD",
            "ICHIMOKU",
            "VWAP",
            "DMI",
            "AROON",
        ],
    },
    "STRENGTH_FILTER_TYPE": {
        "type": "categorical",
        "choices": [
            "NONE",
            "ADX",
            "VHF",
            "MFI",
            "RSI",
            "STOCHASTIC",
            "STOCH_RSI",
            "CMF",
            "HURST",
            "ER",
            "NATR",
            "GARMAN_KLASS",
            "FORCE_INDEX",
            "WILLIAMS_R",
            "OBV",
        ],
    },
    "EXIT_TYPE": {"type": "categorical", "choices": ["ATR", "PARABOLIC_SAR"]},
    "USE_TAKE_PROFIT": {"type": "categorical", "choices": [True, False]},
    "STOP_LOSS_TYPE": {"type": "categorical", "choices": ["FIXED", "ATR"]},
    "USE_VOLUME_FILTER": {"type": "categorical", "choices": [True, False]},
    # Common Indicator Parameters
    "BB_STD": {
        "type": "float",
        "low": 1.5,
        "high": 3.0,
        "step": 0.1,
    },  # Linear (표준편차)
    "KELTNER_ATR_MULT": {
        "type": "float",
        "low": 1.0,
        "high": 2.5,
        "step": 0.1,
    },  # [NEW] Keltner 너비
    "CCI_THRESHOLD": {
        "type": "int",
        "low": 50,
        "high": 150,
        "step": 10,
    },  # [NEW] CCI 돌파 기준
    "SUPERTREND_MULT": {
        "type": "float",
        "low": 1.0,
        "high": 5.0,
        "log": True,
    },  # Log (배수)
    "SUPERTREND_PERIOD": {
        "type": "int",
        "low": 5,
        "high": 50,
        "log": True,
    },  # Log (기간)
    "MACD_FAST": {"type": "int", "low": 5, "high": 30, "log": True},  # Log (기간)
    "MACD_SLOW": {"type": "int", "low": 20, "high": 100, "log": True},  # Log (기간)
    "MACD_SIGNAL": {"type": "int", "low": 5, "high": 20, "log": True},  # Log (기간)
    "ICHIMOKU_TENKAN": {"type": "int", "low": 7, "high": 20, "log": True},  # Log (기간)
    "ICHIMOKU_KIJUN": {"type": "int", "low": 20, "high": 60, "log": True},  # Log (기간)
    "ICHIMOKU_SENKOU_B": {
        "type": "int",
        "low": 40,
        "high": 120,
        "log": True,
    },  # Log (기간)
    "STRENGTH_FILTER_PERIOD": {
        "type": "int",
        "low": 7,
        "high": 50,
        "log": True,
    },  # Log (기간)
    "VHF_THRESHOLD": {
        "type": "float",
        "low": 0.2,
        "high": 0.6,
        "step": 0.01,
    },  # Linear (임계값)
    "MFI_THRESHOLD": {
        "type": "int",
        "low": 10,
        "high": 50,
        "step": 5,
    },  # Linear (임계값)
    "RSI_OVERBOUGHT": {
        "type": "int",
        "low": 65,
        "high": 85,
        "step": 1,
    },  # Linear (임계값)
    "RSI_OVERSOLD": {
        "type": "int",
        "low": 15,
        "high": 35,
        "step": 1,
    },  # Linear (임계값)
    "STOCH_OVERBOUGHT": {
        "type": "int",
        "low": 75,
        "high": 95,
        "step": 1,
    },  # Linear (임계값)
    "STOCH_OVERSOLD": {
        "type": "int",
        "low": 5,
        "high": 25,
        "step": 1,
    },  # Linear (임계값)
    "STOCH_RSI_OVERBOUGHT": {
        "type": "int",
        "low": 70,
        "high": 90,
        "step": 1,
    },  # Linear (임계값)
    "STOCH_RSI_OVERSOLD": {
        "type": "int",
        "low": 10,
        "high": 30,
        "step": 1,
    },  # Linear (임계값)
    "VOLUME_MA_PERIOD": {
        "type": "int",
        "low": 10,
        "high": 50,
        "log": True,
    },  # Log (기간)
    "SAR_STEP": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.005},
    # VWAP Parameters
    "VWAP_STD_MULT": {
        "type": "float",
        "low": 0.5,
        "high": 2.5,
        "step": 0.1,
    },  # VWAP 표준편차 밴드 (Mean Reversion 용)
    # CMF Parameters
    "CMF_PERIOD": {"type": "int", "low": 10, "high": 40, "log": True},  # Log (기간)
    "CMF_THRESHOLD": {
        "type": "float",
        "low": 0.0,
        "high": 0.15,
        "step": 0.01,
    },  # Linear (임계값) - 추세 추종: 양수 자금 유입만 허용
    # ER (Kaufman Efficiency Ratio) Parameters
    "ER_THRESHOLD": {
        "type": "float",
        "low": 0.3,
        "high": 0.8,
        "step": 0.05,
    },  # H > 임계값: Trending
    # NATR (Normalized ATR) Parameters
    "NATR_THRESHOLD": {
        "type": "float",
        "low": 0.5,
        "high": 2.0,
        "step": 0.1,
    },  # 최소 변동성 기준
    # DMI (TREND) Parameters
    "DMI_PERIOD": {"type": "int", "low": 7, "high": 28, "log": True},
    # Aroon (TREND) Parameters
    "AROON_PERIOD": {"type": "int", "low": 7, "high": 28, "log": True},
    # Garman-Klass volatility (STRENGTH) Parameters
    "GK_PERIOD": {"type": "int", "low": 14, "high": 60, "log": True},
    "GK_THRESHOLD": {"type": "float", "low": 1e-5, "high": 0.01, "log": True},
    # Force Index (STRENGTH) Parameters
    "FORCE_INDEX_PERIOD": {"type": "int", "low": 2, "high": 13, "log": True},
    "FORCE_INDEX_THRESHOLD": {"type": "float", "low": 0.0, "high": 1e6, "log": True},
    # Williams %R (STRENGTH) - range [-100, 0], overbought near 0, oversold near -100
    "WILLR_OVERBOUGHT": {"type": "float", "low": -30.0, "high": -10.0, "step": 1.0},
    "WILLR_OVERSOLD": {"type": "float", "low": -90.0, "high": -70.0, "step": 1.0},
    # OBV (STRENGTH) - MA period for OBV vs OBV_MA trend confirmation
    "OBV_MA_PERIOD": {"type": "int", "low": 10, "high": 50, "log": True},
    # Time-Based Exit Parameters
    "TIME_EXIT_PROFIT_THRESHOLD": {
        "type": "float",
        "low": 0.0,
        "high": 2.0,
        "step": 0.1,
    },  # ATR 단위 최소 수익 (0 = 무조건 청산, 2 = 2 ATR 이상 수익일 때만 보유)
    # [NEW] Panic Exit Parameters
    "RSI_EXIT_THRESHOLD": {
        "type": "int",
        "low": 75,
        "high": 95,
        "step": 1,
    },  # 75~95 (Long Exit), 25~5 (Short Exit)
    "ENABLE_TREND_EXIT": {"type": "categorical", "choices": [True, False]},
    # [NEW] Active Position Management (Spot)
    "ENABLE_SCALE_OUT": {"type": "categorical", "choices": [False, True]},
    "SCALE_OUT_TRIGGER_ATR": {"type": "float", "low": 0.8, "high": 3.0, "step": 0.1},
    "SCALE_OUT_RATIO": {"type": "float", "low": 0.30, "high": 0.70, "step": 0.05},
    # [NEW] Safe Entry Filters
    "RSI_ENTRY_MAX": {
        "type": "categorical",
        "choices": [None, 70, 75, 80, 85],
    },  # Don't buy if RSI > X (Overbought Top)
    "NATR_ENTRY_MIN": {
        "type": "float",
        "low": 0.2,
        "high": 1.5,
        "step": 0.1,
    },  # Don't buy if Volatility < X (Dead Market)
    # [NEW] Dynamic Risk Sizing Parameters (Relaxed for Stability)
    "USE_DYNAMIC_RISK": {"type": "categorical", "choices": [False, True]},
    "STRONG_REGIME_HURST": {
        "type": "float",
        "low": 0.52,
        "high": 0.62,
        "step": 0.01,
    },  # 0.52만 넘어도 추세로 인정
    "STRONG_REGIME_NATR": {
        "type": "float",
        "low": 0.7,
        "high": 1.6,
        "step": 0.1,
    },  # 낮은 변동성에서도 공격적 진입
    "STRONG_REGIME_MULTIPLIER": {"type": "float", "low": 1.2, "high": 1.8, "step": 0.1},
    "WEAK_REGIME_HURST": {
        "type": "float",
        "low": 0.40,
        "high": 0.50,
        "step": 0.01,
    },  # 진짜 역추세일 때만 감액
    "WEAK_REGIME_MULTIPLIER": {
        "type": "float",
        "low": 0.5,
        "high": 0.9,
        "step": 0.1,
    },  # 감액 폭을 완화 (최소 0.5배 유지)
    "PANIC_REGIME_NATR": {
        "type": "float",
        "low": 4.5,
        "high": 7.5,
        "step": 0.5,
    },  # 진짜 패닉일 때만 방어
    "PANIC_REGIME_MULTIPLIER": {"type": "float", "low": 0.1, "high": 0.4, "step": 0.05},
    # Hurst Exponent Parameters
    "HURST_PERIOD": {
        "type": "int",
        "low": 100,
        "high": 300,
        "log": True,
    },  # Log (기간) - 통계적 신뢰도를 위한 최소 100
    "HURST_TREND_THRESHOLD": {
        "type": "float",
        "low": 0.52,
        "high": 0.65,
        "step": 0.01,
    },  # H > 임계값: Trending (금융 시계열 현실 반영)
    "HURST_RANDOM_THRESHOLD": {
        "type": "float",
        "low": 0.45,
        "high": 0.50,
        "step": 0.01,
    },  # H < 임계값: Random (진입 금지)
}


def GET_SEARCH_SPACE(mode, market_type="futures"):
    """
    Returns the search space for UNIFIED/ALL mode and market type.
    mode: 'UNIFIED', 'ALL'
    market_type: 'futures', 'spot'
    """
    space = deepcopy(BASE_SEARCH_SPACE)
    mode = mode.upper()
    market_type = market_type.lower()
    if mode not in {"UNIFIED", "ALL"}:
        mode = "UNIFIED"
    cfg = UNIFIED_CONFIG

    # [UNIFIED] Supports both Futures and Spot with stable timeframes
    # 15m, 30m added for Small Capital Rotation
    space["TIMEFRAME"] = {
        "type": "categorical",
        "choices": ["15m", "30m", "1h", "4h", "1d"],
    }
    # Wide ATR range to accommodate all strategies
    space["ATR_STOP_LOSS_MULT"] = {
        "type": "float",
        "low": 1.0,
        "high": 6.0,
        "step": 0.5,
    }
    space["ATR_MULTIPLIER"] = {"type": "float", "low": 1.5, "high": 8.0, "step": 0.5}

    # Apply Mode-Specific Configs
    space["ENTRY_PERIOD"] = {"type": "int", **cfg["ENTRY_PERIOD"]}
    space["MA_PERIOD"] = {"type": "int", **cfg["MA_PERIOD"]}
    space["ATR_PERIOD"] = {"type": "int", **cfg["ATR_PERIOD"]}
    space["STOP_LOSS_PCT"] = {"type": "float", **cfg["SL_PCT"]}
    space["TAKE_PROFIT_ATR_MULT"] = {"type": "float", **cfg["TP_ATR_MULT"]}
    space["ADX_THRESHOLD"] = {"type": "int", **cfg["ADX_THRESH"]}

    # RVOL Filter (Mode-Specific Range)
    if "VOL_THRESHOLD" in cfg:
        space["VOLUME_THRESHOLD_MULT"] = {"type": "float", **cfg["VOL_THRESHOLD"]}

    # Time-Based Exit (Opportunity Cost Management)
    if "MAX_HOLDING_BARS" in cfg:
        space["MAX_HOLDING_BARS"] = {"type": "int", **cfg["MAX_HOLDING_BARS"]}

    # Trailing stop activation (profit-protection)
    if "TRAILING_ACTIVATION_ATR" in cfg:
        space["TRAILING_ACTIVATION_ATR"] = {
            "type": "float",
            **cfg["TRAILING_ACTIVATION_ATR"],
        }

    # === MARKET TYPE OVERRIDES ===
    if market_type == "futures":
        # [Futures] Baseline for small-cap realism:
        # keep enough upside, but avoid immediate account volatility explosion.
        space["RISK_PER_TRADE"] = {
            "type": "float",
            "low": 0.012,
            "high": 0.05,
            "step": 0.001,
        }
        # Leverage resolution upgrade:
        # search in finer half-step granularity and cover full exchange-safe range.
        space["LEVERAGE"] = {"type": "float", "low": 1.0, "high": 10.0, "step": 1.0}

        # [UNIFIED-FUTURES] Small-cap production profile (about 0.8M KRW account)
        if mode == "UNIFIED" or mode == "ALL":
            # Structure choices: Phase A + SMA/HMA for MA diversity (lag spectrum).
            space["ENTRY_TYPE"] = {
                "type": "categorical",
                "choices": ["DONCHIAN", "BOLLINGER", "KELTNER", "CCI"],
            }
            space["TREND_FILTER_TYPE"] = {
                "type": "categorical",
                "choices": [
                    "SMA",
                    "EMA",
                    "HMA",
                    "SUPERTREND",
                    "ICHIMOKU",
                    "MACD",
                    "VWAP",
                    "DMI",
                    "AROON",
                ],
            }
            space["STRENGTH_FILTER_TYPE"] = {
                "type": "categorical",
                "choices": [
                    "NONE",
                    "ADX",
                    "RSI",
                    "NATR",
                    "HURST",
                    "VHF",
                    "MFI",
                    "STOCH_RSI",
                    "STOCHASTIC",
                    "CMF",
                    "ER",
                    "GARMAN_KLASS",
                    "FORCE_INDEX",
                    "WILLIAMS_R",
                    "OBV",
                ],
            }
            space["USE_TAKE_PROFIT"] = {"type": "categorical", "choices": [True, False]}
            space["USE_VOLUME_FILTER"] = {
                "type": "categorical",
                "choices": [True, False],
            }
            space["ENABLE_TREND_EXIT"] = {
                "type": "categorical",
                "choices": [True, False],
            }

            # Timeframes: reduce fee/slippage tax while preserving responsiveness.
            space["TIMEFRAME"] = {"type": "categorical", "choices": ["30m", "1h", "4h"]}

            # Risk / leverage: keep return potential but control drawdown speed.
            space["RISK_PER_TRADE"] = {
                "type": "float",
                "low": 0.012,
                "high": 0.035,
                "step": 0.001,
            }
            space["LEVERAGE"] = {"type": "float", "low": 1.0, "high": 10.0, "step": 1.0}

            # Core execution bounds
            space["ENTRY_PERIOD"] = {"type": "int", "low": 12, "high": 140, "log": True}
            space["MA_PERIOD"] = {"type": "int", "low": 10, "high": 170, "log": True}
            space["ATR_PERIOD"] = {"type": "int", "low": 10, "high": 40, "log": True}
            space["STOP_LOSS_PCT"] = {
                "type": "float",
                "low": 0.008,
                "high": 0.03,
                "step": 0.002,
            }
            space["ATR_STOP_LOSS_MULT"] = {
                "type": "float",
                "low": 1.5,
                "high": 3.5,
                "step": 0.25,
            }
            space["TAKE_PROFIT_ATR_MULT"] = {
                "type": "float",
                "low": 1.8,
                "high": 8.0,
                "log": True,
            }
            space["ATR_MULTIPLIER"] = {
                "type": "float",
                "low": 2.0,
                "high": 5.0,
                "step": 0.5,
            }
            space["MAX_HOLDING_BARS"] = {
                "type": "int",
                "low": 24,
                "high": 300,
                "log": True,
            }
            space["TRAILING_ACTIVATION_ATR"] = {
                "type": "float",
                "low": 1.0,
                "high": 4.0,
                "step": 0.5,
            }
            space["TIME_EXIT_PROFIT_THRESHOLD"] = {
                "type": "float",
                "low": 0.2,
                "high": 1.0,
                "step": 0.1,
            }
            space["RSI_EXIT_THRESHOLD"] = {
                "type": "int",
                "low": 85,
                "high": 98,
                "step": 1,
            }

            # Indicator/detail bounds
            space["KELTNER_ATR_MULT"] = {
                "type": "float",
                "low": 1.2,
                "high": 2.2,
                "step": 0.1,
            }
            space["BB_STD"] = {"type": "float", "low": 1.8, "high": 2.6, "step": 0.1}
            space["CCI_THRESHOLD"] = {"type": "int", "low": 80, "high": 140, "step": 10}
            space["SUPERTREND_MULT"] = {
                "type": "float",
                "low": 1.2,
                "high": 3.2,
                "log": True,
            }
            space["SUPERTREND_PERIOD"] = {
                "type": "int",
                "low": 7,
                "high": 34,
                "log": True,
            }
            space["ICHIMOKU_TENKAN"] = {
                "type": "int",
                "low": 9,
                "high": 18,
                "log": True,
            }
            space["ICHIMOKU_KIJUN"] = {
                "type": "int",
                "low": 24,
                "high": 42,
                "log": True,
            }
            space["ICHIMOKU_SENKOU_B"] = {
                "type": "int",
                "low": 52,
                "high": 90,
                "log": True,
            }
            space["MACD_FAST"] = {"type": "int", "low": 8, "high": 18, "log": True}
            space["MACD_SLOW"] = {"type": "int", "low": 22, "high": 45, "log": True}
            space["MACD_SIGNAL"] = {"type": "int", "low": 6, "high": 14, "log": True}
            space["VWAP_STD_MULT"] = {
                "type": "float",
                "low": 1.0,
                "high": 2.2,
                "step": 0.1,
            }
            space["DMI_PERIOD"] = {"type": "int", "low": 10, "high": 28, "log": True}
            space["AROON_PERIOD"] = {"type": "int", "low": 10, "high": 28, "log": True}
            space["STRENGTH_FILTER_PERIOD"] = {
                "type": "int",
                "low": 10,
                "high": 35,
                "log": True,
            }
            space["ADX_THRESHOLD"] = {"type": "int", "low": 18, "high": 35, "step": 1}
            space["RSI_OVERBOUGHT"] = {"type": "int", "low": 68, "high": 82, "step": 1}
            space["RSI_OVERSOLD"] = {"type": "int", "low": 20, "high": 32, "step": 1}
            space["MFI_THRESHOLD"] = {"type": "int", "low": 20, "high": 45, "step": 5}
            space["STOCH_RSI_OVERBOUGHT"] = {
                "type": "int",
                "low": 75,
                "high": 90,
                "step": 1,
            }
            space["STOCH_RSI_OVERSOLD"] = {
                "type": "int",
                "low": 10,
                "high": 25,
                "step": 1,
            }
            space["STOCH_OVERBOUGHT"] = {
                "type": "int",
                "low": 78,
                "high": 92,
                "step": 1,
            }
            space["STOCH_OVERSOLD"] = {"type": "int", "low": 8, "high": 22, "step": 1}
            space["CMF_PERIOD"] = {"type": "int", "low": 14, "high": 35, "log": True}
            space["CMF_THRESHOLD"] = {
                "type": "float",
                "low": 0.0,
                "high": 0.12,
                "step": 0.01,
            }
            space["ER_THRESHOLD"] = {
                "type": "float",
                "low": 0.35,
                "high": 0.75,
                "step": 0.05,
            }
            space["GK_PERIOD"] = {"type": "int", "low": 14, "high": 45, "log": True}
            space["GK_THRESHOLD"] = {
                "type": "float",
                "low": 1e-5,
                "high": 0.005,
                "log": True,
            }
            space["FORCE_INDEX_PERIOD"] = {
                "type": "int",
                "low": 2,
                "high": 10,
                "log": True,
            }
            space["FORCE_INDEX_THRESHOLD"] = {
                "type": "float",
                "low": 0.0,
                "high": 1e5,
                "log": True,
            }
            space["WILLR_OVERBOUGHT"] = {
                "type": "float",
                "low": -28.0,
                "high": -12.0,
                "step": 1.0,
            }
            space["WILLR_OVERSOLD"] = {
                "type": "float",
                "low": -88.0,
                "high": -72.0,
                "step": 1.0,
            }
            space["OBV_MA_PERIOD"] = {"type": "int", "low": 12, "high": 40, "log": True}
            space["VOLUME_THRESHOLD_MULT"] = {
                "type": "float",
                "low": 1.0,
                "high": 1.6,
                "log": True,
            }
            space["VOLUME_MA_PERIOD"] = {
                "type": "int",
                "low": 12,
                "high": 30,
                "log": True,
            }
            space["SAR_STEP"] = {
                "type": "float",
                "low": 0.01,
                "high": 0.03,
                "step": 0.005,
            }

            # Dynamic risk: keep enabled/usable with moderate multipliers.
            space["USE_DYNAMIC_RISK"] = {
                "type": "categorical",
                "choices": [False, True],
            }
            space["STRONG_REGIME_HURST"] = {
                "type": "float",
                "low": 0.53,
                "high": 0.60,
                "step": 0.01,
            }
            space["STRONG_REGIME_NATR"] = {
                "type": "float",
                "low": 0.8,
                "high": 1.5,
                "step": 0.1,
            }
            space["STRONG_REGIME_MULTIPLIER"] = {
                "type": "float",
                "low": 1.1,
                "high": 1.4,
                "step": 0.1,
            }
            space["WEAK_REGIME_HURST"] = {
                "type": "float",
                "low": 0.43,
                "high": 0.49,
                "step": 0.01,
            }
            space["WEAK_REGIME_MULTIPLIER"] = {
                "type": "float",
                "low": 0.6,
                "high": 0.9,
                "step": 0.1,
            }
            space["PANIC_REGIME_NATR"] = {
                "type": "float",
                "low": 4.5,
                "high": 7.0,
                "step": 0.5,
            }
            space["PANIC_REGIME_MULTIPLIER"] = {
                "type": "float",
                "low": 0.15,
                "high": 0.35,
                "step": 0.05,
            }

        # [ISOLATION] Remove Spot-only parameters to prevent ghost dimensions in Futures
        for key in (
            "NATR_ENTRY_MIN",
            "ENABLE_SCALE_OUT",
            "SCALE_OUT_TRIGGER_ATR",
            "SCALE_OUT_RATIO",
        ):
            if key in space:
                del space[key]

    else:
        # [Spot] Long-only, no leverage
        if "LEVERAGE" in space:
            del space["LEVERAGE"]
        if "ENABLE_TREND_EXIT" in space:
            del space["ENABLE_TREND_EXIT"]

        # Long-only spot: allow broader sizing exploration for bull-alpha capture.
        space["RISK_PER_TRADE_SPOT"] = {
            "type": "float",
            "low": 0.10,
            "high": 0.35,
            "step": 0.02,
        }

        # Spot-specific safety filters: balanced to allow trend-following while preventing extreme entries
        space["RSI_ENTRY_MAX"] = {
            "type": "int",
            "low": 75,
            "high": 96,
            "step": 2,
        }  # Raised lower bound to preserve trend-following
        space["NATR_ENTRY_MIN"] = {
            "type": "float",
            "low": 0.0,
            "high": 0.8,
            "step": 0.1,
        }  # Raised upper bound for volatility filter range
        # Trend gate policy for long-only spot:
        # STRICT=hourly&daily, SOFT=hourly|daily, OFF=hourly only.
        # LONG-only spot: STRICT often over-filters in risk-off/chop regimes.
        # STRICT mode is critical for long-only: upper timeframe trend confirmation
        space["TREND_GATE_MODE"] = {
            "type": "categorical",
            "choices": ["STRICT", "SOFT", "OFF"],
        }

        # Keep dynamic-risk family fully searchable for spot.
        space["USE_DYNAMIC_RISK"] = {"type": "categorical", "choices": [False, True]}

        # Spot needs looser TP ceiling
        if "TAKE_PROFIT_ATR_MULT" in space:
            space["TAKE_PROFIT_ATR_MULT"]["high"] = max(
                space["TAKE_PROFIT_ATR_MULT"]["high"], 15.0
            )

        # [UNIFIED-SPOT] Keep parity with futures-style parameter coverage, with broader upside exploration
        # Structure choices: same family as futures unified (minus leverage dimension)
        space["ENTRY_TYPE"] = {
            "type": "categorical",
            "choices": ["DONCHIAN", "BOLLINGER", "KELTNER"],
        }
        space["TREND_FILTER_TYPE"] = {
            "type": "categorical",
            "choices": ["EMA", "SUPERTREND", "ICHIMOKU"],
        }
        space["STRENGTH_FILTER_TYPE"] = {
            "type": "categorical",
            "choices": ["NONE", "ADX", "RSI", "NATR", "HURST", "VHF"],
        }
        space["USE_TAKE_PROFIT"] = {"type": "categorical", "choices": [True, False]}
        space["USE_VOLUME_FILTER"] = {"type": "categorical", "choices": [True, False]}

        # Long-only spot unified set: 30m, 1h, 4h for spot search.
        space["TIMEFRAME"] = {"type": "categorical", "choices": ["30m", "1h", "4h"]}

        # Core execution bounds (spot UNIFIED): realistic ranges for long-only crypto.
        space["ENTRY_PERIOD"] = {
            "type": "int",
            "low": 12,
            "high": 140,
            "log": True,
        }  # Expanded upper bound for longer trend capture
        space["MA_PERIOD"] = {"type": "int", "low": 10, "high": 120, "log": True}
        space["ATR_PERIOD"] = {"type": "int", "low": 8, "high": 28, "log": True}
        # Range tightened to cap excessive downside tails in long-only spot.
        space["STOP_LOSS_PCT"] = {
            "type": "float",
            "low": 0.010,
            "high": 0.050,
            "step": 0.003,
        }
        space["ATR_STOP_LOSS_MULT"] = {
            "type": "float",
            "low": 1.2,
            "high": 3.2,
            "step": 0.2,
        }  # Reduced from 4.6 (realistic max)
        space["TAKE_PROFIT_ATR_MULT"] = {
            "type": "float",
            "low": 2.0,
            "high": 12.0,
            "log": True,
        }  # TODO #4
        space["ATR_MULTIPLIER"] = {
            "type": "float",
            "low": 1.5,
            "high": 4.5,
            "step": 0.3,
        }
        space["MAX_HOLDING_BARS"] = {"type": "int", "low": 20, "high": 180, "log": True}
        space["TRAILING_ACTIVATION_ATR"] = {
            "type": "float",
            "low": 0.5,
            "high": 4.5,
            "step": 0.5,
        }
        space["TIME_EXIT_PROFIT_THRESHOLD"] = {
            "type": "float",
            "low": 0.0,
            "high": 1.8,
            "step": 0.1,
        }
        space["RSI_EXIT_THRESHOLD"] = {"type": "int", "low": 78, "high": 98, "step": 1}

        # Indicator/detail bounds: reduce over-smoothing and over-filtering in bear/choppy regimes.
        space["KELTNER_ATR_MULT"] = {
            "type": "float",
            "low": 1.0,
            "high": 2.8,
            "step": 0.1,
        }
        space["BB_STD"] = {"type": "float", "low": 1.6, "high": 3.2, "step": 0.1}
        space["SUPERTREND_MULT"] = {
            "type": "float",
            "low": 1.0,
            "high": 3.2,
            "log": True,
        }
        space["SUPERTREND_PERIOD"] = {"type": "int", "low": 7, "high": 24, "log": True}
        space["ICHIMOKU_TENKAN"] = {"type": "int", "low": 7, "high": 24, "log": True}
        space["ICHIMOKU_KIJUN"] = {"type": "int", "low": 20, "high": 42, "log": True}
        space["ICHIMOKU_SENKOU_B"] = {"type": "int", "low": 40, "high": 78, "log": True}
        space["STRENGTH_FILTER_PERIOD"] = {
            "type": "int",
            "low": 8,
            "high": 28,
            "log": True,
        }
        space["ADX_THRESHOLD"] = {
            "type": "int",
            "low": 15,
            "high": 40,
            "step": 1,
        }  # Keep minimum in trending zone
        space["RSI_OVERBOUGHT"] = {
            "type": "int",
            "low": 70,
            "high": 90,
            "step": 1,
        }  # Raised from 65 to preserve trend-following
        space["RSI_OVERSOLD"] = {"type": "int", "low": 10, "high": 35, "step": 1}
        space["VOLUME_THRESHOLD_MULT"] = {
            "type": "float",
            "low": 1.0,
            "high": 1.30,
            "log": True,
        }
        space["VOLUME_MA_PERIOD"] = {"type": "int", "low": 8, "high": 30, "log": True}
        space["SAR_STEP"] = {"type": "float", "low": 0.005, "high": 0.04, "step": 0.005}

        # Dynamic-risk bounds: reduce extreme sensitivity and keep robust regimes.
        space["STRONG_REGIME_HURST"] = {
            "type": "float",
            "low": 0.53,
            "high": 0.65,
            "step": 0.01,
        }
        space["STRONG_REGIME_NATR"] = {
            "type": "float",
            "low": 0.6,
            "high": 1.8,
            "step": 0.1,
        }
        space["STRONG_REGIME_MULTIPLIER"] = {
            "type": "float",
            "low": 1.00,
            "high": 1.35,
            "step": 0.05,
        }
        space["WEAK_REGIME_HURST"] = {
            "type": "float",
            "low": 0.40,
            "high": 0.50,
            "step": 0.01,
        }
        space["WEAK_REGIME_MULTIPLIER"] = {
            "type": "float",
            "low": 0.50,
            "high": 0.85,
            "step": 0.05,
        }
        space["PANIC_REGIME_NATR"] = {
            "type": "float",
            "low": 3.0,
            "high": 8.5,
            "step": 0.5,
        }
        space["PANIC_REGIME_MULTIPLIER"] = {
            "type": "float",
            "low": 0.10,
            "high": 0.30,
            "step": 0.05,
        }
        # Risk-off gate controls (previously fixed in engine defaults): make them searchable.
        space["ENABLE_RISK_OFF_HARD_GATE"] = {
            "type": "categorical",
            "choices": [False, True],
        }
        space["RISK_OFF_EXIT_ON_TRIGGER"] = {
            "type": "categorical",
            "choices": [False, True],
        }
        space["RISK_OFF_COOLDOWN_BARS"] = {
            "type": "int",
            "low": 0,
            "high": 4,
            "step": 1,
        }
        # Breakeven controls: protect PF during choppy/post-breakout reversals.
        space["ENABLE_BREAKEVEN"] = {"type": "categorical", "choices": [False, True]}
        space["BREAKEVEN_BUFFER_PCT"] = {
            "type": "float",
            "low": 0.001,
            "high": 0.008,
            "step": 0.001,
        }

        # Active position management: ENABLED for long-only trend maximization
        space["ENABLE_SCALE_OUT"] = {
            "type": "categorical",
            "choices": [False, True],
        }  # ENABLED
        # Delay partial take-profit to avoid cutting trend winners too early.
        space["SCALE_OUT_TRIGGER_ATR"] = {
            "type": "float",
            "low": 1.8,
            "high": 3.5,
            "step": 0.1,
        }  # Raised lower bound
        space["SCALE_OUT_RATIO"] = {
            "type": "float",
            "low": 0.20,
            "high": 0.40,
            "step": 0.05,
        }  # Adjusted range

    return space
