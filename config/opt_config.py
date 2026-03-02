from typing import Any, Dict, List

# ==============================================================================
# OPTIMIZATION V2 SEARCH SPACE & CONFIGURATION
# ==============================================================================

OPT_V2_CONFIG: Dict[str, Any] = {
    "total_trials": 1500,
    "n_startup_trials": 150, # [NEW] Priority 6: 완전 랜덤 탐색 구간을 20%에서 10%로 축소하여 지능형 탐색 조기 시작
    "seeds": [42, 137],
    "n_jobs": 4,
    "TARGET_TIMEFRAMES": ["1h", "4h"],
}

# 1. Timeframe-Invariant Parameters (Common Base)
BASE_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "W_BREAKOUT": {"type": "categorical", "choices": [0.0, 0.5, 1.0]},
    "W_TREND":    {"type": "categorical", "choices": [0.0, 0.5, 1.0]},
    "W_VOLUME":   {"type": "categorical", "choices": [0.0, 0.5, 1.0]},
    "W_MEAN_REVERSION": {"type": "categorical", "choices": [0.0, 0.5, 1.0]},

    # [INSTITUTIONAL] Adaptive Thresholds (Rolling Percentiles)
    "THRESHOLD_LOOKBACK": {"type": "int", "low": 180, "high": 720, "step": 60}, # 60~360 -> 180~720 (기관급 매크로 시야 확장)
    "THRESHOLD_QUANTILE": {"type": "float", "low": 0.50, "high": 0.65, "step": 0.05}, # 0.60~0.85 -> 0.50~0.65 (지표가 폭발하지 않은 '조용한 돌파(Slow Grind)' 초입도 허용)

    # Layer 4: Exit & Risk Mgmt
    "STOP_LOSS_TYPE": {"type": "categorical", "choices": ["FIXED", "ATR"]},
    "USE_TAKE_PROFIT": {"type": "categorical", "choices": [True, False]},
    "EXIT_TYPE": {"type": "categorical", "choices": ["ATR", "PARABOLIC_SAR"]},
    "LEVERAGE": {"type": "categorical", "choices": [2, 3]},
    
    # PSAR Search Space
    "PSAR_STEP": {"type": "float", "low": 0.01, "high": 0.03, "step": 0.01},
    "PSAR_MAX": {"type": "float", "low": 0.10, "high": 0.30, "step": 0.1},
}

def get_search_space_v2(tf: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns a timeframe-specific search space based on financial engineering principles.
    """
    space: Dict[str, Dict[str, Any]] = {k: v.copy() for k, v in BASE_SEARCH_SPACE.items()}

    # ---------------------------------------------------------
    # 1H (Strict Mean Reversion Specialist)
    # ---------------------------------------------------------
    if tf == "1h":
        space.update({
            "W_BREAKOUT": {"type": "categorical", "choices": [0.0]},
            "W_TREND": {"type": "categorical", "choices": [0.0]},
            "W_MEAN_REVERSION": {"type": "categorical", "choices": [1.0]},

            # [INSTITUTIONAL] 역추세 전용 극단값 탐지 (Panic Selling Catcher)
            # 추세추종과 달리, 역추세는 무조건 단기적인 '극단적 패닉(상위 5~15%)'에만 진입해야 함.
            "THRESHOLD_LOOKBACK": {"type": "int", "low": 24, "high": 120, "step": 24}, # 1일~5일 단기 시야
            "THRESHOLD_QUANTILE": {"type": "float", "low": 0.85, "high": 0.95, "step": 0.05}, # 극단적 패닉만 줍기

            # 역추세 특화: 노출 시간 최소화(치고 빠지기), 넓은 손절폭(꼬리 휩쏘 방어)
            "MAX_HOLDING_BARS": {"type": "int", "low": 6, "high": 24, "step": 6},
            "TIME_EXIT_PROFIT_THRESHOLD": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.2},
            "STOP_LOSS_PCT": {"type": "float", "low": 0.03, "high": 0.08, "step": 0.01},
            "ATR_STOP_LOSS_MULT": {"type": "float", "low": 2.0, "high": 3.5, "step": 0.5},
            "TAKE_PROFIT_ATR_MULT": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
            "ATR_MULTIPLIER": {"type": "float", "low": 2.0, "high": 3.5, "step": 0.5},

            "VWAP_STD_MULT": {"type": "float", "low": 2.0, "high": 3.5, "step": 0.5},
            "STOCH_RSI_PERIOD": {"type": "int", "low": 14, "high": 28, "step": 7},
            "STOCH_RSI_EXTREME": {"type": "int", "low": 10, "high": 25, "step": 5},
            "CMF_PERIOD": {"type": "int", "low": 20, "high": 40, "step": 10},
        })

    # ---------------------------------------------------------
    # 4H (Pure Institutional Macro Trend Following & ER Risk Sizing)
    # ---------------------------------------------------------
    elif tf == "4h":
        space.update({
            # [INSTITUTIONAL] 꼬리 자르기 금지: 4H는 역추세를 하지 않으므로 지정가 익절(TP) 영구 금지
            "USE_TAKE_PROFIT": {"type": "categorical", "choices": [False]},
            # [INSTITUTIONAL] 조급한 청산(PSAR) 금지: 대추세의 잔파도를 견디기 위해 무조건 ATR 트레일링 강제
            "EXIT_TYPE": {"type": "categorical", "choices": ["ATR"]},
            
            # [INSTITUTIONAL] 가랑비(Whipsaw)에 젖어 죽는 '겁쟁이 청산' 로직 영구 철거
            # 오직 물리적 방패(Trailing Stop)가 깨질 때만 청산하여 수익의 오른쪽 꼬리(Fat-tail)를 무한대로 염.
            "RSI_EXIT_THRESHOLD": {"type": "categorical", "choices": [100.0]}, # 폭등장(RSI 80 이상) 패닉 셀 금지
            "ENABLE_TREND_EXIT": {"type": "categorical", "choices": [False]}, # 보조지표 점수가 꺾였다고 미리 도망치는 행위 금지
            
            "W_BREAKOUT": {"type": "categorical", "choices": [0.5, 1.0]},
            "W_TREND": {"type": "categorical", "choices": [0.5, 1.0]},
            "W_MEAN_REVERSION": {"type": "categorical", "choices": [0.0]}, # 역추세 완전 금지 (4H는 순수 추세 전용)
            
            # [INSTITUTIONAL] 암호화폐 선물의 거래량 노이즈(청산, 자전거래)를 배제하기 위해 Volume Factor 강제 종료
            "W_VOLUME": {"type": "categorical", "choices": [0.0]},
            
            # [IMPROVEMENT] 너무 잦은 휩쏘(18봉)와 너무 늦은 막물(96봉)을 피해, 크립토 황금 타점인 5일~14일로 문지기 재배치
            "ENTRY_PERIOD": {"type": "int", "low": 30, "high": 84, "step": 6},
            # [INSTITUTIONAL] 너무 빠른 강제 청산(Time Exit)을 막고 수익의 오른쪽 꼬리를 무한대로 열기 위해 20일~80일로 대폭 확장
            "MAX_HOLDING_BARS": {"type": "int", "low": 120, "high": 480, "step": 24},
            "TIME_EXIT_PROFIT_THRESHOLD": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.5},
            "STOP_LOSS_PCT": {"type": "float", "low": 0.05, "high": 0.10, "step": 0.01},
            "ATR_STOP_LOSS_MULT": {"type": "float", "low": 3.0, "high": 5.5, "step": 0.5},
            
            # [NEW] Priority 3: Chandelier Exit (Turtle Trading Style)
            # 잔파도(-15%~-20% Pullback)에 털리지 않고 대추세를 끝까지 먹기 위해 방패(ATR)를 극단적으로 넓게(기관급) 설정
            "TRAILING_ACTIVATION_ATR": {"type": "categorical", "choices": [0.0]},
            "ATR_MULTIPLIER": {"type": "float", "low": 4.5, "high": 7.0, "step": 0.5},

            # [REMOVED] MACRO_SMA_PERIOD: 기관급 다중 모멘텀 수식의 독립성을 위해 하드 게이트 삭제
        })
    
    return space

SEARCH_SPACE_V2: Dict[str, Dict[str, Any]] = get_search_space_v2("1h")

def get_quarterly_window(reference_date=None) -> tuple[str, str, str, str]:
    import datetime
    from dateutil.relativedelta import relativedelta
    if reference_date is None: reference_date = datetime.date.today()
    elif isinstance(reference_date, str): reference_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d").date()
    elif isinstance(reference_date, datetime.datetime): reference_date = reference_date.date()
    current_quarter_start_month: int = ((reference_date.month - 1) // 3) * 3 + 1
    current_quarter_start: datetime.date = datetime.date(reference_date.year, current_quarter_start_month, 1)
    oos_end: datetime.date = current_quarter_start - datetime.timedelta(days=1)
    oos_start: datetime.date = current_quarter_start - relativedelta(months=6)
    is_start: datetime.date = oos_start - relativedelta(months=24)
    fetch_start: datetime.date = is_start - relativedelta(days=500)
    return (fetch_start.strftime("%Y-%m-%d"), is_start.strftime("%Y-%m-%d"), oos_start.strftime("%Y-%m-%d"), oos_end.strftime("%Y-%m-%d"))
