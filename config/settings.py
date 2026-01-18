import os
from pathlib import Path
from dotenv import load_dotenv

# 로컬 환경 변수 로드 (.env 파일이 있다면)
load_dotenv()

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent

# 데이터 저장 경로
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

# 디렉토리 자동 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 데이터베이스 파일 경로 (SQLite)
DB_PATH = DATA_DIR / "trading_data.db"

# Binance 설정
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

# Upbit 설정
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "")

# 거래 수수료 및 슬리피지 설정 (보수적 적용 - 스트레스 테스트용 강화)
TRADING_FEE_RATE = 0.001       # 0.1% (매수/매도 각각)
SLIPPAGE_RATE = 0.001          # 0.1% (진입/청산 각각, 노이즈 대응용)

# 백테스팅 기본 기간
BACKTEST_START_DATE = "2020-01-01"  # 전체 데이터 시작일
BACKTEST_END_DATE = "2026-01-16"    # 전체 데이터 종료일
TRAIN_CUTOFF_DATE = "2025-01-01"    # [중요] 최적화(Train)와 검증(Test) 분리 기준일
TIMEFRAME = "1h"               # 1시간봉 기준

# 로깅 설정
LOG_LEVEL = "INFO"

# ============================================================
# RealTrader Futures 설정
# ============================================================

# 거래 DB 경로
FUTURES_STRATEGY_DB = BASE_DIR / "futures_strategy.db"
TRADE_HISTORY_DB = DATA_DIR / "trade_history.db"

# 헬스체크 파일 (봇 생존 확인용)
HEARTBEAT_FILE = LOG_DIR / "trader_heartbeat.json"

# --- API 재시도 정책 ---
API_RETRY_ATTEMPTS = 5          # 최대 재시도 횟수
API_RETRY_WAIT_MIN = 1          # 최소 대기 시간 (초)
API_RETRY_WAIT_MAX = 30         # 최대 대기 시간 (초)
API_TIMEOUT_SECONDS = 30        # API 타임아웃

# --- 포지션 사이징 임계값 ---
MIN_BALANCE_USDT = 50           # 최소 운영 잔고 (USDT)
MIN_BALANCE_FOR_TRADE = 10      # 최소 거래 가능 잔고
MIN_ORDER_VALUE_USDT = 6        # Binance 최소 주문 금액
MAX_EXCHANGE_LEVERAGE = 5       # 거래소 설정 레버리지 (버퍼)

# --- 타이밍 설정 ---
LOOP_INTERVAL_SECONDS = 30      # 메인 루프 간격
SYMBOL_DELAY_SECONDS = 2        # 심볼 간 딜레이
ERROR_SLEEP_SECONDS = 60        # 에러 발생 시 대기

# --- 캔들 동기화 오프셋 (분) ---
# 정시 봉 마감 후 N초 뒤에 실행하여 데이터 안정성 확보
CANDLE_SYNC_OFFSET_SECONDS = 15

# --- 로그 회전 설정 ---
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# --- 대상 심볼 ---
FUTURES_TARGET_SYMBOLS = ['BTC/USDT', 'ETH/USDT']

# --- 스터디 이름 (Optuna) ---
OPTUNA_STUDY_NAMES = ['futures_strategy', 'future_strategy']

# --- 포지션 사이징 가중치 (성과 기반) ---
# 근거: 2025-2026 백테스트 결과
# BTC: Return 103%, PF 11.53, WinRate 90% (압도적 성과)
# ETH: Return 31%, PF 2.69, WinRate 76% (보조)
SYMBOL_ALLOCATION_WEIGHTS = {
    'BTC/USDT': 0.75,
    'ETH/USDT': 0.25
}

# ============================================================
# RealTrader Spot 설정 (Upbit)
# ============================================================

# 거래 DB 경로
SPOT_STRATEGY_DB = BASE_DIR / "spot_strategy.db"
SPOT_HEARTBEAT_FILE = LOG_DIR / "spot_trader_heartbeat.json"
SPOT_STATE_FILE = DATA_DIR / "spot_trading_state.json"

# --- Upbit KRW 기준 포지션 사이징 ---
MIN_POSITION_VALUE_KRW = 10_000         # 최소 포지션 인식 금액 (10,000원)
MIN_ORDER_VALUE_KRW = 5_000             # 최소 주문 금액 (5,000원)
MAX_INVEST_CAP_KRW = 100_000_000        # 심볼당 최대 투자 금액 (1억원)
MAX_TOTAL_BALANCE_KRW = 500_000_000     # 총 운영 자금 상한 (5억원)

# --- Spot 타이밍 설정 ---
SPOT_LOOP_INTERVAL_SECONDS = 60         # 메인 루프 간격 (1분)
SPOT_SYMBOL_DELAY_SECONDS = 2           # 심볼 간 딜레이

# --- 대상 심볼 (Upbit) ---
SPOT_TARGET_SYMBOLS = ['KRW-BTC', 'KRW-ETH']

# --- 포지션 사이징 가중치 (성과 기반 - Spot) ---
# 근거: 2025-2026 백테스트 결과 (Spot)
# ETH: Return 6328%, PF 119801, WinRate 99.59% (최고 성과) -> 70% 할당
# BTC: Return 2236%, PF 5385, WinRate 99.32% (우수) -> 30% 할당
SPOT_ALLOCATION_WEIGHTS = {
    'KRW-BTC': 0.30,
    'KRW-ETH': 0.70
}

# --- Optuna Study ---
SPOT_OPTUNA_STUDY_NAME = 'spot_strategy'
