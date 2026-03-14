import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Non-TTY (e.g. Docker -d)에서 stdout/stderr 블록 버퍼링 방지 → docker logs 즉시 반영
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

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

# 매매 기록 데이터베이스 파일 (SQLite)
TRADE_HISTORY_DB = DATA_DIR / "trade_history.db"

# Binance 설정
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

# Upbit 설정
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "")

# 거래 수수료 및 슬리피지 설정 (바이낸스 선물 실제 수수료 반영)
# Binance Futures Fee Structure (VIP 0 기준):
# - Maker (지정가): 0.02%
# - Taker (시장가): 0.05%
# 백테스트는 보수적으로 Taker 수수료 기준 적용
TRADING_FEE_RATE = 0.0005  # 0.05% (Taker 기준, 진입/청산 각각)
SLIPPAGE_RATE = 0.0005  # 0.05% (시장가 주문 시 예상 슬리피지)

# 스마트 주문 설정
MAKER_FEE_RATE = 0.0002  # 0.02% (지정가 주문 성공 시)
TAKER_FEE_RATE = 0.0005  # 0.05% (시장가 주문 시)
SMART_ORDER_OFFSET = 0.0003  # 0.03% (공격적 지정가 오프셋 - Maker 수수료 절감 목적)
SMART_ORDER_TIMEOUT = 10  # 10초 (지정가 체결 대기 시간)

# 선물 펀딩비 설정 (Perpetual Futures Funding Fee)
FUNDING_FEE_RATE = 0.0001  # 0.01% (8시간 마다, Binance Default)
FUNDING_INTERVAL_HOURS = 8

LOG_LEVEL = "DEBUG"

# ============================================================
# RealTrader Futures 설정
# ============================================================

# 헬스체크 파일 (봇 생존 확인용)
HEARTBEAT_FILE = LOG_DIR / "trader_heartbeat.json"
FUTURES_STATE_FILE = DATA_DIR / "futures_trading_state.json"

# --- API 재시도 및 타임아웃 정책 ---
API_READ_TIMEOUT = 20  # 데이터 조회 (OHLCV, 잔고 등)
API_ORDER_TIMEOUT = 10  # 주문 제출/취소
API_CHECK_TIMEOUT = 5  # 주문 상태 확인/조회
API_RETRY_ATTEMPTS = 3  # 최대 재시도 횟수
API_RETRY_WAIT_MIN = 1  # 최소 대기 시간 (초)
API_RETRY_WAIT_MAX = 30  # 최대 대기 시간 (초)

# --- 포지션 사이징 임계값 ---
MIN_BALANCE_USDT = 10.0  # 최소 운영 잔고 (USDT)
MIN_BALANCE_FOR_TRADE = 5.0  # 최소 거래 가능 잔고
MIN_ORDER_VALUE_USDT = 5.0  # Binance 최소 주문 금액 (바이낸스 규정 기반)
MAX_EXCHANGE_LEVERAGE = 20  # 최대 허용 레버리지 

# --- 타이밍 설정 ---
LOOP_INTERVAL_SECONDS = 10  # 메인 루프 간격
SYMBOL_DELAY_SECONDS = 2  # 심볼 간 딜레이
ERROR_SLEEP_SECONDS = 60  # 에러 발생 시 대기

# --- 캔들 동기화 오프셋 (분) ---
CANDLE_SYNC_OFFSET_SECONDS = 15

# --- 로그 회전 설정 ---
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# --- 대상 심볼 (Futures) ---
# 최적화 결과 기반, 100만원 소자본 자산 증식 전용 포트폴리오 (SOL, SUI, BTC)
FUTURES_TARGET_SYMBOLS = ["SOL/USDT", "SUI/USDT", "BTC/USDT"]

# --- 포지션 사이징 가중치 (Futures) ---
# "First-Come, First-Served" 방식: 각 심볼이 자본의 50%까지 활용 가능하도록 가중치 설정
SYMBOL_ALLOCATION_WEIGHTS = {"SOL/USDT": 0.50, "SUI/USDT": 0.50, "BTC/USDT": 0.50}

# ============================================================
# RealTrader Spot 설정 (Upbit)
# ============================================================

# 거래 DB 경로
SPOT_STRATEGY_DB = BASE_DIR / "spot_strategy.db"
SPOT_HEARTBEAT_FILE = LOG_DIR / "spot_trader_heartbeat.json"
SPOT_STATE_FILE = DATA_DIR / "spot_trading_state.json"

# --- Upbit KRW 기준 포지션 사이징 ---
MIN_POSITION_VALUE_KRW = 10_000  # 최소 포지션 인식 금액 (10,000원)
MIN_ORDER_VALUE_KRW = 5_000  # 최소 주문 금액 (5,000원)
MAX_INVEST_CAP_KRW = 100_000_000  # 심볼당 최대 투자 금액 (1억원)
MAX_TOTAL_BALANCE_KRW = 500_000_000  # 총 운영 자금 상한 (5억원)

# --- Spot 타이밍 설정 ---
SPOT_LOOP_INTERVAL_SECONDS = 10  # 메인 루프 간격 (10초)
SPOT_SYMBOL_DELAY_SECONDS = 2  # 심볼 간 딜레이

# --- 대상 심볼 (Upbit) ---
# OOS 성과 최상위 3종목 (ETH, SOL, DOGE) 압축 투자로 복리 증식 극대화
SPOT_TARGET_SYMBOLS = ["KRW-ETH", "KRW-SOL", "KRW-DOGE"]

# --- 포지션 사이징 가중치 (Spot) ---
# 추천 포트폴리오 비중: ETH(40%), SOL(30%), DOGE(30%)
SPOT_ALLOCATION_WEIGHTS = {"KRW-ETH": 0.4, "KRW-SOL": 0.3, "KRW-DOGE": 0.3}

# --- Optuna Study ---
SPOT_OPTUNA_STUDY_NAME = "spot_strategy"

# --- 백테스트 초기 잔고 (Futures) ---
FUTURES_INITIAL_BALANCE = 800.0  # USDT, backtest/optimize/verify 공통

# --- 데이터 기간 설정 (Backtest & Optimization) ---
# Futures 전용 설정
FUTURES_BACKTEST_START_DATE = "2021-01-01"
FUTURES_BACKTEST_END_DATE = "2026-02-17"
FUTURES_TRAIN_CUTOFF_DATE = "2024-04-01"

# Spot 전용 데이터 기간 (최적화: start~cutoff, 검증: cutoff~end)
SPOT_BACKTEST_START_DATE = "2021-01-01"
SPOT_BACKTEST_END_DATE = "2026-02-17"
SPOT_TRAIN_CUTOFF_DATE = "2024-04-01"
