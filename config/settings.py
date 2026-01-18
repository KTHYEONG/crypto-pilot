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
