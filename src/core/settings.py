import contextlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Non-TTY (e.g. Docker -d)에서 stdout/stderr 블록 버퍼링 방지 → docker logs 즉시 반영
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore
    except Exception:
        # Standard stdout/stderr might not support reconfigure in some environments
        ...

# 로컬 환경 변수 로드 (.env 파일이 있다면)
load_dotenv()

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 데이터 저장 경로
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

# 디렉토리 자동 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
FUTURES_DATA_DIR = DATA_DIR / "futures"
FUTURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class FuturesStorageLayout:
    """Manages the partitioned layout of futures data files with automatic migration from flat legacy paths."""

    @staticmethod
    def get_ohlcv_path(symbol: str, timeframe: str, base_dir: Path | None = None) -> Path:
        base = base_dir if base_dir is not None else FUTURES_DATA_DIR
        safe_symbol = symbol.replace("/", "_")
        new_path = base / "ohlcv" / timeframe / f"{safe_symbol}.parquet"
        old_path = base / f"{safe_symbol}_{timeframe}.parquet"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if not new_path.exists() and old_path.exists():
            with contextlib.suppress(FileNotFoundError):
                old_path.rename(new_path)
        return new_path

    @staticmethod
    def get_enriched_path(symbol: str, timeframe: str, base_dir: Path | None = None) -> Path:
        base = base_dir if base_dir is not None else FUTURES_DATA_DIR
        safe_symbol = symbol.replace("/", "_")
        new_path = base / "enriched" / timeframe / f"{safe_symbol}.parquet"
        old_path = base / f"{safe_symbol}_{timeframe}_enriched.parquet"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if not new_path.exists() and old_path.exists():
            with contextlib.suppress(FileNotFoundError):
                old_path.rename(new_path)
        return new_path

    @staticmethod
    def get_funding_path(symbol: str, base_dir: Path | None = None) -> Path:
        base = base_dir if base_dir is not None else FUTURES_DATA_DIR
        safe_symbol = symbol.replace("/", "_")
        new_path = base / "funding" / f"{safe_symbol}.parquet"
        old_path = base / f"{safe_symbol}_funding.parquet"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if not new_path.exists() and old_path.exists():
            with contextlib.suppress(FileNotFoundError):
                old_path.rename(new_path)
        return new_path

    @staticmethod
    def get_metrics_path(symbol: str, base_dir: Path | None = None) -> Path:
        base = base_dir if base_dir is not None else FUTURES_DATA_DIR
        safe_symbol = symbol.replace("/", "_")
        new_path = base / "metrics" / f"{safe_symbol}.parquet"
        old_path = base / f"{safe_symbol}_metrics.parquet"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if not new_path.exists() and old_path.exists():
            with contextlib.suppress(FileNotFoundError):
                old_path.rename(new_path)
        return new_path

    @staticmethod
    def get_metadata_path(filename: str, base_dir: Path | None = None) -> Path:
        base = base_dir if base_dir is not None else FUTURES_DATA_DIR
        new_path = base / "metadata" / filename
        old_path = base / filename
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if not new_path.exists() and old_path.exists():
            with contextlib.suppress(FileNotFoundError):
                old_path.rename(new_path)
        return new_path


# 매매 기록 데이터베이스 파일 (SQLite)
TRADE_HISTORY_DB = DATA_DIR / "trade_history.db"

# Binance 설정
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")

# === Canonical transaction-cost model (single source of truth, per-side, Binance USDⓈ-M VIP0) ===
# 아래 *_BPS 상수가 유일한 수정 지점입니다. Decimal rate는 여기서 파생되므로 직접 수정하지 마세요.
MAKER_FEE_BPS: float = 2.0  # 0.0200% — Maker(지정가) 수수료 per side
TAKER_FEE_BPS: float = 5.0  # 0.0500% — Taker(시장가) 수수료 per side
SLIPPAGE_BPS: float = 2.0  # 시장가 주문 예상 슬리피지 per side
FUNDING_FEE_BPS_PER_8H: float = 1.0  # 펀딩비 per 8h (Binance Default)
FILLS_PER_ROUND_TRIP: int = 2  # 진입 1회 + 청산 1회

# Decimal rates DERIVED from canonical bps. Edit the *_BPS constants above, not these.
MAKER_FEE_RATE = MAKER_FEE_BPS / 10000.0
TAKER_FEE_RATE = TAKER_FEE_BPS / 10000.0
TRADING_FEE_RATE = TAKER_FEE_RATE  # 하위 호환성을 위해 유지 (기본값 Taker)
SLIPPAGE_RATE = SLIPPAGE_BPS / 10000.0

# 스마트 주문 설정
SMART_ORDER_OFFSET = 0.0003  # 0.03% (공격적 지정가 오프셋 - Maker 수수료 절감 목적)

# 선물 펀딩비 설정 (Perpetual Futures Funding Fee)
FUNDING_FEE_RATE = FUNDING_FEE_BPS_PER_8H / 10000.0  # 0.01% (8시간 마다, Binance Default)


def round_trip_cost_bps(*, taker_entry: bool = True, taker_exit: bool = True) -> float:
    """Round-trip transaction cost in bps (entry fee + exit fee + 2x slippage).

    실행 시뮬레이터는 양 leg을 모두 Taker로 체결하므로 기본값은 taker/taker입니다.

    Args:
        taker_entry: 진입을 Taker로 체결하면 True, Maker면 False.
        taker_exit: 청산을 Taker로 체결하면 True, Maker면 False.

    Returns:
        Round-trip 총 비용(bps).

    """
    entry_fee = TAKER_FEE_BPS if taker_entry else MAKER_FEE_BPS
    exit_fee = TAKER_FEE_BPS if taker_exit else MAKER_FEE_BPS
    return entry_fee + exit_fee + FILLS_PER_ROUND_TRIP * SLIPPAGE_BPS


# ============================================================
# RealTrader Futures 설정
# ============================================================

# 헬스체크 파일 (봇 생존 확인용)
HEARTBEAT_FILE = LOG_DIR / "trader_heartbeat.json"
FUTURES_STATE_FILE = DATA_DIR / "futures_trading_state.json"

# --- API 재시도 및 타임아웃 정책 ---
API_READ_TIMEOUT = 30  # 데이터 조회 (OHLCV, 잔고 등)
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
LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB
LOG_BACKUP_COUNT = 3

# --- 대상 심볼 (Futures) ---
# 최적화 및 OOS 검증을 통과한 실거래 대상 Golden 6 심볼
FUTURES_LIVE_SYMBOLS = [
    "AVAX/USDT",
    "DOGE/USDT",
    "ETH/USDT",
    "LINK/USDT",
    "NEAR/USDT",
    "SUI/USDT",
]

# --- 백테스트 초기 잔고 (Futures) ---
FUTURES_INITIAL_BALANCE = 10000.0  # USDT, backtest/optimize/verify 공통

# --- 데이터 기간 설정 (Backtest & Optimization) ---
# Futures 전용 설정
FUTURES_BACKTEST_START_DATE = "2021-01-01"
FUTURES_BACKTEST_END_DATE = "2026-02-17"
