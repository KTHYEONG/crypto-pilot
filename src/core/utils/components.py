import json
import logging
import os
import sqlite3
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Import configuration
try:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    pass

from src.core.settings import CANDLE_SYNC_OFFSET_SECONDS

logger = logging.getLogger(__name__)


# ============================================================
# Trade History DB Manager
# ============================================================
class TradeHistoryDB:
    """거래 기록 영속화 매니저."""

    def __init__(self, db_path: Path) -> None:
        """DB 매니저 초기화 및 테이블 생성."""
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """거래 기록 테이블 생성 (WAL 모드 활성화)."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            # WAL 모드 활성화 (동시 읽기/쓰기 성능 향상)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # 성능 최적화

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    entry_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    reason TEXT,
                    params_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
                ON trades(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                ON trades(symbol)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def record_trade(
        self,
        symbol: str,
        side: str,
        action: str,  # 'ENTRY' or 'EXIT'
        quantity: float,
        price: float,
        entry_price: float | None = None,
        pnl: float | None = None,
        pnl_pct: float | None = None,
        reason: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """거래 기록 저장 (동시 접근 대응)."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._lock:
                    conn = self._get_conn()
                    conn.execute(
                        """
                        INSERT INTO trades 
                        (timestamp, symbol, side, action, quantity, price, 
                         entry_price, pnl, pnl_pct, reason, params_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            datetime.now(UTC).isoformat(),
                            symbol,
                            side,
                            action,
                            quantity,
                            price,
                            entry_price,
                            pnl,
                            pnl_pct,
                            reason,
                            json.dumps(params) if params else None,
                        ),
                    )
                    conn.commit()
                logger.info(f"📝 Trade recorded: {action} {side} {quantity} {symbol} @ {price}")
                break  # Success
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 0.5 * (2**attempt)  # Exponential backoff
                        logger.warning(f"⚠️ DB locked, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Failed to record trade after {max_retries} attempts: {e}")
                else:
                    logger.error(f"❌ Failed to record trade: {e}")
                    break
            except Exception as e:
                logger.error(f"❌ Failed to record trade: {e}")
                break

    def get_recent_trades(self, symbol: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """최근 거래 조회."""
        with self._lock:
            conn = self._get_conn()
            conn.row_factory = sqlite3.Row
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]


# ============================================================
# Health Check Manager
# ============================================================
class HealthCheckManager:
    """봇 생존 확인 매니저."""

    def __init__(self, heartbeat_file: Path) -> None:
        """하트비트 매니저 초기화."""
        self.heartbeat_file = heartbeat_file
        self.start_time = datetime.now(UTC)
        self.loop_count = 0
        self.last_error: Exception | None = None

    def update_heartbeat(
        self,
        status: str = "running",
        positions: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """하트비트 파일 업데이트."""
        self.loop_count += 1
        now = datetime.now(UTC)
        heartbeat_data = {
            "status": status,
            "timestamp": now.isoformat(),
            "uptime_seconds": (now - self.start_time).total_seconds(),
            "loop_count": self.loop_count,
            "last_error": str(self.last_error) if self.last_error else None,
            "positions": positions or {},
            "pid": os.getpid(),
        }
        if extra:
            heartbeat_data.update(extra)

        try:
            with open(self.heartbeat_file, "w", encoding="utf-8") as f:
                json.dump(heartbeat_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Failed to update heartbeat: {e}")

    def record_error(self, error: Exception) -> None:
        """에러 기록."""
        self.last_error = error


# ============================================================
# Utility Functions
# ============================================================
def parse_balance(ret: Any) -> float:
    """BinanceClient.fetch_balance() 반환값 파싱.

    다양한 반환 형식 처리 (dict, tuple).
    """
    usdt_free = 0.0

    # Case A: Dictionary (Standard)
    if isinstance(ret, dict):
        if "USDT" in ret:
            val = ret["USDT"]
            usdt_free = val.get("free", 0.0) if isinstance(val, dict) else float(val)
        elif "free" in ret and isinstance(ret["free"], dict):
            usdt_free = ret["free"].get("USDT", 0.0)

    # Case B: Tuple based (Custom implementation)
    elif isinstance(ret, tuple) and len(ret) >= 2:
        free_part = ret[1]
        if isinstance(free_part, dict):
            usdt_free = free_part.get("USDT", 0.0)
        elif isinstance(free_part, (int, float)):
            usdt_free = float(free_part)

    return float(usdt_free)


def calculate_candle_wait_time(timeframe: str) -> int:
    """다음 캔들 마감까지 대기 시간 계산 (초).

    정확한 봉 마감 시점에 로직 실행.
    """
    now = datetime.now(UTC)

    # 타임프레임별 분 단위 변환
    tf_minutes = 60  # default 1h
    if "m" in timeframe:
        tf_minutes = int(timeframe.replace("m", ""))
    elif "h" in timeframe:
        tf_minutes = int(timeframe.replace("h", "")) * 60
    elif "d" in timeframe:
        tf_minutes = int(timeframe.replace("d", "")) * 1440

    # 현재 시간을 분 단위로 변환
    current_minutes = now.hour * 60 + now.minute

    # 다음 봉 마감 시점 계산
    next_candle_minutes = ((current_minutes // tf_minutes) + 1) * tf_minutes

    # 자정 넘어가는 경우 처리
    if next_candle_minutes >= 1440:
        next_candle_minutes = next_candle_minutes % 1440
        next_candle = (now + timedelta(days=1)).replace(
            hour=next_candle_minutes // 60,
            minute=next_candle_minutes % 60,
            second=int(CANDLE_SYNC_OFFSET_SECONDS),
            microsecond=0,
        )
    else:
        next_candle = now.replace(
            hour=next_candle_minutes // 60,
            minute=next_candle_minutes % 60,
            second=int(CANDLE_SYNC_OFFSET_SECONDS),
            microsecond=0,
        )

    wait_seconds = (next_candle - now).total_seconds()

    # 이미 지났으면 다음 주기로
    if wait_seconds < 0:
        wait_seconds += tf_minutes * 60

    return int(wait_seconds)
