"""범용 클라우드 환경 최적화 유틸리티.

목적:
1. NTP 기반 시간 동기화 검증 (Binance API 호환성).
2. 시스템 리소스 모니터링 (CPU/메모리/디스크).
3. 거래 DB 정리 (디스크 절약).

지원 환경: Windows, Linux (AWS, Azure, GCP, Oracle Cloud 등).
"""

import logging
import socket
import struct
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


class CloudOptimizer:
    """범용 클라우드 환경 최적화."""

    def __init__(self) -> None:
        """최적화 유틸리티 초기화."""
        self.last_time_check = time.time()

    def check_time_sync_ntp(
        self, ntp_server: str = "time.google.com", max_drift_seconds: float = 5.0
    ) -> bool:
        """NTP 서버와 시간 동기화 상태 확인.

        Binance API는 ±1초 이내 타임스탬프 요구.

        Args:
            ntp_server: NTP 서버 주소 (기본: Google Public NTP)
            max_drift_seconds: 허용 오차 (초)

        Returns:
            bool: 동기화 정상 여부.

        """
        try:
            # NTP 프로토콜로 서버 시간 조회
            ntp_time = self._get_ntp_time(ntp_server)
            local_time = time.time()

            drift = abs(ntp_time - local_time)

            if drift > max_drift_seconds:
                logger.warning(
                    f"⏰ Clock drift detected: {drift:.3f}s "
                    f"(threshold: {max_drift_seconds}s, server: {ntp_server})"
                )
                return False

            logger.debug(f"⏰ Time sync OK: drift {drift:.3f}s (server: {ntp_server})")
            return True

        except Exception as e:
            logger.error(f"⏰ Time sync check failed: {e}")
            # 체크 실패는 경고만 하고 True 반환 (과도한 중단 방지)
            return True

    def _get_ntp_time(self, server: str, port: int = 123, timeout: float = 3.0) -> float:
        """NTP 서버로부터 현재 시간(Unix timestamp) 조회.

        NTP 프로토콜 간단 구현 (SNTP).
        """
        # NTP 요청 패킷 생성 (48 bytes)
        # LI=0, VN=3, Mode=3 (Client)
        ntp_packet = b"\x1b" + 47 * b"\0"

        # UDP 소켓 생성 및 타임아웃 설정
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(ntp_packet, (server, port))

            # 응답 수신
            data, _ = sock.recvfrom(1024)

            if len(data) < 48:
                raise ValueError(f"Invalid NTP response from {server}")

            # Transmit Timestamp (bytes 40-47, big-endian)
            # NTP epoch: 1900-01-01, Unix epoch: 1970-01-01
            # Offset: 70년 * 365.25일 * 24시간 * 3600초 = 2208988800초
            ntp_time = struct.unpack("!I", data[40:44])[0]
            unix_time = ntp_time - 2208988800

            return float(unix_time)

    def cleanup_db_old_records(
        self, db_path: Path, table: str = "trades", days_to_keep: int = 90
    ) -> None:
        """거래 기록 DB 정리 (90일 이상 오래된 레코드 삭제).

        디스크 용량 절약.
        """
        import sqlite3
        from datetime import timedelta

        cutoff_date = (datetime.now(UTC) - timedelta(days=days_to_keep)).isoformat()

        allowed_tables = {"spot_signals", "futures_signals", "trades"}
        if table not in allowed_tables:
            logger.error(f"Invalid table name for cleanup: {table}")
            return

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?",  # noqa: S608
                    (cutoff_date,),
                )
                row = cursor.fetchone()
                old_count = row[0] if row else 0

                if old_count > 0:
                    conn.execute(
                        f"DELETE FROM {table} WHERE timestamp < ?",  # noqa: S608
                        (cutoff_date,),
                    )
                    conn.commit()
                    logger.info(f"Cleaned up {old_count} old records from {table}")

                    # VACUUM으로 디스크 공간 회수
                    conn.execute("VACUUM")

                    logger.info(
                        f"🗑️ Cleaned {old_count} old records from {table} "
                        f"(older than {days_to_keep} days)."
                    )
        except Exception as e:
            logger.error(f"❌ DB cleanup failed: {e}")

    def get_resource_usage(self) -> dict[str, float]:
        """시스템 리소스 사용량 조회."""
        # 디스크 경로를 OS에 맞게 자동 감지
        disk_path = "C:/" if psutil.WINDOWS else "/"

        return {
            "cpu_percent": float(psutil.cpu_percent(interval=1)),
            "memory_percent": float(psutil.virtual_memory().percent),
            "disk_percent": float(psutil.disk_usage(disk_path).percent),
            "swap_percent": float(psutil.swap_memory().percent),
        }

    def log_resource_usage(self) -> dict[str, float]:
        """리소스 사용량 로깅 (모니터링용)."""
        usage = self.get_resource_usage()
        logger.info(
            f"📊 Resources: CPU {usage['cpu_percent']:.1f}% | "
            f"Memory {usage['memory_percent']:.1f}% | "
            f"Disk {usage['disk_percent']:.1f}% | "
            f"Swap {usage['swap_percent']:.1f}%"
        )

        # 경고 임계값
        if usage["memory_percent"] > 80:
            logger.warning(f"⚠️ High memory usage: {usage['memory_percent']:.1f}%")

        if usage["disk_percent"] > 85:
            logger.warning(f"⚠️ Disk space low: {usage['disk_percent']:.1f}%")

        if usage["swap_percent"] > 50:
            logger.warning(f"⚠️ High swap usage: {usage['swap_percent']:.1f}% (memory pressure)")

        return usage

    def force_gc(self) -> int:
        """명시적 가비지 컬렉션 (메모리 누수 방지)."""
        import gc

        collected = gc.collect()
        logger.debug(f"🧹 Garbage collection: {collected} objects freed")
        return collected
