"""
Oracle Cloud Free Tier 최적화 유틸리티
======================================
목적:
1. Idle VM 자동 종료 방지 (CPU 사용률 유지)
2. 시간 동기화 검증
3. 디스크/메모리 모니터링
"""

import time
import psutil
import subprocess
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class OracleCloudOptimizer:
    """Oracle Cloud Free Tier 환경 최적화"""
    
    def __init__(self):
        self.cpu_target = 12  # 목표 CPU 사용률 (%)
        self.last_heartbeat = time.time()
    
    def prevent_idle_shutdown(self, duration_seconds: int = 5):
        """
        Idle 판단 방지용 CPU 부하 생성
        - 7일간 평균 10% 미만 시 VM 종료됨
        - 주기적으로 가벼운 CPU 작업 실행
        """
        logger.debug(f"🔥 CPU burn for {duration_seconds}s to prevent idle detection")
        
        end_time = time.time() + duration_seconds
        while time.time() < end_time:
            # 간단한 연산으로 CPU 사용률 증가
            _ = sum(i * i for i in range(10000))
    
    def check_time_sync(self, max_drift_seconds: float = 1.0) -> bool:
        """
        시간 동기화 상태 확인
        Binance API는 ±1초 이내 타임스탬프 요구
        """
        try:
            # chronyd 상태 확인 (Linux)
            result = subprocess.run(
                ['chronyc', 'tracking'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # System time offset 파싱
                for line in result.stdout.split('\n'):
                    if 'System time' in line:
                        # 예: "System time     : 0.000012345 seconds slow of NTP time"
                        parts = line.split(':')
                        if len(parts) > 1:
                            offset_str = parts[1].strip().split()[0]
                            offset = abs(float(offset_str))
                            
                            if offset > max_drift_seconds:
                                logger.warning(
                                    f"⏰ Clock drift detected: {offset:.6f}s "
                                    f"(threshold: {max_drift_seconds}s)"
                                )
                                return False
                            
                            logger.debug(f"⏰ Time sync OK: offset {offset:.6f}s")
                            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            # chronyd 없으면 경고만 (Windows는 다른 메커니즘 사용)
            logger.debug("⏰ chronyd not available, skipping time sync check")
        
        return True
    
    def cleanup_db_old_records(
        self, 
        db_path: Path, 
        table: str = 'trades',
        days_to_keep: int = 90
    ):
        """
        거래 기록 DB 정리 (90일 이상 오래된 레코드 삭제)
        디스크 용량 절약
        """
        import sqlite3
        from datetime import timedelta
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?",
                    (cutoff_date,)
                )
                old_count = cursor.fetchone()[0]
                
                if old_count > 0:
                    conn.execute(
                        f"DELETE FROM {table} WHERE timestamp < ?",
                        (cutoff_date,)
                    )
                    conn.commit()
                    
                    # VACUUM으로 디스크 공간 회수
                    conn.execute("VACUUM")
                    
                    logger.info(
                        f"🗑️ Cleaned {old_count} old records from {table} "
                        f"(older than {days_to_keep} days)"
                    )
        except Exception as e:
            logger.error(f"❌ DB cleanup failed: {e}")
    
    def get_resource_usage(self) -> dict:
        """시스템 리소스 사용량 조회"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'swap_percent': psutil.swap_memory().percent,
        }
    
    def log_resource_usage(self):
        """리소스 사용량 로깅 (모니터링용)"""
        usage = self.get_resource_usage()
        logger.info(
            f"📊 Resources: CPU {usage['cpu_percent']:.1f}% | "
            f"Memory {usage['memory_percent']:.1f}% | "
            f"Disk {usage['disk_percent']:.1f}% | "
            f"Swap {usage['swap_percent']:.1f}%"
        )
        
        # 경고 임계값
        if usage['memory_percent'] > 80:
            logger.warning(f"⚠️ High memory usage: {usage['memory_percent']:.1f}%")
        
        if usage['disk_percent'] > 85:
            logger.warning(f"⚠️ Disk space low: {usage['disk_percent']:.1f}%")
        
        if usage['swap_percent'] > 50:
            logger.warning(f"⚠️ High swap usage: {usage['swap_percent']:.1f}% (memory pressure)")
        
        return usage
    
    def force_gc(self):
        """명시적 가비지 컬렉션 (메모리 누수 방지)"""
        import gc
        collected = gc.collect()
        logger.debug(f"🧹 Garbage collection: {collected} objects freed")
        return collected
