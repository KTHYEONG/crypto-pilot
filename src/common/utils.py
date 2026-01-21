
import os
import sys
import json
import time
import logging
from datetime import datetime
from functools import wraps
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Union

# tenacity import check
try:
    from tenacity import (
        retry, 
        stop_after_attempt, 
        wait_exponential, 
        retry_if_exception_type,
        before_sleep_log
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# Import configuration
try:
    # Add project root to path if needed to find config
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    pass

from config.settings import (
    LOG_DIR,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MIN,
    API_RETRY_WAIT_MAX
)

# ============================================================
# Structured JSON Logger
# ============================================================
class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포맷터 (외부 모니터링 연동용)"""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'extra_data'):
            log_obj["data"] = record.extra_data
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(name: str, log_prefix: str = None) -> logging.Logger:
    """
    통합 로거 설정 (동적 로그 파일명)
    
    Args:
        name: 로거 이름 (예: "RealTraderFutures", "RealTraderSpot")
        log_prefix: 로그 파일 접두사 (None이면 name을 snake_case로 변환하여 사용)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 이미 핸들러가 설정되어 있으면 중복 설정 방지
    if logger.handlers:
        return logger
    
    # 로그 디렉토리 생성
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    
    # 로그 파일명 자동 생성 (CamelCase -> snake_case)
    if log_prefix is None:
        import re
        log_prefix = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    # 1. 일반 텍스트 로그 (Rotating)
    log_file = LOG_DIR / f"{log_prefix}.log"
    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # 2. JSON 로그 (모니터링 연동용)
    json_log_file = LOG_DIR / f"{log_prefix}.jsonl"
    json_handler = RotatingFileHandler(
        str(json_log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    json_handler.setFormatter(JSONFormatter())
    
    # 3. 콘솔 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(json_handler)
    logger.addHandler(stream_handler)
    
    return logger

# 내부 로거 (Retry 로직용)
_internal_logger = setup_logger("CommonUtils")

# ============================================================
# Retry Decorator (API 재시도)
# ============================================================
def create_retry_decorator():
    """tenacity 기반 재시도 데코레이터 생성 (설정값 사용)"""
    if TENACITY_AVAILABLE:
        return retry(
            stop=stop_after_attempt(API_RETRY_ATTEMPTS),
            wait=wait_exponential(
                multiplier=1, 
                min=API_RETRY_WAIT_MIN, 
                max=API_RETRY_WAIT_MAX
            ),
            # ConnectionError, TimeoutError와 일반 Exception 포함
            retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
            before_sleep=before_sleep_log(_internal_logger, logging.WARNING),
            reraise=True
        )
    else:
        # Fallback: 단순 재시도 데코레이터 (tenacity 없을 때)
        def fallback_retry(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(API_RETRY_ATTEMPTS):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        wait_time = min(
                            API_RETRY_WAIT_MIN * (2 ** attempt), 
                            API_RETRY_WAIT_MAX
                        )
                        _internal_logger.warning(
                            f"⚠️ Retry {attempt+1}/{API_RETRY_ATTEMPTS}: {e}. Waiting {wait_time}s..."
                        )
                        time.sleep(wait_time)
                raise last_error
            return wrapper
        return fallback_retry

# 싱글톤으로 재사용 가능한 retry 데코레이터
api_retry = create_retry_decorator()

