import json
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TypeVar, cast

# tenacity import check
try:
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
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

from src.core.settings import (
    API_RETRY_ATTEMPTS,
    API_RETRY_WAIT_MAX,
    API_RETRY_WAIT_MIN,
    LOG_BACKUP_COUNT,
    LOG_DIR,
    LOG_MAX_BYTES,
)

# Performance measurement level (Level 10 is standard DEBUG)
PERF = logging.DEBUG

class CategorizedLogger(logging.Logger):
    """Logger subclass that automatically ensures debug messages start with logical tags."""

    def perf(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log performance-related message at DEBUG level with [PERF] tag."""
        if self.isEnabledFor(logging.DEBUG):
            self.debug(f"[PERF] {msg}", *args, **kwargs)

    def data(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log data-related message at DEBUG level with [DATA] tag."""
        if self.isEnabledFor(logging.DEBUG):
            self.debug(f"[DATA] {msg}", *args, **kwargs)

    def opt(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log optimization-related message at DEBUG level with [OPT] tag."""
        if self.isEnabledFor(logging.DEBUG):
            self.debug(f"[OPT] {msg}", *args, **kwargs)

    def strat(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log strategy-related message at DEBUG level with [STRAT] tag."""
        if self.isEnabledFor(logging.DEBUG):
            self.debug(f"[STRAT] {msg}", *args, **kwargs)

    def _log(
        self,
        level: int,
        msg: Any,
        args: Any,
        exc_info: Any = None,
        extra: Any = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        if level == logging.DEBUG and isinstance(msg, str):
            allowed_tags = ["[PERF]", "[DATA]", "[OPT]", "[STRAT]", "[SYS]"]
            msg_str = msg.strip()
            has_tag = any(msg_str.startswith(tag) for tag in allowed_tags)

            if not has_tag:
                msg = f"[SYS] {msg_str}"

        super()._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )

logging.setLoggerClass(CategorizedLogger)

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# Structured JSON Logger
# ============================================================
class FlushingStreamHandler(logging.StreamHandler[Any]):
    """StreamHandler that flushes after every emit (Docker/non-TTY visibility)."""

    def emit(self, record: logging.LogRecord) -> None:
        """로그 기록 후 명시적 플러시."""
        super().emit(record)
        if self.stream and hasattr(self.stream, "flush"):
            self.stream.flush()


class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포맷터 (외부 모니터링 연동용)."""

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 문자열로 변환."""
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_obj["data"] = record.extra_data
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(
    name: str,
    log_prefix: str | None = None,
    write_file: bool = True,
) -> logging.Logger:
    """통합 로거 설정 (동적 로그 파일명).

    Args:
        name: 로거 이름 (예: "RealTraderFutures", "RealTraderSpot")
        log_prefix: 로그 파일 접두사 (None이면 name을 snake_case로 변환하여 사용)
        write_file: True이면 .jsonl 파일 핸들러를 추가함.
                    False이면 콘솔(StreamHandler)만 사용 — 최적화/검증 프로세스 전용.

    Returns:
        logging.Logger: 설정된 로거 객체.

    """
    import re

    logger = logging.getLogger(name)
    if not isinstance(logger, CategorizedLogger):
        logger.__class__ = CategorizedLogger

    # 이미 핸들러가 설정되어 있으면 중복 설정 방지 (레벨 초기화 방지)
    if logger.handlers:
        return logger

    # 환경 변수에서 로그 레벨 로드 (기본값 INFO)
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        # PERF 등 사용자 정의 레벨 처리
        if log_level_str == "PERF":
            log_level = PERF
        elif log_level_str == "TRACE":
            log_level = 5  # TRACE
        else:
            log_level = logging.INFO

    logger.setLevel(log_level)
    logger.propagate = False

    stream_handler = FlushingStreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    if not write_file:
        return logger

    # 로그 디렉토리 생성
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 로그 파일명 자동 생성 (CamelCase -> snake_case)
    if log_prefix is None:
        log_prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    # Human-readable .log (IDE/호스트에서 확인용)
    text_log_file = LOG_DIR / f"{log_prefix}.log"
    text_handler = RotatingFileHandler(
        str(text_log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    text_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(text_handler)

    # JSON 로그 (모니터링 연동용) — 단일 파일만 유지
    json_log_file = LOG_DIR / f"{log_prefix}.jsonl"
    json_handler = RotatingFileHandler(
        str(json_log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)

    return logger


# 내부 로거 (Retry 로직용) — 파일 로그 불필요
_internal_logger = setup_logger("CommonUtils", write_file=False)


# ============================================================
# Retry Decorator (API 재시도)
# ============================================================
def create_retry_decorator() -> Any:
    """Tenacity 기반 재시도 데코레이터 생성 (설정값 사용)."""
    if TENACITY_AVAILABLE:
        return retry(
            stop=stop_after_attempt(API_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=API_RETRY_WAIT_MIN, max=API_RETRY_WAIT_MAX),
            # ConnectionError, TimeoutError와 일반 Exception 포함
            retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
            before_sleep=before_sleep_log(_internal_logger, logging.WARNING),
            reraise=True,
        )
    else:
        # Fallback: 단순 재시도 데코레이터 (tenacity 없을 때)
        def fallback_retry(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                last_error: Exception | None = None
                for attempt in range(API_RETRY_ATTEMPTS):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        wait_time = min(API_RETRY_WAIT_MIN * (2**attempt), API_RETRY_WAIT_MAX)
                        _internal_logger.warning(
                            f"⚠️ Retry {attempt + 1}/{API_RETRY_ATTEMPTS}: {e}. "
                            f"Waiting {wait_time}s..."
                        )
                        time.sleep(wait_time)
                if last_error:
                    raise last_error
                raise RuntimeError("API_RETRY_ATTEMPTS must be > 0")

            return cast(F, wrapper)

        return fallback_retry


# 싱글톤으로 재사용 가능한 retry 데코레이터
api_retry = create_retry_decorator()
