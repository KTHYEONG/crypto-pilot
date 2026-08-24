"""Live trading error hierarchy and the Binance error-code policy registry.

미등록 오류 코드는 항상 FAIL_CLOSED로 해석된다(I-ERROR-REGISTRY). 추측에 의한
재시도는 실계좌 손실로 직결되므로 레지스트리 확장은 문서 확인 후에만 허용한다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import Enum
from typing import Any


class ErrorAction(str, Enum):  # noqa: UP042 - contract pins the (str, Enum) base
    """정책 액션의 폐쇄 집합."""

    RETRY_BACKOFF = "retry_backoff"
    RETRY_BACKOFF_LONG = "retry_backoff_long"
    RESYNC_CLOCK = "resync_clock"
    RESYNC_THEN_DECIDE = "resync_then_decide"
    BENIGN = "benign"
    BENIGN_REPRICE = "benign_reprice"
    FAIL_CLOSED = "fail_closed"


# 초기 레지스트리: 각 코드의 의미는 전송 전 Binance 문서/테스트넷 실측으로 재확인한다.
# 확인되지 않은 코드는 등록하지 않는다(미등록 = FAIL_CLOSED가 안전한 기본값).
BINANCE_ERROR_POLICY: Mapping[int, ErrorAction] = {
    -1000: ErrorAction.RETRY_BACKOFF,
    -1001: ErrorAction.RETRY_BACKOFF,
    -1003: ErrorAction.RETRY_BACKOFF_LONG,
    -1007: ErrorAction.RESYNC_THEN_DECIDE,
    -1021: ErrorAction.RESYNC_CLOCK,
    -1022: ErrorAction.FAIL_CLOSED,
    -1013: ErrorAction.FAIL_CLOSED,
    -2010: ErrorAction.FAIL_CLOSED,
    -2011: ErrorAction.BENIGN,
    -2019: ErrorAction.FAIL_CLOSED,
    -4046: ErrorAction.BENIGN,
    -5022: ErrorAction.BENIGN_REPRICE,
}


def resolve_error_action(code: int | None) -> ErrorAction:
    """미등록 코드와 None은 반드시 FAIL_CLOSED를 반환한다."""
    if code is None:
        return ErrorAction.FAIL_CLOSED
    return BINANCE_ERROR_POLICY.get(code, ErrorAction.FAIL_CLOSED)


def payload_digest(payload: Mapping[str, Any] | str | None) -> str:
    """서명/키 유출 방지를 위해 payload 원문 대신 sha256 앞 12자만 보관한다."""
    raw = payload if isinstance(payload, str) else repr(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class LiveTradingError(RuntimeError):
    """라이브 트레이딩 계층의 최상위 예외."""


class ShadowModeViolation(LiveTradingError):  # noqa: N818 - contract pins the name
    """SHADOW 모드에서 허용되지 않은 변이 요청이 전송 계층에 도달했다."""


class CausalityViolation(LiveTradingError):  # noqa: N818 - contract pins the name
    """결정 시각 T의 주문이 T+1h 이전에 생성되려 했다(look-ahead)."""


class ReconciliationBreach(LiveTradingError):  # noqa: N818 - contract pins the name
    """거래소 스냅샷과 내부 원장이 허용오차를 초과해 불일치한다."""


class RiskGateBreach(LiveTradingError):  # noqa: N818 - contract pins the name
    """사전 리스크 게이트 위반으로 사이클 전체가 HALT 되었다."""


class VenueError(LiveTradingError):
    """거래소가 오류 응답을 반환했다. payload 원문은 보관하지 않는다."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None,
        http_status: int,
        path: str,
        payload_digest: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.path = path
        self.payload_digest = payload_digest

    def __str__(self) -> str:
        return (
            f"{super().__str__()} (code={self.code} http={self.http_status} "
            f"path={self.path} payload_sha256_12={self.payload_digest})"
        )
