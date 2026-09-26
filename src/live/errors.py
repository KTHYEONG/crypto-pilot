"""Live trading error hierarchy and the Binance error-code policy registry.

미등록 오류 코드는 항상 FAIL_CLOSED로 해석된다(I-ERROR-REGISTRY). 추측에 의한
재시도는 실계좌 손실로 직결되므로 레지스트리 확장은 문서 확인 후에만 허용한다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.live.executor import ExecutionOutcome


class ErrorAction(str, Enum):  # noqa: UP042 - contract pins the (str, Enum) base
    """정책 액션의 폐쇄 집합."""

    RETRY_BACKOFF = "retry_backoff"
    RETRY_BACKOFF_LONG = "retry_backoff_long"
    RESYNC_CLOCK = "resync_clock"
    RESYNC_THEN_DECIDE = "resync_then_decide"
    BENIGN = "benign"
    BENIGN_REPRICE = "benign_reprice"
    # '이 intent 는 무의미해졌다'(-2022 reduceOnly 거절). 사이클 전체가 아니라 해당 intent 만 종료한다.
    BENIGN_ABORT = "benign_abort"
    INTENT_REJECT = "intent_reject"
    MARGIN_WAIT = "margin_wait"
    RISK_INCREASE_FREEZE = "risk_increase_freeze"
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
    -1013: ErrorAction.INTENT_REJECT,  # "Filter failure: {filter}."
    -2010: ErrorAction.FAIL_CLOSED,
    -2011: ErrorAction.BENIGN,
    -2019: ErrorAction.MARGIN_WAIT,  # "Margin is insufficient."
    -4046: ErrorAction.BENIGN,
    -5022: ErrorAction.BENIGN_REPRICE,
    # ReduceOnly Order is rejected: 포지션이 이미 청산됨 -> 해당 intent 만 무의미.
    -2022: ErrorAction.BENIGN_ABORT,
    # PERCENT_PRICE 한계 초과: 밴드 내 재호가 신호.
    -4131: ErrorAction.BENIGN_REPRICE,
    -4164: ErrorAction.INTENT_REJECT,  # "Order's notional must be no smaller than {0} (unless you choose reduce only)."
    -4140: ErrorAction.INTENT_REJECT,  # "Invalid symbol status for opening position."
    -2027: ErrorAction.INTENT_REJECT,  # "Exceeded the maximum allowable position at current leverage."
    -1111: ErrorAction.INTENT_REJECT,  # "Precision is over the maximum defined for this asset."
    -4014: ErrorAction.INTENT_REJECT,  # "Price not increased by tick size."
    -4023: ErrorAction.INTENT_REJECT,  # "Quantity not increased by step size."
    -4003: ErrorAction.INTENT_REJECT,  # "Quantity less than zero."
    -4400: ErrorAction.RISK_INCREASE_FREEZE,  # "Quantitative rules violation; only reduceOnly orders are allowed."
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
    """라이브 트레이딩 계층의 최상위 예외.

    I-LEDGER-DURABLE 채널: 집행기가 중단 시까지 확인된 부분 결과를
    partial_outcomes 에 붙여 재전파하면 원장이 이를 영속한다.
    """

    partial_outcomes: tuple[ExecutionOutcome, ...] | None = None


class ShadowModeViolation(LiveTradingError):  # noqa: N818 - contract pins the name
    """SHADOW 모드에서 허용되지 않은 변이 요청이 전송 계층에 도달했다."""


class CausalityViolation(LiveTradingError):  # noqa: N818 - contract pins the name
    """결정 시각 T의 주문이 T+1h 이전에 생성되려 했다(look-ahead)."""


class ReconciliationBreach(LiveTradingError):  # noqa: N818 - contract pins the name
    """거래소 스냅샷과 내부 원장이 허용오차를 초과해 불일치한다."""


class RiskGateBreach(LiveTradingError):  # noqa: N818 - contract pins the name
    """사전 리스크 게이트 위반으로 사이클 전체가 HALT 되었다."""


class StaleSignalError(LiveTradingError):
    """신호가 max_staleness 보다 오래되어 주문 0건으로 스킵한다."""


class OrderObsolete(LiveTradingError):  # noqa: N818 - contract pins the name
    """거래소가 해당 intent 의 무의미함(-2022)을 통보했다. 사이클 전체가 아니다."""


class ArtifactSealError(LiveTradingError):
    """아티팩트 봉투(seal)의 키/포맷/무결성 실패. 평문 노출 없이 HALT 한다."""


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


class TransientReadError(LiveTradingError):
    """An idempotent venue read failed transiently and its bounded retry budget is exhausted.

    Raised only for GET requests whose failure class is transport-level (socket error, timeout),
    HTTP 5xx, a non-JSON body on a success status, or a registered transient code (-1000/-1001/
    -1007). It never carries mutation semantics: callers may skip the tick that needed the read
    and retry on the next one, because re-reading cannot change venue state.
    """

    def __init__(
        self, message: str, *, path: str, http_status: int, code: int | None, attempts: int
    ) -> None:
        super().__init__(message)
        self.path = path
        self.http_status = http_status
        self.code = code
        self.attempts = attempts

    def __str__(self) -> str:
        return (
            f"{super().__str__()} (path={self.path} http={self.http_status} "
            f"code={self.code} attempts={self.attempts})"
        )
