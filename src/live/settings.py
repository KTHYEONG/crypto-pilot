"""Live execution settings. 모든 기본값은 가장 안전한 쪽(SHADOW, 테스트넷 주문)이다."""

from __future__ import annotations

from enum import Enum

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.mhs.params import GROWTH_RISK_ENVELOPES

#: LIVE_MAINNET 승인 문자열. 이 값과 정확히 일치해야만 실계좌 모드가 생성된다.
MAINNET_TRADING_ACK = "I_UNDERSTAND_REAL_MONEY"

_MAX_RECV_WINDOW_MS = 60_000


class ExecutionMode(str, Enum):  # noqa: UP042 - contract pins the (str, Enum) base
    """실행 모드. SHADOW는 변이 요청을 전송 계층에서 억제한다."""

    SHADOW = "shadow"
    LIVE_TESTNET = "live_testnet"
    LIVE_MAINNET = "live_mainnet"


class LiveSettings(BaseSettings):
    """환경변수(LIVE_*) 또는 .env 로 주입되는 라이브 실행 설정."""

    # env_file 을 직접 지정하지 않는다: pydantic-settings 의 dotenv 소스는 env_prefix 로
    # 비-LIVE_ 키를 걸러내지 않아 공유 .env(BINANCE_API_KEY 등)와 함께 쓰면 extra_forbidden 으로
    # 즉시 크래시한다. docker-compose 의 env_file: .env 가 이미 OS 환경변수로 주입하므로
    # OS 환경변수 소스(정상적으로 prefix 필터링됨)만 신뢰한다.
    model_config = SettingsConfigDict(env_prefix="LIVE_", extra="forbid")

    mode: ExecutionMode = ExecutionMode.SHADOW
    market_data_base_url: str = "https://fapi.binance.com"
    order_base_url: str = "https://testnet.binancefuture.com"
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None
    order_api_key: SecretStr | None = None
    order_api_secret: SecretStr | None = None
    mainnet_trading_ack: str | None = None
    # 사이징 에쿼티의 절대 상한 캡(I-EQUITY-MTM): E = min(계좌 MTM, 이 값).
    # '목표 노셔널'이 아니다. 값과 검증은 불변이다.
    notional_equity_usdt: float = 2_000.0
    recv_window_ms: int = 5_000
    growth_envelope: str = "growth_extreme"
    # None이면 data/state/live_position_ledger.json(default_ledger_path)을 쓴다.
    # 병렬 실행/테스트 격리를 위해 경로를 오버라이드할 수 있다.
    ledger_path: str | None = None
    # 배포 아티팩트 봉투(AES-256-GCM) 키. base64 인코딩 32바이트. env: LIVE_ARTIFACT_KEY.
    artifact_key: SecretStr | None = None
    # 신호 스테일 상한(시간). 초과 신호는 주문 0건으로 스킵한다. env: LIVE_MAX_SIGNAL_STALENESS_HOURS.
    max_signal_staleness_hours: float = 6.0

    # 리스크 게이트(등록 상한). 레버리지 천장은 리스크 엔벨로프 레지스트리에서 유도한다.
    max_gross_leverage: float = GROWTH_RISK_ENVELOPES["growth_extreme"].leverage_ceiling
    max_symbol_notional_fraction: float = 0.05
    max_daily_orders: int = 600
    max_daily_turnover_fraction: float = 2.0
    equity_drawdown_halt: float = -0.45
    min_free_margin_fraction: float = 0.15

    @field_validator("notional_equity_usdt")
    @classmethod
    def _positive_equity(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("notional_equity_usdt must be > 0")
        return value

    @field_validator("recv_window_ms")
    @classmethod
    def _bounded_recv_window(cls, value: int) -> int:
        if not 0 < value <= _MAX_RECV_WINDOW_MS:
            raise ValueError(f"recv_window_ms must be in (0, {_MAX_RECV_WINDOW_MS}]")
        return value

    @model_validator(mode="after")
    def _gate_mainnet(self) -> LiveSettings:
        if self.mode is ExecutionMode.LIVE_MAINNET and (
            self.mainnet_trading_ack != MAINNET_TRADING_ACK
        ):
            raise ValueError(
                "mode='live_mainnet' requires mainnet_trading_ack="
                f"'{MAINNET_TRADING_ACK}'"
            )
        return self
