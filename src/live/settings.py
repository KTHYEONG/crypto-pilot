"""Live execution settings. 모든 기본값은 가장 안전한 쪽(SHADOW, 테스트넷 주문)이다."""

from __future__ import annotations

from enum import Enum

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.mhs.params import GROWTH_RISK_ENVELOPES
from src.mhs.types import ExecutionSpec

#: LIVE_MAINNET 승인 문자열. 이 값과 정확히 일치해야만 실계좌 모드가 생성된다.
MAINNET_TRADING_ACK = "I_UNDERSTAND_REAL_MONEY"

_MAX_RECV_WINDOW_MS = 60_000


class ExecutionMode(str, Enum):  # noqa: UP042 - contract pins the (str, Enum) base
    """실행 모드. SHADOW/PAPER는 변이 요청을 전송 계층에서 억제한다.

    PAPER는 SHADOW와 동일한 전송 억제를 유지하되, 억제된 주문을 관측된
    호가로 로컬 체결 시뮬레이션해 체결 경로(chase/IOC/정산)까지 검증한다.
    """

    SHADOW = "shadow"
    PAPER = "paper"
    LIVE_TESTNET = "live_testnet"
    LIVE_MAINNET = "live_mainnet"

    @property
    def suppresses_mutations(self) -> bool:
        return self in (ExecutionMode.SHADOW, ExecutionMode.PAPER)


class LiveSettings(BaseSettings):
    """환경변수(LIVE_*) 또는 .env 로 주입되는 라이브 실행 설정."""

    # env_file 을 직접 지정하지 않는다: pydantic-settings 의 dotenv 소스는 env_prefix 로
    # 비-LIVE_ 키를 걸러내지 않아 공유 .env(BINANCE_API_KEY 등)와 함께 쓰면 extra_forbidden 으로
    # 즉시 크래시한다. docker-compose 의 env_file: .env 가 이미 OS 환경변수로 주입하므로
    # OS 환경변수 소스(정상적으로 prefix 필터링됨)만 신뢰한다.
    model_config = SettingsConfigDict(env_prefix="LIVE_", extra="forbid", populate_by_name=True)

    mode: ExecutionMode = ExecutionMode.SHADOW
    market_data_base_url: str = "https://fapi.binance.com"
    # 빈 값이면 mode에서 유도한다: LIVE_TESTNET만 테스트넷, 나머지(SHADOW/PAPER는 주문
    # 억제, LIVE_MAINNET)는 메인넷. LIVE_ORDER_BASE_URL로 별도 주문 베뉴 오버라이드 가능.
    order_base_url: str = ""
    # 계정/마켓데이터(메인넷) 자격증명. 데이터 수집용 공유 .env 와 이름을 맞추기 위해
    # 접두사 없는 BINANCE_* 도 대체로 인식한다(order_* 미설정 시 이 값이 주문에도 재사용됨).
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LIVE_API_KEY", "BINANCE_API_KEY"),
    )
    api_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LIVE_API_SECRET", "BINANCE_SECRET_KEY", "BINANCE_SECRET"),
    )
    # 주문 전용 베뉴(기본 테스트넷) 자격증명. 미설정 시 api_key/api_secret 로 폴백한다.
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
    execution_quality_dir: str | None = None
    portfolio_state_dir: str | None = None
    # 배포 아티팩트 봉투(AES-256-GCM) 키. base64 인코딩 32바이트. env: LIVE_ARTIFACT_KEY.
    artifact_key: SecretStr | None = None
    # 신호 스테일 상한(시간). 초과 신호는 주문 0건으로 스킵한다. env: LIVE_MAX_SIGNAL_STALENESS_HOURS.
    max_signal_staleness_hours: float = 6.0
    max_weights_staleness_hours: float = 96.0
    daemon_catchup_buffer_minutes: float = 20.0
    daemon_max_attempts_per_day: int = 5
    heartbeat_path: str | None = None
    maker_fee_bps: float = ExecutionSpec().maker_fee_bps
    taker_fee_bps: float = ExecutionSpec().taker_fee_bps
    taker_slippage_bps: float = ExecutionSpec().taker_slippage_bps
    paper_fill_model: str = "immediate_taker"
    orderbook_capture_enabled: bool = True
    orderbook_capture_interval_s: float = 10.0
    orderbook_capture_duration_s: float = 1800.0
    orderbook_capture_depth_limit: int = 20
    orderbook_capture_max_symbols: int = 40
    orderbook_capture_dir: str | None = None
    fills_dir: str | None = None
    microstructure_dir: str | None = None
    tax_ledger_dir: str | None = None
    tax_collection_enabled: bool = True
    alert_webhook_url: str | None = None
    min_universe_symbols: int = 100
    alert_halt_streak: int = 2

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

    @field_validator("paper_fill_model")
    @classmethod
    def _validate_paper_fill_model(cls, value: str) -> str:
        if value not in {"immediate_taker", "peg_chase"}:
            raise ValueError(f"paper_fill_model must be one of immediate_taker, peg_chase, got {value!r}")
        return value

    @field_validator("recv_window_ms")
    @classmethod
    def _bounded_recv_window(cls, value: int) -> int:
        if not 0 < value <= _MAX_RECV_WINDOW_MS:
            raise ValueError(f"recv_window_ms must be in (0, {_MAX_RECV_WINDOW_MS}]")
        return value

    @model_validator(mode="after")
    def _gate_mainnet(self) -> LiveSettings:
        if not self.order_base_url:
            self.order_base_url = (
                "https://testnet.binancefuture.com"
                if self.mode is ExecutionMode.LIVE_TESTNET
                else "https://fapi.binance.com"
            )
        if self.mode is ExecutionMode.LIVE_MAINNET and (
            self.mainnet_trading_ack != MAINNET_TRADING_ACK
        ):
            raise ValueError(
                "mode='live_mainnet' requires mainnet_trading_ack="
                f"'{MAINNET_TRADING_ACK}'"
            )
        return self
