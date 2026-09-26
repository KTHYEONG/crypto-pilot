"""Live execution settings. 모든 기본값은 가장 안전한 쪽(SHADOW, 테스트넷 주문)이다."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.paths import APP_ROOT, DATA_DIR
from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2
from src.mhs.params import ACCOUNT_EXPOSURE_MAX, SIGNAL_PANEL_WINDOW_DAYS
from src.mhs.types import ExecutionSpec

#: LIVE_MAINNET 승인 문자열. 이 값과 정확히 일치해야만 실계좌 모드가 생성된다.
MAINNET_TRADING_ACK = "I_UNDERSTAND_REAL_MONEY"

#: Mainnet futures REST venue.
MAINNET_FAPI_URL = "https://fapi.binance.com"

#: Testnet futures REST venue.
TESTNET_FAPI_URL = "https://testnet.binancefuture.com"

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


def _venue_host(url: str) -> str:
    """Return the lowercased network location of a venue URL."""
    return urlsplit(url if "://" in url else f"https://{url}").netloc.lower()


class LiveSettings(BaseSettings):
    """환경변수(LIVE_*) 또는 .env 로 주입되는 라이브 실행 설정."""

    # env_file 을 직접 지정하지 않는다: pydantic-settings 의 dotenv 소스는 env_prefix 로
    # 비-LIVE_ 키를 걸러내지 않아 공유 .env(BINANCE_API_KEY 등)와 함께 쓰면 extra_forbidden 으로
    # 즉시 크래시한다. docker-compose 의 env_file: .env 가 이미 OS 환경변수로 주입하므로
    # OS 환경변수 소스(정상적으로 prefix 필터링됨)만 신뢰한다.
    model_config = SettingsConfigDict(env_prefix="LIVE_", extra="forbid", populate_by_name=True)

    mode: ExecutionMode = ExecutionMode.SHADOW
    # 빈 값이면 mode에서 유도(LIVE_TESTNET만 테스트넷), 필터·호가·주문이 같은 베뉴여야 체결 경로 검증이 유효.
    market_data_base_url: str = ""
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
    # 신호 스테일 상한(시간). 결정일 00:00 라벨 행을 d 23:03에 소비하므로 23시간 이상이어야
    # 한다. 다음날 02:00까지만 유효하다. env: LIVE_MAX_SIGNAL_STALENESS_HOURS.
    max_signal_staleness_hours: float = 26.0
    max_weights_staleness_hours: float = 96.0
    daemon_catchup_buffer_minutes: float = 3.0
    daemon_max_attempts_per_day: int = 5
    heartbeat_path: str | None = None
    maker_fee_bps: float = ExecutionSpec().maker_fee_bps
    taker_fee_bps: float = ExecutionSpec().taker_fee_bps
    taker_slippage_bps: float = ExecutionSpec().taker_slippage_bps
    # 등록 백테스트 집행 방식과 같아야 한다. taker 원장 = taker_parity, maker 원장 =
    # strict_passive. env는 LIVE_EXECUTION_POLICY.
    execution_policy: Literal["taker_parity", "strict_passive"] = "taker_parity"
    passive_timeout_minutes: int = ExecutionSpec().passive_timeout_minutes
    paper_fill_model: str = "immediate_taker"
    orderbook_capture_enabled: bool = True
    orderbook_capture_interval_s: float = 10.0
    orderbook_capture_duration_s: float = 1800.0
    orderbook_capture_depth_limit: int = 20
    orderbook_capture_max_symbols: int = 40
    orderbook_capture_dir: str | None = None
    exec_depth_capture_enabled: bool = True
    exec_depth_stream_url: str = "wss://fstream.binance.com/stream"
    exec_depth_levels: int = 5
    exec_depth_update_ms: int = 500
    exec_depth_post_window_s: float = 1800.0
    exec_depth_max_symbols: int = 60
    exec_depth_flush_interval_s: float = 300.0
    exec_depth_max_session_s: float = 7200.0
    fills_dir: str | None = None
    microstructure_dir: str | None = None
    tax_ledger_dir: str | None = None
    tax_collection_enabled: bool = True
    # 시뮬레이션 기록의 float 수수료 반올림을 흡수하는 허용 오차.
    cash_reconcile_tolerance_usdt: float = 0.01
    tax_income_page_limit: int = 1000
    tax_trades_page_limit: int = 1000
    tax_income_window_days: int = 7
    tax_income_overlap_s: float = 21600.0
    tax_income_retention_days: int = 90
    tax_max_pages_per_cycle: int = 200
    alert_webhook_url: str | None = None
    alert_gmail_user: str | None = None
    alert_gmail_app_password: SecretStr | None = None
    alert_outbox_path: str | None = None
    alert_retry_backoff_s: float = 60.0
    alert_retry_backoff_max_s: float = 1800.0
    alert_outbox_max_age_s: float = 259200.0
    alert_outbox_retention_s: float = 604800.0
    alert_outbox_max_records: int = 1000
    liveness_stage_grace_s: float = 900.0
    liveness_execute_budget_s: float = 3960.0
    liveness_heartbeat_stale_s: float = 900.0
    deadman_ping_url: SecretStr | None = None
    deadman_ping_interval_s: float = 300.0
    deadman_ping_timeout_s: float = 10.0
    min_universe_symbols: int = 100
    alert_halt_streak: int = 2
    alert_daily_digest: bool = True
    recorder_watch_enabled: bool = True
    recorder_watch_interval_s: float = 60.0
    recorder_heartbeat_stale_s: float = 600.0
    # 청산 무음 알림은 스트림의 EVENT_STALL 재연결 타임아웃(600 s)보다 길게 잡는다:
    # 스트림이 한 번 스스로 재연결을 시도해 보고, 그래도 이벤트가 없으면 알림한다.
    recorder_liquidation_silence_s: float = 900.0
    recorder_liquidation_max_failed_connections: int = 5
    recorder_sampler_stale_s: float = 1800.0
    recorder_sampler_max_consecutive_failures: int = 5
    recorder_min_capture_ratio: float = 0.9
    recorder_capture_ratio_min_points: int = 10
    recorder_persist_stale_s: float = 1200.0
    recorder_max_consecutive_flush_failures: int = 3
    recorder_reference_grace_s: float = 3600.0
    recorder_rejected_fraction_alert: float = 0.01
    recorder_rejected_max_consecutive_points: int = 60
    # Frozen strategy digest stamped onto execution-quality observations so
    # forward evidence can be attributed to an immutable strategy version.
    strategy_digest: str | None = None
    data_retention_days: int = SIGNAL_PANEL_WINDOW_DAYS + 30
    orderbook_retention_days: int = 365
    refresh_max_workers: int = 12
    refresh_lookback_days: int = 40
    refresh_deadline_s: float = 900.0
    refresh_max_fail_fraction: float = 0.15
    funding_prefetch_enabled: bool = True
    funding_prefetch_offset_hours: float = 20.25
    refresh_decision_bar_max_missing_fraction: float = 0.05
    venue_rules_warn_age_days: float = 2.0
    venue_rules_max_age_days: float = 7.0
    venue_rules_max_rejected_fraction: float = 0.05
    max_market_data_staleness_hours: float = 30.0
    delisting_announcement_horizon_days: int = 365
    delisting_block_lead_hours: float = 48.0
    delisting_settlement_min_flat_bars: int = 3
    delisting_settlement_price_rtol: float = 1e-9
    delisting_settlement_fee_bps: float = ExecutionSpec().taker_fee_bps
    venue_listing_snapshot_max_age_hours: float = 30.0
    venue_listing_retention_days: int | None = None

    # 리스크 게이트(등록 상한). frozen 노출은 증거금 상한 안에서 베이지안 Kelly가 정한다.
    # 리스크 게이트는 그 위의 안전 레일이다.
    max_gross_leverage: float = ACCOUNT_EXPOSURE_MAX
    leverage_buffer_fraction: float = 0.25
    max_daily_orders: int = 600
    max_daily_turnover_fraction: float = 2.0 * ACCOUNT_EXPOSURE_MAX
    min_free_margin_fraction: float = 0.15
    derisk_mode_enabled: bool = True
    venue_force_close_auto_adopt: bool = False
    venue_force_close_lookback_hours: float = 168.0
    ledger_resync_backup_dir: str | None = None
    reject_cluster_alert_min_symbols: int = 3
    # Frozen live 입력: 봉인된 단위수익률 부트스트랩과 베뉴 규칙 폴백 스냅샷.
    unit_bootstrap_path: str = str(APP_ROOT / "deploy" / "mhs" / "frozen_unit_returns_maker.parquet.enc")
    venue_fallback_path: str = str(APP_ROOT / "deploy" / "mhs" / "venue_rules_20260921.json")
    # paper/실거래 기록 실행 단위. 설정하면 전략 의존 기록이 `data/state/runs/<run_id>/`에 저장된다.
    # 전략이나 집행이 바뀌면 새 id를 쓴다. env는 LIVE_RECORD_RUN_ID.
    record_run_id: str | None = None
    order_journal_path: str | None = None
    weights_path: str | None = None
    journal_recovery_lookback_hours: float = 72.0
    execution_shutdown_cleanup_budget_s: float = 20.0

    @field_validator("venue_force_close_lookback_hours")
    @classmethod
    def _bounded_force_close_lookback(cls, value: float) -> float:
        if not 0 < value <= 168:
            raise ValueError("venue_force_close_lookback_hours must be in (0, 168]")
        return value

    @field_validator("journal_recovery_lookback_hours", "execution_shutdown_cleanup_budget_s")
    @classmethod
    def _positive_recovery_seconds(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("deadman_ping_interval_s", "deadman_ping_timeout_s")
    @classmethod
    def _positive_deadman_seconds(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("alert_outbox_max_records")
    @classmethod
    def _bounded_outbox_records(cls, value: int) -> int:
        if value < 1:
            raise ValueError("alert_outbox_max_records must be >= 1")
        return value

    @field_validator(
        "recorder_watch_interval_s",
        "recorder_heartbeat_stale_s",
        "recorder_liquidation_silence_s",
        "recorder_sampler_stale_s",
        "recorder_persist_stale_s",
        "recorder_reference_grace_s",
    )
    @classmethod
    def _positive_recorder_seconds(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("recorder_liquidation_max_failed_connections")
    @classmethod
    def _positive_recorder_max_failed(cls, value: int) -> int:
        if value < 1:
            raise ValueError("recorder_liquidation_max_failed_connections must be >= 1")
        return value

    @field_validator(
        "recorder_sampler_max_consecutive_failures",
        "recorder_capture_ratio_min_points",
        "recorder_max_consecutive_flush_failures",
        "recorder_rejected_max_consecutive_points",
    )
    @classmethod
    def _positive_recorder_counts(cls, value: int, info: ValidationInfo) -> int:
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return value

    @field_validator("recorder_min_capture_ratio", "recorder_rejected_fraction_alert")
    @classmethod
    def _bounded_recorder_ratio(cls, value: float, info: ValidationInfo) -> float:
        if not 0 < value <= 1:
            raise ValueError(f"{info.field_name} must be in (0, 1]")
        return value

    @field_validator("notional_equity_usdt")
    @classmethod
    def _positive_equity(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("notional_equity_usdt must be > 0")
        return value

    @field_validator("leverage_buffer_fraction")
    @classmethod
    def _bounded_buffer_fraction(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("leverage_buffer_fraction must be in [0.0, 1.0)")
        return value

    @field_validator("reject_cluster_alert_min_symbols")
    @classmethod
    def _bounded_reject_cluster_min_symbols(cls, value: int) -> int:
        if value < 2:
            raise ValueError("reject_cluster_alert_min_symbols must be >= 2")
        return value

    @field_validator("paper_fill_model")
    @classmethod
    def _validate_paper_fill_model(cls, value: str) -> str:
        if value not in {"immediate_taker", "peg_chase"}:
            raise ValueError(f"paper_fill_model must be one of immediate_taker, peg_chase, got {value!r}")
        return value

    @field_validator("exec_depth_levels")
    @classmethod
    def _validate_exec_depth_levels(cls, value: int) -> int:
        if value not in {5, 10, 20}:
            raise ValueError(f"exec_depth_levels must be one of 5, 10, 20, got {value!r}")
        return value

    @field_validator("exec_depth_update_ms")
    @classmethod
    def _validate_exec_depth_update_ms(cls, value: int) -> int:
        if value not in {100, 250, 500}:
            raise ValueError(f"exec_depth_update_ms must be one of 100, 250, 500, got {value!r}")
        return value

    @field_validator("exec_depth_max_symbols")
    @classmethod
    def _validate_exec_depth_max_symbols(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"exec_depth_max_symbols must be >= 1, got {value!r}")
        return value

    @field_validator("exec_depth_flush_interval_s", "exec_depth_max_session_s")
    @classmethod
    def _validate_exec_depth_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"exec depth interval must be > 0, got {value!r}")
        return value

    @field_validator("exec_depth_post_window_s")
    @classmethod
    def _validate_exec_depth_post_window(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"exec_depth_post_window_s must be >= 0, got {value!r}")
        return value

    @field_validator("record_run_id")
    @classmethod
    def _validate_record_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if re.fullmatch(r"[a-z0-9][a-z0-9_]{7,63}", value) is None:
            raise ValueError(f"record_run_id must match ^[a-z0-9][a-z0-9_]{{7,63}}$, got {value!r}")
        return value

    @field_validator("recv_window_ms")
    @classmethod
    def _bounded_recv_window(cls, value: int) -> int:
        if not 0 < value <= _MAX_RECV_WINDOW_MS:
            raise ValueError(f"recv_window_ms must be in (0, {_MAX_RECV_WINDOW_MS}]")
        return value

    @field_validator("cash_reconcile_tolerance_usdt")
    @classmethod
    def _positive_reconcile_tolerance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("cash_reconcile_tolerance_usdt must be > 0")
        return value

    @field_validator("tax_income_page_limit", "tax_trades_page_limit")
    @classmethod
    def _bounded_tax_page_limit(cls, value: int, info: ValidationInfo) -> int:
        if not 1 <= value <= 1000:
            raise ValueError(f"{info.field_name} must be in [1, 1000]")
        return value

    @field_validator("tax_income_window_days", "tax_income_retention_days", "tax_max_pages_per_cycle")
    @classmethod
    def _positive_tax_int(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value

    @field_validator("tax_income_overlap_s")
    @classmethod
    def _positive_tax_overlap(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("tax_income_overlap_s must be > 0")
        return value

    @field_validator("data_retention_days")
    @classmethod
    def _bounded_data_retention(cls, value: int) -> int:
        from src.market_data.retention import MARKET_DATA_MIN_RETENTION_DAYS

        if value < MARKET_DATA_MIN_RETENTION_DAYS:
            raise ValueError(f"data_retention_days must be >= {MARKET_DATA_MIN_RETENTION_DAYS}")
        return value

    @field_validator("refresh_max_workers")
    @classmethod
    def _bounded_refresh_workers(cls, value: int) -> int:
        if not 1 <= value <= 64:
            raise ValueError("refresh_max_workers must be in [1, 64]")
        return value

    @field_validator("funding_prefetch_offset_hours")
    @classmethod
    def _bounded_prefetch_offset(cls, value: float) -> float:
        bound = float(FROZEN_MHS_TOP20_V2.release_hour_utc) - 0.5
        if not 0 < value < bound:
            raise ValueError(f"funding_prefetch_offset_hours must be in (0, {bound})")
        return value

    @field_validator("refresh_decision_bar_max_missing_fraction", "venue_rules_max_rejected_fraction")
    @classmethod
    def _bounded_unit_fraction(cls, value: float, info: ValidationInfo) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} must be in [0.0, 1.0]")
        return value

    @field_validator("venue_rules_warn_age_days", "venue_rules_max_age_days")
    @classmethod
    def _non_negative_venue_age(cls, value: float, info: ValidationInfo) -> float:
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return value

    @field_validator("refresh_max_fail_fraction")
    @classmethod
    def _bounded_refresh_fail_fraction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("refresh_max_fail_fraction must be in [0.0, 1.0]")
        return value

    @field_validator("refresh_lookback_days")
    @classmethod
    def _bounded_refresh_lookback(cls, value: int) -> int:
        if value < 7:
            raise ValueError("refresh_lookback_days must be >= 7")
        return value

    @field_validator("max_market_data_staleness_hours")
    @classmethod
    def _bounded_market_data_staleness(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("max_market_data_staleness_hours must be > 0")
        return value

    @field_validator("delisting_announcement_horizon_days")
    @classmethod
    def _bounded_delisting_horizon(cls, value: int) -> int:
        if value < 1:
            raise ValueError("delisting_announcement_horizon_days must be >= 1")
        return value

    @field_validator("delisting_block_lead_hours", "venue_listing_snapshot_max_age_hours")
    @classmethod
    def _positive_delisting_hours(cls, value: float, info: ValidationInfo) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value

    @field_validator("delisting_settlement_min_flat_bars")
    @classmethod
    def _bounded_settlement_flat_bars(cls, value: int) -> int:
        if value < 1:
            raise ValueError("delisting_settlement_min_flat_bars must be >= 1")
        return value

    @field_validator("delisting_settlement_price_rtol")
    @classmethod
    def _positive_settlement_rtol(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("delisting_settlement_price_rtol must be > 0")
        return value

    @field_validator("delisting_settlement_fee_bps")
    @classmethod
    def _bounded_settlement_fee(cls, value: float) -> float:
        if value < 0:
            raise ValueError("delisting_settlement_fee_bps must be >= 0")
        return value

    @field_validator("venue_listing_retention_days")
    @classmethod
    def _bounded_listing_retention(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("venue_listing_retention_days must be >= 1")
        return value

    @model_validator(mode="after")
    def _default_listing_retention(self) -> LiveSettings:
        """Listing snapshots cover the frozen panel window unless overridden."""
        if self.venue_listing_retention_days is None:
            self.venue_listing_retention_days = self.data_retention_days
        return self

    @model_validator(mode="after")
    def _gate_venue_rule_ages(self) -> LiveSettings:
        """The stale-ladder warning must fire strictly before the sizing halt."""
        if not self.venue_rules_warn_age_days < self.venue_rules_max_age_days:
            raise ValueError("venue_rules_warn_age_days must be < venue_rules_max_age_days")
        return self

    @model_validator(mode="after")
    def _gate_deadman(self) -> LiveSettings:
        """Dead-man ping timeout must stay below the ping interval."""
        if not self.deadman_ping_timeout_s < self.deadman_ping_interval_s:
            raise ValueError("deadman_ping_timeout_s must be < deadman_ping_interval_s")
        return self

    @model_validator(mode="after")
    def _gate_execution_policy(self) -> LiveSettings:
        """strict passive는 패시브 루프 시뮬레이터와 양의 타임아웃을 요구한다."""
        if self.passive_timeout_minutes < 1:
            raise ValueError(
                f"passive_timeout_minutes must be >= 1, got {self.passive_timeout_minutes}"
            )
        if (
            self.mode is ExecutionMode.PAPER
            and self.execution_policy == "strict_passive"
            and self.paper_fill_model == "immediate_taker"
        ):
            raise ValueError(
                "execution_policy='strict_passive' requires a loop fill simulator;"
                " paper_fill_model='immediate_taker' bypasses the passive loop"
            )
        return self

    @model_validator(mode="after")
    def _gate_mainnet(self) -> LiveSettings:
        """Derive empty venues from mode and enforce live venue parity."""
        # 빈 값은 mode에서 유도한다: LIVE_TESTNET만 테스트넷
        venue = TESTNET_FAPI_URL if self.mode is ExecutionMode.LIVE_TESTNET else MAINNET_FAPI_URL
        if not self.market_data_base_url:
            self.market_data_base_url = venue
        if not self.order_base_url:
            self.order_base_url = venue
        if self.mode is ExecutionMode.LIVE_MAINNET and (
            self.mainnet_trading_ack != MAINNET_TRADING_ACK
        ):
            raise ValueError(
                "mode='live_mainnet' requires mainnet_trading_ack="
                f"'{MAINNET_TRADING_ACK}'"
            )
        # 반쪽 주문 자격증명은 키/시크릿 출처 혼선을 막기 위해 거부
        if (self.order_api_key is None) != (self.order_api_secret is None):
            raise ValueError("order_api_key and order_api_secret must be set together")
        # 라이브 모드는 필터·호가 베뉴와 주문 베뉴가 같아야 검증이 유효
        if not self.mode.suppresses_mutations and _venue_host(
            self.market_data_base_url
        ) != _venue_host(self.order_base_url):
            raise ValueError(
                f"venue parity: market_data_base_url host {_venue_host(self.market_data_base_url)}"
                f" must equal order_base_url host {_venue_host(self.order_base_url)}"
                f" in mode {self.mode.value}"
            )
        # 메인넷 키가 테스트넷 호스트로 전송되지 않도록 설정 단계에서 거부
        if (
            self.mode is ExecutionMode.LIVE_TESTNET
            and self.api_key is not None
            and self.order_api_key is None
        ):
            raise ValueError(
                "live_testnet requires order_api_key/order_api_secret;"
                " refusing to send mainnet api_key to the testnet venue"
            )
        return self

    @model_validator(mode="after")
    def _derive_run_paths(self) -> LiveSettings:
        if self.record_run_id is None:
            return self
        run_root = DATA_DIR / "state" / "runs" / self.record_run_id
        if self.ledger_path is None:
            self.ledger_path = str(run_root / "position_ledger.json")
        if self.order_journal_path is None:
            self.order_journal_path = str(run_root / "order_journal.jsonl")
        if self.weights_path is None:
            self.weights_path = str(run_root / "target_weights.parquet")
        if self.fills_dir is None:
            self.fills_dir = str(run_root / "fills")
        if self.execution_quality_dir is None:
            self.execution_quality_dir = str(run_root / "execution_quality")
        if self.portfolio_state_dir is None:
            self.portfolio_state_dir = str(run_root / "portfolio_state")
        if self.microstructure_dir is None:
            self.microstructure_dir = str(run_root / "microstructure")
        if self.tax_ledger_dir is None:
            self.tax_ledger_dir = str(run_root / "tax_ledger")
        return self

    def run_root(self) -> Path | None:
        if self.record_run_id is None:
            return None
        return DATA_DIR / "state" / "runs" / self.record_run_id


def refresh_settings_fields() -> tuple[str, ...]:
    return (
        "refresh_max_workers",
        "refresh_lookback_days",
        "refresh_deadline_s",
        "refresh_max_fail_fraction",
        "funding_prefetch_enabled",
        "funding_prefetch_offset_hours",
        "refresh_decision_bar_max_missing_fraction",
        "max_market_data_staleness_hours",
    )


_refresh_settings_fields_ref = refresh_settings_fields()
