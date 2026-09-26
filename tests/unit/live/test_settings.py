"""LiveSettings 기본값과 mainnet 승인 게이트 검증."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from src.live.settings import MAINNET_TRADING_ACK, ExecutionMode, LiveSettings


@pytest.fixture(autouse=True)
def _clean_live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LIVE_MODE", "LIVE_MAINNET_TRADING_ACK", "LIVE_NOTIONAL_EQUITY_USDT"):
        monkeypatch.delenv(key, raising=False)


def test_defaults_are_fail_safe() -> None:
    settings = LiveSettings()
    assert settings.mode is ExecutionMode.SHADOW
    # SHADOW는 주문을 억제하므로 계좌 조회 GET은 실계좌 베뉴(메인넷)로 간다.
    assert settings.order_base_url == "https://fapi.binance.com"
    assert LiveSettings(mode="live_testnet").order_base_url.endswith("testnet.binancefuture.com")
    assert LiveSettings(order_base_url="https://x").order_base_url == "https://x"
    assert settings.recv_window_ms <= 60_000
    assert settings.notional_equity_usdt > 0


def test_recv_window_and_equity_validators() -> None:
    with pytest.raises(ValueError, match="recv_window_ms"):
        LiveSettings(recv_window_ms=61_000)
    with pytest.raises(ValueError, match="notional_equity_usdt"):
        LiveSettings(notional_equity_usdt=0)


def test_cash_reconcile_tolerance_must_be_positive() -> None:
    assert LiveSettings().cash_reconcile_tolerance_usdt == 0.01
    with pytest.raises(ValueError, match="cash_reconcile_tolerance_usdt"):
        LiveSettings(cash_reconcile_tolerance_usdt=0)


def test_recorder_watch_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recorder watchdog thresholds default to the spec'd cadences."""
    for key in (
        "LIVE_RECORDER_WATCH_ENABLED",
        "LIVE_RECORDER_WATCH_INTERVAL_S",
        "LIVE_RECORDER_HEARTBEAT_STALE_S",
        "LIVE_RECORDER_LIQUIDATION_SILENCE_S",
        "LIVE_RECORDER_SAMPLER_STALE_S",
        "LIVE_RECORDER_CAPTURE_STALE_S",
        "LIVE_RECORDER_NORMALIZER_MAX_LAG_S",
        "LIVE_RECORDER_COMPACTION_MAX_DELAY_S",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = LiveSettings()
    assert settings.recorder_watch_enabled is True
    assert settings.recorder_watch_interval_s == 60.0
    assert settings.recorder_heartbeat_stale_s == 600.0
    assert settings.recorder_liquidation_silence_s == 900.0
    assert settings.recorder_sampler_stale_s == 1800.0
    assert settings.recorder_capture_stale_s == 120.0
    assert settings.recorder_capture_ready_grace_s == 900.0
    assert settings.recorder_capture_dual_active_max_s == 1200.0
    assert settings.recorder_normalizer_max_lag_s == 600.0
    assert settings.recorder_normalizer_max_consecutive_failures == 5
    assert settings.recorder_compaction_max_delay_s == 10800.0


def test_recorder_watch_env_override_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env overrides apply; non-positive cadences fail naming the field."""
    from pydantic import ValidationError

    monkeypatch.setenv("LIVE_RECORDER_LIQUIDATION_SILENCE_S", "1200")
    assert LiveSettings().recorder_liquidation_silence_s == 1200.0
    monkeypatch.setenv("LIVE_RECORDER_WATCH_INTERVAL_S", "0")
    with pytest.raises(ValidationError, match="recorder_watch_interval_s"):
        LiveSettings()
    monkeypatch.delenv("LIVE_RECORDER_WATCH_INTERVAL_S")
    with pytest.raises(ValidationError, match="recorder_normalizer_max_consecutive_failures"):
        LiveSettings(recorder_normalizer_max_consecutive_failures=0)


def test_shared_env_with_non_live_keys_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """docker-compose 의 env_file: .env 는 BINANCE_API_KEY 등 비-LIVE_ 키도 함께 주입한다.

    LiveSettings 가 자체 env_file 을 지정해 dotenv 를 다시 파싱하면 pydantic-settings
    가 prefix 필터 없이 전체 키를 extra=forbid 검증에 넣어 즉시 크래시한다. OS 환경변수
    소스만 신뢰해야 이 시나리오에서 안전하다.
    """
    monkeypatch.setenv("BINANCE_API_KEY", "unrelated")
    monkeypatch.setenv("LIVE_ARTIFACT_KEY", "a" * 44)
    settings = LiveSettings()
    assert settings.artifact_key is not None
    assert settings.artifact_key.get_secret_value() == "a" * 44


def test_mainnet_requires_exact_ack_string() -> None:
    with pytest.raises(ValueError, match="mainnet_trading_ack"):
        LiveSettings(mode=ExecutionMode.LIVE_MAINNET)
    with pytest.raises(ValueError, match="mainnet_trading_ack"):
        LiveSettings(mode=ExecutionMode.LIVE_MAINNET, mainnet_trading_ack="ok")

    settings = LiveSettings(
        mode=ExecutionMode.LIVE_MAINNET,
        mainnet_trading_ack=MAINNET_TRADING_ACK,
        api_key=SecretStr("k"),
        api_secret=SecretStr("s"),
    )
    assert settings.mode is ExecutionMode.LIVE_MAINNET
def test_SCENARIO_LIVE_30_PAPER_MODE_SETTINGS_ARE_SAFE() -> None:
    """SCENARIO_LIVE_30_PAPER_MODE_SETTINGS_ARE_SAFE: mode='paper' constructs
    without a mainnet acknowledgement and without any API credentials; PAPER is
    distinct from LIVE_MAINNET so the suppressed transport path stays safe."""
    settings = LiveSettings(mode="paper")
    assert settings.mode is ExecutionMode.PAPER
    assert ExecutionMode.PAPER != ExecutionMode.LIVE_MAINNET
    assert settings.mainnet_trading_ack is None
    assert settings.api_key is None
    assert settings.api_secret is None


def test_paper_fill_model_default_and_validation() -> None:
    from pydantic import ValidationError

    from src.live.settings import LiveSettings

    assert LiveSettings().paper_fill_model == "immediate_taker"
    assert LiveSettings(paper_fill_model="peg_chase").paper_fill_model == "peg_chase"
    with pytest.raises(ValidationError, match="paper_fill_model"):
        LiveSettings(paper_fill_model="bogus")  # type: ignore[arg-type]


def test_orderbook_capture_settings_defaults() -> None:
    from src.mhs.types import ExecutionSpec

    from src.live.settings import LiveSettings

    s = LiveSettings()
    assert s.orderbook_capture_enabled is True
    assert s.orderbook_capture_interval_s == 10.0
    assert s.orderbook_capture_duration_s == 1800.0
    assert s.orderbook_capture_depth_limit == 20
    assert s.orderbook_capture_max_symbols == 40
    assert s.orderbook_capture_dir is None
    assert s.taker_slippage_bps == ExecutionSpec().taker_slippage_bps


def test_settings_data_retention_below_floor_rejected() -> None:
    import pytest
    from pydantic import ValidationError
    from src.live.settings import LiveSettings
    from src.market_data.retention import MARKET_DATA_MIN_RETENTION_DAYS

    with pytest.raises(ValidationError):
        LiveSettings(data_retention_days=MARKET_DATA_MIN_RETENTION_DAYS - 1)

    ok = LiveSettings(data_retention_days=MARKET_DATA_MIN_RETENTION_DAYS + 5)
    assert ok.data_retention_days == MARKET_DATA_MIN_RETENTION_DAYS + 5


def test_live_settings_refresh_field_defaults_and_bounds(monkeypatch) -> None:
    import pytest
    from pydantic import ValidationError
    from src.live.settings import LiveSettings

    for k in list(__import__("os").environ):
        if k.startswith("LIVE_"):
            monkeypatch.delenv(k, raising=False)

    s = LiveSettings()
    assert s.refresh_max_workers == 12
    assert s.refresh_lookback_days == 40
    assert s.refresh_deadline_s == 900.0
    assert s.refresh_max_fail_fraction == 0.15
    assert s.max_market_data_staleness_hours == 30.0

    with pytest.raises(ValidationError):
        LiveSettings(refresh_max_workers=0)
    with pytest.raises(ValidationError):
        LiveSettings(refresh_max_fail_fraction=1.5)
    with pytest.raises(ValidationError):
        LiveSettings(refresh_lookback_days=3)


def test_live_settings_backtest_parity_defaults() -> None:
    import pytest
    from pydantic import ValidationError
    from src.live.settings import LiveSettings
    from src.market_data.retention import MARKET_DATA_MIN_RETENTION_DAYS
    from src.mhs.params import SIGNAL_PANEL_WINDOW_DAYS

    settings = LiveSettings()
    assert SIGNAL_PANEL_WINDOW_DAYS == 400
    assert settings.data_retention_days == MARKET_DATA_MIN_RETENTION_DAYS == 430
    assert settings.daemon_catchup_buffer_minutes == 3.0
    assert settings.max_daily_turnover_fraction == 2.0 * settings.max_gross_leverage
    with pytest.raises(ValidationError):
        LiveSettings(max_symbol_notional_fraction=0.05)
    with pytest.raises(ValidationError):
        LiveSettings(equity_drawdown_halt=-0.45)


def test_live_settings_leverage_buffer_fraction_default_and_bounds(monkeypatch) -> None:
    import pytest
    from pydantic import ValidationError
    from src.live.settings import LiveSettings

    monkeypatch.delenv("LIVE_LEVERAGE_BUFFER_FRACTION", raising=False)
    assert LiveSettings().leverage_buffer_fraction == 0.25
    assert LiveSettings(leverage_buffer_fraction=0.0).leverage_buffer_fraction == 0.0
    for bad in (-0.01, 1.0):
        with pytest.raises(ValidationError, match="leverage_buffer_fraction"):
            LiveSettings(leverage_buffer_fraction=bad)


def test_live_settings_derive_market_and_order_venue_per_mode() -> None:
    from pydantic import SecretStr

    from src.live.settings import MAINNET_FAPI_URL, MAINNET_TRADING_ACK, TESTNET_FAPI_URL, ExecutionMode, LiveSettings

    assert MAINNET_FAPI_URL == "https://fapi.binance.com"
    assert TESTNET_FAPI_URL == "https://testnet.binancefuture.com"
    for mode in (ExecutionMode.SHADOW, ExecutionMode.PAPER):
        settings = LiveSettings(mode=mode)
        assert (settings.market_data_base_url, settings.order_base_url) == (MAINNET_FAPI_URL, MAINNET_FAPI_URL)
    testnet = LiveSettings(mode=ExecutionMode.LIVE_TESTNET)
    assert (testnet.market_data_base_url, testnet.order_base_url) == (TESTNET_FAPI_URL, TESTNET_FAPI_URL)
    mainnet = LiveSettings(mode=ExecutionMode.LIVE_MAINNET, mainnet_trading_ack=MAINNET_TRADING_ACK, api_key=SecretStr("k"), api_secret=SecretStr("s"))
    assert (mainnet.market_data_base_url, mainnet.order_base_url) == (MAINNET_FAPI_URL, MAINNET_FAPI_URL)
    # 명시 오버라이드는 유지(같은 호스트면 라이브 모드도 허용)
    custom = LiveSettings(mode=ExecutionMode.LIVE_TESTNET, market_data_base_url="https://testnet.binancefuture.com/", order_base_url="https://TESTNET.binancefuture.com")
    assert custom.market_data_base_url == "https://testnet.binancefuture.com/"
    # 억제 모드는 주문 베뉴를 쓰지 않으므로 불일치 허용
    assert LiveSettings(mode=ExecutionMode.PAPER, order_base_url="https://x").order_base_url == "https://x"


def test_live_settings_reject_cross_venue_in_live_modes() -> None:
    import pytest
    from pydantic import SecretStr

    from src.live.settings import MAINNET_FAPI_URL, MAINNET_TRADING_ACK, TESTNET_FAPI_URL, ExecutionMode, LiveSettings

    with pytest.raises(ValueError, match="venue parity"):
        LiveSettings(mode=ExecutionMode.LIVE_TESTNET, market_data_base_url=MAINNET_FAPI_URL)
    with pytest.raises(ValueError, match="venue parity"):
        LiveSettings(mode=ExecutionMode.LIVE_TESTNET, order_base_url=MAINNET_FAPI_URL)
    with pytest.raises(ValueError, match="venue parity"):
        LiveSettings(
            mode=ExecutionMode.LIVE_MAINNET, mainnet_trading_ack=MAINNET_TRADING_ACK,
            api_key=SecretStr("k"), api_secret=SecretStr("s"), market_data_base_url=TESTNET_FAPI_URL,
        )


def test_execution_policy_defaults_to_taker_parity() -> None:
    """Default settings replay the taker parity book with the registered timeout."""
    from src.mhs.types import ExecutionSpec

    settings = LiveSettings()
    assert settings.execution_policy == "taker_parity"
    assert settings.passive_timeout_minutes == ExecutionSpec().passive_timeout_minutes


def test_paper_strict_passive_requires_loop_simulator() -> None:
    """PAPER strict passive with the immediate simulator fails; the loop simulator passes."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="strict_passive"):
        LiveSettings(
            mode=ExecutionMode.PAPER, execution_policy="strict_passive",
            paper_fill_model="immediate_taker",
        )
    ok = LiveSettings(
        mode=ExecutionMode.PAPER, execution_policy="strict_passive",
        paper_fill_model="peg_chase",
    )
    assert ok.execution_policy == "strict_passive"


def test_invalid_execution_policy_rejected() -> None:
    """An execution policy outside the closed set fails closed."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="execution_policy"):
        LiveSettings(execution_policy="maker")  # type: ignore[arg-type]


def test_non_positive_passive_timeout_rejected() -> None:
    """A non-positive passive timeout fails closed."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="passive_timeout_minutes"):
        LiveSettings(passive_timeout_minutes=0)


def test_live_settings_testnet_requires_dedicated_order_credentials() -> None:
    import pytest
    from pydantic import SecretStr

    from src.live.settings import MAINNET_TRADING_ACK, ExecutionMode, LiveSettings

    # Given: 메인넷 키만 있는 LIVE_TESTNET -> 테스트넷으로 메인넷 키를 보내지 않는다
    with pytest.raises(ValueError, match="refusing to send mainnet api_key"):
        LiveSettings(mode=ExecutionMode.LIVE_TESTNET, api_key=SecretStr("main-k"), api_secret=SecretStr("main-s"))
    # 주문 키 반쪽 설정은 모든 모드에서 거부
    with pytest.raises(ValueError, match="must be set together"):
        LiveSettings(mode=ExecutionMode.LIVE_TESTNET, order_api_key=SecretStr("t-k"))
    with pytest.raises(ValueError, match="must be set together"):
        LiveSettings(mode=ExecutionMode.PAPER, order_api_secret=SecretStr("t-s"))

    ok = LiveSettings(
        mode=ExecutionMode.LIVE_TESTNET, api_key=SecretStr("main-k"), api_secret=SecretStr("main-s"),
        order_api_key=SecretStr("t-k"), order_api_secret=SecretStr("t-s"),
    )
    assert ok.order_api_key is not None and ok.order_api_key.get_secret_value() == "t-k"  # noqa: PT018
    # 자격증명 없는 LIVE_TESTNET 은 설정 단계에서 막지 않는다(preflight 가 담당)
    assert LiveSettings(mode=ExecutionMode.LIVE_TESTNET).api_key is None
    # 메인넷은 주문 키 폴백 허용(같은 베뉴)
    LiveSettings(mode=ExecutionMode.LIVE_MAINNET, mainnet_trading_ack=MAINNET_TRADING_ACK, api_key=SecretStr("k"), api_secret=SecretStr("s"))



def test_risk_rails_derive_from_account_exposure_max() -> None:
    from src.live.settings import LiveSettings
    from src.mhs.params import ACCOUNT_EXPOSURE_MAX

    settings = LiveSettings()
    assert settings.max_gross_leverage == ACCOUNT_EXPOSURE_MAX
    assert settings.max_daily_turnover_fraction == 2.0 * ACCOUNT_EXPOSURE_MAX
    assert settings.max_signal_staleness_hours >= 24


def test_record_run_id_derives_strategy_record_paths() -> None:
    from src.common.paths import DATA_DIR
    from src.live.settings import LiveSettings

    run_id = "frozen_top20_v2_bayes_maker_20260922"
    settings = LiveSettings(record_run_id=run_id)
    root = DATA_DIR / "state" / "runs" / run_id
    assert settings.run_root() == root
    assert settings.ledger_path == str(root / "position_ledger.json")
    assert settings.order_journal_path == str(root / "order_journal.jsonl")
    assert settings.weights_path == str(root / "target_weights.parquet")
    assert settings.fills_dir == str(root / "fills")
    assert settings.execution_quality_dir == str(root / "execution_quality")
    assert settings.portfolio_state_dir == str(root / "portfolio_state")
    assert settings.microstructure_dir == str(root / "microstructure")
    assert settings.tax_ledger_dir == str(root / "tax_ledger")


def test_record_run_id_never_overrides_explicit_paths() -> None:
    from src.live.settings import LiveSettings

    settings = LiveSettings(record_run_id="frozen_top20_v2_bayes_maker_20260922", fills_dir="/x")
    assert settings.fills_dir == "/x"
    assert settings.ledger_path is not None
    assert "runs" in settings.ledger_path


def test_no_record_run_id_keeps_legacy_defaults() -> None:
    from src.live.settings import LiveSettings

    settings = LiveSettings()
    assert settings.record_run_id is None
    assert settings.run_root() is None
    assert settings.ledger_path is None
    assert settings.weights_path is None
    assert settings.fills_dir is None
    assert settings.execution_quality_dir is None
    assert settings.portfolio_state_dir is None
    assert settings.microstructure_dir is None
    assert settings.tax_ledger_dir is None


def test_unsafe_record_run_id_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    from src.live.settings import LiveSettings

    for bad in ("../evil", "A_UPPER", "short"):
        with pytest.raises(ValidationError):
            LiveSettings(record_run_id=bad)


def test_exec_depth_validator_rejects_bad_variants() -> None:
    import pytest

    from src.live.settings import LiveSettings

    assert LiveSettings().exec_depth_levels == 5
    assert LiveSettings().exec_depth_update_ms == 500
    with pytest.raises(ValueError, match="exec_depth_levels"):
        LiveSettings(exec_depth_levels=7)
    with pytest.raises(ValueError, match="exec_depth_update_ms"):
        LiveSettings(exec_depth_update_ms=700)
    with pytest.raises(ValueError, match="exec_depth_max_symbols"):
        LiveSettings(exec_depth_max_symbols=0)
    with pytest.raises(ValueError, match="exec_depth_flush_interval_s"):
        LiveSettings(exec_depth_flush_interval_s=0)
    with pytest.raises(ValueError, match="exec_depth_max_session_s"):
        LiveSettings(exec_depth_max_session_s=0)
    with pytest.raises(ValueError, match="exec_depth_post_window_s"):
        LiveSettings(exec_depth_post_window_s=-1)
    assert LiveSettings(exec_depth_post_window_s=0).exec_depth_post_window_s == 0


def test_reject_cluster_alert_min_symbols_default_and_bounds() -> None:
    """Spec 02: cluster alert threshold defaults to 3 and requires >= 2."""
    import pytest

    from src.live.settings import LiveSettings

    assert LiveSettings().reject_cluster_alert_min_symbols == 3
    with pytest.raises(ValueError, match="reject_cluster_alert_min_symbols"):
        LiveSettings(reject_cluster_alert_min_symbols=1)


def test_deadman_ping_validators_require_positive_seconds() -> None:
    """deadman 핑 간격·타임아웃은 양수이며 타임아웃은 간격보다 짧다."""
    import pytest

    from src.live.settings import LiveSettings

    with pytest.raises(ValueError, match="deadman_ping_interval_s"):
        LiveSettings(deadman_ping_interval_s=0)
    with pytest.raises(ValueError, match="deadman_ping_timeout_s"):
        LiveSettings(deadman_ping_interval_s=300.0, deadman_ping_timeout_s=300.0)


def test_alert_outbox_max_records_requires_positive() -> None:
    """아웃박스 용량 상한은 1 이상이다."""
    import pytest

    from src.live.settings import LiveSettings

    with pytest.raises(ValueError, match="alert_outbox_max_records"):
        LiveSettings(alert_outbox_max_records=0)


def test_venue_force_close_lookback_hours_bounded() -> None:
    """Force-close lookback is bounded by venue retention."""
    import pytest

    from src.live.settings import LiveSettings

    assert LiveSettings(venue_force_close_lookback_hours=168.0).venue_force_close_lookback_hours == 168.0
    with pytest.raises(ValueError, match="venue_force_close_lookback_hours"):
        LiveSettings(venue_force_close_lookback_hours=0)
    with pytest.raises(ValueError, match="venue_force_close_lookback_hours"):
        LiveSettings(venue_force_close_lookback_hours=169)


def test_bounded_validators_reject_non_positive() -> None:
    """Shared positive/bounded validators fail closed."""
    import pytest

    from src.live.settings import LiveSettings

    with pytest.raises(ValueError, match="journal_recovery_lookback_hours"):
        LiveSettings(journal_recovery_lookback_hours=0)
    with pytest.raises(ValueError, match="tax_income_page_limit"):
        LiveSettings(tax_income_page_limit=0)
    with pytest.raises(ValueError, match="tax_max_pages_per_cycle"):
        LiveSettings(tax_max_pages_per_cycle=0)
    with pytest.raises(ValueError, match="tax_income_overlap_s"):
        LiveSettings(tax_income_overlap_s=0)


def test_delisting_settings_validators_and_retention_default() -> None:
    """Delisting lifecycle knobs fail closed on nonsense; listing retention tracks data retention."""
    import pytest

    from src.live.settings import LiveSettings

    settings = LiveSettings()
    assert settings.delisting_announcement_horizon_days == 365
    assert settings.delisting_block_lead_hours == 48.0
    assert settings.delisting_settlement_min_flat_bars == 3
    assert settings.delisting_settlement_price_rtol == 1e-9
    assert settings.venue_listing_snapshot_max_age_hours == 30.0
    assert settings.venue_listing_retention_days == settings.data_retention_days
    assert LiveSettings(venue_listing_retention_days=7).venue_listing_retention_days == 7

    with pytest.raises(ValueError, match="delisting_announcement_horizon_days"):
        LiveSettings(delisting_announcement_horizon_days=0)
    with pytest.raises(ValueError, match="delisting_block_lead_hours"):
        LiveSettings(delisting_block_lead_hours=0)
    with pytest.raises(ValueError, match="venue_listing_snapshot_max_age_hours"):
        LiveSettings(venue_listing_snapshot_max_age_hours=0)
    with pytest.raises(ValueError, match="delisting_settlement_min_flat_bars"):
        LiveSettings(delisting_settlement_min_flat_bars=0)
    with pytest.raises(ValueError, match="delisting_settlement_price_rtol"):
        LiveSettings(delisting_settlement_price_rtol=0)
    with pytest.raises(ValueError, match="delisting_settlement_fee_bps"):
        LiveSettings(delisting_settlement_fee_bps=-1)
    with pytest.raises(ValueError, match="venue_listing_retention_days"):
        LiveSettings(venue_listing_retention_days=0)


def test_venue_age_validators_reject_negative_and_inverted() -> None:
    import pytest
    from pydantic import ValidationError

    from src.live.settings import LiveSettings

    with pytest.raises(ValidationError, match="venue_rules_warn_age_days"):
        LiveSettings(venue_rules_warn_age_days=-1.0)
    with pytest.raises(ValidationError, match="venue_rules_warn_age_days"):
        LiveSettings(venue_rules_warn_age_days=7.0, venue_rules_max_age_days=7.0)


def test_funding_prefetch_offset_validator_bounds() -> None:
    import pytest
    from pydantic import ValidationError

    from src.live.settings import LiveSettings

    with pytest.raises(ValidationError):
        LiveSettings(funding_prefetch_offset_hours=0.0)
    with pytest.raises(ValidationError):
        LiveSettings(funding_prefetch_offset_hours=22.5)
    assert LiveSettings(funding_prefetch_offset_hours=20.25).funding_prefetch_offset_hours == 20.25


def test_recorder_health_field_defaults() -> None:
    """Watchdog persistence thresholds default to the spec'd values."""
    from src.live.settings import LiveSettings as _Settings

    settings = _Settings()
    assert settings.recorder_sampler_max_consecutive_failures == 5
    assert settings.recorder_min_capture_ratio == 0.9
    assert settings.recorder_capture_ratio_min_points == 10
    assert settings.recorder_persist_stale_s == 1200.0
    assert settings.recorder_max_consecutive_flush_failures == 3
    assert settings.recorder_reference_grace_s == 3600.0
    assert settings.recorder_rejected_fraction_alert == 0.01
    assert settings.recorder_rejected_max_consecutive_points == 60


def test_recorder_health_field_validation() -> None:
    """Non-positive counts and out-of-range ratios fail naming the field."""
    from pydantic import ValidationError

    from src.live.settings import LiveSettings as _Settings

    with pytest.raises(ValidationError, match="recorder_sampler_max_consecutive_failures"):
        _Settings(recorder_sampler_max_consecutive_failures=0)
    with pytest.raises(ValidationError, match="recorder_capture_ratio_min_points"):
        _Settings(recorder_capture_ratio_min_points=0)
    with pytest.raises(ValidationError, match="recorder_max_consecutive_flush_failures"):
        _Settings(recorder_max_consecutive_flush_failures=0)
    with pytest.raises(ValidationError, match="recorder_rejected_max_consecutive_points"):
        _Settings(recorder_rejected_max_consecutive_points=0)
    with pytest.raises(ValidationError, match="recorder_min_capture_ratio"):
        _Settings(recorder_min_capture_ratio=0.0)
    with pytest.raises(ValidationError, match="recorder_min_capture_ratio"):
        _Settings(recorder_min_capture_ratio=1.5)
    with pytest.raises(ValidationError, match="recorder_rejected_fraction_alert"):
        _Settings(recorder_rejected_fraction_alert=0.0)
    with pytest.raises(ValidationError, match="recorder_persist_stale_s"):
        _Settings(recorder_persist_stale_s=0.0)
    with pytest.raises(ValidationError, match="recorder_reference_grace_s"):
        _Settings(recorder_reference_grace_s=-1.0)
