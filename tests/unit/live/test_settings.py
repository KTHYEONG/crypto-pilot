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
