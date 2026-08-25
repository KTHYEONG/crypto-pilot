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
    assert settings.order_base_url.endswith("testnet.binancefuture.com")
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
