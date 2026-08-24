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
