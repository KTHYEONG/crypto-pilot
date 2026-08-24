"""SCENARIO_LIVE_14: CLI에 live 그룹과 shadow-cycle이 등록된다."""

from __future__ import annotations

import pytest
import pandas as pd
from pydantic import SecretStr

from src.cli.main import build_root_parser
from src.live.settings import ExecutionMode, LiveSettings


@pytest.fixture(autouse=True)
def _clean_live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LIVE_MODE",
        "LIVE_MAINNET_TRADING_ACK",
        "LIVE_NOTIONAL_EQUITY_USDT",
        "LIVE_API_KEY",
        "LIVE_API_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)


def test_SCENARIO_LIVE_14_cli_registers_live_group() -> None:
    parser = build_root_parser()
    args = parser.parse_args(
        ["live", "shadow-cycle", "--decision-time", "2026-08-24T00:00:00Z"]
    )
    assert args.group == "live"
    assert args.decision_time == pd.Timestamp("2026-08-24 00:00Z")

    with pytest.raises(SystemExit):
        parser.parse_args(["live", "shadow-cycle"])


def test_live_settings_default_shadow_and_mainnet_ack_gate() -> None:
    assert LiveSettings().mode is ExecutionMode.SHADOW

    with pytest.raises(ValueError, match="mainnet_trading_ack"):
        LiveSettings(mode=ExecutionMode.LIVE_MAINNET)

    acknowledged = LiveSettings(
        mode=ExecutionMode.LIVE_MAINNET,
        mainnet_trading_ack="I_UNDERSTAND_REAL_MONEY",
        api_key=SecretStr("k"),
        api_secret=SecretStr("s"),
    )
    assert acknowledged.mode is ExecutionMode.LIVE_MAINNET

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_14_CLI_REGISTERS_LIVE_GROUP",
)
