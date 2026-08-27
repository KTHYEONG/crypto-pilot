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


def test_SCENARIO_LIVE_40_PREFLIGHT_CLI_EXITS_NONZERO_ON_FAILURE(monkeypatch) -> None:
    """SCENARIO_LIVE_40_PREFLIGHT_CLI_EXITS_NONZERO_ON_FAILURE: ``live
    preflight`` shares the shadow-cycle --artifact default and surfaces a
    failing PreflightReport as a nonzero process exit."""
    import src.live.preflight as preflight_mod

    parser = build_root_parser()
    args = parser.parse_args(["live", "preflight"])
    assert args.artifact.endswith("deployed_target_weights.parquet.enc")

    from src.live.preflight import PreflightCheck, PreflightReport

    failing_report = PreflightReport(
        checks=(PreflightCheck(name="artifact_readable", passed=False, detail="boom"),)
    )
    monkeypatch.setattr(preflight_mod, "run_preflight", lambda *a, **k: failing_report)
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1

    passing_report = PreflightReport(
        checks=(PreflightCheck(name="artifact_readable", passed=True, detail="rows=1"),)
    )
    monkeypatch.setattr(preflight_mod, "run_preflight", lambda *a, **k: passing_report)
    args.handler(args)  # must not raise


def test_SCENARIO_SIGNAL_10_CLI_SUBCOMMANDS_AND_EXIT_CODES(monkeypatch) -> None:
    """SCENARIO_SIGNAL_10 (live side): ``live signal-refresh`` shares the
    shadow-cycle --artifact default; a fail-closed DataIntegrityError from
    refresh_signal_row becomes a clean nonzero exit, NOOP/APPENDED don't raise."""
    import src.mhs.signal_refresh as signal_refresh_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.signal_refresh import SignalRefreshReport

    parser = build_root_parser()
    args = parser.parse_args(["live", "signal-refresh"])
    assert args.artifact.endswith("deployed_target_weights.parquet.enc")

    def raise_binding_error(*a, **k):
        raise DataIntegrityError("params_digest mismatch")

    monkeypatch.setattr(signal_refresh_mod, "refresh_signal_row", raise_binding_error)
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1

    noop_report = SignalRefreshReport(
        status="NOOP", reason="already present", decision_time=pd.Timestamp.now(tz="UTC"),
        n_symbols=0, gross_exposure=0.0, exposure_scale=0.0, elapsed_seconds=0.01,
    )
    monkeypatch.setattr(signal_refresh_mod, "refresh_signal_row", lambda *a, **k: noop_report)
    args.handler(args)  # must not raise

    appended_report = SignalRefreshReport(
        status="APPENDED", reason=None, decision_time=pd.Timestamp.now(tz="UTC"),
        n_symbols=2, gross_exposure=0.04, exposure_scale=1.0, elapsed_seconds=0.5,
    )
    monkeypatch.setattr(signal_refresh_mod, "refresh_signal_row", lambda *a, **k: appended_report)
    args.handler(args)  # must not raise


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
    "SCENARIO_LIVE_40_PREFLIGHT_CLI_EXITS_NONZERO_ON_FAILURE",
    "SCENARIO_SIGNAL_10_CLI_SUBCOMMANDS_AND_EXIT_CODES",
)
