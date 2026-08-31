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
    assert "deployed_target_weights.parquet" in args.artifact

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
    parser = build_root_parser()
    args = parser.parse_args(["live", "signal-step", "--date", "2026-08-25T00:00:00Z"])
    assert args.date == __import__("pandas").Timestamp("2026-08-25T00:00:00Z")
    assert args.handler.__name__ == "_run_signal_step"
    # also check daemon still exists
    args2 = parser.parse_args(["live", "daemon"])
    assert args2.handler.__name__ == "_run_daemon"



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


def test_deploy_check_exits_nonzero_on_missing_bundle(tmp_path) -> None:
    """backtest_cloud_handoff: `live deploy-check` fails closed on a missing bundle."""
    import argparse

    from src.cli.commands.live import _run_deploy_check

    ns = argparse.Namespace(
        bundle=str(tmp_path / "nope.json.enc"), runtime=str(tmp_path / "rt.json")
    )
    with pytest.raises(SystemExit) as ei:
        _run_deploy_check(ns)
    assert ei.value.code != 0


def test_run_shadow_cycle_gates_on_effective_decision_time(monkeypatch, tmp_path) -> None:  # noqa: SIM105,S110
    import contextlib

    import pandas as pd

    import src.live.runner as runner

    seen = {}
    monkeypatch.setattr(runner, "assert_signal_available", lambda eff, now: seen.__setitem__("eff", pd.Timestamp(eff)))
    held = pd.Series({"BTCUSDT": 0.4}, name=pd.Timestamp("2026-08-23", tz="UTC"))
    monkeypatch.setattr(runner, "latest_target_weights", lambda *a, **k: held)

    with contextlib.suppress(Exception):
        runner.run_shadow_cycle(runner.LiveSettings(), pd.Timestamp("2026-08-25", tz="UTC"),
                                tmp_path / "w.parquet", now=pd.Timestamp("2026-08-25 02:00:00", tz="UTC"))
    assert seen["eff"] == pd.Timestamp("2026-08-23", tz="UTC")


def test_live_cli_surface_after_v2() -> None:
    import argparse

    import pytest

    from src.cli.commands.live import add_live_commands

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    add_live_commands(sub.add_parser("live"))

    for gone in ("signal-daemon", "signal-refresh", "deploy-check"):
        with pytest.raises(SystemExit):
            parser.parse_args(["live", gone])
    args = parser.parse_args(["live", "signal-step", "--date", "2026-08-25T00:00:00Z"])
    assert args.handler.__name__ == "_run_signal_step"



def test_run_shadow_cycle_paper_no_credentials() -> None:
    import pandas as pd

    import src.live.runner as runner
    from src.live.settings import ExecutionMode, LiveSettings

    settings = LiveSettings(mode=ExecutionMode.PAPER)
    assert settings.api_key is None
    client = runner._order_client(settings, pd.Timestamp("2026-08-24", tz="UTC"))
    assert isinstance(client, runner.NullOrderClient)
    # 서명 조회는 스텁: 실계좌 GET 없음
    assert client.open_orders() == []
    assert client.sync_server_time() is None
    from src.live.rest import PaperResponse

    assert isinstance(client.new_order({}), PaperResponse)

def test_settings_with_mode_flag_overrides_env(monkeypatch) -> None:
    import argparse

    monkeypatch.setenv("LIVE_MODE", "shadow")
    from src.cli.commands.live import _settings_with_mode
    from src.live.settings import ExecutionMode

    s = _settings_with_mode(argparse.Namespace(mode="paper"))
    assert s.mode is ExecutionMode.PAPER
    s2 = _settings_with_mode(argparse.Namespace(mode=None))
    assert s2.mode is ExecutionMode.SHADOW


# --- auto appended from contract ---
def test_run_status_exit_nonzero_on_halt_heartbeat(tmp_path, monkeypatch) -> None:
    import argparse
    import json
    import pandas as pd
    import pytest
    import src.cli.commands.live as live_mod
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({
        "status": "HALT", "decision_time": "2026-08-31T00:00:00+00:00",
        "consecutive_halts": 3, "attempts": 1,
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
    }))
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb)

    with pytest.raises(SystemExit) as ei:
        live_mod._run_status(argparse.Namespace(mode=None))

    assert ei.value.code == 1


def test_run_status_exit_zero_on_healthy_recent_heartbeat(tmp_path, monkeypatch) -> None:
    import argparse
    import json
    import pandas as pd
    import pytest
    import src.cli.commands.live as live_mod
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({
        "status": "COMPLETE", "decision_time": "2026-08-31T00:00:00+00:00",
        "consecutive_halts": 0, "attempts": 0,
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
    }))
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb)

    with pytest.raises(SystemExit) as ei:
        live_mod._run_status(argparse.Namespace(mode=None))

    assert ei.value.code == 0


def test_signal_step_reconciles_params_digest_change(tmp_path, monkeypatch) -> None:
    import argparse
    import pandas as pd
    import src.cli.commands.live as live_mod

    seen = {}

    def _fake_reconcile(runtime, params, boot):
        seen["called"] = True
        return runtime, "soft_swap"

    def _fake_advance(params, runtime, weights_path, data_root, *, target, artifact_key=None, portfolio_state_dir=None, mode="shadow", **kw):
        seen["mode"] = mode
        return runtime, 1, 1.0

    monkeypatch.setattr("src.mhs.live_runtime.reconcile_runtime_params", _fake_reconcile, raising=False)
    monkeypatch.setattr(live_mod, "advance_to_date", _fake_advance, raising=False)
    monkeypatch.setattr(live_mod, "_settings_with_mode", lambda a: _mk_settings_stub(), raising=False)
    monkeypatch.setattr(
        "src.mhs.live_strategy.load_strategy_params",
        lambda *a, **k: _mk_params_stub(), raising=False,
    )
    monkeypatch.setattr(
        "src.mhs.live_runtime.load_or_bootstrap_runtime",
        lambda *a, **k: _mk_runtime_stub(), raising=False,
    )
    monkeypatch.setattr("src.mhs.live_runtime.save_runtime", lambda *a, **k: None, raising=False)

    live_mod._run_signal_step(argparse.Namespace(date=pd.Timestamp("2026-08-31", tz="UTC"), mode=None))

    assert seen.get("called") is True


def _mk_settings_stub():
    from types import SimpleNamespace
    from src.live.settings import ExecutionMode
    return SimpleNamespace(artifact_key=None, portfolio_state_dir=None, mode=ExecutionMode.PAPER)


def _mk_params_stub():
    from types import SimpleNamespace
    return SimpleNamespace(strategy_digest="x")


def _mk_runtime_stub():
    from types import SimpleNamespace
    import pandas as pd
    return SimpleNamespace(
        params_digest="old", last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
    )


