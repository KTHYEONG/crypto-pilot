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
    args = parser.parse_args(["live", "frozen-step", "--date", "2026-08-25T00:00:00Z"])
    assert args.date == __import__("pandas").Timestamp("2026-08-25T00:00:00Z")
    assert args.handler.__name__ == "_run_frozen_step"
    with pytest.raises(SystemExit):
        parser.parse_args(["live", "signal-step", "--date", "2026-08-25T00:00:00Z"])
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

    for gone in ("signal-daemon", "signal-refresh", "deploy-check", "signal-step"):
        with pytest.raises(SystemExit):
            parser.parse_args(["live", gone])
    args = parser.parse_args(["live", "frozen-step", "--date", "2026-08-25T00:00:00Z"])
    assert args.handler.__name__ == "_run_frozen_step"


def test_run_shadow_cycle_paper_no_credentials() -> None:
    import pandas as pd

    import src.live.runner as runner
    from src.live.settings import ExecutionMode, LiveSettings

    settings = LiveSettings(mode=ExecutionMode.PAPER)
    assert settings.api_key is None
    client = runner._order_client(settings, pd.Timestamp("2026-08-24", tz="UTC"))
    assert isinstance(client, runner.NullOrderClient)
    assert client.mode is ExecutionMode.PAPER
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


def test_run_status_exit_nonzero_on_state_corrupt_heartbeat(tmp_path, monkeypatch) -> None:
    import argparse
    import json
    import pandas as pd
    import pytest
    import src.cli.commands.live as live_mod
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({
        "status": "STATE_CORRUPT", "stage": "idle", "decision_time": "2026-08-31T00:00:00+00:00",
        "consecutive_halts": 0, "attempts": 0,
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


def test_run_daemon_cli_installs_shutdown_handlers_and_process_log(tmp_path, monkeypatch) -> None:
    import argparse
    import logging
    from logging.handlers import RotatingFileHandler
    import src.cli.commands.live as module
    import src.live.lifecycle as lifecycle
    import src.live.scheduler as sched
    from src.live.lifecycle import ShutdownFlag

    monkeypatch.setattr(module, "_LIVE_LOG_DIR", tmp_path / "live_logs")
    monkeypatch.setattr(module, "_settings_with_mode", lambda _args: LiveSettings())
    installed: list[object] = []
    monkeypatch.setattr(lifecycle, "install_shutdown_handlers", lambda flag, **kwargs: installed.append(flag))
    captured: dict[str, object] = {}

    def fake_run_daemon(settings, weights_path, state_path, **kwargs):
        captured.update(kwargs)
        captured["state_path"] = state_path

    monkeypatch.setattr(sched, "run_daemon", fake_run_daemon)
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        module._run_daemon(argparse.Namespace(artifact=str(tmp_path / "w.parquet"), state_path=str(tmp_path / "state.json"), mode=None))
        added = [h for h in root.handlers if h not in before]
    finally:
        for handler in [h for h in root.handlers if h not in before]:
            root.removeHandler(handler)
            handler.close()

    rotating = [h for h in added if isinstance(h, RotatingFileHandler)]
    assert len(installed) == 1
    assert isinstance(installed[0], ShutdownFlag)
    assert captured["shutdown"] is installed[0]
    assert captured["state_path"] == tmp_path / "state.json"
    assert [h.baseFilename for h in rotating] == [str(tmp_path / "live_logs" / "daemon.log")]
    assert rotating[0].maxBytes == 10 * 1024 * 1024
    assert rotating[0].backupCount == 5


def test_attach_process_log_is_idempotent_and_writes_records(tmp_path, monkeypatch) -> None:
    import logging
    from logging.handlers import RotatingFileHandler
    import src.cli.commands.live as module

    monkeypatch.setattr(module, "_LIVE_LOG_DIR", tmp_path / "live_logs")
    root = logging.getLogger()
    before = list(root.handlers)
    previous_level = root.level
    try:
        root.setLevel(logging.INFO)
        first = module._attach_process_log("daemon.log")
        second = module._attach_process_log("daemon.log")
        logging.getLogger("LiveScheduler").info("[SYS] probe record")
        added = [h for h in root.handlers if h not in before]
        for handler in added:
            handler.flush()
        content = first.read_text(encoding="utf-8")
    finally:
        root.setLevel(previous_level)
        for handler in [h for h in root.handlers if h not in before]:
            root.removeHandler(handler)
            handler.close()

    assert first == second == tmp_path / "live_logs" / "daemon.log"
    assert len([h for h in added if isinstance(h, RotatingFileHandler)]) == 1
    assert "[SYS] probe record" in content


def test_run_status_logs_heartbeat_stage(tmp_path, monkeypatch, caplog) -> None:
    import argparse
    import json
    import logging
    import pandas as pd
    import pytest
    import src.cli.commands.live as live_mod
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({
        "status": "RUNNING", "stage": "execute", "decision_time": "2026-08-31T00:00:00+00:00",
        "consecutive_halts": 0, "attempts": 0, "ts": pd.Timestamp.now(tz="UTC").isoformat(),
    }))
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb)

    with caplog.at_level(logging.INFO, logger="LiveCli"), pytest.raises(SystemExit) as exit_info:
        live_mod._run_status(argparse.Namespace(mode=None))

    assert exit_info.value.code == 0
    assert "stage=execute" in caplog.text


def test_run_daemon_cli_uses_default_frozen_step(tmp_path, monkeypatch) -> None:
    import argparse
    import logging
    import src.cli.commands.live as module
    import src.live.lifecycle as lifecycle
    import src.live.scheduler as sched

    monkeypatch.setattr(module, "_LIVE_LOG_DIR", tmp_path / "live_logs")
    monkeypatch.setattr(module, "_settings_with_mode", lambda _args: object())
    monkeypatch.setattr(lifecycle, "install_shutdown_handlers", lambda flag, **kwargs: None)
    monkeypatch.setattr("src.live.recorder_watch.build_recorder_watchdog", lambda *a, **k: None)
    captured: dict[str, object] = {}
    monkeypatch.setattr(sched, "run_daemon", lambda settings, weights_path, state_path, **kwargs: captured.update(kwargs))
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        module._run_daemon(argparse.Namespace(artifact=str(tmp_path / "w.parquet"), state_path=str(tmp_path / "state.json"), mode=None))
    finally:
        for handler in [h for h in root.handlers if h not in before]:
            root.removeHandler(handler)
            handler.close()

    # 데몬은 frozen 단계를 기본값(프로세스 내 호출)으로 쓰며 CLI가 대체 함수를 주입하지 않는다.
    assert "signal_step_fn" not in captured
    assert captured["shutdown"] is not None


def test_run_status_logs_heartbeat_detail(tmp_path, monkeypatch, caplog) -> None:
    import argparse
    import json
    import logging
    import pandas as pd
    import pytest
    import src.cli.commands.live as live_mod
    import src.live.scheduler as sched

    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({
        "status": "HALT", "stage": "idle", "decision_time": "2026-08-31T00:00:00+00:00",
        "consecutive_halts": 1, "attempts": 2, "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "detail": "signal_step ValueError",
    }))
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: hb)

    with caplog.at_level(logging.INFO, logger="LiveCli"), pytest.raises(SystemExit) as exit_info:
        live_mod._run_status(argparse.Namespace(mode=None))

    assert exit_info.value.code == 1
    assert "detail=signal_step ValueError" in caplog.text


def test_cli_paper_funding_backfill_wires_flags_and_exit_codes(monkeypatch) -> None:
    from decimal import Decimal

    import pandas as pd
    import pytest

    import src.live.funding_backfill as backfill_mod
    from src.cli.main import build_root_parser
    from src.common.errors import DataIntegrityError
    from src.live.funding_backfill import BackfillPlan

    parser = build_root_parser()
    args = parser.parse_args(["live", "paper-funding-backfill", "--apply", "--accrual-start", "2026-09-15T01:05:00Z", "--mode", "paper"])
    assert args.apply is True
    assert args.accrual_start == pd.Timestamp("2026-09-15 01:05Z")
    defaults = parser.parse_args(["live", "paper-funding-backfill"])
    assert defaults.apply is False
    assert defaults.accrual_start is None

    calls: list[dict] = []
    plan = BackfillPlan(
        start=pd.Timestamp("2026-09-02 01:26Z"), end=pd.Timestamp("2026-09-15 01:05Z"),
        cash_delta=Decimal("1.9"), by_symbol={"AAAUSDT": Decimal("1.9")}, epochs=10, seeds_accrual_start=False,
    )

    def _ok(settings, *, apply, now, accrual_start=None, **_kwargs):
        calls.append({"mode": settings.mode.value, "apply": apply, "accrual_start": accrual_start, "tz": str(now.tz)})
        return plan

    monkeypatch.setattr(backfill_mod, "run_paper_funding_backfill", _ok)
    args.handler(args)
    assert calls == [{"mode": "paper", "apply": True, "accrual_start": pd.Timestamp("2026-09-15 01:05Z"), "tz": "UTC"}]

    def _fail(*_a, **_k):
        raise DataIntegrityError("paper funding backfill already applied")

    monkeypatch.setattr(backfill_mod, "run_paper_funding_backfill", _fail)
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1


# --- auto appended from contract: live_alert_gaps ---
def test_run_daemon_cli_alerts_and_reraises_on_crash(tmp_path, monkeypatch, caplog) -> None:
    import argparse
    import logging

    import pytest

    import src.cli.commands.live as module
    import src.live.lifecycle as lifecycle
    import src.live.scheduler as sched

    settings = object()
    monkeypatch.setattr(module, "_settings_with_mode", lambda _args: settings)
    monkeypatch.setattr(lifecycle, "install_shutdown_handlers", lambda flag, **kwargs: None)
    monkeypatch.setattr("src.live.recorder_watch.build_recorder_watchdog", lambda *a, **k: None)

    def _crash(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sched, "run_daemon", _crash)
    alerts: list[tuple[object, str, str, object, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda s, sent, *, event, detail, decision_time, now: alerts.append((s, event, detail, decision_time, str(now.tz))) or True,
    )
    caplog.set_level(logging.ERROR, logger="LiveCli")

    # When
    with pytest.raises(OSError, match="disk full"):
        module._run_daemon(argparse.Namespace(artifact=str(tmp_path / "w.parquet"), state_path=str(tmp_path / "state.json"), mode=None))

    # Then: 알림 1회 + traceback 로그 + 원 예외 재전파(컨테이너 재시작 정책 유지)
    assert alerts == [(settings, "daemon_crashed", "error=OSError: disk full", None, "UTC")]
    assert any(record.exc_info and "daemon crashed" in record.getMessage() for record in caplog.records)




def test_run_daemon_cli_brackets_run_daemon_with_watchdog(tmp_path, monkeypatch, caplog) -> None:
    import argparse
    import logging

    import pytest

    import src.cli.commands.live as module
    import src.live.lifecycle as lifecycle
    import src.live.scheduler as sched

    settings = object()
    monkeypatch.setattr(module, "_settings_with_mode", lambda _args: settings)
    monkeypatch.setattr(lifecycle, "install_shutdown_handlers", lambda flag, **kwargs: None)

    events: list[str] = []

    class _FakeWatchdog:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    def _fake_build(*args, **kwargs):
        events.append("build")
        return _FakeWatchdog()

    monkeypatch.setattr("src.live.recorder_watch.build_recorder_watchdog", _fake_build)

    def _crash(*a, **k):
        events.append("run")
        raise OSError("disk full")

    monkeypatch.setattr(sched, "run_daemon", _crash)
    alerts: list[tuple[object, str]] = []
    monkeypatch.setattr(
        sched, "_daemon_alert",
        lambda s, sent, *, event, detail, decision_time, now: alerts.append((s, event)) or True,
    )
    caplog.set_level(logging.ERROR, logger="LiveCli")

    with pytest.raises(OSError, match="disk full"):
        module._run_daemon(argparse.Namespace(artifact=str(tmp_path / "w.parquet"), state_path=str(tmp_path / "state.json"), mode=None))

    assert events == ["build", "start", "run", "stop"]
    assert alerts == [(settings, "daemon_crashed")]
