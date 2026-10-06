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


def test_SCENARIO_LIVE_40_PREFLIGHT_CLI_EXITS_NONZERO_ON_FAILURE(monkeypatch, tmp_path) -> None:
    """SCENARIO_LIVE_40_PREFLIGHT_CLI_EXITS_NONZERO_ON_FAILURE: ``live
    preflight`` shares the shadow-cycle --artifact default and surfaces a
    failing PreflightReport as a nonzero process exit."""
    import src.live.preflight as preflight_mod

    parser = build_root_parser()
    args = parser.parse_args(["live", "preflight"])
    assert args.artifact is None

    settings = LiveSettings(weights_path=str(tmp_path / "run" / "weights.parquet"))
    monkeypatch.setattr("src.cli.commands.live._settings_with_mode", lambda _args: settings)
    captured = []

    from src.live.preflight import PreflightCheck, PreflightReport

    failing_report = PreflightReport(
        checks=(PreflightCheck(name="artifact_readable", passed=False, detail="boom"),)
    )
    def capture_run_preflight(_settings, artifact):
        captured.append(artifact)
        return failing_report

    monkeypatch.setattr(preflight_mod, "run_preflight", capture_run_preflight)
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1
    assert captured[-1] == tmp_path / "run" / "weights.parquet"

    explicit_args = parser.parse_args(["live", "preflight", "--artifact", "/x/w.parquet"])
    with pytest.raises(SystemExit):
        explicit_args.handler(explicit_args)
    assert captured[-1] == __import__("pathlib").Path("/x/w.parquet")

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

    for gone in ("signal-daemon", "signal-refresh", "deploy-check", "signal-step", "microstructure-summary"):
        with pytest.raises(SystemExit):
            parser.parse_args(["live", gone])
    with pytest.raises(SystemExit):
        parser.parse_args(["live", "shadow-cycle", "--decision-time", "2026-08-25T00:00:00Z", "--dry-run"])
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

    def _fake_dispatch(settings_arg, *, event, detail, decision_time, dedupe_key, now):
        alerts.append((settings_arg, event, detail, decision_time, str(now.tz)))
        return True

    monkeypatch.setattr("src.live.alerting.dispatch_alert", _fake_dispatch)
    monkeypatch.setattr("src.live.alerting.drain_alerts", lambda *a, **k: None)
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

    def _fake_dispatch(settings_arg, *, event, detail, decision_time, dedupe_key, now):
        alerts.append((settings_arg, event))
        return True

    monkeypatch.setattr("src.live.alerting.dispatch_alert", _fake_dispatch)
    monkeypatch.setattr("src.live.alerting.drain_alerts", lambda *a, **k: None)
    caplog.set_level(logging.ERROR, logger="LiveCli")

    with pytest.raises(OSError, match="disk full"):
        module._run_daemon(argparse.Namespace(artifact=str(tmp_path / "w.parquet"), state_path=str(tmp_path / "state.json"), mode=None))

    assert events == ["build", "start", "run", "stop"]
    assert alerts == [(settings, "daemon_crashed")]


def test_cli_ledger_resync_wires_flags_and_exit_codes(monkeypatch) -> None:
    from decimal import Decimal

    import pytest

    import src.live.ledger_resync as resync_mod
    from src.cli.main import build_root_parser
    from src.live.account import PositionBreach
    from src.live.errors import LiveTradingError
    from src.live.ledger_resync import ResyncPlan

    parser = build_root_parser()
    args = parser.parse_args(["live", "ledger-resync", "--apply", "--mode", "live_testnet"])
    assert args.apply is True
    defaults = parser.parse_args(["live", "ledger-resync"])
    assert defaults.apply is False

    calls: list[dict] = []
    plan = ResyncPlan(
        adjustments=(PositionBreach(symbol="AAAUSDT", venue_qty=Decimal("0.5"), ledger_qty=Decimal("0")),),
        backup_path=None,
        applied=True,
    )

    def _ok(settings, *, apply, now):
        calls.append({"mode": settings.mode.value, "apply": apply})
        return plan

    monkeypatch.setattr(resync_mod, "run_ledger_resync", _ok)
    args.handler(args)
    assert calls == [{"mode": "live_testnet", "apply": True}]

    def _fail(*_a, **_k):
        raise LiveTradingError("daemon busy stage=execute; retry when idle")

    monkeypatch.setattr(resync_mod, "run_ledger_resync", _fail)
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1


def test_tax_collect_cli_uses_durable_collect_path_and_audits_issues(monkeypatch, tmp_path) -> None:
    """`live tax-collect` routes through collect_and_persist_live_tax (same path as the daemon) and audits every issue."""
    import json

    import src.live.audit as audit_mod
    import src.live.rest as rest_mod
    import src.live.tax_ledger as tax_mod

    monkeypatch.setenv("LIVE_TAX_LEDGER_DIR", str(tmp_path / "tax"))
    audit_path = tmp_path / "tax_collect_audit.jsonl"
    monkeypatch.setattr(audit_mod, "default_audit_log_path", lambda name, for_date=None: audit_path)
    monkeypatch.setattr(rest_mod, "BinanceFuturesRestClient", lambda *a, **k: object())
    seen: dict = {}

    def _collect(client, symbols, tax_dir, mode, *, now, settings):
        seen.update(tax_dir=tax_dir, symbols=list(symbols), now=now)
        return 3, (tax_mod.TaxCollectionIssue(stream="income", stage="page_cap", detail="budget"),)

    monkeypatch.setattr(tax_mod, "collect_and_persist_live_tax", _collect)
    args = build_root_parser().parse_args(["live", "tax-collect"])

    args.handler(args)

    assert seen["tax_dir"] == tmp_path / "tax"
    assert seen["now"].tzinfo is not None
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    issue = next(e for e in events if e["event"] == "tax_collect_issue")
    assert (issue["stream"], issue["stage"]) == ("income", "page_cap")


def test_tax_summary_resolves_run_ledger_and_source_from_mode(tmp_path, monkeypatch) -> None:
    import argparse
    from decimal import Decimal

    import pandas as pd

    import src.cli.commands.live as live_mod
    from src.live.settings import LiveSettings
    from src.live.tax_ledger import append_tax_records
    from src.live.tax_schema import TaxRecord

    run_id = "taxcli2026test01"
    monkeypatch.setenv("LIVE_RECORD_RUN_ID", run_id)
    monkeypatch.setenv("LIVE_MODE", "paper")
    import src.common.paths as paths_mod

    monkeypatch.setattr(paths_mod, "DATA_DIR", tmp_path)
    import src.live.settings as settings_mod

    monkeypatch.setattr(settings_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr("src.common.paths.ohlcv_path", lambda s, tf: tmp_path / "lake" / f"{s}.parquet")
    settings = LiveSettings()
    ledger_dir = tmp_path / "state" / "runs" / run_id / "tax_ledger"
    recs = [
        TaxRecord(record_id="simulated:TRADE:journal:1", kind="TRADE", event_time=pd.Timestamp("2026-12-20 00:00", tz="UTC"), symbol="BTCUSDT", side="BUY", quantity=Decimal(1), price=Decimal(100), quote_qty=Decimal(100), fee=Decimal(0), fee_asset="USDT", realized_pnl=Decimal(0), income_asset="USDT", is_maker=False, venue_id=1, source="simulated", mode="paper"),
        TaxRecord(record_id="simulated:TRADE:journal:2", kind="TRADE", event_time=pd.Timestamp("2027-01-05 00:00", tz="UTC"), symbol="BTCUSDT", side="SELL", quantity=Decimal(1), price=Decimal(120), quote_qty=Decimal(120), fee=Decimal(0), fee_asset="USDT", realized_pnl=Decimal(0), income_asset="USDT", is_maker=False, venue_id=2, source="simulated", mode="paper"),
    ]
    append_tax_records(recs, ledger_dir)
    args = argparse.Namespace(year=2027, mode=None, ledger_dir=None, source=None, boundary_marks=None, output=None)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        live_mod._run_tax_summary(args)
    out_path = __import__("pathlib").Path(buf.getvalue().strip())
    assert out_path == ledger_dir / "summaries" / "tax_summary_2027_simulated.json"
    assert out_path.exists()
    import json

    summary = json.loads(out_path.read_text())
    assert summary["source"] == "simulated"
    assert summary["totals"]["trading_pnl"] == "20"
    opening = summary["opening_inventory"]["BTCUSDT"]
    assert opening["boundary_mark"] is None
    assert opening["boundary_mark_source"] == "ohlcv_1h_close"
    assert opening["boundary_mark_unavailable_reason"] == "ohlcv_file_missing"


def test_tax_summary_overrides(tmp_path, monkeypatch) -> None:
    import argparse

    import src.cli.commands.live as live_mod
    import src.live.tax_boundary_marks as marks_mod

    seen: dict = {}

    def _fake(year, ledger_dir, *, source, config, coverage=None, boundary_marks=None, derive_boundary_marks=None):
        seen.update(ledger_dir=ledger_dir, source=source, boundary_marks=boundary_marks, derive=derive_boundary_marks)
        return {"reconciliation": {"status": "not_applicable", "issues": []}, "mode": "paper"}

    monkeypatch.setattr("src.live.tax_summary.summarize_tax_year", _fake)
    monkeypatch.setattr("src.live.tax_summary.write_tax_summary", lambda summary, path: seen.__setitem__("written", path))
    args = argparse.Namespace(year=2027, mode=None, ledger_dir=str(tmp_path / "X"), source="venue", boundary_marks=None, output=str(tmp_path / "Y.json"))
    live_mod._run_tax_summary(args)
    from pathlib import Path as _Path

    assert seen["ledger_dir"] == _Path(str(tmp_path / "X"))
    assert seen["source"] == "venue"
    assert seen["boundary_marks"] is None
    assert seen["derive"] is marks_mod.derive_ohlcv_boundary_marks
    assert str(seen["written"]) == str(tmp_path / "Y.json")


def test_operator_marks_disable_auto_derivation(tmp_path, monkeypatch) -> None:
    import argparse
    import json

    import src.cli.commands.live as live_mod

    marks_path = tmp_path / "marks.json"
    marks_path.write_text(json.dumps({"BTCUSDT": "105"}))
    seen: dict = {}

    def _fake(year, ledger_dir, *, source, config, coverage=None, boundary_marks=None, derive_boundary_marks=None):
        seen.update(boundary_marks=boundary_marks, derive=derive_boundary_marks)
        return {"reconciliation": {"status": "not_applicable", "issues": []}, "mode": "paper"}

    monkeypatch.setattr("src.live.tax_summary.summarize_tax_year", _fake)
    monkeypatch.setattr("src.live.tax_summary.write_tax_summary", lambda summary, path: None)
    args = argparse.Namespace(year=2027, mode=None, ledger_dir=str(tmp_path), source="simulated", boundary_marks=str(marks_path), output=str(tmp_path / "o.json"))
    live_mod._run_tax_summary(args)
    from decimal import Decimal as _D

    assert seen["derive"] is None
    assert seen["boundary_marks"]["BTCUSDT"].price == _D("105")
    assert seen["boundary_marks"]["BTCUSDT"].source == "operator_supplied"


def test_tax_summary_fails_closed(tmp_path, monkeypatch) -> None:
    import argparse

    import pytest

    import src.cli.commands.live as live_mod

    ledger_dir = tmp_path / "tax"
    ledger_dir.mkdir()
    (ledger_dir / "tax_ledger_202701.jsonl").write_text("not json\n")
    args = argparse.Namespace(year=2027, mode=None, ledger_dir=str(ledger_dir), source="simulated", boundary_marks=None, output=str(tmp_path / "o.json"))
    with pytest.raises(SystemExit) as exc:
        live_mod._run_tax_summary(args)
    assert exc.value.code == 1
    assert not (tmp_path / "o.json").exists()


def test_tax_summary_non_object_marks_fails_closed(tmp_path, monkeypatch) -> None:
    import argparse

    import pytest

    import src.cli.commands.live as live_mod

    marks_path = tmp_path / "marks.json"
    marks_path.write_text("[1, 2]")
    args = argparse.Namespace(year=2027, mode=None, ledger_dir=str(tmp_path), source="simulated", boundary_marks=str(marks_path), output=str(tmp_path / "o.json"))
    with pytest.raises(SystemExit) as exc:
        live_mod._run_tax_summary(args)
    assert exc.value.code == 1


def test_tax_summary_incomplete_warns(tmp_path, monkeypatch, caplog) -> None:
    import argparse
    import logging

    import src.cli.commands.live as live_mod

    def _fake(year, ledger_dir, *, source, config, coverage=None, boundary_marks=None, derive_boundary_marks=None):
        return {"reconciliation": {"status": "incomplete", "issues": [{"code": "coverage_unknown", "symbol": None, "detail": "x"}]}, "mode": "paper"}

    monkeypatch.setattr("src.live.tax_summary.summarize_tax_year", _fake)
    monkeypatch.setattr("src.live.tax_summary.write_tax_summary", lambda summary, path: None)
    args = argparse.Namespace(year=2027, mode=None, ledger_dir=str(tmp_path), source="venue", boundary_marks=None, output=str(tmp_path / "o.json"))
    with caplog.at_level(logging.WARNING, logger="LiveCli"):
        live_mod._run_tax_summary(args)
    assert "coverage_unknown" in caplog.text
