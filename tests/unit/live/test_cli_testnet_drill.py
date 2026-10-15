"""CLI wiring invariants for the testnet-drill command."""

from __future__ import annotations

import argparse
import logging
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.live.testnet_drill.report as report_module
import src.live.testnet_drill.session as session_module
from src.cli.commands.live import add_live_commands
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import ReportPaths
from src.live.testnet_drill.session import SlateReport


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="live")
    add_live_commands(parser)
    return parser.parse_args(argv)


def test_list_groups(caplog: pytest.LogCaptureFixture) -> None:
    args = _parse(["testnet-drill", "--list"])
    with caplog.at_level(logging.INFO):
        args.handler(args)
    assert "group=A" in caplog.text


def test_missing_run_id_exits() -> None:
    args = _parse(["testnet-drill"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 2


def test_unknown_scenario_exits() -> None:
    args = _parse(["testnet-drill", "--run-id", "20261016a", "--scenario", "Z9"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 2


def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_MODE", "live_mainnet")
    monkeypatch.setenv("LIVE_ORDER_API_KEY", "order-key")
    monkeypatch.setenv("LIVE_ORDER_API_SECRET", "order-secret")


def _patch_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[str]) -> None:
    session = SimpleNamespace(workspace=tmp_path / "ws", run_id="20261016a")

    def _open(*args: Any, **kwargs: Any) -> Any:
        calls.append("open")
        return session

    def _prepare(current: Any) -> SlateReport:
        calls.append("prepare")
        return SlateReport(starting_wallet_balance=Decimal("100"), leverage=(), margin_type=(), testnet_reset_suspected=False)

    def _write(results: Any, workspace: Path) -> ReportPaths:
        calls.append("report")
        return ReportPaths(json_path=workspace / "report.json", md_path=workspace / "report.md", evidence_path=workspace / "evidence.jsonl")

    def _teardown(current: Any) -> None:
        calls.append("teardown")

    monkeypatch.setattr(session_module, "open_drill_session", _open)
    monkeypatch.setattr(session_module, "prepare_clean_slate", _prepare)
    monkeypatch.setattr(report_module, "write_report", _write)
    monkeypatch.setattr(session_module, "teardown_session", _teardown)


def test_success_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _hermetic_env(monkeypatch)
    calls: list[str] = []
    _patch_lifecycle(monkeypatch, tmp_path, calls)
    args = _parse(["testnet-drill", "--run-id", "20261016a", "--symbols", "BTCUSDT", "--scenario", "B"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 2
    assert calls == []


def test_resume_skips_prepare_and_no_teardown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _hermetic_env(monkeypatch)
    calls: list[str] = []
    _patch_lifecycle(monkeypatch, tmp_path, calls)
    args = _parse(["testnet-drill", "--run-id", "20261016a", "--resume", "--no-teardown"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 2
    assert calls == []


def test_refused_drill_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    _hermetic_env(monkeypatch)

    def _open(*args: Any, **kwargs: Any) -> Any:
        raise DrillRefused("drill refused: mode")

    monkeypatch.setattr(session_module, "open_drill_session", _open)
    args = _parse(["testnet-drill", "--run-id", "20261016a"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 2


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, results: Any, opened: list[dict[str, Any]]
) -> list[Any]:
    import src.live.settings as settings_module
    import src.live.testnet_drill.guard as guard_module
    import src.live.testnet_drill.registry as registry_module
    from tests.unit.live.testnet_drill._stubs import make_settings

    reports: list[Any] = []
    runs: list[bool] = []
    monkeypatch.setattr(settings_module, "LiveSettings", lambda: make_settings())
    monkeypatch.setattr(guard_module, "drill_scratch_root", lambda: tmp_path)
    monkeypatch.setattr(guard_module, "assert_drill_allowed", lambda settings, workspace: None)
    monkeypatch.setattr(session_module, "open_drill_session", lambda *a, **kw: opened.append(kw) or kw)

    def _run(factory: Any, selection: str, config: Any, *, continue_on_fail: bool = False) -> Any:
        runs.append(continue_on_fail)
        factory()
        factory()
        if isinstance(results, Exception):
            raise results
        return results

    def _write(rows: Any, workspace: Path) -> ReportPaths:
        reports.append(rows)
        return ReportPaths(json_path=workspace / "r.json", md_path=workspace / "r.md", evidence_path=workspace / "e")

    monkeypatch.setattr(registry_module, "run_selected", _run)
    monkeypatch.setattr(report_module, "write_report", _write)
    return reports


def _result(verdict: Any, conservation: bool = True) -> Any:
    from src.live.testnet_drill.report import ScenarioResult

    return ScenarioResult("C1", verdict, expectations=(("ok", True),), conservation_passed=conservation)


def test_run_opens_fresh_numbered_sessions_and_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.live.testnet_drill.report import Verdict

    opened: list[dict[str, Any]] = []
    reports = _patch_run(monkeypatch, tmp_path, (_result(Verdict.PASS),), opened)
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "C1", "--symbols", "BTCUSDT"])
    args.handler(args)
    assert [kw["run_id"] for kw in opened] == ["0001run1", "0002run1"]
    assert [kw["workspace"] for kw in opened] == [tmp_path / "run1" / "scenario_0001", tmp_path / "run1" / "scenario_0002"]
    assert len(reports) == 1


@pytest.mark.parametrize("conservation", [True, False])
def test_failed_or_unconserved_result_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, conservation: bool) -> None:
    from src.live.testnet_drill.report import Verdict

    verdict = Verdict.FAIL if conservation else Verdict.OBSERVED
    _patch_run(monkeypatch, tmp_path, (_result(verdict, conservation),), [])
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "C"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1


def test_runner_failure_exits_1_without_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reports = _patch_run(monkeypatch, tmp_path, DrillRefused("drill refused: x"), [])
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "all"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1
    assert reports == []


def test_existing_workspace_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    opened: list[dict[str, Any]] = []
    _patch_run(monkeypatch, tmp_path, (), opened)
    (tmp_path / "run1").mkdir()
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "A"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 2
    assert opened == []


def test_session_open_refusal_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.live.testnet_drill.guard as guard_module

    _patch_run(monkeypatch, tmp_path, (), [])

    def _refuse(settings: Any, workspace: Path) -> None:
        raise DrillRefused("drill refused: mode")

    monkeypatch.setattr(guard_module, "assert_drill_allowed", _refuse)
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "A"])
    with pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 2


def test_skipped_results_do_not_fail_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.live.testnet_drill.report import Verdict

    _patch_run(monkeypatch, tmp_path, (_result(Verdict.SKIPPED, False), _result(Verdict.PASS)), [])
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "C"])
    args.handler(args)


def test_group_d_selection_and_findings_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.live.testnet_drill.report import Verdict

    _patch_run(monkeypatch, tmp_path, (_result(Verdict.PASS),), [])
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "D", "--symbols", "BTCUSDT", "--findings"])
    args.handler(args)
    assert (tmp_path / "run1" / "findings.md").is_file()


def test_dirty_run_writes_partial_report_and_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> None:
    from src.live.testnet_drill.report import Verdict
    from src.live.testnet_drill.session import DirtySessionError

    dirty = DirtySessionError("DIRTY", residuals={"ETHUSDT": "0.2"})
    dirty.partial_results = (_result(Verdict.FAIL, False),)
    reports = _patch_run(monkeypatch, tmp_path, dirty, [])
    args = _parse(["testnet-drill", "--run-id", "run1", "--scenario", "A", "--continue-on-fail"])
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit) as excinfo:
        args.handler(args)
    assert excinfo.value.code == 1
    assert reports == [dirty.partial_results]
    assert "ETHUSDT" in caplog.text
