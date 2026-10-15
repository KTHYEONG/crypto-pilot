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
