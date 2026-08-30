# ruff: noqa
"""src/cli/commands/live.py 등록 검증 (mirrored unit test)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.scheduler as scheduler_mod
from src.cli.commands.live import _run_daemon, add_live_commands


def _live_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_live_commands(parser)
    return parser


def test_shadow_cycle_parses_utc_decision_time() -> None:
    args = _live_parser().parse_args(
        ["shadow-cycle", "--decision-time", "2026-08-24T00:00:00Z"]
    )
    assert args.decision_time == pd.Timestamp("2026-08-24 00:00Z")
    assert args.artifact.endswith("deployed_target_weights.parquet.enc")
    assert args.dry_run is False


def test_decision_time_is_required() -> None:
    with pytest.raises(SystemExit):
        _live_parser().parse_args(["shadow-cycle"])


def test_SCENARIO_LIVE_DAEMON_09_cli_daemon_subcommand_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _live_parser()
    # daemon/shadow-cycle 서브커맨드가 모두 등록되어 있다.
    shadow_args = parser.parse_args(
        ["shadow-cycle", "--decision-time", "2026-08-24T00:00:00Z"]
    )
    assert shadow_args.decision_time == pd.Timestamp("2026-08-24 00:00Z")
    daemon_args = parser.parse_args(["daemon"])  # --decision-time 없이 파싱 성공
    assert daemon_args.artifact.endswith("deployed_target_weights.parquet.enc")
    assert daemon_args.state_path.endswith("live_daemon_last_run.json")

    calls: list[tuple[Path, Path]] = []

    def fake_run_daemon(settings: Any, artifact_path: Path, state_path: Path, **_: Any) -> None:
        calls.append((artifact_path, state_path))

    monkeypatch.setattr(scheduler_mod, "run_daemon", fake_run_daemon)
    _run_daemon(daemon_args)

    assert len(calls) == 1
    artifact_path, state_path = calls[0]
    assert isinstance(artifact_path, Path)
    assert isinstance(state_path, Path)


def test_SCENARIO_LIVE_CLI_EXECUTION_QUALITY_SUMMARY_SUBCOMMAND(monkeypatch) -> None:
    parser = _live_parser()
    args = parser.parse_args(["live", "execution-quality-summary"]) if False else parser.parse_args(["execution-quality-summary"])
    assert args.live_command == "execution-quality-summary"
    # handler calls summarize_execution_quality exactly once without extra args
    calls: list[int] = []

    import src.cli.commands.live as live_mod

    original = getattr(live_mod, "_run_execution_quality_summary", None)

    def fake_summarize(*_a, **_k):
        calls.append(1)
        return {"n_cycles": 0}

    monkeypatch.setattr("src.live.execution_quality.summarize_execution_quality", fake_summarize)
    # also need live_mod import path for handler's internal import; patch that module too
    monkeypatch.setattr("src.live.execution_quality.summarize_execution_quality", fake_summarize)

    # invoke handler directly
    handler = args.handler
    handler(args)
    assert len(calls) == 1


def test_SCENARIO_REC_12_cli_tax_subcommands(monkeypatch) -> None:
    from src.cli.commands.live import _run_tax_summary, add_live_commands
    from src.cli.main import build_root_parser
    import argparse

    parser = argparse.ArgumentParser()
    add_live_commands(parser)
    args = parser.parse_args(["tax-summary", "--year", "2027"])
    assert args.year == 2027
    assert args.handler is _run_tax_summary
    # --year missing should exit
    with pytest.raises(SystemExit):
        parser.parse_args(["tax-summary"])
    # summary failure handling
    monkeypatch.setattr("src.live.tax_ledger.summarize_tax_year", lambda *a, **k: (_ for _ in ()).throw(__import__("src.common.errors", fromlist=["DataIntegrityError"]).DataIntegrityError("mixed")))
    with pytest.raises(SystemExit) as exc:
        _run_tax_summary(args)
    assert exc.value.code == 1
    # root parser
    root = build_root_parser()
    a = root.parse_args(["live", "tax-summary", "--year", "2027"])
    assert a.handler is _run_tax_summary


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_09_CLI_DAEMON_SUBCOMMAND_REGISTERED",
    "SCENARIO_LIVE_CLI_EXECUTION_QUALITY_SUMMARY_SUBCOMMAND",
    "SCENARIO_REC_12",
)

# SCENARIO_RESIL_11-cli-signal-daemon-registered
def test_SCENARIO_RESIL_11_cli_signal_daemon_registered():  # noqa: D103
    """SCENARIO_RESIL_11-cli-signal-daemon-registered"""
    import argparse

    from src.cli.commands.live import _run_signal_daemon, add_live_commands
    from src.cli.main import build_root_parser

    parser = argparse.ArgumentParser()
    add_live_commands(parser)
    args = parser.parse_args(["signal-daemon"])
    assert args.handler is _run_signal_daemon
    root = build_root_parser()
    a = root.parse_args(["live", "signal-daemon"])
    assert a.handler is _run_signal_daemon
    b = root.parse_args(["live", "daemon"])
    assert b.handler is not None
    c = root.parse_args(["live", "signal-refresh"])
    assert c.handler is not None
# SCENARIO_REC_12-cli-tax-subcommands
