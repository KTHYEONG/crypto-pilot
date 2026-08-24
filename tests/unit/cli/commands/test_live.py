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


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_09_CLI_DAEMON_SUBCOMMAND_REGISTERED",
)
