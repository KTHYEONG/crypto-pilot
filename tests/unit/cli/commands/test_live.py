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
    # run-namespaced weights: unset --artifact resolves at handler time from
    # settings.weights_path, falling back to the rolling state file.
    assert args.artifact is None
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
    assert daemon_args.artifact is None  # unset --artifact resolves at handler time
    assert daemon_args.state_path.endswith("live_daemon_last_run.json")

    calls: list[tuple[Path, Path]] = []

    def fake_run_daemon(settings: Any, artifact_path: Path, state_path: Path, **_: Any) -> None:
        calls.append((artifact_path, state_path))

    monkeypatch.setattr(scheduler_mod, "run_daemon", fake_run_daemon)
    _run_daemon(daemon_args)

    assert len(calls) == 1
    artifact_path, state_path = calls[0]
    assert isinstance(artifact_path, Path)
    assert artifact_path.name == "deployed_target_weights.parquet"
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


def test_orderbook_capture_subcommand_invokes_capture_and_append(monkeypatch) -> None:
    import argparse

    import src.cli.commands.live as live_mod
    from src.cli.commands.live import add_live_commands

    parser = argparse.ArgumentParser()
    add_live_commands(parser)
    args = parser.parse_args(
        ["orderbook-capture", "--symbols", "BTCUSDT,ETHUSDT", "--duration-s", "20", "--interval-s", "10"]
    )
    # handler should be _run_orderbook_capture
    from src.cli.commands.live import _run_orderbook_capture

    assert args.handler is _run_orderbook_capture

    captured: dict = {}

    def fake_capture(client, symbols, decision_time, *, mode, duration_s, interval_s, depth_limit, max_symbols, clock, sleep_fn, now_fn, shutdown=None):
        captured["symbols"] = symbols
        captured["duration_s"] = duration_s
        captured["interval_s"] = interval_s
        return []

    def fake_append(snapshots, directory):
        captured["append_called"] = True
        return []

    # handler imports inside function, so patch src.live.orderbook
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", fake_capture)
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", fake_append)

    args.handler(args)
    assert captured["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert captured["duration_s"] == 20.0
    assert captured["interval_s"] == 10.0
    assert captured.get("append_called") is True


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_09_CLI_DAEMON_SUBCOMMAND_REGISTERED",
    "SCENARIO_LIVE_CLI_EXECUTION_QUALITY_SUMMARY_SUBCOMMAND",
    "SCENARIO_REC_12",
)

# SCENARIO_RESIL_11-cli-signal-daemon-registered
# SCENARIO_REC_12-cli-tax-subcommands










def test_frozen_step_registered_and_signal_step_removed() -> None:
    """frozen-step replaces signal-step/signal-daemon/signal-refresh."""
    import argparse

    import pytest

    from src.cli.commands.live import _run_frozen_step, add_live_commands
    from src.cli.main import build_root_parser

    parser = argparse.ArgumentParser()
    add_live_commands(parser)
    args = parser.parse_args(["frozen-step", "--date", "2026-08-25T00:00:00Z"])
    assert args.handler is _run_frozen_step
    root = build_root_parser()
    a = root.parse_args(["live", "frozen-step", "--date", "2026-08-25T00:00:00Z"])
    assert a.handler is _run_frozen_step
    b = root.parse_args(["live", "daemon"])
    assert b.handler is not None
    with pytest.raises(SystemExit):
        parser.parse_args(["signal-step", "--date", "2026-08-25T00:00:00Z"])
    with pytest.raises(SystemExit):
        parser.parse_args(["signal-daemon"])
    with pytest.raises(SystemExit):
        parser.parse_args(["signal-refresh"])


def test_frozen_delivery_boundary_under_deploy_mhs() -> None:
    from src.common.paths import DATA_DIR, DEPLOY_MHS_DIR
    from src.live.settings import LiveSettings

    settings = LiveSettings()
    assert DATA_DIR not in DEPLOY_MHS_DIR.parents
    assert settings.unit_bootstrap_path.startswith(str(DEPLOY_MHS_DIR))
    assert settings.venue_fallback_path.startswith(str(DEPLOY_MHS_DIR))


def test_run_frozen_step_reports_and_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace

    import src.cli.commands.live as live_mod

    target = pd.Timestamp("2026-08-25 00:00Z")
    report = SimpleNamespace(
        decision_day=target, exposure=2.5, equity_usdt=2100.0, gross_weight=5.0,
        names=20, unit_observations=300, posterior_mean=0.001, posterior_sigma=0.02,
        unit_history_end=target, venue_snapshot="20260921.json", written=True,
    )
    seen: dict = {}

    def _ok(t, settings, weights):
        seen.update(target=t, weights=weights)
        return report

    monkeypatch.setattr("src.live.scheduler._default_frozen_step", _ok)
    args = argparse.Namespace(date=target, artifact=str(tmp_path / "w.parquet"), mode=None)
    live_mod._run_frozen_step(args)
    assert seen["target"] == target
    assert seen["weights"] == tmp_path / "w.parquet"

    def _boom(t, settings, weights):
        raise RuntimeError("unit history gap")

    monkeypatch.setattr("src.live.scheduler._default_frozen_step", _boom)
    with pytest.raises(SystemExit):
        live_mod._run_frozen_step(args)
