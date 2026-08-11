"""Contract coverage for the MHS CLI argument surface (MHS-MEM-03 wiring)."""

from __future__ import annotations

import argparse
import types

from src.cli.commands.research.mhs import _run_mhs_horizon_diagnostic, add_mhs_commands


def _fake_report() -> types.SimpleNamespace:
    return types.SimpleNamespace(status="COMPLETE", books=[], blend=None)


def test_mhs_diagnostic_defaults_and_mark_mode_choices() -> None:
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["mark_mode"] == "cache_required"
    assert defaults["execution_timeframe"] == "5m"
    assert defaults["max_rss_bytes"] is None
    assert defaults["no_log_run"] is False
    assert defaults["touch_diagnostic"] is False
    assert defaults["ladder_diagnostic"] is False
    assert defaults["discovery_gate"] is False
    assert defaults["fold_safe_horizon"] is False
    args = parser.parse_args([])
    assert args.max_rss_bytes is None


def test_mhs_diagnostic_max_rss_bytes_flag_wired_to_request() -> None:
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    args = parser.parse_args(["--max-rss-bytes", "8000000000"])
    assert args.max_rss_bytes == 8_000_000_000
    args2 = parser.parse_args(["--mark-mode", "cache_required_stale_carry"])
    assert args2.mark_mode == "cache_required_stale_carry"


def test_mhs_diagnostic_output_tier_flag_threaded_to_persist(monkeypatch) -> None:
    """``--output-tier full`` is parsed and threaded into the persist call;
    the default stays ``compact``."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["output_tier"] == "compact"

    monkeypatch.setattr(ev, "run_mhs_horizon_diagnostic", lambda request: _fake_report())

    def _spy_persist(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", _spy_persist)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: "docs/results/mhs.json")

    args = parser.parse_args(["--output-tier", "full"])
    assert args.output_tier == "full"
    _run_mhs_horizon_diagnostic(args)
    assert captured["tier"].value == "full"

    captured.clear()
    args = parser.parse_args([])
    assert args.output_tier == "compact"
    _run_mhs_horizon_diagnostic(args)
    assert captured["tier"].value == "compact"


def test_mhs_diagnostic_touch_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_TOUCH_CLI_FLAG: ``--touch-diagnostic`` is parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; omitting it
    defaults to False."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(ev, "run_mhs_horizon_diagnostic", lambda request: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    args = parser.parse_args(["--touch-diagnostic"])
    assert args.touch_diagnostic is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["touch_diagnostic"] is True

    captured.clear()
    args = parser.parse_args([])
    assert args.touch_diagnostic is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["touch_diagnostic"] is False


def test_mhs_diagnostic_fold_safe_horizon_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_FOLD_SAFE_HORIZON_08_CLI_FLAG_THREADS_THROUGH:
    ``--fold-safe-horizon`` is parsed and threaded into the constructed
    ``MhsDiagnosticRequest``; omitting it defaults to False."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(ev, "run_mhs_horizon_diagnostic", lambda request: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    args = parser.parse_args(["--fold-safe-horizon"])
    assert args.fold_safe_horizon is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["fold_safe_horizon_selection"] is True

    captured.clear()
    args = parser.parse_args([])
    assert args.fold_safe_horizon is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["fold_safe_horizon_selection"] is False


def test_mhs_diagnostic_ladder_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_LADDER_CLI_FLAG: ``--ladder-diagnostic`` is parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; omitting it
    defaults to False."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(ev, "run_mhs_horizon_diagnostic", lambda request: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    args = parser.parse_args(["--ladder-diagnostic"])
    assert args.ladder_diagnostic is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["ladder_diagnostic"] is True

    captured.clear()
    args = parser.parse_args([])
    assert args.ladder_diagnostic is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["ladder_diagnostic"] is False
