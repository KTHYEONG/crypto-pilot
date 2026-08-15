"""Contract coverage for the MHS CLI argument surface (MHS-MEM-03 wiring)."""

from __future__ import annotations

import argparse
import logging
import re
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


def test_mhs_diagnostic_crash_tilt_alpha_flag_threaded_to_request(monkeypatch) -> None:
    """The opt-in ``--crash-regime-tilt-alpha`` is parsed and threaded into the
    constructed ``MhsDiagnosticRequest``; the default stays None (disabled,
    byte-identical to the fully dollar-neutral book)."""
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

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["crash_regime_tilt_alpha"] is None

    args = parser.parse_args(["--crash-regime-tilt-alpha", "0.3"])
    assert args.crash_regime_tilt_alpha == 0.3
    _run_mhs_horizon_diagnostic(args)
    assert captured["crash_regime_tilt_alpha"] == 0.3

    captured.clear()
    args = parser.parse_args([])
    assert args.crash_regime_tilt_alpha is None
    _run_mhs_horizon_diagnostic(args)
    assert captured["crash_regime_tilt_alpha"] is None


def test_mhs_diagnostic_trend_sleeve_flags_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_CLI_TREND_SLEEVE_FLAGS: ``--trend-sleeve`` (store_true, default
    False) and ``--trend-sleeve-gross`` (type=float, default 0.0) are parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; omitting both yields
    the off values."""
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

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["trend_sleeve"] is False
    assert defaults["trend_sleeve_gross"] == 0.0

    args = parser.parse_args(["--trend-sleeve", "--trend-sleeve-gross", "0.3"])
    assert args.trend_sleeve is True
    assert args.trend_sleeve_gross == 0.3
    _run_mhs_horizon_diagnostic(args)
    assert captured["trend_sleeve"] is True
    assert captured["trend_sleeve_gross"] == 0.3

    captured.clear()
    args = parser.parse_args([])
    assert args.trend_sleeve is False
    assert args.trend_sleeve_gross == 0.0
    _run_mhs_horizon_diagnostic(args)
    assert captured["trend_sleeve"] is False
    assert captured["trend_sleeve_gross"] == 0.0


def test_mhs_diagnostic_alpha_engine_flags_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_ALPHA_ENGINE_09: ``--slow-book-mode``, ``--rebalance-filter``,
    ``--beta-neutralize`` and ``--ensemble-signal`` parse into the matching
    ``MhsDiagnosticRequest`` fields, and omitting all three reproduces the
    current defaults."""
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

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["slow_book_mode"] == "single_horizon"
    assert defaults["rebalance_filter"] == "per_symbol_deadband"
    assert defaults["beta_neutralize"] is False
    assert defaults["ensemble_signal"] == "raw"

    args = parser.parse_args(
        ["--slow-book-mode", "horizon_ensemble", "--rebalance-filter", "portfolio_trigger",
         "--beta-neutralize", "--ensemble-signal", "vol_normalized"],
    )
    assert args.slow_book_mode == "horizon_ensemble"
    assert args.rebalance_filter == "portfolio_trigger"
    assert args.beta_neutralize is True
    assert args.ensemble_signal == "vol_normalized"
    _run_mhs_horizon_diagnostic(args)
    assert captured["slow_book_mode"] == "horizon_ensemble"
    assert captured["rebalance_filter"] == "portfolio_trigger"
    assert captured["beta_neutralize"] is True
    assert captured["ensemble_signal"] == "vol_normalized"

    captured.clear()
    args = parser.parse_args([])
    _run_mhs_horizon_diagnostic(args)
    assert captured["slow_book_mode"] == "single_horizon"
    assert captured["rebalance_filter"] == "per_symbol_deadband"
    assert captured["beta_neutralize"] is False
    assert captured["ensemble_signal"] == "raw"


def test_mhs_diagnostic_multi_feature_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_CLI_MULTI_FEATURE_FLAG: ``--multi-feature-book`` (store_true,
    default False) is parsed and threaded into the constructed
    ``MhsDiagnosticRequest``; omitting it yields multi_feature_book=False."""
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

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["multi_feature_book"] is False

    args = parser.parse_args(["--multi-feature-book"])
    assert args.multi_feature_book is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["multi_feature_book"] is True

    captured.clear()
    args = parser.parse_args([])
    assert args.multi_feature_book is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["multi_feature_book"] is False


def test_mhs_diagnostic_committee_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_CLI_COMMITTEE_FLAG: ``--committee-book`` (store_true, default
    False) is parsed and threaded into the constructed ``MhsDiagnosticRequest``;
    omitting it yields committee_book=False."""
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

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["committee_book"] is False

    args = parser.parse_args(["--committee-book"])
    assert args.committee_book is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_book"] is True

    captured.clear()
    args = parser.parse_args([])
    assert args.committee_book is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_book"] is False


def test_mhs_diagnostic_committee_capital_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_CAPITAL_CLI_FLAG_THREADED: ``--committee-capital``
    (store_true, default False) is parsed and threaded into the constructed
    ``MhsDiagnosticRequest``; omitting it yields committee_capital=False."""
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

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["committee_capital"] is False

    args = parser.parse_args(["--committee-capital"])
    assert args.committee_capital is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True

    captured.clear()
    args = parser.parse_args([])
    assert args.committee_capital is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is False


def test_mhs_diagnostic_execution_coverage_gate_flag_threaded(monkeypatch) -> None:
    """SCENARIO_MHS_DIAGNOSTIC_EXECUTION_COVERAGE_GATE_CLI_FLAG_THREADED:
    ``--execution-coverage-gate`` (store_true, default False) is parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; omitting it yields
    execution_coverage_gate=False."""
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

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["execution_coverage_gate"] is False

    args = parser.parse_args(["--execution-coverage-gate"])
    assert args.execution_coverage_gate is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_coverage_gate"] is True

    captured.clear()
    args = parser.parse_args([])
    assert args.execution_coverage_gate is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_coverage_gate"] is False


def test_mhs_diagnostic_persist_stage_logged(monkeypatch, caplog) -> None:
    """SCENARIO_MHS_CLI_PERSIST_STAGE_LOGGED: the persist step emits a [SYS]
    stage=persist_report elapsed_ms=<int> log line, so the post-run report
    serialization span is visible to the same [SYS] telemetry."""
    import src.application.research.mhs.evaluation as ev

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    monkeypatch.setattr(ev, "run_mhs_horizon_diagnostic", lambda request: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    args = parser.parse_args([])
    with caplog.at_level(logging.INFO, logger="MhsHorizonDiagnosticCli"):
        _run_mhs_horizon_diagnostic(args)
    assert any(
        re.match(r"^\[SYS\] stage=persist_report elapsed_ms=\d+$", record.message)
        for record in caplog.records
    )


def test_mhs_diagnostic_persist_receives_request_object(monkeypatch) -> None:
    """SCENARIO_MHS_RESULT_LOG_07: ``_run_mhs_horizon_diagnostic`` threads the
    constructed ``MhsDiagnosticRequest`` into the persist call via ``request=``."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}
    requests: list = []
    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        req = real_request(*args, **kwargs)
        requests.append(req)
        return req

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(ev, "run_mhs_horizon_diagnostic", lambda request: _fake_report())

    def _spy_persist(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", _spy_persist)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: "docs/results/mhs.json")

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    args = parser.parse_args(["--slow-book-mode", "horizon_ensemble"])
    _run_mhs_horizon_diagnostic(args)

    assert len(requests) == 1
    assert "request" in captured
    assert captured["request"] is requests[0]
