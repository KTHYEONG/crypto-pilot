"""Contract coverage for the MHS CLI argument surface (MHS-MEM-03 wiring)."""

from __future__ import annotations

import argparse
import logging
import re
import types

import pytest

from src.cli.commands.research.mhs import _run_mhs_horizon_diagnostic, add_mhs_commands
import src.mhs.pipeline.orchestrator as orchestrator


def _fake_report() -> types.SimpleNamespace:
    return types.SimpleNamespace(status="COMPLETE", books=[], blend=None)


def test_mhs_diagnostic_defaults_and_mark_mode_choices() -> None:
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["mark_mode"] == "cache_required"
    assert defaults["execution_timeframe"] == "3m"
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

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())

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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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


def test_mhs_diagnostic_committee_kelly_sizing_defaults_on_and_opt_out(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_KELLY_SIZING_MAIN_LOGIC_DEFAULT: committee Kelly
    sizing (real 3m replay measured CAGR 341.6%/MDD -38.4%/Calmar 8.89 vs the
    Kelly-off baseline's CAGR 349.8%/MDD -45.6%/Calmar 7.68,
    ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING) is on by default whenever
    committee capital is active; ``--no-committee-kelly-sizing`` opts back out
    to the pure vol-target scale while leaving committee capital on."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["no_committee_kelly_sizing"] is False

    args = parser.parse_args([])
    assert args.no_committee_kelly_sizing is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_kelly_sizing"] is True

    captured.clear()
    args = parser.parse_args(["--no-committee-kelly-sizing"])
    assert args.no_committee_kelly_sizing is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_kelly_sizing"] is False


def test_mhs_diagnostic_committee_growth_diagnostic_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_GROWTH_DIAGNOSTIC_CLI_FLAG_THREADED:
    ``--committee-growth-diagnostic`` (store_true, default False, requires
    ``--committee-book``) is parsed and threaded into the constructed
    ``MhsDiagnosticRequest``; omitting it yields committee_growth_diagnostic=False."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["committee_growth_diagnostic"] is False

    args = parser.parse_args(["--committee-book", "--committee-growth-diagnostic"])
    assert args.committee_book is True
    assert args.committee_growth_diagnostic is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_book"] is True
    assert captured["committee_growth_diagnostic"] is True

    captured.clear()
    args = parser.parse_args(["--committee-book"])
    assert args.committee_growth_diagnostic is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_growth_diagnostic"] is False


def test_mhs_committee_kelly_sizing_help_text_no_stale_claims() -> None:
    """SCENARIO_MHS_COMMITTEE_KELLY_SIZING_HELP_TEXT_NO_LONGER_CLAIMS_COMMITTEE_BOOK_REQUIRED:
    the registered ``--no-committee-kelly-sizing`` help no longer carries the
    stale 'requires --committee-book' claim, and parsing it alone succeeds
    now that committee capital is the default (no ``--committee-book``
    needed)."""
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    kelly = next(a for a in parser._actions if a.dest == "no_committee_kelly_sizing")
    assert "requires --committee-book" not in kelly.help

    args = parser.parse_args(["--no-committee-kelly-sizing"])
    assert args.no_committee_capital is False
    assert args.no_committee_kelly_sizing is True


# SCENARIO_MHS_KELLY_TWO_SIDED_07
def test_scenario_mhs_kelly_two_sided_07_help_texts_reflect_two_sided_cap() -> None:
    """The kelly-sizing help no longer carries the falsified de-leverager
    claims, and the universe-size help documents the measured breadth-60
    default matching the cap_60_roster attestation."""
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    kelly = next(a for a in parser._actions if a.dest == "no_committee_kelly_sizing")
    assert "requires --committee-book" not in kelly.help
    assert "capped at 1.0x" not in kelly.help
    assert "net negative for compounded growth" not in kelly.help
    assert "leverage_ceiling" in kelly.help

    universe = next(a for a in parser._actions if a.dest == "execution_universe_size")
    assert "60" in universe.help


def test_mhs_diagnostic_committee_capital_defaults_on_and_opt_out(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_CAPITAL_MAIN_LOGIC_DEFAULT: committee capital (the
    best-measured configuration) is the main-logic default -- omitting any flag
    threads committee_capital=True into MhsDiagnosticRequest, and
    ``--no-committee-capital`` opts back out to committee_capital=False (which
    also forces committee_regime_adaptive_tranche=False, since it requires
    committee capital)."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["no_committee_capital"] is False

    args = parser.parse_args([])
    assert args.no_committee_capital is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_regime_adaptive_tranche"] is True

    captured.clear()
    args = parser.parse_args(["--no-committee-capital"])
    assert args.no_committee_capital is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is False
    assert captured["committee_regime_adaptive_tranche"] is False


def test_mhs_diagnostic_committee_tranche_smoothing_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_TRANCHE_SMOOTHING_CLI_FLAG_THREADED:
    ``--committee-tranche-smoothing`` (store_true, default False) is parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; passing it
    overrides the regime-adaptive main-logic default (the two are mutually
    exclusive) rather than raising."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["committee_tranche_smoothing"] is False

    args = parser.parse_args([])
    assert args.committee_tranche_smoothing is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_tranche_smoothing"] is False
    assert captured["committee_regime_adaptive_tranche"] is True

    captured.clear()
    args = parser.parse_args(["--committee-tranche-smoothing"])
    assert args.committee_tranche_smoothing is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_tranche_smoothing"] is True
    assert captured["committee_regime_adaptive_tranche"] is False


def test_mhs_diagnostic_committee_regime_adaptive_tranche_defaults_on_and_opt_out(
    monkeypatch,
) -> None:
    """SCENARIO_MHS_COMMITTEE_REGIME_ADAPTIVE_TRANCHE_MAIN_LOGIC_DEFAULT: the
    regime-adaptive tranche (the best-measured configuration) is on by default
    whenever committee capital is active; ``--no-committee-regime-adaptive-tranche``
    opts back out to the raw committee book while leaving committee capital on."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["no_committee_regime_adaptive_tranche"] is False

    args = parser.parse_args([])
    assert args.no_committee_regime_adaptive_tranche is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_regime_adaptive_tranche"] is True

    captured.clear()
    args = parser.parse_args(["--no-committee-regime-adaptive-tranche"])
    assert args.no_committee_regime_adaptive_tranche is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_regime_adaptive_tranche"] is False


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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
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
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())

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


def test_mhs_diagnostic_execution_timeframe_3m_default(monkeypatch) -> None:
    """SCENARIO_MHS_CLI_EXECUTION_TIMEFRAME_3M_DEFAULT: parsing
    ``mhs-horizon-diagnostic`` args without ``--execution-timeframe`` yields
    ``args.execution_timeframe == "3m"``; ``--execution-timeframe 3m`` is
    accepted; and the constructed ``MhsDiagnosticRequest`` carries
    ``execution_timeframe="3m"``."""
    import src.application.research.mhs.evaluation as ev

    captured: dict = {}

    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["execution_timeframe"] == "3m"

    args = parser.parse_args([])
    assert args.execution_timeframe == "3m"
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_timeframe"] == "3m"

    captured.clear()
    args = parser.parse_args(["--execution-timeframe", "3m"])
    assert args.execution_timeframe == "3m"
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_timeframe"] == "3m"


def test_cli_flags_threaded(monkeypatch) -> None:
    """SCENARIO_CLI_FLAGS_THREADED: new CLI args are threaded into MhsDiagnosticRequest."""
    import src.application.research.mhs.evaluation as ev
    import src.mhs.pipeline.orchestrator as orchestrator
    from src.mhs.types import FUNDING_CARRY_SLEEVE_WEIGHT

    captured: dict = {}
    real_request = ev.MhsDiagnosticRequest

    def _spy_request(*args, **kwargs):
        captured.update(kwargs)
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ev, "MhsDiagnosticRequest", _spy_request)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: _fake_report())
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["pnl_vol_target_mode"] == "growth_budget"
    assert defaults["no_funding_carry_sleeve"] is False
    assert defaults["funding_carry_weight"] == FUNDING_CARRY_SLEEVE_WEIGHT

    # Default (2026-08-22 main logic): growth_budget, carry sleeve ON
    # (committee_capital default ON)
    captured.clear()
    args = parser.parse_args([])
    assert args.pnl_vol_target_mode == "growth_budget"
    assert args.no_funding_carry_sleeve is False
    assert args.funding_carry_weight == FUNDING_CARRY_SLEEVE_WEIGHT
    _run_mhs_horizon_diagnostic(args)
    assert captured["pnl_vol_target_mode"] == "growth_budget"
    assert captured["funding_carry_sleeve"] is True
    assert captured["funding_carry_weight"] == FUNDING_CARRY_SLEEVE_WEIGHT

    # --no-funding-carry-sleeve disables sleeve
    captured.clear()
    args = parser.parse_args(["--no-funding-carry-sleeve"])
    _run_mhs_horizon_diagnostic(args)
    assert captured["funding_carry_sleeve"] is False
    assert captured["funding_carry_weight"] == 0.0

    # --no-committee-capital disables sleeve
    captured.clear()
    args = parser.parse_args(["--no-committee-capital"])
    _run_mhs_horizon_diagnostic(args)
    assert captured["funding_carry_sleeve"] is False

    # --pnl-vol-target-mode threads through
    captured.clear()
    args = parser.parse_args(["--pnl-vol-target-mode", "median_relative"])
    _run_mhs_horizon_diagnostic(args)
    assert captured["pnl_vol_target_mode"] == "median_relative"


def test_mhs_diagnostic_leverage_frontier_scan_short_circuit_scenario_mhs_leverage_scan_06(monkeypatch) -> None:
    """SCENARIO_MHS_LEVERAGE_SCAN_06: ``--leverage-frontier-scan`` short-circuits
    the handler before any heavy import -- the full pipeline must never run on
    the scan path, while the flag=False path still reaches the pipeline."""
    from src.mhs.params import LEVERAGE_FRONTIER_SCAN_MULTIPLES
    import src.application.research.mhs.leverage_scan as leverage_scan

    def _boom(config):
        raise AssertionError("full pipeline must not run")

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _boom)
    captured: dict = {}

    def _stub_scan(envelope_name, candidate_multiples, artifact_path=None):
        captured["envelope_name"] = envelope_name
        captured["candidate_multiples"] = candidate_multiples
        return ()

    monkeypatch.setattr(leverage_scan, "run_leverage_frontier_scan", _stub_scan)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    args = parser.parse_args(["--leverage-frontier-scan"])
    assert args.leverage_frontier_scan is True
    assert args.leverage_frontier_multiples == LEVERAGE_FRONTIER_SCAN_MULTIPLES
    _run_mhs_horizon_diagnostic(args)
    assert captured["envelope_name"] == "growth_extreme"
    assert captured["candidate_multiples"] == LEVERAGE_FRONTIER_SCAN_MULTIPLES

    captured.clear()
    args = parser.parse_args(
        ["--leverage-frontier-scan", "--growth-envelope", "balanced",
         "--leverage-frontier-multiples", "2.0, 2.5, 3.0"],
    )
    _run_mhs_horizon_diagnostic(args)
    assert captured["envelope_name"] == "balanced"
    assert captured["candidate_multiples"] == (2.0, 2.5, 3.0)

    # The identical run_mhs_diagnostic-raises monkeypatch MUST trip when the
    # flag is off: proves the flag gates the branch instead of the pipeline
    # call having been removed outright.
    args = parser.parse_args([])
    assert args.leverage_frontier_scan is False
    with pytest.raises(AssertionError, match="full pipeline must not run"):
        _run_mhs_horizon_diagnostic(args)


def test_mhs_leverage_frontier_multiples_rejects_non_float_token() -> None:
    from src.cli.commands.research.mhs import _parse_float_csv

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    # argparse converts the type callback's ArgumentTypeError into its own
    # usage error (SystemExit); the offending token is still surfaced.
    with pytest.raises(SystemExit), pytest.raises(argparse.ArgumentTypeError):
        parser.parse_args(["--leverage-frontier-multiples", "1.0,abc"])
    with pytest.raises(argparse.ArgumentTypeError, match="not-a-float"):
        _parse_float_csv("not-a-float")


def test_mhs_emit_target_weights_calls_persist_seam(monkeypatch) -> None:
    """--emit-target-weights invokes emit_deployed_target_weights with the
    completed blend's target_weights/exposure_scale (fail-closed check bugfix)."""
    import pandas as pd

    import src.application.research.mhs.evaluation as ev
    import src.mhs.report.persist as persist_mod

    target_weights = pd.DataFrame({"BTCUSDT": [0.1, -0.1]})
    exposure_scale = pd.Series([1.2, 1.3])
    fake_blend = types.SimpleNamespace(target_weights=target_weights, exposure_scale=exposure_scale)
    fake_report = types.SimpleNamespace(status="COMPLETE", books=[], blend=fake_blend)

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: fake_report)
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: "docs/results/mhs.json")

    captured: dict = {}

    def _spy_emit(tw, scale, artifact_root, *, tail_rows):
        captured["target_weights"] = tw
        captured["exposure_scale"] = scale
        captured["artifact_root"] = artifact_root
        captured["tail_rows"] = tail_rows
        return {"path": str(artifact_root / "deployed_target_weights.parquet"), "rows": tail_rows}

    monkeypatch.setattr(persist_mod, "emit_deployed_target_weights", _spy_emit)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    args = parser.parse_args(["--emit-target-weights"])
    _run_mhs_horizon_diagnostic(args)

    assert captured["target_weights"] is target_weights
    assert captured["exposure_scale"] is exposure_scale
    assert str(captured["artifact_root"]).endswith("mhs_artifacts")
    assert captured["tail_rows"] > 0


def test_mhs_emit_target_weights_fails_closed_without_blend(monkeypatch) -> None:
    """A blend-less/target_weights-less report must never be silently skipped."""
    import src.application.research.mhs.evaluation as ev
    from src.common.errors import DataIntegrityError

    fake_report = types.SimpleNamespace(status="COMPLETE", books=[], blend=None)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config: fake_report)
    monkeypatch.setattr(ev, "persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)
    monkeypatch.setattr(ev, "mhs_horizon_diagnostic_report_path", lambda: "docs/results/mhs.json")

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    args = parser.parse_args(["--emit-target-weights"])
    with pytest.raises(DataIntegrityError):
        _run_mhs_horizon_diagnostic(args)
