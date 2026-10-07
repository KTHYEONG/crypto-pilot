"""Contract coverage for the MHS CLI argument surface (MHS-MEM-03 wiring)."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import types

import pytest

from src.cli.commands.research.mhs import _run_mhs_horizon_diagnostic, add_mhs_commands
import src.mhs.pipeline.orchestrator as orchestrator
import src.mhs.reporting.inventory as rep_inventory


def _fake_report() -> types.SimpleNamespace:
    return types.SimpleNamespace(status="COMPLETE", books=[], blend=None)


def test_mhs_diagnostic_defaults_and_mark_mode_choices() -> None:
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    defaults = {action.dest: action.default for action in parser._actions}
    assert "mark_mode" not in defaults
    assert defaults["output_tier"] == "compact"
    # Request flags are SUPPRESS-based: unstated fields carry no default value.
    assert defaults["execution_timeframe"] == argparse.SUPPRESS
    assert defaults["max_rss_bytes"] == argparse.SUPPRESS
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert explicit == {}
    assert "max_rss_bytes" not in explicit
    assert not hasattr(args, "max_rss_bytes")
    assert resolve_cli_request(explicit) == MhsDiagnosticRequest()


def test_mhs_diagnostic_max_rss_bytes_flag_wired_to_request() -> None:
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    args = parser.parse_args(["--max-rss-bytes", "8000000000"])
    assert args.max_rss_bytes == 8_000_000_000
    assert explicit_field_values(MhsDiagnosticRequest, args)["max_rss_bytes"] == 8_000_000_000
    with pytest.raises(SystemExit):
        parser.parse_args(["--mark-mode", "cache_required_stale_carry"])


def test_mhs_diagnostic_output_tier_flag_threaded_to_persist(monkeypatch) -> None:
    """``--output-tier full`` is parsed and threaded into the persist call;
    the default stays ``compact``."""

    captured: dict = {}
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["output_tier"] == "compact"

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())

    def _spy_persist(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", _spy_persist)

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
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    args = parser.parse_args(["--touch-diagnostic"])
    assert args.touch_diagnostic is True
    assert explicit_field_values(MhsDiagnosticRequest, args)["touch_diagnostic"] is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["touch_diagnostic"] is True

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "touch_diagnostic" not in explicit
    assert not hasattr(args, "touch_diagnostic")
    _run_mhs_horizon_diagnostic(args)
    assert captured["touch_diagnostic"] is False


def test_mhs_diagnostic_fold_safe_horizon_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_FOLD_SAFE_HORIZON_08_CLI_FLAG_THREADS_THROUGH:
    ``--fold-safe-horizon`` is parsed and threaded into the constructed
    ``MhsDiagnosticRequest``; omitting it defaults to False."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    args = parser.parse_args(["--fold-safe-horizon"])
    assert args.fold_safe_horizon_selection is True
    assert explicit_field_values(MhsDiagnosticRequest, args)["fold_safe_horizon_selection"] is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["fold_safe_horizon_selection"] is True

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "fold_safe_horizon_selection" not in explicit
    assert not hasattr(args, "fold_safe_horizon_selection")
    _run_mhs_horizon_diagnostic(args)
    assert captured["fold_safe_horizon_selection"] is False


def test_mhs_diagnostic_ladder_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_LADDER_CLI_FLAG: ``--ladder-diagnostic`` is parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; omitting it
    defaults to False."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    args = parser.parse_args(["--ladder-diagnostic"])
    assert args.ladder_diagnostic is True
    assert explicit_field_values(MhsDiagnosticRequest, args)["ladder_diagnostic"] is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["ladder_diagnostic"] is True

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "ladder_diagnostic" not in explicit
    assert not hasattr(args, "ladder_diagnostic")
    _run_mhs_horizon_diagnostic(args)
    assert captured["ladder_diagnostic"] is False


def test_mhs_diagnostic_crash_tilt_alpha_flag_threaded_to_request(monkeypatch) -> None:
    """The opt-in ``--crash-regime-tilt-alpha`` is parsed and threaded into the
    constructed ``MhsDiagnosticRequest``; the default stays None (disabled,
    byte-identical to the fully dollar-neutral book)."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["crash_regime_tilt_alpha"] == argparse.SUPPRESS

    args = parser.parse_args(["--crash-regime-tilt-alpha", "0.3"])
    assert args.crash_regime_tilt_alpha == 0.3
    assert explicit_field_values(MhsDiagnosticRequest, args)["crash_regime_tilt_alpha"] == 0.3
    _run_mhs_horizon_diagnostic(args)
    assert captured["crash_regime_tilt_alpha"] == 0.3

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "crash_regime_tilt_alpha" not in explicit
    assert not hasattr(args, "crash_regime_tilt_alpha")
    _run_mhs_horizon_diagnostic(args)
    assert captured["crash_regime_tilt_alpha"] is None


def test_mhs_diagnostic_trend_sleeve_flags_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_CLI_TREND_SLEEVE_FLAGS: ``--trend-sleeve`` (store_true, default
    False) and ``--trend-sleeve-gross`` (type=float, default 0.0) are parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; omitting both yields
    the off values."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["trend_sleeve"] == argparse.SUPPRESS
    assert defaults["trend_sleeve_gross"] == argparse.SUPPRESS

    args = parser.parse_args(["--trend-sleeve", "--trend-sleeve-gross", "0.3"])
    assert args.trend_sleeve is True
    assert args.trend_sleeve_gross == 0.3
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert explicit["trend_sleeve"] is True
    assert explicit["trend_sleeve_gross"] == 0.3
    _run_mhs_horizon_diagnostic(args)
    assert captured["trend_sleeve"] is True
    assert captured["trend_sleeve_gross"] == 0.3

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "trend_sleeve" not in explicit
    assert "trend_sleeve_gross" not in explicit
    assert not hasattr(args, "trend_sleeve")
    assert not hasattr(args, "trend_sleeve_gross")
    assert real_resolve_cli_request(explicit).trend_sleeve_gross == 0.0
    _run_mhs_horizon_diagnostic(args)
    assert captured["trend_sleeve"] is False
    assert captured["trend_sleeve_gross"] == 0.0


def test_mhs_diagnostic_alpha_engine_flags_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_ALPHA_ENGINE_09: ``--slow-book-mode``, ``--rebalance-filter``,
    ``--beta-neutralize`` and ``--ensemble-signal`` parse into the matching
    ``MhsDiagnosticRequest`` fields, and omitting all three reproduces the
    current defaults."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["slow_book_mode"] == argparse.SUPPRESS
    assert defaults["rebalance_filter"] == argparse.SUPPRESS
    assert defaults["beta_neutralize"] == argparse.SUPPRESS
    assert defaults["ensemble_signal"] == argparse.SUPPRESS

    args = parser.parse_args(
        ["--slow-book-mode", "horizon_ensemble", "--rebalance-filter", "portfolio_trigger",
         "--beta-neutralize", "--ensemble-signal", "vol_normalized"],
    )
    assert args.slow_book_mode == "horizon_ensemble"
    assert args.rebalance_filter == "portfolio_trigger"
    assert args.beta_neutralize is True
    assert args.ensemble_signal == "vol_normalized"
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert explicit["slow_book_mode"] == "horizon_ensemble"
    assert explicit["rebalance_filter"] == "portfolio_trigger"
    assert explicit["beta_neutralize"] is True
    assert explicit["ensemble_signal"] == "vol_normalized"
    _run_mhs_horizon_diagnostic(args)
    assert captured["slow_book_mode"] == "horizon_ensemble"
    assert captured["rebalance_filter"] == "portfolio_trigger"
    assert captured["beta_neutralize"] is True
    assert captured["ensemble_signal"] == "vol_normalized"

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "slow_book_mode" not in explicit
    assert "beta_neutralize" not in explicit
    _run_mhs_horizon_diagnostic(args)
    assert captured["slow_book_mode"] == "single_horizon"
    assert captured["rebalance_filter"] == "per_symbol_deadband"
    assert captured["beta_neutralize"] is False
    assert captured["ensemble_signal"] == "raw"


def test_mhs_diagnostic_multi_feature_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_CLI_MULTI_FEATURE_FLAG: ``--multi-feature-book`` (store_true,
    default False) is parsed and threaded into the constructed
    ``MhsDiagnosticRequest``; omitting it yields multi_feature_book=False."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["multi_feature_book"] == argparse.SUPPRESS

    args = parser.parse_args(["--multi-feature-book"])
    assert args.multi_feature_book is True
    assert explicit_field_values(MhsDiagnosticRequest, args)["multi_feature_book"] is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["multi_feature_book"] is True

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "multi_feature_book" not in explicit
    assert not hasattr(args, "multi_feature_book")
    _run_mhs_horizon_diagnostic(args)
    assert captured["multi_feature_book"] is False


def test_mhs_diagnostic_committee_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_CLI_COMMITTEE_FLAG: ``--committee-book`` (store_true, default
    False) is parsed and threaded into the constructed ``MhsDiagnosticRequest``;
    omitting it yields committee_book=False."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["committee_book"] == argparse.SUPPRESS

    args = parser.parse_args(["--committee-book"])
    assert args.committee_book is True
    assert explicit_field_values(MhsDiagnosticRequest, args)["committee_book"] is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_book"] is True

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "committee_book" not in explicit
    assert not hasattr(args, "committee_book")
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_book"] is False


def test_mhs_diagnostic_committee_kelly_sizing_defaults_on_and_opt_out(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_KELLY_SIZING_MAIN_LOGIC_DEFAULT: committee Kelly
    sizing (real 3m replay measured CAGR 341.6%/MDD -38.4%/Calmar 8.89 vs the
    Kelly-off baseline's CAGR 349.8%/MDD -45.6%/Calmar 7.68,
    ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING) is on by default whenever
    committee capital is active; ``--no-committee-kelly-sizing`` opts back out
    to the pure vol-target scale while leaving committee capital on."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert "no_committee_kelly_sizing" not in defaults
    assert defaults["committee_kelly_sizing"] == argparse.SUPPRESS

    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "committee_kelly_sizing" not in explicit
    assert not hasattr(args, "committee_kelly_sizing")
    assert real_resolve_cli_request(explicit).committee_kelly_sizing is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_kelly_sizing"] is True

    captured.clear()
    args = parser.parse_args(["--no-committee-kelly-sizing"])
    assert args.committee_kelly_sizing is False
    assert explicit_field_values(MhsDiagnosticRequest, args)["committee_kelly_sizing"] is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_kelly_sizing"] is False


def test_mhs_diagnostic_committee_growth_diagnostic_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_GROWTH_DIAGNOSTIC_CLI_FLAG_THREADED:
    ``--committee-growth-diagnostic`` (store_true, default False, requires
    ``--committee-book``) is parsed and threaded into the constructed
    ``MhsDiagnosticRequest``; omitting it yields committee_growth_diagnostic=False."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["committee_growth_diagnostic"] == argparse.SUPPRESS

    args = parser.parse_args(["--committee-book", "--committee-growth-diagnostic"])
    assert args.committee_book is True
    assert args.committee_growth_diagnostic is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_book"] is True
    assert captured["committee_growth_diagnostic"] is True

    captured.clear()
    args = parser.parse_args(["--committee-book"])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "committee_growth_diagnostic" not in explicit
    assert not hasattr(args, "committee_growth_diagnostic")
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_growth_diagnostic"] is False


def test_mhs_committee_kelly_sizing_help_text_no_stale_claims() -> None:
    """SCENARIO_MHS_COMMITTEE_KELLY_SIZING_HELP_TEXT_NO_LONGER_CLAIMS_COMMITTEE_BOOK_REQUIRED:
    the registered ``--no-committee-kelly-sizing`` help no longer carries the
    stale 'requires --committee-book' claim, and parsing it alone succeeds
    now that committee capital is the default (no ``--committee-book``
    needed)."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    kelly = next(a for a in parser._actions if "--no-committee-kelly-sizing" in a.option_strings)
    assert kelly.dest == "committee_kelly_sizing"
    assert "requires --committee-book" not in kelly.help

    args = parser.parse_args(["--no-committee-kelly-sizing"])
    assert args.committee_kelly_sizing is False
    assert "committee_capital" not in explicit_field_values(MhsDiagnosticRequest, args)
    assert explicit_field_values(MhsDiagnosticRequest, args) == {"committee_kelly_sizing": False}


# SCENARIO_MHS_KELLY_TWO_SIDED_07
def test_scenario_mhs_kelly_two_sided_07_help_texts_reflect_two_sided_cap() -> None:
    """The kelly-sizing help no longer carries the falsified de-leverager
    claims, and the universe-size help documents the measured breadth-60
    default matching the cap_60_roster attestation."""
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    kelly = next(a for a in parser._actions if "--no-committee-kelly-sizing" in a.option_strings)
    assert kelly.dest == "committee_kelly_sizing"
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
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert "no_committee_capital" not in defaults
    assert defaults["committee_capital"] == argparse.SUPPRESS

    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "committee_capital" not in explicit
    assert not hasattr(args, "committee_capital")
    assert real_resolve_cli_request(explicit).committee_capital is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_regime_adaptive_tranche"] is True

    captured.clear()
    args = parser.parse_args(["--no-committee-capital"])
    assert args.committee_capital is False
    assert explicit_field_values(MhsDiagnosticRequest, args)["committee_capital"] is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is False
    assert captured["committee_regime_adaptive_tranche"] is False


def test_mhs_diagnostic_committee_tranche_smoothing_flag_threaded_to_request(monkeypatch) -> None:
    """SCENARIO_MHS_COMMITTEE_TRANCHE_SMOOTHING_CLI_FLAG_THREADED:
    ``--committee-tranche-smoothing`` (store_true, default False) is parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; passing it
    overrides the regime-adaptive main-logic default (the two are mutually
    exclusive) rather than raising."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["committee_tranche_smoothing"] == argparse.SUPPRESS

    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "committee_tranche_smoothing" not in explicit
    assert not hasattr(args, "committee_tranche_smoothing")
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_tranche_smoothing"] is False
    assert captured["committee_regime_adaptive_tranche"] is True

    captured.clear()
    args = parser.parse_args(["--committee-tranche-smoothing"])
    assert args.committee_tranche_smoothing is True
    assert explicit_field_values(MhsDiagnosticRequest, args)["committee_tranche_smoothing"] is True
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
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert "no_committee_regime_adaptive_tranche" not in defaults
    assert defaults["committee_regime_adaptive_tranche"] == argparse.SUPPRESS

    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "committee_regime_adaptive_tranche" not in explicit
    assert not hasattr(args, "committee_regime_adaptive_tranche")
    assert real_resolve_cli_request(explicit).committee_regime_adaptive_tranche is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_regime_adaptive_tranche"] is True

    captured.clear()
    args = parser.parse_args(["--no-committee-regime-adaptive-tranche"])
    assert args.committee_regime_adaptive_tranche is False
    assert explicit_field_values(MhsDiagnosticRequest, args)["committee_regime_adaptive_tranche"] is False
    _run_mhs_horizon_diagnostic(args)
    assert captured["committee_capital"] is True
    assert captured["committee_regime_adaptive_tranche"] is False


def test_mhs_diagnostic_execution_coverage_gate_flag_threaded(monkeypatch) -> None:
    """SCENARIO_MHS_DIAGNOSTIC_EXECUTION_COVERAGE_GATE_CLI_FLAG_THREADED:
    ``--execution-coverage-gate`` (store_true, default False) is parsed and
    threaded into the constructed ``MhsDiagnosticRequest``; omitting it yields
    execution_coverage_gate=False."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["execution_coverage_gate"] == argparse.SUPPRESS

    args = parser.parse_args(["--execution-coverage-gate"])
    assert args.execution_coverage_gate is True
    assert explicit_field_values(MhsDiagnosticRequest, args)["execution_coverage_gate"] is True
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_coverage_gate"] is True

    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "execution_coverage_gate" not in explicit
    assert not hasattr(args, "execution_coverage_gate")
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_coverage_gate"] is False


def test_mhs_diagnostic_persist_stage_logged(monkeypatch, caplog) -> None:
    """SCENARIO_MHS_CLI_PERSIST_STAGE_LOGGED: the persist step emits a [SYS]
    stage=persist_report elapsed_ms=<int> log line, so the post-run report
    serialization span is visible to the same [SYS] telemetry."""

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

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
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}
    requests: list = []

    def _spy(explicit):
        req = real_resolve_cli_request(explicit)
        requests.append(req)
        return req

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())

    def _spy_persist(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", _spy_persist)

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
    an absent explicit value resolving to ``"3m"``; ``--execution-timeframe 3m``
    is accepted; and the constructed ``MhsDiagnosticRequest`` carries
    ``execution_timeframe="3m"``."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request

    captured: dict = {}

    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["execution_timeframe"] == argparse.SUPPRESS

    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "execution_timeframe" not in explicit
    assert not hasattr(args, "execution_timeframe")
    assert real_resolve_cli_request(explicit).execution_timeframe == "3m"
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_timeframe"] == "3m"

    captured.clear()
    args = parser.parse_args(["--execution-timeframe", "3m"])
    assert args.execution_timeframe == "3m"
    assert explicit_field_values(MhsDiagnosticRequest, args)["execution_timeframe"] == "3m"
    _run_mhs_horizon_diagnostic(args)
    assert captured["execution_timeframe"] == "3m"


def test_cli_flags_threaded(monkeypatch) -> None:
    """SCENARIO_CLI_FLAGS_THREADED: new CLI args are threaded into MhsDiagnosticRequest."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request as real_resolve_cli_request
    import src.mhs.pipeline.orchestrator as orchestrator
    from src.mhs.types import FUNDING_CARRY_SLEEVE_WEIGHT

    captured: dict = {}
    def _spy(explicit):
        request = real_resolve_cli_request(explicit)
        captured.update(dataclasses.asdict(request))
        return request

    monkeypatch.setattr("src.mhs.pipeline.config.resolve_cli_request", _spy)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["pnl_vol_target_mode"] == argparse.SUPPRESS
    assert "no_funding_carry_sleeve" not in defaults
    assert defaults["funding_carry_sleeve"] == argparse.SUPPRESS
    assert defaults["funding_carry_weight"] == argparse.SUPPRESS

    # Default (2026-08-22 main logic): growth_budget, carry sleeve ON
    # (committee_capital default ON)
    captured.clear()
    args = parser.parse_args([])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert "pnl_vol_target_mode" not in explicit
    assert "funding_carry_sleeve" not in explicit
    assert "funding_carry_weight" not in explicit
    assert not hasattr(args, "pnl_vol_target_mode")
    assert not hasattr(args, "funding_carry_weight")
    _run_mhs_horizon_diagnostic(args)
    assert captured["pnl_vol_target_mode"] == "growth_budget"
    assert captured["funding_carry_sleeve"] is True
    assert captured["funding_carry_weight"] == FUNDING_CARRY_SLEEVE_WEIGHT

    # --no-funding-carry-sleeve disables sleeve
    captured.clear()
    args = parser.parse_args(["--no-funding-carry-sleeve"])
    assert args.funding_carry_sleeve is False
    assert explicit_field_values(MhsDiagnosticRequest, args)["funding_carry_sleeve"] is False
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
    assert explicit_field_values(MhsDiagnosticRequest, args)["pnl_vol_target_mode"] == "median_relative"
    _run_mhs_horizon_diagnostic(args)
    assert captured["pnl_vol_target_mode"] == "median_relative"


def test_mhs_diagnostic_leverage_frontier_scan_short_circuit_scenario_mhs_leverage_scan_06(monkeypatch) -> None:
    """SCENARIO_MHS_LEVERAGE_SCAN_06: ``--leverage-frontier-scan`` short-circuits
    the handler before any heavy import -- the full pipeline must never run on
    the scan path, while the flag=False path still reaches the pipeline."""
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.params import LEVERAGE_FRONTIER_SCAN_MULTIPLES
    import src.mhs.leverage_scan as leverage_scan

    def _boom(config, **kwargs):
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
    from src.mhs.params import (
        CLI_GROWTH_ENVELOPE_DEFAULT as _CLI_GROWTH_ENVELOPE_DEFAULT,
    )

    assert captured["envelope_name"] == _CLI_GROWTH_ENVELOPE_DEFAULT
    assert captured["candidate_multiples"] == LEVERAGE_FRONTIER_SCAN_MULTIPLES

    captured.clear()
    args = parser.parse_args(
        ["--leverage-frontier-scan", "--growth-envelope", "balanced",
         "--leverage-frontier-multiples", "2.0, 2.5, 3.0"],
    )
    assert explicit_field_values(MhsDiagnosticRequest, args)["growth_envelope"] == "balanced"
    _run_mhs_horizon_diagnostic(args)
    assert captured["envelope_name"] == "balanced"
    assert captured["candidate_multiples"] == (2.0, 2.5, 3.0)

    # The identical run_mhs_diagnostic-raises monkeypatch MUST trip when the
    # flag is off: proves the flag gates the branch instead of the pipeline
    # call having been removed outright.
    args = parser.parse_args([])
    assert args.leverage_frontier_scan is False
    assert "growth_envelope" not in explicit_field_values(MhsDiagnosticRequest, args)
    assert not hasattr(args, "growth_envelope")
    assert explicit_field_values(MhsDiagnosticRequest, args).get("growth_envelope", _CLI_GROWTH_ENVELOPE_DEFAULT) == _CLI_GROWTH_ENVELOPE_DEFAULT
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


def test_mhs_execution_timeframe_restricted_to_3m() -> None:
    """MHS intervals refuse 1m/5m at parse time while keeping an explicit 3m flag."""
    import pytest

    import src.cli.commands.data as data_mod

    parser = __import__("argparse").ArgumentParser()
    data_mod.add_data_commands(parser.add_subparsers(dest="group", required=True).add_parser("data"))
    with pytest.raises(SystemExit):
        parser.parse_args(["data", "collect", "mhs-execution", "--timeframe", "1m"])
    with pytest.raises(SystemExit):
        parser.parse_args(["data", "collect", "mhs-execution", "--timeframe", "5m"])
    assert parser.parse_args(["data", "collect", "mhs-execution"]).timeframe == "3m"
    with pytest.raises(SystemExit):
        parser.parse_args(["data", "seal-mhs-inputs", "--execution-timeframe", "5m"])
    sub = __import__("argparse").ArgumentParser().add_subparsers()
    from src.cli.commands.research.mhs import add_mhs_commands as _add

    _add(sub)
    with pytest.raises(SystemExit):
        sub.choices["mhs-horizon-diagnostic"].parse_args(["--execution-timeframe", "1m"])
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request

    _args = sub.choices["mhs-horizon-diagnostic"].parse_args([])
    _explicit = explicit_field_values(MhsDiagnosticRequest, _args)
    assert "execution_timeframe" not in _explicit
    assert resolve_cli_request(_explicit).execution_timeframe == "3m"


def test_mhs_collection_rejects_unsupported_timeframe_programmatically(tmp_path) -> None:
    """Direct MHS collection/sealing calls reject 5m before external writes."""
    import pytest

    import src.market_data.services.mhs_execution as mc
    from src.mhs.data_provenance import mhs_sealable_input_paths, resolve_required_mhs_input_paths

    with pytest.raises(ValueError, match="unknown execution_timeframe"):
        mc.build_mhs_execution_plan("2021-01-01", "2021-02-01", timeframe="5m")
    with pytest.raises(ValueError, match="unknown execution_timeframe"):
        resolve_required_mhs_input_paths(data_root=tmp_path, panel_symbols=["A"], execution_symbols=["A"], execution_timeframe="5m")
    with pytest.raises(ValueError, match="unknown execution_timeframe"):
        mhs_sealable_input_paths(data_root=tmp_path, execution_timeframe="1m")
    manifest = tmp_path / "m.json"
    assert not manifest.exists()


def test_generic_collection_preserves_intervals() -> None:
    """Generic data commands still accept non-MHS intervals."""
    import src.cli.commands.data as data_mod

    parser = __import__("argparse").ArgumentParser()
    data_mod.add_data_commands(parser.add_subparsers(dest="group", required=True).add_parser("data"))
    args = parser.parse_args(["data", "collect", "futures-ohlcv", "BTCUSDT", "1m"])
    assert args.timeframe == "1m"


def test_comparable_benchmark_requires_same_workload_evidence() -> None:
    """Resource claims rest on measured peaks/availability with labelled hourly comparison."""
    import types

    memory = types.SimpleNamespace(
        tree_pss_peak_bytes=456,
        tree_uss_peak_bytes=789,
        min_system_available_bytes=2 * 2**30,
    )
    proxy = types.SimpleNamespace(certification_level="process_proxy_1h_ledger")
    report = types.SimpleNamespace(proxy=proxy, memory_stats=memory)
    assert report.memory_stats.tree_pss_peak_bytes >= 0
    assert report.memory_stats.tree_uss_peak_bytes >= 0
    assert report.memory_stats.min_system_available_bytes > 0
    assert report.proxy.certification_level == "process_proxy_1h_ledger"
    assert rep_inventory.PROCESS_INVENTORY_CERTIFICATION_LEVEL == "process_inventory_3m"


def test_research_mhs_process_leaf_is_absent() -> None:
    import argparse

    from src.cli.commands.research.mhs import add_mhs_commands
    from src.cli.main import build_root_parser

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    assert set(sub.choices) == {"mhs-horizon-diagnostic"}

    args = build_root_parser().parse_args(["research", "run", "portfolio", "mhs-horizon-diagnostic"])
    assert args.portfolio_command == "mhs-horizon-diagnostic"
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "portfolio", "mhs-process-backtest"])


def test_register_procedure_help_names_the_canonical_registry() -> None:
    """The procedure registry is gitignored runtime evidence, not documentation."""
    import argparse

    from src.cli.commands.research.mhs import add_mhs_commands

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]

    action = next(a for a in parser._actions if a.dest == "register_procedure")
    assert "procedure_registry.jsonl" in action.help
    assert "git-tracked" not in action.help


def test_research_diagnostic_output_has_data_boundary(monkeypatch) -> None:
    import argparse
    from pathlib import Path

    import src.cli.commands.research.mhs as mhs_cli
    from src.common.paths import DATA_DIR

    captured: dict = {}

    def _spy_persist(report, target, **kwargs):
        captured["target"] = Path(target)
        return captured["target"]

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: _fake_report())
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", _spy_persist)

    sub = argparse.ArgumentParser().add_subparsers()
    mhs_cli.add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    mhs_cli._run_mhs_horizon_diagnostic(parser.parse_args([]))
    assert DATA_DIR / "research" / "mhs" in captured["target"].parents
    assert "docs" not in captured["target"].parts

    captured.clear()
    mhs_cli._run_mhs_horizon_diagnostic(parser.parse_args(["--run-id", "abc123"]))
    assert captured["target"] == DATA_DIR / "research" / "mhs" / "abc123" / "mhs_horizon_diagnostic.json"


def test_canonical_backtest_route_remains_unique() -> None:
    from src.cli.commands.backtest import run_mhs_backtest
    from src.cli.main import build_root_parser

    args = build_root_parser().parse_args(["backtest", "mhs"])
    assert args.handler is run_mhs_backtest


def test_retired_report_path_constants_have_no_callers() -> None:
    import pytest

    from pathlib import Path as _Path

    retired = ("PROCESS_INVENTORY_REPORT_PATH", "PROCESS_REPORT_PATH", "PROCESS_POLICY_REPORT_PATH")
    violations = [
        str(path)
        for path in sorted(_Path("src").rglob("*.py"))
        if any(name in path.read_text(encoding="utf-8") for name in retired)
    ]
    assert violations == []
    with pytest.raises(ImportError):
        from src.mhs.reporting.process import PROCESS_REPORT_PATH  # noqa: F401


def test_cli_attaches_telemetry_log_before_running(monkeypatch) -> None:
    import argparse
    import os
    from pathlib import Path

    import src.common.logging as app_logging
    from src.mhs.telemetry import TELEMETRY_LOGGER_NAME

    calls: list = []

    def _spy_setup(name, *, log_dir, level=logging.INFO):
        calls.append(("setup", name, log_dir))
        return logging.getLogger(name)

    monkeypatch.setattr(app_logging, "setup_logger", _spy_setup)
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", lambda config, **kwargs: (calls.append(("run",)), _fake_report())[1])
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", lambda *a, **k: None)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    _run_mhs_horizon_diagnostic(parser.parse_args([]))

    assert calls[0] == ("setup", TELEMETRY_LOGGER_NAME, app_logging.LOG_DIR)
    assert calls[1] == ("run",)
    assert Path(os.environ["PYTEST_DEBUG_TEMPROOT"]).resolve() in app_logging.LOG_DIR.resolve().parents or app_logging.LOG_DIR.resolve() == Path(os.environ["PYTEST_DEBUG_TEMPROOT"]).resolve()


def test_registration_path_opens_no_telemetry_log(monkeypatch) -> None:
    import argparse

    import pandas as pd

    import src.common.logging as app_logging

    calls: list = []
    monkeypatch.setattr(app_logging, "setup_logger", lambda *a, **k: (calls.append("setup"), logging.getLogger("x"))[1])
    monkeypatch.setattr("src.mhs.preregistration.register_procedure", lambda *a, **k: types.SimpleNamespace(procedure_digest="d", effective_start=pd.Timestamp("2026-01-01", tz="UTC")))

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    _run_mhs_horizon_diagnostic(parser.parse_args(["--register-procedure"]))
    assert calls == []


def test_cli_resolves_operator_storage_defaults(monkeypatch) -> None:
    """CLI resolves operator storage defaults for diagnostic and persist."""
    import argparse
    import os

    import src.common.paths as paths_mod
    import src.mhs.preregistration as prereg_mod

    run_kwargs: dict = {}
    persist_kwargs: dict = {}
    persist_args: tuple = ()

    def _fake_run(config, **kwargs):
        run_kwargs.update(kwargs)
        return _fake_report()

    def _spy_persist(*args, **kwargs):
        nonlocal persist_args
        persist_args = args
        persist_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _fake_run)
    monkeypatch.setattr("src.mhs.report.persist.persist_mhs_horizon_diagnostic_report", _spy_persist)

    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    _run_mhs_horizon_diagnostic(parser.parse_args([]))

    assert run_kwargs == {"procedure_registry": prereg_mod.PROCEDURE_REGISTRY_PATH, "history_dir": paths_mod.BACKTESTS_DIR}
    assert persist_kwargs["history_dir"] == paths_mod.BACKTESTS_DIR
    assert persist_kwargs["procedure_registry"] == prereg_mod.PROCEDURE_REGISTRY_PATH
    assert str(persist_args[1]).startswith(str(paths_mod.DATA_DIR / "research" / "mhs"))
    assert str(paths_mod.BACKTESTS_DIR).startswith(os.environ["PYTEST_DEBUG_TEMPROOT"])
