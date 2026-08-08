"""Contract coverage for the MHS CLI argument surface (MHS-MEM-03 wiring)."""

from __future__ import annotations

import argparse

from src.cli.commands.research.mhs import add_mhs_commands


def test_mhs_diagnostic_defaults_and_mark_mode_choices() -> None:
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["mark_mode"] == "cache_required"
    assert defaults["execution_timeframe"] == "5m"
    assert defaults["max_rss_bytes"] is None
    assert defaults["no_log_run"] is False
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
