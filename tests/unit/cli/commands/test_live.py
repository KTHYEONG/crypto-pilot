"""src/cli/commands/live.py 등록 검증 (mirrored unit test)."""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from src.cli.commands.live import add_live_commands


def _live_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_live_commands(parser)
    return parser


def test_shadow_cycle_parses_utc_decision_time() -> None:
    args = _live_parser().parse_args(
        ["shadow-cycle", "--decision-time", "2026-08-24T00:00:00Z"]
    )
    assert args.decision_time == pd.Timestamp("2026-08-24 00:00Z")
    assert args.artifact.endswith("deployed_target_weights.parquet")
    assert args.dry_run is False


def test_decision_time_is_required() -> None:
    with pytest.raises(SystemExit):
        _live_parser().parse_args(["shadow-cycle"])
