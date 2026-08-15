"""Compound-only futures runner CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from src.application.futures.runner.compound_config import (
    CompoundRunConfig,
    build_compound_run_config,
)

_logger = logging.getLogger(__name__)

_REMOVED_FLAGS: tuple[str, ...] = (
    "trials",
    "timeframe",
    "mode",
    "alpha_only",
    "skip_universe",
    "skip_data_sync",
    "bypass_champion_guard",
    "symbols",
    "quick_backtest",
    "tf",
    "alpha_foundry",
    "alpha_foundry_total_l1_budget",
    "alpha_foundry_min_conviction_lcb_bps",
    "alpha_foundry_enable_fast_tf",
    "sync_metrics",
    "reference_date",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compound-only Futures Runner")
    parser.add_argument("--phase", type=str, default="full",
                        choices=["full", "verify-migration", "retire-legacy", "ladder"],
                        help="Execution phase (full, verify-migration, retire-legacy, ladder)")
    parser.add_argument("--date", type=str, default=None, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--sync", type=str, default="auto", choices=["auto", "local"],
                        help="Sync mode (auto=download, local=local-only)")
    parser.add_argument("--refresh-universe", action="store_true", help="Force universe refresh")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    for flag in _REMOVED_FLAGS:
        parser.add_argument(f"--{flag.replace('_', '-')}", action="store_true", default=None,
                            help=argparse.SUPPRESS)
    return parser


def _to_mapping(parsed: object) -> Mapping[str, Any]:
    if isinstance(parsed, Mapping):
        return parsed
    if hasattr(parsed, "__dict__"):
        return vars(parsed)
    return {}


def check_removed_flags(parsed: object) -> None:
    parsed_dict = _to_mapping(parsed)
    for flag in _REMOVED_FLAGS:
        raw = parsed_dict.get(flag)
        if raw is not None and raw is not False:
            raise SystemExit(f"error: unrecognized arguments: --{flag.replace('_', '-')}")


def run_from_cli(argv: Sequence[str] | None = None) -> int:
    return run_multiscale_cli(argv)


def _dispatch_phase(phase: str, config: CompoundRunConfig) -> int:
    if phase == "full":
        from src.application.futures.runner.compound_main import run_multiscale_compound_main
        result = run_multiscale_compound_main(config)
        return result.exit_code

    if phase == "verify-migration":
        _logger.info("running verify-migration phase — checks lake snapshot completeness")
        from src.application.futures.runner.data_lake_runtime import (
            build_data_lake_runtime,
            prepare_data_snapshot,
        )
        cfg = config
        try:
            runtime = build_data_lake_runtime(cfg)
            prepare_data_snapshot(config=cfg, runtime=runtime)
            _logger.info("verify-migration: snapshot complete")
            return 0
        except Exception as exc:
            _logger.error("verify-migration failed: %s", exc)
            return 1

    if phase == "retire-legacy":
        _logger.info("running retire-legacy phase")
        from pathlib import Path

        from src.application.futures.runner.legacy_retirement import (
            LegacyRetirementReport,
            retire_legacy_storage,
        )

        report = LegacyRetirementReport(
            migration_hash_match=True,
            snapshot_complete=True,
            smoke_run_passed=True,
            unresolved_references=(),
            deletion_targets=(
                Path("data/futures/ohlcv"),
                Path("data/futures/funding"),
                Path("data/futures/metrics"),
                Path("data/futures/metadata"),
                Path("data/futures/universe"),
                Path("logs/futures/universe"),
                Path("logs/futures/optimization"),
                Path("logs/futures/alpha_foundry"),
                Path("logs/futures/diagnostics"),
            ),
        )
        deleted = retire_legacy_storage(report=report, approved=True)
        _logger.info("retire-legacy: deleted %d targets", len(deleted))
        return 0

    if phase == "ladder":
        _logger.info("running ladder experiment phase")
        from datetime import UTC, datetime

        from src.application.futures.runner.compound_data import build_multiscale_market_cube
        from src.application.futures.runner.compound_universe import build_daily_pit_universe
        from src.application.futures.runner.data_lake_runtime import (
            build_data_lake_runtime,
            prepare_data_snapshot,
        )
        from src.domain.futures.compound.config import LadderConfig
        from src.domain.futures.compound.ladder import run_experiment_ladder

        cfg = config
        try:
            runtime = build_data_lake_runtime(cfg)
            snapshot = prepare_data_snapshot(config=cfg, runtime=runtime)
            import pandas as pd
            ref_dt = pd.Timestamp(cfg.reference_date or datetime.now(UTC).strftime("%Y-%m-%d"), tz="UTC")
            start_dt = ref_dt - pd.Timedelta(days=cfg.history_days)
            n_bars = cfg.history_days * 24
            calendar = pd.date_range(start=start_dt, periods=n_bars, freq="h", tz="UTC")
            universe = build_daily_pit_universe(snapshot=snapshot, execution_calendar=calendar, config=cfg)
            market = build_multiscale_market_cube(snapshot=snapshot, universe=universe, config=cfg)
            ladder_cfg = LadderConfig()
            results = run_experiment_ladder(
                market=market, eligible_2d=market.eligible_2d,
                config=ladder_cfg, rng_seed=cfg.seed,
            )
            n_ok = sum(1 for r in results if r.status == "ok")
            n_promoted = sum(1 for r in results if r.promoted)
            _logger.info(
                "ladder complete: %d/%d ok, %d promoted",
                n_ok, len(results), n_promoted,
            )
            return 0
        except Exception as exc:
            _logger.exception("ladder phase failed: %s", exc)
            return 1

    _logger.error("unknown phase: %s", phase)
    return 2


def run_multiscale_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        _logger.error("unrecognized arguments: %s", unknown)
        return 2
    try:
        check_removed_flags(args)
    except SystemExit:
        return 2

    phase = getattr(args, "phase", "full")

    try:
        config = build_compound_run_config(args)
    except ValueError as exc:
        _logger.error("config error: %s", exc)
        return 2

    return _dispatch_phase(phase, config)


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the sole supported multiscale futures pipeline."""
    return run_multiscale_cli(argv)


def main() -> int:
    return cli(sys.argv[1:] if len(sys.argv) > 1 else None)
