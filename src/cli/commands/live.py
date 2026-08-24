"""``live shadow-cycle`` CLI: 환경에서 LiveSettings를 로드해 섬도우 사이클을 실행한다."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

logger = logging.getLogger("LiveCli")

_DEFAULT_ARTIFACT = (
    "docs/results/mhs_horizon_diagnostic_artifacts/deployed_target_weights.parquet"
)


def _parse_decision_time(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        raise argparse.ArgumentTypeError("--decision-time must be tz-aware (UTC)")
    return ts.tz_convert("UTC")


def _run_shadow_cycle(args: argparse.Namespace) -> None:
    from src.live.runner import run_shadow_cycle
    from src.live.settings import LiveSettings

    settings = LiveSettings()
    if args.dry_run:
        logger.info("[SYS] live shadow-cycle dry-run requested; mode=%s", settings.mode.value)
    report = run_shadow_cycle(
        settings,
        args.decision_time,
        args.artifact,
    )
    logger.info(
        "[SYS] live shadow-cycle status=%s reason=%s intents=%d",
        report.status,
        report.reason,
        report.intent_count,
    )


def add_live_commands(live_parser: argparse.ArgumentParser) -> None:
    """``live`` 커맨드 그룹에 shadow-cycle 서브커맨드를 등록한다."""
    subparsers = live_parser.add_subparsers(dest="live_command", required=True)

    shadow = subparsers.add_parser("shadow-cycle", help="Run one daily shadow decision cycle")
    shadow.add_argument(
        "--decision-time",
        type=_parse_decision_time,
        required=True,
        help="Decision time T as ISO8601 UTC (e.g. 2026-08-24T00:00:00Z)",
    )
    shadow.add_argument(
        "--artifact",
        type=str,
        default=_DEFAULT_ARTIFACT,
        help="Path to the deployed_target_weights.parquet artifact to consume",
    )
    shadow.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log the cycle without any state-changing intent beyond SHADOW suppression",
    )
    shadow.set_defaults(handler=_run_shadow_cycle, dry_run=False)
