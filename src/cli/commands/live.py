"""``live`` 커맨드 그룹: 섬도우 사이클 1회 실행 또는 24/7 무인 데몬 구동."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common.config import DATA_DIR

logger = logging.getLogger("LiveCli")

_DEFAULT_ARTIFACT = (
    "docs/results/mhs_horizon_diagnostic_artifacts/deployed_target_weights.parquet"
)
_DEFAULT_DAEMON_STATE_PATH = str(DATA_DIR / "state" / "live_daemon_last_run.json")


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


def _run_daemon(args: argparse.Namespace) -> None:
    from src.live.scheduler import run_daemon
    from src.live.settings import LiveSettings

    settings = LiveSettings()
    # 무한루프라 정상 반환하지 않는다(프로세스 시그널로 종료).
    run_daemon(settings, Path(args.artifact), Path(args.state_path))


def add_live_commands(live_parser: argparse.ArgumentParser) -> None:
    """``live`` 커맨드 그룹에 shadow-cycle/daemon 서브커맨드를 등록한다."""
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

    daemon = subparsers.add_parser("daemon", help="Run the 24/7 unattended shadow-cycle scheduler")
    daemon.add_argument(
        "--artifact",
        type=str,
        default=_DEFAULT_ARTIFACT,
        help="Path to the deployed_target_weights.parquet artifact to consume daily",
    )
    daemon.add_argument(
        "--state-path",
        type=str,
        default=_DEFAULT_DAEMON_STATE_PATH,
        help="Path to the daemon last-processed decision_time state JSON",
    )
    daemon.set_defaults(handler=_run_daemon)
