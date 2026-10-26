"""Read-only, incremental Drive-to-local mirror of live-only VPS data."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from logging import getLogger
from pathlib import Path

from src.application.ops.gdrive_cleanup import DRIVE_REMOTE

logger = getLogger("ResearchMirror")

MIRROR_REMOTE_ROOT: str = f"{DRIVE_REMOTE}/live/crypto-pilot/data"
# Must equal deploy/backup/crypto-pilot-backup.sh REMOTE_ROOT plus "/data".

MIRROR_DATASETS: Mapping[str, str] = {
    "book_ticker": "live_capture/book_ticker",
    "premium_index": "live_capture/premium_index",
    "exec_depth": "live_capture/exec_depth",
    "raw_archive": "live_capture/raw/archive",
    "reference": "live_capture/reference",
    "capture_coverage": "live_capture/coverage",
    "liquidations": "futures/liquidations",
    "venue_rules": "futures/venue_rules",
    "state_runs": "state/runs",
    "public_archive": "public_archive",
}

MIRROR_RCLONE_FLAGS: tuple[str, ...] = (
    "--fast-list",
    "--transfers",
    "4",
    "--checkers",
    "8",
    "--stats-one-line",
    "--stats",
    "60s",
)


def build_mirror_commands(
    datasets: Sequence[str], *, rclone: str, dest_root: Path, dry_run: bool
) -> list[list[str]]:
    """argv lists for one ``rclone copy <remote>/<rel> <dest_root>/<rel>`` per dataset; raises ValueError on an unknown dataset or a destination outside dest_root."""
    resolved_root = dest_root.resolve()
    commands: list[list[str]] = []
    for name in dict.fromkeys(datasets):
        rel = MIRROR_DATASETS.get(name)
        if rel is None:
            valid = ", ".join(MIRROR_DATASETS)
            raise ValueError(f"unknown dataset {name!r}; valid datasets: {valid}")
        destination = (dest_root / rel).resolve()
        if destination == resolved_root or not destination.is_relative_to(resolved_root):
            raise ValueError(f"dataset {name!r} resolves outside destination root: {destination}")
        argv = [
            rclone,
            "copy",
            f"{MIRROR_REMOTE_ROOT}/{rel}",
            str(destination),
            *MIRROR_RCLONE_FLAGS,
        ]
        if dry_run:
            argv.append("--dry-run")
        commands.append(argv)
    return commands


def run_mirror(
    datasets: Sequence[str],
    *,
    dest_root: Path,
    dry_run: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    rclone: str = "rclone",
) -> dict[str, int]:
    """Execute the mirror commands sequentially and return rclone exit codes per dataset."""
    commands = build_mirror_commands(datasets, rclone=rclone, dest_root=dest_root, dry_run=dry_run)
    results: dict[str, int] = {}
    for name, argv in zip(dict.fromkeys(datasets), commands, strict=True):
        started = time.monotonic()
        logger.info("[DATA] stage=research_mirror dataset=%s status=start rc=%d elapsed_s=%.1f", name, 0, 0.0)
        try:
            completed = runner(argv, check=False)
            rc = completed.returncode
        except OSError as exc:
            rc = 127 if isinstance(exc, FileNotFoundError) else 126
            logger.error("[SYS] stage=research_mirror dataset=%s status=spawn_failed errno=%s", name, exc.errno)
        elapsed = time.monotonic() - started
        results[name] = rc
        logger.info(
            "[DATA] stage=research_mirror dataset=%s status=done rc=%d elapsed_s=%.1f",
            name,
            rc,
            elapsed,
        )
    return results
