"""Dedicated process entrypoint for the raw-first market normalizer."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.common.paths import BASE_DIR, LIVE_CAPTURE_DIR
from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers
from src.market_data.streams.liquidations import default_liquidations_dir
from src.market_data.streams.normalizer import NormalizerConfig, run_normalizer

NORMALIZER_LOG_DIR: Path = BASE_DIR / "logs" / "normalizer"
NORMALIZER_LOG_FILENAME: str = "normalizer.log"
NORMALIZER_LOG_MAX_BYTES: int = 10 * 1024 * 1024
NORMALIZER_LOG_BACKUP_COUNT: int = 5
BACKUP_STATUS_PATH: Path = Path("/app/backup_status/last_success.json")


def configure_normalizer_logging(log_dir: Path) -> Path | None:
    """stdout + size-rotated file under the mounted ``logs/`` tree (same contract as the retired recorder logger)."""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stream_handler: logging.Handler | None = next(
        (
            handler
            for handler in root.handlers
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler)
        ),
        None,
    )
    if stream_handler is None:
        stream_handler = logging.StreamHandler(sys.stdout)
        root.addHandler(stream_handler)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / NORMALIZER_LOG_FILENAME
        target = log_file.resolve()
        for existing in root.handlers:
            if isinstance(existing, RotatingFileHandler) and Path(
                str(getattr(existing, "baseFilename", ""))
            ).resolve() == target:
                return log_file
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=NORMALIZER_LOG_MAX_BYTES,
            backupCount=NORMALIZER_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        return log_file
    except Exception as exc:
        root.warning(
            "[SYS] stage=normalizer_log status=FILE_LOG_UNAVAILABLE path=%s error=%s",
            log_dir,
            exc,
        )
        return None


def run_normalizer_process(
    *,
    capture_root: Path = LIVE_CAPTURE_DIR,
    liquidations_dir: Path | None = None,
    backup_status_path: Path = BACKUP_STATUS_PATH,
) -> None:
    """Install SIGTERM/SIGINT handlers and run :func:`run_normalizer` until shutdown."""
    flag = ShutdownFlag()
    install_shutdown_handlers(flag)
    liquidations = liquidations_dir if liquidations_dir is not None else default_liquidations_dir()
    run_normalizer(
        Path(capture_root),
        liquidations,
        NormalizerConfig(),
        backup_status_path=Path(backup_status_path),
        shutdown=flag,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.market_data.streams.normalizer_main [--capture-root P] [--liquidations-dir P] [--backup-status P] [--log-dir P]``."""
    parser = argparse.ArgumentParser(description="Derive live-only raw capture into parquet, compact and prune")
    parser.add_argument("--capture-root", type=str, default=None)
    parser.add_argument("--liquidations-dir", type=str, default=None)
    parser.add_argument("--backup-status", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args(argv)
    configure_normalizer_logging(Path(args.log_dir) if args.log_dir else NORMALIZER_LOG_DIR)
    run_normalizer_process(
        capture_root=Path(args.capture_root) if args.capture_root else LIVE_CAPTURE_DIR,
        liquidations_dir=Path(args.liquidations_dir) if args.liquidations_dir else None,
        backup_status_path=Path(args.backup_status) if args.backup_status else BACKUP_STATUS_PATH,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
