"""Dedicated process entrypoint for the always-on live-only market recorder."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pydantic import SecretStr

from src.common.paths import BASE_DIR, LIVE_CAPTURE_DIR
from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers
from src.market_data.streams.liquidations import default_liquidations_dir
from src.market_data.streams.recorder import MarketRecorderConfig, run_market_recorder

RECORDER_LOG_DIR: Path = BASE_DIR / "logs" / "recorder"
RECORDER_LOG_FILENAME: str = "recorder.log"
RECORDER_LOG_MAX_BYTES: int = 10 * 1024 * 1024
RECORDER_LOG_BACKUP_COUNT: int = 5


def configure_recorder_logging(log_dir: Path) -> Path | None:
    """Send root logging to stdout and to a size-rotated file that survives container recreation.

    Docker's json-file log is discarded whenever the container is recreated (every deploy that
    changes the recorder), which erased the evidence of past capture outages; the rotating file under
    the mounted ``logs/`` tree keeps it. Failure to create the directory or open the file never blocks
    capture: the recorder continues with stdout only and logs a warning. Idempotent: a second call with
    the same directory does not add a second file handler.

    Args:
        log_dir: Directory of ``recorder.log`` (created if missing).

    Returns:
        The log file path, or ``None`` when the file handler could not be attached.
    """
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
        log_file = log_dir / RECORDER_LOG_FILENAME
        target = log_file.resolve()
        for existing in root.handlers:
            if isinstance(existing, RotatingFileHandler) and Path(
                str(getattr(existing, "baseFilename", ""))
            ).resolve() == target:
                return log_file
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=RECORDER_LOG_MAX_BYTES,
            backupCount=RECORDER_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        return log_file
    except Exception as exc:
        root.warning(
            "[SYS] stage=recorder_log status=FILE_LOG_UNAVAILABLE path=%s error=%s",
            log_dir,
            exc,
        )
        return None


def run_recorder(
    *,
    capture_root: Path | None = None,
    liquidations_dir: Path | None = None,
) -> None:
    """Run the always-on live-only market recorder until SIGTERM/SIGINT.

    Dedicated process entrypoint: it must not import ``src.cli`` so the recorder's source closure
    (and therefore its deploy fingerprint) excludes unrelated CLI and strategy modules.

    Args:
        capture_root: defaults to ``LIVE_CAPTURE_DIR``.
        liquidations_dir: defaults to ``default_liquidations_dir()``.
    """
    flag = ShutdownFlag()
    install_shutdown_handlers(flag)
    root = capture_root if capture_root is not None else LIVE_CAPTURE_DIR
    liquidations = liquidations_dir if liquidations_dir is not None else default_liquidations_dir()
    recorder_ping_raw = os.environ.get("LIVE_RECORDER_DEADMAN_PING_URL")
    recorder_ping = SecretStr(recorder_ping_raw) if recorder_ping_raw else None
    asyncio.run(
        run_market_recorder(
            MarketRecorderConfig(deadman_ping_url=recorder_ping),
            capture_root=root,
            liquidations_dir=liquidations,
            shutdown=flag,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.market_data.streams.recorder_main [--capture-root P] [--liquidations-dir P] [--log-dir P]``.

    Returns:
        0 after a clean shutdown.
    """
    parser = argparse.ArgumentParser(description="Always-on recorder of live-only market sources")
    parser.add_argument("--capture-root", type=str, default=None)
    parser.add_argument("--liquidations-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args(argv)
    configure_recorder_logging(Path(args.log_dir) if args.log_dir else RECORDER_LOG_DIR)
    capture_root = Path(args.capture_root) if args.capture_root else None
    liquidations_dir = Path(args.liquidations_dir) if args.liquidations_dir else None
    run_recorder(capture_root=capture_root, liquidations_dir=liquidations_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
