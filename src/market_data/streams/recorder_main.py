"""Dedicated process entrypoint for the always-on live-only market recorder."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from src.common.paths import LIVE_CAPTURE_DIR
from src.live.lifecycle import ShutdownFlag, install_shutdown_handlers
from src.market_data.streams.liquidations import default_liquidations_dir
from src.market_data.streams.recorder import MarketRecorderConfig, run_market_recorder


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
    asyncio.run(
        run_market_recorder(
            MarketRecorderConfig(),
            capture_root=root,
            liquidations_dir=liquidations,
            shutdown=flag,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.market_data.streams.recorder_main [--capture-root P] [--liquidations-dir P]``.

    Returns:
        0 after a clean shutdown.
    """
    parser = argparse.ArgumentParser(description="Always-on recorder of live-only market sources")
    parser.add_argument("--capture-root", type=str, default=None)
    parser.add_argument("--liquidations-dir", type=str, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    capture_root = Path(args.capture_root) if args.capture_root else None
    liquidations_dir = Path(args.liquidations_dir) if args.liquidations_dir else None
    run_recorder(capture_root=capture_root, liquidations_dir=liquidations_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
