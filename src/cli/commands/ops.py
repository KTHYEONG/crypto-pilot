"""``ops`` command group: operational provisioning for the VPS runtime."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common.env_provisioning import build_runtime_fragment, install_runtime_fragment

logger = logging.getLogger("OpsCli")


def _run_provision_env(args: argparse.Namespace) -> None:
    fragment = build_runtime_fragment(Path(args.source))
    key_count = len(fragment.splitlines())
    if args.dry_run:
        logger.info("[SYS] ops provision-env dry-run keys=%d host=%s", key_count, args.host)
        return
    install_runtime_fragment(args.host, fragment)
    logger.info("[SYS] ops provision-env installed keys=%d host=%s", key_count, args.host)


def add_ops_commands(parser: argparse.ArgumentParser) -> None:
    """Register the ``provision-env`` subcommand on the ``ops`` group parser."""
    subparsers = parser.add_subparsers(dest="ops_command", required=True)
    provision = subparsers.add_parser("provision-env", help="Provision VPS runtime secrets from the workstation SSOT")
    provision.add_argument("--host", type=str, default="or-vps", help="SSH host for the VPS runtime")
    provision.add_argument(
        "--source",
        type=str,
        default=str(Path.home() / ".quant.env"),
        help="Workstation source file holding the secret SSOT",
    )
    provision.add_argument("--dry-run", action="store_true", default=False, help="Build the fragment without installing")
    provision.set_defaults(handler=_run_provision_env)
