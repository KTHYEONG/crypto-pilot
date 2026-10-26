"""``ops`` command group: operational provisioning for the VPS runtime."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.env_provisioning import build_runtime_fragment, install_runtime_fragment
from src.common.paths import VPS_MIRROR_DIR

if TYPE_CHECKING:
    from src.market_data.services.data_inventory import DatasetInventory

logger = logging.getLogger("OpsCli")


def _run_provision_env(args: argparse.Namespace) -> None:
    fragment = build_runtime_fragment(Path(args.source))
    key_count = len(fragment.splitlines())
    if args.dry_run:
        logger.info("[SYS] ops provision-env dry-run keys=%d host=%s", key_count, args.host)
        return
    install_runtime_fragment(args.host, fragment)
    logger.info("[SYS] ops provision-env installed keys=%d host=%s", key_count, args.host)


def _run_daemon_idle_gate(args: argparse.Namespace) -> None:
    from src.application.ops.daemon_idle_gate import main as gate_main

    argv = [
        "--heartbeat-file", str(args.heartbeat_file),
        "--waited-s", str(args.waited_s),
        "--max-wait-s", str(args.max_wait_s),
        "--stale-after-s", str(args.stale_after_s),
    ]
    if args.now is not None:
        argv += ["--now", str(args.now)]
    rc = gate_main(argv)
    if rc != 0:
        raise SystemExit(rc)


def _run_gdrive_cleanup(args: argparse.Namespace) -> None:
    from src.application.ops.gdrive_cleanup import main as cleanup_main

    argv: list[str] = []
    if args.apply:
        argv.append("--apply")
    if args.empty_trash:
        argv.append("--empty-trash")
    if args.i_understand_irreversible:
        argv.append("--i-understand-irreversible")
    rc = cleanup_main(argv)
    if rc != 0:
        raise SystemExit(rc)


def _run_artifact_seal(args: argparse.Namespace) -> None:
    from src.application.ops.artifact_seal import main as seal_main

    if args.artifact_command == "keygen":
        argv: list[str] = ["keygen"]
    else:
        argv = [str(args.artifact_command), "--in", str(args.input_path), "--out", str(args.output_path)]
    rc = seal_main(argv)
    if rc != 0:
        raise SystemExit(rc)


def _print_inventory(row: DatasetInventory) -> None:
    print(  # noqa: T201 - ops CLI contract output
        f"inventory name={row.name} first={row.first_day} last={row.last_day} "
        f"present={row.present_days} missing={len(row.missing_days)} "
        f"incomplete={len(row.incomplete_days)} bytes={row.bytes}"
    )


def _run_research_mirror(args: argparse.Namespace) -> None:
    from src.application.ops.research_mirror import MIRROR_DATASETS, run_mirror
    from src.market_data.services.data_inventory import build_inventory, write_inventory

    if not args.rclone or shutil.which(str(args.rclone)) is None:
        print("[SYS] stage=research_mirror status=failed error=rclone-not-found", file=sys.stderr)  # noqa: T201
        raise SystemExit(2)
    datasets = list(args.datasets) if args.datasets else list(MIRROR_DATASETS)
    dest = Path(args.dest)
    try:
        codes = run_mirror(
            datasets,
            dest_root=dest,
            dry_run=not args.execute,
            rclone=str(args.rclone),
        )
    except ValueError as exc:
        print(f"[SYS] stage=research_mirror status=failed error={exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(2) from exc
    if args.execute:
        rows = build_inventory(dest, today=datetime.now(UTC).date())
        write_inventory(rows, dest / "inventory.json")
        for row in rows:
            _print_inventory(row)
    if any(codes.values()):
        raise SystemExit(3)


def _run_data_inventory(args: argparse.Namespace) -> None:
    from src.market_data.services.data_inventory import build_inventory, write_inventory

    root = Path(args.root)
    rows = build_inventory(root, today=datetime.now(UTC).date())
    write_inventory(rows, root / "inventory.json")
    for row in rows:
        _print_inventory(row)


def add_ops_commands(parser: argparse.ArgumentParser) -> None:
    """Register the ``ops`` subcommands on the ``ops`` group parser."""
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
    gate = subparsers.add_parser("daemon-idle-gate", help="Wait for the live daemon to go idle")
    gate.add_argument("--heartbeat-file", type=str, required=True)
    gate.add_argument("--waited-s", type=float, required=True)
    gate.add_argument("--max-wait-s", type=float, default=3600.0)
    gate.add_argument("--stale-after-s", type=float, default=2700.0)
    gate.add_argument("--now", type=str, default=None)
    gate.set_defaults(handler=_run_daemon_idle_gate)
    cleanup = subparsers.add_parser("gdrive-cleanup", help="One-time evidence-gated Drive cleanup (dry-run by default)")
    cleanup.add_argument("--apply", action="store_true", default=False)
    cleanup.add_argument("--empty-trash", action="store_true", default=False)
    cleanup.add_argument("--i-understand-irreversible", action="store_true", default=False)
    cleanup.set_defaults(handler=_run_gdrive_cleanup)
    mirror = subparsers.add_parser(
        "research-mirror", help="Copy live-only VPS data from Drive into the local research mirror"
    )
    mirror.add_argument("--datasets", nargs="+", default=None, help="Logical dataset names (default: all)")
    mirror.add_argument("--dest", type=Path, default=VPS_MIRROR_DIR, help="Local mirror root")
    mirror.add_argument("--execute", action="store_true", default=False, help="Copy for real (default: dry-run)")
    mirror.add_argument("--rclone", default=shutil.which("rclone"), help="rclone binary")
    mirror.set_defaults(handler=_run_research_mirror)
    inventory = subparsers.add_parser("data-inventory", help="Report per-dataset day coverage of a mirror root")
    inventory.add_argument("--root", type=Path, default=VPS_MIRROR_DIR, help="Mirror root to scan")
    inventory.set_defaults(handler=_run_data_inventory)
    seal = subparsers.add_parser("artifact-seal", help="Seal deployed artifacts")
    seal_sub = seal.add_subparsers(dest="artifact_command", required=True)
    seal_sub.add_parser("keygen", help="emit a fresh base64 32-byte artifact key")
    seal_leaf = seal_sub.add_parser("seal", help="seal a plaintext file")
    seal_leaf.add_argument("--in", dest="input_path", type=Path, required=True)
    seal_leaf.add_argument("--out", dest="output_path", type=Path, required=True)
    unseal_leaf = seal_sub.add_parser("unseal", help="open a sealed file")
    unseal_leaf.add_argument("--in", dest="input_path", type=Path, required=True)
    unseal_leaf.add_argument("--out", dest="output_path", type=Path, required=True)
    seal.set_defaults(handler=_run_artifact_seal)
