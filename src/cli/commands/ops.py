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


def _run_backtests_migrate(args: argparse.Namespace) -> None:
    from src.backtests.migration import migrate_legacy_backtests

    registry_path = Path(args.registry_path)
    histories = tuple(Path(item) for item in args.history_directory)
    runs = tuple(Path(item) for item in args.run_directory)
    report = migrate_legacy_backtests(
        registry_path=registry_path,
        history_directories=histories,
        run_directories=runs,
        dry_run=not args.apply,
    )
    logger.info(
        "[SYS] ops backtests-migrate imported=%s skipped=%s protected=%s rejected=%s",
        report["imported"],
        report["skipped"],
        report["protected"],
        report["rejected"],
    )


def _run_backtests_verify_history_migration(args: argparse.Namespace) -> None:
    from src.backtests.migration import verify_legacy_history_migration
    from src.common.errors import DataIntegrityError

    try:
        report = verify_legacy_history_migration(
            registry_path=Path(args.registry_path), source=Path(args.history_directory)
        )
    except DataIntegrityError as exc:
        logger.error("[SYS] ops backtests-verify-history-migration failed error=%s", exc)
        raise SystemExit(1) from exc
    logger.info(
        "[SYS] ops backtests-verify-history-migration verified source=%s records=%s trials=%s",
        report["source"],
        report["source_records"],
        report["distinct_trial_identities"],
    )


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


def add_ops_commands(parser: argparse.ArgumentParser) -> None:
    """Register the ``provision-env`` subcommand on the ``ops`` group parser."""
    subparsers = parser.add_subparsers(dest="ops_command", required=True)
    migrate = subparsers.add_parser("backtests-migrate", help="Import explicitly selected legacy backtest evidence")
    migrate.add_argument("--registry-path", type=str, required=True, help="Target registry SQLite path")
    migrate.add_argument("--history-directory", type=str, action="append", default=[], help="Explicit legacy history directory")
    migrate.add_argument("--run-directory", type=str, action="append", default=[], help="Explicit legacy run directory")
    migrate.add_argument("--apply", action="store_true", default=False, help="Write the registry instead of previewing")
    migrate.set_defaults(handler=_run_backtests_migrate)
    verify = subparsers.add_parser(
        "backtests-verify-history-migration", help="Verify a legacy history source is durably represented by the registry"
    )
    verify.add_argument("--registry-path", type=str, required=True, help="Target registry SQLite path")
    verify.add_argument("--history-directory", type=str, required=True, help="Explicit legacy history directory")
    verify.set_defaults(handler=_run_backtests_verify_history_migration)
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
