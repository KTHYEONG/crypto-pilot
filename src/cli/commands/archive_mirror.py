from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.market_data.mirror.build import BuildReport
    from src.market_data.mirror.config import MirrorConfig

_BUSY_EXIT = 75
_CAPACITY_EXIT = 4
_ANOMALY_EXIT = 5
_FAILED_EXIT = 3
_STALE_EXIT = 6
_LOCK_WAIT_S = 60.0


def _root(args: argparse.Namespace) -> Path:
    if args.root:
        return Path(args.root)
    from src.common.paths import ARCHIVE_MIRROR_DIR

    return ARCHIVE_MIRROR_DIR


def _config(args: argparse.Namespace) -> MirrorConfig:
    from src.market_data.mirror.config import MirrorConfig

    overrides: dict[str, Any] = {}
    if args.datasets:
        overrides["datasets"] = tuple(args.datasets)
    for field in ("drive_budget_gib", "drive_free_floor_gib", "outbox_cap_gib", "download_workers"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    return MirrorConfig(root=_root(args), **overrides)


def _parse_deadline(text: str, now: datetime) -> datetime:
    hour, minute = (int(part) for part in text.split(":", 1))
    return datetime.combine(now.date(), time(hour, minute), tzinfo=UTC)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))  # noqa: T201


def _busy(command: str) -> None:
    print(f"archive mirror {command} status=BUSY")  # noqa: T201
    raise SystemExit(_BUSY_EXIT)


def _run_build(args: argparse.Namespace) -> None:
    from src.market_data.binance.vision import BinanceVisionDownloader
    from src.market_data.mirror.build import BuildReport, build_chunk, plan_datasets
    from src.market_data.public_archive import store as store_mod

    now = datetime.now(UTC)
    config = _config(args)
    deadline = _parse_deadline(args.deadline, now) if args.deadline else None
    if deadline is not None and now >= deadline:
        report = BuildReport(0, 0, 0, 0, 0, 0, True, None, (), ())
        _print_json(dataclasses.asdict(report))
        return
    downloader = BinanceVisionDownloader()
    try:
        with store_mod.archive_lock(config.root, wait_s=_LOCK_WAIT_S, holder="mirror-build"):
            if not args.execute:
                planned = plan_datasets(downloader, config, now=now)
                datasets = {
                    name: {
                        "listing_ok": plan.listing_ok,
                        "pending_sources": sum(len(p.sources) for p in plan.partitions),
                        "partitions": len(plan.partitions),
                        "pending_bytes": sum(s.size for p in plan.partitions for s in p.sources),
                    }
                    for name, plan in planned.plans.items()
                }
                _print_json({"mode": "plan", "datasets": datasets, "listing_failed": list(planned.listing_failed)})
                return
            if args.remote_bytes is None or args.drive_free_bytes is None:
                raise SystemExit("build --execute requires --remote-bytes and --drive-free-bytes")
            report = build_chunk(
                downloader,
                config,
                remote_bytes=args.remote_bytes,
                drive_free_bytes=args.drive_free_bytes,
                now=now,
                deadline=deadline,
            )
    except store_mod.ArchiveBusyError:
        _busy("build")
    _print_json(dataclasses.asdict(report))
    raise SystemExit(_build_exit_code(report))


def _build_exit_code(report: BuildReport) -> int:
    if report.capacity_blocked is not None:
        return _CAPACITY_EXIT
    if report.anomalies:
        return _ANOMALY_EXIT
    if report.failed_sources or report.listing_failed:
        return _FAILED_EXIT
    return 0


def _run_confirm(args: argparse.Namespace) -> None:
    from src.market_data.mirror.confirm import confirm_uploads, parse_remote_listing
    from src.market_data.public_archive import store as store_mod

    root = _root(args)
    remote = parse_remote_listing(Path(args.remote_listing).read_text(encoding="utf-8"))
    try:
        with store_mod.archive_lock(root, wait_s=_LOCK_WAIT_S, holder="mirror-confirm"):
            report = confirm_uploads(root, remote, now=datetime.now(UTC))
    except store_mod.ArchiveBusyError:
        _busy("confirm")
    _print_json(dataclasses.asdict(report))
    if report.unconfirmed:
        raise SystemExit(_FAILED_EXIT)


def _outbox_nonempty(root: Path) -> bool:
    objects = root / "outbox" / "objects"
    return objects.is_dir() and any(path.is_file() for path in objects.rglob("*"))


def _run_status(args: argparse.Namespace) -> None:
    from src.market_data.mirror.status import read_status

    root = _root(args)
    now = datetime.now(UTC)
    status = read_status(root / "_status")
    pending = sum(int(v) for v in dict(status.get("pending_sources", {})).values())
    outbox = _outbox_nonempty(root)
    payload = {**status, "pending_sources_total": pending, "outbox_nonempty": outbox}
    _print_json(payload)
    if args.max_confirm_age_hours is None or not (pending > 0 or outbox):
        return
    limit = timedelta(hours=args.max_confirm_age_hours)
    last_confirmed = status.get("last_confirmed_at")
    if last_confirmed is not None:
        stale = now - datetime.fromisoformat(str(last_confirmed)) > limit
    else:
        first_build = status.get("first_build_at")
        stale = first_build is not None and now - datetime.fromisoformat(str(first_build)) > limit
    if stale:
        raise SystemExit(_STALE_EXIT)


def _run_restore(args: argparse.Namespace) -> None:
    from src.market_data.mirror.confirm import restore_source

    restored = restore_source(Path(args.sidecar), Path(args.object), args.key, Path(args.out), etag=args.etag)
    _print_json({"path": str(restored.path), "sha256": restored.sha256})


def _run_format_doc(_: argparse.Namespace) -> None:
    text = (Path(__file__).resolve().parents[2] / "market_data" / "mirror" / "FORMAT.md").read_text(encoding="utf-8")
    print(text, end="")  # noqa: T201


def add_archive_mirror_commands(mirror_parser: argparse.ArgumentParser) -> None:
    """Register the ``archive mirror`` subcommands: build, confirm, status, restore and format-doc."""
    from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS

    subparsers = mirror_parser.add_subparsers(dest="mirror_command", required=True)
    known = tuple(MIRROR_DATASET_SPECS)

    build = subparsers.add_parser("build", help="Plan and build mirror objects into the outbox (dry-run by default)")
    build.add_argument("--root", type=str, default=None)
    build.add_argument("--datasets", nargs="+", choices=known, default=None)
    build.add_argument("--execute", action="store_true", default=False)
    build.add_argument("--remote-bytes", type=int, default=None)
    build.add_argument("--drive-free-bytes", type=int, default=None)
    build.add_argument("--deadline", type=str, default=None, help="HH:MM UTC today; no work when already passed")
    build.add_argument("--drive-budget-gib", type=int, default=None)
    build.add_argument("--drive-free-floor-gib", type=int, default=None)
    build.add_argument("--outbox-cap-gib", type=int, default=None)
    build.add_argument("--download-workers", type=int, default=None)
    build.set_defaults(handler=_run_build)

    confirm = subparsers.add_parser("confirm", help="Commit outbox pairs confirmed by a remote listing")
    confirm.add_argument("--root", type=str, default=None)
    confirm.add_argument("--remote-listing", type=str, required=True)
    confirm.set_defaults(handler=_run_confirm)

    status = subparsers.add_parser("status", help="Print mirror status JSON")
    status.add_argument("--root", type=str, default=None)
    status.add_argument("--max-confirm-age-hours", type=int, default=None)
    status.set_defaults(handler=_run_status)

    restore = subparsers.add_parser("restore", help="Rebuild one source from its object and sidecar")
    restore.add_argument("--root", type=str, default=None)
    restore.add_argument("--sidecar", type=str, required=True)
    restore.add_argument("--object", type=str, required=True)
    restore.add_argument("--key", type=str, required=True)
    restore.add_argument("--etag", type=str, default=None)
    restore.add_argument("--out", type=str, required=True)
    restore.set_defaults(handler=_run_restore)

    format_doc = subparsers.add_parser("format-doc", help="Print the mirror object format description")
    format_doc.set_defaults(handler=_run_format_doc)
