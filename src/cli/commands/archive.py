"""``archive`` command group: mirror public Binance Vision archives into parquet."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

ARCHIVE_BACKUP_MOUNT: Path = Path("/app/backup_status/last_success.json")


def _root(args: argparse.Namespace) -> Path:
    override = getattr(args, "root", None)
    if override:
        return Path(override)
    from src.common.paths import PUBLIC_ARCHIVE_DIR

    return PUBLIC_ARCHIVE_DIR


def _downloader(args: argparse.Namespace) -> object:
    injected = getattr(args, "downloader", None)
    if injected is not None:
        return injected
    from src.market_data.binance.vision import BinanceVisionDownloader

    return BinanceVisionDownloader()


def _parse_day(raw: str | None, default: date) -> date:
    return date.fromisoformat(raw) if raw else default


def _bucket_symbols(downloader: object, dataset: str) -> list[str]:
    from src.market_data.public_archive.catalog import is_perpetual_usdt

    prefix = f"data/futures/um/daily/{dataset}/"
    try:
        if getattr(downloader, "list_all_symbols", None) is not None:
            listed = downloader.list_all_symbols(dataset_prefix=prefix)  # type: ignore[attr-defined]
        else:
            listed = downloader.list_symbols_from_s3_xml_listing(dataset_prefix=prefix)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.error("[DATA] stage=archive_collect listing_failed detail=%s", exc)
        return []
    if listed is None:
        return []
    return sorted({s for s in listed if is_perpetual_usdt(s)})


def _run_collect(args: argparse.Namespace) -> None:
    from src.market_data.public_archive import service as service_mod
    from src.market_data.public_archive.catalog import DATASETS

    root = _root(args)
    downloader = _downloader(args)
    requested_datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    end = _parse_day(args.end, datetime.now(UTC).date() - timedelta(days=1))
    start = date.fromisoformat(args.start) if args.start else None
    wanted = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    datasets = list(requested_datasets)
    symbols: dict[str, list[str]] = {}
    for dataset in datasets:
        if args.symbols == "all":
            listed = _bucket_symbols(downloader, dataset)
            if not listed:
                print("archive collect status=LISTING_FAILED")  # noqa: T201
                raise SystemExit(2)
            symbols[dataset] = listed
        else:
            symbols[dataset] = list(wanted)
    slice_hours = (int(args.slice_hours[0]), int(args.slice_hours[1]))
    incremental = 14 if args.start is None and os.environ.get("ARCHIVE_INCREMENTAL") == "1" else None
    plans = service_mod.plan_collection(
        downloader,
        root=root,
        datasets=datasets,
        symbols=symbols,
        start=start,
        end=end,
        slice_hours=slice_hours,
        slice_symbols=symbols.get("aggTrades", ()),
        incremental_window_days=incremental,
    )
    if plans is None:
        print("archive collect status=LISTING_FAILED")  # noqa: T201
        raise SystemExit(2)
    for dataset in datasets:
        ds_plans = [p for p in plans if p.dataset == dataset]
        ds_days = sum(len(p.days) for p in ds_plans)
        print(f"plan dataset={dataset} symbols={len(ds_plans)} days={ds_days} files={ds_days}")  # noqa: T201
    if not args.execute:
        return
    report = service_mod.execute_plan(downloader, plans, root=root, workers=args.workers)
    print(  # noqa: T201
        f"collect planned={report.planned_days} ok={report.ok} "
        f"failed={report.failed} absent={report.absent} pending={report.pending}"
    )
    if report.failed:
        raise SystemExit(3)


def _run_status(args: argparse.Namespace) -> None:
    from src.market_data.public_archive.catalog import DATASETS
    from src.market_data.public_archive.store import load_coverage

    root = _root(args)
    heartbeat_file = root / "_status" / "collect_heartbeat.json"
    heartbeat: dict[str, object] = {}
    try:
        heartbeat = json.loads(heartbeat_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        heartbeat = {}
    for dataset in DATASETS:
        coverage = load_coverage(root, dataset)
        ok_days = [(sym, row.day) for (sym, _), row in coverage.items() if row.status == "ok"]
        failed = sum(1 for row in coverage.items() if row[1].status == "failed")
        symbols = sorted({sym for sym, _ in ok_days})
        first = min((day for _, day in ok_days), default=None)
        last = max((day for _, day in ok_days), default=None)
        print(f"status dataset={dataset} symbols={len(symbols)} first={first} last={last} failed={failed}")  # noqa: T201
    if heartbeat:
        print(  # noqa: T201
            f"heartbeat ts={heartbeat.get('ts')} done={heartbeat.get('done')}/{heartbeat.get('total')} failed={heartbeat.get('failed')}"
        )
    else:
        print("heartbeat none")  # noqa: T201


def _run_prune(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.common.paths import BASE_DIR
    from src.market_data.public_archive.retention import prune_backed_up_archive
    from src.market_data.streams.retention import read_backup_status

    root = _root(args)
    if getattr(args, "backup_status", None):
        status_path = Path(args.backup_status)
    elif ARCHIVE_BACKUP_MOUNT.exists():
        status_path = ARCHIVE_BACKUP_MOUNT
    else:
        status_path = BASE_DIR / "deploy" / "backup" / "status" / "last_success.json"
    try:
        status = read_backup_status(status_path)
        report = prune_backed_up_archive(
            root, status=status, now=pd.Timestamp.now(tz="UTC"), keep_days=int(args.keep_days)
        )
    except OSError as exc:
        print(json.dumps({"error": type(exc).__name__}))  # noqa: T201
        raise SystemExit(4) from exc
    print(  # noqa: T201
        json.dumps(
            {
                "prune_blocked": report.prune_blocked,
                "blocked_reason": report.blocked_reason,
                "pruned_files": report.pruned_files,
                "kept_files": report.kept_files,
            },
            sort_keys=True,
        )
    )


def add_archive_commands(archive_parser: argparse.ArgumentParser) -> None:
    """Register ``collect``, ``status`` and ``prune`` on the ``archive`` parser."""
    subparsers = archive_parser.add_subparsers(dest="archive_command", required=True)

    collect = subparsers.add_parser("collect", help="Mirror Vision archives into compact parquet (dry-run by default)")
    collect.add_argument("--dataset", choices=["metrics", "bookDepth", "aggTrades", "all"], default="all")
    collect.add_argument("--start", type=str, default=None)
    collect.add_argument("--end", type=str, default=None)
    collect.add_argument("--symbols", type=str, default="all")
    collect.add_argument("--slice-hours", nargs=2, type=int, default=[23, 24])
    collect.add_argument("--workers", type=int, default=4)
    collect.add_argument("--execute", action="store_true", default=False)
    collect.add_argument("--root", type=str, default=None)
    collect.set_defaults(handler=_run_collect)

    status = subparsers.add_parser("status", help="Show archive coverage per dataset")
    status.add_argument("--root", type=str, default=None)
    status.set_defaults(handler=_run_status)

    prune = subparsers.add_parser("prune", help="Prune backup-gated month partitions")
    prune.add_argument("--keep-days", type=int, default=45)
    prune.add_argument("--backup-status", type=str, default=None)
    prune.add_argument("--root", type=str, default=None)
    prune.set_defaults(handler=_run_prune)
