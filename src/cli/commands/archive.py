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


def _daily_quote_volume() -> object:
    import pandas as pd

    from src.common.paths import FUTURES_DATA_DIR
    from src.market_data.public_archive.catalog import is_perpetual_usdt
    from src.market_data.storage.ohlcv import is_temp_artifact

    lake = FUTURES_DATA_DIR / "ohlcv" / "1h"
    series: dict[str, pd.Series] = {}
    if lake.exists():
        for path in sorted(lake.glob("*.parquet")):
            if is_temp_artifact(path.name) or not is_perpetual_usdt(path.stem):
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[DATA] stage=archive_roster skipped path=%s detail=%s", path, exc)
                continue
            if "quote_vol" not in frame.columns:
                continue
            stamps = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            days = stamps.dt.date
            vols = pd.to_numeric(frame["quote_vol"], errors="coerce")
            series[path.stem] = vols.groupby(days.to_numpy()).sum(min_count=1)
    if not series:
        return pd.DataFrame()
    combined = pd.concat(series, axis=1)
    combined.index = pd.to_datetime(combined.index, utc=True)
    return combined.sort_index()


def _run_roster(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.market_data.public_archive import roster as roster_mod

    root = _root(args)
    frame = _daily_quote_volume()
    discovered = roster_mod.ever_liquid_symbols(
        frame, top_n=args.top_n, lookback_days=args.lookback_days
    )
    path = roster_mod.roster_path(root)
    existing = roster_mod.load_roster(path)
    before = set(existing.symbols) if existing is not None else set()
    merged = roster_mod.merge_roster(
        existing,
        discovered,
        as_of=pd.Timestamp.now(tz="UTC"),
        top_n=args.top_n,
        lookback_days=args.lookback_days,
    )
    added = sorted(set(merged.symbols) - before)
    print(f"roster symbols={len(merged.symbols)} version={merged.version} added={added}")  # noqa: T201
    if args.write:
        roster_mod.save_roster(path, merged)
        logger.info("[DATA] stage=archive_roster status=WRITTEN symbols=%d version=%d", len(merged.symbols), merged.version)
    elif added:
        logger.info("[DATA] stage=archive_roster status=DRY_RUN pending_additions=%d", len(added))


def _parse_day(raw: str | None, default: date) -> date:
    return date.fromisoformat(raw) if raw else default


def _run_collect(args: argparse.Namespace) -> None:
    from src.market_data.public_archive import roster as roster_mod
    from src.market_data.public_archive import service as service_mod
    from src.market_data.public_archive.catalog import DATASETS, is_perpetual_usdt

    root = _root(args)
    downloader = _downloader(args)
    requested_datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    end = _parse_day(args.end, datetime.now(UTC).date() - timedelta(days=1))
    start = date.fromisoformat(args.start) if args.start else None
    wanted = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    persisted = roster_mod.load_roster(roster_mod.roster_path(root))
    roster_symbols = persisted.symbols if persisted is not None else ()
    roster_set = set(roster_symbols)
    datasets = list(requested_datasets)
    if "aggTrades" in datasets and not roster_set:
        logger.warning("[DATA] stage=archive_collect status=EMPTY_ROSTER aggTrades skipped")
        print("warning: empty aggTrades roster; aggTrades skipped")  # noqa: T201
        datasets = [d for d in datasets if d != "aggTrades"]
    symbols: dict[str, list[str]] = {}
    for dataset in datasets:
        if dataset == "aggTrades":
            symbols[dataset] = sorted(roster_set if args.symbols == "all" else roster_set.intersection(wanted))
        elif args.symbols == "all":
            prefix = f"data/futures/um/daily/{dataset}/"
            if getattr(downloader, "list_all_symbols", None) is not None:
                listed = downloader.list_all_symbols(dataset_prefix=prefix)  # type: ignore[attr-defined]
            else:
                listed = downloader.list_symbols_from_s3_xml_listing(dataset_prefix=prefix)  # type: ignore[attr-defined]
            if not listed:
                print("archive collect status=LISTING_FAILED")  # noqa: T201
                raise SystemExit(2)
            symbols[dataset] = sorted({s for s in listed if is_perpetual_usdt(s)})
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
        slice_symbols=roster_symbols,
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
    """Register ``roster``, ``collect``, ``status`` and ``prune`` on the ``archive`` parser."""
    subparsers = archive_parser.add_subparsers(dest="archive_command", required=True)

    roster = subparsers.add_parser("roster", help="Rebuild the liquid aggTrades roster")
    roster.add_argument("--top-n", type=int, default=60)
    roster.add_argument("--lookback-days", type=int, default=30)
    roster.add_argument("--write", action="store_true", default=False)
    roster.add_argument("--root", type=str, default=None)
    roster.set_defaults(handler=_run_roster)

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
