"""``archive`` command group: mirror public Binance Vision archives into parquet."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.market_data.public_archive.jobs import BackfillJob
    from src.market_data.public_archive.mask import CensusSpan
    from src.market_data.public_archive.service import CollectionPlan

logger = logging.getLogger(__name__)

ARCHIVE_BACKUP_MOUNT: Path = Path("/app/backup_status/last_archive_success.json")


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


def _masked_universe(
    args: argparse.Namespace,
    downloader: object,
    datasets: list[str],
    start: date | None,
    end: date,
) -> tuple[dict[str, list[str]], dict[str, dict[str, frozenset[date]]], CensusSpan]:
    """Symbols and allowed days per dataset from the PIT seat-day mask; exits 2 without a lake."""
    from src.common.errors import DataIntegrityError
    from src.common.paths import FUTURES_DATA_DIR
    from src.market_data.public_archive.catalog import ARCHIVE_RETRY_HORIZON_DAYS, MASK_POLICIES
    from src.market_data.public_archive.mask import (
        CensusTooShortError,
        allowed_days,
        census_span,
        load_daily_census,
        require_census_coverage,
        seat_day_mask,
    )

    census_override = getattr(args, "census_root", None)
    census_root = Path(str(census_override)) if census_override else (FUTURES_DATA_DIR / "ohlcv" / "1h")
    try:
        daily_close, daily_quote_volume = load_daily_census(census_root)
        span = census_span(daily_close, root=census_root)
    except DataIntegrityError as exc:
        print("archive collect status=LAKE_MISSING")  # noqa: T201
        raise SystemExit(2) from exc
    incremental = start is None and os.environ.get("ARCHIVE_INCREMENTAL") == "1"
    first_days: dict[str, date] = {}
    for dataset in datasets:
        policy = MASK_POLICIES[dataset]
        if start is not None:
            first = start
        elif incremental:
            first = end - timedelta(days=ARCHIVE_RETRY_HORIZON_DAYS)
        else:
            first = policy.start
        first_days[dataset] = first
        try:
            require_census_coverage(span, first=first, last=end)
        except CensusTooShortError as exc:
            print(  # noqa: T201
                f"archive collect status=CENSUS_SHORT census_root={span.root} "
                f"census_first={span.first_day} census_last={span.last_day} "
                f"required_first={exc.required_first} required_last={exc.required_last}"
            )
            raise SystemExit(2) from exc
    symbols: dict[str, list[str]] = {}
    allowed: dict[str, dict[str, frozenset[date]]] = {}
    for dataset in datasets:
        policy = MASK_POLICIES[dataset]
        breadth = args.breadth if getattr(args, "breadth", None) is not None else policy.breadth
        preroll = args.preroll_days if getattr(args, "preroll_days", None) is not None else policy.preroll_days
        mask = seat_day_mask(daily_close, daily_quote_volume, breadth=breadth, preroll_days=preroll)
        by_symbol = allowed_days(mask, first=first_days[dataset], last=end)
        if not by_symbol:
            symbols[dataset], allowed[dataset] = [], {}
            continue
        listed = set(_bucket_symbols(downloader, dataset))
        if not listed:
            print("archive collect status=LISTING_FAILED")  # noqa: T201
            raise SystemExit(2)
        symbols[dataset] = sorted(s for s in by_symbol if s in listed)
        allowed[dataset] = {s: by_symbol[s] for s in symbols[dataset]}
    return symbols, allowed, span


def _plain_universe(args: argparse.Namespace, downloader: object, datasets: list[str]) -> dict[str, list[str]]:
    """Every bucket symbol (``--no-mask``) or the explicit ``--symbols`` list, for each dataset."""
    if args.symbols != "all":
        wanted = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
        return {dataset: list(wanted) for dataset in datasets}
    symbols: dict[str, list[str]] = {}
    for dataset in datasets:
        listed = _bucket_symbols(downloader, dataset)
        if not listed:
            print("archive collect status=LISTING_FAILED")  # noqa: T201
            raise SystemExit(2)
        symbols[dataset] = listed
    return symbols


def _print_plan(
    root: Path,
    datasets: list[str],
    plans: list[CollectionPlan],
    allowed: dict[str, dict[str, frozenset[date]]] | None,
    census_span: CensusSpan | None = None,
) -> None:
    from src.market_data.public_archive.store import load_coverage

    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    for dataset in datasets:
        ds_plans = [p for p in plans if p.dataset == dataset]
        ds_days = sum(len(p.days) for p in ds_plans)
        ds_sealed = sum(len(p.sealed_days) for p in ds_plans)
        line = f"plan dataset={dataset} symbols={len(ds_plans)} days={ds_days} files={ds_days} sealed_days={ds_sealed}"
        if allowed is not None and dataset in allowed:
            coverage = load_coverage(root, dataset)
            amap = allowed[dataset]
            missing = sum(
                1
                for sym, days in amap.items()
                for day in days
                if day <= yesterday and (coverage.get((sym, day)) is None or coverage[(sym, day)].status != "ok")
            )
            line += (
                f" mask_symbols={len(amap)} allowed_days={sum(len(v) for v in amap.values())} missing_days={missing}"
            )
            if census_span is not None:
                line += f" census_first={census_span.first_day} census_last={census_span.last_day}"
        print(line)  # noqa: T201


def _parse_hhmm(raw: str) -> tuple[int, int]:
    try:
        hh_str, mm_str = raw.split(":", 1)
        hh, mm = int(hh_str), int(mm_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid HH:MM: {raw!r}") from exc
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"invalid HH:MM: {raw!r}")
    return hh, mm


def _drain_deadline(hhmm: str, now: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """Today's HH:MM UTC, or None once it has passed: a late run must never drain into the next day."""
    import pandas as pd

    hh, mm = _parse_hhmm(hhmm)
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    deadline = pd.Timestamp(now.year, now.month, now.day, hh, mm, tz="UTC")
    return deadline if deadline > now else None


def _job_allowed(job: BackfillJob) -> dict[str, frozenset[date]]:
    from src.market_data.public_archive.catalog import MASK_POLICIES
    from src.market_data.public_archive.mask import (
        allowed_days,
        census_span,
        load_daily_census,
        require_census_coverage,
        seat_day_mask,
    )

    close, volume = load_daily_census(Path(job.census_root))
    require_census_coverage(census_span(close, root=Path(job.census_root)), first=job.start, last=job.end)
    policy = MASK_POLICIES[job.dataset]
    mask = seat_day_mask(
        close,
        volume,
        breadth=job.breadth if job.breadth is not None else policy.breadth,
        preroll_days=job.preroll_days if job.preroll_days is not None else policy.preroll_days,
    )
    return allowed_days(mask, first=job.start, last=job.end)


def _job_remaining(root: Path, job: BackfillJob) -> int:
    from src.market_data.public_archive.store import load_coverage

    coverage = load_coverage(root, job.dataset)
    return sum(
        1
        for symbol, days in _job_allowed(job).items()
        for day in days
        if (symbol, day) not in coverage or coverage[symbol, day].status != "ok"
    )


def _run_backfill_add(args: argparse.Namespace) -> None:
    from src.common.errors import DataIntegrityError
    from src.market_data.public_archive import jobs as jobs_mod
    from src.market_data.public_archive import store as store_mod
    from src.market_data.public_archive.catalog import DATASETS
    from src.market_data.public_archive.mask import census_span, load_daily_census, require_census_coverage

    root = _root(args)
    dataset = str(args.dataset)
    if dataset not in DATASETS:
        print(f"archive backfill add status=UNKNOWN_DATASET dataset={dataset}")  # noqa: T201
        raise SystemExit(2)
    try:
        with store_mod.archive_lock(root, wait_s=60.0, holder="backfill-add"):
            census_root = Path(str(args.census_root))
            try:
                daily_close, _daily_qv = load_daily_census(census_root)
                span = census_span(daily_close, root=census_root)
            except DataIntegrityError as exc:
                print("archive backfill add status=LAKE_MISSING")  # noqa: T201
                raise SystemExit(2) from exc
            end = date.fromisoformat(args.end) if getattr(args, "end", None) else span.last_day
            start = date.fromisoformat(args.start)
            try:
                require_census_coverage(span, first=start, last=end)
            except Exception as exc:
                from src.market_data.public_archive.mask import CensusTooShortError

                if isinstance(exc, CensusTooShortError):
                    print(  # noqa: T201
                        f"archive backfill add status=CENSUS_SHORT census_root={span.root} "
                        f"census_first={span.first_day} census_last={span.last_day} "
                        f"required_first={exc.required_first} required_last={exc.required_last}"
                    )
                else:
                    print(f"archive backfill add status=CENSUS_SHORT detail={exc}")  # noqa: T201
                raise SystemExit(2) from exc
            import pandas as pd

            created = pd.Timestamp.now(tz="UTC")
            job_id = f"{dataset}-{start.isoformat()}-{end.isoformat()}-{created.strftime('%Y%m%dT%H%M%SZ')}"
            job = jobs_mod.BackfillJob(
                job_id=job_id,
                dataset=dataset,
                start=start,
                end=end,
                census_root=str(census_root),
                breadth=getattr(args, "breadth", None),
                preroll_days=getattr(args, "preroll_days", None),
                created_at=created,
            )
            try:
                jobs_mod.enqueue_backfill(root, job)
            except ValueError as exc:
                print(f"archive backfill add status=OVERLAP detail={exc}")  # noqa: T201
                raise SystemExit(1) from exc
            print(job_id)  # noqa: T201
    except store_mod.ArchiveBusyError as exc:
        print(f"archive backfill add status=BUSY holder={exc.holder or 'null'}")  # noqa: T201
        raise SystemExit(75) from None


def _run_backfill_list(args: argparse.Namespace) -> None:
    from src.market_data.public_archive import jobs as jobs_mod
    from src.market_data.public_archive import store as store_mod

    root = _root(args)
    try:
        with store_mod.archive_lock(root, wait_s=60.0, holder="backfill-list"):
            for state in ("pending", "done", "cancelled"):
                if state == "pending":
                    items = jobs_mod.pending_jobs(root)
                elif state == "done":
                    items = jobs_mod.done_jobs(root)
                else:
                    items = jobs_mod.cancelled_jobs(root)
                for job in items:
                    remaining = _job_remaining(root, job)
                    coverage = store_mod.load_coverage(root, job.dataset)
                    failed = sum(
                        row.status == "failed" and job.start <= row.day <= job.end for row in coverage.values()
                    )
                    print(  # noqa: T201
                        f"backfill {state} job={job.job_id} dataset={job.dataset} "
                        f"start={job.start.isoformat()} end={job.end.isoformat()} remaining={remaining} failed={failed} "
                        f"status={state.upper()}"
                    )
    except store_mod.ArchiveBusyError as exc:
        print(f"archive backfill list status=BUSY holder={exc.holder or 'null'}")  # noqa: T201
        raise SystemExit(75) from None


def _run_backfill_cancel(args: argparse.Namespace) -> None:
    from src.market_data.public_archive import jobs as jobs_mod
    from src.market_data.public_archive import store as store_mod

    root = _root(args)
    job_id = str(args.job_id)
    try:
        with store_mod.archive_lock(root, wait_s=60.0, holder="backfill-cancel"):
            try:
                jobs_mod.cancel_job(root, job_id)
            except FileNotFoundError as exc:
                print(f"archive backfill cancel status=NOT_FOUND job={job_id}")  # noqa: T201
                raise SystemExit(1) from exc
            print(f"backfill cancelled job={job_id}")  # noqa: T201
    except store_mod.ArchiveBusyError as exc:
        print(f"archive backfill cancel status=BUSY holder={exc.holder or 'null'}")  # noqa: T201
        raise SystemExit(75) from None


def _run_import(args: argparse.Namespace) -> None:
    from src.market_data.public_archive import store as store_mod
    from src.market_data.public_archive.catalog import DATASETS, DEFAULT_DATASETS
    from src.market_data.public_archive.importer import import_archive

    root = _root(args)
    source = Path(args.from_dir)
    raw_datasets = args.dataset
    if not raw_datasets:
        datasets = list(DEFAULT_DATASETS)
    else:
        datasets = list(raw_datasets)
        for dataset in datasets:
            if dataset not in DATASETS:
                print(f"archive import status=UNKNOWN_DATASET dataset={dataset}")  # noqa: T201
                raise SystemExit(2)
    try:
        with store_mod.archive_lock(root, wait_s=float(getattr(args, "lock_wait_s", 1800.0)), holder="import"):
            report = import_archive(source, root, datasets=datasets)
    except store_mod.ArchiveBusyError as exc:
        print(f"archive import status=BUSY holder={exc.holder or 'null'}")  # noqa: T201
        raise SystemExit(75) from None
    print(  # noqa: T201
        json.dumps(
            {
                "days_recorded": report.days_recorded,
                "months_copied": report.months_copied,
                "months_merged": report.months_merged,
                "months_skipped_conflict": report.months_skipped_conflict,
                "months_skipped_sealed": report.months_skipped_sealed,
            },
            sort_keys=True,
        )
    )


def _drain_job(root: Path, downloader: object, job: BackfillJob, deadline: pd.Timestamp, workers: int) -> None:
    from src.market_data.public_archive import jobs, service
    from src.market_data.public_archive.catalog import TAPE_SLICE_HOURS

    amap = _job_allowed(job)
    listed = set(_bucket_symbols(downloader, job.dataset)) if amap else set()
    if amap and not listed:
        raise RuntimeError("backfill bucket listing failed")
    symbols = sorted(set(amap) & listed)
    plans = service.plan_collection(
        downloader,
        root=root,
        datasets=[job.dataset],
        symbols={job.dataset: symbols},
        start=job.start,
        end=job.end,
        slice_hours=TAPE_SLICE_HOURS,
        slice_symbols=symbols if job.dataset == "aggTrades" else (),
        incremental_window_days=None,
        allowed={job.dataset: {s: amap[s] for s in symbols}},
    )
    if plans is None:
        raise RuntimeError("backfill day listing failed")
    report = service.execute_plan(downloader, plans, root=root, workers=workers, deadline=deadline)
    remaining = report.remaining_days + report.failed + report.pending
    done = remaining == 0
    if done:
        jobs.finish_job(
            root, job.job_id, summary={"ok": report.ok, "failed": 0, "sealed": report.sealed, "remaining": 0}
        )
    print(  # noqa: T201
        f"backfill job={job.job_id} ok={report.ok} failed={report.failed} "
        f"sealed={report.sealed} remaining={remaining} status={'DONE' if done else 'PARTIAL'}"
    )


def _drain_jobs(
    args: argparse.Namespace, root: Path, downloader: object, deadline: pd.Timestamp, job_workers: int
) -> None:
    import pandas as pd

    from src.market_data.public_archive.jobs import pending_jobs

    try:
        pending = pending_jobs(root)
    except Exception:
        logger.exception("[DATA] stage=archive_backfill pending_jobs_failed")
        return
    for job in pending:
        if pd.Timestamp.now(tz="UTC") >= deadline:
            break
        try:
            _drain_job(root, downloader, job, deadline, job_workers)
        except Exception:
            logger.exception("[DATA] stage=archive_backfill job_failed job=%s", job.job_id)
            print(f"backfill job={job.job_id} ok=0 failed=1 sealed=0 remaining=unknown status=PARTIAL")  # noqa: T201


def _run_collect(args: argparse.Namespace) -> None:
    from src.market_data.public_archive import store

    if not args.execute:
        _collect_under_lock(args)
        return
    try:
        with store.archive_lock(_root(args), wait_s=float(getattr(args, "lock_wait_s", 1800.0)), holder="collect"):
            _collect_under_lock(args)
    except store.ArchiveBusyError as exc:
        print(f"archive collect status=BUSY holder={exc.holder or 'null'}")  # noqa: T201
        raise SystemExit(75) from None


def _collect_under_lock(args: argparse.Namespace) -> None:
    from src.market_data.public_archive import service as service_mod
    from src.market_data.public_archive.catalog import ARCHIVE_RETRY_HORIZON_DAYS, DEFAULT_DATASETS

    root = _root(args)
    drain_until = getattr(args, "drain_jobs_until", None)
    deadline = _drain_deadline(drain_until) if drain_until else None
    if drain_until and deadline is None:
        logger.info("[DATA] stage=archive_backfill status=SKIPPED reason=past_deadline until=%s", drain_until)
    override = getattr(args, "job_workers", None)
    job_workers = args.workers if override is None else override
    if not 1 <= args.workers <= 8 or not 1 <= job_workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    downloader = _downloader(args)
    datasets = list(DEFAULT_DATASETS) if args.dataset == "all" else [args.dataset]
    end = _parse_day(args.end, datetime.now(UTC).date() - timedelta(days=1))
    start = date.fromisoformat(args.start) if args.start else None
    allowed: dict[str, dict[str, frozenset[date]]] | None = None
    span = None
    if args.symbols == "all" and not getattr(args, "no_mask", False):
        symbols, allowed, span = _masked_universe(args, downloader, datasets, start, end)
    else:
        symbols = _plain_universe(args, downloader, datasets)
    plans = service_mod.plan_collection(
        downloader,
        root=root,
        datasets=datasets,
        symbols=symbols,
        start=start,
        end=end,
        slice_hours=(int(args.slice_hours[0]), int(args.slice_hours[1])),
        slice_symbols=symbols.get("aggTrades", ()),
        incremental_window_days=ARCHIVE_RETRY_HORIZON_DAYS
        if args.start is None and os.environ.get("ARCHIVE_INCREMENTAL") == "1"
        else None,
        allowed=allowed,
    )
    if plans is None:
        print("archive collect status=LISTING_FAILED")  # noqa: T201
        raise SystemExit(2)
    _print_plan(root, datasets, plans, allowed, span)
    if not args.execute:
        return
    report = service_mod.execute_plan(downloader, plans, root=root, workers=args.workers)
    print(  # noqa: T201
        f"collect planned={report.planned_days} ok={report.ok} "
        f"failed={report.failed} absent={report.absent} pending={report.pending} sealed={report.sealed}"
    )
    if deadline is not None:
        _drain_jobs(args, root, downloader, deadline, int(job_workers))
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
    from src.market_data.public_archive import store as store_mod
    from src.market_data.public_archive.retention import prune_backed_up_archive
    from src.market_data.streams.retention import read_backup_status

    root = _root(args)
    try:
        with store_mod.archive_lock(root, wait_s=float(getattr(args, "lock_wait_s", 1800.0)), holder="prune"):
            if getattr(args, "backup_status", None):
                status_path = Path(args.backup_status)
            elif ARCHIVE_BACKUP_MOUNT.exists():
                status_path = ARCHIVE_BACKUP_MOUNT
            else:
                status_path = BASE_DIR / "deploy" / "backup" / "status" / "last_archive_success.json"
            try:
                status = read_backup_status(status_path)
                report = prune_backed_up_archive(
                    root, status=status, now=pd.Timestamp.now(tz="UTC"), keep_days=int(args.keep_days)
                )
            except OSError as exc:
                print(json.dumps({"error": type(exc).__name__}))  # noqa: T201
                raise SystemExit(4) from exc
    except store_mod.ArchiveBusyError:
        print(  # noqa: T201
            json.dumps(
                {
                    "prune_blocked": True,
                    "blocked_reason": "busy",
                    "pruned_files": 0,
                    "kept_files": 0,
                    "sealed_after": 0,
                },
                sort_keys=True,
            )
        )
        return
    print(  # noqa: T201
        json.dumps(
            {
                "prune_blocked": report.prune_blocked,
                "blocked_reason": report.blocked_reason,
                "pruned_files": report.pruned_files,
                "kept_files": report.kept_files,
                "sealed_after": report.sealed_after,
            },
            sort_keys=True,
        )
    )


def add_archive_commands(archive_parser: argparse.ArgumentParser) -> None:
    """Register ``collect``, ``status`` and ``prune`` on the ``archive`` parser."""
    from src.market_data.public_archive.catalog import TAPE_SLICE_HOURS

    subparsers = archive_parser.add_subparsers(dest="archive_command", required=True)

    collect = subparsers.add_parser("collect", help="Mirror Vision archives into compact parquet (dry-run by default)")
    collect.add_argument("--dataset", choices=["metrics", "bookDepth", "aggTrades", "all"], default="all")
    collect.add_argument("--start", type=str, default=None)
    collect.add_argument("--end", type=str, default=None)
    collect.add_argument("--symbols", type=str, default="all")
    collect.add_argument("--slice-hours", nargs=2, type=int, default=list(TAPE_SLICE_HOURS))
    collect.add_argument("--workers", type=int, default=4)
    collect.add_argument("--execute", action="store_true", default=False)
    collect.add_argument("--root", type=str, default=None)
    collect.add_argument("--breadth", type=int, default=None)
    collect.add_argument("--preroll-days", type=int, default=None)
    collect.add_argument("--no-mask", action="store_true", default=False)
    collect.add_argument("--census-root", type=Path, default=None)
    collect.add_argument("--lock-wait-s", type=float, default=1800.0)
    collect.add_argument("--drain-jobs-until", type=str, default=None)
    collect.add_argument("--job-workers", type=int, default=None)
    collect.set_defaults(handler=_run_collect)

    status = subparsers.add_parser("status", help="Show archive coverage per dataset")
    status.add_argument("--root", type=str, default=None)
    status.set_defaults(handler=_run_status)

    prune = subparsers.add_parser("prune", help="Prune backup-gated month partitions")
    prune.add_argument("--keep-days", type=int, default=45)
    prune.add_argument("--backup-status", type=str, default=None)
    prune.add_argument("--root", type=str, default=None)
    prune.add_argument("--lock-wait-s", type=float, default=1800.0)
    prune.set_defaults(handler=_run_prune)

    backfill = subparsers.add_parser("backfill", help="Manage resumable backfill jobs")
    backfill_sub = backfill.add_subparsers(dest="backfill_command", required=True)
    add_cmd = backfill_sub.add_parser("add", help="Enqueue a backfill job")
    add_cmd.add_argument("--dataset", choices=["metrics", "bookDepth", "aggTrades"], required=True)
    add_cmd.add_argument("--start", type=str, required=True)
    add_cmd.add_argument("--end", type=str, default=None)
    add_cmd.add_argument("--census-root", type=str, required=True)
    add_cmd.add_argument("--breadth", type=int, default=None)
    add_cmd.add_argument("--preroll-days", type=int, default=None)
    add_cmd.add_argument("--root", type=str, default=None)
    add_cmd.set_defaults(handler=_run_backfill_add)
    list_cmd = backfill_sub.add_parser("list", help="List backfill jobs")
    list_cmd.add_argument("--root", type=str, default=None)
    list_cmd.set_defaults(handler=_run_backfill_list)
    cancel_cmd = backfill_sub.add_parser("cancel", help="Cancel a pending backfill job")
    cancel_cmd.add_argument("job_id", type=str)
    cancel_cmd.add_argument("--root", type=str, default=None)
    cancel_cmd.set_defaults(handler=_run_backfill_cancel)

    imp = subparsers.add_parser("import", help="Merge another archive root")
    imp.add_argument("--from", dest="from_dir", type=str, required=True)
    imp.add_argument(
        "--dataset", nargs="+", action="extend", choices=["metrics", "bookDepth", "aggTrades"], default=None
    )
    imp.add_argument("--root", type=str, default=None)
    imp.add_argument("--lock-wait-s", type=float, default=1800.0)
    imp.set_defaults(handler=_run_import)
