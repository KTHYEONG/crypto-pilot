from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.domain.futures.data_lake.contracts import (
    BinanceDataClient,
    DataCatalog,
    DataLakeConfig,
    DatasetKind,
    DataSnapshot,
    IngestionPlan,
    PartitionManifest,
)

_logger = logging.getLogger(__name__)
_MAX_PLAN_SYMBOLS = 120


class ChecksumMismatchError(RuntimeError):
    ...


class StorageBudgetError(RuntimeError):
    ...


class DataCoverageError(RuntimeError):
    ...


def build_ingestion_plan(
    *, config: DataLakeConfig, reference_date: date
) -> IngestionPlan:
    if reference_date > date.today():
        msg = f"reference_date {reference_date} cannot be in the future"
        raise ValueError(msg)

    _logger.info("building ingestion plan for %s with market=%s", reference_date, config.market)

    source_root = config.root.parent
    ohlcv_root = source_root / "ohlcv" / "1h"
    candidates = tuple(sorted(path.stem for path in ohlcv_root.glob("*.parquet")))
    if not candidates:
        ohlcv_root = source_root / "ohlcv" / "1m"
        candidates = tuple(sorted(path.stem for path in ohlcv_root.glob("*.parquet")))
    lake_root = config.root / DatasetKind.KLINES_1H.value
    if not candidates and lake_root.exists():
        candidates = tuple(sorted(path.name.removeprefix("symbol=") for path in lake_root.glob("symbol=*")))
    liquidity: list[tuple[float, str]] = []
    for symbol in candidates:
        normalized_symbol = symbol.replace("_", "")
        if (
            not normalized_symbol.isascii()
            or not normalized_symbol.isalnum()
            or not normalized_symbol.endswith(config.quote_asset)
        ):
            _logger.warning("skipping invalid Binance symbol candidate: %s", symbol)
            continue
        try:
            import pandas as pd

            raw_path = ohlcv_root / f"{symbol}.parquet"
            lake_paths = tuple((lake_root / f"symbol={symbol}").glob("year=*/month=*/part.parquet"))
            path = raw_path if raw_path.exists() else max(lake_paths, default=raw_path)
            sample = pd.read_parquet(path)
            volume_column = "quote_volume" if "quote_volume" in sample.columns else "quote_vol"
            score = float(pd.to_numeric(sample[volume_column], errors="coerce").tail(24 * 30).median())
            if score > 0:
                liquidity.append((score, symbol))
        except (KeyError, OSError, ValueError):
            continue
    selected_symbols: list[str] = []
    seen_symbols: set[str] = set()
    for _, raw_symbol in sorted(liquidity, key=lambda item: (-item[0], item[1])):
        normalized_symbol = raw_symbol.replace("_", "")
        if normalized_symbol in seen_symbols:
            continue
        selected_symbols.append(normalized_symbol)
        seen_symbols.add(normalized_symbol)
        if len(selected_symbols) == _MAX_PLAN_SYMBOLS:
            break
    broad_symbols = tuple(selected_symbols)
    if not broad_symbols:
        _logger.warning("no local 1h source files found under %s", ohlcv_root)

    return IngestionPlan(
        reference_date=reference_date,
        broad_symbols=broad_symbols,
        selected_symbols=broad_symbols,
        datasets=(
            DatasetKind.KLINES_1H,
            DatasetKind.KLINES_1M,
            DatasetKind.FUNDING_EVENT,
            DatasetKind.PREMIUM_5M,
            DatasetKind.MARK_1M,
            DatasetKind.INDEX_1M,
            DatasetKind.METRICS_5M,
            DatasetKind.COST_CALIBRATION,
        ),
        config=config,
        start_date=reference_date - timedelta(days=730),
    )


def _month_starts(start: date, end: date) -> tuple[date, ...]:
    current = start.replace(day=1)
    terminal = end.replace(day=1)
    months: list[date] = []
    while current <= terminal:
        months.append(current)
        current = (
            current.replace(year=current.year + 1, month=1)
            if current.month == 12
            else current.replace(month=current.month + 1)
        )
    return tuple(months)


def _partition_path(root: Path, dataset: DatasetKind, symbol: str, month: date) -> Path:
    return (
        root
        / dataset.value
        / f"symbol={symbol}"
        / f"year={month.year:04d}"
        / f"month={month.month:02d}"
        / "part.parquet"
    )


def _payload_manifest(
    *, dataset: DatasetKind, symbol: str, month: date, payload: bytes, root: Path
) -> PartitionManifest:
    path = _partition_path(root, dataset, symbol, month)
    import pandas as pd

    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001
        raise DataCoverageError(
            f"invalid parquet payload for {dataset.value}/{symbol}/{month}: {type(exc).__name__}"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    start_ms = int(datetime(month.year, month.month, 1, tzinfo=UTC).timestamp() * 1000)
    return PartitionManifest(
        dataset=dataset,
        symbol=symbol,
        start_time_ms=start_ms,
        end_time_ms=(
            int(pd.to_datetime(frame["timestamp"], unit="ms", utc=True).max().timestamp() * 1000)
            if not frame.empty and "timestamp" in frame.columns
            else start_ms
        ),
        row_count=len(frame),
        sha256=hashlib.sha256(payload).hexdigest(),
        source="binance_vision_or_local_cache",
        is_final=True,
        path=path,
    )


def _download_pending_partitions(
    *,
    client: BinanceDataClient,
    pending: tuple[tuple[DatasetKind, str, date, int], ...],
    max_workers: int,
) -> Iterator[tuple[tuple[DatasetKind, str, date, int], bytes]]:
    """Download with a bounded in-flight queue.

    The client owns exchange pacing/retry policy.  This layer only overlaps archive
    latency and keeps at most ``max_workers`` Parquet payloads in memory.
    """
    tasks = iter(pending)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: dict[Future[bytes], tuple[DatasetKind, str, date, int]] = {}
        for _ in range(max_workers):
            try:
                dataset, symbol, _month, start_time_ms = next(tasks)
            except StopIteration:
                break
            future = executor.submit(client.download_partition, dataset, symbol, start_time_ms)
            in_flight[future] = (dataset, symbol, _month, start_time_ms)

        while in_flight:
            completed = next(as_completed(in_flight))
            task = in_flight.pop(completed)
            yield task, completed.result()
            try:
                dataset, symbol, _month, start_time_ms = next(tasks)
            except StopIteration:
                continue
            future = executor.submit(client.download_partition, dataset, symbol, start_time_ms)
            in_flight[future] = (dataset, symbol, _month, start_time_ms)


def sync_futures_data_lake(
    *, plan: IngestionPlan, client: BinanceDataClient, catalog: DataCatalog
) -> DataSnapshot:
    if plan.config.hard_cap_gib <= 0:
        msg = f"invalid hard_cap_gib: {plan.config.hard_cap_gib}"
        raise ValueError(msg)

    current_bytes = catalog.total_bytes()
    projected_gib = current_bytes / (1024**3)

    if projected_gib >= plan.config.hard_cap_gib:
        raise StorageBudgetError(
            f"current {projected_gib:.1f} GiB >= hard cap {plan.config.hard_cap_gib} GiB"
        )

    partitions: list[PartitionManifest] = []
    start_date = plan.start_date or plan.reference_date

    for dataset_kind in plan.datasets:
        symbols = plan.broad_symbols if dataset_kind is DatasetKind.KLINES_1H else plan.selected_symbols
        dataset_start = start_date
        if dataset_kind in (DatasetKind.KLINES_1M, DatasetKind.METRICS_5M):
            dataset_start = max(
                start_date,
                plan.reference_date - timedelta(days=180),
            )
        pending = tuple(
            (dataset_kind, symbol, month, start_time_ms)
            for month in _month_starts(dataset_start, plan.reference_date)
            for start_time_ms in (int(datetime(month.year, month.month, 1, tzinfo=UTC).timestamp() * 1000),)
            for symbol in symbols
            if not catalog.partition_exists(dataset_kind, symbol, start_time_ms)
        )
        for (dataset, symbol, month, start_time_ms), payload in _download_pending_partitions(
            client=client,
            pending=pending,
            max_workers=plan.config.max_workers,
        ):
            if not payload:
                continue
            expected_checksum = client.download_checksum(dataset, symbol, start_time_ms)
            actual_checksum = hashlib.sha256(payload).hexdigest()

            if actual_checksum != expected_checksum:
                raise ChecksumMismatchError(
                    f"checksum mismatch for {dataset}/{symbol}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )
            try:
                manifest = _payload_manifest(
                    dataset=dataset,
                    symbol=symbol,
                    month=month,
                    payload=payload,
                    root=plan.config.root,
                )
            except DataCoverageError as exc:
                _logger.error("quarantining invalid partition %s/%s: %s", dataset, symbol, exc)
                continue
            catalog.commit_partition(manifest)
            partitions.append(manifest)

    _logger.info("sync complete: %d partitions, %d bytes", len(partitions), 0)

    reference_time_ms = int(datetime.combine(plan.reference_date, datetime.max.time(), tzinfo=UTC).timestamp() * 1000)
    return catalog.load_snapshot(reference_time_ms)
