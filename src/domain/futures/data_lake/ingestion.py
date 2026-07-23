from __future__ import annotations

import hashlib
import logging
import os
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
        symbol = raw_symbol.replace("_", "")
        if symbol in seen_symbols:
            continue
        selected_symbols.append(symbol)
        seen_symbols.add(symbol)
        if len(selected_symbols) == _MAX_PLAN_SYMBOLS:
            break
    broad_symbols = tuple(selected_symbols)
    if not broad_symbols:
        _logger.warning("no local 1h source files found under %s", ohlcv_root)

    return IngestionPlan(
        reference_date=reference_date,
        broad_symbols=broad_symbols,
        selected_symbols=broad_symbols,
        datasets=(DatasetKind.KLINES_1H, DatasetKind.FUNDING_EVENT),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    import pandas as pd

    frame = pd.read_parquet(path)
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
        for month in _month_starts(start_date, plan.reference_date):
            start_time_ms = int(datetime(month.year, month.month, 1, tzinfo=UTC).timestamp() * 1000)
            for sym in symbols:
                if catalog.partition_exists(dataset_kind, sym, start_time_ms):
                    continue
                payload = client.download_partition(dataset_kind, sym, start_time_ms)
                if not payload:
                    continue
                expected_checksum = client.download_checksum(dataset_kind, sym, start_time_ms)
                actual_checksum = hashlib.sha256(payload).hexdigest()

                if actual_checksum != expected_checksum:
                    raise ChecksumMismatchError(
                        f"checksum mismatch for {dataset_kind}/{sym}: "
                        f"expected {expected_checksum}, got {actual_checksum}"
                    )
                manifest = _payload_manifest(
                    dataset=dataset_kind,
                    symbol=sym,
                    month=month,
                    payload=payload,
                    root=plan.config.root,
                )
                catalog.commit_partition(manifest)
                partitions.append(manifest)

    _logger.info("sync complete: %d partitions, %d bytes", len(partitions), 0)

    reference_time_ms = int(datetime.combine(plan.reference_date, datetime.max.time(), tzinfo=UTC).timestamp() * 1000)
    return catalog.load_snapshot(reference_time_ms)
