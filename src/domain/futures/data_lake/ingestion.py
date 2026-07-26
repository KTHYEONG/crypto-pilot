from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.futures.data_lake.contracts import (
    BinanceDataClient,
    DataCatalog,
    DataLakeConfig,
    DatasetKind,
    DataSnapshot,
    IngestionPlan,
    PartitionManifest,
)
from src.domain.futures.data_lake.reconciliation import (
    _FUNDING_SCHEMA_VERSION,
    _FUNDING_VALIDATOR_VERSION,
    FundingRepairRequest,
    _compute_sidecar_path,
    _write_sidecar,
    validate_funding_frame,
)

_logger = logging.getLogger(__name__)
_MAX_PLAN_SYMBOLS = 120
_DEFAULT_LOOKBACK_DAYS = 730


class ChecksumMismatchError(RuntimeError):
    ...


class StorageBudgetError(RuntimeError):
    ...


class DataCoverageError(RuntimeError):
    ...


def build_ingestion_plan(
    *, config: DataLakeConfig, reference_date: date, start_date: date | None = None,
) -> IngestionPlan:
    if reference_date > date.today():
        msg = f"reference_date {reference_date} cannot be in the future"
        raise ValueError(msg)

    _logger.info("building ingestion plan for %s with market=%s", reference_date, config.market)

    lake_root = config.root
    exchange_info_root = lake_root / DatasetKind.EXCHANGE_INFO.value
    universe_root = lake_root / DatasetKind.UNIVERSE_STATE.value
    candidates: tuple[str, ...] = ()
    if exchange_info_root.exists():
        candidates = tuple(sorted(
            path.name.removeprefix("symbol=") for path in exchange_info_root.glob("symbol=*")
        ))
    if not candidates and universe_root.exists():
        candidates = tuple(sorted({
            path.name.removeprefix("symbol=") for path in universe_root.glob("symbol=*")
        }))
    if not candidates:
        kline_root = lake_root / DatasetKind.KLINES_1H.value
        if kline_root.exists():
            candidates = tuple(sorted(
                path.name.removeprefix("symbol=") for path in kline_root.glob("symbol=*")
            ))

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

            lake_paths = tuple((lake_root / DatasetKind.KLINES_1H.value / f"symbol={symbol}").glob("year=*/month=*/part.parquet"))
            if lake_paths:
                sample = pd.read_parquet(max(lake_paths))
                volume_column = "quote_volume" if "quote_volume" in sample.columns else "quote_vol"
                score = float(pd.to_numeric(sample[volume_column], errors="coerce").tail(min(24 * 30, len(sample))).median())
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
        _logger.warning("no lake source files found under %s", lake_root)

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
            DatasetKind.UNIVERSE_STATE,
        ),
        config=config,
        start_date=start_date or reference_date - timedelta(days=_DEFAULT_LOOKBACK_DAYS),
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


def restrict_to_historically_available_core_symbols(
    *, plan: IngestionPlan, client: BinanceDataClient | None, catalog: DataCatalog,
) -> IngestionPlan:
    """Return a CORE-only plan limited to symbols with data at the required start month.

    A zero-byte Vision archive is an expected consequence of listing after the
    requested historical start.  It is not retried for every later month.
    """
    start_date = plan.start_date or plan.reference_date
    month = start_date.replace(day=1)
    start_time_ms = int(datetime(month.year, month.month, 1, tzinfo=UTC).timestamp() * 1000)
    core_datasets = (DatasetKind.KLINES_1H, DatasetKind.FUNDING_EVENT)
    eligible_symbols: list[str] = []

    for symbol in plan.broad_symbols:
        symbol_complete = True
        for dataset in core_datasets:
            if catalog.partition_exists(dataset, symbol, start_time_ms):
                continue
            if client is None:
                symbol_complete = False
                break
            payload = client.download_partition(dataset, symbol, start_time_ms)
            if not payload:
                symbol_complete = False
                break
            expected_checksum = client.download_checksum(dataset, symbol, start_time_ms)
            if hashlib.sha256(payload).hexdigest() != expected_checksum:
                raise ChecksumMismatchError(
                    f"checksum mismatch for {dataset}/{symbol} at {month}",
                )
            manifest = _payload_manifest(
                dataset=dataset, symbol=symbol, month=month,
                payload=payload, root=plan.config.root,
            )
            catalog.commit_partition(manifest)
        if symbol_complete:
            eligible_symbols.append(symbol)

    if not eligible_symbols:
        raise DataCoverageError(
            f"no symbols have both CORE datasets at required start month {month}",
        )
    return replace(
        plan,
        broad_symbols=tuple(eligible_symbols),
        selected_symbols=tuple(eligible_symbols),
        datasets=core_datasets,
    )


def restrict_to_complete_core_symbols(
    *, plan: IngestionPlan, catalog: DataCatalog,
) -> IngestionPlan:
    """Exclude symbols with a missing CORE partition in the requested history."""
    start_date = plan.start_date or plan.reference_date
    complete_symbols: list[str] = []
    for symbol in plan.broad_symbols:
        complete = all(
            catalog.partition_exists(
                dataset,
                symbol,
                int(datetime(month.year, month.month, 1, tzinfo=UTC).timestamp() * 1000),
            )
            for dataset in (DatasetKind.KLINES_1H, DatasetKind.FUNDING_EVENT)
            for month in _month_starts(start_date, plan.reference_date)
        )
        if complete:
            complete_symbols.append(symbol)
    if not complete_symbols:
        raise DataCoverageError("no symbols have complete CORE history")
    return replace(
        plan,
        broad_symbols=tuple(complete_symbols),
        selected_symbols=tuple(complete_symbols),
    )


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

    if dataset is DatasetKind.FUNDING_EVENT:
        month_start_ms = int(datetime(month.year, month.month, 1, tzinfo=UTC).timestamp() * 1000)
        try:
            validate_funding_frame(
                frame,
                source=f"{dataset.value}/{symbol}/{month}",
                month_start_ms=month_start_ms,
            )
        except Exception as exc:  # noqa: BLE001
            raise DataCoverageError(
                f"funding integrity failure for {dataset.value}/{symbol}/{month}: {exc}"
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


def repair_funding_partitions(
    *,
    requests: tuple[FundingRepairRequest, ...],
    client: BinanceDataClient,
    catalog: DataCatalog,
    root: Path,
    max_workers: int,
) -> tuple[PartitionManifest, ...]:
    """Re-download only funding partitions rejected by reconciliation."""
    pending = tuple(
        (
            DatasetKind.FUNDING_EVENT,
            request.symbol,
            datetime.fromtimestamp(request.start_time_ms / 1000, tz=UTC).date(),
            request.start_time_ms,
        )
        for request in requests
    )
    repaired: list[PartitionManifest] = []
    for (dataset, symbol, month, start_time_ms), payload in _download_pending_partitions(
        client=client,
        pending=pending,
        max_workers=max_workers,
    ):
        if not payload:
            raise DataCoverageError(
                f"empty funding repair payload for {symbol}/{start_time_ms}"
            )
        expected_checksum = client.download_checksum(dataset, symbol, start_time_ms)
        actual_checksum = hashlib.sha256(payload).hexdigest()
        if actual_checksum != expected_checksum:
            raise ChecksumMismatchError(
                f"checksum mismatch for funding repair {symbol}/{start_time_ms}"
            )
        manifest = _payload_manifest(
            dataset=dataset,
            symbol=symbol,
            month=month,
            payload=payload,
            root=root,
        )
        stat = manifest.path.stat()
        frame = pd.read_parquet(manifest.path, engine="pyarrow")
        _write_sidecar(
            _compute_sidecar_path(manifest.path),
            {
                "size": stat.st_size,
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": manifest.sha256,
                "schema_version": _FUNDING_SCHEMA_VERSION,
                "validator_version": _FUNDING_VALIDATOR_VERSION,
                "row_count": len(frame),
                "min_timestamp": int(frame["timestamp"].min()),
                "max_timestamp": int(frame["timestamp"].max()),
                "min_rate": float(pd.to_numeric(frame["funding_rate"]).min()),
                "max_rate": float(pd.to_numeric(frame["funding_rate"]).max()),
            },
        )
        catalog.commit_partition(manifest)
        repaired.append(manifest)
    if len(repaired) != len(requests):
        raise DataCoverageError(
            f"funding repair incomplete: expected {len(requests)}, got {len(repaired)}"
        )
    return tuple(repaired)


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
        if dataset_kind in (DatasetKind.EXCHANGE_INFO, DatasetKind.UNIVERSE_STATE):
            continue
        symbols = plan.broad_symbols if dataset_kind is DatasetKind.KLINES_1H else plan.selected_symbols
        dataset_start = start_date
        if dataset_kind is DatasetKind.KLINES_1M:
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


def migrate_legacy_universe_state(
    *, source_ledger: Path, catalog: DataCatalog, root: Path
) -> str:
    _logger.info("migrating legacy universe state from %s", source_ledger)
    if not source_ledger.exists():
        raise DataCoverageError(f"source ledger not found: {source_ledger}")  # pragma: no cover - defensive input guard

    import sqlite3


    conn = sqlite3.connect(str(source_ledger))
    try:
        df = pd.read_sql_query(
            "SELECT symbol, date, knowledge_date, is_listed, is_trading, status, "
            "adv_usdt_median, listing_age_days, taker_fee_bps, contract_type, quote_asset "
            "FROM ledger ORDER BY symbol, date, knowledge_date",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        raise DataCoverageError(f"no rows in ledger: {source_ledger}")  # pragma: no cover - defensive input guard

    lake_symbols = {
        path.name.removeprefix("symbol=")
        for path in (root / DatasetKind.KLINES_1H.value).glob("symbol=*")
        if path.name.removeprefix("symbol=").isascii()
        and path.name.removeprefix("symbol=").isalnum()
        and path.name.removeprefix("symbol=").endswith("USDT")
    }
    if not lake_symbols:
        raise DataCoverageError("no valid klines_1h symbols available for universe migration")
    df = df.loc[df["symbol"].isin(lake_symbols)].copy()
    if df.empty:
        raise DataCoverageError("legacy ledger has no rows for lake symbols")

    knowledge_time = pd.to_datetime(df["knowledge_date"], utc=True)
    df["knowledge_time_ns"] = knowledge_time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    df["effective_time_ns"] = (
        (knowledge_time + pd.Timedelta(days=1))
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    df["eligible"] = df["is_listed"].fillna(False).astype(bool) & df["is_trading"].fillna(False).astype(bool)
    df["entry_block"] = ~df["eligible"]
    df["exit_required"] = False
    df["capacity_usdt"] = df["adv_usdt_median"].fillna(0.0).astype(float) * 0.1
    df["risk_scale"] = 1.0
    df["execution_cost_bps"] = (df["taker_fee_bps"].fillna(5.0).astype(float) * 2.0)
    df["state_reason"] = ""
    df["universe_config_hash"] = "migration-v1"
    df["source_manifest_hash"] = "migration-v1"

    output_cols = [
        "effective_time_ns", "knowledge_time_ns", "symbol", "eligible", "entry_block",
        "exit_required", "capacity_usdt", "risk_scale", "execution_cost_bps",
        "state_reason", "universe_config_hash", "source_manifest_hash",
    ]
    output = df[output_cols].copy()
    output["month"] = (
        (knowledge_time + pd.Timedelta(days=1)).dt.tz_localize(None).dt.to_period("M")
    )

    total_rows = 0
    for month_period, group in output.groupby("month"):
        month_start = month_period.start_time
        year = month_start.year
        month_num = month_start.month
        payload_buffer = io.BytesIO()
        monthly = group.drop(columns=["month"])
        monthly.to_parquet(payload_buffer, index=False, compression="zstd")
        payload = payload_buffer.getvalue()

        path = (
            root
            / DatasetKind.UNIVERSE_STATE.value
            / "symbol=__all__"
            / f"year={year:04d}"
            / f"month={month_num:02d}"
            / "part.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.parquet")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

        start_ms = int(datetime(year, month_num, 1, tzinfo=UTC).timestamp() * 1000)
        manifest = PartitionManifest(
            dataset=DatasetKind.UNIVERSE_STATE,
            symbol="__all__",
            start_time_ms=start_ms,
            end_time_ms=start_ms,
            row_count=len(monthly),
            sha256=hashlib.sha256(payload).hexdigest(),
            source="legacy_migration",
            is_final=True,
            path=path,
        )
        catalog.commit_partition(manifest)
        total_rows += len(monthly)
        _logger.info("migrated universe_state partition %s-%02d: %d rows", year, month_num, len(monthly))

    latest = df.sort_values(["symbol", "knowledge_date"]).groupby("symbol", as_index=False).tail(1)
    for row in latest.itertuples(index=False):
        observed = pd.Timestamp(row.knowledge_date, tz="UTC")
        exchange_frame = pd.DataFrame(({
            "timestamp": int(observed.timestamp() * 1000),
            "symbol": str(row.symbol),
            "quote_asset": str(row.quote_asset),
            "contract_type": str(row.contract_type),
            "is_trading": bool(row.is_trading),
        },))
        exchange_payload = io.BytesIO()
        exchange_frame.to_parquet(exchange_payload, index=False, compression="zstd")
        payload = exchange_payload.getvalue()
        month = observed.date().replace(day=1)
        manifest = _payload_manifest(
            dataset=DatasetKind.EXCHANGE_INFO,
            symbol=str(row.symbol),
            month=month,
            payload=payload,
            root=root,
        )
        catalog.commit_partition(manifest)

    _logger.info("migration complete: %d total rows across %d partitions", total_rows, len(output["month"].unique()))
    reference_time_ms = int(datetime.now(UTC).timestamp() * 1000)
    snap = catalog.load_snapshot(reference_time_ms)
    return snap.universe_state_hash


def refresh_live_universe_state(
    *, root: Path, catalog: DataCatalog, client: BinanceDataClient,
    knowledge_time_ns: int,
) -> str:
    """Persist one causal daily PIT state from current Binance exchangeInfo."""
    exchange_info = client.fetch_exchange_info()
    records = exchange_info.get("symbols", [])
    if not isinstance(records, list):
        raise DataCoverageError("exchangeInfo symbols must be a list")

    lake_symbols = {
        path.name.removeprefix("symbol=")
        for path in (root / DatasetKind.KLINES_1H.value).glob("symbol=*")
        if path.name.removeprefix("symbol=").isascii()
        and path.name.removeprefix("symbol=").isalnum()
        and path.name.removeprefix("symbol=").endswith("USDT")
    }
    knowledge = pd.Timestamp(knowledge_time_ns, unit="ns", tz="UTC")
    effective = (knowledge.normalize() + pd.Timedelta(days=1)).value
    state_rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        symbol = str(record.get("symbol", "")).upper()
        if (
            not symbol.isascii() or not symbol.isalnum()
            or str(record.get("quoteAsset", "")).upper() != "USDT"
            or str(record.get("contractType", "")).upper() != "PERPETUAL"
            or (lake_symbols and symbol not in lake_symbols)
        ):
            continue
        eligible = str(record.get("status", "")).upper() == "TRADING"
        state_rows.append({
            "effective_time_ns": effective,
            "knowledge_time_ns": knowledge_time_ns,
            "symbol": symbol,
            "eligible": eligible,
            "entry_block": not eligible,
            "exit_required": False,
            "capacity_usdt": 1_000_000.0 if eligible else 0.0,
            "risk_scale": 1.0 if eligible else 0.0,
            "execution_cost_bps": 12.0,
            "state_reason": "live_exchange_info",
            "universe_config_hash": "live-exchange-info-v1",
            "source_manifest_hash": "exchange-info-live",
        })
    if not state_rows:
        raise DataCoverageError("live exchangeInfo produced no USDT perpetual symbols")

    frame = pd.DataFrame(state_rows)
    observed = pd.Timestamp(effective, unit="ns", tz="UTC")
    path = (
        root / DatasetKind.UNIVERSE_STATE.value / "symbol=__all__"
        / f"year={observed.year:04d}" / f"month={observed.month:02d}"
        / f"day={observed.day:02d}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_buffer = io.BytesIO()
    frame.to_parquet(payload_buffer, index=False, compression="zstd")
    payload = payload_buffer.getvalue()
    temporary = path.with_suffix(".tmp.parquet")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    manifest = PartitionManifest(
        dataset=DatasetKind.UNIVERSE_STATE,
        symbol="__all__",
        start_time_ms=effective // 1_000_000,
        end_time_ms=effective // 1_000_000,
        row_count=len(frame),
        sha256=hashlib.sha256(payload).hexdigest(),
        source="binance_exchange_info_live",
        is_final=True,
        path=path,
    )
    catalog.commit_partition(manifest)
    return catalog.load_snapshot(manifest.end_time_ms).universe_state_hash
