from __future__ import annotations

import hashlib
import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.exchange.binance_vision import BinanceVisionDownloader
from src.core.settings import FUTURES_DATA_DIR
from src.domain.futures.data_lake.contracts import (
    DatasetKind,
    DataSnapshot,
    GridRequest,
    IngestionPlan,
    NativeFeatureGrid,
    PartitionManifest,
)

_logger = logging.getLogger(__name__)


class BinanceQueryClient:
    """Read cached Binance futures data first and use Vision only for missing months."""

    def __init__(self, source_root: Path = FUTURES_DATA_DIR) -> None:
        self._source_root = source_root
        self._vision = BinanceVisionDownloader()
        self._payloads: dict[tuple[DatasetKind, str, int], bytes] = {}
        self.download_calls = 0

    @staticmethod
    def _month(start_time_ms: int) -> datetime:
        return datetime.fromtimestamp(start_time_ms / 1000, tz=UTC)

    @staticmethod
    def _normalize_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        if "quote_volume" not in normalized.columns and "quote_vol" in normalized.columns:
            normalized = normalized.rename(columns={"quote_vol": "quote_volume"})
        if {"quote_volume", "close", "volume"}.issubset(normalized.columns):
            quote_volume = pd.to_numeric(normalized["quote_volume"], errors="coerce")
            close = pd.to_numeric(normalized["close"], errors="coerce")
            volume = pd.to_numeric(normalized["volume"], errors="coerce")
            normalized["quote_volume"] = quote_volume.fillna(close * volume)
        if "timestamp" not in normalized.columns:
            if "datetime" not in normalized.columns:
                return pd.DataFrame()
            normalized["timestamp"] = (
                pd.to_datetime(normalized["datetime"], utc=True, errors="coerce")
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
                // 1_000_000
            )
        normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], errors="coerce")
        return normalized.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")

    @staticmethod
    def _normalize_vision_klines(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        columns = ("timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base_volume", "taker_quote_volume", "ignore")
        data = frame.iloc[:, : len(columns)].copy()
        data.columns = columns[: len(data.columns)]
        return BinanceQueryClient._normalize_timestamp(data)

    @staticmethod
    def _normalize_vision_funding(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.iloc[:, :2].copy()
        data.columns = ("timestamp", "funding_rate")[: len(data.columns)]
        return BinanceQueryClient._normalize_timestamp(data)

    @staticmethod
    def _to_parquet_bytes(frame: pd.DataFrame) -> bytes:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False, compression="zstd")
        return buffer.getvalue()

    def _has_local_source(self, dataset: DatasetKind, symbol: str) -> bool:
        if dataset is DatasetKind.KLINES_1H:
            return any(
                path.exists()
                for path in (
                    self._source_root / "ohlcv" / "1h" / f"{symbol}.parquet",
                    self._source_root / "ohlcv" / "1m" / f"{symbol}.parquet",
                    self._source_root / "ohlcv" / "1m" / f"{symbol[:-4]}_USDT.parquet",
                )
            )
        if dataset is DatasetKind.KLINES_1M:
            return any(
                path.exists()
                for path in (
                    self._source_root / "ohlcv" / "1m" / f"{symbol}.parquet",
                    self._source_root / "ohlcv" / "1m" / f"{symbol[:-4]}_USDT.parquet",
                )
            )
        if dataset is DatasetKind.FUNDING_EVENT:
            return (self._source_root / "funding" / f"{symbol}.parquet").exists()
        if dataset is DatasetKind.METRICS_5M:
            return (self._source_root / "metrics" / f"{symbol}.parquet").exists()
        return False

    def _local_frame(self, dataset: DatasetKind, symbol: str) -> pd.DataFrame:
        if dataset in (DatasetKind.KLINES_1H, DatasetKind.KLINES_1M):
            source_interval = "1h" if dataset is DatasetKind.KLINES_1H else "1m"
            path = self._source_root / "ohlcv" / source_interval / f"{symbol}.parquet"
            if not path.exists():
                if dataset is DatasetKind.KLINES_1M:
                    path = self._source_root / "ohlcv" / source_interval / f"{symbol[:-4]}_USDT.parquet"
                    if not path.exists():
                        return pd.DataFrame()
                one_minute_path = self._source_root / "ohlcv" / "1m" / f"{symbol}.parquet"
                if not one_minute_path.exists():
                    one_minute_path = self._source_root / "ohlcv" / "1m" / f"{symbol[:-4]}_USDT.parquet"
                if not one_minute_path.exists():
                    return pd.DataFrame()
                minute = self._normalize_timestamp(pd.read_parquet(one_minute_path))
                if minute.empty:
                    return minute
                minute["datetime"] = pd.to_datetime(minute["timestamp"], unit="ms", utc=True)
                aggregations = {
                    "open": "first", "high": "max", "low": "min", "close": "last",
                    "volume": "sum", "quote_volume": "sum",
                }
                usable = {key: value for key, value in aggregations.items() if key in minute.columns}
                hourly = minute.set_index("datetime").resample("1h").agg(usable).dropna(subset=["close"])
                hourly["timestamp"] = (
                    hourly.index.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 1_000_000
                )
                return hourly.reset_index(drop=True)
        elif dataset is DatasetKind.FUNDING_EVENT:
            path = self._source_root / "funding" / f"{symbol}.parquet"
        elif dataset is DatasetKind.METRICS_5M:
            path = self._source_root / "metrics" / f"{symbol}.parquet"
        else:
            return pd.DataFrame()
        if not path.exists():
            return pd.DataFrame()
        return self._normalize_timestamp(pd.read_parquet(path))

    def _local_partition_frame(
        self,
        dataset: DatasetKind,
        symbol: str,
        month: datetime,
    ) -> pd.DataFrame:
        """Read only one calendar month from a local Parquet source.

        Avoiding a full-file read per monthly partition keeps the 1m materialization
        memory-bounded and prevents repeated scans of multi-gigabyte raw files.
        """
        period_start = int(datetime(month.year, month.month, 1, tzinfo=UTC).timestamp() * 1000)
        period_end = int((pd.Timestamp(month) + pd.offsets.MonthBegin(1)).timestamp() * 1000)
        filters = [("timestamp", ">=", period_start), ("timestamp", "<", period_end)]

        if dataset is DatasetKind.KLINES_1H:
            hourly_path = self._source_root / "ohlcv" / "1h" / f"{symbol}.parquet"
            if hourly_path.exists():
                return self._normalize_timestamp(pd.read_parquet(hourly_path, filters=filters))
            minute_path = self._source_root / "ohlcv" / "1m" / f"{symbol}.parquet"
            if not minute_path.exists():
                minute_path = self._source_root / "ohlcv" / "1m" / f"{symbol[:-4]}_USDT.parquet"
            if not minute_path.exists():
                return pd.DataFrame()
            minute = self._normalize_timestamp(pd.read_parquet(minute_path, filters=filters))
            if minute.empty:
                return minute
            minute["datetime"] = pd.to_datetime(minute["timestamp"], unit="ms", utc=True)
            aggregations = {
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum", "quote_volume": "sum",
            }
            usable = {key: value for key, value in aggregations.items() if key in minute.columns}
            hourly = minute.set_index("datetime").resample("1h").agg(usable).dropna(subset=["close"])
            hourly["timestamp"] = (
                hourly.index.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 1_000_000
            )
            return hourly.reset_index(drop=True)

        if dataset is DatasetKind.KLINES_1M:
            path = self._source_root / "ohlcv" / "1m" / f"{symbol}.parquet"
            if not path.exists():
                path = self._source_root / "ohlcv" / "1m" / f"{symbol[:-4]}_USDT.parquet"
        elif dataset is DatasetKind.METRICS_5M:
            path = self._source_root / "metrics" / f"{symbol}.parquet"
        elif dataset is DatasetKind.FUNDING_EVENT:
            path = self._source_root / "funding" / f"{symbol}.parquet"
        else:
            return pd.DataFrame()
        if not path.exists():
            return pd.DataFrame()
        return self._normalize_timestamp(pd.read_parquet(path, filters=filters))

    def _vision_frame(self, dataset: DatasetKind, symbol: str, month: datetime) -> pd.DataFrame:
        if dataset is DatasetKind.KLINES_1H:
            return self._normalize_vision_klines(
                self._vision.fetch_klines_archive_monthly(symbol, "1h", month.year, month.month)
            )
        if dataset is DatasetKind.FUNDING_EVENT:
            return self._normalize_vision_funding(
                self._vision.fetch_funding_rate_monthly(symbol, month.year, month.month)
            )
        indicator_map = {
            DatasetKind.PREMIUM_5M: ("premiumIndexKlines", "5m"),
            DatasetKind.MARK_1M: ("markPriceKlines", "1h"),
            DatasetKind.INDEX_1M: ("indexPriceKlines", "1h"),
        }
        if dataset is DatasetKind.KLINES_1M:
            return self._normalize_vision_klines(
                self._vision.fetch_klines_archive_monthly(symbol, "1m", month.year, month.month)
            )
        if dataset in indicator_map:
            archive, interval = indicator_map[dataset]
            frame = self._vision.fetch_indicator_klines_monthly(
                archive, symbol, interval, month.year, month.month
            )
            return self._normalize_vision_klines(frame)
        if dataset is DatasetKind.METRICS_5M:
            start = month.replace(day=1)
            end = (pd.Timestamp(start) + pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)).to_pydatetime()
            return BinanceQueryClient._normalize_timestamp(
                self._vision.fetch_range_metrics(symbol, start, end)
            )
        return pd.DataFrame()

    def download_partition(self, dataset: DatasetKind, symbol: str, start_time_ms: int = 0, **_: Any) -> bytes:
        self.download_calls += 1
        key = (dataset, symbol, start_time_ms)
        month = self._month(start_time_ms)
        frame = self._local_partition_frame(dataset, symbol, month)
        if frame.empty:
            frame = self._vision_frame(dataset, symbol, month)
        payload = self._to_parquet_bytes(frame) if not frame.empty else b""
        self._payloads[key] = payload
        return payload

    def download_checksum(self, dataset: DatasetKind, symbol: str, start_time_ms: int = 0, **_: Any) -> str:
        payload = self._payloads.get((dataset, symbol, start_time_ms))
        if payload is None:
            payload = self.download_partition(dataset, symbol, start_time_ms)
        return hashlib.sha256(payload).hexdigest()


class LocalDataCatalog:
    """Durable DuckDB manifest catalog for immutable Parquet partitions."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._database = self._root / "catalog.duckdb"
        try:
            connection = duckdb.connect(str(self._database))
        except duckdb.IOException as error:
            _logger.warning("catalog write lock unavailable; using recovery catalog: %s", error)
            self._database = self._root / "catalog_recovered.duckdb"
            connection = duckdb.connect(str(self._database))
        self._connection = connection
        connection.execute(
                """
                CREATE TABLE IF NOT EXISTS partitions (
                    dataset VARCHAR NOT NULL, symbol VARCHAR NOT NULL,
                    start_time_ms BIGINT NOT NULL, end_time_ms BIGINT NOT NULL,
                    row_count BIGINT NOT NULL, sha256 VARCHAR NOT NULL,
                    source VARCHAR NOT NULL, is_final BOOLEAN NOT NULL, path VARCHAR NOT NULL,
                    PRIMARY KEY (dataset, symbol, start_time_ms)
                )
                """
        )

    @staticmethod
    def _manifest(row: tuple[Any, ...]) -> PartitionManifest:
        return PartitionManifest(
            dataset=DatasetKind(row[0]), symbol=str(row[1]), start_time_ms=int(row[2]),
            end_time_ms=int(row[3]), row_count=int(row[4]), sha256=str(row[5]),
            source=str(row[6]), is_final=bool(row[7]), path=Path(str(row[8])),
        )

    def commit_partition(self, manifest: PartitionManifest) -> None:
        self._connection.execute(
                "INSERT OR REPLACE INTO partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [manifest.dataset.value, manifest.symbol, manifest.start_time_ms, manifest.end_time_ms,
                 manifest.row_count, manifest.sha256, manifest.source, manifest.is_final, str(manifest.path)],
        )

    def partition_exists(self, dataset: DatasetKind, symbol: str, start_time_ms: int) -> bool:
        return self._connection.execute(
                "SELECT 1 FROM partitions WHERE dataset = ? AND symbol = ? AND start_time_ms = ?",
                [dataset.value, symbol, start_time_ms],
        ).fetchone() is not None

    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._root.rglob("*.parquet"))

    def load_snapshot(self, reference_time_ms: int) -> DataSnapshot:
        rows = self._connection.execute(
                "SELECT dataset, symbol, start_time_ms, end_time_ms, row_count, sha256, source, is_final, path "
                "FROM partitions WHERE start_time_ms <= ? ORDER BY dataset, symbol, start_time_ms",
                [reference_time_ms],
        ).fetchall()
        partitions = tuple(self._manifest(row) for row in rows if Path(str(row[8])).exists())
        manifest_body = "\n".join(
            f"{p.dataset}|{p.symbol}|{p.start_time_ms}|{p.sha256}|{p.row_count}" for p in partitions
        )
        manifest_hash = hashlib.sha256(manifest_body.encode()).hexdigest()
        return DataSnapshot(
            snapshot_id=f"local-{reference_time_ms}-{manifest_hash[:12]}",
            reference_time_ms=reference_time_ms, partitions=partitions,
            manifest_hash=manifest_hash, total_bytes=sum(p.path.stat().st_size for p in partitions),
        )

    def has_complete_coverage(self, snapshot: DataSnapshot, plan: IngestionPlan) -> bool:
        if not plan.broad_symbols:
            return False
        required_datasets = tuple(
            dataset for dataset in plan.datasets
            if dataset is not DatasetKind.COST_CALIBRATION
        )
        required = {
            (dataset, symbol)
            for dataset in required_datasets
            for symbol in plan.broad_symbols
        }
        present = {(p.dataset, p.symbol) for p in snapshot.partitions if p.row_count > 0}
        return required.issubset(present)


def materialize_native_grid(*, request: GridRequest, snapshot: DataSnapshot) -> NativeFeatureGrid:
    if not request.symbols:
        raise ValueError("grid request must specify at least one symbol")
    if not request.fields:
        raise ValueError("grid request must specify at least one field")
    if request.timeframe != request.source_timeframe:
        raise ValueError("native grid requires matching request and source timeframe")
    if request.source_timeframe != "1h":
        raise ValueError(f"unsupported native timeframe: {request.source_timeframe}")

    timestamps = np.arange(request.start_time_ns, request.end_time_ns, 3_600_000_000_000, dtype=np.int64)
    fields: dict[str, NDArray[np.float64] | NDArray[np.float32]] = {
        field: np.full((len(timestamps), len(request.symbols)), np.nan, dtype=np.float64)
        for field in request.fields
    }
    available: dict[str, NDArray[np.bool_]] = {
        field: np.zeros((len(timestamps), len(request.symbols)), dtype=np.bool_)
        for field in request.fields
    }
    selected: dict[tuple[DatasetKind, str], list[Path]] = {}
    for partition in snapshot.partitions:
        if partition.dataset is DatasetKind.KLINES_1H and partition.symbol in request.symbols:
            selected.setdefault((partition.dataset, partition.symbol), []).append(partition.path)

    for column, symbol in enumerate(request.symbols):
        paths = selected.get((DatasetKind.KLINES_1H, symbol), [])
        if not paths:
            continue
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        frame = BinanceQueryClient._normalize_timestamp(frame)
        source_ns = frame["timestamp"].to_numpy(dtype=np.int64) * 1_000_000
        positions = np.searchsorted(source_ns, timestamps)
        exact = (positions < len(source_ns)) & (source_ns[np.minimum(positions, len(source_ns) - 1)] == timestamps)
        for field in request.fields:
            if field not in frame.columns:
                continue
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)
            valid = exact & np.isfinite(values[np.minimum(positions, len(values) - 1)])
            fields[field][valid, column] = values[positions[valid]]
            available[field][valid, column] = True

    return NativeFeatureGrid(timestamps_ns=timestamps, symbols=request.symbols, fields=fields, available=available, data_manifest_hash=snapshot.manifest_hash)
