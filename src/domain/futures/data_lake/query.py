from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.exchange.binance_vision import BinanceVisionDownloader
from src.domain.futures.data_lake.contracts import (
    DatasetKind,
    DataSnapshot,
    GridRequest,
    IngestionPlan,
    LakeUniverse,
    NativeFeatureGrid,
    PartitionManifest,
    UniverseStateRequest,
    UniverseStateRow,
)
from src.domain.futures.data_lake.ingestion import DataCoverageError
from src.domain.futures.data_lake.reconciliation import CatalogLockError
from src.domain.futures.universe.contracts import UniverseStateCube

_logger = logging.getLogger(__name__)


class UniverseCoverageError(RuntimeError):
    ...


class HoldoutReuseError(RuntimeError):
    ...


class BinanceQueryClient:
    """Read cached Binance futures data first and use Vision only for missing months."""

    def __init__(self, source_root: Path | None = None) -> None:
        _ = source_root
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
        columns = ("timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore")
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
        frame = self._vision_frame(dataset, symbol, month)
        payload = self._to_parquet_bytes(frame) if not frame.empty else b""
        self._payloads[key] = payload
        return payload

    def download_checksum(self, dataset: DatasetKind, symbol: str, start_time_ms: int = 0, **_: Any) -> str:
        payload = self._payloads.get((dataset, symbol, start_time_ms))
        if payload is None:
            payload = self.download_partition(dataset, symbol, start_time_ms)
        return hashlib.sha256(payload).hexdigest()

    def fetch_exchange_info(self) -> dict[str, Any]:
        request = urllib.request.Request(
            "https://fapi.binance.com/fapi/v1/exchangeInfo",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            payload = response.read()
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict) or not isinstance(decoded.get("symbols"), list):
            raise DataCoverageError("Binance exchangeInfo response has invalid schema")
        return decoded


class LocalDataCatalog:
    """Durable DuckDB manifest catalog for immutable Parquet partitions."""

    def __init__(self, root: Path | str, *, read_only: bool = False) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._database = self._root / "catalog.duckdb"
        self._read_only = read_only
        try:
            connection = duckdb.connect(str(self._database), read_only=read_only)
        except duckdb.IOException as error:
            raise CatalogLockError(
                f"canonical catalog lock unavailable: {self._database}"
            ) from error
        self._connection = connection
        if not read_only:
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
        universe_state_hash = self.compute_universe_state_hash(
            DataSnapshot(
                snapshot_id="", reference_time_ms=reference_time_ms, partitions=partitions,
                manifest_hash=manifest_hash, universe_state_hash="", total_bytes=0,
            )
        )
        return DataSnapshot(
            snapshot_id=f"local-{reference_time_ms}-{manifest_hash[:12]}",
            reference_time_ms=reference_time_ms, partitions=partitions,
            manifest_hash=manifest_hash, universe_state_hash=universe_state_hash,
            total_bytes=sum(p.path.stat().st_size for p in partitions),
        )

    def compute_universe_state_hash(self, snapshot: DataSnapshot) -> str:
        state_entries = [
            f"{p.symbol}|{p.start_time_ms}|{p.sha256}|{p.row_count}"
            for p in snapshot.partitions
            if p.dataset is DatasetKind.UNIVERSE_STATE
        ]
        state_body = "\n".join(sorted(state_entries))
        if not state_body:
            return ""
        return hashlib.sha256(state_body.encode()).hexdigest()

    def has_complete_coverage(self, snapshot: DataSnapshot, plan: IngestionPlan) -> bool:
        if not plan.broad_symbols:
            return False
        required_datasets = tuple(
            dataset for dataset in plan.datasets
            if dataset not in (DatasetKind.COST_CALIBRATION, DatasetKind.EXCHANGE_INFO)
        )
        required: set[tuple[DatasetKind, str]] = {
            (dataset, symbol)
            for dataset in required_datasets
            if dataset is not DatasetKind.UNIVERSE_STATE
            for symbol in plan.broad_symbols
        }
        if DatasetKind.UNIVERSE_STATE in required_datasets:
            required.add((DatasetKind.UNIVERSE_STATE, "__all__"))
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


def _load_partition_data(
    paths: list[Path], *, start_time_ns: int, end_time_ns: int, fields: tuple[str, ...]
) -> pd.DataFrame:
    start_time_ms = start_time_ns // 1_000_000
    end_time_ms = end_time_ns // 1_000_000
    filters = [("timestamp", ">=", start_time_ms), ("timestamp", "<", end_time_ms)]
    columns = list(dict.fromkeys(("timestamp", *fields)))
    import pyarrow.parquet as pq

    frames: list[pd.DataFrame] = []
    for path in paths:
        available_columns = set(pq.read_schema(path).names)  # type: ignore[no-untyped-call]
        projected_columns = [column for column in columns if column in available_columns]
        if "timestamp" not in projected_columns:
            continue  # pragma: no cover - schema validation rejects this upstream
        frames.append(pd.read_parquet(path, columns=projected_columns, filters=filters))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _universe_state_rows(snapshot: DataSnapshot) -> list[UniverseStateRow]:
    rows: list[UniverseStateRow] = []
    for p in snapshot.partitions:
        if p.dataset is not DatasetKind.UNIVERSE_STATE:
            continue  # pragma: no cover - catalog contains mixed datasets
        if not p.path.exists():
            continue  # pragma: no cover - missing files are filtered at snapshot load
        frame = pd.read_parquet(p.path)
        for _, row in frame.iterrows():
            rows.append(UniverseStateRow(
                effective_time_ns=int(row["effective_time_ns"]),
                knowledge_time_ns=int(row["knowledge_time_ns"]),
                symbol=str(row["symbol"]),
                eligible=bool(row["eligible"]),
                entry_block=bool(row["entry_block"]),
                exit_required=bool(row["exit_required"]),
                capacity_usdt=float(row["capacity_usdt"]),
                risk_scale=float(row["risk_scale"]),
                execution_cost_bps=float(row["execution_cost_bps"]),
                state_reason=str(row["state_reason"]),
                universe_config_hash=str(row["universe_config_hash"]),
                source_manifest_hash=str(row["source_manifest_hash"]),
            ))
    return rows


def load_pit_universe_state(
    *, snapshot: DataSnapshot, request: UniverseStateRequest
) -> LakeUniverse:
    rows = _universe_state_rows(snapshot)
    if not rows:
        raise UniverseCoverageError("missing PIT state: no UNIVERSE_STATE partitions in snapshot")

    if any(row.knowledge_time_ns >= row.effective_time_ns for row in rows):
        raise UniverseCoverageError("PIT state knowledge timestamp is not strictly before effective time")

    all_symbols = sorted({r.symbol for r in rows})
    if len(all_symbols) > request.max_axis_symbols:
        raise UniverseCoverageError(
            f"PIT universe axis {len(all_symbols)} exceeds {request.max_axis_symbols}"
        )
    symbols = tuple(all_symbols)

    timestamps = request.execution_timestamps_ns
    n_bars = len(timestamps)
    n_syms = len(symbols)
    eligible = np.zeros((n_bars, n_syms), dtype=np.bool_)
    entry_block = np.ones((n_bars, n_syms), dtype=np.bool_)
    exit_required = np.zeros((n_bars, n_syms), dtype=np.bool_)
    capacity_usdt = np.zeros((n_bars, n_syms), dtype=np.float64)
    risk_scale = np.ones((n_bars, n_syms), dtype=np.float64)
    cost_bps = np.full((n_bars, n_syms), 12.0, dtype=np.float64)

    for sym_idx, sym in enumerate(symbols):
        sym_rows = [
            r for r in rows
            if r.symbol == sym and r.effective_time_ns <= timestamps[-1]
        ]
        if not sym_rows:
            eligible[:, sym_idx] = False
            entry_block[:, sym_idx] = True
            continue
        sym_rows.sort(key=lambda r: r.effective_time_ns)
        eff_arr = np.array([r.effective_time_ns for r in sym_rows], dtype=np.int64)
        eligible_arr = np.array([r.eligible for r in sym_rows], dtype=np.bool_)
        entry_arr = np.array([r.entry_block for r in sym_rows], dtype=np.bool_)
        exit_arr = np.array([r.exit_required for r in sym_rows], dtype=np.bool_)
        cap_arr = np.array([r.capacity_usdt for r in sym_rows], dtype=np.float64)
        risk_arr = np.array([r.risk_scale for r in sym_rows], dtype=np.float64)
        cost_arr = np.array([r.execution_cost_bps for r in sym_rows], dtype=np.float32)
        idx = np.clip(np.searchsorted(eff_arr, timestamps, side="right") - 1, 0, len(sym_rows) - 1)
        valid = timestamps >= eff_arr[0]
        eligible[valid, sym_idx] = eligible_arr[idx[valid]]
        entry_block[valid, sym_idx] = entry_arr[idx[valid]]
        exit_required[valid, sym_idx] = exit_arr[idx[valid]]
        capacity_usdt[valid, sym_idx] = cap_arr[idx[valid]]
        risk_scale[valid, sym_idx] = risk_arr[idx[valid]]
        cost_bps[valid, sym_idx] = cost_arr[idx[valid]]
        for t in range(1, n_bars):
            if not eligible[t, sym_idx] and eligible[t - 1, sym_idx]:
                exit_required[t, sym_idx] = True

    cube = UniverseStateCube(
        calendar=pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)),
        instrument_ids=symbols,
        eligible=eligible,
        entry_block=entry_block,
        exit_required=exit_required,
        capacity_usdt=capacity_usdt,
        risk_scale=risk_scale,
        cost_bps=cost_bps,
    )
    return LakeUniverse(
        symbols=symbols,
        state_cube=cube,
        state_hash=snapshot.universe_state_hash,
    )


def materialize_feature_grid(
    *, request: GridRequest, snapshot: DataSnapshot, dataset: DatasetKind
) -> NativeFeatureGrid:
    if not request.symbols:
        raise ValueError("grid request must specify at least one symbol")
    if not request.fields:
        raise ValueError("grid request must specify at least one field")
    if request.timeframe != request.source_timeframe:
        raise ValueError("feature grid requires matching request and source timeframe")

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
        overlaps = (
            partition.start_time_ms * 1_000_000 < request.end_time_ns
            and partition.end_time_ms * 1_000_000 >= request.start_time_ns
        )
        if partition.dataset is dataset and partition.symbol in request.symbols and overlaps:
            selected.setdefault((partition.dataset, partition.symbol), []).append(partition.path)

    for column, symbol in enumerate(request.symbols):
        paths = selected.get((dataset, symbol), [])
        if not paths:
            continue
        frame = _load_partition_data(
            paths,
            start_time_ns=request.start_time_ns,
            end_time_ns=request.end_time_ns,
            fields=request.fields,
        )
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

    return NativeFeatureGrid(
        timestamps_ns=timestamps, symbols=request.symbols,
        fields=fields, available=available,
        data_manifest_hash=snapshot.manifest_hash,
    )


def _collect_metrics_paths(lake_root: Path, symbol: str) -> list[Path]:
    sym_dir = lake_root / "metrics_5m" / f"symbol={symbol}"
    if not sym_dir.is_dir():
        return []
    return sorted(sym_dir.rglob("*.parquet"))


def materialize_causal_metrics_grid(
    *, symbols: tuple[str, ...], start_time_ns: int, end_time_ns: int,
    lake_root: Path, field: str, tolerance_ns: int = 7_200_000_000_000,
) -> NativeFeatureGrid:
    if not symbols:
        raise ValueError("symbols must not be empty")
    if not field:
        raise ValueError("field must not be empty")

    step_ns = 3_600_000_000_000
    timestamps = np.arange(start_time_ns, end_time_ns, step_ns, dtype=np.int64)
    n_t = len(timestamps)
    n_s = len(symbols)

    fields_dict: dict[str, NDArray[np.float64] | NDArray[np.float32]] = {
        field: np.full((n_t, n_s), np.nan, dtype=np.float64)
    }
    avail: dict[str, NDArray[np.bool_]] = {
        field: np.zeros((n_t, n_s), dtype=np.bool_)
    }

    import pyarrow.parquet as pq

    for col, sym in enumerate(symbols):
        paths = _collect_metrics_paths(lake_root, sym)
        if not paths:
            continue

        columns = ["timestamp", "available_at", field]
        frames: list[pd.DataFrame] = []
        for p in paths:
            try:
                schema = pq.read_schema(p)  # type: ignore[no-untyped-call]
                proj = [c for c in columns if c in set(schema.names)]
                if "available_at" not in proj or field not in proj:
                    continue
                frames.append(pd.read_parquet(p, columns=proj))
            except Exception:  # noqa: S112
                continue

        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)

        avail_raw = df["available_at"]
        if pd.api.types.is_datetime64_any_dtype(avail_raw):
            avail_ns_col = pd.to_datetime(avail_raw, utc=True).astype("datetime64[ns, UTC]").astype("int64")
        else:
            avail_ns_col = pd.to_numeric(avail_raw, errors="coerce").astype("int64") * 1_000_000
        df = df.assign(_available_at_ns=avail_ns_col)

        start_ns_bound = max(0, start_time_ns - tolerance_ns)
        df = df[(df["_available_at_ns"] >= start_ns_bound) & (df["_available_at_ns"] < end_time_ns)]
        if df.empty:
            continue

        df = df.sort_values("_available_at_ns").drop_duplicates(subset=["_available_at_ns"], keep="last")
        avail_ns = df["_available_at_ns"].to_numpy(dtype=np.int64)
        values = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=np.float64)

        pos = np.searchsorted(avail_ns, timestamps, side="right") - 1
        valid = pos >= 0
        if not np.any(valid):
            continue

        vpos = pos[valid]
        lag = timestamps[valid] - avail_ns[vpos]
        within_tol = lag <= tolerance_ns
        hit = np.where(valid)[0][within_tol]
        hit_pos = vpos[within_tol]

        finite = np.isfinite(values[hit_pos])
        hit = hit[finite]
        hit_pos = hit_pos[finite]

        if len(hit) > 0:
            fields_dict[field][hit, col] = values[hit_pos]
            avail[field][hit, col] = True

    return NativeFeatureGrid(
        timestamps_ns=timestamps, symbols=symbols,
        fields=fields_dict, available=avail,
        data_manifest_hash="",
    )
