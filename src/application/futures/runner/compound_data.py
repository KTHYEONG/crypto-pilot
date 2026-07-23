from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.compound_universe import DailyPITUniverse
from src.core.settings import FUTURES_DATA_DIR
from src.domain.futures.compound.contracts import MarketFeatureCube
from src.domain.futures.data_lake.contracts import DataSnapshot

_logger = logging.getLogger(__name__)


def _to_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _load_symbol_ohlcv(
    symbol: str,
    timeframe: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame | None:
    data_dir = base_dir or FUTURES_DATA_DIR
    path = data_dir / "ohlcv" / timeframe / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        dt_col = None
        if "datetime" in df.columns:
            dt_col = "datetime"
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            dt_col = "datetime"
        if dt_col is None:
            return None
        if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df[dt_col] = pd.to_datetime(df[dt_col], ut=True)
        df = df.sort_values(dt_col).reset_index(drop=True)
        mask = (df[dt_col] >= start_dt) & (df[dt_col] <= end_dt)
        return df.loc[mask].copy()
    except Exception as exc:
        _logger.warning("failed to load %s %s: %s", symbol, timeframe, exc)
        return None


def _load_symbol_funding(
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame | None:
    data_dir = base_dir or FUTURES_DATA_DIR
    path = data_dir / "funding" / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        if "datetime" not in df.columns and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        if "datetime" not in df.columns:
            return None
        dt_col = "datetime"
        if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df[dt_col] = pd.to_datetime(df[dt_col], ut=True)
        df = df.sort_values(dt_col).reset_index(drop=True)
        mask = (df[dt_col] >= start_dt) & (df[dt_col] <= end_dt)
        return df.loc[mask].copy()
    except Exception as exc:
        _logger.warning("failed to load funding %s: %s", symbol, exc)
        return None


def _load_symbol_enriched(
    symbol: str,
    timeframe: str,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame | None:
    data_dir = base_dir or FUTURES_DATA_DIR
    path = data_dir / "enriched" / timeframe / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df
    except Exception as exc:
        _logger.warning("failed to load enriched %s %s: %s", symbol, timeframe, exc)
        return None


def _load_symbol_metrics(
    symbol: str,
    *,
    base_dir: Path | None = None,
) -> pd.DataFrame | None:
    data_dir = base_dir or FUTURES_DATA_DIR
    path = data_dir / "metrics" / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as exc:
        _logger.warning("failed to load metrics %s: %s", symbol, exc)
        return None


def load_compound_market_data(
    *,
    config: CompoundRunConfig,
    symbols: tuple[str, ...],
    execution_calendar: pd.DatetimeIndex,
) -> dict[str, dict[str, pd.DataFrame]]:
    ref_date = _to_date(config.reference_date) if config.reference_date else date.today()
    start_dt = pd.Timestamp(ref_date - pd.Timedelta(days=config.history_days), tz="UTC")
    end_dt = pd.Timestamp(ref_date, tz="UTC") + pd.Timedelta(hours=23)

    data_maps: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        sym_data: dict[str, pd.DataFrame] = {}

        ohlcv = _load_symbol_ohlcv(sym, "1h", start_dt, end_dt)
        if ohlcv is not None:
            sym_data["1h"] = ohlcv

        funding = _load_symbol_funding(sym, start_dt, end_dt)
        if funding is not None:
            sym_data["funding"] = funding

        enriched = _load_symbol_enriched(sym, "1h")
        if enriched is not None:
            sym_data["enriched_1h"] = enriched

        metrics = _load_symbol_metrics(sym)
        if metrics is not None:
            sym_data["metrics"] = metrics

        data_maps[sym] = sym_data

    return data_maps


def check_data_readiness(
    data_maps: dict[str, dict[str, pd.DataFrame]],
    *,
    min_ready_pct: float = 0.80,
) -> bool:
    if not data_maps:
        return False
    ready = sum(1 for sym_data in data_maps.values() if "1h" in sym_data and not sym_data["1h"].empty)
    ratio = ready / max(len(data_maps), 1)
    if ratio < min_ready_pct:
        _logger.error(
            "data readiness %.1f%% < %.1f%% (%d/%d symbols ready)",
            ratio * 100, min_ready_pct * 100, ready, len(data_maps),
        )
        return False
    return True


def build_multiscale_market_cube(
    *, snapshot: DataSnapshot, universe: DailyPITUniverse, config: CompoundRunConfig
) -> MarketFeatureCube:
    _logger.info(
        "building multiscale market cube: %d symbols from snapshot %s",
        len(universe.symbols), snapshot.snapshot_id,
    )
    symbols = universe.symbols
    n_syms = len(symbols)

    from datetime import UTC, datetime

    ref_date_str = config.reference_date or datetime.now(UTC).strftime("%Y-%m-%d")
    ref_dt = pd.Timestamp(ref_date_str, tz="UTC")
    start_dt = ref_dt - pd.Timedelta(days=config.history_days)
    n_bars = config.history_days * 24
    execution_calendar = pd.date_range(start=start_dt, periods=n_bars, freq="h", tz="UTC")
    timestamps_ns = execution_calendar.asi8.astype(np.int64)

    core_names = ("open", "high", "low", "close", "quote_volume")
    arrays: dict[str, NDArray[np.float64]] = {
        name: np.full((n_bars, n_syms), np.nan, dtype=np.float64)
        for name in core_names
    }
    funding = np.zeros((n_bars, n_syms), dtype=np.float32)
    available_core = np.zeros((n_bars, n_syms), dtype=np.bool_)
    eligible = np.ones((n_bars, n_syms), dtype=np.bool_)
    entry_block = np.zeros((n_bars, n_syms), dtype=np.bool_)
    exit_required = np.zeros((n_bars, n_syms), dtype=np.bool_)
    capacity = np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64)
    costs = np.full((n_bars, n_syms), 12.0, dtype=np.float32)

    for col, symbol in enumerate(symbols):
        ohlcv = _load_symbol_ohlcv(symbol, "1h", start_dt, ref_dt)
        if ohlcv is not None and not ohlcv.empty:
            frame = ohlcv.sort_values("datetime").reset_index(drop=True)
            source_ns = frame["datetime"].values.astype("datetime64[ns]").astype(np.int64)
            positions = np.searchsorted(source_ns, timestamps_ns, side="right") - 1
            valid_pos = positions >= 0
            for name in core_names:
                if name in frame.columns:
                    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
                    arrays[name][valid_pos, col] = values[positions[valid_pos]]
            funding_col = None
            if "funding_rate_sum" in frame.columns:
                funding_col = frame["funding_rate_sum"]
            elif "funding_rate" in frame.columns:
                funding_col = frame["funding_rate"]
            if funding_col is not None:
                funding[valid_pos, col] = pd.to_numeric(funding_col, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)[positions[valid_pos]]
            available_core[:, col] = valid_pos & np.all(
                np.isfinite(np.column_stack([arrays[name][:, col] for name in core_names])),
                axis=1,
            )

    fields: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {
        **arrays,
        "quote_volume": arrays["quote_volume"].astype(np.float32),
        "funding": funding,
    }

    cube = MarketFeatureCube(
        timestamps_ns=timestamps_ns,
        symbols=symbols,
        fields_2d=fields,
        available_2d={"core": available_core},
        eligible_2d=eligible,
        entry_block_2d=entry_block,
        exit_required_2d=exit_required,
        capacity_usdt_2d=capacity,
        execution_cost_bps_2d=costs,
        data_manifest_hash=snapshot.manifest_hash,
    )

    _logger.info(
        "multiscale market cube built: %d bars x %d symbols",
        n_bars, n_syms,
    )
    return cube


__all__ = [
    "build_multiscale_market_cube",
    "check_data_readiness",
    "load_compound_market_data",
]
