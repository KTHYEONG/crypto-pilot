# mypy: ignore-errors
# ruff: noqa: F401, F821, I001, E402
from __future__ import annotations  # mypy: ignore-errors

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _regime_reference_characterization(close: pd.Series) -> dict[str, float] | None:
    """Pure-function regime descriptor for a reference symbol's 1h close series.

    Computes annualized realized volatility, total return, and 24h direction
    flip rate.  Returns ``None`` when fewer than 49 non-null bars remain after
    ``dropna`` (need >=1 full 24h-return pair beyond the 24-bar lookback).
    """
    clean = close.dropna()
    if len(clean) < 49:
        return None
    log_ret = np.log(clean).diff().dropna()
    ann_vol = float(log_ret.std(ddof=1) * np.sqrt(24 * 365))
    total_ret = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    roll_24h = clean.pct_change(24).dropna()
    flip_signs = np.abs(np.diff(np.sign(roll_24h))) > 0
    flip_rate = float(np.mean(flip_signs))
    return {
        "annualized_realized_vol": ann_vol,
        "total_return": total_ret,
        "direction_flip_rate_24h": flip_rate,
    }


def _load_reference_close(
    root: str, start: pd.Timestamp, end: pd.Timestamp, reference_symbol: str = "BTCUSDT",
) -> pd.Series | None:
    """I/O wrapper: reads one reference symbol's 1h close over ``[start, end]``.

    Independent, exogenous price level for regime labelling (never the
    strategy's own equity) -- see ``causal_regime_labels``. Returns ``None``
    when the reference parquet is absent so the caller can fail open into no
    regime evidence rather than raise.
    """
    parquet_path = Path(root) / "1h" / f"{reference_symbol}.parquet"
    if not parquet_path.exists():
        return None
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    table = pq.read_table(
        str(parquet_path),
        columns=["timestamp", "close"],
        filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
    )
    df = table.to_pandas().sort_values("timestamp").reset_index(drop=True)
    return pd.Series(
        df["close"].to_numpy(dtype="float64"),
        index=pd.to_datetime(df["timestamp"], unit="ms", utc=True),
    )


def _fold_regime_characterization(
    root: str, fold: AnchoredPurgedFold, reference_symbol: str = "BTCUSDT",
) -> dict[str, float] | None:
    """I/O wrapper: reads one reference symbol's 1h parquet for a fold's validation window."""
    parquet_path = Path(root) / "1h" / f"{reference_symbol}.parquet"
    if not parquet_path.exists():
        return None
    start_ms = int(fold.validation_start.timestamp() * 1000)
    end_ms = int(fold.validation_end.timestamp() * 1000)
    table = pq.read_table(
        str(parquet_path),
        columns=["timestamp", "close"],
        filters=[
            [("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)],
        ],
    )
    df = table.to_pandas().sort_values("timestamp").reset_index(drop=True)
    close = pd.Series(df["close"].to_numpy(dtype="float64"), index=pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return _regime_reference_characterization(close)


