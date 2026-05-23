from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.optimization.optimizer import compute_multi_alignment_info


@dataclass(slots=True, frozen=True)
class AlignedMarketData:
    """Aligned market arrays on common time grid."""

    datetimes: NDArray[np.datetime64]
    symbols: tuple[str, ...]
    open_2d: NDArray[np.float64]
    high_2d: NDArray[np.float64]
    low_2d: NDArray[np.float64]
    close_2d: NDArray[np.float64]
    volume_2d: NDArray[np.float64]
    funding_2d: NDArray[np.float64]
    active_mask: NDArray[np.bool_]
    warm_mask: NDArray[np.bool_]
    entry_block_mask: NDArray[np.bool_]
    kill_mask: NDArray[np.bool_]
    basis_2d: NDArray[np.float64] | None = None
    oi_2d: NDArray[np.float64] | None = None
    adv_usdt_2d: NDArray[np.float64] | None = None
    execution_cost_bps_2d: NDArray[np.float64] | None = None


def align_data_maps(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> AlignedMarketData:
    """Align symbol frames into dense [T, N] arrays."""
    info = compute_multi_alignment_info(data_maps, symbols, tf, embargo=0)
    if info is None:
        raise ValueError("unable to align data maps")
    eff_len = int(info["eff_ref_len"])
    offsets: dict[str, int] = info["alignment_offsets"]
    valid_symbols = tuple(
        sym for sym in symbols if sym in offsets and sym in data_maps and tf in data_maps[sym]
    )
    if not valid_symbols:
        raise ValueError("no symbols available after alignment")

    n = len(valid_symbols)
    open_2d: NDArray[np.float64] = np.zeros((eff_len, n), dtype=np.float64)
    high_2d: NDArray[np.float64] = np.zeros((eff_len, n), dtype=np.float64)
    low_2d: NDArray[np.float64] = np.zeros((eff_len, n), dtype=np.float64)
    close_2d: NDArray[np.float64] = np.zeros((eff_len, n), dtype=np.float64)
    volume_2d: NDArray[np.float64] = np.zeros((eff_len, n), dtype=np.float64)
    funding_2d: NDArray[np.float64] = np.zeros((eff_len, n), dtype=np.float64)
    basis_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    oi_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    adv_usdt_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    execution_cost_bps_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    active_mask: NDArray[np.bool_] = np.ones((eff_len, n), dtype=bool)
    warm_mask: NDArray[np.bool_] = np.ones((eff_len, n), dtype=bool)
    entry_block_mask: NDArray[np.bool_] = np.zeros((eff_len, n), dtype=bool)
    kill_mask: NDArray[np.bool_] = np.zeros((eff_len, n), dtype=bool)
    datetimes: NDArray[np.datetime64] | None = None

    for col, sym in enumerate(valid_symbols):
        frame = data_maps[sym][tf]
        start = int(offsets[sym])
        end = start + eff_len
        for required in ("open", "high", "low", "close", "volume", "datetime"):
            if required not in frame.columns:
                raise ValueError(f"missing required column: {required} symbol={sym}")
        open_2d[:, col] = frame["open"].iloc[start:end].to_numpy(dtype=np.float64)
        high_2d[:, col] = frame["high"].iloc[start:end].to_numpy(dtype=np.float64)
        low_2d[:, col] = frame["low"].iloc[start:end].to_numpy(dtype=np.float64)
        close_2d[:, col] = frame["close"].iloc[start:end].to_numpy(dtype=np.float64)
        volume_2d[:, col] = frame["volume"].iloc[start:end].to_numpy(dtype=np.float64)
        if "funding_rate_sum" in frame.columns:
            funding_2d[:, col] = frame["funding_rate_sum"].iloc[start:end].to_numpy(
                dtype=np.float64
            )
        elif "funding_rate" in frame.columns:
            funding_2d[:, col] = frame["funding_rate"].iloc[start:end].to_numpy(dtype=np.float64)
        if "basis" in frame.columns:
            basis_2d[:, col] = frame["basis"].iloc[start:end].to_numpy(dtype=np.float64)
        elif "basis_rate" in frame.columns:
            basis_2d[:, col] = frame["basis_rate"].iloc[start:end].to_numpy(dtype=np.float64)
        if "open_interest" in frame.columns:
            oi_2d[:, col] = frame["open_interest"].iloc[start:end].to_numpy(dtype=np.float64)
        elif "oi" in frame.columns:
            oi_2d[:, col] = frame["oi"].iloc[start:end].to_numpy(dtype=np.float64)
        if "adv_usdt" in frame.columns:
            adv_usdt_2d[:, col] = frame["adv_usdt"].iloc[start:end].to_numpy(dtype=np.float64)
        if "execution_cost_bps" in frame.columns:
            execution_cost_bps_2d[:, col] = frame["execution_cost_bps"].iloc[start:end].to_numpy(
                dtype=np.float64
            )
        if "universe_active_mask" in frame.columns:
            active_mask[:, col] = frame["universe_active_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if "universe_entry_warm_mask" in frame.columns:
            warm_mask[:, col] = frame["universe_entry_warm_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if "entry_block_mask" in frame.columns:
            entry_block_mask[:, col] = frame["entry_block_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if "kill_signal" in frame.columns:
            kill_mask[:, col] = frame["kill_signal"].iloc[start:end].to_numpy(dtype=bool)
        if datetimes is None:
            datetimes = np.asarray(
                frame["datetime"].iloc[start:end].to_numpy(),
                dtype="datetime64[ns]",
            )

    if datetimes is None:
        raise ValueError("datetime alignment failed")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=valid_symbols,
        open_2d=open_2d,
        high_2d=high_2d,
        low_2d=low_2d,
        close_2d=close_2d,
        volume_2d=volume_2d,
        funding_2d=funding_2d,
        basis_2d=basis_2d,
        oi_2d=oi_2d,
        adv_usdt_2d=adv_usdt_2d,
        execution_cost_bps_2d=execution_cost_bps_2d,
        active_mask=active_mask,
        warm_mask=warm_mask,
        entry_block_mask=entry_block_mask,
        kill_mask=kill_mask,
    )
