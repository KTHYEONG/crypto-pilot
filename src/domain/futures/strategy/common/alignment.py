from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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
    taker_buy_2d: NDArray[np.float64] | None = None
    trades_2d: NDArray[np.float64] | None = None
    adv_usdt_2d: NDArray[np.float64] | None = None
    execution_cost_bps_2d: NDArray[np.float64] | None = None
    # Phase D: C1 inference panel 전용 마스크 (Stage5 timeline 기반). None이면 미사용.
    inference_active_mask: NDArray[np.bool_] | None = None
    inference_entry_warm_mask: NDArray[np.bool_] | None = None
    vol_30d_1d: NDArray[np.float32] | None = None
    friction_score_1d: NDArray[np.float32] | None = None
    alpha_capacity_score_1d: NDArray[np.float32] | None = None
    diversification_score_1d: NDArray[np.float32] | None = None
    tradeable_score_1d: NDArray[np.float32] | None = None
    cluster_id_1d: NDArray[np.float32] | None = None
    beta_vs_market_1d: NDArray[np.float32] | None = None
    cluster_size_1d: NDArray[np.float32] | None = None
    anchor_cluster_1d: NDArray[np.float32] | None = None
    # Phase D/E: per-symbol 정적 메타데이터 (sample weighting용). {col: [N] array}
    symbol_meta: dict[str, NDArray[np.float32]] | None = None



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
    taker_buy_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    trades_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    adv_usdt_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    execution_cost_bps_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
    active_mask: NDArray[np.bool_] = np.ones((eff_len, n), dtype=bool)
    warm_mask: NDArray[np.bool_] = np.ones((eff_len, n), dtype=bool)
    entry_block_mask: NDArray[np.bool_] = np.zeros((eff_len, n), dtype=bool)
    kill_mask: NDArray[np.bool_] = np.zeros((eff_len, n), dtype=bool)
    # Phase D: inference 마스크 초기화 — 데이터에 컬럼 있을 때만 채움
    _inf_active: NDArray[np.bool_] | None = None
    _inf_warm: NDArray[np.bool_] | None = None
    # Phase D/E: per-symbol 메타 컬럼 수집 (coverage, cluster, beta 등)
    _meta_cols_to_read: tuple[str, ...] = (
        "coverage_60d",
        "last_60d_coverage",
        "vol_30d",
        "friction_score",
        "alpha_capacity_score",
        "diversification_score",
        "tradeable_score",
        "cluster_id",
        "beta_vs_market",
        "cluster_size",
        "anchor_cluster_member",
    )
    _sym_meta_lists: dict[str, list[float]] = {col: [] for col in _meta_cols_to_read}
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
        if "taker_buy_base" in frame.columns:
            taker_buy_2d[:, col] = frame["taker_buy_base"].iloc[start:end].to_numpy(dtype=np.float64)
        elif "taker_buy_quote" in frame.columns:
            taker_buy_2d[:, col] = frame["taker_buy_quote"].iloc[start:end].to_numpy(dtype=np.float64)
        if "trades" in frame.columns:
            trades_2d[:, col] = frame["trades"].iloc[start:end].to_numpy(dtype=np.float64)
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
        # Phase D/E: per-symbol 정적 메타 (PIT-safe: aligned window의 첫 유효값 사용)
        for _mc in _meta_cols_to_read:
            if _mc in frame.columns:
                _meta_series = pd.to_numeric(frame[_mc].iloc[start:end], errors="coerce").dropna()
                _meta_value = float(_meta_series.iloc[0]) if not _meta_series.empty else float("nan")
                _sym_meta_lists[_mc].append(_meta_value)
            else:
                _sym_meta_lists[_mc].append(float("nan"))
        # Phase D: inference panel 마스크 (Stage5 timeline 기반)
        if "inference_active_mask" in frame.columns:
            if _inf_active is None:
                _inf_active = np.ones((eff_len, n), dtype=bool)
            _inf_active[:, col] = frame["inference_active_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if "inference_entry_warm_mask" in frame.columns:
            if _inf_warm is None:
                _inf_warm = np.ones((eff_len, n), dtype=bool)
            _inf_warm[:, col] = frame["inference_entry_warm_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if datetimes is None:
            datetimes = np.asarray(
                frame["datetime"].iloc[start:end].to_numpy(),
                dtype="datetime64[ns]",
            )

    if datetimes is None:
        raise ValueError("datetime alignment failed")
    symbol_meta = {
        col: np.array(vals, dtype=np.float32)
        for col, vals in _sym_meta_lists.items()
        if any(np.isfinite(v) for v in vals)
    } or None
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
        taker_buy_2d=taker_buy_2d,
        trades_2d=trades_2d,
        adv_usdt_2d=adv_usdt_2d,
        execution_cost_bps_2d=execution_cost_bps_2d,
        active_mask=active_mask,
        warm_mask=warm_mask,
        entry_block_mask=entry_block_mask,
        kill_mask=kill_mask,
        inference_active_mask=_inf_active,
        inference_entry_warm_mask=_inf_warm,
        vol_30d_1d=None if symbol_meta is None else symbol_meta.get("vol_30d"),
        friction_score_1d=None if symbol_meta is None else symbol_meta.get("friction_score"),
        alpha_capacity_score_1d=(
            None if symbol_meta is None else symbol_meta.get("alpha_capacity_score")
        ),
        diversification_score_1d=(
            None if symbol_meta is None else symbol_meta.get("diversification_score")
        ),
        tradeable_score_1d=None if symbol_meta is None else symbol_meta.get("tradeable_score"),
        cluster_id_1d=None if symbol_meta is None else symbol_meta.get("cluster_id"),
        beta_vs_market_1d=None if symbol_meta is None else symbol_meta.get("beta_vs_market"),
        cluster_size_1d=None if symbol_meta is None else symbol_meta.get("cluster_size"),
        anchor_cluster_1d=None if symbol_meta is None else symbol_meta.get("anchor_cluster_member"),
        symbol_meta=symbol_meta,
    )
