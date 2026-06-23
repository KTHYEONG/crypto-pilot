from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.optimization.optimizer import compute_multi_alignment_info
from src.domain.futures.universe.contracts import UniverseStateCube

_logger = logging.getLogger(__name__)


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
    lsr_2d: NDArray[np.float64] | None = None
    taker_buy_2d: NDArray[np.float64] | None = None
    trades_2d: NDArray[np.float64] | None = None
    adv_usdt_2d: NDArray[np.float64] | None = None
    execution_cost_bps_2d: NDArray[np.float64] | None = None
    # Phase D: C1 inference panel 전용 마스크 (Stage5 timeline 기반). None이면 미사용.
    inference_active_mask: NDArray[np.bool_] | None = None
    inference_entry_warm_mask: NDArray[np.bool_] | None = None
    execution_eligibility_mask: NDArray[np.bool_] | None = None
    strategy_readiness_mask: NDArray[np.bool_] | None = None
    promotion_active_mask: NDArray[np.bool_] | None = None
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



_ALIGNED_DATA_MAPS_CACHE: dict[
    tuple[int, tuple[str, ...], str],
    tuple[AlignedMarketData, dict[str, tuple[int, int]]],
] = {}


def align_data_maps(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str] | tuple[str, ...],
    tf: str,
    *,
    state_cube: UniverseStateCube | None = None,
    readiness_strategy: str | None = None,
    readiness_cube: Any | None = None,
) -> AlignedMarketData:
    """Align symbol frames into dense [T, N] arrays.

    Args:
        data_maps: Per-symbol per-timeframe DataFrames.
        symbols: Symbol list to align.
        tf: Timeframe key (e.g. "4h").
        state_cube: Optional PIT universe state cube. When provided, overwrites
            ``active_mask``, ``entry_block_mask``, ``adv_usdt_2d``, and
            ``execution_cost_bps_2d`` via vectorized forward-asof join.
            Cache is bypassed when state_cube is not None.
        readiness_strategy: Strategy name for readiness_cube lookup.
        readiness_cube: Optional StrategyReadinessCube; fills
            ``strategy_readiness_mask`` when both this and readiness_strategy
            are supplied.

    Returns:
        AlignedMarketData with [T, N] arrays on the common time grid.
    """
    # Skip cache when state_cube is provided — result depends on external mutable cube.
    if state_cube is None:
        cache_key = (id(data_maps), tuple(sorted(symbols)), tf)
        if cache_key in _ALIGNED_DATA_MAPS_CACHE:
            cached_val, expected_shapes = _ALIGNED_DATA_MAPS_CACHE[cache_key]
            match = True
            for sym in symbols:
                if sym in data_maps and tf in data_maps[sym]:
                    df = data_maps[sym][tf]
                    if sym not in expected_shapes or (len(df), len(df.columns)) != expected_shapes[sym]:
                        match = False
                        break
                else:
                    if sym in expected_shapes:
                        match = False
                        break
            if match:
                return cached_val

    info = compute_multi_alignment_info(data_maps, list(symbols), tf, embargo=0)

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
    lsr_2d: NDArray[np.float64] = np.full((eff_len, n), np.nan, dtype=np.float64)
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
    _execution_eligibility: NDArray[np.bool_] | None = None
    _strategy_readiness: NDArray[np.bool_] | None = None
    _promotion_active: NDArray[np.bool_] | None = None
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
        if "sum_open_interest" in frame.columns:
            oi_2d[:, col] = frame["sum_open_interest"].iloc[start:end].to_numpy(dtype=np.float64)
        elif "open_interest" in frame.columns:
            oi_2d[:, col] = frame["open_interest"].iloc[start:end].to_numpy(dtype=np.float64)
        elif "oi" in frame.columns:
            oi_2d[:, col] = frame["oi"].iloc[start:end].to_numpy(dtype=np.float64)
        if "long_short_ratio" in frame.columns:
            lsr_2d[:, col] = frame["long_short_ratio"].iloc[start:end].to_numpy(dtype=np.float64)
        elif "global_long_short_ratio" in frame.columns:
            lsr_2d[:, col] = frame["global_long_short_ratio"].iloc[start:end].to_numpy(dtype=np.float64)
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
                # Fast numpy-backed array scanning to extract first non-NaN value
                arr = (
                    frame[_mc].values[start:end]
                    if hasattr(frame[_mc], "values")
                    else frame[_mc].iloc[start:end].to_numpy()
                )
                mask_valid = ~np.isnan(arr)
                _meta_value = float(arr[mask_valid][0]) if mask_valid.any() else float("nan")
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
        if "execution_eligibility_mask" in frame.columns:
            if _execution_eligibility is None:
                _execution_eligibility = np.ones((eff_len, n), dtype=bool)
            _execution_eligibility[:, col] = frame["execution_eligibility_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if "strategy_readiness_mask" in frame.columns:
            if _strategy_readiness is None:
                _strategy_readiness = np.ones((eff_len, n), dtype=bool)
            _strategy_readiness[:, col] = frame["strategy_readiness_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if "promotion_active_mask" in frame.columns:
            if _promotion_active is None:
                _promotion_active = np.ones((eff_len, n), dtype=bool)
            _promotion_active[:, col] = frame["promotion_active_mask"].iloc[start:end].to_numpy(
                dtype=bool
            )
        if datetimes is None:
            datetimes = np.asarray(
                frame["datetime"].iloc[start:end].to_numpy(),
                dtype="datetime64[ns]",
            )

    if datetimes is None:
        raise ValueError("datetime alignment failed")

    # ── PIT state_cube join ──────────────────────────────────────────────────
    # Vectorized forward-asof: for each aligned bar t, find latest cube bar
    # with cube_ts <= datetimes[t].  O(N * log(T_cube)) — no Python loop over t.
    # Vectorization of readiness_cube join is deferred (same pattern applies).
    if state_cube is not None:
        aligned_ts_ns = datetimes.astype("datetime64[ns]").view(np.int64)
        cube_ts_ns = np.asarray(state_cube.calendar.as_unit("ns").asi8, dtype=np.int64)
        cube_sym_idx: dict[str, int] = {}
        for _cube_n, _cube_iid in enumerate(state_cube.instrument_ids):
            _sym_key = _cube_iid.split(":")[-1] if ":" in _cube_iid else _cube_iid
            cube_sym_idx[_sym_key] = _cube_n

        _logger.debug(
            "[ALIGN-CUBE] injecting state_cube: calendar_range=[%s, %s] instruments=%d "
            "aligned_datetimes=%d symbols=%d cube_valid_symbols=%d",
            state_cube.calendar[0],
            state_cube.calendar[-1],
            len(state_cube.instrument_ids),
            len(datetimes),
            n,
            sum(1 for sym in valid_symbols if sym in cube_sym_idx),
        )
        # OPT-1: hoist loop-invariant searchsorted outside the col/sym loop.
        # Complexity: O(T·log T_cube + N·V) vs previous O(N·T·log T_cube).
        # positions: [T]  valid_pos_mask: [T bool]  t_valid/p_valid: [V]
        positions = np.searchsorted(cube_ts_ns, aligned_ts_ns, side="right") - 1
        valid_pos_mask = positions >= 0
        t_valid = np.where(valid_pos_mask)[0]
        p_valid = positions[valid_pos_mask]
        if t_valid.size == 0:
            pass  # all columns keep initial values; loop body becomes no-op
        else:
            for col, sym in enumerate(valid_symbols):
                if sym not in cube_sym_idx:
                    continue
                cube_n = cube_sym_idx[sym]
                active_mask[t_valid, col] = state_cube.eligible[p_valid, cube_n]
                entry_block_mask[t_valid, col] = state_cube.entry_block[p_valid, cube_n]
                adv_usdt_2d[t_valid, col] = state_cube.capacity_usdt[p_valid, cube_n]
                execution_cost_bps_2d[t_valid, col] = state_cube.cost_bps[p_valid, cube_n]
        active_ratio = float(active_mask.mean())
        if active_ratio < 0.99:
            _logger.debug(
                "[ALIGN-CUBE] post-join active_mask mean=%.4f entry_block_mean=%.4f "
                "(was 1.0 / 0.0 before cube injection)",
                active_ratio,
                float(entry_block_mask.mean()),
            )
    # ── end state_cube join ──────────────────────────────────────────────────

    # ── PIT readiness_cube join ───────────────────────────────────────────────
    if readiness_cube is not None and readiness_strategy is not None:
        strat_names: tuple[str, ...] = getattr(readiness_cube, "strategies", ())
        if readiness_strategy in strat_names:
            s_idx = list(strat_names).index(readiness_strategy)
            r_calendar = getattr(readiness_cube, "calendar", None)
            r_iid_tuple: tuple[str, ...] = getattr(readiness_cube, "instrument_ids", ())
            r_sym_idx: dict[str, int] = {
                (iid.split(":")[-1] if ":" in iid else iid): r_n
                for r_n, iid in enumerate(r_iid_tuple)
            }
            if r_calendar is not None:
                r_cal_ns = np.asarray(
                    pd.DatetimeIndex(r_calendar, tz="UTC").view(np.int64),
                    dtype=np.int64,
                )
                aligned_ts_ns_r = datetimes.astype("datetime64[ns]").view(np.int64)
                if _strategy_readiness is None:
                    _strategy_readiness = np.ones((eff_len, n), dtype=np.bool_)
                # OPT-1: hoist readiness searchsorted outside col/sym loop.
                # positions_r: [T]  t_valid_r/p_valid_r: [V_r]
                positions_r = np.searchsorted(r_cal_ns, aligned_ts_ns_r, side="right") - 1
                valid_r_mask = positions_r >= 0
                t_valid_r = np.where(valid_r_mask)[0]
                p_valid_r = positions_r[valid_r_mask]
                if t_valid_r.size > 0:
                    for col, sym in enumerate(valid_symbols):
                        if sym not in r_sym_idx:
                            continue
                        r_n = r_sym_idx[sym]
                        _strategy_readiness[t_valid_r, col] = readiness_cube.ready[
                            s_idx, p_valid_r, r_n
                        ]
    # ── end readiness_cube join ───────────────────────────────────────────────

    symbol_meta = {
        col: np.array(vals, dtype=np.float32)
        for col, vals in _sym_meta_lists.items()
        if any(np.isfinite(v) for v in vals)
    } or None
    result = AlignedMarketData(
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
        lsr_2d=lsr_2d,
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
        execution_eligibility_mask=_execution_eligibility,
        strategy_readiness_mask=_strategy_readiness,
        promotion_active_mask=_promotion_active,
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
    if state_cube is None:
        shapes: dict[str, tuple[int, int]] = {}
        for sym in symbols:
            if sym in data_maps and tf in data_maps[sym]:
                df = data_maps[sym][tf]
                shapes[sym] = (len(df), len(df.columns))
        _ALIGNED_DATA_MAPS_CACHE[cache_key] = (result, shapes)
    return result
