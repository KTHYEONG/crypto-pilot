from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig

LtfAlphaTimeframe = Literal["5m", "15m", "30m"]

_VALID_LTFS: tuple[LtfAlphaTimeframe, ...] = ("5m", "15m", "30m")

_LTF_TO_PANDAS: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}

_LTF_NATIVE_FAMILIES: tuple[str, ...] = (
    "funding_session_orb_flow",
    "liquidity_sweep_reclaim",
    "volume_participation_breakout",
)

_FAMILY_LTF_GRID: dict[str, tuple[LtfAlphaTimeframe, ...]] = {
    "funding_session_orb_flow": ("15m", "30m"),
    "liquidity_sweep_reclaim": ("5m",),
    "volume_participation_breakout": ("15m", "30m"),
}

_FAMILY_VARIANT_BY_LTF: dict[tuple[str, str], str] = {
    ("funding_session_orb_flow", "15m"): "fs_orb_15m",
    ("funding_session_orb_flow", "30m"): "fs_orb_30m",
    ("liquidity_sweep_reclaim", "5m"): "lsr_5m_36",
    ("volume_participation_breakout", "15m"): "vpb_15m_48",
    ("volume_participation_breakout", "30m"): "vpb_30m_48",
}

_FAMILY_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "funding_session_orb_flow": (),
    "liquidity_sweep_reclaim": (),
    "volume_participation_breakout": (),
}

_FAMILY_ARCHETYPE: dict[str, str] = {
    "funding_session_orb_flow": "trend",
    "liquidity_sweep_reclaim": "mean_reversion",
    "volume_participation_breakout": "trend",
}

_FAMILY_MAX_TURNOVER: dict[str, float] = {
    "funding_session_orb_flow": 240.0,
    "liquidity_sweep_reclaim": 365.0,
    "volume_participation_breakout": 240.0,
}

_EPS = 1e-12
_CVD_WINDOW = 12
_ORB_OPENING_MINUTES = 30
_DONCHIAN_WINDOW = 48
_SWEEP_WINDOW = 36
_ATR_WINDOW = 48


@dataclass(slots=True, frozen=True)
class LtfAlphaFeatureGrid:
    ltf: LtfAlphaTimeframe
    datetimes: NDArray[np.datetime64]
    symbols: tuple[str, ...]
    open_2d: NDArray[np.float64]
    high_2d: NDArray[np.float64]
    low_2d: NDArray[np.float64]
    close_2d: NDArray[np.float64]
    volume_2d: NDArray[np.float64]
    taker_buy_2d: NDArray[np.float64]
    quote_volume_2d: NDArray[np.float64]
    trades_2d: NDArray[np.float64] | None
    valid_mask_2d: NDArray[np.bool_]
    coverage_by_symbol: Mapping[str, float]


def _filter_1m_range(
    df: pd.DataFrame,
    start: np.datetime64,
    end: np.datetime64,
) -> pd.DataFrame:
    if df.empty:
        return df
    if hasattr(df["datetime"].dtype, "tz") and df["datetime"].dtype.tz is not None:
        s = pd.Timestamp(start).tz_localize("UTC")
        e = pd.Timestamp(end).tz_localize("UTC")
    else:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
    return df[(df["datetime"] >= s) & (df["datetime"] <= e)]


def _resample_1m_to_ltf(
    df: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:
    indexed = df.set_index("datetime")
    pandas_rule = _LTF_TO_PANDAS.get(rule, rule)
    resampled = indexed.resample(pandas_rule, label="right").agg({
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_vol": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
        "trades": "sum",
    })
    resampled = resampled.dropna()
    return resampled


def _normalize_exec_1m_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "high", "low", "close", "volume", "taker_buy_base_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required column: {sorted(missing)[0]}")
    if "quote_vol" in df.columns and "trades" in df.columns and "taker_buy_quote_volume" in df.columns:
        return df

    normalized = df.copy()
    if "quote_vol" not in normalized.columns:
        normalized["quote_vol"] = normalized["volume"].astype(float) * normalized["close"].astype(float)
    if "trades" not in normalized.columns:
        normalized["trades"] = normalized["volume"].astype(float)
    if "taker_buy_quote_volume" not in normalized.columns:
        normalized["taker_buy_quote_volume"] = (
            normalized["taker_buy_base_volume"].astype(float) * normalized["close"].astype(float)
        )
    return normalized


def build_ltf_alpha_feature_grid(
    *,
    exec_1m_by_symbol: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    ltf: LtfAlphaTimeframe,
    start: np.datetime64,
    end: np.datetime64,
    min_coverage: float = 0.95,
) -> LtfAlphaFeatureGrid:
    """[ADR_20260708_LTF_NATIVE_SIGNAL_EXPANSION] Resample closed 1m caches into LTF grids.

    Args:
        exec_1m_by_symbol: Mapping from symbol to 1m OHLCV+taker DataFrame.
        symbols: Ordered symbol universe matching base alignment.
        ltf: Target lower timeframe.
        start: Inclusive UTC start timestamp.
        end: Inclusive UTC end timestamp.
        min_coverage: Minimum observed/expected 1m rows per symbol.

    Returns:
        Dense LTF arrays. Symbols below coverage threshold have `valid_mask_2d=False`.

    Raises:
        ValueError: If `ltf` is unsupported or required OHLCV columns are absent.
    """
    if ltf not in _VALID_LTFS:
        raise ValueError(f"unsupported ltf: {ltf}")

    rule = ltf
    n_sym = len(symbols)

    grid_dt: NDArray[np.datetime64] | None = None
    for sym in symbols:
        df = exec_1m_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        df_norm = _normalize_exec_1m_columns(df)
        df_clean = df_norm.sort_values("datetime").drop_duplicates(subset="datetime", keep="last")
        df_range = _filter_1m_range(df_clean, start, end)
        if df_range.empty:
            continue
        resampled = _resample_1m_to_ltf(df_range, rule)
        if resampled.empty:
            continue
        grid_dt = resampled.index.to_numpy(dtype="datetime64[ns]")
        break

    if grid_dt is None:
        n_grid = 0
        grid_dt = np.array([], dtype="datetime64[ns]")
    else:
        n_grid = len(grid_dt)

    open_2d: NDArray[np.float64] = np.zeros((n_grid, n_sym), dtype=np.float64)
    high_2d = np.zeros_like(open_2d)
    low_2d = np.zeros_like(open_2d)
    close_2d = np.zeros_like(open_2d)
    volume_2d = np.zeros_like(open_2d)
    taker_buy_2d = np.zeros_like(open_2d)
    quote_volume_2d = np.zeros_like(open_2d)
    trades_2d = np.zeros_like(open_2d)
    valid_mask_2d: NDArray[np.bool_] = np.zeros((n_grid, n_sym), dtype=np.bool_)
    coverage_by_symbol: dict[str, float] = {}

    expected_1m = int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60) + 1

    for i, sym in enumerate(symbols):
        df = exec_1m_by_symbol.get(sym)
        if df is None or df.empty:
            coverage_by_symbol[sym] = 0.0
            continue

        df_norm = _normalize_exec_1m_columns(df)
        df_clean = df_norm.sort_values("datetime").drop_duplicates(subset="datetime", keep="last")
        df_range = _filter_1m_range(df_clean, start, end)
        coverage = len(df_range) / max(expected_1m, 1)
        coverage_by_symbol[sym] = coverage

        if coverage < min_coverage or df_range.empty:
            continue

        resampled = _resample_1m_to_ltf(df_range, rule)
        if resampled.empty:
            continue

        resampled_dt = resampled.index.to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(grid_dt, resampled_dt, side="left")
        mask = idx < n_grid
        idx_valid = idx[mask]
        high_2d[idx_valid, i] = resampled["high"].values[mask]
        low_2d[idx_valid, i] = resampled["low"].values[mask]
        close_2d[idx_valid, i] = resampled["close"].values[mask]
        volume_2d[idx_valid, i] = resampled["volume"].values[mask]
        taker_buy_2d[idx_valid, i] = resampled["taker_buy_base_volume"].values[mask]
        quote_volume_2d[idx_valid, i] = resampled["quote_vol"].values[mask]
        trades_2d[idx_valid, i] = resampled["trades"].values[mask]
        valid_mask_2d[idx_valid, i] = True

    has_trades = bool(np.any(np.isfinite(trades_2d))) if n_grid > 0 else False

    return LtfAlphaFeatureGrid(
        ltf=ltf,
        datetimes=grid_dt,
        symbols=tuple(symbols),
        open_2d=open_2d,
        high_2d=high_2d,
        low_2d=low_2d,
        close_2d=close_2d,
        volume_2d=volume_2d,
        taker_buy_2d=taker_buy_2d,
        quote_volume_2d=quote_volume_2d,
        trades_2d=trades_2d if has_trades else None,
        valid_mask_2d=valid_mask_2d,
        coverage_by_symbol=coverage_by_symbol,
    )


def _safe_divide(num: NDArray[np.float64], den: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros_like(num, dtype=np.float64)
    mask = np.abs(den) > _EPS
    out[mask] = num[mask] / den[mask]
    return out


def _rolling_robust_z(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    series = pd.Series(values, copy=False)
    min_periods = max(4, window // 2)
    median = series.rolling(window=window, min_periods=min_periods).median()
    q75 = series.rolling(window=window, min_periods=min_periods).quantile(0.75)
    q25 = series.rolling(window=window, min_periods=min_periods).quantile(0.25)
    std = series.rolling(window=window, min_periods=min_periods).std()
    scale = ((q75 - q25) / 1.349).mask(lambda s: s <= _EPS, std).replace(0.0, np.nan)
    z = ((series - median) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cast(NDArray[np.float64], np.clip(z.to_numpy(dtype=np.float64), -5.0, 5.0))


def _rolling_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        pd.Series(values, copy=False)
        .rolling(window=window, min_periods=max(4, window // 2))
        .mean()
        .fillna(0.0)
        .to_numpy(dtype=np.float64),
    )


def _shifted_rolling_max(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        pd.Series(values, copy=False)
        .rolling(window=window, min_periods=max(4, window // 2))
        .max()
        .shift(1)
        .to_numpy(dtype=np.float64),
    )


def _shifted_rolling_min(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        pd.Series(values, copy=False)
        .rolling(window=window, min_periods=max(4, window // 2))
        .min()
        .shift(1)
        .to_numpy(dtype=np.float64),
    )


def _rising_edge(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    prev = np.zeros_like(mask)
    prev[1:] = mask[:-1]
    return mask & ~prev


def _session_opening_range(
    datetimes: NDArray[np.datetime64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    *,
    opening_minutes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ts = pd.DatetimeIndex(datetimes)
    session_start = ts.floor("8h")
    minutes_from_start = ((ts - session_start) / pd.Timedelta(minutes=1)).to_numpy(dtype=np.float64)
    open_high = np.full(len(datetimes), np.nan, dtype=np.float64)
    open_low = np.full(len(datetimes), np.nan, dtype=np.float64)
    current_session = None
    cur_high = np.nan
    cur_low = np.nan
    finalized_high = np.nan
    finalized_low = np.nan
    for i, session in enumerate(session_start):
        if current_session is None or session != current_session:
            current_session = session
            cur_high = np.nan
            cur_low = np.nan
            finalized_high = np.nan
            finalized_low = np.nan
        if minutes_from_start[i] <= opening_minutes:
            cur_high = high[i] if np.isnan(cur_high) else max(cur_high, high[i])
            cur_low = low[i] if np.isnan(cur_low) else min(cur_low, low[i])
            if minutes_from_start[i] >= opening_minutes:
                finalized_high = cur_high
                finalized_low = cur_low
        open_high[i] = finalized_high
        open_low[i] = finalized_low
    return open_high, open_low


def _family_signal_arrays(
    *,
    grid: LtfAlphaFeatureGrid,
    family: str,
) -> tuple[NDArray[np.float64], NDArray[np.int8], NDArray[np.bool_]]:
    t_ltf, n_sym = grid.close_2d.shape
    score = np.zeros((t_ltf, n_sym), dtype=np.float64)
    side = np.zeros((t_ltf, n_sym), dtype=np.int8)
    valid = np.zeros((t_ltf, n_sym), dtype=np.bool_)

    for col in range(n_sym):
        active = grid.valid_mask_2d[:, col]
        if not active.any():
            continue

        high = grid.high_2d[:, col]
        low = grid.low_2d[:, col]
        close = grid.close_2d[:, col]
        volume = grid.volume_2d[:, col]
        taker_buy = grid.taker_buy_2d[:, col]
        quote_volume = grid.quote_volume_2d[:, col]
        trades = grid.trades_2d[:, col] if grid.trades_2d is not None else volume

        imbalance = np.clip(_safe_divide(2.0 * taker_buy, volume) - 1.0, -1.0, 1.0)
        cvd = _safe_divide(_rolling_mean(imbalance * volume, _CVD_WINDOW), _rolling_mean(volume, _CVD_WINDOW))
        vol_z = _rolling_robust_z(quote_volume, _DONCHIAN_WINDOW)
        trades_z = _rolling_robust_z(trades, _DONCHIAN_WINDOW)
        spread = np.maximum(high - low, _EPS)
        clv = ((close - low) - (high - close)) / spread
        atr = _rolling_mean(spread, _ATR_WINDOW)

        if family == "funding_session_orb_flow":
            open_high, open_low = _session_opening_range(
                grid.datetimes,
                high,
                low,
                opening_minutes=_ORB_OPENING_MINUTES,
            )
            long_base = close > open_high
            short_base = close < open_low
            long_mask = active & long_base & (vol_z >= 1.5) & (cvd > 0.15) & (clv > 0.10)
            short_mask = active & short_base & (vol_z >= 1.5) & (cvd < -0.15) & (clv < -0.10)
        elif family == "liquidity_sweep_reclaim":
            prior_low = _shifted_rolling_min(low, _SWEEP_WINDOW)
            prior_high = _shifted_rolling_max(high, _SWEEP_WINDOW)
            long_base = (low < prior_low - 0.25 * atr) & (close > prior_low)
            short_base = (high > prior_high + 0.25 * atr) & (close < prior_high)
            long_mask = active & long_base & (cvd < -0.15) & (clv > 0.35)
            short_mask = active & short_base & (cvd > 0.15) & (clv < -0.35)
        elif family == "volume_participation_breakout":
            don_high = _shifted_rolling_max(high, _DONCHIAN_WINDOW)
            don_low = _shifted_rolling_min(low, _DONCHIAN_WINDOW)
            long_base = close > don_high
            short_base = close < don_low
            long_mask = active & long_base & (vol_z >= 2.0) & (trades_z >= 0.5) & (cvd > 0.10) & (clv > 0.20)
            short_mask = active & short_base & (vol_z >= 2.0) & (trades_z >= 0.5) & (cvd < -0.10) & (clv < -0.20)
        else:
            continue

        long_entry = _rising_edge(long_mask)
        short_entry = _rising_edge(short_mask)
        side[long_entry, col] = 1
        side[short_entry, col] = -1
        signed_strength = np.abs(vol_z) + np.abs(cvd) + np.abs(clv)
        score[long_entry, col] = signed_strength[long_entry]
        score[short_entry, col] = -signed_strength[short_entry]
        valid[:, col] = long_entry | short_entry

    return score, side, valid


def _build_sparse_signal(
    *,
    grid: LtfAlphaFeatureGrid,
    aligned: AlignedMarketData,
    family: str,
) -> CandidateSignalPanel:
    score, side_hint, valid = _family_signal_arrays(grid=grid, family=family)
    turnover = valid.astype(np.float64)
    metadata: dict[str, object] = {
        "source_tf": grid.ltf,
        "release_lag_bars": 1,
        "archetype": _FAMILY_ARCHETYPE.get(family, "trend"),
        "ltf_event_count": int(valid.sum()),
        "max_turnover_per_year": _FAMILY_MAX_TURNOVER.get(family, 240.0),
    }
    return CandidateSignalPanel(
        family=family,
        variant=_FAMILY_VARIANT_BY_LTF.get((family, grid.ltf), f"{family}_{grid.ltf}"),
        params={"ltf": grid.ltf},
        datetimes=grid.datetimes,
        symbols=aligned.symbols,
        signed_score_2d=score,
        side_hint_2d=side_hint,
        expected_holding_bars=1,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=turnover,
        valid_mask_2d=valid,
        metadata=metadata,
        archetype=_FAMILY_ARCHETYPE.get(family, "trend"),
        exit_policies=(),
    )


def _check_family_required_fields(family: str, aligned: AlignedMarketData) -> bool:
    required = _FAMILY_REQUIRED_FIELDS.get(family, ())
    return all(getattr(aligned, field, None) is not None for field in required)


def build_ltf_native_alpha_panels(
    *,
    aligned: AlignedMarketData,
    exec_1m_by_symbol: Mapping[str, pd.DataFrame],
    cfg: CandidateStrategyConfig,
    ltf_grid: tuple[LtfAlphaTimeframe, ...] = ("5m", "15m", "30m"),
    family_filter: tuple[str, ...] | None = None,
) -> tuple[CandidateSignalPanel, ...]:
    """[ADR_20260708_LTF_NATIVE_SIGNAL_EXPANSION] Build sparse LTF-native panels.

    Native 1m caches are resampled to 5m/15m/30m, projected onto the base grid,
    and emitted as sparse directional panels with rising-edge entries only.
    """
    del cfg
    families = family_filter if family_filter is not None else _LTF_NATIVE_FAMILIES
    if not exec_1m_by_symbol:
        return ()

    result: list[CandidateSignalPanel] = []
    grid_cache: dict[LtfAlphaTimeframe, LtfAlphaFeatureGrid] = {}
    start = aligned.datetimes[0]
    end = aligned.datetimes[-1]
    for family in families:
        if not _check_family_required_fields(family, aligned):
            continue
        family_ltfs = _FAMILY_LTF_GRID.get(family, ltf_grid)
        for ltf in ltf_grid:
            if ltf not in _VALID_LTFS or ltf not in family_ltfs:
                continue
            grid = grid_cache.get(ltf)
            if grid is None:
                grid = build_ltf_alpha_feature_grid(
                    exec_1m_by_symbol=exec_1m_by_symbol,
                    symbols=aligned.symbols,
                    ltf=ltf,
                    start=start,
                    end=end,
                )
                grid_cache[ltf] = grid
            if grid.datetimes.size == 0:
                continue
            panel = _build_sparse_signal(grid=grid, aligned=aligned, family=family)
            if not panel.valid_mask_2d.any():
                continue
            projected = project_ltf_panel_to_base_grid(
                panel=panel,
                base_datetimes=aligned.datetimes,
                base_valid_mask_2d=aligned.active_mask,
            )
            result.append(projected)

    return tuple(result)


def project_ltf_panel_to_base_grid(
    *,
    panel: CandidateSignalPanel,
    base_datetimes: NDArray[np.datetime64],
    base_valid_mask_2d: NDArray[np.bool_],
) -> CandidateSignalPanel:
    """[ADR_20260708_LTF_NATIVE_SIGNAL_EXPANSION] Project closed LTF bars onto base timestamps."""
    if len(base_datetimes) > 1:
        diffs = np.diff(base_datetimes.astype("int64"))
        if np.any(diffs < 0):
            raise ValueError("base_datetimes must be monotonic increasing")

    t_base = len(base_datetimes)
    n_sym = panel.signed_score_2d.shape[1]
    t_panel = panel.signed_score_2d.shape[0]

    if base_valid_mask_2d.shape != (t_base, n_sym):
        raise ValueError("panel/base shape mismatch")

    score = np.zeros((t_base, n_sym), dtype=np.float64)
    side = np.zeros((t_base, n_sym), dtype=np.int8)
    turnover = np.zeros((t_base, n_sym), dtype=np.float64)
    valid = np.zeros((t_base, n_sym), dtype=np.bool_)

    if t_panel > 0:
        panel_dt: NDArray[np.datetime64] = np.asarray(panel.datetimes, dtype="datetime64[ns]")
        idx = np.searchsorted(base_datetimes, panel_dt, side="left")
        in_range = idx < t_base
        for src_i, base_i in zip(np.flatnonzero(in_range), idx[in_range], strict=False):
            event_cols = np.flatnonzero(panel.valid_mask_2d[src_i] & base_valid_mask_2d[base_i])
            if event_cols.size == 0:
                continue
            existing_abs = np.abs(score[base_i, event_cols])
            incoming_abs = np.abs(panel.signed_score_2d[src_i, event_cols])
            take = incoming_abs >= existing_abs
            cols = event_cols[take]
            score[base_i, cols] = panel.signed_score_2d[src_i, cols]
            side[base_i, cols] = panel.side_hint_2d[src_i, cols]
            turnover[base_i, cols] = panel.turnover_proxy_2d[src_i, cols]
            valid[base_i, cols] = True

    return CandidateSignalPanel(
        family=panel.family,
        variant=panel.variant,
        params=panel.params,
        datetimes=base_datetimes,
        symbols=panel.symbols,
        signed_score_2d=score,
        side_hint_2d=side,
        expected_holding_bars=panel.expected_holding_bars,
        min_holding_bars=panel.min_holding_bars,
        stop_atr_mult=panel.stop_atr_mult,
        take_profit_atr_mult=panel.take_profit_atr_mult,
        turnover_proxy_2d=turnover,
        valid_mask_2d=valid,
        metadata=panel.metadata,
        archetype=panel.archetype,
        exit_policies=panel.exit_policies or (),
    )


def build_ltf_native_alpha_panels_streaming(
    *,
    aligned: AlignedMarketData,
    plan: Any,
    load_frame: Any,
    budget: Any,
) -> tuple[CandidateSignalPanel, ...]:
    """Build causally projected LTF panels without a universe-wide dense grid.

    [ADR_20260712_L0_MEMORY_BOUND_DATAFLOW]
    One symbol is loaded and reduced at a time.  Only base-grid accumulators
    survive between symbols, so peak source memory is independent of universe
    width.  ``load_frame`` must return a bounded 1m frame for the symbol.
    """
    if getattr(plan, "skip_reason", None) is not None:
        return ()
    symbols = tuple(getattr(plan, "symbols", ()))
    if not symbols:
        return ()

    families = tuple(_LTF_NATIVE_FAMILIES)
    base_dt = np.asarray(aligned.datetimes, dtype="datetime64[ns]")
    t_base, n_base = aligned.close_2d.shape
    accumulators: dict[tuple[str, str], dict[str, NDArray[Any]]] = {}
    metadata_by_key: dict[tuple[str, str], dict[str, object]] = {}

    for family in families:
        for ltf_raw in _FAMILY_LTF_GRID.get(family, _VALID_LTFS):
            ltf = ltf_raw
            key = (family, ltf)
            accumulators[key] = {
                "score": np.zeros((t_base, n_base), dtype=np.float64),
                "side": np.zeros((t_base, n_base), dtype=np.int8),
                "turnover": np.zeros((t_base, n_base), dtype=np.float64),
                "valid": np.zeros((t_base, n_base), dtype=np.bool_),
            }

    symbol_to_col = {symbol: index for index, symbol in enumerate(aligned.symbols)}
    for symbol in symbols:
        col = symbol_to_col.get(symbol)
        if col is None:
            continue
        frame = load_frame(symbol)
        if frame is None or frame.empty:
            continue
        symbol_grid: dict[str, LtfAlphaFeatureGrid] = {}
        symbol_aligned = dataclasses.replace(aligned, symbols=(symbol,))
        try:
            for family in families:
                for ltf_raw in _FAMILY_LTF_GRID.get(family, _VALID_LTFS):
                    ltf = ltf_raw
                    grid = symbol_grid.get(ltf)
                    if grid is None:
                        grid = build_ltf_alpha_feature_grid(
                            exec_1m_by_symbol={symbol: frame},
                            symbols=(symbol,),
                            ltf=ltf,
                            start=base_dt[0],
                            end=base_dt[-1],
                        )
                        symbol_grid[ltf] = grid
                    if grid.datetimes.size == 0:
                        continue
                    sparse = _build_sparse_signal(
                        grid=grid,
                        aligned=symbol_aligned,
                        family=family,
                    )
                    projected = project_ltf_panel_to_base_grid(
                        panel=sparse,
                        base_datetimes=base_dt,
                        base_valid_mask_2d=aligned.active_mask[:, col : col + 1],
                    )
                    key = (family, ltf)
                    acc = accumulators[key]
                    acc["score"][:, col] = projected.signed_score_2d[:, 0]
                    acc["side"][:, col] = projected.side_hint_2d[:, 0]
                    acc["turnover"][:, col] = projected.turnover_proxy_2d[:, 0]
                    acc["valid"][:, col] = projected.valid_mask_2d[:, 0]
                    metadata_by_key[key] = dict(projected.metadata)
        finally:
            del symbol_grid, symbol_aligned, frame

    result: list[CandidateSignalPanel] = []
    for (family, ltf_name), acc in accumulators.items():
        valid = acc["valid"]
        if not valid.any():
            continue
        metadata = metadata_by_key.get(
            (family, ltf_name),
            {
                "source_tf": ltf_name,
                "release_lag_bars": 1,
                "archetype": _FAMILY_ARCHETYPE.get(family, "trend"),
            },
        )
        result.append(
            CandidateSignalPanel(
                family=family,
                variant=_FAMILY_VARIANT_BY_LTF.get(
                    (family, ltf_name), f"{family}_{ltf_name}"
                ),
                params={"ltf": ltf_name},
                datetimes=base_dt,
                symbols=aligned.symbols,
                signed_score_2d=acc["score"],
                side_hint_2d=acc["side"],
                expected_holding_bars=1,
                min_holding_bars=1,
                stop_atr_mult=2.0,
                take_profit_atr_mult=4.0,
                turnover_proxy_2d=acc["turnover"],
                valid_mask_2d=valid,
                metadata=metadata,
                archetype=_FAMILY_ARCHETYPE.get(family, "trend"),
                exit_policies=(),
            )
        )
    return tuple(result)
