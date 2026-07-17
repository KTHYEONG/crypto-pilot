"""Entry-timing refinement layer: CVD impulse, anchored VWAP sigma-band, trend quality gate.

[ADR_20260707_LTF_ENTRY_TIMING_LAYER][ADR_20260714_L0_LTF_STREAM_PARALLEL]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.contracts import (
    EntryTimingGateConfig,
    EntryTimingWindow,
    Universe1mCoverageTier,
)
from src.domain.futures.optimization.metrics import hurst_dfa, kaufman_efficiency_ratio, variance_ratio
from src.domain.futures.signals.rules import safe_taker_imbalance_2d
from src.domain.futures.strategy.timeframe_contracts import resample_alias
from src.domain.futures.universe.storage import run_historical_sync


def compute_cvd_delta_z(
    taker_buy_1m: NDArray[np.float64],
    volume_1m: NDArray[np.float64],
    *,
    lookback_bars: int,
) -> NDArray[np.float64]:
    imbalance, _valid = safe_taker_imbalance_2d(taker_buy_1m, volume_1m)
    cum_imbalance = np.zeros_like(imbalance)
    n = len(imbalance)
    for i in range(0, n, lookback_bars):
        end = min(i + lookback_bars, n)
        seg = imbalance[i:end]
        seg_cum = np.cumsum(seg)
        cum_imbalance[i:end] = seg_cum
    z = np.zeros_like(cum_imbalance)
    for i in range(lookback_bars, n):
        window = cum_imbalance[i - lookback_bars : i]
        mu = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std > 1e-12:
            z[i] = (cum_imbalance[i] - mu) / std
    return z


def compute_anchored_vwap_dev_sigma(
    high_1m: NDArray[np.float64],
    low_1m: NDArray[np.float64],
    close_1m: NDArray[np.float64],
    volume_1m: NDArray[np.float64],
    *,
    anchor_pos: int,
) -> NDArray[np.float64]:
    if not (high_1m.shape == low_1m.shape == close_1m.shape == volume_1m.shape):
        raise ValueError("shape mismatch among input arrays")
    typical = (high_1m + low_1m + close_1m) / 3.0
    n = len(typical)
    result = np.zeros(n, dtype=np.float64)
    for i in range(anchor_pos, n):
        seg = typical[anchor_pos : i + 1]
        vol = volume_1m[anchor_pos : i + 1]
        vol_sum = float(np.sum(vol))
        if vol_sum < 1e-12:
            continue
        vwap = float(np.sum(seg * vol)) / vol_sum
        var = float(np.sum(vol * (seg - vwap) ** 2)) / vol_sum
        sigma = float(np.sqrt(max(var, 1e-12)))
        result[i] = (close_1m[i] - vwap) / sigma
    return result


def evaluate_trend_quality_gate(rets_1m_window: NDArray[np.float64]) -> bool:
    n_votes = 0
    er = kaufman_efficiency_ratio(rets_1m_window)
    if er > 0.3:
        n_votes += 1
    h = hurst_dfa(rets_1m_window)
    if h > 0.55:
        n_votes += 1
    vr_val, _m2 = variance_ratio(rets_1m_window, q=4)
    if vr_val > 1.0:
        n_votes += 1
    return n_votes >= 2


def refine_entry_indices(
    events: pd.DataFrame,
    *,
    base_datetimes: NDArray[np.datetime64],
    ltf_1m_frames_by_symbol: Mapping[str, pd.DataFrame],
    config: EntryTimingGateConfig,
    coverage_tier: Universe1mCoverageTier | None = None,
) -> tuple[pd.DataFrame, tuple[EntryTimingWindow, ...]]:
    required = {"entry_idx", "side", "family", "symbol", "variant", "expected_holding_bars", "handoff_tier"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {missing}")
    valid_events = events[events["handoff_tier"].isin({"seed", "candidate"})].copy()
    if valid_events.empty:
        return events.copy(), ()
    windows: list[EntryTimingWindow] = []
    result_events = events.copy()
    for idx, row in valid_events.iterrows():
        symbol = row["symbol"]
        side = int(row["side"])
        base_entry_idx = int(row["entry_idx"])
        expected_holding_bars = int(row["expected_holding_bars"])
        max_wait_bars_base = max(1, round(expected_holding_bars * config.max_wait_bars_ratio))
        if symbol not in ltf_1m_frames_by_symbol:
            cov_status: Literal["covered", "uncovered_fallback"] = (
                "uncovered_fallback"
                if coverage_tier is not None and not coverage_tier.is_covered(symbol)
                else "covered"
            )
            windows.append(
                EntryTimingWindow(
                    episode_id=f"{row['family']}_{row['variant']}_{idx}",
                    ltf=config.ltf_grid[0],
                    max_wait_bars_base=max_wait_bars_base,
                    triggered=False,
                    refined_entry_idx=base_entry_idx + max_wait_bars_base,
                    price_improvement_bps=0.0,
                    opportunity_cost_bps=0.0,
                    net_timing_edge_bps=0.0,
                    coverage_status=cov_status,
                )
            )
            continue
        entry_ts = pd.Timestamp(base_datetimes[base_entry_idx], tz="UTC")
        window_end_idx = min(base_entry_idx + max_wait_bars_base, len(base_datetimes) - 1)
        window_end_ts = pd.Timestamp(base_datetimes[window_end_idx], tz="UTC")
        frame = ltf_1m_frames_by_symbol[symbol]
        mask = (frame["datetime"] >= entry_ts) & (frame["datetime"] < window_end_ts)
        window_1m = frame[mask].copy()
        if window_1m.empty:
            windows.append(
                EntryTimingWindow(
                    episode_id=f"{row['family']}_{row['variant']}_{idx}",
                    ltf=config.ltf_grid[0],
                    max_wait_bars_base=max_wait_bars_base,
                    triggered=False,
                    refined_entry_idx=base_entry_idx + max_wait_bars_base,
                    price_improvement_bps=0.0,
                    opportunity_cost_bps=0.0,
                    net_timing_edge_bps=0.0,
                )
            )
            continue
        ltf = config.ltf_grid[0]
        ltf_alias = resample_alias(ltf)
        window_1m.set_index("datetime", inplace=True)
        ltf_grouped = window_1m.resample(ltf_alias)
        triggered = False
        refined_entry_idx = base_entry_idx + max_wait_bars_base
        for ltf_bar_start, ltf_bar in ltf_grouped:
            if len(ltf_bar) == 0:
                continue
            taker_buy_arr = ltf_bar["taker_buy_volume"].to_numpy(dtype=np.float64)
            vol_arr = ltf_bar["volume"].to_numpy(dtype=np.float64)
            high_arr = ltf_bar["high"].to_numpy(dtype=np.float64)
            low_arr = ltf_bar["low"].to_numpy(dtype=np.float64)
            close_arr = ltf_bar["close"].to_numpy(dtype=np.float64)
            rets_arr = np.diff(np.log(close_arr), prepend=np.log(close_arr[0]))
            lookback = min(config.cvd_lookback_bars, len(taker_buy_arr))
            cvd_z = compute_cvd_delta_z(taker_buy_arr, vol_arr, lookback_bars=lookback)
            current_cvd_z = float(cvd_z[-1]) if len(cvd_z) > 0 else 0.0
            vwap_dev = compute_anchored_vwap_dev_sigma(high_arr, low_arr, close_arr, vol_arr, anchor_pos=0)
            current_vwap_dev = float(vwap_dev[-1]) if len(vwap_dev) > 0 else 0.0
            trend_pass = evaluate_trend_quality_gate(rets_arr)
            if not trend_pass:
                continue
            cvd_agree = (current_cvd_z > 0 and side > 0) or (current_cvd_z < 0 and side < 0)
            w_cvd = config.confluence_weights.get("cvd", 0.5)
            w_vwap = config.confluence_weights.get("vwap", 0.5)
            # score is a signed "alignment with side" confidence in [-1, 1] (positive = confirms side),
            # not a raw directional score — both terms must be normalized against `side` before summing.
            vwap_alignment = float(np.tanh(current_vwap_dev)) * side
            score = w_cvd * (1.0 if cvd_agree else -1.0) + w_vwap * vwap_alignment
            score = float(np.clip(score, -1.0, 1.0))
            if trend_pass and score > 0:
                ltf_bar_end = ltf_bar_start + pd.Timedelta(ltf_alias)
                ltf_end_ns = np.datetime64(ltf_bar_end.to_pydatetime().replace(tzinfo=None), "ns")
                refined_entry_idx = int(np.searchsorted(base_datetimes, ltf_end_ns, side="right"))
                refined_entry_idx = max(base_entry_idx, min(refined_entry_idx, base_entry_idx + max_wait_bars_base))
                triggered = True
                break
        price_improvement_bps = 0.0
        opportunity_cost_bps = 0.0
        net_edge = 0.0
        if triggered:
            naive_price = float(window_1m["close"].iloc[0])
            trigger_ts = (
                pd.Timestamp(base_datetimes[refined_entry_idx], tz="UTC")
                if refined_entry_idx < len(base_datetimes)
                else window_end_ts
            )
            elapsed = window_1m.loc[window_1m.index < trigger_ts]
            if elapsed.empty:
                elapsed = window_1m.iloc[:1]
            timed_price = float(elapsed["close"].iloc[-1])
            price_improvement_bps = float(side) * (naive_price - timed_price) / naive_price * 1e4
            if side > 0:
                adverse = max(0.0, naive_price - float(elapsed["low"].min()))
            else:
                adverse = max(0.0, float(elapsed["high"].max()) - naive_price)
            opportunity_cost_bps = adverse / naive_price * 1e4
            net_edge = price_improvement_bps - opportunity_cost_bps
        windows.append(
            EntryTimingWindow(
                episode_id=f"{row['family']}_{row['variant']}_{idx}",
                ltf=ltf,
                max_wait_bars_base=max_wait_bars_base,
                triggered=triggered,
                refined_entry_idx=refined_entry_idx,
                price_improvement_bps=price_improvement_bps,
                opportunity_cost_bps=opportunity_cost_bps,
                net_timing_edge_bps=net_edge,
            )
        )
        result_events.at[idx, "entry_idx"] = refined_entry_idx
    return result_events, tuple(windows)


def aggregate_entry_timing_evidence(
    windows: Sequence[EntryTimingWindow],
) -> Mapping[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for w in windows:
        parts = w.episode_id.split("_")
        if len(parts) >= 2:
            family = parts[0]
            variant = parts[1]
        else:
            family = "unknown"
            variant = "unknown"
        key = (family, variant, w.ltf)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(w.net_timing_edge_bps)
    result: dict[tuple[str, str, str], float] = {}
    for key, edges in grouped.items():
        arr = np.array(edges, dtype=np.float64)
        if arr.size < 2:
            result[key] = float(np.mean(arr)) if arr.size > 0 else 0.0
            continue
        n_boot = 1000
        rng = np.random.default_rng(42)
        n = len(arr)
        block_size = max(1, int(np.sqrt(n)))
        boot_means = np.zeros(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sample: list[float] = []
            while len(sample) < n:
                start = rng.integers(0, max(1, n - block_size + 1))
                sample.extend(edges[start : start + block_size])
            boot_means[b] = float(np.mean(sample[:n]))
        lcb = float(np.percentile(boot_means, 5))
        result[key] = lcb
    return result


def _compute_1m_coverage_ratio(
    path: Path,
    *,
    start_date: date,
    end_date: date,
) -> float:
    """[ADR_20260710_L0_SIGNAL_BREADTH_DIVERSITY_REDESIGN] Fraction of expected 1-minute bars present in ``path``.

    Returns 0.0 if the file is absent, empty, zero-byte, or lacks a ``datetime`` column —
    mirrors ``evaluate_symbol_data_sufficiency()``'s ``exec_1m_coverage`` arithmetic
    (``opt_data_utils.py:144-146``) so the two coverage definitions cannot drift apart.
    [LIMIT-12]
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0.0
    try:
        try:
            frame = pd.read_parquet(path, columns=["timestamp"])
            frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        except Exception:
            frame = pd.read_parquet(path, columns=["datetime"])
    except Exception:
        return 0.0
    if frame.empty or "datetime" not in frame.columns:
        return 0.0
    dt = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dropna()
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    actual = int(((dt >= start_ts) & (dt <= end_ts)).sum())
    required = max(1, int((end_ts - start_ts).total_seconds() // 60))
    return float(actual) / float(required)


def resolve_1m_backfill_targets(
    universe_symbols: tuple[str, ...],
    data_root: Path = Path("data/futures"),
    *,
    start_date: date = date(2019, 1, 1),
    end_date: date,
    min_coverage_ratio: float = 0.95,
) -> tuple[str, ...]:
    """Return symbols whose 1m coverage ratio over (start_date, end_date) is below the floor.

    Replaces the prior ``path.exists()``-only check: a file that exists but only
    covers a recent slice is still reported as a backfill target for the full
    requested window. [LIMIT-12][LIMIT-14]
    """
    missing: list[str] = []
    for symbol in universe_symbols:
        from src.core.settings import FUTURES_DATA_DIR, FuturesStorageLayout

        if Path(data_root).resolve() == FUTURES_DATA_DIR.resolve():
            path = FuturesStorageLayout.get_ohlcv_path(symbol, "1m")
        else:
            path = Path(data_root) / f"{symbol.replace('/', '_')}_1m.parquet"
        ratio = _compute_1m_coverage_ratio(path, start_date=start_date, end_date=end_date)
        if ratio < min_coverage_ratio:
            missing.append(symbol)
    return tuple(missing)


def run_1m_backfill(
    missing_symbols: tuple[str, ...],
    *,
    start_date: date = date(2019, 1, 1),
    end_date: date,
) -> None:
    """Backfill 1m data for missing symbols via run_historical_sync. No-op if empty.

    [ADR_20260708_LTF_NATIVE_DIRECTIONAL_SEARCH]
    """
    if not missing_symbols:
        return
    run_historical_sync(
        start_date=start_date,
        end_date=end_date,
        symbols=list(missing_symbols),
        sync_1m=True,
        sync_1d=False,
        sync_4h=False,
    )


def resolve_1m_coverage_tier(
    universe_symbols: tuple[str, ...],
    *,
    data_root: Path = Path("data/futures"),
    start_date: date = date(2019, 1, 1),
    end_date: date,
    min_coverage_ratio: float = 0.95,
) -> Universe1mCoverageTier:
    """Build Universe1mCoverageTier using the same date-range coverage ratio as
    resolve_1m_backfill_targets() — [LIMIT-14] keeps the two functions consistent.

    Coverage scan is parallelised across symbols via ThreadPoolExecutor.
    """
    from src.core.settings import FUTURES_DATA_DIR, FuturesStorageLayout

    def _check_one(symbol: str) -> str | None:
        if Path(data_root).resolve() == FUTURES_DATA_DIR.resolve():
            path = FuturesStorageLayout.get_ohlcv_path(symbol, "1m")
        else:
            path = Path(data_root) / f"{symbol.replace('/', '_')}_1m.parquet"
        ratio = _compute_1m_coverage_ratio(path, start_date=start_date, end_date=end_date)
        return symbol if ratio >= min_coverage_ratio else None

    covered: set[str] = set()
    n_workers = max(1, min(8, len(universe_symbols)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_check_one, sym) for sym in universe_symbols]
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                covered.add(res)
    return Universe1mCoverageTier(
        covered_symbols=frozenset(covered),
        universe_symbols=frozenset(universe_symbols),
    )
