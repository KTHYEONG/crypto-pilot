"""Timeframe alpha probe: (symbol x family x tf) cell-level signal diagnostics."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm, spearmanr

from src.domain.futures.strategy.tiered_workflow.metrics import (
    _newey_west_ic_tstat,
    hurst_dfa,
    variance_ratio,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HPB: dict[str, float] = {
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "8h": 8.0,
    "12h": 12.0,
}
_BASE_TF: str = "4h"
_BASE_HPB: float = _HPB[_BASE_TF]

_RESAMPLE_ALIAS: dict[str, str] = {
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
}

# VR multi-q majority vote thresholds
_VR_TREND_THRESH: float = 1.05
_VR_MEAN_REV_THRESH: float = 0.95
_VR_Q_LIST: tuple[int, ...] = (2, 4, 8, 16)

# Minimum observations for meaningful IC estimation
_MIN_IC_OBS: int = 30

# Bars per year for 4h base
_BARS_PER_YEAR_4H: float = 2190.0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class TfCellEvidence:
    """Per-(symbol, family, variant, tf) diagnostic evidence cell."""

    symbol: str
    family: str
    variant: str
    archetype: str
    tf: str
    n_obs: int
    n_events: int
    ic_mean: float
    ic_tstat_hac: float
    ic_fold_sign_consistency: float
    alpha_half_life_h: float
    net_edge_bps: float
    turnover_per_year: float
    vr_label: str
    hurst: float
    passed_fdr: bool


@dataclass(slots=True, frozen=True)
class TfProbeManifest:
    """Full probe output: all (symbol x family x tf) diagnostic cells."""

    cells: tuple[TfCellEvidence, ...]
    tf_grid: tuple[str, ...]
    coverage_by_tf: dict[str, int]
    diversity_corr: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _hpb(tf: str) -> float:
    """Hours per bar for a given tf string."""
    return _HPB.get(tf, 4.0)


def _resample_ohlcv(df: pd.DataFrame, alias: str) -> pd.DataFrame:
    """Resample OHLCV to target timeframe. Drop last incomplete bar."""
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "funding_rate" in df.columns:
        agg["funding_rate"] = "mean"
    resampled = (
        df.resample(alias, label="right", closed="right")
        .agg(agg)
        .dropna(subset=["close"])
    )
    return resampled.iloc[:-1]  # drop last potentially incomplete bar


def _scale_bar_param(bars_base: int, tf: str, base_tf: str = _BASE_TF) -> int:
    """Scale a bar-count parameter to maintain real-time horizon across tf."""
    hours_target = bars_base * _hpb(base_tf)
    return max(1, round(hours_target / _hpb(tf)))


def _bars_per_year(tf: str) -> float:
    """Annual bar count for a given tf."""
    return (24.0 * 365.0) / _hpb(tf)


def _compute_forward_returns(
    close: NDArray[np.float64], holding_bars: int
) -> NDArray[np.float64]:
    """Compute forward returns with entry at t+1 (look-ahead bias prevention).

    fwd[t] = close[t+1+H] / close[t+1] - 1 for t in 0..T-2-H
    """
    n_bars = len(close)
    fwd = np.full(n_bars, np.nan, dtype=np.float64)
    h_bars = holding_bars
    # Entry at t+1, exit at t+1+h_bars; requires t+1+h_bars <= n_bars-1
    limit = n_bars - 2 - h_bars
    if limit < 0:
        return fwd
    for t in range(limit + 1):
        entry_price = close[t + 1]
        exit_price = close[t + 1 + h_bars]
        if entry_price > 1e-20:
            fwd[t] = exit_price / entry_price - 1.0
    return fwd


def _fold_sign_consistency(
    ic_vals: NDArray[np.float64], overall_ic: float
) -> float:
    """Fraction of folds whose IC sign matches overall_ic sign."""
    if len(ic_vals) == 0:
        return 0.0
    sign_ref = 1.0 if overall_ic >= 0.0 else -1.0
    matches = np.sum(np.sign(ic_vals) == sign_ref)
    return float(matches) / len(ic_vals)


def _alpha_half_life(
    signal: NDArray[np.float64],
    close: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    holding_bars: int,
    hpb_val: float,
) -> float:
    """Compute alpha half-life via log-linear IC decay fit.

    Returns NaN if lambda <= 0 or insufficient data.
    """
    h_bars = holding_bars
    lag_bars = [
        math.ceil(0.5 * h_bars),
        h_bars,
        2 * h_bars,
        4 * h_bars,
    ]
    ic_vals: list[float] = []
    lags_used: list[int] = []

    for lag in lag_bars:
        fwd = _compute_forward_returns(close, lag)
        valid = valid_mask & np.isfinite(fwd)
        n_valid = int(np.sum(valid))
        if n_valid < _MIN_IC_OBS:
            continue
        ic, _ = spearmanr(signal[valid], fwd[valid])
        if np.isfinite(ic):
            ic_vals.append(float(ic))
            lags_used.append(lag)

    if len(ic_vals) < 2:
        return float("nan")

    # log-linear: log|IC(h)| = log|IC0| - lambda * h_bars
    # Use h_hours = lag * hpb_val
    log_ic = np.log(np.maximum(np.abs(ic_vals), 1e-12))
    h_hours = np.array([lg * hpb_val for lg in lags_used], dtype=np.float64)

    if len(log_ic) < 2:
        return float("nan")

    coeffs = np.polyfit(h_hours, log_ic, 1)
    lambda_val = -coeffs[0]  # decay rate (should be > 0 for decaying IC)
    if lambda_val <= 0.0:
        return float("nan")
    return float(math.log(2.0) / lambda_val)


def _vr_label_majority(close_col: NDArray[np.float64]) -> str:
    """Compute vr_label via majority vote over q in {2,4,8,16}."""
    log_rets = np.diff(np.log(np.maximum(close_col, 1e-20))).astype(np.float64)
    votes: list[str] = []
    for q in _VR_Q_LIST:
        vr, _ = variance_ratio(log_rets, q)
        if vr > _VR_TREND_THRESH:
            votes.append("trend")
        elif vr < _VR_MEAN_REV_THRESH:
            votes.append("mean_rev")
        else:
            votes.append("flat")
    if not votes:
        return "flat"
    counter = Counter(votes)
    top_label, top_count = counter.most_common(1)[0]
    second_count = counter.most_common(2)[-1][1] if len(counter) > 1 else 0
    if top_count == second_count:
        return "flat"
    return top_label


def _bh_fdr(pvals: NDArray[np.float64], q: float) -> NDArray[np.bool_]:
    """Benjamini-Hochberg FDR correction. Returns bool mask of discoveries."""
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvals)
    thresholds = (np.arange(1, n + 1) / n) * q
    rejected = np.zeros(n, dtype=bool)
    max_k = -1
    for k in range(n - 1, -1, -1):
        if pvals[order[k]] <= thresholds[k]:
            max_k = k
            break
    if max_k >= 0:
        rejected[order[: max_k + 1]] = True
    return rejected


def _fold_ic_values(
    signal: NDArray[np.float64],
    fwd: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    fold_boundaries: Sequence[pd.Timestamp] | None,
    datetimes: NDArray[np.datetime64],
) -> NDArray[np.float64]:
    """Compute per-fold IC values for sign-consistency estimation."""
    n = len(signal)

    # Build fold index boundaries
    if fold_boundaries is not None and len(fold_boundaries) > 0:
        dt_pd = pd.DatetimeIndex(datetimes)
        split_indices: list[int] = []
        for ts in fold_boundaries:
            idx = int(np.searchsorted(dt_pd, ts))
            if 0 < idx < n:
                split_indices.append(idx)
        split_indices = sorted(set(split_indices))
    else:
        n_folds = 4
        split_indices = [n * i // n_folds for i in range(1, n_folds)]

    # Build fold slices
    starts = [0, *split_indices]
    ends = [*split_indices, n]

    ic_list: list[float] = []
    for s, e in zip(starts, ends, strict=False):
        if e <= s:
            continue
        sl_valid = valid_mask[s:e] & np.isfinite(fwd[s:e]) & np.isfinite(signal[s:e])
        n_valid = int(np.sum(sl_valid))
        if n_valid < _MIN_IC_OBS:
            continue
        ic, _ = spearmanr(signal[s:e][sl_valid], fwd[s:e][sl_valid])
        if np.isfinite(ic):
            ic_list.append(float(ic))

    return np.array(ic_list, dtype=np.float64)


def _compute_net_edge_bps(
    signal: NDArray[np.float64],
    fwd: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    turnover_proxy: NDArray[np.float64],
    round_trip_cost_bps: float,
    tf: str,
) -> tuple[float, float]:
    """Compute net_edge_bps and turnover_per_year.

    net_edge_bps = gross_edge_bps - mean_turnover_per_trade * round_trip_cost_bps
    turnover_per_year = mean_turnover_per_bar * bars_per_year
    """
    valid = valid_mask & np.isfinite(fwd)
    if int(np.sum(valid)) < _MIN_IC_OBS:
        return 0.0, 0.0

    pos = np.sign(signal[valid])
    gross_bps = float(np.mean(pos * fwd[valid])) * 1e4

    mean_to_per_bar = float(np.mean(turnover_proxy[valid]))
    bpy = _bars_per_year(tf)
    turnover_per_year = mean_to_per_bar * bpy

    # net = gross - turnover_per_trade * cost (turnover_per_bar ≈ turnover_per_trade for binary positions)
    net_bps = gross_bps - mean_to_per_bar * round_trip_cost_bps

    return float(net_bps), float(turnover_per_year)


# ---------------------------------------------------------------------------
# Per-TF worker (module-level for picklability)
# ---------------------------------------------------------------------------
def _probe_tf_worker(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    """ProcessPoolExecutor worker: probe one tf, return list of cell dicts."""
    (
        resampled_maps_for_tf,  # dict[symbol, pd.DataFrame] for this tf
        symbols,
        tf,
        base_cfg_kwargs,  # serializable cfg fields as dict
        fold_boundary_strs,  # list[str] ISO timestamps or None
        round_trip_cost_bps,
    ) = args

    import logging as _logging

    from src.domain.futures.strategy.common.alignment import align_data_maps
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.rule_signals import build_rule_signal_panels

    _wlog = _logging.getLogger(__name__)

    # Reconstruct cfg with tf substitution and scaled bar params
    cfg = CandidateStrategyConfig(**{**base_cfg_kwargs, "timeframe": tf})

    # Build data_maps wrapper {symbol: {tf: df}}
    data_maps_tf: dict[str, dict[str, Any]] = {
        sym: {tf: df} for sym, df in resampled_maps_for_tf.items()
    }

    # Align
    try:
        aligned_tf = align_data_maps(data_maps_tf, list(symbols), tf)
    except Exception as exc:
        _wlog.warning("align_data_maps failed for tf=%s: %s", tf, exc)
        return []

    # Build signal panels
    try:
        panels = build_rule_signal_panels(aligned=aligned_tf, cfg=cfg)
    except Exception as exc:
        _wlog.warning("build_rule_signal_panels failed for tf=%s: %s", tf, exc)
        return []

    close_2d = aligned_tf.close_2d  # [T, N]
    datetimes = aligned_tf.datetimes
    syms = aligned_tf.symbols

    # Parse fold boundaries
    fold_boundaries_ts: list[pd.Timestamp] | None = None
    if fold_boundary_strs is not None:
        fold_boundaries_ts = [pd.Timestamp(s) for s in fold_boundary_strs]

    cell_dicts: list[dict[str, Any]] = []

    hpb_val = _hpb(tf)

    # Pre-compute VR/Hurst per (symbol x tf) — cache to avoid redundant re-computation per panel
    sym_vr_cache: dict[int, tuple[str, float]] = {}
    for n_idx in range(min(len(syms), close_2d.shape[1])):
        close_col_c = close_2d[:, n_idx]
        log_rets_c = np.diff(np.log(np.maximum(close_col_c, 1e-20))).astype(np.float64)
        finite_rets_c = log_rets_c[np.isfinite(log_rets_c)]
        sym_vr_cache[n_idx] = (_vr_label_majority(close_col_c), hurst_dfa(finite_rets_c))

    for panel in panels:
        family = panel.family
        variant = panel.variant
        archetype = str(panel.archetype)
        h_hold = panel.expected_holding_bars

        for n_idx, sym in enumerate(syms):
            if n_idx >= close_2d.shape[1]:
                continue
            close_col = close_2d[:, n_idx]
            signal_col = panel.signed_score_2d[:, n_idx]
            valid_col = panel.valid_mask_2d[:, n_idx]
            to_col = panel.turnover_proxy_2d[:, n_idx]

            fwd = _compute_forward_returns(close_col, h_hold)
            valid = valid_col & np.isfinite(fwd) & np.isfinite(signal_col)
            n_events = int(np.sum(valid))

            if n_events < _MIN_IC_OBS:
                cell_dicts.append(
                    {
                        "symbol": sym,
                        "family": family,
                        "variant": variant,
                        "archetype": archetype,
                        "tf": tf,
                        "n_obs": len(signal_col),
                        "n_events": n_events,
                        "ic_mean": 0.0,
                        "ic_tstat_hac": 0.0,
                        "ic_fold_sign_consistency": 0.0,
                        "alpha_half_life_h": float("nan"),
                        "net_edge_bps": 0.0,
                        "turnover_per_year": 0.0,
                        "vr_label": "flat",
                        "hurst": 0.5,
                        "passed_fdr": False,
                    }
                )
                continue

            ic_mean, _ = spearmanr(signal_col[valid], fwd[valid])
            if not np.isfinite(ic_mean):
                ic_mean = 0.0
            ic_mean = float(ic_mean)

            ic_tstat = _newey_west_ic_tstat(
                signal_col[valid].astype(np.float64),
                fwd[valid].astype(np.float64),
                max_lag=h_hold,
            )

            fold_ics = _fold_ic_values(
                signal_col, fwd, valid_col, fold_boundaries_ts, datetimes
            )
            fold_consistency = _fold_sign_consistency(fold_ics, ic_mean)

            half_life = _alpha_half_life(signal_col, close_col, valid_col, h_hold, hpb_val)

            net_bps, turnover_yr = _compute_net_edge_bps(
                signal_col, fwd, valid_col, to_col, round_trip_cost_bps, tf
            )

            # VR/Hurst from cache (computed once per symbol x tf, shared across panels)
            vr_lbl, hurst_val = sym_vr_cache.get(n_idx, ("flat", 0.5))

            cell_dicts.append(
                {
                    "symbol": sym,
                    "family": family,
                    "variant": variant,
                    "archetype": archetype,
                    "tf": tf,
                    "n_obs": len(signal_col),
                    "n_events": n_events,
                    "ic_mean": ic_mean,
                    "ic_tstat_hac": float(ic_tstat),
                    "ic_fold_sign_consistency": float(fold_consistency),
                    "alpha_half_life_h": float(half_life),
                    "net_edge_bps": float(net_bps),
                    "turnover_per_year": float(turnover_yr),
                    "vr_label": vr_lbl,
                    "hurst": float(hurst_val),
                    "passed_fdr": False,  # filled post-hoc
                }
            )

    return cell_dicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def probe_timeframe_alpha(
    *,
    data_maps: dict[str, dict[str, pd.DataFrame]],
    symbols: Sequence[str],
    base_cfg: Any,
    tf_grid: Sequence[str] = ("15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h"),
    fold_boundaries: Sequence[pd.Timestamp] | None = None,
    round_trip_cost_bps: float = 6.0,
    fdr_q: float = 0.10,
    max_workers: int = 12,
) -> TfProbeManifest:
    """Probe alpha signal quality for every (symbol x family x tf) cell.

    Args:
        data_maps: Per-symbol, per-tf OHLCV DataFrames.
        symbols: Symbols to probe.
        base_cfg: CandidateStrategyConfig instance as base configuration.
        tf_grid: Target timeframes to evaluate.
        fold_boundaries: Optional explicit fold split timestamps (4-equal split if None).
        round_trip_cost_bps: Full round-trip transaction cost in bps.
        fdr_q: Benjamini-Hochberg FDR target false-discovery rate.
        max_workers: Max parallel workers (one task per tf).

    Returns:
        TfProbeManifest with all diagnostic cells and summary metadata.
    """
    syms_list = list(symbols)
    coverage_by_tf: dict[str, int] = {}

    # Serialize fold boundaries for IPC
    fold_boundary_strs: list[str] | None = None
    if fold_boundaries is not None:
        fold_boundary_strs = [str(ts) for ts in fold_boundaries]

    # Extract serializable cfg fields
    import dataclasses

    try:
        base_cfg_kwargs: dict[str, Any] = {
            k: v
            for k, v in dataclasses.asdict(base_cfg).items()
            if not k.startswith("_")
        }
        # Remove non-init fields if any
        base_cfg_kwargs.pop("_purge_bars_input", None)
        base_cfg_kwargs.pop("_embargo_bars_input", None)
    except Exception as exc:
        _logger.warning("Failed to serialize base_cfg: %s", exc)
        base_cfg_kwargs = {}

    # Resample data for each tf and gather coverage
    all_cell_dicts: list[dict[str, Any]] = []

    def _build_resampled_maps(tf: str) -> dict[str, pd.DataFrame]:
        alias = _RESAMPLE_ALIAS.get(tf, tf)
        hpb_tf = _hpb(tf)
        resampled: dict[str, pd.DataFrame] = {}
        for sym in syms_list:
            sym_maps = data_maps.get(sym, {})
            # Pick finest available base tf (hpb <= target tf) for accurate resample
            source_df: pd.DataFrame | None = None
            for candidate_tf in ("1m", "5m", "15m", "30m", "1h", "4h", tf):
                if candidate_tf in sym_maps:
                    candidate_hpb = _hpb(candidate_tf)
                    if candidate_hpb <= hpb_tf:
                        source_df = sym_maps[candidate_tf]
                        if candidate_tf == tf:
                            break  # exact match — no resample needed
            if source_df is None:
                _logger.debug("No suitable base tf for sym=%s, tf=%s", sym, tf)
                continue
            try:
                if source_df.index.tz is not None:
                    source_df = source_df.copy()
                    source_df.index = source_df.index.tz_localize(None)
                resampled[sym] = _resample_ohlcv(source_df, alias)
            except Exception as exc:
                _logger.debug("Resample failed sym=%s tf=%s: %s", sym, tf, exc)
        return resampled

    # Build tf-worker args
    worker_args: list[tuple[Any, ...]] = []
    for tf in tf_grid:
        resampled_maps = _build_resampled_maps(tf)
        available_syms = [s for s in syms_list if s in resampled_maps]
        if not available_syms:
            _logger.warning("No data available for tf=%s, skipping.", tf)
            coverage_by_tf[tf] = 0
            continue

        # Coverage = median usable bar count across symbols
        bar_counts = [len(resampled_maps[s]) for s in available_syms]
        coverage_by_tf[tf] = int(np.median(bar_counts)) if bar_counts else 0

        worker_args.append(
            (
                resampled_maps,
                tuple(available_syms),
                tf,
                base_cfg_kwargs,
                fold_boundary_strs,
                round_trip_cost_bps,
            )
        )

    n_workers = min(max_workers, len(worker_args))
    if n_workers <= 0:
        _logger.warning("No valid tf tasks to run.")
        return TfProbeManifest(
            cells=(),
            tf_grid=tuple(tf_grid),
            coverage_by_tf=coverage_by_tf,
            diversity_corr={},
        )

    _logger.info("Launching tf probe: %d tf tasks, %d workers", len(worker_args), n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {
            executor.submit(_probe_tf_worker, args): args[2] for args in worker_args
        }
        for future in as_completed(future_map):
            tf_label = future_map[future]
            try:
                cells = future.result()
                all_cell_dicts.extend(cells)
                _logger.info("tf=%s: %d cells computed", tf_label, len(cells))
            except Exception as exc:
                _logger.error("tf=%s worker failed: %s", tf_label, exc)

    # ---------------------------------------------------------------------------
    # Post-hoc: BH-FDR across all cells
    # ---------------------------------------------------------------------------
    if all_cell_dicts:
        tstats = np.array(
            [float(c["ic_tstat_hac"]) for c in all_cell_dicts], dtype=np.float64
        )
        pvals = 2.0 * norm.sf(np.abs(tstats))
        discoveries = _bh_fdr(pvals, fdr_q)
        for i, cell_d in enumerate(all_cell_dicts):
            cell_d["passed_fdr"] = bool(discoveries[i])

    # ---------------------------------------------------------------------------
    # Diversity correlation: same (symbol, family) across tf pairs
    # ---------------------------------------------------------------------------
    diversity_corr: dict[str, float] = {}
    try:
        diversity_corr = _compute_diversity_corr(
            all_cell_dicts, data_maps, syms_list, tf_grid
        )
    except Exception as exc:
        _logger.warning("Diversity corr computation failed: %s", exc)

    built_cells: tuple[TfCellEvidence, ...] = tuple(
        TfCellEvidence(**d) for d in all_cell_dicts
    )
    return TfProbeManifest(
        cells=built_cells,
        tf_grid=tuple(tf_grid),
        coverage_by_tf=coverage_by_tf,
        diversity_corr=diversity_corr,
    )


def _compute_diversity_corr(
    cell_dicts: list[dict[str, Any]],
    data_maps: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    tf_grid: Sequence[str],
) -> dict[str, float]:
    """Compute Pearson correlation of forward-return proxies across (symbol,family) tf pairs."""
    from scipy.stats import pearsonr

    diversity: dict[str, float] = {}
    tf_list = list(tf_grid)

    # Build unique (symbol, family) pairs
    sym_family_pairs: set[tuple[str, str]] = set()
    for c in cell_dicts:
        sym_family_pairs.add((c["symbol"], c["family"]))

    # Build close-return series resampled to 4h grid for comparability
    base_alias = _RESAMPLE_ALIAS.get(_BASE_TF, _BASE_TF)

    for sym, family in sym_family_pairs:
        sym_maps = data_maps.get(sym, {})
        if not sym_maps:
            continue

        # Get 4h close series
        ref_df: pd.DataFrame | None = None
        for candidate in ("4h", "1h", "30m", "15m"):
            if candidate in sym_maps:
                ref_df = sym_maps[candidate]
                break
        if ref_df is None:
            continue

        try:
            if ref_df.index.tz is not None:
                ref_df = ref_df.copy()
                ref_df.index = ref_df.index.tz_localize(None)
            base_close = _resample_ohlcv(ref_df, base_alias)["close"]
        except Exception as exc:
            _logger.debug("Resample failed for diversity corr sym=%s: %s", sym, exc)
            continue

        base_rets = base_close.pct_change().dropna()
        if len(base_rets) < 30:
            continue

        # For each tf pair, use close returns at target tf aligned to common 4h grid
        tf_close_at_base: dict[str, pd.Series] = {"_base": base_rets}

        for tf in tf_list:
            if tf == _BASE_TF:
                tf_close_at_base[tf] = base_rets
                continue
            alias = _RESAMPLE_ALIAS.get(tf, tf)
            src_df = None
            for candidate in (tf, "1h", "4h"):
                if candidate in sym_maps:
                    src_df = sym_maps[candidate]
                    break
            if src_df is None:
                continue
            try:
                if src_df.index.tz is not None:
                    src_df = src_df.copy()
                    src_df.index = src_df.index.tz_localize(None)
                resampled_tf = _resample_ohlcv(src_df, alias)
                # Resample to 4h grid for alignment
                tf_close = resampled_tf["close"].resample(base_alias).last().dropna()
                tf_rets = tf_close.pct_change().dropna()
                tf_close_at_base[tf] = tf_rets
            except Exception as exc:
                _logger.debug("Resample failed for diversity tf=%s sym=%s: %s", tf, sym, exc)
                continue

        # Compute Pearson r for all pairs of tfs that have data
        available_tfs = [t for t in tf_list if t in tf_close_at_base]
        for i_idx in range(len(available_tfs)):
            for j_idx in range(i_idx + 1, len(available_tfs)):
                tf_a = available_tfs[i_idx]
                tf_b = available_tfs[j_idx]
                s_a = tf_close_at_base[tf_a]
                s_b = tf_close_at_base[tf_b]
                # Align on common index
                common = s_a.index.intersection(s_b.index)
                if len(common) < 20:
                    continue
                try:
                    r, _ = pearsonr(s_a.loc[common].values, s_b.loc[common].values)
                    key = f"{sym}:{family}:{tf_a}~{tf_b}"
                    diversity[key] = float(r)
                except Exception as exc:
                    _logger.debug("pearsonr failed %s~%s sym=%s: %s", tf_a, tf_b, sym, exc)

    return diversity


def select_tf_family_cells(
    manifest: TfProbeManifest,
    *,
    min_ic_tstat: float = 2.0,
    require_fdr: bool = True,
    min_net_edge_bps: float = 0.0,
    min_fold_sign_consistency: float = 0.75,
) -> tuple[TfCellEvidence, ...]:
    """Filter and rank promotable cells from a TfProbeManifest.

    Args:
        manifest: Probe manifest from probe_timeframe_alpha.
        min_ic_tstat: Minimum NW HAC IC t-stat (inclusive).
        require_fdr: If True, require passed_fdr == True.
        min_net_edge_bps: Minimum cost-adjusted edge in bps (inclusive).
        min_fold_sign_consistency: Minimum fraction of folds with consistent IC sign.

    Returns:
        Ranked tuple of promotable TfCellEvidence (ic_tstat_hac desc, net_edge_bps desc).
    """
    selected: list[TfCellEvidence] = []
    for cell in manifest.cells:
        if cell.ic_tstat_hac < min_ic_tstat:
            continue
        if require_fdr and not cell.passed_fdr:
            continue
        if cell.net_edge_bps < min_net_edge_bps:
            continue
        if cell.ic_fold_sign_consistency < min_fold_sign_consistency:
            continue
        selected.append(cell)

    selected.sort(key=lambda c: (-c.ic_tstat_hac, -c.net_edge_bps))
    return tuple(selected)
