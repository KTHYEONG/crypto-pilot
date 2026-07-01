# src/domain/futures/strategy/tiered_workflow/diagnostics.py

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

from src.domain.futures.allocation.contracts import (
    LayerUniverseAudit,
    PredictionDecompositionDiag,
    StrategySignal,
    SymbolRealizedStat,
)
from src.domain.futures.allocation.metrics import (
    _is_non_constant_finite_array,
    _nw_tstat_realized,
)

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.walk_forward import WFFold

logger = logging.getLogger(__name__)


def _is_trained_fold_output(fold_out: Any) -> bool:
    """Return whether the fold completed training with non-degenerate predictions."""
    return getattr(fold_out, "fit_status", "trained") == "trained"


def _as_bool_mask(value: Any, *, shape: tuple[int, int], default: bool) -> NDArray[np.bool_]:
    if value is None:
        return np.full(shape, default, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        return np.full(shape, default, dtype=bool)
    return arr


def _resolve_layer_window_bounds(
    *,
    n_bars: int,
    start_idx: int,
    end_idx: int,
) -> tuple[int, int]:
    start = max(0, min(int(start_idx), n_bars))
    end = max(start, min(int(end_idx), n_bars))
    return start, end


def _date_label(datetimes: NDArray[np.datetime64], idx: int) -> str:
    if idx < 0 or idx >= len(datetimes):
        return "—"
    return str(pd.Timestamp(datetimes[idx]).date())


def _resolve_audit_symbols(
    *,
    aligned: AlignedMarketData,
    symbols: Sequence[str] | None,
) -> tuple[str, ...]:
    if symbols is None:
        return tuple(aligned.symbols)
    available = set(aligned.symbols)
    return tuple(sym for sym in symbols if sym in available)


def build_layer_universe_audit(
    *,
    aligned: AlignedMarketData,
    layer: str,
    start_idx: int,
    end_idx: int,
    symbols: Sequence[str] | None = None,
) -> LayerUniverseAudit:
    """Build a causal universe audit for a layer window."""
    n_bars = len(aligned.datetimes)
    start, end = _resolve_layer_window_bounds(n_bars=n_bars, start_idx=start_idx, end_idx=end_idx)
    audit_symbols = _resolve_audit_symbols(aligned=aligned, symbols=symbols)
    symbol_index = {sym: idx for idx, sym in enumerate(aligned.symbols)}
    symbol_positions = tuple(symbol_index[sym] for sym in audit_symbols if sym in symbol_index)
    if not audit_symbols or end <= start or not symbol_positions:
        return LayerUniverseAudit(
            layer=layer,
            start_idx=start,
            end_idx=end,
            start_date=_date_label(aligned.datetimes, start),
            end_date=_date_label(aligned.datetimes, max(end - 1, start)),
            symbol_count=len(symbol_positions),
            active_symbol_count_min=0,
            active_symbol_count_median=0.0,
            active_symbol_count_max=0,
            entry_block_count=0,
            kill_count=0,
            symbols=audit_symbols,
            warnings=("empty_window",),
        )

    col_idx = np.asarray(symbol_positions, dtype=np.intp)
    slice_obj = slice(start, end)
    active_mask = getattr(aligned, "inference_active_mask", None)
    if not isinstance(active_mask, np.ndarray):
        active_mask = getattr(aligned, "active_mask", None)
    active_mask = _as_bool_mask(active_mask, shape=(n_bars, len(aligned.symbols)), default=True)
    warm_mask = getattr(aligned, "inference_entry_warm_mask", None)
    if not isinstance(warm_mask, np.ndarray):
        warm_mask = getattr(aligned, "warm_mask", None)
    warm_mask = _as_bool_mask(warm_mask, shape=(n_bars, len(aligned.symbols)), default=True)
    entry_block_mask = _as_bool_mask(
        getattr(aligned, "entry_block_mask", None),
        shape=(n_bars, len(aligned.symbols)),
        default=False,
    )
    kill_mask = _as_bool_mask(
        getattr(aligned, "kill_mask", None),
        shape=(n_bars, len(aligned.symbols)),
        default=False,
    )

    effective_mask = (
        active_mask[slice_obj][:, col_idx]
        & warm_mask[slice_obj][:, col_idx]
        & ~entry_block_mask[slice_obj][:, col_idx]
        & ~kill_mask[slice_obj][:, col_idx]
    )
    active_counts = np.sum(effective_mask, axis=1, dtype=np.int64)
    entry_counts = np.sum(entry_block_mask[slice_obj][:, col_idx], axis=1, dtype=np.int64)
    kill_counts = np.sum(kill_mask[slice_obj][:, col_idx], axis=1, dtype=np.int64)

    window_len = int(end - start)
    tail_len = max(int(np.ceil(window_len * 0.2)), 1)
    tail_slice = slice(max(end - tail_len, start), end)
    tail_active_counts = np.sum(
        active_mask[tail_slice][:, col_idx]
        & warm_mask[tail_slice][:, col_idx]
        & ~entry_block_mask[tail_slice][:, col_idx]
        & ~kill_mask[tail_slice][:, col_idx],
        axis=1,
        dtype=np.int64,
    )
    tail_entry_counts = np.sum(entry_block_mask[tail_slice][:, col_idx], axis=1, dtype=np.int64)
    tail_kill_counts = np.sum(kill_mask[tail_slice][:, col_idx], axis=1, dtype=np.int64)

    active_median = float(np.median(active_counts)) if active_counts.size else 0.0
    tail_active_median = float(np.median(tail_active_counts)) if tail_active_counts.size else 0.0
    entry_median = float(np.median(entry_counts)) if entry_counts.size else 0.0
    tail_entry_median = float(np.median(tail_entry_counts)) if tail_entry_counts.size else 0.0
    kill_median = float(np.median(kill_counts)) if kill_counts.size else 0.0
    tail_kill_median = float(np.median(tail_kill_counts)) if tail_kill_counts.size else 0.0

    warnings_list: list[str] = []
    if window_len <= 0:
        warnings_list.append("empty_window")
    else:
        if active_median > 0.0 and tail_active_median < 0.8 * active_median:
            warnings_list.append("low_active_tail")
        if tail_entry_median > 2.0 * entry_median:
            warnings_list.append("entry_block_spike")
        if tail_kill_median > 2.0 * kill_median:
            warnings_list.append("kill_spike")

    return LayerUniverseAudit(
        layer=layer,
        start_idx=start,
        end_idx=end,
        start_date=_date_label(aligned.datetimes, start),
        end_date=_date_label(aligned.datetimes, max(end - 1, start)),
        symbol_count=len(symbol_positions),
        active_symbol_count_min=int(np.min(active_counts)) if active_counts.size else 0,
        active_symbol_count_median=active_median,
        active_symbol_count_max=int(np.max(active_counts)) if active_counts.size else 0,
        entry_block_count=int(np.sum(entry_counts)) if entry_counts.size else 0,
        kill_count=int(np.sum(kill_counts)) if kill_counts.size else 0,
        symbols=audit_symbols,
        warnings=tuple(warnings_list),
    )


def _fold_eligible_symbol_mask(
    *,
    aligned: AlignedMarketData,
    fold: WFFold,
    min_bar_coverage: float = 0.80,
) -> NDArray[np.bool_]:
    """Compute fold-local PIT eligible symbols from OOS universe and warm/kill masks."""
    if fold.oos_end <= fold.oos_start:
        return np.zeros(len(aligned.symbols), dtype=bool)

    active_mask = getattr(aligned, "inference_active_mask", None)
    if not isinstance(active_mask, np.ndarray):
        active_mask = getattr(aligned, "active_mask", None)
    if not isinstance(active_mask, np.ndarray):
        active_mask = np.ones((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    warm_mask = getattr(aligned, "inference_entry_warm_mask", None)
    if not isinstance(warm_mask, np.ndarray):
        warm_mask = getattr(aligned, "warm_mask", None)
    if not isinstance(warm_mask, np.ndarray):
        warm_mask = np.ones((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    entry_block_mask = getattr(aligned, "entry_block_mask", None)
    if not isinstance(entry_block_mask, np.ndarray):
        entry_block_mask = np.zeros((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    kill_mask = getattr(aligned, "kill_mask", None)
    if not isinstance(kill_mask, np.ndarray):
        kill_mask = np.zeros((len(aligned.datetimes), len(aligned.symbols)), dtype=bool)

    oos_slice = slice(fold.oos_start, fold.oos_end)
    eligible_2d = (
        active_mask[oos_slice]
        & warm_mask[oos_slice]
        & ~entry_block_mask[oos_slice]
        & ~kill_mask[oos_slice]
    )
    coverage = np.mean(eligible_2d.astype(np.float64), axis=0)
    return np.asarray(coverage >= float(min_bar_coverage), dtype=bool)


def compute_per_symbol_ic(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
) -> dict[str, float]:
    """심볼별 time-series Spearman rank IC (expected_net_bps vs oos_set.y_return_bps)."""
    sym_ic_lists: dict[str, list[float]] = defaultdict(list)

    for _, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue

        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        if y_realized is None:
            continue

        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if events_df.empty or "symbol" not in events_df.columns:
            continue

        pred: NDArray[np.float64] = np.asarray(
            fold_out.model_output.expected_net_bps, dtype=np.float64
        )
        realized: NDArray[np.float64] = np.asarray(y_realized, dtype=np.float64)

        if len(pred) != len(realized) or len(pred) != len(events_df):
            continue

        symbols_arr = events_df["symbol"].to_numpy()

        for sym in np.unique(symbols_arr):
            sym_mask = symbols_arr == sym
            p = pred[sym_mask]
            r = realized[sym_mask]
            valid_mask = np.isfinite(p) & np.isfinite(r)
            if valid_mask.sum() < 4:
                continue
            if not _is_non_constant_finite_array(p[valid_mask]):
                continue
            if not _is_non_constant_finite_array(r[valid_mask]):
                continue
            ic_val, _ = spearmanr(p[valid_mask], r[valid_mask])
            if not np.isnan(ic_val):
                sym_ic_lists[str(sym)].append(float(ic_val))

    return {sym: float(np.mean(ics)) for sym, ics in sym_ic_lists.items() if ics}


def _compute_fold_realized_valid_set(
    fold_out: Any,
    *,
    min_obs: int = 20,
    t_stat_floor: float = 1.96,
) -> frozenset[str]:
    """Per-fold: symbols passing realized NW t-stat QC (BUG-B 방어)."""
    if not _is_trained_fold_output(fold_out):
        return frozenset()
    oos_set = getattr(fold_out, "oos_set", None)
    if oos_set is None:
        return frozenset()
    y_realized = getattr(oos_set, "y_return_bps", None)
    if y_realized is None:
        y_realized = getattr(oos_set, "y_edge_bps", None)
    if y_realized is None:
        return frozenset()
    events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
    if events_df.empty or "symbol" not in events_df.columns:
        return frozenset()

    realized = np.asarray(y_realized, dtype=np.float64)
    symbols_arr = events_df["symbol"].to_numpy()
    if len(realized) != len(symbols_arr):
        return frozenset()

    valid_syms: set[str] = set()
    for sym in np.unique(symbols_arr):
        mask = symbols_arr == sym
        r_sym = realized[mask]
        r_sym = r_sym[np.isfinite(r_sym)]
        if len(r_sym) < min_obs:
            continue
        t = _nw_tstat_realized(r_sym)
        if abs(t) >= t_stat_floor:
            valid_syms.add(str(sym))
    return frozenset(valid_syms)


def compute_per_symbol_realized_stats(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
    min_obs: int,
    t_stat_floor: float,
    per_symbol_ic: dict[str, float],
) -> dict[str, SymbolRealizedStat]:
    """fold-pooled 실현 수익 기반 per-symbol QC (BUG-A+B 교정)."""
    sym_returns: dict[str, list[float]] = defaultdict(list)

    for _, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue
        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        if y_realized is None:
            continue
        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if events_df.empty or "symbol" not in events_df.columns:
            continue

        realized = np.asarray(y_realized, dtype=np.float64)
        symbols_arr = events_df["symbol"].to_numpy()
        if len(realized) != len(symbols_arr):
            continue

        for sym in np.unique(symbols_arr):
            mask = symbols_arr == sym
            r_sym = realized[mask]
            r_valid = r_sym[np.isfinite(r_sym)]
            sym_returns[str(sym)].extend(r_valid.tolist())

    result: dict[str, SymbolRealizedStat] = {}
    for sym, returns_list in sym_returns.items():
        r_arr = np.asarray(returns_list, dtype=np.float64)
        n = len(r_arr)
        mu = float(np.mean(r_arr)) if n > 0 else 0.0
        t = _nw_tstat_realized(r_arr) if n >= 4 else 0.0
        ic = per_symbol_ic.get(sym, 0.0)
        valid = (
            n >= min_obs
            and abs(t) >= t_stat_floor
            and bool(np.isfinite(mu))
            and bool(np.isfinite(t))
            and ic > 0.0
        )
        result[sym] = SymbolRealizedStat(
            realized_mu_bps=mu,
            t_stat=t,
            n_obs=n,
            ic=ic,
            valid=valid,
        )
    return result


def compute_per_strategy_oos_validation(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
    min_obs: int = 30,
    t_stat_floor: float = 1.5,
    consistency_floor: float = 0.60,
) -> tuple[StrategySignal, ...]:
    """rule-family:variant별 OOS 독립검증."""
    per_strategy_realized: dict[str, list[float]] = defaultdict(list)
    per_strategy_fold_edge: dict[str, dict[int, float]] = defaultdict(dict)

    for fold_idx, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue

        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if y_realized is None or events_df.empty:
            continue

        realized = np.asarray(y_realized, dtype=np.float64)
        if len(realized) != len(events_df):
            continue

        if "family" in events_df.columns:
            family_col = events_df["family"].astype(str)
        elif "archetype" in events_df.columns:
            family_col = events_df["archetype"].astype(str)
        else:
            family_col = pd.Series(["_unknown"] * len(events_df), index=events_df.index, dtype="object")
        if "variant" in events_df.columns:
            variant_col = events_df["variant"].astype(str)
        else:
            variant_col = pd.Series(["_unknown"] * len(events_df), index=events_df.index, dtype="object")

        fold_bucket: dict[str, list[float]] = defaultdict(list)
        for idx, value in enumerate(realized):
            if not np.isfinite(value):
                continue
            strategy_id = f"{family_col.iat[idx]}:{variant_col.iat[idx]}"
            per_strategy_realized[strategy_id].append(float(value))
            fold_bucket[strategy_id].append(float(value))

        for strategy_id, values in fold_bucket.items():
            if values:
                per_strategy_fold_edge[strategy_id][fold_idx] = float(np.mean(values))

    panel: list[StrategySignal] = []
    for strategy_id in sorted(per_strategy_realized):
        realized_clean = np.asarray(per_strategy_realized[strategy_id], dtype=np.float64)
        if len(realized_clean) == 0:
            continue
        fold_edges = tuple(
            sorted((int(fold_id), float(edge)) for fold_id, edge in per_strategy_fold_edge.get(strategy_id, {}).items())
        )
        n_folds = len(fold_edges)
        fold_consistency = (
            float(sum(1 for _, edge in fold_edges if edge > 0.0) / n_folds)
            if n_folds > 0 else 0.0
        )
        nw_tstat = _nw_tstat_realized(realized_clean)
        panel.append(
            StrategySignal(
                strategy_id=strategy_id,
                oos_edge_bps=float(np.mean(realized_clean)),
                oos_nw_tstat=nw_tstat,
                hit_rate=float(np.mean(realized_clean > 0.0)),
                fold_sign_consistency=fold_consistency,
                n_obs=len(realized_clean),
                n_folds=n_folds,
                valid=bool(
                    len(realized_clean) >= min_obs
                    and nw_tstat >= t_stat_floor
                    and fold_consistency >= consistency_floor
                ),
                _fold_edges=fold_edges,
            )
        )
    return tuple(panel)


def _compute_fold_ts_ic(*, fold_out: Any) -> float | None:
    """fold OOS pooled time-series Spearman rank IC."""
    if not _is_trained_fold_output(fold_out):
        return None
    oos_set = getattr(fold_out, "oos_set", None)
    if oos_set is None:
        return None

    y_realized = getattr(oos_set, "y_return_bps", None)
    if y_realized is None:
        y_realized = getattr(oos_set, "y_edge_bps", None)
    if y_realized is None:
        return None

    pred: NDArray[np.float64] = np.asarray(
        fold_out.model_output.expected_net_bps, dtype=np.float64
    )
    realized: NDArray[np.float64] = np.asarray(y_realized, dtype=np.float64)

    if len(pred) != len(realized) or len(pred) < 4:
        return None

    mask = np.isfinite(pred) & np.isfinite(realized)
    if mask.sum() < 4:
        return None
    if not _is_non_constant_finite_array(pred[mask]):
        return None
    if not _is_non_constant_finite_array(realized[mask]):
        return None

    ic_val, _ = spearmanr(pred[mask], realized[mask])
    return float(ic_val) if not np.isnan(ic_val) else None


def compute_prediction_decomposition_diag(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
) -> PredictionDecompositionDiag:
    """OOS 이벤트에서 예측의 정적/동적 분산 분해 + archetype 엣지 + decile lift (진단 전용)."""
    all_pred: list[NDArray[np.float64]] = []
    all_real: list[NDArray[np.float64]] = []
    all_archetype: list[list[str]] = []
    all_regime: list[list[int]] = []
    all_variant: list[list[str]] = []
    score_cal_ratios: list[float] = []

    for _, _, fold_out in fold_tuples:
        if not _is_trained_fold_output(fold_out):
            continue
        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            continue

        y_realized = getattr(oos_set, "y_return_bps", None)
        if y_realized is None:
            y_realized = getattr(oos_set, "y_edge_bps", None)
        if y_realized is None:
            continue

        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        if events_df.empty:
            continue

        pred: NDArray[np.float64] = np.asarray(
            fold_out.model_output.expected_net_bps, dtype=np.float64
        )
        real: NDArray[np.float64] = np.asarray(y_realized, dtype=np.float64)

        if len(pred) != len(real) or len(pred) != len(events_df):
            continue

        mask = np.isfinite(pred) & np.isfinite(real)
        if mask.sum() < 4:
            continue

        pred_m = pred[mask]
        real_m = real[mask]
        ev_m = events_df.loc[mask.tolist() if not isinstance(mask, np.ndarray) else events_df.index[mask]]

        n_m = int(mask.sum())
        arch_col = (
            ev_m["archetype"].astype(str).tolist() if "archetype" in ev_m.columns
            else ["_unknown"] * n_m
        )
        regime_col: list[int] = []
        if "entry_regime_code" in ev_m.columns:
            regime_col = [int(v) if pd.notna(v) else -1 for v in ev_m["entry_regime_code"]]
        else:
            regime_col = [-1] * n_m
        variant_col = (
            ev_m["variant"].astype(str).tolist() if "variant" in ev_m.columns
            else ["_unknown"] * n_m
        )

        all_pred.append(pred_m)
        all_real.append(real_m)
        all_archetype.append(arch_col)
        all_regime.append(regime_col)
        all_variant.append(variant_col)

        val_diag = getattr(fold_out.model_output, "validation_diagnostics", {})
        ens_diag = val_diag.get("ensemble_diagnostics", {}) if isinstance(val_diag, dict) else {}
        n_valid_reg = int(ens_diag.get("num_valid_regimes", 0)) if isinstance(ens_diag, dict) else 0
        n_unique_reg = max(len(set(regime_col) - {-1}), 1)
        score_cal_ratios.append(float(n_valid_reg) / float(n_unique_reg))

    if not all_pred:
        return PredictionDecompositionDiag(
            static_variance_share=0.0,
            dynamic_variance_share=0.0,
            score_cal_valid_ratio=0.0,
            per_archetype_oos_edge={},
            decile_lift_bps=0.0,
        )

    pred_arr = np.concatenate(all_pred, axis=0)
    real_arr = np.concatenate(all_real, axis=0)
    arch_arr = np.array([a for sub in all_archetype for a in sub])
    regime_arr = np.array([r for sub in all_regime for r in sub], dtype=np.int32)
    variant_arr = np.array([v for sub in all_variant for v in sub])

    n_total = len(pred_arr)

    total_var = float(np.var(pred_arr)) if n_total > 1 else 0.0
    static_var = 0.0
    if total_var > 1e-20:
        group_keys = [f"{a}|{r}|{v}" for a, r, v in zip(arch_arr, regime_arr, variant_arr, strict=True)]
        group_key_arr = np.array(group_keys)
        group_mean_arr: NDArray[np.float64] = np.zeros(n_total, dtype=np.float64)
        for gk in np.unique(group_key_arr):
            gm = group_key_arr == gk
            n_g = int(gm.sum())
            if n_g < 2:
                group_mean_arr[gm] = pred_arr[gm]
                continue
            group_mean_arr[gm] = float(np.mean(pred_arr[gm]))
        static_var = float(np.var(group_mean_arr)) if n_total > 1 else 0.0

    static_share = float(np.clip(static_var / (total_var + 1e-20), 0.0, 1.0))
    dynamic_share = float(max(0.0, 1.0 - static_share))

    per_archetype_oos_edge: dict[str, tuple[float, float]] = {}
    for arch in np.unique(arch_arr):
        a_mask = arch_arr == arch
        r_sub: NDArray[np.float64] = real_arr[a_mask]
        if len(r_sub) < 4:
            continue
        mu_arch = float(np.mean(r_sub))
        t_arch = _nw_tstat_realized(r_sub)
        per_archetype_oos_edge[str(arch)] = (mu_arch, t_arch)

    decile_lift_bps = 0.0
    if n_total >= 20:
        n10 = max(1, n_total // 10)
        order = np.argsort(pred_arr)
        top_real = real_arr[order[-n10:]]
        bot_real = real_arr[order[:n10]]
        decile_lift_bps = float(np.mean(top_real) - np.mean(bot_real))

    score_cal_valid_ratio = float(np.mean(score_cal_ratios)) if score_cal_ratios else 0.0

    return PredictionDecompositionDiag(
        static_variance_share=static_share,
        dynamic_variance_share=dynamic_share,
        score_cal_valid_ratio=score_cal_valid_ratio,
        per_archetype_oos_edge=per_archetype_oos_edge,
        decile_lift_bps=decile_lift_bps,
    )


def _log_fold_regime_analysis(
    *,
    fold_tuples: list[tuple[int, Any, Any]],
    datetimes: NDArray[np.datetime64],
) -> None:
    """각 fold OOS 날짜 범위·regime 분포·archetype별 실현mu를 로깅."""
    n_bars = len(datetimes)

    for fold_idx, wf_fold, fold_out in fold_tuples:
        oos_s = int(getattr(wf_fold, "oos_start", 0))
        oos_e = int(getattr(wf_fold, "oos_end", 0))
        oos_s_clamp = max(0, min(oos_s, n_bars - 1))
        oos_e_clamp = max(0, min(oos_e - 1, n_bars - 1))
        date_start = str(datetimes[oos_s_clamp])[:10]
        date_end = str(datetimes[oos_e_clamp])[:10]

        if not _is_trained_fold_output(fold_out):
            logger.debug(
                "[SWF-DIAG-FOLD%d] %s~%s fit_status=%s SKIP",
                fold_idx + 1, date_start, date_end,
                getattr(fold_out, "fit_status", "unknown"),
            )
            continue

        oos_set = getattr(fold_out, "oos_set", None)
        if oos_set is None:
            logger.debug("[SWF-DIAG-FOLD%d] %s~%s oos_set=None SKIP", fold_idx + 1, date_start, date_end)
            continue

        events_df: pd.DataFrame = getattr(oos_set, "event_index", pd.DataFrame())
        y_raw = getattr(oos_set, "y_return_bps", None)
        if events_df.empty or y_raw is None:
            logger.debug("[SWF-DIAG-FOLD%d] %s~%s no_events SKIP", fold_idx + 1, date_start, date_end)
            continue

        y_arr: NDArray[np.float64] = np.asarray(y_raw, dtype=np.float64)
        n_ev = len(y_arr)

        regime_dist = ""
        if "entry_regime_code" in events_df.columns and n_ev > 0:
            rc = events_df["entry_regime_code"].dropna().astype(int)
            counts = rc.value_counts().sort_index()
            regime_dist = " ".join(f"r{k}:{v}" for k, v in counts.items())

        arch_mu_str = ""
        if "archetype" in events_df.columns and n_ev == len(events_df):
            arch_col = events_df["archetype"].astype(str).to_numpy()
            arch_parts = []
            for arch in np.unique(arch_col):
                mask = arch_col == arch
                r_sub = y_arr[mask]
                finite = r_sub[np.isfinite(r_sub)]
                if len(finite) >= 4:
                    mu = float(np.mean(finite))
                    label = arch.replace("_reversion", "").replace("_continuation", "").replace("time_series_", "ts_")
                    arch_parts.append(f"{label}:{mu:.1f}(n={len(finite)})")
            arch_mu_str = " | ".join(arch_parts)

        pred_arr: NDArray[np.float64] = np.asarray(
            fold_out.model_output.expected_net_bps, dtype=np.float64
        )
        cs_ic_str = "ic=N/A"
        if len(pred_arr) == n_ev and n_ev >= 4:
            mask_f = np.isfinite(pred_arr) & np.isfinite(y_arr)
            if mask_f.sum() >= 4:
                from scipy.stats import spearmanr as _sr
                _ic, _ = _sr(pred_arr[mask_f], y_arr[mask_f])
                cs_ic_str = f"ic={float(_ic):.4f}"

        logger.debug(
            "[SWF-DIAG-FOLD%d] %s~%s n=%d %s | regime[%s] | arch[%s]",
            fold_idx + 1, date_start, date_end, n_ev, cs_ic_str, regime_dist, arch_mu_str,
        )
