from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.market_regime import (
    compute_market_regime_context,
    compute_risk_overlay,
)

_logger = logging.getLogger(__name__)
_ROBUST_Z_EPS = 1e-9
_ROBUST_Z_CLIP = 3.0

_ARCHETYPE_REGIME_AFFINITY: dict[tuple[str, str], float] = {
    ("trend", "bull_quiet"): 1.0,
    ("trend", "bull_volatile"): 0.5,
    ("trend", "bear_quiet"): -0.5,
    ("trend", "bear_volatile"): -1.0,
    ("trend", "transition"): 0.0,
    ("trend", "crash"): -1.0,
    ("ts_mom", "bull_quiet"): 1.0,
    ("ts_mom", "bull_volatile"): 0.7,
    ("ts_mom", "bear_quiet"): -0.3,
    ("ts_mom", "bear_volatile"): -0.7,
    ("ts_mom", "transition"): 0.0,
    ("ts_mom", "crash"): -1.0,
    ("mean_rev", "bull_quiet"): 0.0,
    ("mean_rev", "bull_volatile"): 0.8,
    ("mean_rev", "bear_quiet"): 0.0,
    ("mean_rev", "bear_volatile"): 0.8,
    ("mean_rev", "transition"): 0.5,
    ("mean_rev", "crash"): -0.5,
    ("carry_rev", "bull_quiet"): 0.3,
    ("carry_rev", "bull_volatile"): 0.5,
    ("carry_rev", "bear_quiet"): 0.3,
    ("carry_rev", "bear_volatile"): 0.5,
    ("carry_rev", "transition"): 0.8,
    ("carry_rev", "crash"): 0.0,
    ("flow_rev", "bull_quiet"): 0.2,
    ("flow_rev", "bull_volatile"): 0.7,
    ("flow_rev", "bear_quiet"): 0.2,
    ("flow_rev", "bear_volatile"): 0.7,
    ("flow_rev", "transition"): 0.5,
    ("flow_rev", "crash"): 0.3,
    ("unwind", "bull_quiet"): -0.3,
    ("unwind", "bull_volatile"): 0.3,
    ("unwind", "bear_quiet"): 0.0,
    ("unwind", "bear_volatile"): 0.5,
    ("unwind", "transition"): 0.3,
    ("unwind", "crash"): 1.0,
    ("beta_neut", "bull_quiet"): -0.2,
    ("beta_neut", "bull_volatile"): 0.5,
    ("beta_neut", "bear_quiet"): 0.2,
    ("beta_neut", "bear_volatile"): 0.5,
    ("beta_neut", "transition"): 0.3,
    ("beta_neut", "crash"): -0.3,
}


_ALIGNED_FEATURE_CACHE: dict[int, dict[str, Any]] = {}


@dataclass(slots=True, frozen=True)
class CandidateDataset:
    """Candidate tabular dataset contract."""

    X: NDArray[np.float32]
    y_gate: NDArray[np.int8]
    y_edge_bps: NDArray[np.float32]
    y_q10_bps: NDArray[np.float32]
    y_mfe_bps: NDArray[np.float32]
    gate_weight: NDArray[np.float32]
    edge_weight: NDArray[np.float32]
    groups: NDArray[np.int32]
    event_index: pd.DataFrame
    feature_names: tuple[str, ...]
    effective_sample_size: float = 0.0
    feature_schema_version: str = "candidate_v6"
    y_return_r: NDArray[np.float32] | None = None
    y_return_bps: NDArray[np.float32] | None = None
    y_gross_return_bps: NDArray[np.float32] | None = None
    y_gross_return_r: NDArray[np.float32] | None = None
    y_mae_r: NDArray[np.float32] | None = None
    risk_unit_bps: NDArray[np.float32] | None = None


@dataclass(slots=True, frozen=True)
class CandidateFeatureSchema:
    """Train-fit feature schema contract for candidate datasets."""

    feature_names: tuple[str, ...]
    identity_categories: tuple[str, ...]
    version: str = "candidate_v6"


def _find_symbol_index(symbols: tuple[str, ...], symbol: str) -> int:
    for idx, value in enumerate(symbols):
        if value == symbol:
            return idx
    raise KeyError(f"unknown symbol: {symbol}")


def _rolling_robust_z_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    median = series.rolling(window, min_periods=1).median()
    mad = series.rolling(window, min_periods=1).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))),
        raw=True,
    )
    z = (series - median) / np.maximum(mad * 1.4826, _ROBUST_Z_EPS)
    return np.asarray(
        z.fillna(0.0).clip(-_ROBUST_Z_CLIP, _ROBUST_Z_CLIP).to_numpy(),
        dtype=np.float64,
    )


def _rolling_robust_z_2d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    out = np.zeros_like(values, dtype=np.float64)
    for col_idx in range(values.shape[1]):
        out[:, col_idx] = _rolling_robust_z_1d(values[:, col_idx], window)
    return out


def _cross_sectional_robust_z_2d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros_like(values, dtype=np.float64)
    for row_idx in range(values.shape[0]):
        row = values[row_idx]
        finite_mask = np.isfinite(row)
        if not bool(finite_mask.any()):
            continue
        finite_row = row[finite_mask]
        median = float(np.median(finite_row))
        mad = float(np.median(np.abs(finite_row - median)) * 1.4826)
        if mad > _ROBUST_Z_EPS:
            out[row_idx, finite_mask] = np.clip((finite_row - median) / mad, -_ROBUST_Z_CLIP, _ROBUST_Z_CLIP)
    return out


def _compute_score_pct_variant_hist(
    events: pd.DataFrame,
    window_bars: int = 2160,
) -> NDArray[np.float32]:
    """Causal percentile of raw_score within this variant's window_bars event history.

    For each event, fraction of prior same-variant events (within window_bars)
    with strictly lower raw_score. Defaults to 0.5 if history < 5 events.
    """
    result = np.full(len(events), 0.5, dtype=np.float32)
    if "family" not in events.columns or "variant" not in events.columns:
        return result
    score_col = "raw_score" if "raw_score" in events.columns else "score"
    if score_col not in events.columns:
        return result

    variant_key = (events["family"].astype(str) + ":" + events["variant"].astype(str)).values
    entry_idx_arr = events["entry_idx"].values
    score_arr = events[score_col].fillna(0.0).values.astype(np.float64)

    for vk in np.unique(variant_key):
        v_mask = variant_key == vk
        v_pos = np.where(v_mask)[0]
        sort_order = np.argsort(entry_idx_arr[v_pos], kind="stable")
        v_pos_sorted = v_pos[sort_order]
        v_entry_sorted = entry_idx_arr[v_pos_sorted]
        v_score_sorted = score_arr[v_pos_sorted]

        # Use np.searchsorted to find the left boundary index O(log N)
        left_idxs = np.searchsorted(v_entry_sorted, v_entry_sorted - window_bars, side="right")
        for i in range(len(v_pos_sorted)):
            left_idx = left_idxs[i]
            if i - left_idx >= 5:
                hist_scores = v_score_sorted[left_idx:i]
                result[v_pos_sorted[i]] = float(np.mean(hist_scores < v_score_sorted[i]))

    return result


def _impute_feature_matrix(
    x_mat: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], int]:
    imputed = x_mat.copy()
    if imputed.shape[0] == 0:
        return imputed, 0
    valid_rows = np.flatnonzero(valid_mask)
    if valid_rows.size == 0:
        return imputed, 0
    replacements = 0
    for col_idx in range(imputed.shape[1]):
        column = imputed[:, col_idx].astype(np.float64, copy=False)
        valid_values = column[valid_rows]
        finite_valid = valid_values[np.isfinite(valid_values)]
        fill_value = float(np.median(finite_valid)) if finite_valid.size > 0 else 0.0
        missing_rows = valid_rows[~np.isfinite(valid_values)]
        if missing_rows.size > 0:
            imputed[missing_rows, col_idx] = np.float32(fill_value)
            replacements += int(missing_rows.size)
    return imputed, replacements


def _target_cost_hurdle_bps(events: pd.DataFrame) -> NDArray[np.float32]:
    """Return per-event cost+hurdle already embedded in edge_after_hurdle_bps."""
    size = len(events)
    cost_hurdle = np.zeros(size, dtype=np.float32)
    if "ex_ante_cost_bps" in events.columns:
        cost = pd.to_numeric(events["ex_ante_cost_bps"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        np.nan_to_num(cost, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cost_hurdle = cost_hurdle + cost
    if "hurdle_bps" in events.columns:
        hurdle = pd.to_numeric(events["hurdle_bps"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        np.nan_to_num(hurdle, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cost_hurdle = cost_hurdle + hurdle
    return cost_hurdle


def _resolve_gross_target_bps(
    events: pd.DataFrame,
    *,
    allow_label_free: bool,
) -> NDArray[np.float32]:
    for column in ("gross_return_bps", "gross_event_bps", "gross_fwd_bps"):
        if column in events.columns:
            values = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            return np.asarray(values, dtype=np.float32)
    if allow_label_free:
        return np.zeros((len(events),), dtype=np.float32)
    for legacy_column in ("net_event_bps", "edge_after_hurdle_bps", "net_return_bps"):
        if legacy_column in events.columns:
            _logger.warning("[DATASET] legacy_target_fallback=%s", legacy_column)
            values = pd.to_numeric(events[legacy_column], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            return np.asarray(values, dtype=np.float32)
    raise ValueError("missing gross target columns: ['gross_return_bps', 'gross_event_bps', 'gross_fwd_bps']")


def _resolve_gross_target_r(
    events: pd.DataFrame,
    *,
    gross_bps: NDArray[np.float32],
    risk_unit: NDArray[np.float32],
    allow_label_free: bool,
) -> NDArray[np.float32]:
    if "gross_return_r" in events.columns:
        values = pd.to_numeric(events["gross_return_r"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        return np.asarray(values, dtype=np.float32)
    if allow_label_free and not any(
        column in events.columns for column in ("gross_return_bps", "gross_event_bps", "gross_fwd_bps")
    ):
        return np.zeros((len(events),), dtype=np.float32)
    return (gross_bps / np.maximum(risk_unit, 1e-6)).astype(np.float32, copy=False)


def _resolve_effective_exit_idx(
    events: pd.DataFrame,
    *,
    split_end: int,
) -> NDArray[np.int32]:
    if "exit_idx" in events.columns:
        exit_idx = pd.to_numeric(events["exit_idx"], errors="coerce")
        if exit_idx.notna().all():
            return np.asarray(exit_idx.to_numpy(dtype=np.int32, copy=False), dtype=np.int32)
    entry_idx = np.asarray(
        pd.to_numeric(events["entry_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False),
        dtype=np.int32,
    )
    if "expected_holding_bars" in events.columns:
        holding = np.asarray(
            pd.to_numeric(events["expected_holding_bars"], errors="coerce")
            .fillna(1)
            .clip(lower=1)
            .to_numpy(dtype=np.int32, copy=False),
            dtype=np.int32,
        )
    else:
        holding = np.ones((len(events),), dtype=np.int32)
    inferred = entry_idx + np.maximum(holding - 1, 0)
    return np.clip(inferred, entry_idx, max(split_end - 1, 0)).astype(np.int32, copy=False)


@njit(cache=True)  # type: ignore
def _compute_uniqueness_weights_numba(
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
    inv_active: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_events = starts.shape[0]
    weights = np.zeros(n_events, dtype=np.float64)
    for idx in range(n_events):
        start = starts[idx]
        end = ends[idx]
        segment = inv_active[start : end + 1]
        weights[idx] = segment.mean() if segment.size > 0 else 0.0
    return weights


def _event_uniqueness_weights(
    *,
    entry_idx: NDArray[np.int32],
    exit_idx: NDArray[np.int32],
    split_start: int,
    split_end: int,
) -> tuple[NDArray[np.float32], float]:
    """Return overlap-aware uniqueness weights and effective sample size."""
    n_events = int(entry_idx.shape[0])
    if n_events == 0 or split_end <= split_start:
        return np.zeros((n_events,), dtype=np.float32), 0.0

    horizon = split_end - split_start
    concurrency = np.zeros(horizon + 1, dtype=np.int32)
    starts = np.clip(entry_idx.astype(np.int64) - split_start, 0, max(horizon - 1, 0))
    ends = np.clip(exit_idx.astype(np.int64) - split_start, 0, max(horizon - 1, 0))
    for start, end in zip(starts, ends, strict=True):
        concurrency[int(start)] += 1
        concurrency[int(end) + 1] -= 1
    active = np.cumsum(concurrency[:-1], dtype=np.int32)
    inv_active = np.zeros(active.shape[0], dtype=np.float64)
    positive_mask = active > 0
    inv_active[positive_mask] = 1.0 / active[positive_mask]

    # Numba-compiled 가중치 연산 호출
    weights = _compute_uniqueness_weights_numba(starts, ends, inv_active)

    positive = weights > 0.0
    if positive.any():
        weights[positive] = weights[positive] / float(np.mean(weights[positive]))
    ess_den = float(np.sum(np.square(weights)))
    ess = float((np.sum(weights) ** 2) / ess_den) if ess_den > 0.0 else 0.0
    return weights.astype(np.float32, copy=False), ess


def _effective_sample_size_from_weights(weights: NDArray[np.float32]) -> float:
    weights64 = np.asarray(weights, dtype=np.float64)
    positive = weights64[np.isfinite(weights64) & (weights64 > 0.0)]
    if positive.size == 0:
        return 0.0
    denom = float(np.sum(np.square(positive)))
    if denom <= 0.0:
        return 0.0
    total = float(np.sum(positive))
    return (total * total) / denom


def _weighted_tstat(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(mask.any()):
        return 0.0, 0.0
    x = values[mask]
    w = weights[mask]
    w_sum = float(np.sum(w))
    w_sq_sum = float(np.sum(np.square(w)))
    if w_sum <= 0.0 or w_sq_sum <= 0.0:
        return 0.0, 0.0
    n_eff = (w_sum * w_sum) / w_sq_sum
    if n_eff <= 1.0:
        return 0.0, n_eff
    mean = float(np.sum(w * x) / w_sum)
    centered = x - mean
    var_num = float(np.sum(w * centered * centered))
    var = var_num / max(w_sum, 1e-12)
    if not np.isfinite(var) or var <= 0.0:
        return 0.0, n_eff
    se = math.sqrt(var / n_eff)
    if se <= 0.0 or not np.isfinite(se):
        return 0.0, n_eff
    return mean / se, n_eff


@njit(cache=True)  # type: ignore
def _compute_bootstrap_means_numba(
    x: NDArray[np.float64],
    w: NDArray[np.float64],
    start_idxs: NDArray[np.int64],
    block: int,
) -> NDArray[np.float64]:
    n_obs = x.shape[0]
    bootstrap_n = start_idxs.shape[0]
    num_blocks = start_idxs.shape[1]
    boot_means = np.zeros(bootstrap_n, dtype=np.float64)

    for boot_idx in range(bootstrap_n):
        sx = np.zeros(n_obs, dtype=np.float64)
        sw = np.zeros(n_obs, dtype=np.float64)
        idx = 0
        for b in range(num_blocks):
            start = start_idxs[boot_idx, b]
            length = block
            if start + length > n_obs:
                length = n_obs - start
            if idx + length > n_obs:
                length = n_obs - idx
            if length <= 0:
                break

            sx[idx : idx + length] = x[start : start + length]
            sw[idx : idx + length] = w[start : start + length]
            idx += length

        w_sum = sw.sum()
        if w_sum > 0.0:
            boot_means[boot_idx] = (sx * sw).sum() / w_sum
        else:
            boot_means[boot_idx] = 0.0

    return boot_means


def _block_bootstrap_tstat(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    entry_idx: NDArray[np.int32],
    holding_bars: NDArray[np.int32],
    *,
    bootstrap_n: int,
    seed: int,
) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(mask.sum()) < 2:
        return 0.0
    x = values[mask]
    w = weights[mask]
    order = np.argsort(entry_idx[mask], kind="stable")
    x = x[order]
    w = w[order]
    holding = holding_bars[mask][order]
    n_obs = x.shape[0]
    block = max(1, int(np.rint(np.median(holding))))

    num_blocks = (n_obs + block - 1) // block
    rng = np.random.default_rng(seed)

    # 2D 난수 인덱스 사전 일괄 샘플링
    start_idxs = rng.integers(0, n_obs, size=(bootstrap_n, num_blocks))

    # Numba 컴파일 함수 실행
    boot_means = _compute_bootstrap_means_numba(x, w, start_idxs, block)

    boot_std = float(np.std(boot_means, ddof=1)) if bootstrap_n > 1 else 0.0
    if not np.isfinite(boot_std) or boot_std <= 0.0:
        return 0.0
    return float(np.average(x, weights=w)) / boot_std


def _variant_proof_tstat(
    *,
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    entry_idx: NDArray[np.int32],
    holding_bars: NDArray[np.int32],
    method: str,
    bootstrap_n: int,
    seed: int,
) -> tuple[float, float]:
    if method == "concurrency_t":
        return _weighted_tstat(values, weights)
    if method == "block_bootstrap":
        tstat = _block_bootstrap_tstat(
            values,
            weights,
            entry_idx,
            holding_bars,
            bootstrap_n=bootstrap_n,
            seed=seed,
        )
        _, n_eff = _weighted_tstat(values, weights)
        return tstat, n_eff
    mean = 0.0
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if bool(mask.any()):
        mean = float(np.average(values[mask], weights=weights[mask]))
    _, n_eff = _weighted_tstat(values, weights)
    return mean, n_eff


def _stable_variant_key(family: str, variant: str) -> str:
    return f"{family}:{variant}"


def _ordered_identity_feature_names(
    labeled_events: pd.DataFrame,
    *,
    cfg: CandidateStrategyConfig,
) -> tuple[str, ...]:
    if not cfg.candidate_identity_features_enabled:
        return ()

    names: list[str] = []
    if labeled_events.empty:
        return ("side_is_long", "side_is_short")

    if "family" in labeled_events.columns:
        families = sorted({str(value) for value in labeled_events["family"].dropna().astype(str) if value})
        names.extend(f"family={family}" for family in families)
    if "archetype" in labeled_events.columns:
        archetypes = sorted({str(value) for value in labeled_events["archetype"].dropna().astype(str) if value})
        names.extend(f"archetype={archetype}" for archetype in archetypes)
    names.extend(("side_is_long", "side_is_short"))
    return tuple(names)


def _btc_symbol_index(symbols: tuple[str, ...]) -> int:
    for idx, symbol in enumerate(symbols):
        if "BTC" in symbol.upper():
            return idx
    return 0


def _market_state_feature_names(cfg: CandidateStrategyConfig) -> tuple[str, ...]:
    if not cfg.market_state_features_enabled:
        return ()
    return (
        "btc_ret_1",
        "btc_ret_5",
        "btc_trend_20_100",
        "mkt_vol_z120",
        "mkt_dispersion_z120",
        "market_breadth_20",
        "symbol_ret_rank_20",
        "symbol_vol_z120",
        "funding_cross_section_z",
        "cost_to_vol_bps",
    )


def _signal_context_feature_names(cfg: CandidateStrategyConfig) -> tuple[str, ...]:
    if not bool(getattr(cfg, "signal_context_features_enabled", True)):
        return ()
    return (
        "overlay_mult_entry",
        "crisis_active_entry",
        "funding_side_alignment",
        "score_pct_variant_hist_90d",
        "archetype_regime_match",
        "n_same_dir_variants_log",
    )


def _universe_feature_names(cfg: CandidateStrategyConfig) -> tuple[str, ...]:
    if not cfg.static_universe_features_enabled:
        return ()
    return (
        "universe_vol_30d",
        "universe_friction_score",
        "universe_alpha_capacity_score",
        "universe_diversification_score",
        "universe_tradeable_score",
        "universe_cluster_id",
        "universe_beta_vs_market",
        "universe_cluster_size",
        "universe_anchor_cluster_member",
    )


def _base_feature_names(cfg: CandidateStrategyConfig) -> tuple[str, ...]:
    exclude_leaky = bool(getattr(cfg, "exclude_immediate_return_features", False))
    names = [
        "side",
        "raw_score",
        "score_z",
        "turnover_proxy",
        "expected_holding_bars",
        "min_holding_bars",
        "stop_atr_mult",
        "sl_thr_bps",
        "take_profit_atr_mult",
    ]
    if not exclude_leaky:
        names.append("sym_ret_1")
    names.extend(["sym_ret_5", "sym_vol_20", "sym_volume_z20"])
    if not exclude_leaky:
        names.append("mkt_ret_1")
    names.extend(["mkt_vol_20", "mkt_dispersion_20", "ex_ante_cost_bps", "funding_z20"])
    return tuple(names)


def fit_candidate_feature_schema(
    *,
    labeled_events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    split_start: int,
    split_end: int,
) -> CandidateFeatureSchema:
    """Fit a feature schema from the train split only."""
    mask = (labeled_events["entry_idx"] >= split_start) & (labeled_events["entry_idx"] < split_end)
    events = labeled_events.loc[mask].copy()
    base_feat_names = _base_feature_names(cfg)
    uni_feat_names = _universe_feature_names(cfg)
    mkt_feat_names = _market_state_feature_names(cfg)
    sig_feat_names = _signal_context_feature_names(cfg)
    id_feat_names = _ordered_identity_feature_names(events, cfg=cfg)
    return CandidateFeatureSchema(
        feature_names=base_feat_names + uni_feat_names + mkt_feat_names + sig_feat_names + id_feat_names,
        identity_categories=id_feat_names,
    )


def build_candidate_dataset(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    schema: CandidateFeatureSchema | None = None,
    split_start: int,
    split_end: int,
    require_label_within_split: bool = True,
    is_fit_split: bool = False,
) -> CandidateDataset:
    """Build model matrix for candidate gate and edge models.

    Fully vectorized implementation for high performance.
    """
    if split_end <= split_start:
        raise ValueError("split_end must be greater than split_start")

    active_schema = schema or fit_candidate_feature_schema(
        labeled_events=labeled_events,
        cfg=cfg,
        split_start=split_start,
        split_end=split_end,
    )
    base_feat_names = _base_feature_names(cfg)
    uni_feat_names = _universe_feature_names(cfg)
    sig_feat_names = _signal_context_feature_names(cfg)
    id_feat_names = active_schema.identity_categories
    feature_names = active_schema.feature_names
    exclude_leaky = bool(getattr(cfg, "exclude_immediate_return_features", False))

    mask = (labeled_events["entry_idx"] >= split_start) & (labeled_events["entry_idx"] < split_end)
    if require_label_within_split:
        if "exit_idx" not in labeled_events.columns:
            mask &= False
        else:
            exit_idx = pd.to_numeric(labeled_events["exit_idx"], errors="coerce")
            mask &= exit_idx.ge(0) & exit_idx.lt(split_end)
    events = labeled_events.loc[mask].copy()
    if not events.empty:
        sym_set = set(aligned.symbols)
        events = events[events["symbol"].isin(sym_set)].copy()

    if events.empty:
        return CandidateDataset(
            X=np.zeros((0, len(feature_names)), dtype=np.float32),
            y_gate=np.zeros((0,), dtype=np.int8),
            y_edge_bps=np.zeros((0,), dtype=np.float32),
            y_q10_bps=np.zeros((0,), dtype=np.float32),
            y_mfe_bps=np.zeros((0,), dtype=np.float32),
            gate_weight=np.zeros((0,), dtype=np.float32),
            edge_weight=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=np.int32),
            event_index=events,
            feature_names=feature_names,
            effective_sample_size=0.0,
            feature_schema_version=active_schema.version,
            y_return_r=np.zeros((0,), dtype=np.float32),
            y_return_bps=np.zeros((0,), dtype=np.float32),
            y_gross_return_bps=np.zeros((0,), dtype=np.float32),
            y_gross_return_r=np.zeros((0,), dtype=np.float32),
            y_mae_r=np.zeros((0,), dtype=np.float32),
            risk_unit_bps=np.zeros((0,), dtype=np.float32),
        )

    # Pre-calculate features
    close = aligned.close_2d
    volume = aligned.volume_2d
    funding = aligned.funding_2d
    t_len, _ = close.shape

    global _ALIGNED_FEATURE_CACHE
    aligned_id = id(aligned)
    if aligned_id not in _ALIGNED_FEATURE_CACHE:
        _ALIGNED_FEATURE_CACHE[aligned_id] = {"__aligned_ref__": aligned}
    cache = _ALIGNED_FEATURE_CACHE[aligned_id]
    if cache.get("__aligned_ref__") is not aligned:
        cache.clear()
        cache["__aligned_ref__"] = aligned

    if "sym_ret_1" not in cache:
        sym_ret_1 = (close[1:] / np.maximum(close[:-1], 1e-12)) - 1.0
        sym_ret_5 = (close[5:] / np.maximum(close[:-5], 1e-12)) - 1.0

        log_ret_2d = np.zeros_like(close)
        log_ret_2d[1:] = np.diff(np.log(np.maximum(close, 1e-12)), axis=0)
        sym_vol_20 = pd.DataFrame(log_ret_2d).rolling(20, min_periods=1).std(ddof=0).values

        sym_volume_z20 = _rolling_robust_z_2d(volume, window=20)

        f_df = pd.DataFrame(funding)
        funding_z20 = _rolling_robust_z_2d(funding, window=20)

        # 2. Market-wide technicals (T)
        mkt_ret_1 = np.nanmean(sym_ret_1, axis=1)
        # pad to T length
        mkt_ret_1_padded = np.zeros(t_len)
        mkt_ret_1_padded[1:] = mkt_ret_1

        mkt_vol_20 = np.nanmean(sym_vol_20, axis=1)

        ret20_2d = np.zeros_like(close)
        ret20_2d[20:] = (close[20:] / np.maximum(close[:-20], 1e-12)) - 1.0
        mkt_dispersion_20 = np.nanstd(ret20_2d, axis=1)
        market_breadth_20 = np.nanmean(ret20_2d > 0, axis=1)

        cache["sym_ret_1"] = sym_ret_1
        cache["sym_ret_5"] = sym_ret_5
        cache["sym_vol_20"] = sym_vol_20
        cache["sym_volume_z20"] = sym_volume_z20
        cache["funding_z20"] = funding_z20
        cache["mkt_ret_1_padded"] = mkt_ret_1_padded
        cache["mkt_vol_20"] = mkt_vol_20
        cache["mkt_dispersion_20"] = mkt_dispersion_20
        cache["market_breadth_20"] = market_breadth_20
        cache["ret20_2d"] = ret20_2d
    else:
        sym_ret_1 = cache["sym_ret_1"]
        sym_ret_5 = cache["sym_ret_5"]
        sym_vol_20 = cache["sym_vol_20"]
        sym_volume_z20 = cache["sym_volume_z20"]
        funding_z20 = cache["funding_z20"]
        mkt_ret_1_padded = cache["mkt_ret_1_padded"]
        mkt_vol_20 = cache["mkt_vol_20"]
        mkt_dispersion_20 = cache["mkt_dispersion_20"]
        market_breadth_20 = cache["market_breadth_20"]
        ret20_2d = cache["ret20_2d"]
        f_df = pd.DataFrame(funding)

    # 3. Market state features (if enabled)
    btc_ret_1_ser = np.zeros(t_len)
    btc_ret_5_ser = np.zeros(t_len)
    btc_trend_20_100_ser = np.zeros(t_len)
    mkt_vol_z120_ser = np.zeros(t_len)
    mkt_disp_z120_ser = np.zeros(t_len)
    symbol_ret_rank_20_2d = np.zeros_like(close)
    symbol_vol_z120_2d = np.zeros_like(close)
    funding_cs_z_2d = np.zeros_like(close)

    if cfg.market_state_features_enabled:
        if "btc_ret_1_ser" not in cache:
            btc_idx = _btc_symbol_index(aligned.symbols)
            btc_close = close[:, btc_idx]
            btc_ret_1_ser[1:] = (btc_close[1:] / np.maximum(btc_close[:-1], 1e-12)) - 1.0
            btc_ret_5_ser[5:] = (btc_close[5:] / np.maximum(btc_close[:-5], 1e-12)) - 1.0
            btc_ma20 = pd.Series(btc_close).rolling(20, min_periods=1).mean().values
            btc_ma100 = pd.Series(btc_close).rolling(100, min_periods=1).mean().values
            btc_trend_20_100_ser = (btc_ma20 >= btc_ma100).astype(float)

            mkt_vol_df = pd.Series(mkt_vol_20).fillna(0)
            mkt_vol_z120_ser = _rolling_robust_z_1d(mkt_vol_df.to_numpy(dtype=np.float64), window=120)

            mkt_disp_df = pd.Series(mkt_dispersion_20).fillna(0)
            mkt_disp_z120_ser = _rolling_robust_z_1d(mkt_disp_df.to_numpy(dtype=np.float64), window=120)

            symbol_ret_rank_20_2d[20:] = pd.DataFrame(ret20_2d[20:]).rank(axis=1, pct=True).values

            sv_df = pd.DataFrame(sym_vol_20).fillna(0)
            symbol_vol_z120_2d = _rolling_robust_z_2d(sv_df.to_numpy(dtype=np.float64), window=120)

            funding_cs_z_2d = _cross_sectional_robust_z_2d(f_df.fillna(0).to_numpy(dtype=np.float64))

            cache["btc_ret_1_ser"] = btc_ret_1_ser
            cache["btc_ret_5_ser"] = btc_ret_5_ser
            cache["btc_trend_20_100_ser"] = btc_trend_20_100_ser
            cache["mkt_vol_z120_ser"] = mkt_vol_z120_ser
            cache["mkt_disp_z120_ser"] = mkt_disp_z120_ser
            cache["symbol_ret_rank_20_2d"] = symbol_ret_rank_20_2d
            cache["symbol_vol_z120_2d"] = symbol_vol_z120_2d
            cache["funding_cs_z_2d"] = funding_cs_z_2d
        else:
            btc_ret_1_ser = cache["btc_ret_1_ser"]
            btc_ret_5_ser = cache["btc_ret_5_ser"]
            btc_trend_20_100_ser = cache["btc_trend_20_100_ser"]
            mkt_vol_z120_ser = cache["mkt_vol_z120_ser"]
            mkt_disp_z120_ser = cache["mkt_disp_z120_ser"]
            symbol_ret_rank_20_2d = cache["symbol_ret_rank_20_2d"]
            symbol_vol_z120_2d = cache["symbol_vol_z120_2d"]
            funding_cs_z_2d = cache["funding_cs_z_2d"]

    # 4. Identity Features
    id_matrix: NDArray[np.float32] | None = None
    if cfg.candidate_identity_features_enabled and len(id_feat_names) > 0:
        id_matrix = np.zeros((len(events), len(id_feat_names)), dtype=np.float32)
        if "family" in events.columns:
            for i, name in enumerate(id_feat_names):
                if name.startswith("family="):
                    fam = name.split("=", 1)[1]
                    id_matrix[:, i] = (events["family"].astype(str) == fam).astype(float)
        if "archetype" in events.columns:
            for i, name in enumerate(id_feat_names):
                if name.startswith("archetype="):
                    archetype = name.split("=", 1)[1]
                    id_matrix[:, i] = (events["archetype"].astype(str) == archetype).astype(float)
        long_idx = id_feat_names.index("side_is_long") if "side_is_long" in id_feat_names else -1
        short_idx = id_feat_names.index("side_is_short") if "side_is_short" in id_feat_names else -1
        if long_idx >= 0:
            id_matrix[:, long_idx] = (events["side"] > 0).astype(float)
        if short_idx >= 0:
            id_matrix[:, short_idx] = (events["side"] < 0).astype(float)

    # 5. Universe Features
    uni_matrix = np.zeros((len(events), len(uni_feat_names)), dtype=np.float32)
    sym_to_idx = {s: i for i, s in enumerate(aligned.symbols)}
    event_sym_idxs = events["symbol"].map(sym_to_idx).values
    if len(uni_feat_names) > 0 and aligned.vol_30d_1d is not None:
        uni_matrix[:, 0] = aligned.vol_30d_1d[event_sym_idxs]
    if len(uni_feat_names) > 1 and aligned.friction_score_1d is not None:
        uni_matrix[:, 1] = aligned.friction_score_1d[event_sym_idxs]
    if len(uni_feat_names) > 2 and aligned.alpha_capacity_score_1d is not None:
        uni_matrix[:, 2] = aligned.alpha_capacity_score_1d[event_sym_idxs]
    if len(uni_feat_names) > 3 and aligned.diversification_score_1d is not None:
        uni_matrix[:, 3] = aligned.diversification_score_1d[event_sym_idxs]
    if len(uni_feat_names) > 4 and aligned.tradeable_score_1d is not None:
        uni_matrix[:, 4] = aligned.tradeable_score_1d[event_sym_idxs]
    if len(uni_feat_names) > 5 and aligned.cluster_id_1d is not None:
        uni_matrix[:, 5] = aligned.cluster_id_1d[event_sym_idxs]
    if len(uni_feat_names) > 6 and aligned.beta_vs_market_1d is not None:
        uni_matrix[:, 6] = aligned.beta_vs_market_1d[event_sym_idxs]
    if len(uni_feat_names) > 7 and aligned.cluster_size_1d is not None:
        uni_matrix[:, 7] = aligned.cluster_size_1d[event_sym_idxs]
    if len(uni_feat_names) > 8 and aligned.anchor_cluster_1d is not None:
        uni_matrix[:, 8] = aligned.anchor_cluster_1d[event_sym_idxs]

    # 6. Assembly
    event_t = events["entry_idx"].values - 1
    valid_mask = event_t >= 20
    
    if "overlay_ctx" not in cache:
        cache["overlay_ctx"] = compute_risk_overlay(aligned=aligned)
    overlay_ctx = cache["overlay_ctx"]

    if "regime_ctx" not in cache:
        cache["regime_ctx"] = compute_market_regime_context(aligned=aligned)
    regime_ctx = cache["regime_ctx"]

    x_mat = np.zeros((len(events), len(feature_names)), dtype=np.float32)

    # Helper function to set feature values dynamically based on name index
    def _set_feat(name: str, values: NDArray[np.float32] | NDArray[np.float64]) -> None:
        if name in feature_names:
            idx = feature_names.index(name)
            x_mat[:, idx] = values

    cols_from_events = [
        "side",
        "raw_score",
        "score_z",
        "turnover_proxy",
        "expected_holding_bars",
        "min_holding_bars",
        "stop_atr_mult",
        "sl_thr_bps",
        "take_profit_atr_mult",
    ]
    for col in cols_from_events:
        if col in events.columns:
            _set_feat(col, events[col].fillna(0).values.astype(np.float32))
        elif col == "raw_score" and "score" in events.columns:
            _set_feat("raw_score", events["score"].fillna(0).values.astype(np.float32))

    if not exclude_leaky:
        idx_ret1 = feature_names.index("sym_ret_1")
        x_mat[valid_mask, idx_ret1] = sym_ret_1[
            event_t[valid_mask] - 1, event_sym_idxs[valid_mask]
        ]

    x_mat[valid_mask, feature_names.index("sym_ret_5")] = sym_ret_5[
        event_t[valid_mask] - 5, event_sym_idxs[valid_mask]
    ]
    x_mat[valid_mask, feature_names.index("sym_vol_20")] = sym_vol_20[
        event_t[valid_mask], event_sym_idxs[valid_mask]
    ]
    x_mat[valid_mask, feature_names.index("sym_volume_z20")] = sym_volume_z20[
        event_t[valid_mask], event_sym_idxs[valid_mask]
    ]

    if not exclude_leaky:
        idx_mkt_ret1 = feature_names.index("mkt_ret_1")
        x_mat[valid_mask, idx_mkt_ret1] = mkt_ret_1_padded[event_t[valid_mask]]

    x_mat[valid_mask, feature_names.index("mkt_vol_20")] = mkt_vol_20[event_t[valid_mask]]
    x_mat[valid_mask, feature_names.index("mkt_dispersion_20")] = mkt_dispersion_20[
        event_t[valid_mask]
    ]

    if "ex_ante_cost_bps" in events.columns:
        _set_feat("ex_ante_cost_bps", events["ex_ante_cost_bps"].fillna(0).values.astype(np.float32))
    elif aligned.execution_cost_bps_2d is not None:
        idx_cost = feature_names.index("ex_ante_cost_bps")
        x_mat[valid_mask, idx_cost] = aligned.execution_cost_bps_2d[
            event_t[valid_mask], event_sym_idxs[valid_mask]
        ]

    x_mat[valid_mask, feature_names.index("funding_z20")] = funding_z20[
        event_t[valid_mask], event_sym_idxs[valid_mask]
    ]

    curr = len(base_feat_names)
    x_mat[:, curr : curr + len(uni_feat_names)] = uni_matrix
    curr += len(uni_feat_names)

    if cfg.market_state_features_enabled:
        x_mat[valid_mask, curr] = btc_ret_1_ser[event_t[valid_mask]]
        x_mat[valid_mask, curr + 1] = btc_ret_5_ser[event_t[valid_mask]]
        x_mat[valid_mask, curr + 2] = btc_trend_20_100_ser[event_t[valid_mask]]
        x_mat[valid_mask, curr + 3] = mkt_vol_z120_ser[event_t[valid_mask]]
        x_mat[valid_mask, curr + 4] = mkt_disp_z120_ser[event_t[valid_mask]]
        x_mat[valid_mask, curr + 5] = market_breadth_20[event_t[valid_mask]]
        x_mat[valid_mask, curr + 6] = symbol_ret_rank_20_2d[event_t[valid_mask], event_sym_idxs[valid_mask]]
        x_mat[valid_mask, curr + 7] = symbol_vol_z120_2d[event_t[valid_mask], event_sym_idxs[valid_mask]]
        x_mat[valid_mask, curr + 8] = funding_cs_z_2d[event_t[valid_mask], event_sym_idxs[valid_mask]]
        vol_20_vals = x_mat[:, feature_names.index("sym_vol_20")]
        cost_bps_vals = x_mat[:, feature_names.index("ex_ante_cost_bps")]
        x_mat[:, curr + 9] = cost_bps_vals / np.maximum(vol_20_vals * 1e4, 1.0)
        curr += 10

    if sig_feat_names:
        win = int(getattr(cfg, "score_pct_variant_hist_window_bars", 2160))
        score_pct = _compute_score_pct_variant_hist(events, window_bars=win)

        side_arr = x_mat[:, feature_names.index("side")]
        funding_z20_arr = x_mat[:, feature_names.index("funding_z20")]
        fsa = np.tanh(funding_z20_arr * side_arr).astype(np.float32)

        regime_names_arr = np.asarray(regime_ctx.name_by_code, dtype=object)[regime_ctx.code_1d[event_t]]
        archetype_arr = (
            events.get("archetype", pd.Series("", index=events.index))
            .fillna("")
            .astype(str)
            .values
        )
        
        _archetypes = [
            "trend", "ts_mom", "mean_rev",
            "carry_rev", "flow_rev", "unwind",
            "beta_neut"
        ]
        _regimes = [
            "bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile",
            "transition", "crash"
        ]
        _arch_to_idx = {a: idx for idx, a in enumerate(_archetypes)}
        _reg_to_idx = {r: idx for idx, r in enumerate(_regimes)}
        
        _affinity_matrix = np.zeros((len(_archetypes) + 1, len(_regimes) + 1), dtype=np.float32)
        for (arch, reg), val in _ARCHETYPE_REGIME_AFFINITY.items():
            if arch in _arch_to_idx and reg in _reg_to_idx:
                _affinity_matrix[_arch_to_idx[arch], _reg_to_idx[reg]] = val
                
        arch_idxs = np.array([_arch_to_idx.get(a, len(_archetypes)) for a in archetype_arr], dtype=np.int32)
        reg_idxs = np.array([_reg_to_idx.get(r, len(_regimes)) for r in regime_names_arr], dtype=np.int32)
        arm = _affinity_matrix[arch_idxs, reg_idxs]

        if "entry_idx" in events.columns and "symbol" in events.columns and "side" in events.columns:
            side_sign = pd.Series(np.sign(events["side"].fillna(0)).astype(np.int32), index=events.index)
            counts = events.groupby(["entry_idx", "symbol", side_sign]).transform("size")
            n_same = np.log1p(np.maximum(counts.values - 1, 0)).astype(np.float32)
        else:
            n_same = np.zeros(len(events), dtype=np.float32)

        def _set_sig(name: str, values: NDArray[np.float32]) -> None:
            if name in feature_names:
                x_mat[:, feature_names.index(name)] = values

        _set_sig("overlay_mult_entry", overlay_ctx.overlay_mult_1d[event_t].astype(np.float32))
        x_mat[valid_mask, feature_names.index("crisis_active_entry")] = (
            overlay_ctx.crisis_active_1d[event_t[valid_mask]].astype(np.float32)
        )
        _set_sig("funding_side_alignment", fsa)
        _set_sig("score_pct_variant_hist_90d", score_pct)
        _set_sig("archetype_regime_match", arm)
        _set_sig("n_same_dir_variants_log", n_same)
        curr += len(sig_feat_names)

    if id_matrix is not None:
        x_mat[:, curr : curr + id_matrix.shape[1]] = id_matrix

    x_mat, n_imputed = _impute_feature_matrix(x_mat, valid_mask)
    finite_mask = np.all(np.isfinite(x_mat), axis=1) & valid_mask
    x_final = x_mat[finite_mask]
    kept_events = events[finite_mask].copy()
    groups_final = event_t[finite_mask].astype(np.int32)
    event_t_kept = event_t[finite_mask].astype(np.int32, copy=False)
    if kept_events.shape[0] > 0:
        kept_events["overlay_mult"] = overlay_ctx.overlay_mult_1d[event_t_kept]
        kept_events["crisis_active"] = overlay_ctx.crisis_active_1d[event_t_kept]
        kept_events["entry_regime_code"] = regime_ctx.code_1d[event_t_kept]
        kept_events["entry_regime"] = np.asarray(
            [regime_ctx.name_by_code[int(code)] for code in regime_ctx.code_1d[event_t_kept]],
            dtype=object,
        )
    if n_imputed > 0:
        _logger.debug("[DATASET] imputed_missing_values=%d valid_events=%d", n_imputed, int(valid_mask.sum()))
    dropped_events = int(valid_mask.sum()) - int(finite_mask.sum())
    if dropped_events > 0:
        _logger.debug("[DATASET] dropped_invalid_events=%d valid_events=%d", dropped_events, int(valid_mask.sum()))

    if x_final.shape[0] == 0:
        return CandidateDataset(
            X=np.zeros((0, len(feature_names)), dtype=np.float32),
            y_gate=np.zeros((0,), dtype=np.int8),
            y_edge_bps=np.zeros((0,), dtype=np.float32),
            y_q10_bps=np.zeros((0,), dtype=np.float32),
            y_mfe_bps=np.zeros((0,), dtype=np.float32),
            gate_weight=np.zeros((0,), dtype=np.float32),
            edge_weight=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=np.int32),
            event_index=kept_events,
            feature_names=feature_names,
            effective_sample_size=0.0,
            feature_schema_version=active_schema.version,
            y_return_r=np.zeros((0,), dtype=np.float32),
            y_return_bps=np.zeros((0,), dtype=np.float32),
            y_gross_return_bps=np.zeros((0,), dtype=np.float32),
            y_gross_return_r=np.zeros((0,), dtype=np.float32),
            y_mae_r=np.zeros((0,), dtype=np.float32),
            risk_unit_bps=np.zeros((0,), dtype=np.float32),
        )

    gate_label_col = cfg.gate_label_column
    if gate_label_col not in kept_events.columns and require_label_within_split:
        raise ValueError(f"missing configured gate label column: {gate_label_col}")
    if gate_label_col in kept_events.columns:
        y_gate = kept_events[gate_label_col].to_numpy(dtype=np.int8, copy=False)
    else:
        y_gate = np.zeros((kept_events.shape[0],), dtype=np.int8)
    min_risk_unit = np.float32(getattr(cfg, "min_risk_unit_bps", 25.0))
    if "risk_unit_bps" in kept_events.columns:
        risk_unit = kept_events["risk_unit_bps"].to_numpy(dtype=np.float32, copy=False)
    elif "sl_thr_bps" in kept_events.columns:
        risk_unit = kept_events["sl_thr_bps"].to_numpy(dtype=np.float32, copy=False)
    else:
        risk_unit = np.full((kept_events.shape[0],), min_risk_unit, dtype=np.float32)
    risk_unit = np.maximum(risk_unit, min_risk_unit)
    y_gross_return_bps = _resolve_gross_target_bps(
        kept_events,
        allow_label_free=not require_label_within_split,
    )
    y_gross_return_r = _resolve_gross_target_r(
        kept_events,
        gross_bps=y_gross_return_bps,
        risk_unit=risk_unit,
        allow_label_free=not require_label_within_split,
    )
    y_return_bps = y_gross_return_bps
    y_return_r = y_gross_return_r
    if "mae_r" in kept_events.columns:
        y_mae_r = kept_events["mae_r"].to_numpy(dtype=np.float32, copy=False)
    elif "mae_bps" in kept_events.columns:
        mae_raw = kept_events["mae_bps"].to_numpy(dtype=np.float32, copy=False)
        y_mae_r = (mae_raw / np.maximum(risk_unit, 1e-6)).astype(np.float32, copy=False)
    else:
        y_mae_r = np.zeros((kept_events.shape[0],), dtype=np.float32)
    legacy_return_bps = y_gross_return_bps.astype(np.float32, copy=False)
    uniqueness_weight, effective_sample_size = _event_uniqueness_weights(
        entry_idx=kept_events["entry_idx"].to_numpy(dtype=np.int32, copy=False),
        exit_idx=_resolve_effective_exit_idx(kept_events, split_end=split_end),
        split_start=split_start,
        split_end=split_end,
    )

    # Layer 0: Signal Pre-Qualification — fit split only
    # Variants with IS mean_edge < 0 or insufficient obs have edge_weight zeroed out.
    min_obs_prequalify = int(getattr(cfg, "signal_prequalify_min_obs", 0))
    if is_fit_split and min_obs_prequalify > 0:
        method = str(getattr(cfg, "signal_prequalify_method", "mean"))
        min_tstat = float(getattr(cfg, "signal_prequalify_min_tstat", 0.0))
        bootstrap_n = int(getattr(cfg, "signal_prequalify_bootstrap_n", 1000))
        seed = int(getattr(cfg, "seed", 42))
        edge_col = "edge_after_hurdle_bps" if "edge_after_hurdle_bps" in kept_events.columns else None
        if (
            edge_col is not None
            and "family" in kept_events.columns
            and "variant" in kept_events.columns
        ):
            variant_keys = (
                kept_events["family"].astype(str) + ":" + kept_events["variant"].astype(str)
            ).to_numpy(dtype=object)
            edge_values = pd.to_numeric(kept_events[edge_col], errors="coerce").to_numpy(
                dtype=np.float64,
                copy=False,
            )
            holding_bars = pd.to_numeric(
                kept_events.get("expected_holding_bars", pd.Series(1, index=kept_events.index)),
                errors="coerce",
            ).fillna(1).clip(lower=1).to_numpy(dtype=np.int32, copy=False)
            disq_arr = np.zeros(kept_events.shape[0], dtype=bool)
            for key in pd.unique(variant_keys):
                key_mask = variant_keys == key
                obs = int(np.isfinite(edge_values[key_mask]).sum())
                if obs < min_obs_prequalify:
                    disq_arr[key_mask] = True
                    continue
                finite_variant = np.isfinite(edge_values[key_mask]) & (uniqueness_weight[key_mask] > 0.0)
                if not bool(finite_variant.any()):
                    disq_arr[key_mask] = True
                    continue
                mean_edge = float(
                    np.average(
                        edge_values[key_mask][finite_variant],
                        weights=uniqueness_weight[key_mask][finite_variant],
                    )
                )
                if method == "mean":
                    disqualify = mean_edge <= 0.0
                else:
                    tstat, _ = _variant_proof_tstat(
                        values=edge_values[key_mask],
                        weights=uniqueness_weight[key_mask].astype(np.float64, copy=False),
                        entry_idx=kept_events.loc[key_mask, "entry_idx"].to_numpy(dtype=np.int32, copy=False),
                        holding_bars=holding_bars[key_mask],
                        method=method,
                        bootstrap_n=bootstrap_n,
                        seed=seed + int(np.sum(key_mask)),
                    )
                    disqualify = mean_edge <= 0.0 or tstat < min_tstat
                if disqualify:
                    disq_arr[key_mask] = True
            uniqueness_weight[disq_arr] = 0.0
            n_disqualified = int(disq_arr.sum())
            if n_disqualified > 0:
                _logger.debug(
                    "[SIGNAL_PREQUALIFY] method=%s disqualified=%d/%d events",
                    method,
                    n_disqualified,
                    len(kept_events),
                )
                for _dv in sorted(set(variant_keys[disq_arr]))[:10]:
                    _dv_mask = variant_keys == _dv
                    _dv_obs = int(np.isfinite(edge_values[_dv_mask]).sum())
                    _dv_mean = float(np.nanmean(edge_values[_dv_mask])) if _dv_obs > 0 else float("nan")
                    _logger.debug(
                        "[SIGNAL_PREQUALIFY][DISQ] variant=%-45s obs=%d  is_mean_edge=%.2f bps",
                        _dv,
                        _dv_obs,
                        _dv_mean,
                    )
            effective_sample_size = _effective_sample_size_from_weights(uniqueness_weight)

    return CandidateDataset(
        X=x_final,
        y_gate=y_gate,
        y_edge_bps=legacy_return_bps,
        y_q10_bps=legacy_return_bps,
        y_mfe_bps=legacy_return_bps,
        gate_weight=uniqueness_weight,
        edge_weight=uniqueness_weight.copy(),
        groups=groups_final,
        event_index=kept_events.reset_index(drop=True),
        feature_names=feature_names,
        effective_sample_size=effective_sample_size,
        feature_schema_version=active_schema.version,
        y_return_r=y_return_r.astype(np.float32, copy=False),
        y_return_bps=y_return_bps.astype(np.float32, copy=False),
        y_gross_return_bps=y_gross_return_bps.astype(np.float32, copy=False),
        y_gross_return_r=y_gross_return_r.astype(np.float32, copy=False),
        y_mae_r=y_mae_r.astype(np.float32, copy=False),
        risk_unit_bps=risk_unit.astype(np.float32, copy=False),
    )
