from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig


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
    feature_schema_version: str = "candidate_v4"


def _find_symbol_index(symbols: tuple[str, ...], symbol: str) -> int:
    for idx, value in enumerate(symbols):
        if value == symbol:
            return idx
    raise KeyError(f"unknown symbol: {symbol}")


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

    weights = np.zeros((n_events,), dtype=np.float64)
    for idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        segment = inv_active[int(start) : int(end) + 1]
        weights[idx] = float(segment.mean()) if segment.size > 0 else 0.0

    positive = weights > 0.0
    if positive.any():
        weights[positive] = weights[positive] / float(np.mean(weights[positive]))
    ess_den = float(np.sum(np.square(weights)))
    ess = float((np.sum(weights) ** 2) / ess_den) if ess_den > 0.0 else 0.0
    return weights.astype(np.float32, copy=False), ess


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
    if {"family", "variant"}.issubset(labeled_events.columns):
        variants = sorted(
            {
                _stable_variant_key(str(row.family), str(row.variant))
                for row in labeled_events.loc[:, ["family", "variant"]].itertuples(index=False)
                if str(row.family) and str(row.variant)
            }
        )
        names.extend(f"variant={variant}" for variant in variants)
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


def _universe_feature_names() -> tuple[str, ...]:
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


def build_candidate_dataset(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    split_start: int,
    split_end: int,
    require_label_within_split: bool = True,
) -> CandidateDataset:
    """Build model matrix for candidate gate and edge models.

    Fully vectorized implementation for high performance.
    """
    if split_end <= split_start:
        raise ValueError("split_end must be greater than split_start")

    id_feat_names = _ordered_identity_feature_names(labeled_events, cfg=cfg)
    mkt_feat_names = _market_state_feature_names(cfg)
    uni_feat_names = _universe_feature_names()

    exclude_leaky = bool(getattr(cfg, "exclude_immediate_return_features", False))
    base_feat_names_list = [
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
        base_feat_names_list.append("sym_ret_1")
    base_feat_names_list.extend([
        "sym_ret_5",
        "sym_vol_20",
        "sym_volume_z20",
    ])
    if not exclude_leaky:
        base_feat_names_list.append("mkt_ret_1")
    base_feat_names_list.extend([
        "mkt_vol_20",
        "mkt_dispersion_20",
        "ex_ante_cost_bps",
        "funding_z20",
    ])
    base_feat_names = tuple(base_feat_names_list)
    feature_names = base_feat_names + uni_feat_names + mkt_feat_names + id_feat_names

    mask = (labeled_events["entry_idx"] >= split_start) & (labeled_events["entry_idx"] < split_end)
    if require_label_within_split:
        if "exit_idx" not in labeled_events.columns:
            mask &= False
        else:
            exit_idx = pd.to_numeric(labeled_events["exit_idx"], errors="coerce")
            mask &= exit_idx.ge(0) & exit_idx.lt(split_end)
    events = labeled_events.loc[mask].copy()
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
        )

    # Pre-calculate features
    close = aligned.close_2d
    volume = aligned.volume_2d
    funding = aligned.funding_2d
    t_len, _ = close.shape

    # 1. Base technicals (T x N)
    sym_ret_1 = (close[1:] / np.maximum(close[:-1], 1e-12)) - 1.0
    sym_ret_5 = (close[5:] / np.maximum(close[:-5], 1e-12)) - 1.0

    log_ret_2d = np.zeros_like(close)
    log_ret_2d[1:] = np.diff(np.log(np.maximum(close, 1e-12)), axis=0)
    sym_vol_20 = pd.DataFrame(log_ret_2d).rolling(20, min_periods=1).std(ddof=0).values

    vol_df = pd.DataFrame(volume)
    vol_mean_20 = vol_df.rolling(20, min_periods=1).mean().values
    vol_std_20 = vol_df.rolling(20, min_periods=1).std(ddof=0).values
    sym_volume_z20 = np.zeros_like(volume)
    z_mask = vol_std_20 > 0
    sym_volume_z20[z_mask] = (volume[z_mask] - vol_mean_20[z_mask]) / vol_std_20[z_mask]

    f_df = pd.DataFrame(funding)
    funding_mean_20 = f_df.rolling(20, min_periods=1).mean().values
    funding_std_20 = f_df.rolling(20, min_periods=1).std(ddof=0).values
    funding_z20 = np.zeros_like(funding)
    fz_mask = funding_std_20 > 0
    funding_z20[fz_mask] = (funding[fz_mask] - funding_mean_20[fz_mask]) / funding_std_20[fz_mask]

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
        btc_idx = _btc_symbol_index(aligned.symbols)
        btc_close = close[:, btc_idx]
        btc_ret_1_ser[1:] = (btc_close[1:] / np.maximum(btc_close[:-1], 1e-12)) - 1.0
        btc_ret_5_ser[5:] = (btc_close[5:] / np.maximum(btc_close[:-5], 1e-12)) - 1.0
        btc_ma20 = pd.Series(btc_close).rolling(20, min_periods=1).mean().values
        btc_ma100 = pd.Series(btc_close).rolling(100, min_periods=1).mean().values
        btc_trend_20_100_ser = (btc_ma20 >= btc_ma100).astype(float)

        mkt_vol_df = pd.Series(mkt_vol_20).fillna(0)
        mkt_vol_rolling = mkt_vol_df.rolling(120, min_periods=1)
        mkt_vol_z120_ser = ((mkt_vol_df - mkt_vol_rolling.mean()) / mkt_vol_rolling.std(ddof=0)).fillna(0).values

        mkt_disp_df = pd.Series(mkt_dispersion_20).fillna(0)
        mkt_disp_rolling = mkt_disp_df.rolling(120, min_periods=1)
        mkt_disp_z120_ser = ((mkt_disp_df - mkt_disp_rolling.mean()) / mkt_disp_rolling.std(ddof=0)).fillna(0).values

        symbol_ret_rank_20_2d[20:] = pd.DataFrame(ret20_2d[20:]).rank(axis=1, pct=True).values

        sv_df = pd.DataFrame(sym_vol_20).fillna(0)
        sv_rolling = sv_df.rolling(120, min_periods=1)
        symbol_vol_z120_2d = ((sv_df - sv_rolling.mean()) / sv_rolling.std(ddof=0)).fillna(0).values

        funding_cs_z_2d = (
            (f_df.sub(f_df.mean(axis=1), axis=0)).div(f_df.std(axis=1, ddof=0), axis=0)
        ).fillna(0).values

    # 4. Identity Features
    id_matrix: NDArray[np.float32] | None = None
    if cfg.candidate_identity_features_enabled:
        id_matrix = np.zeros((len(events), len(id_feat_names)), dtype=np.float32)
        if "family" in events.columns:
            for i, name in enumerate(id_feat_names):
                if name.startswith("family="):
                    fam = name.split("=", 1)[1]
                    id_matrix[:, i] = (events["family"].astype(str) == fam).astype(float)
        if "family" in events.columns and "variant" in events.columns:
            v_keys = events.apply(lambda r: _stable_variant_key(str(r.family), str(r.variant)), axis=1)
            for i, name in enumerate(id_feat_names):
                if name.startswith("variant="):
                    var = name.split("=", 1)[1]
                    id_matrix[:, i] = (v_keys == var).astype(float)
        long_idx = id_feat_names.index("side_is_long")
        short_idx = id_feat_names.index("side_is_short")
        id_matrix[:, long_idx] = (events["side"] > 0).astype(float)
        id_matrix[:, short_idx] = (events["side"] < 0).astype(float)

    # 5. Universe Features
    uni_matrix = np.zeros((len(events), len(uni_feat_names)), dtype=np.float32)
    sym_to_idx = {s: i for i, s in enumerate(aligned.symbols)}
    event_sym_idxs = events["symbol"].map(sym_to_idx).values
    if aligned.vol_30d_1d is not None:
        uni_matrix[:, 0] = aligned.vol_30d_1d[event_sym_idxs]
    if aligned.friction_score_1d is not None:
        uni_matrix[:, 1] = aligned.friction_score_1d[event_sym_idxs]
    if aligned.alpha_capacity_score_1d is not None:
        uni_matrix[:, 2] = aligned.alpha_capacity_score_1d[event_sym_idxs]
    if aligned.diversification_score_1d is not None:
        uni_matrix[:, 3] = aligned.diversification_score_1d[event_sym_idxs]
    if aligned.tradeable_score_1d is not None:
        uni_matrix[:, 4] = aligned.tradeable_score_1d[event_sym_idxs]
    if aligned.cluster_id_1d is not None:
        uni_matrix[:, 5] = aligned.cluster_id_1d[event_sym_idxs]
    if aligned.beta_vs_market_1d is not None:
        uni_matrix[:, 6] = aligned.beta_vs_market_1d[event_sym_idxs]
    if aligned.cluster_size_1d is not None:
        uni_matrix[:, 7] = aligned.cluster_size_1d[event_sym_idxs]
    if aligned.anchor_cluster_1d is not None:
        uni_matrix[:, 8] = aligned.anchor_cluster_1d[event_sym_idxs]

    # 6. Assembly
    event_t = events["entry_idx"].values - 1
    valid_mask = event_t >= 20

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

    if id_matrix is not None:
        x_mat[:, curr : curr + id_matrix.shape[1]] = id_matrix

    finite_mask = np.all(np.isfinite(x_mat), axis=1) & valid_mask
    x_final = x_mat[finite_mask]
    kept_events = events[finite_mask].copy()
    groups_final = event_t[finite_mask].astype(np.int32)

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
        )

    gate_label_col = cfg.gate_label_column
    if gate_label_col not in kept_events.columns:
        raise ValueError(f"missing configured gate label column: {gate_label_col}")
    y_gate = kept_events[gate_label_col].to_numpy(dtype=np.int8, copy=False)
    y_edge = kept_events["edge_after_hurdle_bps"].to_numpy(dtype=np.float32, copy=False)
    cost_hurdle = _target_cost_hurdle_bps(kept_events)
    mae_raw = kept_events["mae_bps"].to_numpy(dtype=np.float32, copy=False)
    # RC3: clip paper-MAE to the realizable stop-loss level so the q10 model learns
    # the bounded worst-case, not the unbounded close-based paper drawdown.
    # mae_bps is negative (min path return); -sl_thr_bps is also negative; max() is
    # "less negative" = the stop-bounded floor.
    if bool(getattr(cfg, "q10_bound_to_stop", True)) and "sl_thr_bps" in kept_events.columns:
        sl_thr_raw = kept_events["sl_thr_bps"].to_numpy(dtype=np.float32, copy=False)
        mae_raw = np.maximum(mae_raw, -sl_thr_raw)
    y_q10 = np.minimum(
        mae_raw - cost_hurdle,
        y_edge,
    )
    y_mfe = kept_events["mfe_bps"].to_numpy(dtype=np.float32, copy=False) - cost_hurdle
    uniqueness_weight, effective_sample_size = _event_uniqueness_weights(
        entry_idx=kept_events["entry_idx"].to_numpy(dtype=np.int32, copy=False),
        exit_idx=kept_events["exit_idx"].to_numpy(dtype=np.int32, copy=False),
        split_start=split_start,
        split_end=split_end,
    )

    return CandidateDataset(
        X=x_final,
        y_gate=y_gate,
        y_edge_bps=y_edge,
        y_q10_bps=y_q10.astype(np.float32, copy=False),
        y_mfe_bps=y_mfe,
        gate_weight=uniqueness_weight,
        edge_weight=uniqueness_weight.copy(),
        groups=groups_final,
        event_index=kept_events.reset_index(drop=True),
        feature_names=feature_names,
        effective_sample_size=effective_sample_size,
    )
