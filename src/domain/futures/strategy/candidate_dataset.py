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
    sample_weight: NDArray[np.float32]
    groups: NDArray[np.int32]
    event_index: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_schema_version: str = "candidate_v2"


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


def _safe_mean(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size > 0 else 0.0


def _safe_std(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size > 0 else 0.0


def _rolling_series_mean_std(
    series: NDArray[np.float64],
    end_idx: int,
    window: int,
) -> tuple[float, float]:
    start = max(0, end_idx - window + 1)
    hist = series[start : end_idx + 1]
    finite = hist[np.isfinite(hist)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite)), float(np.std(finite))


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


def build_candidate_dataset(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    split_start: int,
    split_end: int,
) -> CandidateDataset:
    """Build model matrix for candidate gate and edge models.

    Time Complexity: O(E) for selected events.
    Space Complexity: O(E * F).
    """
    if split_end <= split_start:
        raise ValueError("split_end must be greater than split_start")

    identity_feature_names = _ordered_identity_feature_names(labeled_events, cfg=cfg)
    market_feature_names = _market_state_feature_names(cfg)

    if labeled_events.empty:
        return CandidateDataset(
            X=np.zeros((0, 0), dtype=np.float32),
            y_gate=np.zeros((0,), dtype=np.int8),
            y_edge_bps=np.zeros((0,), dtype=np.float32),
            y_q10_bps=np.zeros((0,), dtype=np.float32),
            y_mfe_bps=np.zeros((0,), dtype=np.float32),
            sample_weight=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=np.int32),
            event_index=labeled_events.copy(),
            feature_names=(),
        )

    mask = (labeled_events["entry_idx"] >= split_start) & (labeled_events["entry_idx"] < split_end)
    events = labeled_events.loc[mask].copy()
    if events.empty:
        return CandidateDataset(
            X=np.zeros((0, 0), dtype=np.float32),
            y_gate=np.zeros((0,), dtype=np.int8),
            y_edge_bps=np.zeros((0,), dtype=np.float32),
            y_q10_bps=np.zeros((0,), dtype=np.float32),
            y_mfe_bps=np.zeros((0,), dtype=np.float32),
            sample_weight=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=np.int32),
            event_index=events,
            feature_names=(),
        )

    feature_rows: list[list[float]] = []
    groups: list[int] = []
    kept_positions: list[int] = []

    close = aligned.close_2d
    volume = aligned.volume_2d
    funding = aligned.funding_2d
    btc_idx = _btc_symbol_index(aligned.symbols)
    close_safe = np.maximum(close, 1e-12)
    log_ret_2d = np.zeros_like(close, dtype=np.float64)
    log_ret_2d[1:] = np.diff(np.log(close_safe), axis=0)
    mkt_vol_series = np.zeros(close.shape[0], dtype=np.float64)
    mkt_disp_series = np.zeros(close.shape[0], dtype=np.float64)
    market_breadth_series = np.zeros(close.shape[0], dtype=np.float64)
    sym_vol_20_series = np.zeros_like(close, dtype=np.float64)

    for t in range(close.shape[0]):
        if t < 20:
            continue
        hist_ret = log_ret_2d[t - 19 : t + 1]
        mkt_vol_series[t] = _safe_mean(np.nanstd(hist_ret, axis=0))
        ret20 = (close[t] / np.maximum(close[t - 20], 1e-12)) - 1.0
        mkt_disp_series[t] = _safe_std(ret20.astype(np.float64, copy=False))
        market_breadth_series[t] = float(np.mean(ret20 > 0.0))
        sym_vol_20_series[t] = np.nanstd(hist_ret, axis=0)

    for pos, row in enumerate(events.itertuples(index=False)):
        symbol = str(row.symbol)
        sym_idx = _find_symbol_index(aligned.symbols, symbol)
        t = int(row.entry_idx) - 1
        if t < 20:
            continue

        side = float(row.side)
        raw_score = float(getattr(row, "raw_score", getattr(row, "score", 0.0)))
        score_z = float(getattr(row, "score_z", 0.0))
        turnover_proxy = float(getattr(row, "turnover_proxy", 0.0))
        expected_holding_bars = float(getattr(row, "expected_holding_bars", 0.0))
        min_holding_bars = float(getattr(row, "min_holding_bars", 0.0))
        stop_atr_mult = float(getattr(row, "stop_atr_mult", 0.0))
        take_profit_atr_mult = float(getattr(row, "take_profit_atr_mult", 0.0))

        ret_1 = (close[t, sym_idx] / close[t - 1, sym_idx]) - 1.0
        ret_5 = (close[t, sym_idx] / close[t - 5, sym_idx]) - 1.0
        vol_20 = float(np.std(np.diff(np.log(np.maximum(close[t - 20 : t + 1, sym_idx], 1e-12)))))
        vol_hist = volume[t - 20 : t, sym_idx]
        vol_std = float(np.nanstd(vol_hist))
        vol_z = (
            float((volume[t, sym_idx] - np.nanmean(vol_hist)) / vol_std)
            if np.isfinite(vol_std) and vol_std > 0.0
            else 0.0
        )

        mkt_ret_1 = float(np.nanmean((close[t] / np.maximum(close[t - 1], 1e-12)) - 1.0))
        mkt_disp_20 = float(np.nanstd((close[t] / np.maximum(close[t - 20], 1e-12)) - 1.0))
        mkt_vol_20 = float(np.nanstd(np.diff(np.log(np.maximum(close[t - 20 : t + 1], 1e-12))), axis=0).mean())

        cost_bps = float(getattr(row, "ex_ante_cost_bps", 0.0))
        if not np.isfinite(cost_bps):
            if aligned.execution_cost_bps_2d is not None:
                cost_bps = float(aligned.execution_cost_bps_2d[t, sym_idx])
            else:
                cost_bps = 0.0

        funding_hist = funding[max(0, t - 20) : t + 1, sym_idx]
        funding_std = float(np.nanstd(funding_hist))
        funding_z = (
            float((funding[t, sym_idx] - np.nanmean(funding_hist)) / funding_std)
            if np.isfinite(funding_std) and funding_std > 0.0
            else 0.0
        )

        row_features = [
            side,
            raw_score,
            score_z,
            turnover_proxy,
            expected_holding_bars,
            min_holding_bars,
            stop_atr_mult,
            take_profit_atr_mult,
            ret_1,
            ret_5,
            vol_20,
            vol_z,
            mkt_ret_1,
            mkt_vol_20,
            mkt_disp_20,
            cost_bps,
            funding_z,
        ]

        if cfg.market_state_features_enabled:
            btc_ret_1 = (close[t, btc_idx] / np.maximum(close[t - 1, btc_idx], 1e-12)) - 1.0
            btc_ret_5 = (close[t, btc_idx] / np.maximum(close[t - 5, btc_idx], 1e-12)) - 1.0
            btc_trend_20 = float(np.nanmean(close[max(0, t - 19) : t + 1, btc_idx]))
            btc_trend_100 = float(np.nanmean(close[max(0, t - 99) : t + 1, btc_idx]))
            btc_trend_20_100 = 1.0 if btc_trend_20 >= btc_trend_100 else 0.0
            mkt_vol_mean_120, mkt_vol_std_120 = _rolling_series_mean_std(mkt_vol_series, t, 120)
            mkt_disp_mean_120, mkt_disp_std_120 = _rolling_series_mean_std(mkt_disp_series, t, 120)
            sym_vol_mean_120, sym_vol_std_120 = _rolling_series_mean_std(sym_vol_20_series[:, sym_idx], t, 120)
            ret20_cross = (close[t] / np.maximum(close[t - 20], 1e-12)) - 1.0
            cross_rank = pd.Series(ret20_cross).rank(method="average", pct=True)
            funding_cross = funding[t]
            funding_cross_std = float(np.nanstd(funding_cross))
            funding_cross_mean = float(np.nanmean(funding_cross))
            funding_cross_section_z = (
                float((funding[t, sym_idx] - funding_cross_mean) / funding_cross_std)
                if np.isfinite(funding_cross_std) and funding_cross_std > 0.0
                else 0.0
            )
            cost_to_vol_bps = float(cost_bps / max(vol_20 * 1e4, 1.0))
            row_features.extend(
                [
                    btc_ret_1,
                    btc_ret_5,
                    btc_trend_20_100,
                    float((mkt_vol_series[t] - mkt_vol_mean_120) / mkt_vol_std_120)
                    if np.isfinite(mkt_vol_std_120) and mkt_vol_std_120 > 0.0
                    else 0.0,
                    float((mkt_disp_series[t] - mkt_disp_mean_120) / mkt_disp_std_120)
                    if np.isfinite(mkt_disp_std_120) and mkt_disp_std_120 > 0.0
                    else 0.0,
                    float(market_breadth_series[t]),
                    float(cross_rank.iloc[sym_idx]),
                    float((sym_vol_20_series[t, sym_idx] - sym_vol_mean_120) / sym_vol_std_120)
                    if np.isfinite(sym_vol_std_120) and sym_vol_std_120 > 0.0
                    else 0.0,
                    funding_cross_section_z,
                    cost_to_vol_bps,
                ]
            )

        if cfg.candidate_identity_features_enabled:
            family_value = str(getattr(row, "family", ""))
            variant_value = _stable_variant_key(family_value, str(getattr(row, "variant", "")))
            family_identity = [
                1.0 if name == f"family={family_value}" else 0.0
                for name in identity_feature_names
                if name.startswith("family=")
            ]
            variant_identity = [
                1.0 if name == f"variant={variant_value}" else 0.0
                for name in identity_feature_names
                if name.startswith("variant=")
            ]
            row_features.extend(family_identity)
            row_features.extend(variant_identity)
            row_features.extend([1.0 if side > 0.0 else 0.0, 1.0 if side < 0.0 else 0.0])

        if not np.all(np.isfinite(np.asarray(row_features, dtype=np.float64))):
            continue
        feature_rows.append(row_features)
        groups.append(int(t))
        kept_positions.append(pos)

    feature_names: tuple[str, ...] = (
        "side",
        "raw_score",
        "score_z",
        "turnover_proxy",
        "expected_holding_bars",
        "min_holding_bars",
        "stop_atr_mult",
        "take_profit_atr_mult",
        "sym_ret_1",
        "sym_ret_5",
        "sym_vol_20",
        "sym_volume_z20",
        "mkt_ret_1",
        "mkt_vol_20",
        "mkt_dispersion_20",
        "ex_ante_cost_bps",
        "funding_z20",
    )
    feature_names = feature_names + market_feature_names + identity_feature_names

    if not feature_rows:
        return CandidateDataset(
            X=np.zeros((0, len(feature_names)), dtype=np.float32),
            y_gate=np.zeros((0,), dtype=np.int8),
            y_edge_bps=np.zeros((0,), dtype=np.float32),
            y_q10_bps=np.zeros((0,), dtype=np.float32),
            y_mfe_bps=np.zeros((0,), dtype=np.float32),
            sample_weight=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=np.int32),
            event_index=events.iloc[0:0].copy(),
            feature_names=feature_names,
        )

    kept_events = events.iloc[kept_positions].copy()

    gate_label_col = (
        "profitable_after_hurdle_label"
        if "profitable_after_hurdle_label" in kept_events.columns
        else "triple_barrier_label"
    )
    y_gate = kept_events[gate_label_col].to_numpy(dtype=np.int8, copy=False)
    y_edge = kept_events["edge_after_hurdle_bps"].to_numpy(dtype=np.float32, copy=False)
    cost_hurdle = _target_cost_hurdle_bps(kept_events)
    y_q10 = np.minimum(
        kept_events["mae_bps"].to_numpy(dtype=np.float32, copy=False) - cost_hurdle,
        y_edge,
    )
    y_mfe = kept_events["mfe_bps"].to_numpy(dtype=np.float32, copy=False) - cost_hurdle
    sw = np.clip(np.abs(y_edge) / 10.0, 0.5, 5.0).astype(np.float32, copy=False)

    return CandidateDataset(
        X=np.asarray(feature_rows, dtype=np.float32),
        y_gate=y_gate,
        y_edge_bps=y_edge,
        y_q10_bps=y_q10.astype(np.float32, copy=False),
        y_mfe_bps=y_mfe,
        sample_weight=sw,
        groups=np.asarray(groups, dtype=np.int32),
        event_index=kept_events.reset_index(drop=True),
        feature_names=feature_names,
    )
