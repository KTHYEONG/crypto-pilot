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
    sample_weight: NDArray[np.float32]
    groups: NDArray[np.int32]
    event_index: pd.DataFrame
    feature_names: tuple[str, ...]


def _find_symbol_index(symbols: tuple[str, ...], symbol: str) -> int:
    for idx, value in enumerate(symbols):
        if value == symbol:
            return idx
    raise KeyError(f"unknown symbol: {symbol}")


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
    del cfg
    if split_end <= split_start:
        raise ValueError("split_end must be greater than split_start")

    if labeled_events.empty:
        return CandidateDataset(
            X=np.zeros((0, 0), dtype=np.float32),
            y_gate=np.zeros((0,), dtype=np.int8),
            y_edge_bps=np.zeros((0,), dtype=np.float32),
            y_q10_bps=np.zeros((0,), dtype=np.float32),
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
            sample_weight=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=np.int32),
            event_index=events,
            feature_names=(),
        )

    feature_rows: list[list[float]] = []
    groups: list[int] = []

    close = aligned.close_2d
    volume = aligned.volume_2d
    funding = aligned.funding_2d

    for row in events.itertuples(index=False):
        symbol = str(row.symbol)
        sym_idx = _find_symbol_index(aligned.symbols, symbol)
        t = int(row.entry_idx) - 1
        if t < 20:
            continue

        side = float(row.side)
        raw_score = float(getattr(row, "raw_score", getattr(row, "score", 0.0)))
        score_z = float(getattr(row, "score_z", 0.0))
        turnover_proxy = float(getattr(row, "turnover_proxy", 0.0))

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
        if not np.all(np.isfinite(np.asarray(row_features, dtype=np.float64))):
            continue
        feature_rows.append(row_features)
        groups.append(int(t))

    feature_names = (
        "side",
        "raw_score",
        "score_z",
        "turnover_proxy",
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

    if not feature_rows:
        return CandidateDataset(
            X=np.zeros((0, len(feature_names)), dtype=np.float32),
            y_gate=np.zeros((0,), dtype=np.int8),
            y_edge_bps=np.zeros((0,), dtype=np.float32),
            y_q10_bps=np.zeros((0,), dtype=np.float32),
            sample_weight=np.zeros((0,), dtype=np.float32),
            groups=np.zeros((0,), dtype=np.int32),
            event_index=events.iloc[0:0].copy(),
            feature_names=feature_names,
        )

    valid_len = len(feature_rows)
    kept_events = events.iloc[:valid_len].copy()

    y_gate = kept_events["triple_barrier_label"].to_numpy(dtype=np.int8, copy=False)
    y_edge = kept_events["edge_after_hurdle_bps"].to_numpy(dtype=np.float32, copy=False)
    y_q10 = np.minimum(
        kept_events["mae_bps"].to_numpy(dtype=np.float32, copy=False),
        y_edge,
    )
    sw = np.clip(np.abs(y_edge) / 10.0, 0.5, 5.0).astype(np.float32, copy=False)

    return CandidateDataset(
        X=np.asarray(feature_rows, dtype=np.float32),
        y_gate=y_gate,
        y_edge_bps=y_edge,
        y_q10_bps=y_q10.astype(np.float32, copy=False),
        sample_weight=sw,
        groups=np.asarray(groups, dtype=np.int32),
        event_index=kept_events.reset_index(drop=True),
        feature_names=feature_names,
    )
