from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

if TYPE_CHECKING:
    from src.research.contracts import CostModel


def _default_cost_model() -> CostModel:
    from src.research.contracts import CostModel

    return CostModel()


@dataclass(frozen=True, slots=True)
class NetConstructionSpec:
    """Immutable net-of-turnover construction controls.

    ``rebalance_bars`` is the bar cadence at which targets are re-examined and
    ``no_trade_band`` the hysteresis band that suppresses trades whose gross
    edge is below it; ``costs`` reuses ``src.research.contracts.CostModel`` and
    never redeclares fee or slippage literals.
    """

    rebalance_bars: int = 1
    no_trade_band: float = 0.0
    costs: CostModel = field(default_factory=_default_cost_model)

    def __post_init__(self) -> None:
        if self.rebalance_bars < 1:
            raise ValueError(f"rebalance_bars must be >= 1, got {self.rebalance_bars}")
        if self.no_trade_band < 0:
            raise ValueError(f"no_trade_band must be >= 0, got {self.no_trade_band}")


def apply_no_trade_band(target: np.ndarray, held: np.ndarray, band: float) -> np.ndarray:
    """Hysteresis that turns gross edge into net edge.

    Returns ``held`` wherever ``abs(target - held) <= band`` and ``target``
    otherwise; pure vectorized ``np.where`` that never mutates its inputs.
    """
    if band < 0:
        raise ValueError(f"band must be >= 0, got {band}")
    target_arr = np.asarray(target, dtype=np.float64)
    held_arr = np.asarray(held, dtype=np.float64)
    if target_arr.shape != held_arr.shape:
        raise ValueError("target and held must have identical shapes")
    return np.where(np.abs(target_arr - held_arr) <= band, held_arr, target_arr)


@dataclass(frozen=True)
class NetReturnStream:
    gross: pd.Series
    cost: pd.Series
    funding: pd.Series
    net: pd.Series
    turnover: pd.Series
    realized_weights: pd.DataFrame


def _validate_frames(target_weights: pd.DataFrame, forward_returns: pd.DataFrame) -> None:
    if not isinstance(target_weights.index, pd.DatetimeIndex) or not isinstance(
        forward_returns.index, pd.DatetimeIndex
    ):
        raise DataIntegrityError("target_weights and forward_returns must have a DatetimeIndex")
    if target_weights.index.tz is None or forward_returns.index.tz is None:
        raise DataIntegrityError("frames must have a tz-aware UTC index")
    if not target_weights.index.is_monotonic_increasing or not forward_returns.index.is_monotonic_increasing:
        raise DataIntegrityError("frames must have a monotonic increasing index")
    if not target_weights.index.equals(forward_returns.index):
        raise DataIntegrityError("frames must share an identical index")
    if list(target_weights.columns) != list(forward_returns.columns):
        raise DataIntegrityError("frames must share identical columns")


def compute_net_return_stream(
    target_weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    spec: NetConstructionSpec,
    forward_funding: pd.DataFrame | None = None,
) -> NetReturnStream:
    """Compute the net return stream with explicit turnover and funding accounting.

    On non-rebalance bars the previous realized weights are held; on rebalance
    bars ``apply_no_trade_band`` is applied to the target.  One-way turnover
    ``sum(abs(realized[t] - realized[t-1]))`` (the first position is traded from
    cash) is charged at ``fee_rate + slippage_rate``, and
    ``net = gross - cost - funding`` where
    ``gross[t] = sum(realized[t] * forward_returns[t])`` and
    ``funding[t] = sum(realized[t] * forward_funding[t])``.  A positive funding
    rate therefore debits a long and credits a short.  ``forward_returns`` is
    already decision-to-fill aligned, so no shift happens here.

    ``forward_funding`` is optional and, when supplied, must share the identical
    UTC index and columns with ``target_weights``.  Omitting it is exactly a
    zero matrix and preserves every existing caller's result byte-for-byte.
    """
    _validate_frames(target_weights, forward_returns)
    symbols = list(target_weights.columns)
    n = len(target_weights)
    rate = spec.costs.fee_rate + spec.costs.slippage_rate

    target_arr = target_weights.to_numpy(dtype=np.float64)
    forward_arr = forward_returns.to_numpy(dtype=np.float64)
    if forward_funding is not None:
        _validate_frames(target_weights, forward_funding)
        funding_arr = forward_funding.to_numpy(dtype=np.float64)
    else:
        funding_arr = np.zeros((n, len(symbols)), dtype=np.float64)

    realized = np.zeros((n, len(symbols)), dtype=np.float64)
    gross = np.zeros(n, dtype=np.float64)
    cost = np.zeros(n, dtype=np.float64)
    funding = np.zeros(n, dtype=np.float64)
    turnover = np.zeros(n, dtype=np.float64)
    prev = np.zeros(len(symbols), dtype=np.float64)

    for t in range(n):
        if t % spec.rebalance_bars == 0:
            realized[t] = apply_no_trade_band(target_arr[t], prev, spec.no_trade_band)
        else:
            realized[t] = prev
        turnover[t] = float(np.abs(realized[t] - prev).sum())
        cost[t] = turnover[t] * rate
        # A symbol that is not held (realized weight 0) never contributes its
        # forward return, so a missing/NaN return for an unlisted or backfill
        # symbol cannot poison the held portfolio's gross (0 * NaN = NaN).
        nonzero = realized[t] != 0.0
        gross[t] = float(np.sum(np.where(nonzero, realized[t] * forward_arr[t], 0.0)))
        funding[t] = float(np.sum(np.where(nonzero, realized[t] * funding_arr[t], 0.0)))
        prev = realized[t]

    index = target_weights.index
    net = gross - cost - funding
    return NetReturnStream(
        gross=pd.Series(gross, index=index),
        cost=pd.Series(cost, index=index),
        funding=pd.Series(funding, index=index),
        net=pd.Series(net, index=index),
        turnover=pd.Series(turnover, index=index),
        realized_weights=pd.DataFrame(realized, index=index, columns=symbols),
    )
