from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.allocator import compute_funding_4h_2d
from src.domain.futures.compound.config import DenseSimConfig
from src.domain.futures.compound.contracts import ExecutionLedger, TimeframeBarCube

_logger = logging.getLogger(__name__)


def simulate_dense_portfolio(
    bars_4h: TimeframeBarCube,
    target_weights_2d: NDArray[np.float64],
    funding_1h_2d: NDArray[np.float32],
    cost_bps: float | NDArray[np.float32],
    config: DenseSimConfig,
) -> ExecutionLedger:
    _logger.info("[SYS] simulate_dense_portfolio: T=%d N=%d", bars_4h.timestamps_ns.size, len(bars_4h.symbols))

    n_bars = bars_4h.timestamps_ns.size
    n_syms = len(bars_4h.symbols)

    if target_weights_2d.shape != (n_bars, n_syms):
        raise ValueError(f"target_weights_2d shape {target_weights_2d.shape} != ({n_bars}, {n_syms})")
    if funding_1h_2d.shape != (n_bars * 4, n_syms):
        raise ValueError(f"funding_1h_2d shape {funding_1h_2d.shape} != ({n_bars * 4}, {n_syms})")

    net_returns = np.zeros(n_bars, dtype=np.float64)
    equity = np.ones(n_bars, dtype=np.float64)
    fee_returns = np.zeros(n_bars, dtype=np.float64)
    slippage_returns = np.zeros(n_bars, dtype=np.float64)
    impact_returns = np.zeros(n_bars, dtype=np.float64)
    funding_returns = np.zeros(n_bars, dtype=np.float64)

    prev_w = np.zeros(n_syms, dtype=np.float64)
    pending_turnover_cost = 0.0
    integrity_failures: list[str] = []

    funding_4h = compute_funding_4h_2d(funding_1h_2d, n_bars)

    for t in range(n_bars):
        if t == 0:
            prev_w = target_weights_2d[0].copy()
            net_returns[t] = 0.0
            equity[t] = 1.0
            continue

        close_prev = bars_4h.close_2d[t - 1].astype(np.float64)
        close_t = bars_4h.close_2d[t].astype(np.float64)

        sym_ret = np.where(
            (close_prev > 0) & (close_t > 0),
            close_t / close_prev - 1.0,
            0.0,
        )
        bar_return = float(np.nansum(prev_w * sym_ret))

        funding_ret = -float(np.sum(prev_w * funding_4h[t]))
        funding_returns[t] = funding_ret

        fee_returns[t] = -pending_turnover_cost

        net_returns[t] = bar_return + funding_ret - pending_turnover_cost
        equity[t] = equity[t - 1] * max(1.0 + net_returns[t], 1e-12)

        delta_w = target_weights_2d[t] - prev_w
        if isinstance(cost_bps, np.ndarray) and cost_bps.ndim == 2 and cost_bps.shape[1] > 1:
            sym_cost = cost_bps[t].astype(np.float64) if t < cost_bps.shape[0] else np.full(n_syms, float(cost_bps))
            sym_cost = np.maximum(sym_cost, 1e-8)
            turnover_cost = float(np.sum(np.abs(delta_w) * sym_cost)) * 1e-4
        else:
            taker_bps = float(np.mean(cost_bps[t])) if isinstance(cost_bps, np.ndarray) else float(cost_bps)
            turnover_cost = float(np.sum(np.abs(delta_w))) * taker_bps * 1e-4
        pending_turnover_cost = turnover_cost

        prev_w = target_weights_2d[t].copy()

        if not np.isfinite(net_returns[t]):
            integrity_failures.append(f"bar {t}: non-finite net return {net_returns[t]}")

    integrity_ok = len(integrity_failures) == 0
    return ExecutionLedger(
        timestamps_ns=bars_4h.timestamps_ns,
        net_returns_1d=net_returns,
        equity_1d=equity,
        target_weights_2d=target_weights_2d.astype(np.float32),
        fee_returns_1d=fee_returns,
        slippage_returns_1d=slippage_returns,
        impact_returns_1d=impact_returns,
        funding_returns_1d=funding_returns,
        integrity_ok=integrity_ok,
        integrity_reasons=tuple(integrity_failures),
    )
