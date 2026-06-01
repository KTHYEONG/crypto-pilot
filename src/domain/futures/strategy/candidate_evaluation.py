from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(slots=True, frozen=True)
class CompoundEvaluationReport:
    """Evaluation metrics representing compounding growth and OOS performance robustness."""

    mean_log_growth: float
    cagr: float
    max_drawdown: float
    mar: float
    final_equity: float
    net_pnl: float
    fees: float
    funding: float
    turnover: float
    block_pass_ratio: float
    worst_block_return: float
    dsr: float
    pbo: float
    liquidation_count: int
    pass_compound_gate: bool
    fail_reasons: tuple[str, ...]


def evaluate_compound_backtest(
    *,
    trades: pd.DataFrame,
    equity_curve: NDArray[np.float64],
    diag: NDArray[np.float64] | None = None,
    cfg: CandidateStrategyConfig,
) -> CompoundEvaluationReport:
    """Evaluate geometric capital growth and execution realism of a backtest."""
    del diag
    n_bars = equity_curve.shape[0]
    if n_bars < 2:
        return CompoundEvaluationReport(
            mean_log_growth=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            mar=0.0,
            final_equity=1.0 if n_bars == 0 else float(equity_curve[0]),
            net_pnl=0.0,
            fees=0.0,
            funding=0.0,
            turnover=0.0,
            block_pass_ratio=0.0,
            worst_block_return=0.0,
            dsr=0.0,
            pbo=0.0,
            liquidation_count=0,
            pass_compound_gate=False,
            fail_reasons=("insufficient bars",),
        )

    # 1. Log Growth
    returns = equity_curve[1:] / np.maximum(equity_curve[:-1], 1e-12)
    log_returns = np.log(np.maximum(returns, 1e-12))
    mean_log_growth = float(np.mean(log_returns))

    # 2. CAGR Calculation
    bars_per_year = 2190.0  # Default 4h timeframe (365 * 6)
    if cfg.timeframe == "1h":
        bars_per_year = 8760.0
    elif cfg.timeframe == "1d":
        bars_per_year = 365.0
    
    years = max(n_bars / bars_per_year, 1e-9)
    initial_eq = max(float(equity_curve[0]), 1e-12)
    final_eq = max(float(equity_curve[-1]), 0.0)
    cagr = float((final_eq / initial_eq) ** (1.0 / years) - 1.0) if final_eq > 0.0 else -1.0

    # 3. Drawdown and MAR
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / np.maximum(peaks, 1e-12)
    max_dd = float(np.abs(np.min(drawdowns)))
    mar = float(cagr / max_dd) if max_dd > 1e-9 else 0.0

    # 4. Trades Metrics
    net_pnl = 0.0
    fees = 0.0
    funding = 0.0
    liquidation_count = 0
    turnover = 0.0

    if not trades.empty:
        net_pnl = float(trades["pnl"].sum()) if "pnl" in trades.columns else 0.0
        fees = float(trades["fee"].sum()) if "fee" in trades.columns else 0.0
        funding = float(trades["funding"].sum()) if "funding" in trades.columns else 0.0
        if "is_liquidation" in trades.columns:
            liquidation_count = int(trades["is_liquidation"].sum())
        elif "liquidation" in trades.columns:
            liquidation_count = int(trades["liquidation"].sum())
        
        if "size" in trades.columns and n_bars > 0:
            turnover = float(trades["size"].sum() / initial_eq / n_bars)

    # 5. Non-overlapping 6-month block evaluation
    # 6 months = 180 days = 1080 bars in 4h timeframe
    block_size = int(bars_per_year / 2.0)
    n_blocks = max(1, n_bars // block_size)
    block_returns: list[float] = []

    for i in range(n_blocks):
        st = i * block_size
        ed = min((i + 1) * block_size, n_bars - 1)
        if ed > st:
            b_ret = float((equity_curve[ed] / max(equity_curve[st], 1e-12)) - 1.0)
            block_returns.append(b_ret)

    passed_blocks = sum(1 for r in block_returns if r > 0.0)
    block_pass_ratio = float(passed_blocks / len(block_returns)) if block_returns else 0.0
    worst_block_return = float(min(block_returns)) if block_returns else 0.0

    # 6. Promotion Gate Check
    fail_reasons: list[str] = []
    if mean_log_growth <= 0.0:
        fail_reasons.append("negative log growth")
    if cagr <= 0.0:
        fail_reasons.append("negative CAGR")
    if max_dd > cfg.gross_cap:
        fail_reasons.append(f"max drawdown {max_dd:.3f} exceeds gross cap limit {cfg.gross_cap:.3f}")
    if mar < 0.75:
        fail_reasons.append(f"MAR ratio {mar:.3f} is below 0.75 target")
    if worst_block_return <= -0.3:
        fail_reasons.append(f"worst block return {worst_block_return:.3f} exceeds maximum loss target")
    if liquidation_count > 0:
        fail_reasons.append("liquidation occurred during simulation")
    if block_pass_ratio < 0.70:
        fail_reasons.append(f"block pass ratio {block_pass_ratio:.3f} is below 0.70 threshold")
    if net_pnl <= 0.0:
        fail_reasons.append("negative net pnl after costs")

    pass_gate = len(fail_reasons) == 0

    return CompoundEvaluationReport(
        mean_log_growth=mean_log_growth,
        cagr=cagr,
        max_drawdown=max_dd,
        mar=mar,
        final_equity=final_eq,
        net_pnl=net_pnl,
        fees=fees,
        funding=funding,
        turnover=turnover,
        block_pass_ratio=block_pass_ratio,
        worst_block_return=worst_block_return,
        dsr=0.0,  # Hook for future DSR
        pbo=0.0,  # Hook for future PBO
        liquidation_count=liquidation_count,
        pass_compound_gate=pass_gate,
        fail_reasons=tuple(fail_reasons),
    )
