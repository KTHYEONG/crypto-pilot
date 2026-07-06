from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from src.domain.futures.strategy.config import CandidateStrategyConfig


def _calc_dsr(block_returns: list[float], n_trials: int = 1) -> float:
    """Simplified DSR: Sharpe ratio deflated for multiple comparisons.

    Args:
        block_returns: List of OOS block returns.
        n_trials: Number of strategy trials evaluated (for deflation).
            Defaults to 1 (no deflation). Pass the actual Optuna trial count
            to enable Bailey-Lopez de Prado multi-test correction.

    Returns:
        DSR value in [0.0, 1.0].
    """
    if len(block_returns) < 2:
        return 0.0
    returns_arr = np.array(block_returns, dtype=np.float64)
    std_val = float(np.std(returns_arr, ddof=1))
    sr = float(np.mean(returns_arr) / (std_val + 1e-12))

    # Bailey-Lopez de Prado deflation factor
    n = len(block_returns)
    # E[max(SR)] for n_trials i.i.d. SR ~ N(0,1)
    e_max_sr = (1.0 - 0.5772156649) / math.log(max(n_trials, 2)) if n_trials > 1 else 0.0

    mean_val = float(np.mean(returns_arr))
    demeaned = returns_arr - mean_val
    std_safe = std_val + 1e-12
    gamma_1 = float(np.mean(demeaned**3) / std_safe**3)
    gamma_2 = float(np.mean(demeaned**4) / std_safe**4) - 3.0

    deflation_denom = n - 1 + sr**2 * (gamma_1 * sr / 6.0 - gamma_2 * sr**2 / 24.0)
    sr_adj = sr * math.sqrt(n) / math.sqrt(max(deflation_denom, 1e-12))

    dsr_val = float(norm.cdf((sr_adj - e_max_sr) * math.sqrt(n)))
    return float(np.clip(dsr_val, 0.0, 1.0))


def _calc_pbo(block_returns: list[float]) -> float:
    """Simplified PBO: fraction of OOS blocks with negative return.

    Args:
        block_returns: List of OOS block returns.

    Returns:
        PBO value in [0.0, 1.0].
    """
    if not block_returns:
        return 1.0
    n_neg = sum(1 for r in block_returns if r < 0.0)
    return float(n_neg / len(block_returns))


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
    fold_oos_boundaries: tuple[tuple[int, int], ...] | None = None,
    deployed_bar_fraction: float | None = None,
    trade_count: int | None = None,
) -> CompoundEvaluationReport:
    """Evaluate geometric capital growth and execution realism of a backtest.

    Args:
        deployed_bar_fraction: Fraction of bars with non-zero exposure.  Pass
            ``None`` (default) to skip the deployment enforcement check; always
            provide an explicit value when ``enforce_deployment_in_compound_gate``
            matters.
        trade_count: Number of round-trip trades.  Same sentinel semantics as
            ``deployed_bar_fraction``.
    """
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
    # RC2 guard: max_dd below the floor means cagr/max_dd is a ratio of two noise-level
    # numbers that can explode (e.g. 0.0001% / 0.00005% = MAR 2). Treat as MAR = 0.
    mar_min_dd = float(getattr(cfg, "mar_min_drawdown_floor", 0.01))
    mar = float(cagr / max_dd) if max_dd > mar_min_dd else 0.0

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

    # 5. Block evaluation: fold OOS boundaries if provided, else 6-month fallback
    block_returns: list[float] = []
    if fold_oos_boundaries:
        for oos_s, oos_e in fold_oos_boundaries:
            st = max(0, min(oos_s, n_bars - 1))
            ed = max(0, min(oos_e, n_bars - 1))
            if ed > st:
                b_ret = float((equity_curve[ed] / max(equity_curve[st], 1e-12)) - 1.0)
                block_returns.append(b_ret)
    else:
        block_size = int(bars_per_year / 2.0)
        n_blocks = max(1, n_bars // block_size)
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
    # RC2: absolute CAGR floor prevents noise-level gains from passing
    min_cagr = float(getattr(cfg, "min_cagr_for_promotion", 0.02))
    if cagr < min_cagr:
        fail_reasons.append(f"CAGR {cagr:.4f} below min_cagr_for_promotion {min_cagr:.4f}")
    drawdown_cap = float(getattr(cfg, "max_drawdown_cap", 0.25))
    if max_dd > drawdown_cap:
        fail_reasons.append(f"max drawdown {max_dd:.3f} exceeds max_drawdown_cap {drawdown_cap:.3f}")
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
    # RC2: deployment enforcement — near-zero-trading variants must not pass.
    # Only applied when explicit deployment metrics are provided (not None sentinel).
    if (
        bool(getattr(cfg, "enforce_deployment_in_compound_gate", True))
        and deployed_bar_fraction is not None
        and trade_count is not None
    ):
        min_deploy_frac = float(getattr(cfg, "min_deployment_capital_fraction", 0.05))
        min_deploy_trades = int(getattr(cfg, "min_deployment_trade_count", 20))
        if deployed_bar_fraction < min_deploy_frac:
            fail_reasons.append(f"deployed_bar_fraction {deployed_bar_fraction:.3f} below {min_deploy_frac:.3f}")
        if trade_count < min_deploy_trades:
            fail_reasons.append(f"trade_count {trade_count} below min_deployment_trade_count {min_deploy_trades}")

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
        dsr=_calc_dsr(block_returns),
        pbo=_calc_pbo(block_returns),
        liquidation_count=liquidation_count,
        pass_compound_gate=pass_gate,
        fail_reasons=tuple(fail_reasons),
    )
