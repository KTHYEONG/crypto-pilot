from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger
from src.core.types import CostModel, StrategySpec
from src.engine.backtest import BacktestResult, run_backtest
from src.validation.metrics import compute_metrics

_logger = setup_logger("ReliabilityGate")

_SECONDS_PER_YEAR = 365.25 * 86400


@dataclass(frozen=True, slots=True)
class ReliabilityGateConfig:
    hurdle_rate: float = 0.15
    block_size: int | None = None
    n_bootstrap: int = 3000
    seed: int = 0
    min_trades: int = 30
    mdd_floor: float = -0.25
    t_stat_floor: float = 2.0
    lcb_confidence: float = 0.90
    max_period_contribution: float = 0.40

    def __post_init__(self) -> None:
        if self.hurdle_rate < 0:
            raise ValueError(f"hurdle_rate must be >= 0, got {self.hurdle_rate}")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100, got {self.n_bootstrap}")
        if self.min_trades < 1:
            raise ValueError(f"min_trades must be >= 1, got {self.min_trades}")
        if self.mdd_floor >= 0:
            raise ValueError(f"mdd_floor must be < 0, got {self.mdd_floor}")
        if self.t_stat_floor < 0:
            raise ValueError(f"t_stat_floor must be >= 0, got {self.t_stat_floor}")
        if not 0.5 < self.lcb_confidence < 1.0:
            raise ValueError(f"lcb_confidence must be in (0.5, 1.0), got {self.lcb_confidence}")
        if not 0.0 < self.max_period_contribution <= 1.0:
            raise ValueError(
                f"max_period_contribution must be in (0.0, 1.0], got {self.max_period_contribution}"
            )


@dataclass(frozen=True, slots=True)
class ReliabilityGateResult:
    lcb90_cagr: float
    lcb95_cagr: float
    p_negative: float
    point_cagr: float
    t_stat: float
    trade_count: int
    block_size_used: int
    verdict: Literal["PASS", "FAIL", "PENDING"]


@dataclass(frozen=True, slots=True)
class FoldDistributionResult:
    n_folds: int
    median_fold_cagr: float
    worst_fold_cagr: float
    median_fold_calmar: float
    max_period_contribution: float
    gate_pass: bool


@dataclass(frozen=True, slots=True)
class HoldoutSegment:
    observation_trades: pd.DataFrame
    holdout_trades: pd.DataFrame
    observation_equity: pd.Series
    holdout_equity: pd.Series
    holdout_years: float
    holdout_mdd: float
    holdout_cagr_sign: float


def derive_block_size(returns: np.ndarray) -> int:
    """Data-driven block length from the run's own trade-return autocorrelation.

    White noise (no significant lag) returns 1, reducing to a plain bootstrap.
    Never raises; tiny samples (n<10) return 1 rather than an unreliable ACF.
    """
    n = len(returns)
    if n < 10:
        return 1
    x = returns - returns.mean()
    denom = float(np.sum(x**2))
    if denom <= 0.0:
        return 1
    max_lag = min(20, n // 4)
    band = 1.96 / np.sqrt(n)
    block = 1
    for lag in range(1, max_lag + 1):
        acf = float(np.sum(x[:-lag] * x[lag:])) / denom
        if abs(acf) > band:
            block = lag
    return min(block, max(1, n // 5))


def _block_bootstrap_cagr(
    rets: np.ndarray,
    *,
    years: float,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    n = len(rets)
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(n_bootstrap, n_blocks * block_size)[:, :n]
    resampled = rets[idx]
    equity_b = np.asarray(np.prod(1.0 + resampled, axis=1), dtype=np.float64)
    return np.asarray(equity_b ** (1.0 / years) - 1.0, dtype=np.float64)


def compute_reliability_gate(
    trades: pd.DataFrame,
    years: float,
    sharpe: float,
    mdd: float,
    config: ReliabilityGateConfig = ReliabilityGateConfig(),  # noqa: B008
) -> ReliabilityGateResult:
    if years <= 0:
        raise ValueError(f"years must be > 0, got {years}")
    if "return_pct" not in trades.columns:
        raise ValueError("trades must contain a 'return_pct' column")
    if len(trades) == 0:
        result = ReliabilityGateResult(
            lcb90_cagr=0.0, lcb95_cagr=0.0, p_negative=0.0,
            point_cagr=0.0, t_stat=0.0, trade_count=0,
            block_size_used=1, verdict="PENDING",
        )
        _logger.info(
            "lcb90=%.4f lcb95=%.4f p_neg=%.3f point_cagr=%.4f t_stat=%.3f trades=0 block=%d verdict=%s",
            result.lcb90_cagr, result.lcb95_cagr, result.p_negative,
            result.point_cagr, result.t_stat, result.block_size_used, result.verdict,
            extra={"tag": "EVAL"},
        )
        return result

    rets = trades["return_pct"].to_numpy(dtype=np.float64)
    trade_count = len(trades)
    block_size = config.block_size if config.block_size is not None else derive_block_size(rets)

    cagr_b = _block_bootstrap_cagr(
        rets, years=years, block_size=block_size,
        n_bootstrap=config.n_bootstrap, seed=config.seed,
    )
    lcb90_cagr = float(np.percentile(cagr_b, 100.0 * (1.0 - config.lcb_confidence)))
    lcb95_cagr = float(np.percentile(cagr_b, 5.0))
    p_negative = float(np.mean(cagr_b < 0.0))
    point_cagr = float(np.prod(1.0 + rets) ** (1.0 / years) - 1.0)
    t_stat = float(sharpe * np.sqrt(years))

    if trade_count < config.min_trades:
        verdict: Literal["PASS", "FAIL", "PENDING"] = "PENDING"
    elif (
        mdd > config.mdd_floor
        and t_stat > config.t_stat_floor
        and lcb90_cagr > config.hurdle_rate
    ):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    result = ReliabilityGateResult(
        lcb90_cagr=lcb90_cagr, lcb95_cagr=lcb95_cagr, p_negative=p_negative,
        point_cagr=point_cagr, t_stat=t_stat, trade_count=trade_count,
        block_size_used=block_size, verdict=verdict,
    )
    _logger.info(
        "lcb90=%.4f lcb95=%.4f p_neg=%.3f point_cagr=%.4f t_stat=%.3f trades=%d block=%d verdict=%s",
        lcb90_cagr, lcb95_cagr, p_negative, point_cagr, t_stat, trade_count,
        block_size, verdict, extra={"tag": "EVAL"},
    )
    return result


def compute_equity_reliability_gate(
    equity: pd.Series,
    closed_trade_count: int,
    config: ReliabilityGateConfig = ReliabilityGateConfig(),  # noqa: B008
) -> ReliabilityGateResult:
    """Canonical promotion gate on the causal marked total-equity return stream.

    The sampled return stream is the single marked ledger, never independent
    trade/sleeve returns compounded separately, so concurrent positions cannot
    inflate the LCB by multiplying per-sleeve paths. All numerical gate limits
    are the frozen ones from ``ReliabilityGateConfig`` (15% LCB90, -25% MDD,
    2.0 t-stat, 30 closes); the block size is derived from the return
    autocorrelation. ``closed_trade_count`` is only a sample-size guard and
    never supplies returns.
    """
    if not isinstance(equity.index, pd.DatetimeIndex) or len(equity) < 2:
        raise ValueError("equity must be a DatetimeIndex series with at least 2 points")
    if not equity.index.is_monotonic_increasing:
        raise ValueError("equity index must be monotonic increasing")
    values = equity.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("equity must contain only finite values")
    if (values <= 0).any():
        raise ValueError("equity must be strictly positive")
    if closed_trade_count < 0:
        raise ValueError(f"closed_trade_count must be >= 0, got {closed_trade_count}")

    years = (equity.index[-1] - equity.index[0]).total_seconds() / _SECONDS_PER_YEAR
    if years <= 0:
        raise ValueError("equity must span a positive time range")

    returns = equity.pct_change().dropna().to_numpy(dtype=np.float64)
    if len(returns) == 0:
        result = ReliabilityGateResult(
            lcb90_cagr=0.0, lcb95_cagr=0.0, p_negative=0.0,
            point_cagr=0.0, t_stat=0.0, trade_count=closed_trade_count,
            block_size_used=1, verdict="PENDING",
        )
        _logger.info(
            "lcb90=%.4f lcb95=%.4f p_neg=%.3f point_cagr=%.4f t_stat=%.3f trades=%d block=%d verdict=%s",
            result.lcb90_cagr, result.lcb95_cagr, result.p_negative,
            result.point_cagr, result.t_stat, result.trade_count,
            result.block_size_used, result.verdict, extra={"tag": "EVAL"},
        )
        return result

    block_size = config.block_size if config.block_size is not None else derive_block_size(returns)
    cagr_b = _block_bootstrap_cagr(
        returns, years=years, block_size=block_size,
        n_bootstrap=config.n_bootstrap, seed=config.seed,
    )
    lcb90_cagr = float(np.percentile(cagr_b, 100.0 * (1.0 - config.lcb_confidence)))
    lcb95_cagr = float(np.percentile(cagr_b, 5.0))
    p_negative = float(np.mean(cagr_b < 0.0))
    point_cagr = float(np.prod(1.0 + returns) ** (1.0 / years) - 1.0)

    std_ret = float(np.std(returns))
    sharpe = float(np.mean(returns) / std_ret * np.sqrt(2190.0)) if std_ret > 0 else 0.0
    t_stat = sharpe * np.sqrt(years)
    mdd = float((equity / equity.cummax() - 1.0).min())

    if closed_trade_count < config.min_trades:
        verdict: Literal["PASS", "FAIL", "PENDING"] = "PENDING"
    elif (
        mdd > config.mdd_floor
        and t_stat > config.t_stat_floor
        and lcb90_cagr > config.hurdle_rate
    ):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    result = ReliabilityGateResult(
        lcb90_cagr=lcb90_cagr, lcb95_cagr=lcb95_cagr, p_negative=p_negative,
        point_cagr=point_cagr, t_stat=float(t_stat), trade_count=closed_trade_count,
        block_size_used=block_size, verdict=verdict,
    )
    _logger.info(
        "lcb90=%.4f lcb95=%.4f p_neg=%.3f point_cagr=%.4f t_stat=%.3f trades=%d block=%d verdict=%s",
        lcb90_cagr, lcb95_cagr, p_negative, point_cagr, t_stat,
        closed_trade_count, block_size, verdict, extra={"tag": "EVAL"},
    )
    return result


def compute_portfolio_reliability_gate(
    equity: pd.Series,
    closed_trade_count: int,
    config: ReliabilityGateConfig = ReliabilityGateConfig(),  # noqa: B008
) -> ReliabilityGateResult:
    """Compatibility delegator for promotion: portfolio evidence uses the canonical equity gate."""
    return compute_equity_reliability_gate(equity, closed_trade_count, config)


def _year_log_return_contributions(equity: pd.Series) -> dict[int, float]:
    """Annual marked log-return contributions.

    Groups ``log(E_t / E_{t-1})`` by the timestamp of the observed mark ``E_t``,
    so cross-year open positions contribute to every year containing their marked
    return. Exact under compounding because the per-year sums telescope to the
    total log growth.
    """
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("equity must have a DatetimeIndex")
    if len(equity) < 2:
        raise ValueError("equity must contain at least 2 points")
    if not equity.index.is_monotonic_increasing:
        raise ValueError("equity index must be monotonic increasing")
    values = equity.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("equity must contain only finite values")
    if (values <= 0).any():
        raise ValueError("equity must be strictly positive")

    log_returns = np.log(values[1:] / values[:-1])
    years = equity.index[1:].year
    contributions: dict[int, float] = {}
    for year, log_ret in zip(years, log_returns, strict=True):
        contributions[year] = contributions.get(year, 0.0) + float(log_ret)
    return contributions


def split_holdout_segment(result: BacktestResult, cutoff: pd.Timestamp) -> HoldoutSegment:
    equity = result.equity
    if equity.index.tz is not None and cutoff.tzinfo is None:
        raise ValueError("cutoff is tz-naive while equity index is tz-aware")
    if equity.index.tz is None and cutoff.tzinfo is not None:
        raise ValueError("cutoff is tz-aware while equity index is tz-naive")
    if not (equity.index[0] <= cutoff <= equity.index[-1]):
        raise ValueError(
            f"cutoff {cutoff} outside equity index range [{equity.index[0]}, {equity.index[-1]}]"
        )

    cutoff_idx = int(equity.index.searchsorted(cutoff, side="right")) - 1
    if len(result.trades) > 0:
        if "exit_time" in result.trades.columns:
            exit_ts = pd.to_datetime(result.trades["exit_time"], utc=True, errors="raise")
        elif "exit_bar" in result.trades.columns:
            exit_ts = equity.index[result.trades["exit_bar"].astype(int).to_numpy()]
        else:
            raise ValueError(
                "trades must carry 'exit_bar' (single symbol) or 'exit_time' "
                "(portfolio) for holdout attribution"
            )
        holdout_mask = exit_ts > cutoff
    else:
        holdout_mask = pd.Series(False, index=result.trades.index)
    holdout_trades = result.trades[holdout_mask]
    observation_trades = result.trades[~holdout_mask]

    observation_equity = equity.iloc[: cutoff_idx + 1]
    holdout_equity = equity.iloc[cutoff_idx:] / equity.iloc[cutoff_idx]
    holdout_years = (equity.index[-1] - cutoff).days / 365.25
    holdout_mdd = float((holdout_equity / holdout_equity.cummax() - 1.0).min())
    growth = float(holdout_equity.iloc[-1] / holdout_equity.iloc[0])
    holdout_cagr_sign = growth ** (1.0 / holdout_years) - 1.0 if holdout_years > 0 else 0.0

    return HoldoutSegment(
        observation_trades=observation_trades,
        holdout_trades=holdout_trades,
        observation_equity=observation_equity,
        holdout_equity=holdout_equity,
        holdout_years=holdout_years,
        holdout_mdd=holdout_mdd,
        holdout_cagr_sign=holdout_cagr_sign,
    )


def _fold_equity_metrics(equity: pd.Series) -> tuple[float, float] | None:
    if len(equity) < 2 or equity.iloc[-1] <= 0:
        return None
    empty_trades = pd.DataFrame(columns=["entry_bar", "pnl", "return_pct"])
    metrics = compute_metrics(equity / equity.iloc[0], empty_trades)
    return metrics.cagr, metrics.calmar


def compute_fold_distribution(
    result: BacktestResult,
    config: ReliabilityGateConfig = ReliabilityGateConfig(),  # noqa: B008
) -> FoldDistributionResult:
    equity = result.equity
    years_present = set(equity.index.year)
    if len(years_present) < 2:
        raise ValueError(
            f"equity spans fewer than 2 distinct calendar years: {sorted(years_present)}"
        )
    if len(result.trades) == 0:
        return FoldDistributionResult(
            n_folds=0, median_fold_cagr=0.0, worst_fold_cagr=0.0,
            median_fold_calmar=0.0, max_period_contribution=0.0, gate_pass=True,
        )

    year_log_returns = _year_log_return_contributions(equity)
    net_total = abs(sum(year_log_returns.values()))
    max_period_contribution = (
        max(abs(v) for v in year_log_returns.values()) / net_total if net_total > 0 else 0.0
    )
    gate_pass = max_period_contribution <= config.max_period_contribution

    cagrs: list[float] = []
    calmars: list[float] = []
    for year in sorted(years_present):
        segment = equity[equity.index.year == year]
        fold_metrics = _fold_equity_metrics(segment)
        if fold_metrics is not None:
            cagrs.append(fold_metrics[0])
            calmars.append(fold_metrics[1])

    median_fold_cagr = float(np.median(cagrs)) if cagrs else 0.0
    worst_fold_cagr = float(np.min(cagrs)) if cagrs else 0.0
    median_fold_calmar = float(np.median(calmars)) if calmars else 0.0
    n_folds = len(cagrs)

    result_out = FoldDistributionResult(
        n_folds=n_folds, median_fold_cagr=median_fold_cagr,
        worst_fold_cagr=worst_fold_cagr, median_fold_calmar=median_fold_calmar,
        max_period_contribution=max_period_contribution, gate_pass=gate_pass,
    )
    _logger.info(
        "n_folds=%d max_period_contribution=%.4f gate_pass=%s median_fold_cagr=%.4f "
        "worst_fold_cagr=%.4f median_fold_calmar=%.4f",
        n_folds, max_period_contribution, gate_pass, median_fold_cagr,
        worst_fold_cagr, median_fold_calmar, extra={"tag": "EVAL"},
    )
    return result_out


def compute_stress_test_gate(
    df: pd.DataFrame,
    spec: StrategySpec,
    costs: CostModel,
    config: ReliabilityGateConfig = ReliabilityGateConfig(),  # noqa: B008
    *,
    cost_fee_mult: float = 1.5,
    cost_slip_mult: float = 2.0,
    delay_bars: int = 1,
    funding_rates: pd.Series | None = None,
) -> ReliabilityGateResult:
    stressed_costs = CostModel(
        fee_rate=costs.fee_rate * cost_fee_mult,
        slippage_rate=costs.slippage_rate * cost_slip_mult,
    )
    stressed_result = run_backtest(
        df, spec, stressed_costs, signal_delay_bars=delay_bars, funding_rates=funding_rates,
    )
    gate_config = dataclasses.replace(config, hurdle_rate=0.0)
    return compute_equity_reliability_gate(
        stressed_result.equity, len(stressed_result.trades), config=gate_config,
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen contract surface."""
    config = ReliabilityGateConfig()
    assert (config.hurdle_rate, config.block_size, config.n_bootstrap, config.seed,
            config.min_trades, config.mdd_floor, config.t_stat_floor,
            config.max_period_contribution) == (0.15, None, 3000, 0, 30, -0.25, 2.0, 0.40)
    assert {f.name for f in fields(ReliabilityGateResult)} == {
        "lcb90_cagr", "lcb95_cagr", "p_negative", "point_cagr",
        "t_stat", "trade_count", "block_size_used", "verdict",
    }
    assert {f.name for f in fields(FoldDistributionResult)} == {
        "n_folds", "median_fold_cagr", "worst_fold_cagr",
        "median_fold_calmar", "max_period_contribution", "gate_pass",
    }
    assert compute_equity_reliability_gate.__name__ == "compute_equity_reliability_gate"
    assert compute_portfolio_reliability_gate.__name__ == "compute_portfolio_reliability_gate"


_check_contract()
