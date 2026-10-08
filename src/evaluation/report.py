"""Decision-grade statistics of one daily strategy return path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.bootstrap import (
    iter_stationary_bootstrap_index_chunks,
    stationary_bootstrap_max_blocks,
)
from src.core.params import REPORT_BOOTSTRAP_PATHS, REPORT_BOOTSTRAP_SEED
from src.quant.evaluation.reliability import derive_block_size


@dataclass(frozen=True, slots=True)
class YearStatistics:
    """One calendar year of a daily return path."""

    year: int
    days: int
    total_return: float
    log_growth: float
    sharpe: float
    max_drawdown: float
    funding_share: float


@dataclass(frozen=True, slots=True)
class BootstrapStatistics:
    """Same-length stationary block-bootstrap distribution of CAGR and drawdown."""

    horizon_days: int
    n_paths: int
    mean_block_days: int
    seed: int
    cagr_q05: float
    cagr_q50: float
    cagr_q95: float
    max_drawdown_q50: float
    max_drawdown_q95: float


@dataclass(frozen=True, slots=True)
class FundingConcentration:
    """Funding income concentration of one return path.

    Charge convention matches the ledger: per-symbol values are funding charge
    divided by ``initial_equity`` (negative = income). ``top5_symbols`` is
    ordered by income (most negative first). Shares are income shares in
    ``[0, 1]`` using gross receiving-symbol income (0 when no symbol received income).
    """

    total_contribution: float
    top5_symbols: tuple[tuple[str, float], ...]
    top5_share: float
    max_day_share: float
    max_day: pd.Timestamp


@dataclass(frozen=True, slots=True)
class StrategyStatistics:
    """Decision-grade statistics of one daily return path."""

    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    skew: float
    excess_kurtosis: float
    years: tuple[YearStatistics, ...]
    bootstrap: BootstrapStatistics
    funding: FundingConcentration
    in_sample_days: int
    out_of_sample_days: int


def _annualized_sharpe(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(values.mean() / std * np.sqrt(365.0))


def _sortino(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    std = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(values.mean() / std * np.sqrt(365.0))


def _cagr(values: np.ndarray) -> float:
    growth = float(np.prod(1.0 + values))
    return float(growth ** (365.0 / values.size) - 1.0)


def _max_drawdown(values: np.ndarray) -> float:
    path = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peak = np.maximum.accumulate(path)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (peak - path) / peak, 0.0)
    return float(drawdown.max())


def _skew(values: np.ndarray) -> float:
    if values.size < 3:
        return float("nan")
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(pd.Series(values).skew())


def _excess_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return float("nan")
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float(pd.Series(values).kurt())


def _require_returns(daily_returns: pd.Series) -> np.ndarray:
    if not isinstance(daily_returns, pd.Series) or daily_returns.empty:
        raise DataIntegrityError("daily_returns must be a non-empty Series")
    raw: Any = daily_returns.to_numpy(dtype="float64")
    values = np.asarray(raw, dtype="float64")
    if not bool(np.isfinite(values).all()):
        raise DataIntegrityError("daily_returns must be finite (NaN/inf fail closed)")
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        raise DataIntegrityError("daily_returns must carry a DatetimeIndex")
    index = daily_returns.index
    if (
        index.tz is None
        or str(index.tz) != "UTC"
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise DataIntegrityError("daily_returns must have a unique increasing UTC index")
    if bool((values < -1.0).any()):
        raise DataIntegrityError("daily_returns cannot fall below -100%")
    return values


def strategy_statistics(
    daily_returns: pd.Series,
    *,
    daily_funding_share: pd.Series,
    funding_by_symbol: Mapping[str, float],
    initial_equity: float,
    design_data_cutoff: pd.Timestamp,
    seed: int = REPORT_BOOTSTRAP_SEED,
    n_paths: int = REPORT_BOOTSTRAP_PATHS,
) -> StrategyStatistics:
    """Compute decision-grade statistics of one daily strategy return path."""
    values = _require_returns(daily_returns)
    if not isinstance(daily_funding_share, pd.Series) or len(daily_funding_share) != len(daily_returns):
        raise DataIntegrityError("daily_funding_share must align one-to-one with daily_returns")
    if not daily_funding_share.index.equals(daily_returns.index):
        raise DataIntegrityError("daily_funding_share must share the daily_returns index")
    funding_share = daily_funding_share.to_numpy(dtype="float64")
    if not bool(np.isfinite(funding_share).all()):
        raise DataIntegrityError("daily_funding_share must be finite")
    if not isinstance(funding_by_symbol, Mapping):
        raise DataIntegrityError("funding_by_symbol must be a mapping")
    per_symbol = {str(sym): float(val) for sym, val in funding_by_symbol.items()}
    for val in per_symbol.values():
        if not np.isfinite(val):
            raise DataIntegrityError("funding_by_symbol values must be finite")
    if isinstance(initial_equity, bool) or not np.isfinite(float(initial_equity)) or not float(initial_equity) > 0.0:
        raise DataIntegrityError("initial_equity must be a positive finite capital")
    equity0 = float(initial_equity)
    if not isinstance(design_data_cutoff, pd.Timestamp) or pd.isna(design_data_cutoff):
        raise DataIntegrityError("design_data_cutoff must be a valid timestamp")
    if design_data_cutoff.tzinfo is None or design_data_cutoff.utcoffset().total_seconds() != 0:
        raise DataIntegrityError("design_data_cutoff must be timezone-aware UTC")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise DataIntegrityError(f"seed must be an integer, got {seed!r}")
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or int(n_paths) < 1:
        raise DataIntegrityError(f"n_paths must be a positive integer, got {n_paths!r}")
    n_paths = int(n_paths)
    n = int(values.size)

    equity = equity0 * np.cumprod(1.0 + values)
    start_equity = np.empty(n, dtype="float64")
    start_equity[0] = equity0
    start_equity[1:] = equity[:-1]
    funding_income_day = -funding_share * equity0

    years: list[YearStatistics] = []
    for year, group in daily_returns.groupby(daily_returns.index.year):
        arr = group.to_numpy(dtype="float64")
        positions = daily_returns.index.get_indexer_for(group.index)
        year_start_equity = float(start_equity[int(positions[0])])
        year_income = float(funding_income_day[positions].sum())
        years.append(
            YearStatistics(
                year=int(year),
                days=int(arr.size),
                total_return=float(np.prod(1.0 + arr) - 1.0),
                log_growth=float("-inf") if bool((arr == -1).any()) else float(np.log1p(arr).sum()),
                sharpe=_annualized_sharpe(arr),
                max_drawdown=_max_drawdown(arr),
                funding_share=float(year_income / year_start_equity) if year_start_equity > 0 else float("nan"),
            )
        )
    years.sort(key=lambda y: y.year)

    mean_block = max(1, int(derive_block_size(values)))
    max_blocks = stationary_bootstrap_max_blocks(n, mean_block)
    rng = np.random.default_rng(int(seed))
    cagrs = np.empty(n_paths, dtype="float64")
    mdds = np.empty(n_paths, dtype="float64")
    filled = 0
    chunk_size = min(n_paths, 500)
    for chunk in iter_stationary_bootstrap_index_chunks(
        rng,
        source_len=n,
        path_len=n,
        n_replicates=n_paths,
        mean_block=mean_block,
        chunk_size=chunk_size,
        max_blocks=max_blocks,
    ):
        paths = values[chunk.indices]
        for row in range(paths.shape[0]):
            rep = paths[row]
            cagrs[filled] = _cagr(rep)
            mdds[filled] = _max_drawdown(rep)
            filled += 1
    bootstrap = BootstrapStatistics(
        horizon_days=n,
        n_paths=n_paths,
        mean_block_days=mean_block,
        seed=int(seed),
        cagr_q05=float(np.quantile(cagrs, 0.05)),
        cagr_q50=float(np.quantile(cagrs, 0.50)),
        cagr_q95=float(np.quantile(cagrs, 0.95)),
        max_drawdown_q50=float(np.quantile(mdds, 0.50)),
        max_drawdown_q95=float(np.quantile(mdds, 0.95)),
    )

    contributions = {sym: val / equity0 for sym, val in per_symbol.items()}
    total_contribution = float(sum(contributions.values()))
    total_income = float(sum(max(-val, 0.0) for val in per_symbol.values()))
    ranked = sorted(((sym, val) for sym, val in per_symbol.items() if val < 0), key=lambda kv: (kv[1], kv[0]))
    top = tuple((sym, float(val / equity0)) for sym, val in ranked[:5])
    if total_income > 0:
        top_income = float(sum(-val for _, val in ranked[:5]))
        top_share = float(top_income / total_income)
    else:
        top_share = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        day_share = np.where(start_equity > 0, funding_income_day / start_equity, 0.0)
    best_pos = int(np.argmax(day_share))
    funding = FundingConcentration(
        total_contribution=total_contribution,
        top5_symbols=top,
        top5_share=float(min(1.0, max(0.0, top_share))),
        max_day_share=float(day_share[best_pos]),
        max_day=pd.Timestamp(daily_returns.index[best_pos]).tz_convert("UTC"),
    )

    cutoff = pd.Timestamp(design_data_cutoff).tz_convert("UTC")
    in_sample = int((daily_returns.index <= cutoff).sum())
    return StrategyStatistics(
        sharpe=_annualized_sharpe(values),
        sortino=_sortino(values),
        cagr=_cagr(values),
        max_drawdown=_max_drawdown(values),
        skew=_skew(values),
        excess_kurtosis=_excess_kurtosis(values),
        years=tuple(years),
        bootstrap=bootstrap,
        funding=funding,
        in_sample_days=in_sample,
        out_of_sample_days=n - in_sample,
    )


def _json_float(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def statistics_payload(stats: StrategyStatistics) -> dict[str, Any]:
    """JSON-safe mapping of one ``StrategyStatistics`` (timestamps ISO-8601)."""
    return {
        "sharpe": _json_float(stats.sharpe),
        "sortino": _json_float(stats.sortino),
        "cagr": _json_float(stats.cagr),
        "max_drawdown": _json_float(stats.max_drawdown),
        "skew": _json_float(stats.skew),
        "excess_kurtosis": _json_float(stats.excess_kurtosis),
        "in_sample_days": stats.in_sample_days,
        "out_of_sample_days": stats.out_of_sample_days,
        "years": [
            {
                "year": y.year,
                "days": y.days,
                "total_return": _json_float(y.total_return),
                "log_growth": _json_float(y.log_growth),
                "sharpe": _json_float(y.sharpe),
                "max_drawdown": _json_float(y.max_drawdown),
                "funding_share": _json_float(y.funding_share),
            }
            for y in stats.years
        ],
        "bootstrap": {
            "horizon_days": stats.bootstrap.horizon_days,
            "n_paths": stats.bootstrap.n_paths,
            "mean_block_days": stats.bootstrap.mean_block_days,
            "seed": stats.bootstrap.seed,
            "cagr_q05": _json_float(stats.bootstrap.cagr_q05),
            "cagr_q50": _json_float(stats.bootstrap.cagr_q50),
            "cagr_q95": _json_float(stats.bootstrap.cagr_q95),
            "max_drawdown_q50": _json_float(stats.bootstrap.max_drawdown_q50),
            "max_drawdown_q95": _json_float(stats.bootstrap.max_drawdown_q95),
        },
        "funding": {
            "total_contribution": _json_float(stats.funding.total_contribution),
            "top5_symbols": [[sym, _json_float(val)] for sym, val in stats.funding.top5_symbols],
            "top5_share": _json_float(stats.funding.top5_share),
            "max_day_share": _json_float(stats.funding.max_day_share),
            "max_day": stats.funding.max_day.isoformat(),
        },
    }
