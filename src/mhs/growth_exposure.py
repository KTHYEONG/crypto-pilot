"""Growth-optimal exposure for a dollar-neutral book under selection deflation and empirical single-name gap risk."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.mhs.source_gaps import SourceGapPlane, active_intervals

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GapSample:
    """Empirical single-name gap magnitudes and their yearly frequency among roster members."""

    magnitudes: np.ndarray
    events_per_year: float


@dataclass(frozen=True, slots=True)
class LogGrowthExposureSolution:
    """Stressed log-growth curve over the exposure grid and the selected rung."""

    grid: tuple[float, ...]
    growth: tuple[float, ...]
    ruin_probability: tuple[float, ...]
    argmax: float
    chosen: float
    gap_events_per_year: float
    gap_sample_size: int


def structurally_excluded_symbols(*, plane: SourceGapPlane = "ohlcv_3m", path: Path | None = None) -> frozenset[str]:
    """Symbols the ledger can never trade because their exit is unevidencable.

    A registry reason of ``DELISTED`` means no forced exit can ever be priced with a real
    settlement, so the strategy has excluded the symbol from trading for its whole life
    (see ``src/mhs/policy/source_gaps.jsonl``). That symbol's realized price history is the
    only source of genuine single-name gap risk that is structurally absent from every
    historical ledger return series, because the ledger has never held it and never will.
    Every other symbol's crashes, however large, already occurred inside the ledger's own
    realized returns and must never be re-added on top of them as a separate gap stress.

    Args:
        plane: Source-gap plane to query (the execution plane governs tradability).
        path: Registry override forwarded to the loader.
    Returns:
        Symbols with an unresolved ``DELISTED`` record on the given plane.
    """
    return frozenset(iv.symbol for iv in active_intervals(plane=plane, path=path) if iv.reason == "DELISTED")


def roster_gap_sample(daily_close: pd.DataFrame, roster: pd.DataFrame, *, threshold: float) -> GapSample:
    """Collect absolute next-day moves of rostered names that reach ``threshold``.

    Either direction is adverse to one leg of a dollar-neutral book, so magnitudes are absolute.
    The roster passed here must not apply trading exclusions: a name the ledger cannot settle
    (e.g. delisted without a published price) still revealed how far a rostered name can move.

    Args:
        daily_close: UTC-daily closes, census columns.
        roster: Boolean decision-day roster aligned to ``daily_close``.
        threshold: Minimum absolute move counted as a gap, in (0, 1].
    Returns:
        Gap magnitudes and events per year over the roster's covered span.
    Raises:
        ValueError: Misaligned inputs, non-positive closes, or threshold outside (0, 1].
    """
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.floating))
        or not np.isfinite(float(threshold))
        or not 0.0 < float(threshold) <= 1.0
    ):
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    if (
        not isinstance(daily_close, pd.DataFrame)
        or not isinstance(roster, pd.DataFrame)
        or daily_close.empty
        or not roster.index.equals(daily_close.index)
        or not roster.columns.equals(daily_close.columns)
    ):
        raise ValueError("roster must be a boolean frame aligned to daily_close index and columns")
    closes = daily_close.to_numpy(dtype="float64")
    observed = ~np.isnan(closes)
    # 미상장·상폐 이후 NaN은 정상 census 상태이며, 관측된 가격만 유한 양수여야 한다.
    if not bool((np.isfinite(closes[observed]) & (closes[observed] > 0.0)).all()):
        raise ValueError("daily_close must hold finite positive prices where observed")
    mask = roster.to_numpy(dtype=bool)[:-1] & observed[:-1] & observed[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        moves = np.abs(closes[1:] / closes[:-1] - 1.0)[mask]
    magnitudes = np.sort(moves[moves >= float(threshold)])
    covered_years = len(daily_close) / 365.0
    return GapSample(magnitudes=magnitudes, events_per_year=float(magnitudes.size / covered_years))


def _stationary_bootstrap_matrix(
    source: np.ndarray, n_paths: int, horizon_days: int, mean_block_days: int, rng: np.random.Generator,
) -> np.ndarray:
    """Draw one stationary block-bootstrap index matrix reused for every rung."""
    n = int(source.size)
    prob = 1.0 / float(mean_block_days)
    lengths = rng.geometric(prob, size=(n_paths, horizon_days))
    starts = rng.integers(0, n, size=(n_paths, horizon_days))
    ends = np.cumsum(lengths, axis=1)
    ends_trunc = np.minimum(ends, horizon_days)
    used = ends_trunc - np.concatenate([np.zeros((n_paths, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1)
    flat_used = used.ravel()
    flat_starts = starts.ravel()
    keep = flat_used > 0
    flat_used = flat_used[keep]
    flat_starts = flat_starts[keep]
    block_start = np.cumsum(flat_used) - flat_used
    offsets = np.arange(n_paths * horizon_days, dtype=np.int64) - np.repeat(block_start, flat_used)
    indices = (np.repeat(flat_starts, flat_used) + offsets) % n
    return cast(np.ndarray, source[indices].reshape(n_paths, horizon_days))


def solve_log_growth_exposure(
    unit_returns: pd.Series,
    *,
    max_name_weight: float,
    gaps: GapSample,
    mean_haircut: float,
    grid: tuple[float, ...],
    plateau_tolerance: float,
    n_paths: int,
    horizon_years: float,
    mean_block_days: int,
    seed: int,
) -> LogGrowthExposureSolution:
    """Choose the exposure that maximizes stressed expected log growth, preferring the low end of its plateau.

    Unit (x1) daily returns are resampled with a stationary block bootstrap to keep volatility
    clustering, shifted down by ``mean_haircut`` times their sample mean to remove selection
    optimism, and hit by single-name gaps drawn from ``gaps`` at its empirical frequency, each
    costing ``rung * max_name_weight * magnitude``. Every rung sees the same draws, so the curve
    reflects exposure alone. A rung with any ruined path (equity <= 0) is inadmissible because
    one ruin sends expected log growth to minus infinity.

    Args:
        unit_returns: Ledger daily net returns of the book at exposure 1.0.
        max_name_weight: Mean daily largest |weight| of the book at exposure 1.0.
        gaps: Empirical gap sample (``roster_gap_sample``).
        mean_haircut: Fraction of the sample mean removed, in [0, 1).
        grid: Strictly increasing positive exposure rungs.
        plateau_tolerance: Relative distance to the best growth still counted as the plateau, in [0, 1).
        n_paths: Bootstrap paths.
        horizon_years: Simulated horizon.
        mean_block_days: Mean stationary-bootstrap block length.
        seed: Generator seed; identical inputs give bit-identical output.
    Returns:
        Curve, ruin probabilities, argmax rung, and chosen rung.
    Raises:
        ValueError: Fewer than 365 finite returns, non-finite inputs, invalid grid or parameters,
            or no admissible rung with positive growth.
    """
    if not isinstance(unit_returns, pd.Series):
        raise ValueError("unit_returns must be a daily return Series")
    finite = unit_returns.to_numpy(dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size < 365:
        raise ValueError(f"unit_returns must hold at least 365 finite rows, got {finite.size}")
    if isinstance(max_name_weight, bool) or not isinstance(max_name_weight, (int, float, np.floating)):
        raise ValueError(f"max_name_weight must be a finite non-negative weight, got {max_name_weight!r}")
    if not np.isfinite(float(max_name_weight)) or float(max_name_weight) < 0.0:
        raise ValueError(f"max_name_weight must be a finite non-negative weight, got {max_name_weight!r}")
    if (
        not isinstance(gaps, GapSample)
        or not isinstance(gaps.magnitudes, np.ndarray)
        or gaps.magnitudes.ndim != 1
        or (gaps.magnitudes.size and not bool(np.isfinite(gaps.magnitudes).all()))
        or (gaps.magnitudes.size and not bool((gaps.magnitudes >= 0.0).all()))
        or not isinstance(gaps.events_per_year, (int, float, np.floating))
        or not np.isfinite(float(gaps.events_per_year))
        or float(gaps.events_per_year) < 0.0
        or (float(gaps.events_per_year) > 0.0 and gaps.magnitudes.size == 0)
    ):
        raise ValueError("gaps must hold finite non-negative magnitudes with a finite non-negative yearly rate")
    if (
        isinstance(mean_haircut, bool)
        or not isinstance(mean_haircut, (int, float, np.floating))
        or not np.isfinite(float(mean_haircut))
        or not 0.0 <= float(mean_haircut) < 1.0
    ):
        raise ValueError(f"mean_haircut must be in [0, 1), got {mean_haircut!r}")
    grid_vals = tuple(float(v) for v in grid) if isinstance(grid, (tuple, list)) else ()
    if (
        not isinstance(grid, (tuple, list))
        or len(grid_vals) < 1
        or not all(np.isfinite(grid_vals))
        or not all(v > 0.0 for v in grid_vals)
        or any(b <= a for a, b in itertools.pairwise(grid_vals))
    ):
        raise ValueError("grid must be strictly increasing positive rungs")
    if (
        isinstance(plateau_tolerance, bool)
        or not isinstance(plateau_tolerance, (int, float, np.floating))
        or not np.isfinite(float(plateau_tolerance))
        or not 0.0 <= float(plateau_tolerance) < 1.0
    ):
        raise ValueError(f"plateau_tolerance must be in [0, 1), got {plateau_tolerance!r}")
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or int(n_paths) < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
    if (
        isinstance(horizon_years, bool)
        or not isinstance(horizon_years, (int, float, np.floating))
        or not np.isfinite(float(horizon_years))
        or float(horizon_years) <= 0.0
    ):
        raise ValueError(f"horizon_years must be positive, got {horizon_years!r}")
    if isinstance(mean_block_days, bool) or not isinstance(mean_block_days, (int, np.integer)) or int(mean_block_days) < 1:
        raise ValueError(f"mean_block_days must be a positive integer, got {mean_block_days!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    horizon_days = round(float(horizon_years) * 365.0)
    if horizon_days < 1:
        raise ValueError(f"horizon_years must span at least one day, got {horizon_years!r}")
    rng = np.random.default_rng(int(seed))
    resampled = _stationary_bootstrap_matrix(finite, int(n_paths), horizon_days, int(mean_block_days), rng)
    stressed_unit = resampled - float(mean_haircut) * float(finite.mean())
    magnitudes = np.asarray(gaps.magnitudes, dtype="float64")
    if float(gaps.events_per_year) <= 0.0 or magnitudes.size == 0 or float(max_name_weight) == 0.0:
        gap_unit_loss = np.zeros((int(n_paths), horizon_days), dtype="float64")
    else:
        gap_hits = rng.random((int(n_paths), horizon_days)) < min(1.0, float(gaps.events_per_year) / 365.0)
        drawn = magnitudes[rng.integers(0, magnitudes.size, size=(int(n_paths), horizon_days))]
        gap_unit_loss = np.where(gap_hits, float(max_name_weight) * drawn, 0.0)
    base = stressed_unit - gap_unit_loss
    growth: list[float] = []
    ruin: list[float] = []
    for rung in grid_vals:
        leg = rung * base
        ruined = leg <= -1.0
        ruined_paths = ruined.any(axis=1)
        ruin.append(float(ruined_paths.mean()))
        if bool(ruined_paths.any()):
            growth.append(float("-inf"))
        else:
            growth.append(float(np.log1p(leg).sum(axis=1).mean() * (365.0 / horizon_days)))
    admissible = [i for i, (g, r) in enumerate(zip(growth, ruin, strict=True)) if r == 0.0 and np.isfinite(g)]
    best_admissible = [i for i in admissible if growth[i] > 0.0]
    if not best_admissible:
        raise ValueError("no admissible rung with positive stressed growth")
    best = max(growth[i] for i in best_admissible)
    argmax = grid_vals[max(i for i in best_admissible if growth[i] == best)]
    plateau = [i for i in best_admissible if growth[i] >= best - float(plateau_tolerance) * abs(best)]
    chosen = grid_vals[min(plateau)]
    _logger.debug(
        "[EVAL] log_growth_exposure chosen=%.2f argmax=%.2f best=%.4f paths=%d horizon_days=%d gap_sample=%d",
        chosen, argmax, best, int(n_paths), horizon_days, magnitudes.size,
    )
    return LogGrowthExposureSolution(
        grid=grid_vals,
        growth=tuple(growth),
        ruin_probability=tuple(ruin),
        argmax=float(argmax),
        chosen=float(chosen),
        gap_events_per_year=float(gaps.events_per_year),
        gap_sample_size=int(magnitudes.size),
    )
