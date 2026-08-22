from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from src.research.evaluation.reliability import derive_block_size


@dataclass(frozen=True, slots=True)
class GrowthSizingConfig:
    """Immutable parameters for constraint-first growth-optimal risk selection.

    ``bars_per_year`` defaults to 2190 as the 4h calendar invariant ``6 * 365``,
    not a fitted constant.
    """

    risk_grid: tuple[float, ...]
    reference_risk: float = 0.005
    max_drawdown: float = 0.20
    max_drawdown_prob: float = 0.05
    ruin_fraction: float = 0.50
    max_ruin_prob: float = 0.001
    horizon_years: float = 5.0
    n_paths: int = 2000
    seed: int = 0
    plateau_fraction: float = 0.95
    bars_per_year: int = 2190

    def __post_init__(self) -> None:
        if not self.risk_grid:
            raise ValueError("risk_grid must not be empty")
        if len(self.risk_grid) > 1 and not all(
            self.risk_grid[i] < self.risk_grid[i + 1] for i in range(len(self.risk_grid) - 1)
        ):
            raise ValueError("risk_grid must be strictly ascending")
        if self.reference_risk <= 0:
            raise ValueError(f"reference_risk must be > 0, got {self.reference_risk}")
        if self.n_paths < 100:
            raise ValueError(f"n_paths must be >= 100, got {self.n_paths}")
        if not 0 < self.max_drawdown_prob <= 1:
            raise ValueError(f"max_drawdown_prob must be in (0, 1], got {self.max_drawdown_prob}")
        if not 0 < self.max_ruin_prob < 1:
            raise ValueError(f"max_ruin_prob must be in (0, 1), got {self.max_ruin_prob}")


@dataclass(frozen=True, slots=True)
class GrowthSizingResult:
    selected_risk: float | None
    median_log_growth: float
    mdd_breach_prob: float
    ruin_prob: float
    feasible_risks: tuple[float, ...]
    binding_constraint: str
    block_size_used: int


@dataclass(frozen=True, slots=True)
class GrowthHeadroomDiagnostic:
    """Observability-only headroom report, mirroring ``block_size_search_hit_cap``.

    Records where the selected risk sits relative to the best feasible point of
    the *tested* grid and whether any higher-risk point is walled off by tail
    risk instead of merely sitting off the 95% plateau. This diagnostic passes
    or fails nothing -- it never re-selects a risk and never feeds back into the
    ``GrowthSizingResult`` it describes.
    """

    selected_risk: float | None
    selected_median_log_growth: float
    peak_feasible_risk: float | None
    peak_feasible_median_log_growth: float
    headroom_ratio: float
    risk_constrained: bool
    block_size_used: int


def drawdown_risk_multiplier(drawdown: np.ndarray) -> np.ndarray:
    """Vectorized piecewise de-risk ladder on a positive drawdown fraction.

    ``1.0`` up to 5%, a linear taper to ``0.25`` at 15%, a final taper to zero at
    20%, and ``0.0`` at or beyond 20%.  Implemented with ``np.select``; the
    overlay only ever reduces exposure.
    """
    dd = np.asarray(drawdown, dtype=np.float64)
    if dd.size and (np.any(dd < 0) or not np.isfinite(dd).all()):
        raise ValueError("drawdown must contain only finite non-negative values")
    out = np.select(
        [dd <= 0.05, dd <= 0.15, dd < 0.20],
        [
            np.ones_like(dd),
            1.0 - 0.75 * (dd - 0.05) / 0.10,
            0.25 * (0.20 - dd) / 0.05,
        ],
        default=0.0,
    )
    return np.asarray(out, dtype=np.float64)


def _block_bootstrap_paths(
    unit_returns: np.ndarray,
    *,
    n_paths: int,
    path_len: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    n = len(unit_returns)
    n_blocks = int(np.ceil(path_len / block_size))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(n_paths, n_blocks * block_size)[:, :path_len]
    out: np.ndarray = unit_returns[idx]
    return out


def _simulate_with_drawdown_overlay(scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Path-dependent overlay: per-bar risk is scaled by the running drawdown.

    The only permitted Python loop is over bars because the overlay is path
    dependent; all path dimensions stay vectorized.
    """
    n_paths, n_bars = scaled.shape
    equity = np.ones(n_paths)
    peak = np.ones(n_paths)
    mdd = np.zeros(n_paths)
    for b in range(n_bars):
        multiplier = drawdown_risk_multiplier(mdd)
        equity = equity * (1.0 + scaled[:, b] * multiplier)
        peak = np.maximum(peak, equity)
        mdd = np.maximum(mdd, 1.0 - equity / peak)
    return equity, mdd


def solve_growth_optimal_risk(
    unit_returns: np.ndarray,
    config: GrowthSizingConfig,
    *,
    use_drawdown_overlay: bool = True,
) -> GrowthSizingResult:
    """Select the growth-optimal per-bar risk, constraints before the plateau rule.

    A stationary block bootstrap (block length from the reused
    ``derive_block_size``) draws ``config.n_paths`` paths of
    ``horizon_years * bars_per_year`` bars, seeded by ``config.seed`` for exact
    reproducibility.  The feasible set is defined FIRST by the two constraints
    ``P(MDD > max_drawdown) <= max_drawdown_prob`` and
    ``P(final < ruin_fraction) <= max_ruin_prob``; only then is the lowest
    feasible grid risk whose median log growth reaches
    ``plateau_fraction`` of the feasible maximum selected.  Applying the plateau
    rule before the constraints is the defect this function exists to prevent.
    """
    arr = np.asarray(unit_returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("unit_returns must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("unit_returns must contain only finite values")

    path_len = round(config.horizon_years * config.bars_per_year)
    if path_len < 1:
        raise ValueError("horizon_years * bars_per_year must be >= 1")

    block_size = derive_block_size(arr)
    paths = _block_bootstrap_paths(
        arr,
        n_paths=config.n_paths,
        path_len=path_len,
        block_size=block_size,
        seed=config.seed,
    )

    feasible: list[tuple[float, float, float, float]] = []
    best_median_g = -np.inf
    for risk in config.risk_grid:
        scale = risk / config.reference_risk
        scaled = paths * scale
        if use_drawdown_overlay:
            finals, mdd = _simulate_with_drawdown_overlay(scaled)
        else:
            cum = np.cumprod(1.0 + scaled, axis=1)
            finals = cum[:, -1]
            mdd = (1.0 - cum / np.maximum.accumulate(cum, axis=1)).max(axis=1)
        mdd_breach_prob = float(np.mean(mdd > config.max_drawdown))
        ruin_prob = float(np.mean(finals < config.ruin_fraction))
        median_g = float(np.median(np.log(np.maximum(finals, 1e-12))))
        if mdd_breach_prob <= config.max_drawdown_prob and ruin_prob <= config.max_ruin_prob:
            feasible.append((risk, median_g, mdd_breach_prob, ruin_prob))
            best_median_g = max(best_median_g, median_g)

    if not feasible:
        return GrowthSizingResult(
            None, 0.0, 0.0, 0.0, (), "infeasible", block_size,
        )

    plateau_target = config.plateau_fraction * best_median_g
    candidates = [item for item in feasible if item[1] >= plateau_target]
    if not candidates:
        return GrowthSizingResult(
            None, 0.0, 0.0, 0.0, tuple(item[0] for item in feasible),
            "infeasible", block_size,
        )
    selected = min(candidates, key=lambda item: item[0])
    return GrowthSizingResult(
        selected_risk=selected[0],
        median_log_growth=selected[1],
        mdd_breach_prob=selected[2],
        ruin_prob=selected[3],
        feasible_risks=tuple(item[0] for item in feasible),
        binding_constraint="none",
        block_size_used=block_size,
    )


def diagnose_growth_headroom(
    unit_returns: np.ndarray,
    config: GrowthSizingConfig,
    selected: GrowthSizingResult,
    *,
    use_drawdown_overlay: bool = True,
) -> GrowthHeadroomDiagnostic:
    """Report whether leverage is exhausted and by which constraint.

    Purely observational: independently re-runs the same risk-grid feasibility
    loop :func:`solve_growth_optimal_risk` runs (same block-bootstrap draw,
    same per-grid-point median/mdd/ruin computation, same overlay branch --
    deliberate duplication so the frozen solver contract is never touched) and
    reports, for grid points strictly above the selected risk, the best
    feasible median log growth vs. the selected point and whether any higher
    point that *would* beat the running peak is blocked by tail risk rather
    than by the 95% plateau rule. Like ``block_size_search_hit_cap`` in
    ``reliability.py`` this is a pure observability flag: it passes or fails
    nothing, never re-selects, and never mutates ``selected``.

    When ``selected.selected_risk`` is ``None`` (infeasible at every grid
    point) the bootstrap is skipped and a degenerate diagnostic is returned.
    """
    arr = np.asarray(unit_returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("unit_returns must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("unit_returns must contain only finite values")

    if selected.selected_risk is None:
        return GrowthHeadroomDiagnostic(
            None, 0.0, None, 0.0, 0.0, False, selected.block_size_used,
        )

    path_len = round(config.horizon_years * config.bars_per_year)
    block_size = derive_block_size(arr)
    paths = _block_bootstrap_paths(
        arr,
        n_paths=config.n_paths,
        path_len=path_len,
        block_size=block_size,
        seed=config.seed,
    )

    peak_risk: float | None = selected.selected_risk
    peak_median_g = selected.median_log_growth
    risk_constrained = False
    for risk in config.risk_grid:
        if risk <= selected.selected_risk:
            continue
        scale = risk / config.reference_risk
        scaled = paths * scale
        if use_drawdown_overlay:
            finals, mdd = _simulate_with_drawdown_overlay(scaled)
        else:
            cum = np.cumprod(1.0 + scaled, axis=1)
            finals = cum[:, -1]
            mdd = (1.0 - cum / np.maximum.accumulate(cum, axis=1)).max(axis=1)
        mdd_breach_prob = float(np.mean(mdd > config.max_drawdown))
        ruin_prob = float(np.mean(finals < config.ruin_fraction))
        median_g = float(np.median(np.log(np.maximum(finals, 1e-12))))
        if mdd_breach_prob <= config.max_drawdown_prob and ruin_prob <= config.max_ruin_prob:
            if median_g > peak_median_g:
                peak_median_g = median_g
                peak_risk = risk
        elif median_g > peak_median_g:
            risk_constrained = True

    headroom_ratio = (
        peak_median_g / selected.median_log_growth - 1.0
        if selected.median_log_growth > 0.0 else 0.0
    )
    return GrowthHeadroomDiagnostic(
        selected_risk=selected.selected_risk,
        selected_median_log_growth=selected.median_log_growth,
        peak_feasible_risk=peak_risk,
        peak_feasible_median_log_growth=peak_median_g,
        headroom_ratio=headroom_ratio,
        risk_constrained=risk_constrained,
        block_size_used=block_size,
    )


def apply_realised_risk_overlay(
    net: pd.Series,
    weights: pd.DataFrame,
    selected_risk: float,
    reference_risk: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Deterministic realised-risk overlay on net returns and realised weights.

    Every bar is scaled by ``selected_risk / reference_risk`` and by the causal
    drawdown multiplier ``drawdown_risk_multiplier(drawdown of the deployed
    equity through the preceding bar)``; the first multiplier is exactly one.
    The overlay is applied to BOTH the net return series and the realised
    weights, so ``selected_risk`` changes the published equity and reported
    weights by the defined scale.  The drawdown ladder only ever reduces
    exposure; an invalid, non-finite, or non-shared-index input fails closed
    with ``ValueError`` instead of silently mutating the ledger.
    """
    if not isinstance(net.index, pd.DatetimeIndex) or not isinstance(weights.index, pd.DatetimeIndex):
        raise ValueError("net and weights must have a DatetimeIndex")
    if not net.index.equals(weights.index):
        raise ValueError("net and weights must share an identical index")
    if not net.index.is_monotonic_increasing:
        raise ValueError("net index must be monotonic increasing")
    values = net.to_numpy(dtype=np.float64)
    w_arr = weights.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(w_arr).all():
        raise ValueError("net and weights must contain only finite values")
    if selected_risk <= 0 or reference_risk <= 0:
        raise ValueError("selected_risk and reference_risk must be > 0")

    scale = selected_risk / reference_risk
    n = len(values)
    scaled_net = np.empty(n, dtype=np.float64)
    scaled_w = np.empty_like(w_arr)
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for t in range(n):
        multiplier = drawdown_risk_multiplier(np.asarray([mdd], dtype=np.float64))[0]
        factor = scale * multiplier
        scaled_net[t] = factor * values[t]
        scaled_w[t] = factor * w_arr[t]
        equity *= 1.0 + scaled_net[t]
        peak = max(peak, equity)
        mdd = max(mdd, 1.0 - equity / peak)
    return (
        pd.Series(scaled_net, index=net.index, dtype=np.float64),
        pd.DataFrame(scaled_w, index=weights.index, columns=weights.columns, dtype=np.float64),
    )


def compute_discovery_target_vol(discovery_net: pd.Series, window: int) -> float:
    """Frozen vol-target anchor: median causal trailing realised vol.

    Returns ``discovery_net.rolling(window, min_periods=window).std().
    shift(1).dropna().median()`` -- the median trailing vol over strictly
    prior bars only, so the bar being scaled never enters its own estimate.
    The return value is the single frozen constant callers must apply over
    the full deployed history; it must never be re-fit on qualification or
    holdout data.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    values = discovery_net.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("discovery_net must contain only finite values")
    trailing = discovery_net.rolling(window, min_periods=window).std().shift(1).dropna()
    if trailing.empty:
        raise ValueError(
            "discovery_net must have at least window + 1 finite bars "
            "to compute a median trailing vol"
        )
    return float(trailing.median())


def apply_vol_target_overlay(
    net: pd.Series,
    weights: pd.DataFrame,
    window: int,
    target_vol: float,
    multiplier_bounds: tuple[float, float],
) -> tuple[pd.Series, pd.DataFrame]:
    """Proactive causal trailing-vol targeting overlay on net and weights.

    Every bar ``t`` is scaled by ``clip(target_vol / trailing_vol_t,
    multiplier_bounds)`` where ``trailing_vol_t`` is the rolling std over the
    strictly-prior ``window`` bars (``net.rolling(window,
    min_periods=window).std().shift(1)``). Where history is insufficient or
    the trailing vol is zero/non-finite the multiplier falls back to ``1.0``
    (never CASH, never NaN, never a divide-by-zero), the same
    fail-closed-to-neutral convention used by
    ``_causal_family_inverse_vol_weights`` in ``cross_sectional.py``. Both
    ``net`` and ``weights`` are scaled by the identical per-bar multiplier.
    """
    if not isinstance(net.index, pd.DatetimeIndex) or not isinstance(weights.index, pd.DatetimeIndex):
        raise ValueError("net and weights must have a DatetimeIndex")
    if not net.index.equals(weights.index):
        raise ValueError("net and weights must share an identical index")
    if not net.index.is_monotonic_increasing:
        raise ValueError("net index must be monotonic increasing")
    values = net.to_numpy(dtype=np.float64)
    w_arr = weights.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(w_arr).all():
        raise ValueError("net and weights must contain only finite values")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if not np.isfinite(target_vol) or target_vol <= 0:
        raise ValueError("target_vol must be finite and > 0")
    lo, hi = multiplier_bounds
    if not (0.0 < lo < hi and np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("multiplier_bounds must be a strictly-ascending positive pair")

    trailing = net.rolling(window, min_periods=window).std().shift(1)
    trailing_arr = trailing.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        multiplier = target_vol / trailing_arr
    invalid = ~np.isfinite(trailing_arr) | (trailing_arr <= 0.0)
    multiplier = np.where(invalid, 1.0, np.clip(multiplier, lo, hi))
    scaled_net = multiplier * values
    scaled_weights = multiplier[:, None] * w_arr
    return (
        pd.Series(scaled_net, index=net.index, dtype=np.float64),
        pd.DataFrame(scaled_weights, index=weights.index, columns=weights.columns, dtype=np.float64),
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen growth-sizing contract surface."""
    config = GrowthSizingConfig(risk_grid=(0.0005, 0.001, 0.005))
    assert config.bars_per_year == 2190
    assert config.max_drawdown == 0.20
    assert config.plateau_fraction == 0.95
    assert {f.name for f in fields(GrowthSizingConfig)} == {
        "risk_grid", "reference_risk", "max_drawdown", "max_drawdown_prob",
        "ruin_fraction", "max_ruin_prob", "horizon_years", "n_paths", "seed",
        "plateau_fraction", "bars_per_year",
    }
    assert {f.name for f in fields(GrowthSizingResult)} == {
        "selected_risk", "median_log_growth", "mdd_breach_prob", "ruin_prob",
        "feasible_risks", "binding_constraint", "block_size_used",
    }
    assert {f.name for f in fields(GrowthHeadroomDiagnostic)} == {
        "selected_risk", "selected_median_log_growth", "peak_feasible_risk",
        "peak_feasible_median_log_growth", "headroom_ratio", "risk_constrained",
        "block_size_used",
    }
    dd = np.array([0.0, 0.05, 0.10, 0.15, 0.175, 0.20, 0.30])
    assert np.allclose(drawdown_risk_multiplier(dd), np.array([1.0, 1.0, 0.625, 0.25, 0.125, 0.0, 0.0]))
    assert apply_realised_risk_overlay.__name__ == "apply_realised_risk_overlay"


_check_contract()
