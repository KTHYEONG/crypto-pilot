from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from src.quant.evaluation.reliability import derive_block_size


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


@dataclass(frozen=True, slots=True)
class FrontierScanPoint:
    """Per-candidate leverage multiple frontier observation (diagnostic-only)."""

    multiple: float
    mdd_breach_prob: float
    ruin_prob: float
    feasible: bool


def scan_leverage_frontier(
    unit_returns: np.ndarray,
    config: GrowthSizingConfig,
    candidate_multiples: tuple[float, ...],
) -> tuple[FrontierScanPoint, ...]:
    """Report per-multiple mdd-breach/ruin feasibility over one bootstrap draw.

    Diagnostic sibling of :func:`diagnose_growth_headroom`, following the same
    deliberate-duplication philosophy: the frozen solver is never called here
    because its plateau rule zeroes out any candidate whose median log growth
    is negative, silently hiding the true constraint feasibility a wide
    leverage scan exists to expose. The block-bootstrap draw happens exactly
    once and every candidate multiple reuses those identical paths (same
    comparability principle as the solver's own grid loop). Observability
    only: nothing is selected, persisted, or fed back into sizing.
    """
    arr = np.asarray(unit_returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("unit_returns must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("unit_returns must contain only finite values")
    if len(candidate_multiples) == 0:
        raise ValueError("candidate_multiples must not be empty")
    offending = [m for m in candidate_multiples if not m > 0]
    if offending:
        raise ValueError(
            f"candidate_multiples must contain only positive values, got {offending}"
        )

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

    points: list[FrontierScanPoint] = []
    for multiple in candidate_multiples:
        scaled = paths * multiple
        cum = np.cumprod(1.0 + scaled, axis=1)
        finals = cum[:, -1]
        mdd = (1.0 - cum / np.maximum.accumulate(cum, axis=1)).max(axis=1)
        mdd_breach_prob = float(np.mean(mdd > config.max_drawdown))
        ruin_prob = float(np.mean(finals < config.ruin_fraction))
        feasible = (
            mdd_breach_prob <= config.max_drawdown_prob
            and ruin_prob <= config.max_ruin_prob
        )
        points.append(FrontierScanPoint(float(multiple), mdd_breach_prob, ruin_prob, feasible))
    return tuple(points)


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


_check_contract()
