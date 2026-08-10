"""Evaluation layer: phase diagnostics, cost/tail sensitivity, Sharpe, folds, readiness.

The target-weight proxy (``mhs_ledger_pnl``) is pre-screen only; Research GO,
OOS, capital, and participation numbers come from the simulated inventory
ledger. ``phase_diagnostic_metrics`` answers "is the result robust to an
arbitrary decision-clock offset?" and nothing else.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.mhs.contracts import MEASURED_EXECUTION_COST_TIERS_BPS
from src.mhs.execution import mhs_ledger_pnl


def _zero_variance(sd: float, mean: float) -> bool:
    """Treat numerically-zero sample dispersion as zero variance."""
    return sd <= 1e-12 * max(1.0, abs(mean))


def _sharpe(series: pd.Series, periods_per_year: float) -> float:
    """Annualized sample Sharpe, zero-variance-safe."""
    sd = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    mean = float(series.mean())
    if _zero_variance(sd, mean):
        return float("inf") if mean > 0 else float("-inf")
    return float(mean / sd * math.sqrt(periods_per_year))


@dataclass(frozen=True, slots=True)
class PhaseDiagnosticResult:
    """Robustness diagnostic over independently-run, non-capital-shared phases.

    Not a tradable portfolio; ``degenerate`` flags a phase spread that exceeds
    the absolute mean performance, in which case the numbers must not be
    reported as a go/no-go input.
    """

    n_phases: int
    ensemble_ann: float
    ensemble_sharpe: float
    mean_phase_ann: float
    min_phase_ann: float
    max_phase_ann: float
    phase_spread_ann: float
    degenerate: bool


@dataclass(frozen=True, slots=True)
class CostResponsePoint:
    net_ann: float
    net_sharpe: float
    net_t: float


@dataclass(frozen=True, slots=True)
class TailSensitivityResult:
    base_net_ann: float
    base_sharpe: float
    winsor_curve: dict[int, tuple[float, float]]
    event_window_bars: int
    event_count: int
    top1_event_share: float
    top5_event_share: float
    top1pct_events_share: float
    leave_worst_event_out_sharpe: float


def phase_diagnostic_metrics(
    phase_nets: Mapping[int, pd.Series],
    periods_per_year: float,
) -> PhaseDiagnosticResult:
    """Mean of N independently-run single-phase books, annualized.

    ``ensemble_ann`` is the annualized mean of the equal-weight average across
    the independent ``phase_nets``; per-phase annualized means are summarized
    by mean/min/max and ``phase_spread_ann`` is max minus min.
    """
    if not phase_nets:
        raise ValueError("phase_nets must not be empty")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    series = list(phase_nets.values())
    first = series[0].index
    if any(not s.index.equals(first) for s in series):
        raise ValueError("all phase series must be identically indexed")
    per_phase_ann = [float(s.mean()) * periods_per_year for s in series]
    ensemble = pd.concat(series, axis=1).mean(axis=1)
    mean_phase_ann = float(np.mean(per_phase_ann))
    spread = float(max(per_phase_ann) - min(per_phase_ann))
    return PhaseDiagnosticResult(
        n_phases=len(series),
        ensemble_ann=float(ensemble.mean()) * periods_per_year,
        ensemble_sharpe=_sharpe(ensemble, periods_per_year),
        mean_phase_ann=mean_phase_ann,
        min_phase_ann=min(per_phase_ann),
        max_phase_ann=max(per_phase_ann),
        phase_spread_ann=spread,
        degenerate=spread > abs(mean_phase_ann),
    )


def cost_response_curve(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    cost_grid_bps: tuple[float, ...],
    periods_per_year: float,
) -> dict[float, CostResponsePoint]:
    """Target-weight pre-screen cost sensitivity at every one-way rate.

    Pre-screen only; the simulated inventory ledger remains the Research-GO
    PnL. The grid must include the three ``MEASURED_EXECUTION_COST_TIERS_BPS``
    values in addition to any diagnostic literals.
    """
    if not cost_grid_bps:
        raise ValueError("cost_grid_bps must not be empty")
    if any(c < 0 for c in cost_grid_bps):
        raise ValueError("cost rates must be >= 0")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    out: dict[float, CostResponsePoint] = {}
    for rate in cost_grid_bps:
        net, _turnover = mhs_ledger_pnl(weights, opens, bar_funding, rate)
        sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        t_stat = (
            float(net.mean() / sd * math.sqrt(len(net))) if sd > 0 else float("nan")
        )
        out[rate] = CostResponsePoint(
            net_ann=float(net.mean()) * periods_per_year,
            net_sharpe=_sharpe(net, periods_per_year),
            net_t=t_stat,
        )
    return out


def _event_clusters(
    base: pd.Series, window_bars: int,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Coalesce holding-horizon windows around the top positive bars.

    Seeds are the highest ``ceil(1% * n)`` positive-contribution bars; each
    seed maps to its inclusive ``[i - window, i + window]`` positional window
    and overlapping windows are merged, so adjacent bars from one market move
    become one event rather than independent samples.
    """
    n = len(base)
    if n == 0:
        return [], []
    sorted_desc = base.sort_values(ascending=False)
    positive = sorted_desc[sorted_desc > 0]
    n_seeds = max(1, math.ceil(0.01 * n))
    seeds = list(positive.index[:n_seeds])
    positions = {t: i for i, t in enumerate(base.index)}
    intervals = [
        (max(0, positions[t] - window_bars), min(n - 1, positions[t] + window_bars))
        for t in seeds
    ]
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    contributions = [float(base.iloc[lo : hi + 1].sum()) for lo, hi in merged]
    return merged, contributions


def tail_sensitivity_curve(
    weights: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    turnover: pd.Series,
    one_way_bps: float,
    periods_per_year: float,
    event_window_bars: int,
) -> TailSensitivityResult:
    """Winsor curve + deterministic event-cluster tail diagnostics.

    The winsor curve clips the PER-SYMBOL forward-return panel at each cap in
    (0.50, 0.30, 0.20, 0.10) before aggregating with weights; event clusters
    are computed on the uncapped base net-return series with the frozen
    holding-horizon radius ``event_window_bars``.
    """
    if not weights.index.equals(fwd_returns.index) or list(weights.columns) != list(fwd_returns.columns):
        raise ValueError("weights and fwd_returns must be identically indexed and columned")
    if not weights.index.equals(turnover.index):
        raise ValueError("weights and turnover must share an index")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    if event_window_bars < 1:
        raise ValueError(f"event_window_bars must be >= 1, got {event_window_bars}")

    def net_from(fwd: pd.DataFrame) -> pd.Series:
        gross = (weights * fwd).sum(axis=1)
        return gross - turnover * (one_way_bps * 1e-4)

    base = net_from(fwd_returns)
    curve: dict[int, tuple[float, float]] = {}
    for cap in (0.50, 0.30, 0.20, 0.10):
        capped = net_from(fwd_returns.clip(-cap, cap))
        curve[int(cap * 100)] = (
            float(capped.mean() * periods_per_year),
            _sharpe(capped, periods_per_year),
        )

    total = float(base.sum())
    _intervals, contributions = _event_clusters(base, event_window_bars)
    event_count = len(contributions)
    if not contributions or total == 0:
        top1 = top5 = top1pct = float("nan")
    else:
        ordered = sorted(contributions, reverse=True)
        n_top1pct = max(1, math.ceil(0.01 * event_count))
        top1 = ordered[0] / total
        top5 = float(np.sum(ordered[:5])) / total
        top1pct = float(np.sum(ordered[:n_top1pct])) / total

    worst_idx = int(np.argmin(base.to_numpy(dtype="float64"))) if len(base) else 0
    n = len(base)
    lo = max(0, worst_idx - event_window_bars)
    hi = min(n - 1, worst_idx + event_window_bars)
    mask = np.ones(n, dtype=bool)
    mask[lo : hi + 1] = False
    without_worst = base.iloc[mask]

    return TailSensitivityResult(
        base_net_ann=float(base.mean() * periods_per_year),
        base_sharpe=_sharpe(base, periods_per_year),
        winsor_curve=curve,
        event_window_bars=event_window_bars,
        event_count=event_count,
        top1_event_share=top1,
        top5_event_share=top5,
        top1pct_events_share=top1pct,
        leave_worst_event_out_sharpe=_sharpe(without_worst, periods_per_year),
    )


def autocorrelation_adjusted_sharpe(
    daily_net_returns: pd.Series,
    annualization_days: int = 365,
    max_lag_days: int = 7,
) -> float:
    """Daily-compounded annualized Sharpe with the frozen autocorrelation adjustment.

    Divides the annualized sample Sharpe by
    ``sqrt(1 + 2 * sum((1 - k/(max_lag_days + 1)) * rho_k))`` over lags
    ``k = 1..max_lag_days``. Phase 1 fixes ``annualization_days=365`` and
    ``max_lag_days=7`` (the longest frozen holding horizon is 168h).
    """
    if daily_net_returns.index.tz is None:
        raise ValueError("daily_net_returns must be tz-aware")
    if not daily_net_returns.index.is_monotonic_increasing:
        raise ValueError("daily_net_returns must be monotonic in time")
    if len(daily_net_returns) < max_lag_days + 2:
        raise ValueError(
            f"need at least {max_lag_days + 2} observations, got {len(daily_net_returns)}"
        )
    if annualization_days < 1:
        raise ValueError(f"annualization_days must be >= 1, got {annualization_days}")
    if max_lag_days < 1:
        raise ValueError(f"max_lag_days must be >= 1, got {max_lag_days}")

    mean = float(daily_net_returns.mean())
    std = float(daily_net_returns.std(ddof=1))
    if _zero_variance(std, mean):
        if mean > 0:
            return float("inf")
        if mean < 0:
            return float("-inf")
        return float("nan")
    sample_sharpe = mean / std * math.sqrt(annualization_days)
    acf_sum = 0.0
    n = len(daily_net_returns)
    demeaned = daily_net_returns.to_numpy(dtype="float64") - mean
    var = float(np.dot(demeaned, demeaned))
    for k in range(1, max_lag_days + 1):
        rho = float(np.dot(demeaned[k:], demeaned[: n - k])) / var if var > 0 else 0.0
        acf_sum += (1.0 - k / (max_lag_days + 1)) * rho
    denom = max(1.0 + 2.0 * acf_sum, 1e-12)
    return sample_sharpe / math.sqrt(denom)


@dataclass(frozen=True, slots=True)
class AnchoredPurgedFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    forward_dependency_hours: int
    purge_hours: int

    def __post_init__(self) -> None:
        bounds = (self.train_start, self.train_end, self.validation_start, self.validation_end)
        if any(b.tzinfo is None for b in bounds):
            raise ValueError("fold bounds must be tz-aware")
        if not (
            self.train_start < self.train_end < self.validation_start < self.validation_end
        ):
            raise ValueError("fold bounds must be strictly ascending")
        if self.forward_dependency_hours < 1:
            raise ValueError("forward_dependency_hours must be >= 1")
        if self.purge_hours < self.forward_dependency_hours:
            raise ValueError("purge_hours must be >= forward_dependency_hours")
        if self.validation_start <= self.train_end:
            raise ValueError("the purge embargo must be positive")


def phase_1_anchored_purged_folds() -> tuple[AnchoredPurgedFold, AnchoredPurgedFold, AnchoredPurgedFold]:
    """The three preregistered Level 2 anchored purged folds.

    ``purge_hours`` derives from the maximum forward dependency (frozen at
    168h for Phase 1); these folds are internal historical robustness, never
    labelled OOS.
    """
    purge = 168
    return (
        AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31", tz="UTC"),
            pd.Timestamp("2023-01-08", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            168,
            purge,
        ),
        AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2023-12-31", tz="UTC"),
            pd.Timestamp("2024-01-08", tz="UTC"),
            pd.Timestamp("2024-12-31", tz="UTC"),
            168,
            purge,
        ),
        AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2024-12-31", tz="UTC"),
            pd.Timestamp("2025-01-08", tz="UTC"),
            pd.Timestamp("2025-12-31", tz="UTC"),
            168,
            purge,
        ),
    )


@dataclass(frozen=True, slots=True)
class SyntheticStressScenario:
    name: str
    description: str


def synthetic_stress_scenarios() -> tuple[SyntheticStressScenario, ...]:
    """The nine deterministic, parameter-free preregistered stress scenarios.

    Distinct from historical event removal. Unsupported components are reported
    rather than fabricated from OHLCV, and no scenario may tune Phase 1
    parameters.
    """
    return (
        SyntheticStressScenario("BTC_DOWN_10", "BTC spot falls 10% intraday"),
        SyntheticStressScenario("BTC_DOWN_20", "BTC spot falls 20% intraday"),
        SyntheticStressScenario("ALT_BETA_UP", "altcoin beta to BTC spikes"),
        SyntheticStressScenario("XS_CORRELATION_ONE", "cross-sectional correlation equals 1"),
        SyntheticStressScenario("SPREAD_AND_COST_X3", "spread and costs triple"),
        SyntheticStressScenario("PASSIVE_FILL_DEGRADATION", "passive fill rate degrades"),
        SyntheticStressScenario("FUNDING_EXTREME", "funding rate reaches an extreme"),
        SyntheticStressScenario(
            "LIQUIDITY_DETERIORATION_50PCT", "50% of symbols face liquidity deterioration",
        ),
        SyntheticStressScenario("VENUE_API_OUTAGE_30M", "venue/API outage of 30 minutes"),
    )


@dataclass(frozen=True, slots=True)
class DeploymentReadinessResult:
    geometric_cagr: float
    max_drawdown: float
    calmar: float
    expected_shortfall: float
    worst_1d: float
    worst_7d: float
    worst_event: float
    time_under_water_bars: int
    recovery_bars: int | None
    probability_final_wealth_below_initial: float
    probability_mdd_over_20pct: float
    probability_mdd_over_30pct: float
    leverage_ruin_probabilities: Mapping[float, float]
    concentration: Mapping[str, float]
    participation_warnings: Mapping[str, float]
    research_go_eligible: bool
    execution_go_eligible: bool
    pilot_go_eligible: bool
    scale_go_eligible: bool


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def _time_under_water(equity: pd.Series) -> int:
    running_max = equity.cummax()
    underwater = (equity < running_max).to_numpy()
    if not underwater.any():
        return 0
    # Vectorized run-length encoding of contiguous True runs: diff over a
    # zero-padded boolean array marks each run start (+1) and end (-1).
    padded = np.concatenate(([0], underwater.astype(np.int8), [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def _recovery_bars(equity: pd.Series) -> int | None:
    if equity.empty:
        return None
    running_max = equity.cummax()
    ratio = equity / running_max
    trough_idx = int(np.argmin(ratio.to_numpy(dtype="float64")))
    peak_before = running_max.iloc[trough_idx]
    after = equity.iloc[trough_idx:]
    recovered = after[after >= peak_before]
    if len(recovered):
        return int(recovered.index.get_loc(recovered.index[0]))
    return None


def _bootstrap_chunk_size(n: int) -> int:
    """Per-chunk replicate count keeping the (chunk, n) sample matrix bounded.

    The vectorized bootstrap materialises one ``(chunk, n)`` float64 sample
    matrix plus a few same-shaped temporaries per chunk.  At production scale
    the equity is the 5-minute grid (~525,600 bars), so a fixed ``chunk=500``
    would allocate ~2.1GB per array and spike RSS far above the 8GB soft
    budget.  ``chunk`` is capped to keep a single sample matrix <= 128MB
    (spec O10); the MDD path also allocates a same-sized running-max temporary,
    so a 128MB sample keeps the combined transient ~1.1GB.
    """
    if n <= 0:
        return 500
    return max(1, min(500, int((128 * 2**20) // (n * 8))))


def _stationary_block_bootstrap_paths(
    net_returns: np.ndarray, n_replicates: int, mean_block: int, seed: int,
) -> np.ndarray:
    """Wealth multipliers (final wealth / initial) per replicate, vectorized.

    Statistically equivalent to the scalar while-loop block composition
    (MHS_PERF_OPT_003 precedent): block lengths are ``geometric(p_block)`` and
    block starts are uniform, matching the scalar length law.  A 6x block-count
    safety margin makes running short effectively impossible; any shortfall
    still falls back to the scalar replicate path.
    """
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be > 0, got {n_replicates}")
    if mean_block < 0:
        raise ValueError(f"mean_block must be >= 0, got {mean_block}")
    rng = np.random.default_rng(seed)
    n = len(net_returns)
    if n == 0:
        return np.ones(n_replicates)
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    if p_block <= 0.0:
        # Degenerate mean_block == 0: every block is a single element (the
        # scalar ``while length < n and rng.random() > 0`` never advances).
        starts = rng.integers(0, n, size=n_replicates)
        return np.asarray(np.prod(1.0 + net_returns[starts], axis=0), dtype="float64")
    outcomes = np.empty(n_replicates, dtype="float64")
    chunk = _bootstrap_chunk_size(n)
    for r0 in range(0, n_replicates, chunk):
        r1 = min(r0 + chunk, n_replicates)
        k = r1 - r0
        max_blocks = min(n, int(np.ceil(n * 6.0 / mean_block)) + 16)
        lengths = rng.geometric(p_block, size=(k, max_blocks))
        starts = rng.integers(0, n, size=(k, max_blocks))
        ends = np.cumsum(lengths, axis=1)
        short = ends[:, -1] < n
        for r in np.flatnonzero(short).tolist():
            outcomes[r0 + r] = _block_bootstrap_replicate_wealth(
                net_returns, n, p_block, rng,
            )
        valid = ~short
        if valid.any():
            ends_trunc = np.minimum(ends, n)
            used = ends_trunc - np.concatenate(
                [np.zeros((k, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1,
            )
            u = used[valid].ravel()
            s = starts[valid].ravel()
            keep = u > 0
            u = u[keep]
            s = s[keep]
            block_start = np.cumsum(u) - u
            offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
            arr_idx = (np.repeat(s, u) + offsets) % n
            sample = net_returns[arr_idx].reshape(int(valid.sum()), n)
            outcomes[r0 + np.flatnonzero(valid)] = np.prod(1.0 + sample, axis=1)
    return outcomes


def _block_bootstrap_replicate_wealth(
    arr: np.ndarray, n: int, p_block: float, rng: np.random.Generator,
) -> float:
    """Wealth multiplier of one scalar block-bootstrap replicate (fallback)."""
    blocks: list[float] = []
    while len(blocks) < n:
        start = int(rng.integers(0, n))
        length = 1
        while length < n and rng.random() > p_block:
            length += 1
        length = min(length, n - len(blocks))
        blocks.extend(arr[start : start + length].tolist())
    path = np.array(blocks[:n], dtype="float64")
    return float(np.prod(1.0 + path))


def _bootstrap_mdd_paths(
    net_returns: np.ndarray, n_replicates: int, mean_block: int, seed: int,
) -> np.ndarray:
    """Per-replicate max drawdown of the block-bootstrap equity path, vectorized."""
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be > 0, got {n_replicates}")
    if mean_block < 0:
        raise ValueError(f"mean_block must be >= 0, got {mean_block}")
    rng = np.random.default_rng(seed + 1)
    n = len(net_returns)
    if n == 0:
        return np.zeros(n_replicates)
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    if p_block <= 0.0:
        starts = rng.integers(0, n, size=n_replicates)
        equity = np.cumprod(1.0 + net_returns[starts], axis=0)
        running_max = np.maximum.accumulate(equity, axis=0)
        return np.asarray((equity / running_max - 1.0).min(axis=0), dtype="float64")
    mdd = np.empty(n_replicates, dtype="float64")
    chunk = _bootstrap_chunk_size(n)
    for r0 in range(0, n_replicates, chunk):
        r1 = min(r0 + chunk, n_replicates)
        k = r1 - r0
        max_blocks = min(n, int(np.ceil(n * 6.0 / mean_block)) + 16)
        lengths = rng.geometric(p_block, size=(k, max_blocks))
        starts = rng.integers(0, n, size=(k, max_blocks))
        ends = np.cumsum(lengths, axis=1)
        short = ends[:, -1] < n
        for r in np.flatnonzero(short).tolist():
            mdd[r0 + r] = _block_bootstrap_replicate_mdd(
                net_returns, n, p_block, rng,
            )
        valid = ~short
        if valid.any():
            ends_trunc = np.minimum(ends, n)
            used = ends_trunc - np.concatenate(
                [np.zeros((k, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1,
            )
            u = used[valid].ravel()
            s = starts[valid].ravel()
            keep = u > 0
            u = u[keep]
            s = s[keep]
            block_start = np.cumsum(u) - u
            offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
            arr_idx = (np.repeat(s, u) + offsets) % n
            sample = net_returns[arr_idx].reshape(int(valid.sum()), n)
            sample += 1.0
            np.cumprod(sample, axis=1, out=sample)
            running_max = np.maximum.accumulate(sample, axis=1)
            mdd[r0 + np.flatnonzero(valid)] = (sample / running_max - 1.0).min(axis=1)
    return mdd


def _block_bootstrap_replicate_mdd(
    arr: np.ndarray, n: int, p_block: float, rng: np.random.Generator,
) -> float:
    """Max drawdown of one scalar block-bootstrap replicate (fallback)."""
    blocks: list[float] = []
    while len(blocks) < n:
        start = int(rng.integers(0, n))
        length = 1
        while length < n and rng.random() > p_block:
            length += 1
        length = min(length, n - len(blocks))
        blocks.extend(arr[start : start + length].tolist())
    path = np.array(blocks[:n], dtype="float64")
    equity = np.cumprod(1.0 + path)
    running_max = np.maximum.accumulate(equity)
    return float((equity / running_max - 1.0).min())


def compute_deployment_readiness(
    equity: pd.Series,
    periods_per_year: float,
    concentration: Mapping[str, float] | None = None,
    participation_warnings: Mapping[str, float] | None = None,
    primary_valid: bool = True,
    research_go_eligible: bool | None = None,
    mean_block_bars: int = 168,
    n_bootstrap: int = 2000,
    seed: int = 20260807,
    leverage_grid: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> DeploymentReadinessResult:
    """Research/execution/pilot/scale readiness from strict-proxy simulated equity.

    Research GO can be true from historical strict-proxy evidence only when the
    primary ledger is valid (``primary_valid``). When ``research_go_eligible``
    is passed, it is the explicit gate decision and overrides the ``primary_valid``
    shortcut entirely, so a caller can route a fold-based Research-GO decision
    without weakening the fail-closed default. Execution, Pilot, and Scale GO
    require forward evidence that is absent by construction until the respective
    calibration data exists.
    """
    if equity.index.tz is None:
        raise ValueError("equity must be tz-aware")
    if not equity.index.is_monotonic_increasing:
        raise ValueError("equity must be monotonic in time")
    if (equity <= 0).any():
        raise ValueError("equity must be strictly positive")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    if not isinstance(primary_valid, bool):
        raise ValueError("primary_valid must be a bool")
    if research_go_eligible is not None and not isinstance(research_go_eligible, bool):
        raise ValueError("research_go_eligible must be a bool or None")
    research_go = primary_valid if research_go_eligible is None else research_go_eligible

    net = equity.pct_change().dropna()
    if len(net) == 0:
        raise ValueError("equity must contain at least two observations")
    final = float(equity.iloc[-1])
    initial = float(equity.iloc[0])
    n = len(net)
    geometric_cagr = (final / initial) ** (periods_per_year / n) - 1.0
    max_drawdown = _max_drawdown_from_equity(equity)
    calmar = geometric_cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    sorted_net = np.sort(net.to_numpy(dtype="float64"))
    tail_count = max(1, math.ceil(0.05 * n))
    expected_shortfall = float(np.mean(sorted_net[:tail_count]))
    worst_1d = float(net.min())

    k7 = min(7, n)
    # Vectorized sliding-window 7-bar sum: identical arithmetic to the scalar
    # slice loop (each window is summed in the same element order).
    if k7 > 1:
        from numpy.lib.stride_tricks import sliding_window_view

        worst_7d = float(
            sliding_window_view(net.to_numpy(dtype="float64"), k7).sum(axis=1).min()
        )
    else:
        worst_7d = worst_1d
    worst_event = worst_7d if k7 > 1 else worst_1d

    net_arr = net.to_numpy(dtype="float64")
    wealth = _stationary_block_bootstrap_paths(net_arr, n_bootstrap, mean_block_bars, seed)
    mdd_paths = _bootstrap_mdd_paths(net_arr, n_bootstrap, mean_block_bars, seed)
    probability_final_wealth_below_initial = float(np.mean(wealth < 1.0))
    probability_mdd_over_20pct = float(np.mean(mdd_paths < -0.20))
    probability_mdd_over_30pct = float(np.mean(mdd_paths < -0.30))

    # The scalar leverage-ruin loop recomputed the identical deterministic
    # leveraged path ``n_bootstrap`` times; a single cumulative-product check
    # per leverage is exactly equivalent (ruin_probs[lev] is 0.0 or 1.0).
    ruin_probs: dict[float, float] = {}
    for lev in leverage_grid:
        ruin_probs[lev] = float(
            bool((np.cumprod(1.0 + lev * net_arr) <= 0.0).any())
        )

    return DeploymentReadinessResult(
        geometric_cagr=geometric_cagr,
        max_drawdown=max_drawdown,
        calmar=calmar,
        expected_shortfall=expected_shortfall,
        worst_1d=worst_1d,
        worst_7d=worst_7d,
        worst_event=worst_event,
        time_under_water_bars=_time_under_water(equity),
        recovery_bars=_recovery_bars(equity),
        probability_final_wealth_below_initial=probability_final_wealth_below_initial,
        probability_mdd_over_20pct=probability_mdd_over_20pct,
        probability_mdd_over_30pct=probability_mdd_over_30pct,
        leverage_ruin_probabilities=ruin_probs,
        concentration=dict(concentration or {}),
        participation_warnings=dict(participation_warnings or {}),
        research_go_eligible=research_go,
        execution_go_eligible=False,
        pilot_go_eligible=False,
        scale_go_eligible=False,
    )


def required_cost_tiers() -> tuple[float, ...]:
    """The three named measured execution tiers, always reported together."""
    return tuple(
        MEASURED_EXECUTION_COST_TIERS_BPS[tier]
        for tier in ("optimistic", "base", "stress")
    )
