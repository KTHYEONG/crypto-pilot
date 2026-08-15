from __future__ import annotations

import dataclasses
import logging
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_1h_as
from src.research.contracts import CostModel, GrowthEngineEvaluationRequest
from src.research.evaluation.falsification import FalsificationConfig, evaluate_falsification, evaluate_parameter_plateau
from src.research.evaluation.policy import resolve_evaluation_end
from src.research.evaluation.promotion import PromotionResult, compose_promotion_verdict
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    compute_equal_duration_fold_distribution,
    compute_equity_reliability_gate,
)
from src.research.evaluation.reliability import count_closed_trades
from src.research.execution.intrabar_audit import intrabar_audit_required
from src.research.portfolio.growth_router import (
    CONTEXT_WINDOW_BARS,
    DISCOVERY_MONTHS,
    MIN_CONTEXT_SAMPLES,
    build_rolling_segments,
    causal_router,
    compute_context_features,
    context_state_for,
    enough_deployment_folds,
)
from src.research.portfolio.growth_strategy_library import (
    FAMILY_SIZE,
    STRATEGY_REGISTRY,
    GrowthStrategyScreen,
    align_funding_bars,
    registry_definition,
    screen_growth_strategy_weights,
)
from src.research.portfolio.net_construction import NetConstructionSpec, compute_net_return_stream
from src.research.risk.growth_sizing import (
    GrowthSizingConfig,
    apply_realised_risk_overlay,
    solve_growth_optimal_risk,
)
from src.research.universe.pit_universe import PitUniverseSpec, SymbolCoverage, build_universe_schedule, earliest_admissible_start, symbol_partition, derive_backfill_candidates

if TYPE_CHECKING:
    from src.research.evaluation.falsification import FalsificationVerdict
    from src.research.portfolio.growth_router import ContextState
    from src.research.portfolio.net_construction import NetReturnStream
    from src.research.risk.growth_sizing import GrowthSizingResult

_logger = logging.getLogger("GrowthEngineEvaluation")

BARS_PER_YEAR = 2190
_EMPTY_TRADE_COLUMNS = ("entry_bar", "exit_bar", "pnl", "return_pct")


@dataclass(frozen=True)
class GrowthEngineReport:
    """Fail-closed result of a growth-engine evaluation.

    ``status`` is ``"NO_ADMISSIBLE_ALPHA"`` when the admissible start cannot be
    derived, no family survives the discovery plateau rule, or the falsification
    verdict fails; the equity curve is then a flat CASH series at
    ``initial_equity`` with zero trades.  ``scorecard`` is the reproducible
    per-candidate evidence surface and ``selected_strategy`` the promoted
    identity (or ``None`` when holding CASH).
    """

    status: Literal["PASS", "NO_ADMISSIBLE_ALPHA"]
    equity: pd.Series
    trades: pd.DataFrame
    falsification: FalsificationVerdict | None
    sizing: GrowthSizingResult | None
    universe_schedule: dict[pd.Timestamp, tuple[str, ...]]
    start: pd.Timestamp | None
    promotion: PromotionResult | None
    record: dict[str, object] | None = None
    scorecard: GrowthCandidateScorecard | None = None
    selected_strategy: str | None = None


@dataclass(frozen=True)
class GrowthCandidateScoreEntry:
    """One immutable discovery-screen row for a single family/parameter candidate."""

    strategy_id: str
    parameter: int
    dev_discovery_score: float | None
    status: Literal["SCREENED", "DATA_INVALID"]
    family_passed: bool


@dataclass(frozen=True)
class GrowthCandidateScorecard:
    """Deterministic, immutable per-family evidence surface published on every run."""

    family_size: int
    entries: tuple[GrowthCandidateScoreEntry, ...]
    selected_strategy_id: str | None
    selected_parameter: int | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class _FamilyScreen:
    strategy_id: str
    chosen_parameter: int | None
    chosen_score: float | None
    passed: bool
    parameter_scores: dict[float, float]


def _empty_scorecard(reason: str) -> GrowthCandidateScorecard:
    return GrowthCandidateScorecard(
        family_size=FAMILY_SIZE, entries=(), selected_strategy_id=None,
        selected_parameter=None, reason=reason,
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_EMPTY_TRADE_COLUMNS))


def _list_symbols() -> tuple[str, ...]:
    pattern = ohlcv_path("", "1h")
    if not pattern.parent.exists():
        return ()
    return tuple(sorted(p.stem for p in pattern.parent.glob("*.parquet")))


def _load_4h_frame(
    symbol: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame | None:
    path = ohlcv_path(symbol, "1h")
    if not path.exists():
        return None
    try:
        return load_ohlcv_1h_as(path, "4h", start=start, end=end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.warning("[EVAL] excluding %s: %s", symbol, exc)
        return None


def _coverage_from_frame(frame: pd.DataFrame, symbol: str) -> SymbolCoverage | None:
    if frame is None or frame.empty:
        return None
    span = frame.index[-1] - frame.index[0]
    expected = span // pd.Timedelta(hours=4) + 1
    bar_coverage = min(float(len(frame) / expected), 1.0)
    return SymbolCoverage(symbol, frame.index[0], frame.index[-1], bar_coverage)


def _build_universe_inputs(
    symbols: Sequence[str],
    spec: PitUniverseSpec,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> tuple[list[SymbolCoverage], dict[str, pd.Series], dict[str, pd.DataFrame]]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _load_4h_frame(symbol, start, end)
        if frame is not None:
            frames[symbol] = frame
    coverage = [
        cov
        for symbol, frame in frames.items()
        if (cov := _coverage_from_frame(frame, symbol)) is not None
    ]
    liquidity = {
        symbol: frame["quote_vol"]
        for symbol, frame in frames.items()
        if "quote_vol" in frame.columns
    }
    return coverage, liquidity, frames


def _build_rebalance_dates(
    data_start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    begin = data_start.tz_convert("UTC").normalize() - pd.offsets.MonthBegin(0)
    return list(pd.date_range(begin, end, freq="MS", tz="UTC"))


def _apply_scope(
    schedule: dict[pd.Timestamp, tuple[str, ...]],
    scope: str,
    spec: PitUniverseSpec,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    if scope == "all":
        return schedule
    return {
        date: tuple(
            sym for sym in roster if symbol_partition(sym, spec.dev_fraction) == scope
        )
        for date, roster in schedule.items()
    }


def _subset_schedule(
    schedule: dict[pd.Timestamp, tuple[str, ...]],
    members: set[str],
) -> dict[pd.Timestamp, tuple[str, ...]]:
    return {
        date: tuple(sym for sym in roster if sym in members)
        for date, roster in schedule.items()
    }


def _build_price_panel(
    panel_symbols: list[str],
    frames: Mapping[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = pd.date_range(start, end, freq="4h", tz="UTC", inclusive="left")
    closes: dict[str, pd.Series] = {}
    taker: dict[str, pd.Series] = {}
    for symbol in panel_symbols:
        frame = frames[symbol]
        closes[symbol] = frame["close"].reindex(grid)
        if "taker_buy_ratio" in frame.columns:
            taker[symbol] = frame["taker_buy_ratio"].reindex(grid)
        else:
            taker[symbol] = pd.Series(np.nan, index=grid)
    px = pd.DataFrame(closes)
    taker_ratio = pd.DataFrame(taker)

    # Decision-to-fill aligned forward returns: fwd[t] = close[t+1] / close[t] - 1,
    # so a signal decided at bar t earns the return realised on the following bar.
    arr = px.to_numpy(dtype=np.float64)
    fwd_arr = np.full_like(arr, np.nan)
    fwd_arr[:-1] = arr[1:] / arr[:-1] - 1.0
    fwd = pd.DataFrame(fwd_arr, index=px.index, columns=px.columns)
    return px, fwd, taker_ratio


def _compute_stream(
    target_weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    construction: NetConstructionSpec,
    forward_funding: pd.DataFrame | None = None,
) -> NetReturnStream:
    return compute_net_return_stream(
        target_weights, forward_returns, construction, forward_funding,
    )


def _gross_sharpe(pnl: pd.Series) -> float:
    rets = pnl.dropna()
    if len(rets) < 2:
        return 0.0
    std = float(rets.std())
    if std <= 0:
        return 0.0
    return float(rets.mean() / std * np.sqrt(BARS_PER_YEAR))


def _oos_t_stat(net: pd.Series, *, test_fraction: float = 0.5) -> float:
    rets = net.dropna()
    if len(rets) < 10:
        return 0.0
    # Recency-restricted OOS window (default: trailing half); dev_qualification_score
    # separately uses the full qualification range -- see docs/specs/growth_engine_gate_diagnostics.md s3.
    test = rets.iloc[int(len(rets) * (1.0 - test_fraction)) :]
    if len(test) < 2:
        return 0.0
    std = float(test.std())
    if std <= 0:
        return 0.0
    return float(test.mean() / std * np.sqrt(len(test)))


_FOLD_DURATION = "6MS"


def _qualification_fold_gate_pass(net: pd.Series) -> bool:
    """Fail-closed fold-concentration check on the qualification net-return stream.

    Reuses ``compute_equal_duration_fold_distribution`` (unchanged reliability
    contract, docs/specs/growth_engine_fold_concentration_gate.md) on the
    cumulative qualification equity split into 6-month folds, guarding against
    an ``oos_t_stat`` that clears the multiplicity floor only because
    performance concentrates in one favourable sub-period rather than being
    distributed across the qualification window.  A span too short to admit at
    least one fold, or fewer than two return observations, fails closed
    (returns ``False``) rather than raising.
    """
    rets = net.dropna()
    if len(rets) < 2:
        return False
    equity = (1.0 + rets).cumprod()
    try:
        result = compute_equal_duration_fold_distribution(
            equity, ReliabilityGateConfig(), fold_duration=_FOLD_DURATION,
        )
    except ValueError as exc:
        _logger.warning("[EVAL] fold_gate fail-closed reason=%s", exc)
        return False
    _logger.info(
        "[EVAL] fold_gate n_folds=%d concentration=%.3f threshold=%.3f gate_pass=%s "
        "median_fold_cagr=%.3f worst_fold_cagr=%.3f",
        result.n_folds, result.fold_concentration, result.fold_concentration_threshold,
        result.gate_pass, result.median_fold_cagr, result.worst_fold_cagr,
    )
    return result.gate_pass


def family_window_correlation(
    net_return_streams: Mapping[int, pd.Series],
) -> dict[tuple[int, int], float]:
    """Pairwise Pearson correlation of per-window net-return streams.

    Measurement-only diagnostic (docs/specs/growth_engine_gate_diagnostics.md
    section 3): computes every window pair's correlation for one strategy
    family and never feeds into falsification or promotion.  A pair is omitted
    when fewer than two windows are provided, the shared non-null overlap has
    fewer than 10 bars, or either stream is constant on the overlap (undefined
    correlation).
    """
    windows = sorted(net_return_streams)
    if len(windows) < 2:
        return {}
    out: dict[tuple[int, int], float] = {}
    for index, left in enumerate(windows):
        for right in windows[index + 1:]:
            x = net_return_streams[left].dropna()
            y = net_return_streams[right].dropna()
            overlap = x.index.intersection(y.index)
            if len(overlap) < 10:
                continue
            xv = x.loc[overlap].to_numpy(dtype=np.float64)
            yv = y.loc[overlap].to_numpy(dtype=np.float64)
            if float(np.std(xv)) <= 0.0 or float(np.std(yv)) <= 0.0:
                continue
            out[(left, right)] = float(np.corrcoef(xv, yv)[0, 1])
    return out


def _cash_curve_index(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    if start is None or end is None or end <= start:
        return pd.DatetimeIndex([], tz="UTC")
    return _build_rebalance_dates(start, end)


def _git_head() -> tuple[str | None, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, False
    return sha, dirty


def _record_run(
    request: GrowthEngineEvaluationRequest,
    report: GrowthEngineReport,
    end: str | pd.Timestamp | None,
) -> dict[str, object]:
    from src.research.provenance.ledger import (
        RUNS_LOG_PATH,
        append_event,
        build_evaluation_event,
    )

    git_sha, git_dirty = _git_head()
    event = build_evaluation_event(
        workflow="growth_engine",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics={
            "final_equity": (
                float(report.equity.iloc[-1]) if len(report.equity) else float(request.initial_equity)
            ),
            "n_bars": len(report.equity),
            "trade_count": len(report.trades),
            "selected_strategy": report.selected_strategy,
        },
        reliability={
            "binding_constraint": (
                report.falsification.binding_constraint
                if report.falsification is not None
                else None
            ),
            "oos_t_stat": (
                report.falsification.oos_t_stat if report.falsification is not None else None
            ),
            "selected_risk": (
                report.sizing.selected_risk if report.sizing is not None else None
            ),
        },
        promotion={
            "status": report.promotion.status if report.promotion is not None else "REJECTED",
        },
        status=report.status,
        start=str(report.start) if report.start is not None else None,
        end=str(end) if end is not None else None,
        initial_equity=request.initial_equity,
        symbol_scope=request.symbol_scope,
        universe_size=request.universe.universe_size,
        family_size=FAMILY_SIZE,
    )
    appended = append_event(event, ledger_path=RUNS_LOG_PATH)
    return dict(appended.payload)


def _no_admissible_alpha(
    request: GrowthEngineEvaluationRequest,
    end: pd.Timestamp,
    schedule: dict[pd.Timestamp, tuple[str, ...]],
    *,
    start: pd.Timestamp | None,
    falsification: FalsificationVerdict | None,
    sizing: GrowthSizingResult | None,
    scorecard: GrowthCandidateScorecard | None = None,
    selected_strategy: str | None = None,
) -> GrowthEngineReport:
    _logger.warning(
        "[EVAL] status=NO_ADMISSIBLE_ALPHA start=%s strategy=%s scorecard_reason=%s "
        "falsification=%s sizing=%s -- holding CASH",
        start,
        selected_strategy,
        scorecard.reason if scorecard is not None else "unavailable",
        falsification.binding_constraint if falsification is not None else "unavailable",
        sizing.binding_constraint if sizing is not None else "unavailable",
    )
    index = _cash_curve_index(start, end)
    equity = pd.Series(float(request.initial_equity), index=index)
    return GrowthEngineReport(
        status="NO_ADMISSIBLE_ALPHA",
        equity=equity,
        trades=_empty_trades(),
        falsification=falsification,
        sizing=sizing,
        universe_schedule=schedule,
        start=start,
        promotion=None,
        record=None,
        scorecard=scorecard,
        selected_strategy=selected_strategy,
    )


def _market_context_inputs(
    forward_returns: pd.DataFrame,
    panel_symbols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar equal-weight market return and cross-sectional breadth arrays."""
    arr = forward_returns[list(panel_symbols)].to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    counts = finite.sum(axis=1)
    with np.errstate(invalid="ignore"):
        zeroed = np.where(finite, arr, 0.0)
        market = np.where(
            counts > 0, zeroed.sum(axis=1) / np.maximum(counts, 1), 0.0,
        )
    breadth = finite.mean(axis=1)
    return np.asarray(market, dtype=np.float64), np.asarray(breadth, dtype=np.float64)


def _context_features_at(
    market_returns: np.ndarray,
    breadth: np.ndarray,
    bar_index: int,
    *,
    window: int = CONTEXT_WINDOW_BARS,
) -> tuple[float, float, float]:
    """Context features from completed bars strictly before ``bar_index``."""
    return compute_context_features(market_returns, breadth, end_idx=bar_index, window=window)


def _discovery_context_thresholds(
    market_returns: np.ndarray,
    breadth: np.ndarray,
    discovery_bars: np.ndarray,
) -> tuple[float, float]:
    """Discovery-median vol/breadth thresholds used to partition context states."""
    vols: list[float] = []
    breadths: list[float] = []
    for idx in discovery_bars:
        _mean, vol, mean_breadth = _context_features_at(market_returns, breadth, int(idx))
        if np.isfinite(vol):
            vols.append(vol)
        if np.isfinite(mean_breadth):
            breadths.append(mean_breadth)
    if not vols or not breadths:
        return float("nan"), float("nan")
    return float(np.median(vols)), float(np.median(breadths))


def _segment_context_state(
    market_returns: np.ndarray,
    breadth: np.ndarray,
    deploy_start_index: int,
    vol_threshold: float,
    breadth_threshold: float,
) -> ContextState | None:
    """Pre-decision context state from the trailing window before deployment."""
    features = _context_features_at(market_returns, breadth, deploy_start_index)
    if not all(np.isfinite(value) for value in features):
        return None
    if not np.isfinite(vol_threshold) or not np.isfinite(breadth_threshold):
        return None
    try:
        return context_state_for(
            features, vol_threshold=vol_threshold, breadth_threshold=breadth_threshold,
        )
    except ValueError:
        return None


def _build_settled_funding(
    symbols: Sequence[str],
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Load raw per-symbol funding events into one aligned-able frame.

    Symbols with missing or unreadable funding files are simply absent from the
    frame; the funding screen then records ``DATA_INVALID`` for the candidate
    instead of inventing zero funding.
    """
    directory = funding_path("").parent
    if not directory.exists():
        return pd.DataFrame(index=grid)
    columns: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            rates = load_funding_rates(funding_path(symbol))
        except (DataIntegrityError, FileNotFoundError):
            continue
        if len(rates):
            columns[symbol] = rates
    if not columns:
        return pd.DataFrame(index=grid)
    frame = pd.DataFrame(columns)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _build_forward_funding(
    raw_funding: pd.DataFrame,
    grid: pd.DatetimeIndex,
    target_columns: Sequence[str],
) -> pd.DataFrame | None:
    """Post-decision interval funding aligned to the trading panel columns."""
    present = [column for column in target_columns if column in raw_funding.columns]
    if not present:
        return None
    aligned = align_funding_bars(raw_funding[list(present)], grid, forward=True)
    return aligned.reindex(columns=list(target_columns)).fillna(0.0)


def _screen_discovery_candidates(
    schedule: dict[pd.Timestamp, tuple[str, ...]],
    px: pd.DataFrame,
    fwd: pd.DataFrame,
    taker: pd.DataFrame,
    settled_funding: pd.DataFrame,
    bars: pd.DatetimeIndex,
    max_positions: int,
) -> tuple[tuple[GrowthCandidateScoreEntry, ...], tuple[_FamilyScreen, ...]]:
    """Screen every frozen family/window on a discovery evidence window only."""
    entries: list[GrowthCandidateScoreEntry] = []
    families: list[_FamilyScreen] = []
    for definition in STRATEGY_REGISTRY:
        scores: dict[float, float] = {}
        invalid_windows: list[int] = []
        for window in definition.windows:
            screen = screen_growth_strategy_weights(
                definition.strategy_id, window, schedule, px, taker,
                settled_funding, max_positions,
            )
            if screen.status == "DATA_INVALID":
                invalid_windows.append(window)
                continue
            pnl = (screen.weights * fwd).sum(axis=1)
            scores[float(window)] = _gross_sharpe(pnl.loc[bars])
        chosen = max(scores, key=lambda key: (scores[key], -key)) if scores else None
        passed = False
        if chosen is not None:
            passed = evaluate_parameter_plateau(scores, chosen, FalsificationConfig()).passed
        for window in definition.windows:
            if window in invalid_windows:
                entries.append(GrowthCandidateScoreEntry(
                    strategy_id=definition.strategy_id, parameter=window,
                    dev_discovery_score=None, status="DATA_INVALID",
                    family_passed=passed,
                ))
            else:
                entries.append(GrowthCandidateScoreEntry(
                    strategy_id=definition.strategy_id, parameter=window,
                    dev_discovery_score=scores[float(window)], status="SCREENED",
                    family_passed=passed,
                ))
        families.append(_FamilyScreen(
            strategy_id=definition.strategy_id,
            chosen_parameter=int(chosen) if chosen is not None else None,
            chosen_score=scores[chosen] if chosen is not None else None,
            passed=passed,
            parameter_scores=scores,
        ))
    return tuple(entries), tuple(families)

def _sleeve_net_stream(
    screen: GrowthStrategyScreen,
    fwd: pd.DataFrame,
    construction: NetConstructionSpec,
    settled_funding: pd.DataFrame,
    grid: pd.DatetimeIndex,
) -> NetReturnStream:
    """Net-of-turnover return stream for one frozen screen on the trading panel."""
    forward_returns = fwd[list(screen.weights.columns)]
    definition = registry_definition(screen.strategy_id)
    if definition.requires_funding:
        forward_funding = _build_forward_funding(
            settled_funding, grid, list(screen.weights.columns),
        )
        return _compute_stream(
            screen.weights, forward_returns, construction, forward_funding,
        )
    return _compute_stream(screen.weights, forward_returns, construction)


@dataclass(frozen=True)
class _SleeveEvidence:
    """Discovery-only evidence for one independently admitted sleeve.

    ``admission_lcb`` is the block-aware lower confidence bound over the full
    discovery window; ``context_lcb`` is the same bound restricted to discovery
    bars whose pre-decision context matches the frozen deployment context.  A
    sleeve with a non-positive admission LCB is CASH and is never routed.
    """

    family: _FamilyScreen
    discovery_stream: NetReturnStream
    discovery_net: pd.Series
    discovery_realized: pd.DataFrame
    discovery_net_sharpe: float
    admission_lcb: float
    context_lcb: float


def _discovery_sleeve_evidence(
    family: _FamilyScreen,
    discovery_schedule: dict[pd.Timestamp, tuple[str, ...]],
    px: pd.DataFrame,
    fwd: pd.DataFrame,
    taker: pd.DataFrame,
    settled_funding: pd.DataFrame,
    construction: NetConstructionSpec,
    grid: pd.DatetimeIndex,
    discovery_bars: pd.DatetimeIndex,
    max_positions: int,
    router_config: ReliabilityGateConfig,
) -> _SleeveEvidence | None:
    """Measure one sleeve's discovery-only net evidence; fail closed on invalid data."""
    if family.chosen_parameter is None:
        return None
    screen = screen_growth_strategy_weights(
        family.strategy_id, family.chosen_parameter, discovery_schedule, px, taker,
        settled_funding, max_positions,
    )
    if screen.status != "SCREENED":
        return None
    stream = _sleeve_net_stream(screen, fwd, construction, settled_funding, grid)
    net = stream.net.loc[discovery_bars].dropna()
    realized = stream.realized_weights.loc[discovery_bars]
    if len(net) < 2:
        return None
    equity = (1.0 + net).cumprod()
    closed = count_closed_trades(realized)
    try:
        admission_lcb = compute_equity_reliability_gate(
            equity, closed, config=router_config,
        ).lcb90_cagr
    except ValueError:
        return None
    return _SleeveEvidence(
        family=family,
        discovery_stream=stream,
        discovery_net=net,
        discovery_realized=realized,
        discovery_net_sharpe=_gross_sharpe(net),
        admission_lcb=float(admission_lcb),
        context_lcb=float("nan"),
    )


def _discovery_context_evidence(
    market_returns: np.ndarray,
    breadth: np.ndarray,
    discovery_bars: pd.DatetimeIndex,
    grid: pd.DatetimeIndex,
) -> tuple[float, float, dict[int, ContextState]]:
    """Discovery-median vol/breadth thresholds and per-bar pre-decision context states.

    The medians are used only to partition the decision-time context; they are
    derived from the discovery window alone and never touch the outer
    deployment or symbol-holdout returns.
    """
    positions = grid.get_indexer(discovery_bars)
    means = np.full(len(positions), np.nan)
    vols = np.full(len(positions), np.nan)
    breadths = np.full(len(positions), np.nan)
    for i, idx in enumerate(positions):
        mean_ret, vol, mean_breadth = _context_features_at(market_returns, breadth, int(idx))
        means[i] = mean_ret
        vols[i] = vol
        breadths[i] = mean_breadth
    finite_vols = vols[np.isfinite(vols)]
    finite_breadths = breadths[np.isfinite(breadths)]
    vol_threshold = float(np.nanmedian(finite_vols)) if len(finite_vols) else float("nan")
    breadth_threshold = (
        float(np.nanmedian(finite_breadths)) if len(finite_breadths) else float("nan")
    )
    states: dict[int, ContextState] = {}
    for i, idx in enumerate(positions):
        if not (np.isfinite(means[i]) and np.isfinite(vols[i]) and np.isfinite(breadths[i])):
            continue
        try:
            states[int(idx)] = context_state_for(
                (means[i], vols[i], breadths[i]),
                vol_threshold=vol_threshold,
                breadth_threshold=breadth_threshold,
            )
        except ValueError:
            continue
    return vol_threshold, breadth_threshold, states


def _context_sleeve_lcb(
    evidence: _SleeveEvidence,
    context_state: ContextState,
    states: Mapping[int, ContextState],
    grid: pd.DatetimeIndex,
    router_config: ReliabilityGateConfig,
) -> float:
    """Lower confidence bound of a sleeve restricted to its matching context.

    Returns ``nan`` (fail closed) when the context-sleeve pair has fewer than
    ``MIN_CONTEXT_SAMPLES`` discovery bars or the restricted equity is invalid,
    so a rare context can never fabricate a positive LCB.
    """
    matched = [
        int(idx) for idx, state in states.items()
        if state == context_state and int(idx) < len(grid)
    ]
    if len(matched) < MIN_CONTEXT_SAMPLES:
        return float("nan")
    net = evidence.discovery_net.loc[grid[matched]].dropna()
    realized = evidence.discovery_realized.loc[grid[matched]]
    if len(net) < 2:
        return float("nan")
    equity = (1.0 + net).cumprod()
    try:
        return compute_equity_reliability_gate(
            equity, count_closed_trades(realized), config=router_config,
        ).lcb90_cagr
    except ValueError:
        return float("nan")


def _segment_sizing(
    discovery_stream: NetReturnStream,
    sizing_config: GrowthSizingConfig,
) -> GrowthSizingResult:
    """Size a segment strictly from returns known before its deployment window."""
    unit_returns = discovery_stream.net.to_numpy(dtype=np.float64)
    unit_returns = unit_returns[np.isfinite(unit_returns)]
    return solve_growth_optimal_risk(unit_returns, sizing_config)


def _stressed_construction(spec: NetConstructionSpec) -> NetConstructionSpec:
    """1.5x fee, 2.0x slippage stress construction; cadence and band unchanged."""
    costs = spec.costs
    return NetConstructionSpec(
        rebalance_bars=spec.rebalance_bars,
        no_trade_band=spec.no_trade_band,
        costs=CostModel(
            fee_rate=costs.fee_rate * 1.5,
            slippage_rate=costs.slippage_rate * 2.0,
        ),
    )


def _family_tiebreak(family: _FamilyScreen) -> tuple[float, tuple[int, ...]]:
    score = family.chosen_score if family.chosen_score is not None else float("-inf")
    return score, tuple(-ord(character) for character in family.strategy_id)


def _diagnostic_falsification(family: _FamilyScreen) -> FalsificationVerdict | None:
    """Compose a fail-closed verdict when no family survives the discovery plateau."""
    if family.chosen_parameter is None or family.chosen_score is None:
        return None
    return evaluate_falsification(
        parameter_scores=family.parameter_scores,
        chosen_parameter=float(family.chosen_parameter),
        oos_t_stat=0.0,
        family_size=FAMILY_SIZE,
        dev_score=family.chosen_score,
        holdout_score=0.0,
        fold_gate_pass=False,
        config=FalsificationConfig(),
    )


def _scorecard(
    entries: tuple[GrowthCandidateScoreEntry, ...],
    *,
    selected: _FamilyScreen | None,
    reason: str | None,
) -> GrowthCandidateScorecard:
    return GrowthCandidateScorecard(
        family_size=FAMILY_SIZE,
        entries=entries,
        selected_strategy_id=selected.strategy_id if selected is not None else None,
        selected_parameter=selected.chosen_parameter if selected is not None else None,
        reason=reason,
    )


def run_growth_engine_evaluation(request: GrowthEngineEvaluationRequest) -> GrowthEngineReport:
    """Rolling, context-conditioned multi-sleeve growth-engine evaluation.

    ``resolve_evaluation_end`` -> ``earliest_admissible_start`` ->
    ``build_universe_schedule`` -> for every outer deployment segment: frozen
    discovery on the immediately preceding 12 calendar months, plateau selection
    of at most one parameter per source family, discovery-only block-aware
    lower-confidence sleeve admission, and a pre-decision context router that
    freezes identity/parameters/sleeve weights/risk before each frozen
    3-month deployment segment -> stitched out-of-sample net returns ->
    ``evaluate_falsification`` (plateau, Bonferroni multiplicity
    ``family_size=12``, equal-duration 6-month fold concentration, symbol
    holdout) -> realised-risk overlay -> ``compose_promotion_verdict`` from real
    observation, fold, stress, and symbol-holdout reliability evidence.
    Fail-closed: insufficient data/span/breadth, a missing fold, an infeasible
    risk, or any observation/fold/stress/multiplicity/holdout gate failure
    yields ``NO_ADMISSIBLE_ALPHA`` with a flat CASH equity curve, zero trades,
    an empty promotion result, and a deterministic candidate scorecard.
    """
    end = resolve_evaluation_end(request.end, unseal_holdout=request.unseal_holdout)
    if end is None:
        end_ts = pd.Timestamp.now(tz="UTC")
    elif isinstance(end, pd.Timestamp):
        end_ts = end.tz_convert("UTC")
    else:
        end_ts = pd.Timestamp(end, tz="UTC")

    coverage, liquidity, frames = _build_universe_inputs(
        _list_symbols(), request.universe, request.start, end,
    )
    if not coverage:
        return _no_admissible_alpha(
            request, end_ts, {}, start=None, falsification=None, sizing=None,
            scorecard=_empty_scorecard("insufficient_data"),
        )

    data_start = min(cov.first_bar for cov in coverage)
    data_last = max(cov.last_bar for cov in coverage)
    rebalance_dates = _build_rebalance_dates(data_start, min(end_ts, data_last))

    start = earliest_admissible_start(coverage, rebalance_dates, request.universe)
    if start is None:
        return _no_admissible_alpha(
            request, end_ts, {}, start=None, falsification=None, sizing=None,
            scorecard=_empty_scorecard("no_admissible_start"),
        )

    discovery_start = start - pd.DateOffset(months=DISCOVERY_MONTHS)
    schedule_start = min(discovery_start, rebalance_dates[0])

    full_schedule = build_universe_schedule(
        coverage, liquidity,
        [date for date in rebalance_dates if date >= schedule_start],
        request.universe,
    )

    backfill_candidates = derive_backfill_candidates(
        coverage, liquidity, [date for date in rebalance_dates if date >= start],
        request.universe,
    )
    _logger.info(
        "[EVAL] start=%s discovery_start=%s backfill_candidates=%d universe_dates=%d",
        start, discovery_start, len(backfill_candidates), len(full_schedule),
    )

    all_symbols = sorted({sym for roster in full_schedule.values() for sym in roster})
    if not all_symbols:
        return _no_admissible_alpha(
            request, end_ts, full_schedule, start=start, falsification=None, sizing=None,
            scorecard=_empty_scorecard("empty_universe"),
        )

    px, fwd, taker = _build_price_panel(all_symbols, frames, schedule_start, end_ts)

    dev_members = {
        sym for sym in all_symbols
        if symbol_partition(sym, request.universe.dev_fraction) == "dev"
    }
    holdout_members = {
        sym for sym in all_symbols
        if symbol_partition(sym, request.universe.dev_fraction) == "holdout"
    }
    holdout_schedule = _subset_schedule(full_schedule, holdout_members)

    schedule = _apply_scope(full_schedule, request.symbol_scope, request.universe)
    panel_symbols = sorted({sym for roster in schedule.values() for sym in roster})
    if not panel_symbols:
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start, falsification=None, sizing=None,
            scorecard=_empty_scorecard("empty_panel"),
        )

    segments = build_rolling_segments(sorted(schedule))
    segments = [segment for segment in segments if segment.deployment_dates[0] >= start]
    if not enough_deployment_folds(segments):
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start, falsification=None, sizing=None,
            scorecard=_empty_scorecard("insufficient_rolling_span"),
        )

    grid = px.index
    month_key = grid.normalize() - pd.to_timedelta(grid.day - 1, unit="D")
    settled_funding = _build_settled_funding(all_symbols, grid)
    market_returns, breadth = _market_context_inputs(fwd, panel_symbols)

    sizing_config = GrowthSizingConfig(
        risk_grid=(0.001, 0.005, 0.02),
        horizon_years=5.0,
        n_paths=500,
    )
    router_config = dataclasses.replace(ReliabilityGateConfig(), n_bootstrap=500)

    stitched_nets: list[pd.Series] = []
    stitched_weights: list[pd.DataFrame] = []
    stress_nets: list[pd.Series] = []
    stress_weights: list[pd.DataFrame] = []
    scorecard_entries: tuple[GrowthCandidateScoreEntry, ...] = ()
    families: tuple[_FamilyScreen, ...] = ()
    finalist: _FamilyScreen | None = None
    finalist_sizing: GrowthSizingResult | None = None
    finalist_context: ContextState | None = None
    finalist_dev_score = 0.0
    deployed_sleeve: str | None = None
    binding: str | None = None
    had_passing_family = False
    had_admitted_sleeve = False

    for segment in segments:
        disc_schedule = {
            date: schedule[date] for date in segment.discovery_dates if date in schedule
        }
        dep_schedule = {
            date: schedule[date] for date in segment.deployment_dates if date in schedule
        }
        discovery_bars = grid[month_key.isin(set(disc_schedule))]
        deployment_bars = grid[month_key.isin(set(dep_schedule))]
        if len(discovery_bars) < 2 or len(deployment_bars) < 2:
            binding = "insufficient_bars"
            continue

        entries, families = _screen_discovery_candidates(
            disc_schedule, px, fwd, taker, settled_funding, discovery_bars,
            request.universe.max_positions,
        )
        scorecard_entries = entries
        passing = [f for f in families if f.passed and f.chosen_parameter is not None]
        if not passing:
            binding = "no_passing_family"
            continue
        had_passing_family = True

        evidence: dict[str, _SleeveEvidence] = {}
        for family in passing:
            sleeve_ev = _discovery_sleeve_evidence(
                family, disc_schedule, px, fwd, taker, settled_funding,
                request.construction, grid, discovery_bars,
                request.universe.max_positions, router_config,
            )
            if sleeve_ev is not None and sleeve_ev.admission_lcb > 0.0:
                evidence[family.strategy_id] = sleeve_ev
        if not evidence:
            binding = "no_admitted_sleeve"
            continue
        had_admitted_sleeve = True

        vol_threshold, breadth_threshold, states = _discovery_context_evidence(
            market_returns, breadth, discovery_bars, grid,
        )
        deploy_start_index = int(grid.get_indexer([deployment_bars[0]])[0])
        context = _segment_context_state(
            market_returns, breadth, deploy_start_index, vol_threshold, breadth_threshold,
        )
        if context is None:
            binding = "no_context"
            continue

        for sid, sleeve_ev in evidence.items():
            evidence[sid] = _SleeveEvidence(
                family=sleeve_ev.family,
                discovery_stream=sleeve_ev.discovery_stream,
                discovery_net=sleeve_ev.discovery_net,
                discovery_realized=sleeve_ev.discovery_realized,
                discovery_net_sharpe=sleeve_ev.discovery_net_sharpe,
                admission_lcb=sleeve_ev.admission_lcb,
                context_lcb=_context_sleeve_lcb(
                    sleeve_ev, context, states, grid, router_config,
                ),
            )
        chosen_sid = causal_router({
            sid: sleeve_ev.context_lcb for sid, sleeve_ev in evidence.items()
        })
        if chosen_sid is None:
            binding = "no_context_sleeve"
            continue
        chosen_evidence = evidence[chosen_sid]
        chosen = chosen_evidence.family
        if chosen.chosen_parameter is None:
            binding = "no_chosen_parameter"
            continue

        screen = screen_growth_strategy_weights(
            chosen.strategy_id, chosen.chosen_parameter, dep_schedule, px, taker,
            settled_funding, request.universe.max_positions,
        )
        if screen.status == "DATA_INVALID":
            binding = "finalist_data_invalid"
            continue
        dep_stream = _sleeve_net_stream(
            screen, fwd, request.construction, settled_funding, grid,
        )
        dep_net = dep_stream.net.loc[deployment_bars].dropna()
        if len(dep_net) < 2:
            binding = "no_deployment_returns"
            continue

        sizing = _segment_sizing(chosen_evidence.discovery_stream, sizing_config)
        if sizing.selected_risk is None:
            binding = "infeasible_risk"
            continue

        realized = dep_stream.realized_weights.loc[dep_net.index]
        scaled_net, scaled_weights = apply_realised_risk_overlay(
            dep_net, realized, sizing.selected_risk, sizing_config.reference_risk,
        )

        stress_construction = _stressed_construction(request.construction)
        delayed_weights = screen.weights.shift(1).fillna(0.0)
        forward_funding = (
            _build_forward_funding(
                settled_funding, grid, list(delayed_weights.columns),
            )
            if registry_definition(chosen.strategy_id).requires_funding
            else None
        )
        stress_stream = _compute_stream(
            delayed_weights,
            fwd[list(delayed_weights.columns)],
            stress_construction,
            forward_funding,
        )
        stress_net = stress_stream.net.loc[deployment_bars].dropna()
        if len(stress_net) < 2:
            binding = "no_stress_returns"
            continue

        stitched_nets.append(scaled_net)
        stitched_weights.append(scaled_weights)
        stress_nets.append(stress_net)
        stress_weights.append(stress_stream.realized_weights.loc[stress_net.index])
        finalist = chosen
        finalist_sizing = sizing
        finalist_context = context
        finalist_dev_score = evidence[chosen_sid].discovery_net_sharpe
        deployed_sleeve = chosen.strategy_id
        binding = None

    if not stitched_nets:
        diagnostic = next(
            (f for f in families if f.chosen_parameter is not None), None,
        )
        if not had_passing_family:
            reason = "no_passing_family"
        elif not had_admitted_sleeve:
            reason = "no_admitted_sleeve"
        else:
            reason = binding or "no_deployed_segment"
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start,
            falsification=(
                _diagnostic_falsification(diagnostic) if diagnostic is not None else None
            ),
            sizing=None,
            scorecard=_scorecard(
                scorecard_entries, selected=diagnostic, reason=reason,
            ),
            selected_strategy=diagnostic.strategy_id if diagnostic is not None else None,
        )

    oos_net = pd.concat(stitched_nets)
    oos_net = oos_net[~oos_net.index.duplicated(keep="last")].sort_index()
    oos_weights = pd.concat(stitched_weights)
    oos_weights = oos_weights[~oos_weights.index.duplicated(keep="last")].sort_index()
    closed_trades = count_closed_trades(oos_weights)
    deployed_equity = (1.0 + oos_net).cumprod()

    oos_t_stat = _oos_t_stat(oos_net)
    try:
        folds = compute_equal_duration_fold_distribution(
            deployed_equity, ReliabilityGateConfig(), fold_duration=_FOLD_DURATION,
        )
        fold_gate_pass = folds.gate_pass
    except ValueError as exc:
        _logger.warning("[EVAL] fold evidence fail-closed reason=%s", exc)
        folds = FoldDistributionResult(
            n_folds=0, median_fold_cagr=0.0, worst_fold_cagr=0.0,
            median_fold_calmar=0.0, max_period_contribution=0.0, gate_pass=False,
        )
        fold_gate_pass = False

    observation = compute_equity_reliability_gate(deployed_equity, closed_trades)

    stress_net = pd.concat(stress_nets)
    stress_net = stress_net[~stress_net.index.duplicated(keep="last")].sort_index()
    stress_weights_concat = pd.concat(stress_weights)
    stress_weights_concat = stress_weights_concat[
        ~stress_weights_concat.index.duplicated(keep="last")
    ].sort_index()
    stress_equity = (1.0 + stress_net).cumprod()
    stress_gate = compute_equity_reliability_gate(
        stress_equity, count_closed_trades(stress_weights_concat),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )

    assert finalist is not None
    assert finalist.chosen_parameter is not None
    holdout_gate: ReliabilityGateResult | None = None
    holdout_score = 0.0
    if holdout_schedule and any(roster for roster in holdout_schedule.values()):
        holdout_screen = screen_growth_strategy_weights(
            finalist.strategy_id, finalist.chosen_parameter, holdout_schedule, px, taker,
            settled_funding, request.universe.max_positions,
        )
        if holdout_screen.status == "SCREENED":
            holdout_stream = _sleeve_net_stream(
                holdout_screen, fwd, request.construction, settled_funding, grid,
            )
            holdout_bars = grid[month_key.isin(set(holdout_schedule))]
            holdout_net = holdout_stream.net.loc[holdout_bars].dropna()
            if len(holdout_net) >= 2:
                holdout_score = _gross_sharpe(holdout_net)
                holdout_equity = (1.0 + holdout_net).cumprod()
                holdout_gate = compute_equity_reliability_gate(
                    holdout_equity,
                    count_closed_trades(holdout_stream.realized_weights.loc[holdout_net.index]),
                )

    falsification = evaluate_falsification(
        parameter_scores=finalist.parameter_scores,
        chosen_parameter=float(finalist.chosen_parameter),
        oos_t_stat=oos_t_stat,
        family_size=FAMILY_SIZE,
        dev_score=finalist_dev_score,
        holdout_score=holdout_score,
        fold_gate_pass=fold_gate_pass,
        config=FalsificationConfig(),
    )

    if not falsification.passed:
        # The falsification verdict already composes plateau, multiplicity, the
        # equal-duration fold-concentration gate, and the symbol-holdout
        # retention gate, so its binding constraint is reported directly.
        binding_gate = falsification.binding_constraint
    elif observation.verdict != "PASS":
        binding_gate = "observation"
    elif stress_gate.verdict != "PASS":
        binding_gate = "stress"
    elif holdout_gate is None or holdout_gate.verdict != "PASS":
        binding_gate = "symbol_holdout"
    else:
        binding_gate = "none"

    scorecard = _scorecard(scorecard_entries, selected=finalist, reason=None)
    active_mask = oos_weights.to_numpy(dtype=np.float64) != 0.0
    active_symbols = (
        float(np.mean(np.count_nonzero(active_mask, axis=1)))
        if active_mask.size else 0.0
    )
    weight_delta = np.abs(np.diff(oos_weights.to_numpy(dtype=np.float64), axis=0))
    turnover = float(np.mean(weight_delta.sum(axis=1))) if weight_delta.size else 0.0
    audit_required = intrabar_audit_required(
        competing_intrabar_exits=1, stop_atr_mult=3.0,
    )
    _logger.info(
        "[EVAL] symbols=%d traded=%d dev=%d holdout=%d source=%s context=%s "
        "fold=%d active_symbols=%.2f turnover=%.4f net_t=%.3f lcb=%.4f "
        "stress=%s holdout=%s binding=%s audit_required=%s",
        len(all_symbols), len(panel_symbols), len(dev_members), len(holdout_members),
        deployed_sleeve, str(finalist_context), folds.n_folds, active_symbols,
        turnover, oos_t_stat, observation.lcb90_cagr, stress_gate.verdict,
        holdout_gate.verdict if holdout_gate is not None else "MISSING",
        binding_gate, audit_required,
    )

    if binding_gate != "none":
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start,
            falsification=falsification, sizing=finalist_sizing,
            scorecard=_scorecard(
                scorecard_entries, selected=finalist, reason=binding_gate,
            ),
            selected_strategy=finalist.strategy_id,
        )

    equity = request.initial_equity * deployed_equity
    promotion = compose_promotion_verdict(observation, folds, stress_gate, holdout_gate)
    report = GrowthEngineReport(
        status="PASS",
        equity=equity,
        trades=_empty_trades(),
        falsification=falsification,
        sizing=finalist_sizing,
        universe_schedule=schedule,
        start=start,
        promotion=promotion,
        scorecard=scorecard,
        selected_strategy=finalist.strategy_id,
    )
    record = _record_run(request, report, end) if request.log_run else None
    return dataclasses.replace(report, record=record)
