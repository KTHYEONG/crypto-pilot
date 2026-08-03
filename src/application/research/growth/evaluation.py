from __future__ import annotations

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
from src.research.contracts import GrowthEngineEvaluationRequest
from src.research.evaluation.falsification import FalsificationConfig, evaluate_falsification, evaluate_parameter_plateau
from src.research.evaluation.policy import resolve_evaluation_end
from src.research.evaluation.promotion import PromotionResult, compose_promotion_verdict
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    compute_equal_duration_fold_distribution,
)
from src.research.execution.intrabar_audit import intrabar_audit_required
from src.research.portfolio.growth_strategy_library import (
    FAMILY_SIZE,
    STRATEGY_REGISTRY,
    align_funding_bars,
    registry_definition,
    screen_growth_strategy_weights,
)
from src.research.portfolio.net_construction import NetConstructionSpec, compute_net_return_stream
from src.research.risk.growth_sizing import GrowthSizingConfig, solve_growth_optimal_risk
from src.research.universe.pit_universe import PitUniverseSpec, SymbolCoverage, build_universe_schedule, earliest_admissible_start, symbol_partition, derive_backfill_candidates

if TYPE_CHECKING:
    from src.research.evaluation.falsification import FalsificationVerdict
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


def _split_dev_schedule(
    dev_schedule: dict[pd.Timestamp, tuple[str, ...]],
) -> tuple[dict[pd.Timestamp, tuple[str, ...]], dict[pd.Timestamp, tuple[str, ...]]]:
    """Split the dev evaluation period chronologically at its midpoint.

    The split uses whole monthly rebalance boundaries so a monthly roster never
    straddles the discovery/qualification seam.  The discovery half alone is
    allowed to participate in family selection.
    """
    dates = sorted(dev_schedule)
    mid = len(dates) // 2
    return (
        {date: dev_schedule[date] for date in dates[:mid]},
        {date: dev_schedule[date] for date in dates[mid:]},
    )


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
) -> tuple[tuple[GrowthCandidateScoreEntry, ...], tuple[_FamilyScreen, ...]]:
    """Screen every frozen family/window on dev-discovery evidence only."""
    entries: list[GrowthCandidateScoreEntry] = []
    families: list[_FamilyScreen] = []
    for definition in STRATEGY_REGISTRY:
        scores: dict[float, float] = {}
        invalid_windows: list[int] = []
        for window in definition.windows:
            screen = screen_growth_strategy_weights(
                definition.strategy_id, window, schedule, px, taker, settled_funding,
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
    """Single orchestration path for the growth-engine evaluation.

    ``resolve_evaluation_end`` -> ``earliest_admissible_start`` ->
    ``build_universe_schedule`` -> frozen dev discovery (every family/window on
    the discovery half) -> dev finalist qualification (net OOS t-stat and gross
    score) + symbol-holdout gross score -> ``evaluate_falsification`` (plateau,
    Bonferroni multiplicity with ``family_size=12``,
    ``compute_equal_duration_fold_distribution`` fold-concentration on the
    qualification equity, symbol holdout) -> ``compute_net_return_stream`` ->
    ``solve_growth_optimal_risk`` -> ``compose_promotion_verdict``.  Fail-closed:
    when the admissible start
    cannot be derived, no family passes the discovery plateau, or the
    falsification verdict fails, the report is ``NO_ADMISSIBLE_ALPHA`` with a
    flat CASH equity curve, zero trades, and a deterministic candidate
    scorecard.  The holdout partition only ever inspects the dev-selected
    finalist.
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

    dates_from_start = [date for date in rebalance_dates if date >= start]

    # full_schedule (unscoped) is the realized universe used to derive the
    # dev/holdout split for falsification; the symbol_scope filter is applied
    # only afterwards, to the trading panel, so a default symbol_scope="dev"
    # restricts capital deployment without silently emptying the holdout side
    # of the falsification's symbol_holdout test.
    full_schedule = build_universe_schedule(coverage, liquidity, rebalance_dates, request.universe)
    full_schedule = {date: roster for date, roster in full_schedule.items() if date >= start}

    backfill_candidates = derive_backfill_candidates(
        coverage, liquidity, dates_from_start, request.universe,
    )
    _logger.info(
        "[EVAL] start=%s backfill_candidates=%d universe_dates=%d",
        start, len(backfill_candidates), len(full_schedule),
    )

    all_symbols = sorted({sym for roster in full_schedule.values() for sym in roster})
    if not all_symbols:
        return _no_admissible_alpha(
            request, end_ts, full_schedule, start=start, falsification=None, sizing=None,
            scorecard=_empty_scorecard("empty_universe"),
        )

    px, fwd, taker = _build_price_panel(all_symbols, frames, start, end_ts)

    dev_members = {
        sym for sym in all_symbols
        if symbol_partition(sym, request.universe.dev_fraction) == "dev"
    }
    holdout_members = {
        sym for sym in all_symbols
        if symbol_partition(sym, request.universe.dev_fraction) == "holdout"
    }
    dev_schedule = _subset_schedule(full_schedule, dev_members)
    holdout_schedule = _subset_schedule(full_schedule, holdout_members)

    schedule = _apply_scope(full_schedule, request.symbol_scope, request.universe)
    panel_symbols = sorted({sym for roster in schedule.values() for sym in roster})
    if not panel_symbols:
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start, falsification=None, sizing=None,
            scorecard=_empty_scorecard("empty_panel"),
        )

    discovery_schedule, qualification_schedule = _split_dev_schedule(dev_schedule)
    if len(discovery_schedule) < 2 or len(qualification_schedule) < 2:
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start, falsification=None, sizing=None,
            scorecard=_empty_scorecard("insufficient_dev_rosters"),
        )

    grid = px.index
    month_key = grid.normalize() - pd.to_timedelta(grid.day - 1, unit="D")
    discovery_bars = grid[month_key.isin(set(discovery_schedule))]
    qualification_bars = grid[month_key.isin(set(qualification_schedule))]

    settled_funding = _build_settled_funding(all_symbols, grid)
    entries, families = _screen_discovery_candidates(
        discovery_schedule, px, fwd, taker, settled_funding, discovery_bars,
    )
    for definition in STRATEGY_REGISTRY:
        # C4 measurement-only diagnostic: the discovery screens carry weights
        # only over discovery bars, so the qualification-period window streams
        # are screened here against the qualification schedule. This pass feeds
        # only the [EVAL] family_correlation log; it never alters FAMILY_SIZE,
        # the multiplicity floor, or any falsification/promotion decision.
        qualification_streams: dict[int, pd.Series] = {}
        for window in definition.windows:
            screen = screen_growth_strategy_weights(
                definition.strategy_id, window, qualification_schedule, px, taker,
                settled_funding,
            )
            if screen.status != "SCREENED":
                continue
            qualification_streams[window] = (
                (screen.weights * fwd).sum(axis=1).loc[qualification_bars]
            )
        correlations = family_window_correlation(qualification_streams)
        if correlations:
            pairs = ",".join(
                f"{left}-{right}:{value:.3f}"
                for (left, right), value in sorted(correlations.items())
            )
            _logger.info(
                "[EVAL] family_correlation strategy=%s windows=%d pairs=%s",
                definition.strategy_id, len(qualification_streams), pairs,
            )

    passing = [f for f in families if f.passed and f.chosen_parameter is not None]
    screened = [f for f in families if f.chosen_parameter is not None]
    if passing:
        selected = max(passing, key=_family_tiebreak)
    elif screened:
        selected = max(screened, key=_family_tiebreak)
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start,
            falsification=_diagnostic_falsification(selected), sizing=None,
            scorecard=_scorecard(entries, selected=selected, reason="no_passing_family"),
            selected_strategy=selected.strategy_id,
        )
    else:
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start, falsification=None, sizing=None,
            scorecard=_scorecard(entries, selected=None, reason="no_screenable_candidate"),
        )

    assert selected.chosen_parameter is not None
    selected_parameter = int(selected.chosen_parameter)
    screen = screen_growth_strategy_weights(
        selected.strategy_id, selected_parameter, schedule, px, taker, settled_funding,
    )
    if screen.status == "DATA_INVALID":
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start, falsification=None, sizing=None,
            scorecard=_scorecard(entries, selected=selected, reason="finalist_data_invalid"),
            selected_strategy=selected.strategy_id,
        )

    target_weights = screen.weights
    forward_returns = fwd[list(target_weights.columns)]
    definition = registry_definition(selected.strategy_id)
    if definition.requires_funding:
        forward_funding = _build_forward_funding(
            settled_funding, grid, list(target_weights.columns),
        )
        stream = _compute_stream(
            target_weights, forward_returns, request.construction, forward_funding,
        )
    else:
        stream = _compute_stream(target_weights, forward_returns, request.construction)

    unit_returns = stream.net.to_numpy(dtype=np.float64)
    unit_returns = unit_returns[np.isfinite(unit_returns)]
    sizing_config = GrowthSizingConfig(
        risk_grid=(0.001, 0.005, 0.02),
        horizon_years=5.0,
        n_paths=500,
    )
    sizing = solve_growth_optimal_risk(unit_returns, sizing_config)

    full_pnl = (target_weights * fwd).sum(axis=1)
    dev_qualification_score = _gross_sharpe(full_pnl.loc[qualification_bars])
    qualification_net = stream.net.loc[qualification_bars]
    oos_t_stat = _oos_t_stat(qualification_net)
    fold_gate_pass = _qualification_fold_gate_pass(qualification_net)

    holdout_screen = screen_growth_strategy_weights(
        selected.strategy_id, selected_parameter, holdout_schedule, px, taker,
        settled_funding,
    )
    if holdout_screen.status == "SCREENED":
        holdout_pnl = (holdout_screen.weights * fwd).sum(axis=1)
        holdout_bars = grid[month_key.isin(set(holdout_schedule))]
        holdout_score = _gross_sharpe(holdout_pnl.loc[holdout_bars])
    else:
        holdout_score = 0.0

    falsification = evaluate_falsification(
        parameter_scores=selected.parameter_scores,
        chosen_parameter=float(selected_parameter),
        oos_t_stat=oos_t_stat,
        family_size=FAMILY_SIZE,
        dev_score=dev_qualification_score,
        holdout_score=holdout_score,
        fold_gate_pass=fold_gate_pass,
        config=FalsificationConfig(),
    )

    audit_required = intrabar_audit_required(
        competing_intrabar_exits=1, stop_atr_mult=3.0,
    )
    _logger.info(
        "[EVAL] symbols=%d traded=%d dev=%d holdout=%d strategy=%s parameter=%s "
        "oos_t=%.3f falsification=%s sizing=%s audit_required=%s",
        len(all_symbols), len(panel_symbols), len(dev_members), len(holdout_members),
        selected.strategy_id, selected_parameter, oos_t_stat,
        falsification.binding_constraint, sizing.binding_constraint, audit_required,
    )

    scorecard = _scorecard(entries, selected=selected, reason=None)
    if not falsification.passed or sizing.selected_risk is None:
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start,
            falsification=falsification, sizing=sizing,
            scorecard=scorecard, selected_strategy=selected.strategy_id,
        )

    equity = request.initial_equity * (1.0 + stream.net).cumprod()
    promotion = compose_promotion_verdict(
        observation=ReliabilityGateResult(
            lcb90_cagr=0.0, lcb95_cagr=0.0, p_negative=0.0, point_cagr=0.0,
            t_stat=falsification.oos_t_stat, trade_count=0, block_size_used=1,
            verdict="PASS" if falsification.passed else "FAIL",
        ),
        folds=FoldDistributionResult(
            n_folds=0, median_fold_cagr=0.0, worst_fold_cagr=0.0,
            median_fold_calmar=0.0, max_period_contribution=0.0,
            gate_pass=sizing.selected_risk is not None,
        ),
        stress=ReliabilityGateResult(
            lcb90_cagr=0.0, lcb95_cagr=0.0, p_negative=0.0, point_cagr=0.0,
            t_stat=0.0, trade_count=0, block_size_used=1, verdict="PASS",
        ),
        holdout=None,
    )

    report = GrowthEngineReport(
        status="PASS",
        equity=equity,
        trades=_empty_trades(),
        falsification=falsification,
        sizing=sizing,
        universe_schedule=schedule,
        start=start,
        promotion=promotion,
        scorecard=scorecard,
        selected_strategy=selected.strategy_id,
    )
    record = _record_run(request, report, end) if request.log_run else None
    return GrowthEngineReport(
        status="PASS",
        equity=equity,
        trades=_empty_trades(),
        falsification=falsification,
        sizing=sizing,
        universe_schedule=schedule,
        start=start,
        promotion=promotion,
        record=record,
        scorecard=scorecard,
        selected_strategy=selected.strategy_id,
    )
