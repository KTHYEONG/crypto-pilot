"""MHS Phase 1 orchestrator: dev-only diagnostics and the Research-GO evidence path.

This module composes the frozen ``src.mhs`` primitives; no alpha, cost,
ranking, liquidity, funding, or inventory arithmetic is reimplemented here.
The target-weight ``cost_response_curve`` is pre-screen only, the strict-proxy
replay + simulated inventory ledger is the primary Research-GO evidence, and
every report separates the two.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.market_data.services.futures_collection import DataCollector
from src.mhs.evaluation import cost_response_curve, phase_diagnostic_metrics, tail_sensitivity_curve
from src.mhs.evaluation import AnchoredPurgedFold, DeploymentReadinessResult, autocorrelation_adjusted_sharpe, synthetic_stress_scenarios
from src.mhs.execution import simulated_inventory_ledger
from src.mhs.execution import strategy_aware_execution_replay
from src.mhs.panel import liquid_half_eligibility, load_base_panel
from src.research.evaluation.policy import resolve_evaluation_end

from src.common.config import FUTURES_DATA_DIR, funding_path
from src.common.errors import DataIntegrityError
from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.contracts import (
    PHASE_1_BOOK_BLEND_WEIGHTS,
    PHASE_1_BOOK_SPECS,
    BookSpec,
    ExecutionSpec,
)
from src.mhs.evaluation import (
    CostResponsePoint,
    PhaseDiagnosticResult,
    TailSensitivityResult,
    compute_deployment_readiness,
    phase_1_anchored_purged_folds,
    required_cost_tiers,
)
from src.mhs.execution import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult, bar_funding_panel, mhs_ledger_pnl
from src.mhs.horizons import efficiency_ratio, horizon_log_return, realized_vol
from src.research.evaluation.policy import HOLDOUT_CUTOFF
from src.market_data.storage.loaders import load_funding_rates

__all__ = ["simulated_inventory_ledger"]

_logger = logging.getLogger("MhsHorizonDiagnostic")

MHS_DISCOVERY_START = pd.Timestamp("2021-01-01", tz="UTC")
_MHS_FEATURE = "multi_horizon_market_state"
_PERIODS_PER_YEAR_1H = 365.0 * 24
_BOOTSTRAP_SEED = 20260807
_BOOTSTRAP_REPLICATES = 2000
_BOOTSTRAP_MEAN_BLOCK = 168

# Frozen strict-proxy Research-GO criterion (spec §3.3): the primary
# autocorrelation-adjusted Sharpe must be >= 0.6 for a candidate to pass.
MHS_GO_PRIMARY_SHARPE_FLOOR = 0.6

MHS_ARTIFACT_SCHEMA_VERSION = 1

MHS_GO_REASON_INCOMPLETE_FOLD = "INCOMPLETE_ANCHORED_FOLD"
MHS_GO_REASON_INVALID_PRIMARY = "INVALID_PRIMARY_LEDGER"
MHS_GO_REASON_NONFINITE_EQUITY = "NONFINITE_EQUITY"
MHS_GO_REASON_EXECUTION_GAP = "RELEVANT_EXECUTION_DATA_GAP"
MHS_GO_REASON_PRIMARY_SHARPE = "PRIMARY_AUTOCORR_SHARPE_BELOW_0_6"
MHS_GO_REASON_STRESS_SHARPE = "STRESS_SHARPE_NOT_POSITIVE"
MHS_GO_REASON_CAPITAL_BREACH = "CAPITAL_INVARIANT_BREACH"
MHS_GO_REASON_UNSPECIFIED_POLICY = "UNSPECIFIED_POLICY"


@dataclass(frozen=True, slots=True)
class MhsDiagnosticRequest:
    """Immutable request; the CLI carries ``--start``/``--end``/``--mark-mode``/``--no-log-run``.

    ``partition`` is forced to ``'dev'`` (a holdout request raises); ``data_root``
    allows tests to run against a synthetic market. ``mark_mode`` is a
    reproducibility parameter: ``cache_required`` builds the causal mark panel
    and fails closed on a missing/non-positive decision mark or invalid strict
    ledger, while ``ohlcv_close_fallback`` deliberately passes ``None`` and
    reports ``OHLCV_CLOSE_FALLBACK`` (fixtures and explicit comparison runs
    only). Collection is a separate, resumable cache-building operation and is
    never triggered by a sealed evaluation run.
    """

    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    partition: Literal["dev", "holdout", "all"] = "dev"
    data_root: str | None = None
    mark_mode: Literal["cache_required", "ohlcv_close_fallback"] = "cache_required"
    execution_timeframe: Literal["1m", "5m"] = "5m"
    execution_universe_size: int = 30
    log_run: bool = True

    def __post_init__(self) -> None:
        if self.partition not in ("dev", "holdout", "all"):
            raise ValueError(f"unknown partition '{self.partition}'")
        if self.mark_mode not in ("cache_required", "ohlcv_close_fallback"):
            raise ValueError(f"unknown mark_mode '{self.mark_mode}'")
        if self.execution_timeframe not in ("1m", "5m"):
            raise ValueError(f"unknown execution_timeframe '{self.execution_timeframe}'")
        if self.execution_universe_size < 8:
            raise ValueError("execution_universe_size must be >= 8")


@dataclass(frozen=True, slots=True)
class MhsBookReport:
    name: str
    band: str
    horizon_hours: int
    step_hours: int
    tranche_count: int
    n_symbols: int
    phase: PhaseDiagnosticResult
    prescreen: dict[float, CostResponsePoint]
    tail: TailSensitivityResult
    primary: StrategyExecutionReplayResult
    stress: StrategyExecutionReplayResult
    primary_autocorr_sharpe: float
    primary_naive_sharpe: float
    primary_net_ann: float
    primary_geometric_cagr: float
    primary_max_drawdown: float
    primary_annualized_turnover: float
    stress_naive_sharpe: float


@dataclass(frozen=True, slots=True)
class MhsResearchGoResult:
    """Machine-readable Research-GO gate decision built from the fold evidence.

    ``eligible`` is false unless every anchored fold passed and no policy gate
    is left unspecified; the exact blocking reasons are carried as stable codes.
    """

    eligible: bool
    reason_codes: tuple[str, ...]
    evaluated_folds: int
    folds_passed: int


@dataclass(frozen=True, slots=True)
class MhsFoldReport:
    """One independently flat anchored-fold replay over the blend book.

    ``strict``/``stress`` are ``None`` for an incomplete fold; ``failures``
    carries the stable reason codes that blocked this fold's evidence.
    """

    fold_index: int
    validation_start: str
    validation_end: str
    strict: StrategyExecutionReplayResult | None
    stress: StrategyExecutionReplayResult | None
    primary_valid: bool
    primary_autocorr_sharpe: float
    primary_naive_sharpe: float
    primary_net_ann: float
    primary_geometric_cagr: float
    primary_max_drawdown: float
    stress_naive_sharpe: float
    decision_intents: int
    termination_counts: dict[str, int]
    failures: tuple[str, ...]
    strict_elapsed_seconds: float
    stress_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class MhsHorizonDiagnosticReport:
    feature: str
    status: str
    start: str
    end: str
    resolved_end: str
    partition: str
    execution_tiers_bps: tuple[float, ...]
    books: dict[str, MhsBookReport]
    blend: MhsBookReport | None
    blend_target_gross: float
    blend_cash_fraction: float
    eligible_symbols: int
    trials_attempted: int
    deflated_sharpe_ratio: float | None
    xs_rank_ic: dict[str, float]
    date_clustered_regression: dict[str, float]
    horizon_diagnostics: dict[str, float]
    bootstrap_ci: tuple[float, float] | None
    placebo_sharpe_percentile: float | None
    deployment_readiness: DeploymentReadinessResult
    synthetic_stress: dict[str, dict[str, Any]]
    participation_warnings: dict[str, float]
    termination_counts: dict[str, int]
    unsupported_assumptions: tuple[str, ...]
    anchored_folds: tuple[AnchoredPurgedFold, ...]
    folds: tuple[MhsFoldReport, ...]
    research_go: MhsResearchGoResult
    fill_source: str
    mark_source: str
    execution_timeframe: str
    execution_universe_size: int
    execution_symbols: tuple[str, ...]
    run_elapsed_seconds: float

    def to_payload(self) -> Any:
        return _jsonable(dataclasses.asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        if not value.index.is_unique:
            return _jsonable(value.to_dict(orient="records"))
        return {
            str(k): _jsonable(v)
            for k, v in value.astype(object).to_dict(orient="index").items()
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _load_funding_series(symbols: list[str]) -> dict[str, pd.Series]:
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        path = funding_path(sym)
        if not path.exists():
            continue
        try:
            rates = load_funding_rates(str(path))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[MHS] funding load failed symbol=%s error=%s", sym, exc)
            continue
        if len(rates):
            series[sym] = rates
    return series


def _pit_execution_mask(
    quote_volume: pd.DataFrame,
    eligible: pd.DataFrame,
    universe_size: int,
) -> pd.DataFrame:
    """Select the PIT top-volume execution roster without changing signals."""
    trailing = quote_volume.rolling(720, min_periods=720).mean()
    ranked = trailing.where(eligible).rank(axis=1, ascending=False, method="first")
    return ranked.le(universe_size).fillna(False)


def _load_minute_frames(
    root: str, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp,
    timeframe: Literal["1m", "5m"],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    for sym in symbols:
        path = os.path.join(root, timeframe, f"{sym}.parquet")
        if not os.path.exists(path):
            continue
        table = pq.read_table(
            path,
            columns=["timestamp", "high", "low", "close", "quote_vol"],
            filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
        )
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        frame = pd.DataFrame(
            {
                c: table.column(c).to_numpy().astype("float64")
                for c in ("high", "low", "close", "quote_vol")
            },
            index=idx,
        )
        frame = frame[(frame.index >= start) & (frame.index <= end)]
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if not frame.empty:
            frames[sym] = frame
    return frames


def _align_minute_frames(
    frames: dict[str, pd.DataFrame], timeframe: Literal["1m", "5m"],
    start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not frames:
        return None
    if start >= end:
        return None
    # The requested evaluation grid is the replay grid. A late listing is kept
    # as NaN on that grid and never trims the global start, so the replay
    # horizon is never shortened by the union of first-observed timestamps.
    grid = pd.date_range(
        start, end,
        freq={"1m": "1min", "5m": "5min"}[timeframe],
        tz="UTC",
    )
    highs = pd.DataFrame({s: f["high"] for s, f in frames.items()}).reindex(grid)
    lows = pd.DataFrame({s: f["low"] for s, f in frames.items()}).reindex(grid)
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()}).reindex(grid)
    quote_vol = pd.DataFrame({s: f["quote_vol"] for s, f in frames.items()}).reindex(grid)
    return highs, lows, closes, quote_vol


def _book_weights(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    spec: BookSpec,
    step_grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    sig = horizon_log_return(log_close, spec.horizon_hours)
    sig_step = sig.reindex(step_grid)
    el_step = eligible.reindex(step_grid)
    weights = rank_weight_book(sig_step, el_step, spec.band.sign, spec.min_symbols)
    return phase_tranche_book(weights, spec.tranche_count())


def _phase_diagnostics(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    spec: BookSpec,
) -> PhaseDiagnosticResult:
    phase_nets: dict[int, pd.Series] = {}
    for offset in range(spec.step_hours):
        phase_grid = grid_1h[offset :: spec.step_hours]
        sig = horizon_log_return(log_close, spec.horizon_hours).reindex(phase_grid)
        el = eligible.reindex(phase_grid)
        weights = rank_weight_book(sig, el, spec.band.sign, spec.min_symbols)
        weights_1h = weights.reindex(grid_1h).ffill().fillna(0.0)
        net, _turnover = mhs_ledger_pnl(weights_1h, opens, bar_funding, 8.0)
        phase_nets[offset] = net
    return phase_diagnostic_metrics(phase_nets, _PERIODS_PER_YEAR_1H)


def _xs_rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> dict[str, float]:
    common = signal.index.intersection(fwd.index)
    ics: dict[pd.Timestamp, float] = {}
    for t in common:
        row_sig = signal.loc[t]
        row_fwd = fwd.loc[t]
        valid = row_sig.notna() & row_fwd.notna()
        if int(valid.sum()) < 5:
            continue
        s = row_sig[valid].rank()
        f = row_fwd[valid].rank()
        if s.std() == 0 or f.std() == 0:
            continue
        ics[t] = float(np.corrcoef(s, f)[0, 1])
    if not ics:
        return {}
    series = pd.Series(ics)
    n_dates = len(series)
    mean_ic = float(series.mean())
    sd = float(series.std(ddof=1)) if n_dates > 1 else 0.0
    t_stat = mean_ic / (sd / np.sqrt(n_dates)) if sd > 0 else float("nan")
    return {"n_dates": n_dates, "mean_ic": mean_ic, "t_stat": t_stat}


def _date_clustered_ols(fwd: pd.DataFrame, past: pd.DataFrame) -> dict[str, float]:
    """Pooled panel regression ``fwd ~ past`` with date-clustered standard errors."""
    x_vals: list[float] = []
    y_vals: list[float] = []
    cluster_ids: list[int] = []
    date_to_id: dict[pd.Timestamp, int] = {}
    for t in past.index.intersection(fwd.index):
        row_past = past.loc[t]
        row_fwd = fwd.loc[t]
        valid = row_past.notna() & row_fwd.notna()
        for sym in past.columns:
            if sym not in valid.index or not bool(valid[sym]):
                continue
            day = pd.Timestamp(t).normalize()
            if day not in date_to_id:
                date_to_id[day] = len(date_to_id)
            x_vals.append(float(row_past[sym]))
            y_vals.append(float(row_fwd[sym]))
            cluster_ids.append(date_to_id[day])
    n = len(x_vals)
    if n < 10:
        return {"n": n, "past_beta": float("nan"), "past_t": float("nan")}
    design = np.column_stack([np.ones(n), np.asarray(x_vals, dtype="float64")])
    outcome = np.asarray(y_vals, dtype="float64")
    inv_xtx = np.linalg.inv(design.T @ design)
    beta = inv_xtx @ (design.T @ outcome)
    resid = outcome - design @ beta
    cids = np.asarray(cluster_ids)
    meat = np.zeros((2, 2))
    for c in np.unique(cids):
        cluster_mask = cids == c
        score = design[cluster_mask].T @ resid[cluster_mask]
        meat += np.outer(score, score)
    cov = inv_xtx @ meat @ inv_xtx
    se = np.sqrt(np.diag(cov))
    t_beta = beta[1] / se[1] if se[1] > 0 else float("nan")
    return {"n": n, "n_dates": len(date_to_id), "past_beta": float(beta[1]), "past_t": float(t_beta)}


def _bootstrap_ci(net: pd.Series, n_replicates: int, mean_block: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = net.to_numpy(dtype="float64")
    n = len(arr)
    means: list[float] = []
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    for _r in range(n_replicates):
        blocks: list[float] = []
        while len(blocks) < n:
            start = int(rng.integers(0, n))
            length = 1
            while length < n and rng.random() > p_block:
                length += 1
            length = min(length, n - len(blocks))
            blocks.extend(arr[start : start + length].tolist())
        means.append(float(np.mean(blocks[:n])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _placebo_sharpe_percentile(
    signal: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    spec: BookSpec,
    observed_sharpe: float,
    n_placebos: int,
    seed: int,
) -> float | None:
    rng = np.random.default_rng(seed)
    ranks: list[float] = []
    cols = list(signal.columns)
    sig_step = signal.reindex(grid_1h)
    el_step = eligible.reindex(grid_1h)
    for _p in range(n_placebos):
        perm = rng.permutation(len(cols))
        shuffled = sig_step.copy()
        permuted_cols = [cols[i] for i in perm]
        shuffled.columns = permuted_cols
        el_shuffled = el_step.copy()
        el_shuffled.columns = permuted_cols
        weights_p = rank_weight_book(shuffled, el_shuffled, spec.band.sign, spec.min_symbols)
        weights_p = phase_tranche_book(weights_p, spec.tranche_count())
        weights_1h = weights_p.reindex(grid_1h).ffill().fillna(0.0)
        try:
            net, _t = mhs_ledger_pnl(
                weights_1h, opens[permuted_cols], bar_funding[permuted_cols], 8.0,
            )
        except DataIntegrityError:
            # A shuffled placebo can assign a non-zero weight to a symbol
            # outside its lifecycle. Such a placebo is invalid, not evidence
            # that the production ledger should relax its active-cell guard.
            continue
        sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        if sd > 0:
            ranks.append(float(net.mean() / sd * np.sqrt(_PERIODS_PER_YEAR_1H)))
    if not ranks:
        return None
    return float(np.mean([1.0 if observed_sharpe >= r else 0.0 for r in ranks]))


def _participation_warnings(
    replay: StrategyExecutionReplayResult,
    quote_volume_1m: pd.DataFrame,
) -> dict[str, float]:
    if replay.simulated_fills.empty:
        return {}
    fills = replay.simulated_fills
    notional = float((fills["quantity_delta"].abs() * fills["fill_price"]).sum())
    warnings: dict[str, float] = {}
    for window_label, minutes in (("1m", 1), ("30m", 30)):
        total_volume = 0.0
        for _i, row in fills.iterrows():
            t = row["timestamp"]
            sym = row["symbol"]
            if sym not in quote_volume_1m.columns or t not in quote_volume_1m.index:
                continue
            window_end = t + pd.Timedelta(minutes=minutes)
            total_volume += float(quote_volume_1m.loc[t:window_end, sym].sum())
        warnings[f"fill_notional_to_{window_label}_quote_volume"] = (
            notional / total_volume if total_volume > 0 else float("nan")
        )
    daily_notional = notional
    daily_volume = float(quote_volume_1m.sum().sum()) if not quote_volume_1m.empty else 0.0
    warnings["daily_trade_notional_to_daily_quote_volume"] = (
        daily_notional / daily_volume if daily_volume > 0 else float("nan")
    )
    return warnings


def _daily_autocorr_sharpe(ledger: SimulatedInventoryLedgerResult) -> float:
    if ledger.equity.empty:
        return float("nan")
    daily = ledger.equity.resample("1D").last().dropna()
    if len(daily) < 9:
        return float("nan")
    return autocorrelation_adjusted_sharpe(daily.pct_change().dropna(), 365, 7)


def _naive_sharpe(ledger: SimulatedInventoryLedgerResult) -> float:
    net = ledger.net_returns
    if len(net) < 2:
        return float("nan")
    sd = float(net.std(ddof=1))
    if sd <= 0:
        return float("inf") if float(net.mean()) > 0 else float("-inf")
    return float(net.mean() / sd * np.sqrt(_PERIODS_PER_YEAR_1H))


def _mean_ann(series: pd.Series, periods_per_year: float) -> float:
    return float(series.mean()) * periods_per_year if len(series) else float("nan")


def _geometric_cagr(equity: pd.Series) -> float:
    if equity.empty or float(equity.iloc[0]) <= 0 or float(equity.iloc[-1]) <= 0:
        return float("nan")
    n = len(equity)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (_PERIODS_PER_YEAR_1H / n) - 1.0)


def _mdd(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def _assert_cache_required_marks(
    name: str,
    target_replay: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    minute_marks: pd.DataFrame,
) -> None:
    """Fail closed when a replay symbol lacks a finite positive mark at a decision point.

    A mark is required at every decision time where the target weight is
    non-zero. ``minute_marks`` is exactly aligned to the minute closes; a
    missing/non-positive mark at a required decision point raises
    ``DataIntegrityError`` rather than silently falling back to OHLCV closes.
    """
    grid_set = set(minute_marks.index)
    for i, decision_time in enumerate(target_replay.index):
        signal_time = signal_available_at[i]
        for sym in target_replay.columns:
            weight = float(target_replay.loc[decision_time, sym])
            if not np.isfinite(weight) or weight == 0.0:
                continue
            mark = float("nan")
            if decision_time in grid_set:
                mark = float(minute_marks.loc[decision_time, sym])
            if not (np.isfinite(mark) and mark > 0):
                prior = minute_marks.index[(minute_marks.index <= signal_time)]
                if len(prior):
                    mark = float(minute_marks.loc[prior[-1], sym])
            if not (np.isfinite(mark) and mark > 0):
                after = minute_marks.index[minute_marks.index > signal_time]
                if len(after):
                    mark = float(minute_marks.loc[after[0], sym])
            if not (np.isfinite(mark) and mark > 0):
                raise DataIntegrityError(
                    f"cache_required: no finite positive mark for {name} symbol={sym} "
                    f"decision={decision_time}"
                )


def _book_outcome(
    name: str,
    spec: BookSpec,
    n_symbols: int,
    step_grid: pd.DatetimeIndex,
    weights_step: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    phase: PhaseDiagnosticResult,
    minute_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    minute_bar_funding: pd.DataFrame,
    minute_marks: pd.DataFrame | None,
    event_window_bars: int,
    initial_equity: float,
    replay_weights_step: pd.DataFrame | None = None,
) -> MhsBookReport:
    weights_1h = weights_step.reindex(grid_1h).ffill().fillna(0.0)
    cost_grid = tuple(dict.fromkeys((0.0, 2.0, 4.0, 8.0, *required_cost_tiers())))
    prescreen = cost_response_curve(
        weights_1h, opens, bar_funding, cost_grid, _PERIODS_PER_YEAR_1H,
    )

    effective_weights = weights_1h.shift(2).fillna(0.0)
    fwd = opens.pct_change()
    _net, turnover = mhs_ledger_pnl(weights_1h, opens, bar_funding, 8.0)
    tail = tail_sensitivity_curve(
        effective_weights, fwd, turnover, 8.0, _PERIODS_PER_YEAR_1H, event_window_bars,
    )

    target_weights = (replay_weights_step if replay_weights_step is not None else weights_step).reindex(step_grid)
    signal_available_at = step_grid + pd.Timedelta(hours=1)
    highs, lows, closes, _quote_vol = minute_frames
    replay_symbols = [s for s in target_weights.columns if s in highs.columns]
    target_replay = target_weights[replay_symbols]
    if minute_marks is not None:
        _assert_cache_required_marks(name, target_replay, signal_available_at, minute_marks)
    primary = strategy_aware_execution_replay(
        target_replay, signal_available_at, highs, lows, closes, minute_marks,
        minute_bar_funding, initial_equity, "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    stress = strategy_aware_execution_replay(
        target_replay, signal_available_at, highs, lows, closes, minute_marks,
        minute_bar_funding, initial_equity, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
    )
    if minute_marks is not None and not primary.ledger.primary_valid:
            raise DataIntegrityError(
                f"cache_required strict primary ledger invalid for {name}: "
                f"{', '.join(primary.ledger.invalid_reasons)}"
            )
    return MhsBookReport(
        name=name,
        band=spec.band.name,
        horizon_hours=spec.horizon_hours,
        step_hours=spec.step_hours,
        tranche_count=spec.tranche_count(),
        n_symbols=n_symbols,
        phase=phase,
        prescreen=prescreen,
        tail=tail,
        primary=primary,
        stress=stress,
        primary_autocorr_sharpe=_daily_autocorr_sharpe(primary.ledger),
        primary_naive_sharpe=_naive_sharpe(primary.ledger),
        primary_net_ann=_mean_ann(primary.ledger.net_returns, _PERIODS_PER_YEAR_1H),
        primary_geometric_cagr=_geometric_cagr(primary.ledger.equity),
        primary_max_drawdown=_mdd(primary.ledger.equity),
        primary_annualized_turnover=_mean_ann(primary.ledger.fill_turnover, _PERIODS_PER_YEAR_1H),
        stress_naive_sharpe=_naive_sharpe(stress.ledger),
    )


def _incomplete_fold_report(
    fold: AnchoredPurgedFold, fold_index: int, failures: tuple[str, ...],
) -> MhsFoldReport:
    """A fold that could not be replayed, failed closed with its reason codes."""
    return MhsFoldReport(
        fold_index=fold_index,
        validation_start=str(fold.validation_start),
        validation_end=str(fold.validation_end),
        strict=None,
        stress=None,
        primary_valid=False,
        primary_autocorr_sharpe=float("nan"),
        primary_naive_sharpe=float("nan"),
        primary_net_ann=float("nan"),
        primary_geometric_cagr=float("nan"),
        primary_max_drawdown=float("nan"),
        stress_naive_sharpe=float("nan"),
        decision_intents=0,
        termination_counts={},
        failures=tuple(sorted(set(failures))),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )


def _run_anchored_fold(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    initial_equity: float,
    fold_index: int,
) -> MhsFoldReport:
    """One independently flat strict/immediate-taker blend replay per fold.

    The 1h panel spans ``[train_start, validation_end]`` so warm-up history
    feeds features only; the replay decisions and the fresh flat ledger cover
    only the validation window. A fold that cannot be replayed is reported
    (not raised) with machine-readable failure codes.
    """
    try:
        ts = fold.train_start
        vs = fold.validation_start
        ve = fold.validation_end
        panel = load_base_panel(
            root, "1h", ("close", "open", "quote_vol"), ts, ve,
            partition="dev", min_bars=2000,
        )
        close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
        grid_1h = close.index
        symbols = list(close.columns)
        funded = [s for s in symbols if s in funding_by_symbol]
        if not funded:
            return _incomplete_fold_report(fold, fold_index, (MHS_GO_REASON_INCOMPLETE_FOLD,))
        close = close[funded]
        opens = opens[funded]
        quote_vol = quote_vol[funded]
        bar_period = grid_1h[1] - grid_1h[0]
        funding_window = {
            s: funding_by_symbol[s].loc[
                (funding_by_symbol[s].index >= grid_1h[0])
                & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
            ]
            for s in funded
        }
        bar_funding = bar_funding_panel(funding_window, grid_1h)
        aligned_symbols = list(bar_funding.columns)
        if not aligned_symbols:
            return _incomplete_fold_report(fold, fold_index, (MHS_GO_REASON_INCOMPLETE_FOLD,))
        close = close[aligned_symbols]
        opens = opens[aligned_symbols]
        quote_vol = quote_vol[aligned_symbols]
        bar_funding = bar_funding[aligned_symbols]

        eligible = liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
        log_close = np.log(close)
        fast = PHASE_1_BOOK_SPECS["fast_reversal"]
        slow = PHASE_1_BOOK_SPECS["slow_momentum"]
        fast_grid = pd.date_range(ts, ve, freq="6h", tz="UTC")
        slow_grid = pd.date_range(ts, ve, freq="24h", tz="UTC")
        w_fast = _book_weights(log_close, eligible, fast, fast_grid)
        w_slow = _book_weights(log_close, eligible, slow, slow_grid)
        execution_mask = _pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
        w_fast_execution = w_fast.where(
            execution_mask.reindex(w_fast.index).fillna(False), other=0.0,
        )
        w_slow_execution = w_slow.where(
            execution_mask.reindex(w_slow.index).fillna(False), other=0.0,
        )
        blend_1h = (
            PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_execution.reindex(grid_1h).ffill().fillna(0.0)
            + PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_execution.reindex(grid_1h).ffill().fillna(0.0)
        )
        decision_grid = blend_1h.index[(blend_1h.index >= vs) & (blend_1h.index <= ve)]
        target_weights = blend_1h.loc[decision_grid]
        if target_weights.empty:
            return _incomplete_fold_report(fold, fold_index, (MHS_GO_REASON_INCOMPLETE_FOLD,))
        execution_symbols = sorted(
            target_weights.columns[target_weights.ne(0.0).any(axis=0)]
        )
        minute_frames = _align_minute_frames(
            _load_minute_frames(root, execution_symbols, vs, ve, request.execution_timeframe),
            request.execution_timeframe,
            vs, ve,
        )
        if minute_frames is None:
            return _incomplete_fold_report(fold, fold_index, (MHS_GO_REASON_INCOMPLETE_FOLD,))
        highs, _lows, closes, _quote_vol_1m = minute_frames
        minute_grid = highs.index
        minute_marks = (
            DataCollector().load_mark_price_panel(
                list(closes.columns), "1h", minute_grid,
            )
            if request.mark_mode == "cache_required"
            else None
        )
        minute_period = minute_grid[1] - minute_grid[0]
        minute_funding_window = {
            s: funding_by_symbol[s].loc[
                (funding_by_symbol[s].index >= minute_grid[0])
                & (funding_by_symbol[s].index < minute_grid[-1] + minute_period)
            ]
            for s in aligned_symbols
        }
        minute_funding = bar_funding_panel(minute_funding_window, minute_grid)
        minute_funding = minute_funding.reindex(columns=list(closes.columns))

        target_replay = target_weights[list(closes.columns)]
        signal_available_at = target_replay.index + pd.Timedelta(hours=1)
        decision_intents = int(np.isfinite(target_replay.to_numpy()).sum())
        if minute_marks is not None:
            _assert_cache_required_marks("fold", target_replay, signal_available_at, minute_marks)
        primary = strategy_aware_execution_replay(
            target_replay, signal_available_at, highs, _lows, closes, minute_marks,
            minute_funding, initial_equity, "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        stress = strategy_aware_execution_replay(
            target_replay, signal_available_at, highs, _lows, closes, minute_marks,
            minute_funding, initial_equity, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )

        failures: list[str] = []
        equity = primary.ledger.equity
        if not np.isfinite(equity.to_numpy()).all() or not (equity > 0).all():
            failures.append(MHS_GO_REASON_NONFINITE_EQUITY)
        if not primary.ledger.primary_valid:
            failures.append(MHS_GO_REASON_INVALID_PRIMARY)
        if (
            primary.termination_counts.get("MISSING_DATA", 0) > 0
            or primary.termination_counts.get("UNKNOWN_TERMINATION", 0) > 0
        ):
            failures.append(MHS_GO_REASON_EXECUTION_GAP)
        primary_autocorr = _daily_autocorr_sharpe(primary.ledger)
        if not np.isfinite(primary_autocorr) or primary_autocorr < MHS_GO_PRIMARY_SHARPE_FLOOR:
            failures.append(MHS_GO_REASON_PRIMARY_SHARPE)
        stress_sharpe = _naive_sharpe(stress.ledger)
        if not np.isfinite(stress_sharpe) or stress_sharpe <= 0.0:
            failures.append(MHS_GO_REASON_STRESS_SHARPE)

        return MhsFoldReport(
            fold_index=fold_index,
            validation_start=str(vs),
            validation_end=str(ve),
            strict=primary,
            stress=stress,
            primary_valid=primary.ledger.primary_valid,
            primary_autocorr_sharpe=primary_autocorr,
            primary_naive_sharpe=_naive_sharpe(primary.ledger),
            primary_net_ann=_mean_ann(primary.ledger.net_returns, _PERIODS_PER_YEAR_1H),
            primary_geometric_cagr=_geometric_cagr(equity),
            primary_max_drawdown=_mdd(equity),
            stress_naive_sharpe=stress_sharpe,
            decision_intents=decision_intents,
            termination_counts=dict(primary.termination_counts),
            failures=tuple(sorted(set(failures))),
            strict_elapsed_seconds=primary.elapsed_seconds,
            stress_elapsed_seconds=stress.elapsed_seconds,
        )
    except DataIntegrityError as exc:
        message = str(exc).lower()
        if "pre-trade equity" in message or "capital" in message or "equity must be" in message:
            code = MHS_GO_REASON_CAPITAL_BREACH
        elif "finite" in message:
            code = MHS_GO_REASON_NONFINITE_EQUITY
        else:
            code = MHS_GO_REASON_INVALID_PRIMARY
        return _incomplete_fold_report(fold, fold_index, (code,))
    except (RuntimeError, ValueError):
        return _incomplete_fold_report(fold, fold_index, (MHS_GO_REASON_INCOMPLETE_FOLD,))


def _mhs_research_go(folds: tuple[MhsFoldReport, ...]) -> MhsResearchGoResult:
    """Fail-closed top-level Research-GO decision built from the fold reports.

    A fold that was not replayed, an invalid primary, non-finite equity, a
    relevant execution gap, a strict-Sharpe failure, or a non-positive stress
    Sharpe each block the decision with a stable reason code. The cap-30 and
    primary annual-return gate thresholds are not preregistered in source
    contracts, so those checks are reported as ``UNSPECIFIED_POLICY`` and keep
    Research GO conservative (false) until they are explicitly registered.
    """
    reasons: list[str] = []
    passed = 0
    for fold_report in folds:
        if fold_report.strict is None:
            reasons.append(MHS_GO_REASON_INCOMPLETE_FOLD)
            continue
        if not fold_report.failures:
            passed += 1
        reasons.extend(fold_report.failures)
    reasons.append(MHS_GO_REASON_UNSPECIFIED_POLICY)
    reasons = sorted(set(reasons))
    return MhsResearchGoResult(
        eligible=not reasons,
        reason_codes=tuple(reasons),
        evaluated_folds=len(folds),
        folds_passed=passed,
    )


def run_mhs_horizon_diagnostic(request: MhsDiagnosticRequest) -> MhsHorizonDiagnosticReport:
    """Compose the dev-only Phase 1 diagnostic: pre-screen + strict-proxy evidence.

    Forces ``partition='dev'`` and resolves the sealed evaluation end; a holdout
    partition or an end past ``HOLDOUT_CUTOFF`` raises ``RuntimeError``. The
    diagnostic-ensemble and executable-tranche numbers are reported as separate
    fields; only strict simulated inventory is primary Research evidence.
    """
    resolved_end = resolve_evaluation_end(request.end, unseal_holdout=False)
    _run_start = time.perf_counter()
    if request.partition != "dev":
        raise RuntimeError(
            "MHS Phase 1 is dev-only; the holdout partition requires an "
            "architecture-freeze final-OOS command"
        )
    start = pd.Timestamp(request.start, tz="UTC") if request.start is not None else MHS_DISCOVERY_START
    end = pd.Timestamp(resolved_end, tz="UTC") if resolved_end is not None else HOLDOUT_CUTOFF
    if end > HOLDOUT_CUTOFF:
        raise RuntimeError(f"Holdout sealed: requested end {end} past {HOLDOUT_CUTOFF}")

    root = request.data_root or str(FUTURES_DATA_DIR / "ohlcv")
    panel = load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), start, end, partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    symbols = list(close.columns)

    funding_by_symbol = _load_funding_series(symbols)
    fold_funding = dict(funding_by_symbol)
    funded = [s for s in symbols if s in funding_by_symbol]
    if not funded:
        raise RuntimeError("no dev symbol has funding coverage; the MHS ledger requires funding")
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = bar_funding_panel(funding_window, grid_1h)
    aligned_symbols = list(bar_funding.columns)
    if not aligned_symbols:
        raise RuntimeError("no dev symbol has causally aligned funding coverage")
    close = close[aligned_symbols]
    opens = opens[aligned_symbols]
    quote_vol = quote_vol[aligned_symbols]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned_symbols}
    bar_funding = bar_funding[aligned_symbols]

    eligible = liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)

    specs = PHASE_1_BOOK_SPECS
    fast = specs["fast_reversal"]
    slow = specs["slow_momentum"]
    fast_grid = pd.date_range(start, end, freq="6h", tz="UTC")
    slow_grid = pd.date_range(start, end, freq="24h", tz="UTC")

    w_fast = _book_weights(log_close, eligible, fast, fast_grid)
    w_slow = _book_weights(log_close, eligible, slow, slow_grid)
    w_fast_1h = w_fast.reindex(grid_1h).ffill().fillna(0.0)
    w_slow_1h = w_slow.reindex(grid_1h).ffill().fillna(0.0)
    execution_mask = _pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    w_fast_execution = w_fast.where(
        execution_mask.reindex(w_fast.index).fillna(False), other=0.0,
    )
    w_slow_execution = w_slow.where(
        execution_mask.reindex(w_slow.index).fillna(False), other=0.0,
    )
    blend_1h = (
        PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_1h
        + PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_1h
    )
    blend_gross = float(blend_1h.abs().sum(axis=1).mean())
    blend_cash_fraction = float((1.0 - blend_1h.abs().sum(axis=1)).mean())

    phase_fast = _phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    phase_slow = _phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, slow)
    phase_blend = _phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)

    execution_symbols = sorted(
        set(w_fast_execution.columns[w_fast_execution.ne(0.0).any(axis=0)])
        | set(w_slow_execution.columns[w_slow_execution.ne(0.0).any(axis=0)])
    )
    minute_frames = _align_minute_frames(
        _load_minute_frames(root, execution_symbols, start, end, request.execution_timeframe),
        request.execution_timeframe,
        start, end,
    )
    initial_equity = 1.0
    quote_volume_1m = pd.DataFrame()
    if minute_frames is not None:
        highs, _lows, closes, quote_volume_1m = minute_frames
        minute_grid = highs.index
        replay_symbols = list(closes.columns)
        minute_marks = (
            DataCollector().load_mark_price_panel(replay_symbols, "1h", minute_grid)
            if request.mark_mode == "cache_required"
            else None
        )
        minute_period = minute_grid[1] - minute_grid[0]
        minute_funding_window = {
            s: funding_by_symbol[s].loc[
                (funding_by_symbol[s].index >= minute_grid[0])
                & (funding_by_symbol[s].index < minute_grid[-1] + minute_period)
            ]
            for s in aligned_symbols
        }
        minute_funding = bar_funding_panel(minute_funding_window, minute_grid)
        minute_symbols = list(closes.columns)
        minute_funding = minute_funding.reindex(columns=minute_symbols)
        book_report_fast = _book_outcome(
            "fast_reversal", fast, len(funded), fast_grid, w_fast, grid_1h,
            opens, bar_funding, phase_fast, minute_frames, minute_funding, minute_marks,
            fast.horizon_hours, initial_equity, w_fast_execution,
        )
        book_report_slow = _book_outcome(
            "slow_momentum", slow, len(funded), slow_grid, w_slow, grid_1h,
            opens, bar_funding, phase_slow, minute_frames, minute_funding, minute_marks,
            slow.horizon_hours, initial_equity, w_slow_execution,
        )
        blend_step = blend_1h.reindex(fast_grid)
        book_report_blend = _book_outcome(
            "blend", fast, len(funded), fast_grid, blend_step, grid_1h,
            opens, bar_funding, phase_blend, minute_frames, minute_funding, minute_marks,
            168, initial_equity,
            blend_1h.where(execution_mask, other=0.0),
        )
        books = {"fast_reversal": book_report_fast, "slow_momentum": book_report_slow}
        blend_report = book_report_blend
    else:
        books = {}
        blend_report = None

    trials_attempted = 1
    deflated_sharpe_ratio = None

    xs_ic = _xs_rank_ic(horizon_log_return(log_close, 48), opens.pct_change())
    regression = _date_clustered_ols(opens.pct_change(), horizon_log_return(log_close, 48))
    horizon_diagnostics = {
        "realized_vol_48h_mean": float(
            realized_vol(log_close, 48).mean().mean()
        ),
        "efficiency_ratio_48h_mean": float(
            efficiency_ratio(log_close, 48).mean().mean()
        ),
    }

    bootstrap_ci: tuple[float, float] | None = None
    placebo_percentile: float | None = None
    participation: dict[str, float] = {}
    termination_counts: dict[str, int] = {}
    unsupported = ("partial_fill", "queue_position", "post_only_rejection", "cancel_replace_latency", "order_size_impact")
    if blend_report is not None:
        equity_1h = blend_report.primary.ledger.equity.resample("1h").last().dropna()
        net_1h = equity_1h.pct_change().dropna()
        if len(net_1h) >= 2:
            bootstrap_ci = _bootstrap_ci(
                net_1h, _BOOTSTRAP_REPLICATES, _BOOTSTRAP_MEAN_BLOCK, _BOOTSTRAP_SEED,
            )
        participation = _participation_warnings(blend_report.primary, quote_volume_1m)
        termination_counts = dict(blend_report.primary.termination_counts)
        placebo_percentile = _placebo_sharpe_percentile(
            horizon_log_return(log_close, 48), eligible, opens, bar_funding, grid_1h,
            fast, blend_report.primary_naive_sharpe, 500, _BOOTSTRAP_SEED,
        )

    folds = tuple(
        _run_anchored_fold(root, fold, request, fold_funding, initial_equity, idx)
        for idx, fold in enumerate(phase_1_anchored_purged_folds())
    )
    research_go = _mhs_research_go(folds)

    if blend_report is not None:
        deployment = compute_deployment_readiness(
            blend_report.primary.ledger.equity,
            _PERIODS_PER_YEAR_1H,
            participation_warnings=participation,
            primary_valid=blend_report.primary.ledger.primary_valid,
            research_go_eligible=research_go.eligible,
            n_bootstrap=_BOOTSTRAP_REPLICATES,
            mean_block_bars=_BOOTSTRAP_MEAN_BLOCK,
            seed=_BOOTSTRAP_SEED,
        )
    else:
        deployment = compute_deployment_readiness(
            pd.Series(
                [1.0, 1.0],
                index=pd.DatetimeIndex([start, start + pd.Timedelta(hours=1)]),
            ),
            _PERIODS_PER_YEAR_1H,
            research_go_eligible=research_go.eligible,
            n_bootstrap=_BOOTSTRAP_REPLICATES,
        )

    synthetic_stress = {s.name: {"description": s.description} for s in synthetic_stress_scenarios()}

    mark_source = "NOT_RUN_NO_EXECUTION_DATA"
    fill_source = "NOT_RUN_NO_EXECUTION_DATA"
    if blend_report is not None:
        mark_source = blend_report.primary.ledger.mark_source
        fill_source = "OHLCV_STRICT_PROXY"

    run_elapsed_seconds = time.perf_counter() - _run_start

    return MhsHorizonDiagnosticReport(
        feature=_MHS_FEATURE,
        status="COMPLETE",
        start=str(start),
        end=str(end),
        resolved_end=str(resolved_end),
        partition="dev",
        execution_tiers_bps=required_cost_tiers(),
        books=books,
        blend=blend_report,
        blend_target_gross=blend_gross,
        blend_cash_fraction=blend_cash_fraction,
        eligible_symbols=len(funded),
        trials_attempted=trials_attempted,
        deflated_sharpe_ratio=deflated_sharpe_ratio,
        xs_rank_ic=xs_ic,
        date_clustered_regression=regression,
        horizon_diagnostics=horizon_diagnostics,
        bootstrap_ci=bootstrap_ci,
        placebo_sharpe_percentile=placebo_percentile,
        deployment_readiness=deployment,
        synthetic_stress=synthetic_stress,
        participation_warnings=participation,
        termination_counts=termination_counts,
        unsupported_assumptions=unsupported,
        anchored_folds=phase_1_anchored_purged_folds(),
        folds=folds,
        research_go=research_go,
        fill_source=fill_source,
        mark_source=mark_source,
        execution_timeframe=request.execution_timeframe,
        execution_universe_size=request.execution_universe_size,
        execution_symbols=tuple(execution_symbols),
        run_elapsed_seconds=run_elapsed_seconds,
    )


def mhs_horizon_diagnostic_report_path() -> str:
    """Single source-controlled report path, sibling to the other ``*_report_path`` helpers."""
    return str(Path("docs/results") / "mhs_horizon_diagnostic.json")


def persist_mhs_horizon_diagnostic_report(report: MhsHorizonDiagnosticReport, path: str | Path) -> Path:
    """Persist a compact summary JSON and columnar replay audit artifacts.

    Fill events and ledger time series are intentionally kept out of the JSON
    summary.  They are losslessly persisted as compressed Parquet tables and
    referenced from the corresponding replay section.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = target.parent / f"{target.stem}_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = report.to_payload()

    for book_name, book_report in report.books.items():
        book_payload = payload["books"][book_name]
        book_payload["primary"] = _persist_replay_artifact(
            book_report.primary, artifact_root, f"{book_name}_primary"
        )
        book_payload["stress"] = _persist_replay_artifact(
            book_report.stress, artifact_root, f"{book_name}_stress"
        )
    if report.blend is not None:
        payload["blend"]["primary"] = _persist_replay_artifact(
            report.blend.primary, artifact_root, "blend_primary"
        )
        payload["blend"]["stress"] = _persist_replay_artifact(
            report.blend.stress, artifact_root, "blend_stress"
        )
    for fold_report in report.folds:
        fold_payload = payload["folds"][fold_report.fold_index]
        if fold_report.strict is not None:
            fold_payload["strict"] = _persist_replay_artifact(
                fold_report.strict, artifact_root, f"fold{fold_report.fold_index}_strict"
            )
        if fold_report.stress is not None:
            fold_payload["stress"] = _persist_replay_artifact(
                fold_report.stress, artifact_root, f"fold{fold_report.fold_index}_stress"
            )

    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    _logger.info("[MHS] report persisted path=%s", target)
    return target


def _to_timestamped_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Promote a DatetimeIndex into an explicit UTC timestamp column.

    The returned table carries a physical ``datetime64[ns, UTC]`` column named
    ``timestamp`` and a RangeIndex, so readers need no string-parsing guess.
    """
    out = frame.copy()
    if len(out):
        out.insert(0, "timestamp", out.index)
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    else:
        out["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    return out.reset_index(drop=True)


def _artifact_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_reference(table: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Row count, time bounds, schema version, and content checksum per table."""
    ts = (
        pd.to_datetime(table["timestamp"], utc=True)
        if "timestamp" in table.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    return {
        "file": path.name,
        "schema_version": MHS_ARTIFACT_SCHEMA_VERSION,
        "row_count": len(table),
        "time_bounds": {
            "start": None if len(ts) == 0 else str(ts.iloc[0]),
            "end": None if len(ts) == 0 else str(ts.iloc[-1]),
        },
        "checksum_sha256": _artifact_checksum(path),
    }


def _persist_replay_artifact(
    replay: StrategyExecutionReplayResult,
    artifact_root: Path,
    name: str,
) -> dict[str, Any]:
    """Write one replay's detailed tables and return a compact JSON reference."""
    prefix = artifact_root / name

    fills = replay.simulated_fills.copy()
    if not fills.empty and "timestamp" in fills.columns:
        fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
    fills_path = prefix.with_name(f"{name}_fills.parquet")
    fills.to_parquet(fills_path, index=False, compression="zstd")

    units_table = _to_timestamped_table(replay.simulated_units)
    units_path = prefix.with_name(f"{name}_units.parquet")
    units_table.to_parquet(units_path, index=False, compression="zstd")

    notional_table = _to_timestamped_table(replay.simulated_notional_weights)
    notional_path = prefix.with_name(f"{name}_notional_weights.parquet")
    notional_table.to_parquet(notional_path, index=False, compression="zstd")

    ledger = pd.concat(
        {
            "equity": replay.ledger.equity,
            "net_returns": replay.ledger.net_returns,
            "mark_to_market_pnl": replay.ledger.mark_to_market_pnl,
            "funding_charge": replay.ledger.funding_charge,
            "fee_charge": replay.ledger.fee_charge,
            "fill_turnover": replay.ledger.fill_turnover,
        },
        axis=1,
    )
    ledger_table = _to_timestamped_table(ledger)
    ledger_path = prefix.with_name(f"{name}_ledger.parquet")
    ledger_table.to_parquet(ledger_path, index=False, compression="zstd")

    times = pd.DataFrame(
        {"submit_time": replay.submit_times, "fill_time": replay.fill_times}
    )
    times["submit_time"] = pd.to_datetime(times["submit_time"], utc=True)
    times["fill_time"] = pd.to_datetime(times["fill_time"], utc=True)
    times_path = prefix.with_name(f"{name}_times.parquet")
    times.to_parquet(times_path, index=False, compression="zstd")

    return {
        "artifact_format": "parquet",
        "artifact_dir": str(artifact_root),
        "fills": _artifact_reference(fills, fills_path),
        "units": _artifact_reference(units_table, units_path),
        "notional_weights": _artifact_reference(notional_table, notional_path),
        "ledger": _artifact_reference(ledger_table, ledger_path),
        "times": _artifact_reference(times, times_path),
        "fill_source": replay.fill_source,
        "mark_source": replay.mark_source,
        "fill_count": replay.fill_count,
        "unfilled_count": replay.unfilled_count,
        "fallback_count": replay.fallback_count,
        "all_intent_shortfall_bps": replay.all_intent_shortfall_bps,
        "forced_exit_count": replay.forced_exit_count,
        "forced_exit_notional": replay.forced_exit_notional,
        "termination_counts": dict(replay.termination_counts),
        "unsupported_assumptions": list(replay.unsupported_assumptions),
        "elapsed_seconds": replay.elapsed_seconds,
    }
