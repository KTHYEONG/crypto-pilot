from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from src.common.config import ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_ohlcv_1h_as
from src.research.contracts import GrowthEngineEvaluationRequest
from src.research.evaluation.falsification import FalsificationConfig, evaluate_falsification
from src.research.evaluation.policy import resolve_evaluation_end
from src.research.evaluation.promotion import PromotionResult, compose_promotion_verdict
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateResult,
)
from src.research.execution.intrabar_audit import intrabar_audit_required
from src.research.portfolio.net_construction import NetConstructionSpec, compute_net_return_stream
from src.research.risk.growth_sizing import GrowthSizingConfig, solve_growth_optimal_risk
from src.research.universe.pit_universe import PitUniverseSpec, SymbolCoverage, build_universe_schedule, earliest_admissible_start, symbol_partition, derive_backfill_candidates

if TYPE_CHECKING:
    from src.research.evaluation.falsification import FalsificationVerdict
    from src.research.portfolio.net_construction import NetReturnStream
    from src.research.risk.growth_sizing import GrowthSizingResult

_logger = logging.getLogger("GrowthEngineEvaluation")

BARS_PER_YEAR = 2190
# Pre-registered xs_momentum hypothesis family, lookbacks in 4h bars.
XS_MOMENTUM_LOOKBACKS: tuple[int, ...] = (6, 18, 42, 84, 180)  # 1d / 3d / 7d / 14d / 30d
FAMILY_SIZE = 9
XS_QUANTILE = 0.30
_EMPTY_TRADE_COLUMNS = ("entry_bar", "exit_bar", "pnl", "return_pct")


@dataclass(frozen=True)
class GrowthEngineReport:
    """Fail-closed result of a growth-engine evaluation.

    ``status`` is ``"NO_ADMISSIBLE_ALPHA"`` when the admissible start cannot be
    derived or the falsification verdict fails; the equity curve is then a flat
    CASH series at ``initial_equity`` with zero trades.
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = pd.date_range(start, end, freq="4h", tz="UTC", inclusive="left")
    closes: dict[str, pd.Series] = {}
    for symbol in panel_symbols:
        closes[symbol] = frames[symbol]["close"].reindex(grid)
    px = pd.DataFrame(closes)

    # Decision-to-fill aligned forward returns: fwd[t] = close[t+1] / close[t] - 1,
    # so a signal decided at bar t earns the return realised on the following bar.
    arr = px.to_numpy(dtype=np.float64)
    fwd_arr = np.full_like(arr, np.nan)
    fwd_arr[:-1] = arr[1:] / arr[:-1] - 1.0
    fwd = pd.DataFrame(fwd_arr, index=px.index, columns=px.columns)
    return px, fwd


def _build_signal_weights(
    schedule: dict[pd.Timestamp, tuple[str, ...]],
    px: pd.DataFrame,
    fwd: pd.DataFrame,
    lookback_bars: int,
    quantile: float = XS_QUANTILE,
) -> pd.DataFrame:
    grid = px.index
    weights = pd.DataFrame(0.0, index=grid, columns=px.columns, dtype="float64")
    momentum = px.pct_change(lookback_bars)
    month_key = grid - pd.offsets.MonthBegin(0)
    for date, roster in schedule.items():
        if not roster:
            continue
        bars = grid[month_key == date]
        sub = momentum.loc[bars, list(roster)]
        valid = sub.notna()
        cnt = valid.sum(axis=1)
        rank = sub.rank(axis=1, ascending=False)
        k = (cnt * quantile).round().astype(int).clip(lower=1)
        longs = rank.le(k, axis=0) & valid
        shorts = rank.gt(cnt - k, axis=0) & valid
        w_long = longs.astype(float).div(longs.sum(axis=1).replace(0, np.nan), axis=0)
        w_short = shorts.astype(float).div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
        w = (w_long - w_short) / 2
        weights.loc[bars, list(roster)] = w.fillna(0.0).to_numpy()
    return weights


def _signal_pnl(
    schedule: dict[pd.Timestamp, tuple[str, ...]],
    px: pd.DataFrame,
    fwd: pd.DataFrame,
    lookback_bars: int,
    quantile: float = XS_QUANTILE,
) -> pd.Series:
    weights = _build_signal_weights(schedule, px, fwd, lookback_bars, quantile)
    return (weights * fwd).sum(axis=1)


def _compute_stream(
    target_weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    construction: NetConstructionSpec,
) -> NetReturnStream:
    return compute_net_return_stream(target_weights, forward_returns, construction)


def _gross_sharpe(pnl: pd.Series) -> float:
    rets = pnl.dropna()
    if len(rets) < 2:
        return 0.0
    std = float(rets.std())
    if std <= 0:
        return 0.0
    return float(rets.mean() / std * np.sqrt(BARS_PER_YEAR))


def _oos_t_stat(net: pd.Series) -> float:
    rets = net.dropna()
    if len(rets) < 10:
        return 0.0
    test = rets.iloc[len(rets) // 2 :]
    if len(test) < 2:
        return 0.0
    std = float(test.std())
    if std <= 0:
        return 0.0
    return float(test.mean() / std * np.sqrt(len(test)))


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
) -> GrowthEngineReport:
    _logger.warning(
        "[EVAL] status=NO_ADMISSIBLE_ALPHA start=%s falsification=%s sizing=%s -- holding CASH",
        start,
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
    )


def run_growth_engine_evaluation(request: GrowthEngineEvaluationRequest) -> GrowthEngineReport:
    """Single orchestration path for the growth-engine evaluation.

    ``resolve_evaluation_end`` -> build coverage -> ``earliest_admissible_start``
    -> ``build_universe_schedule`` -> ``symbol_partition`` filter ->
    ``compute_net_return_stream`` -> ``solve_growth_optimal_risk`` ->
    ``evaluate_falsification`` -> ``compose_promotion_verdict``.  Fail-closed:
    when the admissible start cannot be derived or the falsification verdict
    fails the report is ``NO_ADMISSIBLE_ALPHA`` with a flat CASH equity curve and
    zero trades.
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
        )

    data_start = min(cov.first_bar for cov in coverage)
    data_last = max(cov.last_bar for cov in coverage)
    rebalance_dates = _build_rebalance_dates(data_start, min(end_ts, data_last))

    start = earliest_admissible_start(coverage, rebalance_dates, request.universe)
    if start is None:
        return _no_admissible_alpha(
            request, end_ts, {}, start=None, falsification=None, sizing=None,
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
        )

    px, fwd = _build_price_panel(all_symbols, frames, start, end_ts)

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
        )

    parameter_scores = {
        float(lookback): _gross_sharpe(_signal_pnl(dev_schedule, px, fwd, lookback))
        for lookback in XS_MOMENTUM_LOOKBACKS
    }
    chosen_parameter = max(
        parameter_scores, key=lambda lb: (parameter_scores[lb], -lb),
    )

    target_weights = _build_signal_weights(schedule, px, fwd, int(chosen_parameter))
    forward_returns = fwd[list(target_weights.columns)]
    stream = _compute_stream(target_weights, forward_returns, request.construction)

    unit_returns = stream.net.to_numpy(dtype=np.float64)
    unit_returns = unit_returns[np.isfinite(unit_returns)]
    sizing_config = GrowthSizingConfig(
        risk_grid=(0.001, 0.005, 0.02),
        horizon_years=5.0,
        n_paths=500,
    )
    sizing = solve_growth_optimal_risk(unit_returns, sizing_config)

    oos_t_stat = _oos_t_stat(stream.net)
    dev_score = _gross_sharpe(_signal_pnl(dev_schedule, px, fwd, int(chosen_parameter)))
    holdout_score = _gross_sharpe(
        _signal_pnl(holdout_schedule, px, fwd, int(chosen_parameter))
    )
    falsification = evaluate_falsification(
        parameter_scores=parameter_scores,
        chosen_parameter=chosen_parameter,
        oos_t_stat=oos_t_stat,
        family_size=FAMILY_SIZE,
        dev_score=dev_score,
        holdout_score=holdout_score,
        config=FalsificationConfig(),
    )

    audit_required = intrabar_audit_required(
        competing_intrabar_exits=1, stop_atr_mult=3.0,
    )
    _logger.info(
        "[EVAL] symbols=%d traded=%d dev=%d holdout=%d chosen_lb=%s oos_t=%.3f "
        "falsification=%s sizing=%s audit_required=%s",
        len(all_symbols), len(panel_symbols), len(dev_members), len(holdout_members),
        chosen_parameter, oos_t_stat, falsification.binding_constraint,
        sizing.binding_constraint, audit_required,
    )

    if not falsification.passed or sizing.selected_risk is None:
        return _no_admissible_alpha(
            request, end_ts, schedule, start=start,
            falsification=falsification, sizing=sizing,
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
    )
