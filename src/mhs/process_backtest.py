"""Single continuous causal replay of the algorithmic MHS process (1h ledger proxy).

Quarterly evidence is sliced from one out-of-sample path instead of replaying
each fold separately, so fold statistics and the deployed path can never
diverge. The hourly ledger is a proxy (``mhs_ledger_pnl`` contract): the report
is research evidence, never a deploy verdict, until Phase 2 routes the same
targets through the minute execution replay.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.mhs.books import rank_weight_book, scale_book_to_target_gross
from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
from src.mhs.deploy_gate import DeployGateResult, evaluate_deploy_gate
from src.mhs.evaluation.integrity import SOURCE_GAP_EXCLUDED_SYMBOLS
from src.mhs.execution.contracts import bar_funding_panel
from src.mhs.execution.pnl import mhs_ledger_pnl
from src.mhs.features import FEATURE_REGISTRY
from src.mhs.funding import funding_carry_signal
from src.mhs.marks import _load_funding_series, _pit_execution_mask
from src.mhs.panel import liquid_half_eligibility, load_base_panel
from src.mhs.params import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    DISCOVERY_START,
    GROWTH_RISK_ENVELOPES,
    PANEL_MIN_HISTORY_BARS,
    PROCESS_EVALUATION_CEILING,
    PROCESS_FEATURE_CANDIDATES,
    PROCESS_FUNDING_CARRY_CANDIDATES_HOURS,
    PROCESS_MIN_SYMBOLS,
    PROCESS_SMOOTHING_HALFLIFE_DAYS,
    STRESS_COST_MULTIPLIER,
)
from src.mhs.pipeline.config import (
    CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
    CLI_GROWTH_ENVELOPE_DEFAULT,
)
from src.mhs.process import (
    RefitPoint,
    ema_smoothing_rate,
    estimation_adjusted_mean,
    ledoit_wolf_covariance,
    long_only_growth_weights,
    monthly_refit_schedule,
    smoothed_book_path,
    step_proxy_net_returns,
    volatility_scaled_exposure,
)
from src.mhs.regime import beta_neutralize_weights, causal_market_beta
from src.mhs.types import ExecutionSpec

_logger = logging.getLogger(__name__)

PROCESS_REPORT_PATH: Path = Path("docs") / "results" / "mhs_process_backtest.json"
PROCESS_CERTIFICATION_LEVEL: str = "process_proxy_1h_ledger"


@dataclass(frozen=True, slots=True)
class ProcessMarketData:
    """Causally aligned inputs shared by every cost tier of one process run."""

    grid_1h: pd.DatetimeIndex
    decision_grid: pd.DatetimeIndex
    opens_1h: pd.DataFrame
    bar_funding_1h: pd.DataFrame
    log_close_step: pd.DataFrame
    funding_step: pd.DataFrame
    member_books: dict[str, pd.DataFrame]


@dataclass(frozen=True, slots=True)
class RefitRecord:
    """Audit record of one refit's decisions."""

    point: RefitPoint
    member_weights: dict[str, float]
    smoothing_halflife_days: float


@dataclass(frozen=True, slots=True)
class ProcessPath:
    """One cost tier's out-of-sample path."""

    one_way_bps: float
    daily_returns: pd.Series
    unit_daily_returns: pd.Series
    exposure: pd.Series
    refits: tuple[RefitRecord, ...]
    leverage_cap: float


@dataclass(frozen=True, slots=True)
class ProcessBacktestReport:
    """Proxy evidence for the continuous process; ``gate`` is never a deploy verdict."""

    start: pd.Timestamp
    end: pd.Timestamp
    certification_level: str
    n_candidates: int
    base: ProcessPath
    stress: ProcessPath
    gate: DeployGateResult


def build_candidate_member_books(
    panels: Mapping[str, pd.DataFrame],
    bar_funding_1h: pd.DataFrame,
    eligible_1h: pd.DataFrame,
    execution_mask_1h: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Beta-neutral unit-gross decision-grid books for every declared candidate.

    Books are not admission-filtered by coverage: a coverage audit over the full
    window would decide early membership with later data, and a member without
    evidence already receives zero weight from the growth weights.
    """
    registry = {spec.name: spec for spec in FEATURE_REGISTRY}
    books_1h: dict[str, pd.DataFrame] = {}
    for name in PROCESS_FEATURE_CANDIDATES:
        spec = registry[name]
        feature = spec.builder(panels)
        book = rank_weight_book(feature, execution_mask_1h, 1, PROCESS_MIN_SYMBOLS)
        books_1h[name] = book
        del feature
    for lookback in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS:
        key = f"funding_carry_{lookback}h"
        signal = funding_carry_signal(bar_funding_1h, lookback)
        books_1h[key] = rank_weight_book(signal, execution_mask_1h, -1, PROCESS_MIN_SYMBOLS)
        del signal
    beta_1h = causal_market_beta(
        np.log(panels["close"]), eligible_1h, CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS
    )
    beta_grid = beta_1h.reindex(decision_grid)
    mask_grid = execution_mask_1h.reindex(decision_grid).fillna(False)
    out: dict[str, pd.DataFrame] = {}
    for name, book in books_1h.items():
        grid_book = book.reindex(decision_grid).fillna(0.0)
        out[name] = beta_neutralize_weights(grid_book, beta_grid, mask_grid, PROCESS_MIN_SYMBOLS)
    return out


def load_process_market_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    data_root: str | None = None,
) -> ProcessMarketData:
    """Load the dev 1h panel, align funding, and build candidate books.

    Raises:
        RuntimeError: no dev symbol has aligned funding.
    """
    root = data_root or str(FUTURES_DATA_DIR / "ohlcv")
    panel = load_base_panel(
        root, "1h",
        ("close", "open", "high", "low", "quote_vol", "taker_buy_quote"),
        start, end, partition="dev", min_bars=PANEL_MIN_HISTORY_BARS,
        data_policy=MHS_DATA_POLICY_DEFAULT,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    _logger.info("[DATA] stage=base_1h_panel bars=%d symbols=%d", len(grid_1h), len(close.columns))
    funding_by_symbol, _dropped = _load_funding_series(list(close.columns))
    funded = [
        s for s in close.columns
        if s in funding_by_symbol and s not in SOURCE_GAP_EXCLUDED_SYMBOLS
    ]
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
    aligned = list(bar_funding.columns)
    if not aligned:
        raise RuntimeError("no dev symbol has causally aligned funding coverage")
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    _logger.info("[DATA] stage=funding_alignment bars=%d symbols=%d", len(grid_1h), len(aligned))
    eligible = liquid_half_eligibility(quote_vol, PANEL_MIN_HISTORY_BARS, PANEL_MIN_HISTORY_BARS)
    mask = _pit_execution_mask(quote_vol, eligible, CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT)
    decision_grid = pd.date_range(start, end, freq="24h", tz="UTC")
    log_close_step = np.log(close).reindex(decision_grid)
    # (t, t+24h] 구간 합을 누적합 차분으로 계산한다(결정일마다 전체 스캔 금지).
    day = pd.Timedelta(hours=24)
    prefix = np.vstack([np.zeros((1, len(aligned))), np.cumsum(bar_funding.to_numpy(dtype="float64"), axis=0)])
    left = np.searchsorted(bar_funding.index.to_numpy(), decision_grid.to_numpy(), side="right")
    right = np.searchsorted(bar_funding.index.to_numpy(), (decision_grid + day).to_numpy(), side="right")
    funding_step = pd.DataFrame(prefix[right] - prefix[left], index=decision_grid, columns=aligned)
    _logger.info("[DATA] stage=decision_grid days=%d symbols=%d", len(decision_grid), len(aligned))
    panels: dict[str, pd.DataFrame] = {k: panel[k][aligned] for k in ("close", "open", "high", "low", "quote_vol", "taker_buy_quote")}
    member_books = build_candidate_member_books(panels, bar_funding, eligible, mask, decision_grid)
    _logger.info("[DATA] stage=member_books candidates=%d", len(member_books))
    return ProcessMarketData(
        grid_1h=grid_1h,
        decision_grid=decision_grid,
        opens_1h=opens,
        bar_funding_1h=bar_funding,
        log_close_step=log_close_step,
        funding_step=funding_step,
        member_books=member_books,
    )


def run_process_paths(
    data: ProcessMarketData,
    schedule: tuple[RefitPoint, ...],
    *,
    decision_bps: float,
    evaluation_bps: tuple[float, ...],
    leverage_cap: float,
) -> tuple[ProcessPath, ...]:
    """Replay one set of process decisions continuously and evaluate it at each cost tier.

    Member weights, execution smoothing, and exposure are decided once at
    ``decision_bps``; every ``evaluation_bps`` tier replays those identical
    decisions through the hourly ledger, so a stress tier measures cost
    sensitivity of the same strategy rather than a re-optimized one. Member
    evidence is scored on books smoothed by ``PROCESS_SMOOTHING_HALFLIFE_DAYS``,
    the same smoothing the executed book receives.

    Raises:
        ValueError: empty ``schedule``, no candidate books, or empty ``evaluation_bps``.
    """
    if not schedule:
        raise ValueError("schedule must not be empty")
    if not data.member_books:
        raise ValueError("no candidate books")
    if not evaluation_bps:
        raise ValueError("evaluation_bps must not be empty")
    rate = ema_smoothing_rate(PROCESS_SMOOTHING_HALFLIFE_DAYS)
    names = list(data.member_books.keys())
    smoothed_members = {
        name: smoothed_book_path(
            data.member_books[name], pd.Series(rate, index=data.member_books[name].index)
        )
        for name in names
    }
    member_net = pd.DataFrame(
        {
            name: step_proxy_net_returns(
                smoothed_members[name], data.log_close_step, data.funding_step, decision_bps
            )
            for name in names
        }
    )
    step = member_net.index[1] - member_net.index[0]
    refit_targets: list[pd.DataFrame] = []
    records: list[RefitRecord] = []
    for point in schedule:
        train_rows = member_net.index[member_net.index + step <= point.train_end]
        train = member_net.loc[train_rows]
        weights = long_only_growth_weights(
            estimation_adjusted_mean(train), ledoit_wolf_covariance(train)
        )
        combined = sum(
            (weights[name] * data.member_books[name] for name in names),
            start=data.member_books[names[0]] * 0.0,
        )
        target = scale_book_to_target_gross(combined, 1.0)
        refit_targets.append(target)
        records.append(
            RefitRecord(
                point=point,
                member_weights={n: float(weights[n]) for n in names if float(weights[n]) != 0.0},
                smoothing_halflife_days=PROCESS_SMOOTHING_HALFLIFE_DAYS,
            )
        )
    oos_start = schedule[0].effective_from
    oos_end = schedule[-1].effective_to
    oos_days = data.decision_grid[(data.decision_grid >= oos_start) & (data.decision_grid < oos_end)]
    starts = pd.DatetimeIndex([p.effective_from for p in schedule])
    target_rows: list[pd.Series] = []
    for day in oos_days:
        idx = int(starts.searchsorted(day, side="right")) - 1
        target_rows.append(refit_targets[idx].loc[day])
    targets_oos = pd.DataFrame(target_rows, index=oos_days)
    sized_targets = smoothed_book_path(targets_oos, pd.Series(rate, index=oos_days))
    unit_1h = sized_targets.reindex(data.grid_1h, method="ffill").fillna(0.0)
    unit_net_1h, _ = mhs_ledger_pnl(unit_1h, data.opens_1h, data.bar_funding_1h, decision_bps)
    decision_unit_daily = (
        ((1.0 + unit_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
    )
    exposure = volatility_scaled_exposure(decision_unit_daily, cap=leverage_cap)
    sized = sized_targets.mul(exposure.reindex(sized_targets.index).fillna(0.0), axis=0)
    sized_1h = sized.reindex(data.grid_1h, method="ffill").fillna(0.0)
    paths: list[ProcessPath] = []
    for bps in evaluation_bps:
        net_1h, _ = mhs_ledger_pnl(sized_1h, data.opens_1h, data.bar_funding_1h, bps)
        daily_returns = ((1.0 + net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
        if bps == decision_bps:
            unit_daily_returns = decision_unit_daily
        else:
            alt_net_1h, _ = mhs_ledger_pnl(unit_1h, data.opens_1h, data.bar_funding_1h, bps)
            unit_daily_returns = (
                ((1.0 + alt_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
            )
        paths.append(
            ProcessPath(
                one_way_bps=bps,
                daily_returns=daily_returns,
                unit_daily_returns=unit_daily_returns,
                exposure=exposure,
                refits=tuple(records),
                leverage_cap=leverage_cap,
            )
        )
    return tuple(paths)


def quarter_fold_returns(daily_returns: pd.Series) -> tuple[pd.Series, ...]:
    """Split a daily path into non-empty calendar quarters (``QE-DEC``), in order."""
    if daily_returns.empty:
        return ()
    periods = daily_returns.index.tz_convert("UTC").tz_localize(None).to_period("Q-DEC")
    folds: list[pd.Series] = []
    for period in sorted(set(periods)):
        fold = daily_returns.loc[periods == period]
        if not fold.empty:
            folds.append(fold)
    return tuple(folds)


def evaluate_process_backtest(
    start: pd.Timestamp = DISCOVERY_START,
    end: pd.Timestamp = PROCESS_EVALUATION_CEILING,
    *,
    data_root: str | None = None,
) -> ProcessBacktestReport:
    """Evaluate base and same-decision stress paths with the deploy gate.

    Raises:
        DataIntegrityError: ``end`` exceeds ``PROCESS_EVALUATION_CEILING``.
        ValueError: naive timestamps or ``start >= end``.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    if start >= end:
        raise ValueError("start must be before end")
    if end > PROCESS_EVALUATION_CEILING:
        raise DataIntegrityError(f"end {end} exceeds PROCESS_EVALUATION_CEILING")
    envelope = GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT]
    base_bps = ExecutionSpec().one_way_taker_bps()
    stress_bps = base_bps * STRESS_COST_MULTIPLIER
    data = load_process_market_data(start, end, data_root=data_root)
    schedule = monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])
    base, stress = run_process_paths(
        data,
        schedule,
        decision_bps=base_bps,
        evaluation_bps=(base_bps, stress_bps),
        leverage_cap=envelope.leverage_ceiling,
    )
    gate = evaluate_deploy_gate(
        fold_returns=quarter_fold_returns(base.daily_returns),
        fold_stress_returns=quarter_fold_returns(stress.daily_returns),
        integrity_reasons=(),
        envelope=envelope,
    )
    return ProcessBacktestReport(
        start=start,
        end=end,
        certification_level=PROCESS_CERTIFICATION_LEVEL,
        n_candidates=len(data.member_books),
        base=base,
        stress=stress,
        gate=gate,
    )


def _tier_payload(path: ProcessPath) -> dict[str, object]:
    daily = {ts.isoformat(): float(v) for ts, v in path.daily_returns.items()}
    exposure = {ts.isoformat(): float(v) for ts, v in path.exposure.items()}
    log_growth = float(np.log1p(path.daily_returns.to_numpy(dtype="float64")).mean() * 365.0) if len(path.daily_returns) else 0.0
    exposure_values = path.exposure.to_numpy(dtype="float64")
    zero_share = float((exposure_values == 0.0).mean()) if len(exposure_values) else 0.0
    cap_share = float((exposure_values >= path.leverage_cap).mean()) if len(exposure_values) else 0.0
    return {
        "one_way_bps": path.one_way_bps,
        "leverage_cap": path.leverage_cap,
        "daily_returns": daily,
        "exposure": exposure,
        "refits": [
            {
                "effective_from": r.point.effective_from.isoformat(),
                "effective_to": r.point.effective_to.isoformat(),
                "train_end": r.point.train_end.isoformat(),
                "member_weights": dict(r.member_weights),
                "smoothing_halflife_days": r.smoothing_halflife_days,
            }
            for r in path.refits
        ],
        "ann_log_growth": log_growth,
        "exposure_zero_share": zero_share,
        "exposure_cap_share": cap_share,
    }


def persist_process_report(report: ProcessBacktestReport, path: Path = PROCESS_REPORT_PATH) -> Path:
    """Write the report as JSON (daily series as ISO-date keyed maps) and return the path."""
    payload = {
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "certification_level": report.certification_level,
        "n_candidates": report.n_candidates,
        "gate": {
            "go": report.gate.go,
            "reason_codes": list(report.gate.reason_codes),
            "metrics": dict(report.gate.metrics),
        },
        "base": _tier_payload(report.base),
        "stress": _tier_payload(report.stress),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return out
