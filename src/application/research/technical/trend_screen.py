"""Pre-registered baseline-gate trend screen: 450-cell discovery then qualification.

Loads funding-complete 4h data once per symbol, executes exactly the 30
frozen screen identities on the exact 15-symbol universe (450 cells), evaluates
each cell at unit risk and under its causal fractional-Kelly/MDD policy ledger,
screens discovery (2022-04-01..2023-12-31) by data validity, a complete causal
policy lookback, one closed trade per complete discovery month, and a positive
unit-risk LCB90 (raw CAGR / t-stat / bootstrap-negative fraction stay
diagnostics, never hard gates), forms a maximum-five-sleeve portfolio ranked and
weighted on policy discovery ledgers, and generates the qualification
total-equity ledgers under the frozen base-derived schedule. Admission requires
the unchanged observation, derived fold-concentration, stressed rerun with the
same frozen schedule, and (when unsealed) holdout gates; otherwise the binding
constraint is recorded and CASH is retained.

The screen persists a deterministic report and never registers a production
candidate: it is research evidence only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.core.settings import effective_worker_count
from src.market_data.storage.loaders import load_funding_rates, load_ohlcv_1h_as
from src.research.baseline.backtest import BacktestResult, _align_funding_rates
from src.research.contracts import CostModel
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    split_holdout_segment,
)
from src.research.sleeve_blend.contracts import (
    CausalFractionalKellySpec,
    CausalLeverageSpec,
)
from src.research.sleeve_blend.fixed import (
    build_causal_fractional_kelly_schedule,
    build_causal_leverage_schedule,
)
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.provenance import technical_data_hashes
from src.research.technical_experts.trend_screen_catalog import (
    DISCOVERY_END,
    DISCOVERY_START,
    QUALIFICATION_END,
    QUALIFICATION_START,
    TREND_SCREEN_CANDIDATES,
    TREND_SCREEN_PROFILE_ID,
    TREND_SCREEN_SYMBOLS,
)

_logger = logging.getLogger("TrendScreen")

_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0
_BASE_DELAY_BARS = 1
_STRESS_DELAY_BARS = 2
_INITIAL_EQUITY = 10_000.0
_BARS_PER_YEAR = 2190

__all__ = [
    "TREND_SCREEN_CANDIDATES",
    "TREND_SCREEN_PROFILE_ID",
    "TREND_SCREEN_SYMBOLS",
    "TrendScreenCell",
    "TrendScreenQualification",
    "TrendScreenReport",
    "TrendScreenSelection",
    "persist_trend_screen_report",
    "run_trend_screen",
    "trend_screen_report_path",
]


@dataclass(frozen=True, slots=True)
class TrendScreenCell:
    """One (identity x symbol) discovery cell of the pre-registered screen.

    Raw fields (``net_cagr``/``lcb90``/``t_stat``/``p_negative``) are the
    unit-risk alpha diagnostics; ``policy_*`` fields describe the same return
    stream under the causal fractional-Kelly/MDD policy ledger and drive
    ranking, correlation, and portfolio construction. The latter never replace
    the raw diagnostics.
    """

    return_source: str
    family: str
    side: str
    symbol: str
    data_valid: bool
    funding_coverage: float
    trade_count: int
    net_cagr: float
    mdd: float
    lcb90: float
    t_stat: float
    fold_score: float
    stress_verdict: str
    p_negative: float
    fingerprint: dict[str, str]
    discovery_pass: bool
    rejected_reason: str | None
    policy_lcb90: float = 0.0
    policy_cagr: float = 0.0
    policy_mdd: float = 0.0
    policy_trade_count: int = 0
    policy_schedule_hash: str = ""
    kelly_fraction: float = 0.25
    kelly_lookback_days: int = 365
    allocation_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class TrendScreenSelection:
    """Frozen equal-risk sleeve selection and its discovery boundary weights."""

    return_sources: tuple[str, ...]
    symbols: tuple[str, ...]
    weights: tuple[float, ...]
    discovery_lcb90: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrendScreenQualification:
    """Qualification evidence, admission verdict, and the binding constraint."""

    admitted: bool
    observation_verdict: str
    fold_gate_pass: bool
    stress_verdict: str
    holdout_verdict: str | None
    binding_constraint: str | None


@dataclass(frozen=True, slots=True)
class TrendScreenReport:
    """Deterministic persisted outcome of one sealed screen profile."""

    profile: str
    universe: tuple[str, ...]
    cells: tuple[TrendScreenCell, ...]
    selection: TrendScreenSelection | None
    schedule_hash: str
    qualification: TrendScreenQualification

    def to_payload(self) -> dict[str, object]:
        """Canonical, deterministic JSON-ready payload (fingerprint included)."""
        payload: dict[str, object] = {
            "profile": self.profile,
            "universe": list(self.universe),
            "discovery_start": DISCOVERY_START.isoformat(),
            "discovery_end": DISCOVERY_END.isoformat(),
            "qualification_start": QUALIFICATION_START.isoformat(),
            "qualification_end": QUALIFICATION_END.isoformat(),
            "cells": [
                {
                    "return_source": cell.return_source,
                    "family": cell.family,
                    "side": cell.side,
                    "symbol": cell.symbol,
                    "data_valid": cell.data_valid,
                    "funding_coverage": round(cell.funding_coverage, 8),
                    "trade_count": cell.trade_count,
                    "net_cagr": round(cell.net_cagr, 8),
                    "mdd": round(cell.mdd, 8),
                    "lcb90": round(cell.lcb90, 8),
                    "t_stat": round(cell.t_stat, 8),
                    "fold_score": round(cell.fold_score, 8),
                    "stress_verdict": cell.stress_verdict,
                    "p_negative": round(cell.p_negative, 8),
                    "fingerprint": cell.fingerprint,
                    "discovery_pass": cell.discovery_pass,
                    "rejected_reason": cell.rejected_reason,
                    "policy_lcb90": round(cell.policy_lcb90, 8),
                    "policy_cagr": round(cell.policy_cagr, 8),
                    "policy_mdd": round(cell.policy_mdd, 8),
                    "policy_trade_count": cell.policy_trade_count,
                    "policy_schedule_hash": cell.policy_schedule_hash,
                    "kelly_fraction": cell.kelly_fraction,
                    "kelly_lookback_days": cell.kelly_lookback_days,
                    "allocation_cost": round(cell.allocation_cost, 8),
                }
                for cell in self.cells
            ],
            "selection": None,
            "schedule_hash": self.schedule_hash,
            "qualification": {
                "admitted": self.qualification.admitted,
                "observation_verdict": self.qualification.observation_verdict,
                "fold_gate_pass": self.qualification.fold_gate_pass,
                "stress_verdict": self.qualification.stress_verdict,
                "holdout_verdict": self.qualification.holdout_verdict,
                "binding_constraint": self.qualification.binding_constraint,
            },
        }
        if self.selection is not None:
            payload["selection"] = [
                {
                    "return_source": rs,
                    "symbol": symbol,
                    "weight": round(weight, 8),
                    "lcb90": round(lcb90, 8),
                }
                for rs, symbol, weight, lcb90 in zip(
                    self.selection.return_sources,
                    self.selection.symbols,
                    self.selection.weights,
                    self.selection.discovery_lcb90,
                    strict=True,
                )
            ]
        payload["report_fingerprint"] = _fingerprint_without_self(payload)
        return payload

    def to_json(self) -> str:
        """Byte-deterministic JSON serialization of the report payload."""
        return json.dumps(self.to_payload(), sort_keys=True, indent=2) + "\n"


def _fingerprint_without_self(payload: dict[str, object]) -> str:
    body = {k: v for k, v in payload.items() if k != "report_fingerprint"}
    encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _CellRun:
    """Internal evidence for one cell while the screen is running."""

    __slots__ = (
        "allocation_cost",
        "cell",
        "disc_equity",
        "policy_equity",
        "policy_schedule",
        "result",
    )

    def __init__(self, cell: TrendScreenCell, result: BacktestResult | None = None) -> None:
        self.cell = cell
        self.result = result
        self.disc_equity: pd.Series | None = None
        self.policy_equity: pd.Series | None = None
        self.policy_schedule: pd.Series | None = None
        self.allocation_cost: float = 0.0


def _trades_in_window(
    trades: pd.DataFrame,
    equity_index: pd.DatetimeIndex,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    if len(trades) == 0:
        return trades
    if "exit_time" in trades.columns:
        exit_ts = pd.to_datetime(trades["exit_time"], utc=True, errors="raise")
    else:
        exit_ts = equity_index[trades["exit_bar"].astype(int).to_numpy()]
    mask = (exit_ts >= start_ts) & (exit_ts <= end_ts)
    return trades[mask]


def _portfolio_trades(
    trade_rows: list[tuple[pd.DataFrame, pd.DatetimeIndex]],
) -> pd.DataFrame:
    """Combine single-symbol trades with timestamp exits for portfolio ledgers."""
    frames: list[pd.DataFrame] = []
    for trades, equity_index in trade_rows:
        if len(trades) == 0:
            continue
        frame = trades.copy()
        bars = frame["exit_bar"].astype(int).to_numpy()
        if (bars < 0).any() or (bars >= len(equity_index)).any():
            raise DataIntegrityError("trade exit_bar is outside its equity index")
        frame["exit_time"] = equity_index[bars]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _funding_coverage(funding: pd.Series, grid: pd.DatetimeIndex) -> float:
    if len(grid) < 2:
        return 0.0
    bar_funding = _align_funding_rates(funding, grid)
    return float(np.mean(bar_funding != 0.0))


def _load_symbol_data(
    symbol: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, str], float]:
    """Load and fail-closed validate one symbol's 4h bars and aligned funding.

    ``start`` is ``None`` by default so each symbol's full history (warm-up bars
    before the discovery boundary) is preserved. Missing funding is never a
    zero-cost fallback.
    """
    bars = load_ohlcv_1h_as(ohlcv_path(symbol, "1h"), "4h", start=start, end=end)
    if len(bars) < 2:
        raise DataIntegrityError(f"bars data has fewer than 2 bars for {symbol}")
    funding = load_funding_rates(funding_path(symbol))
    period = bars.index[1] - bars.index[0]
    window_end = bars.index[-1] + period
    funding = funding[(funding.index >= bars.index[0]) & (funding.index < window_end)]
    if len(funding) == 0:
        raise DataIntegrityError(f"no settled funding events in window for {symbol}")
    coverage = _funding_coverage(funding, bars.index)
    fingerprint = technical_data_hashes(symbol)
    return bars, funding, fingerprint, coverage


def _candidate_by_source(return_source: str) -> TechnicalCandidate:
    for candidate in TREND_SCREEN_CANDIDATES:
        if candidate.return_source == return_source:
            return candidate
    raise ValueError(f"unknown trend screen return source '{return_source}'")


def _run_cell(
    frame: pd.DataFrame,
    funding: pd.Series,
    candidate: TechnicalCandidate,
    symbol: str,
    fingerprint: dict[str, str],
    funding_coverage: float,
) -> _CellRun:
    """Execute the base discovery path for one candidate on already loaded data.

    Stress is deliberately deferred until after discovery selection: it cannot
    affect candidate ranking and the qualification stage replays only the at
    most five frozen sleeves under the same causal schedule.
    """
    costs = CostModel()
    result = run_technical_expert_backtest(
        frame, candidate, costs, funding,
        initial_equity=_INITIAL_EQUITY, signal_delay_bars=_BASE_DELAY_BARS,
    )
    policy_schedule = build_causal_fractional_kelly_schedule(
        result.equity, CausalLeverageSpec(), CausalFractionalKellySpec(),
    )
    policy_equity, allocation_cost = _apply_policy_schedule(
        result.equity, policy_schedule, costs,
    )
    kelly_spec = CausalFractionalKellySpec()
    disc_equity = result.equity[
        (result.equity.index >= DISCOVERY_START) & (result.equity.index <= DISCOVERY_END)
    ]
    policy_disc_equity = policy_equity[
        (policy_equity.index >= DISCOVERY_START) & (policy_equity.index <= DISCOVERY_END)
    ]
    disc_trades = _trades_in_window(
        result.trades, result.equity.index, DISCOVERY_START, DISCOVERY_END,
    )

    metrics = compute_metrics(disc_equity, disc_trades)
    gate = compute_equity_reliability_gate(
        disc_equity, len(disc_trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )
    policy_metrics = compute_metrics(policy_disc_equity, disc_trades)
    policy_gate = compute_equity_reliability_gate(
        policy_disc_equity, len(disc_trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )
    disc_result = BacktestResult(
        equity=disc_equity, trades=disc_trades, signals=pd.DataFrame(),
    )
    try:
        folds = compute_fold_distribution(disc_result)
    except ValueError:
        folds = FoldDistributionResult(
            n_folds=0, median_fold_cagr=0.0, worst_fold_cagr=0.0,
            median_fold_calmar=0.0, max_period_contribution=0.0, gate_pass=True,
        )
    cell = TrendScreenCell(
        return_source=candidate.return_source,
        family=candidate.family,
        side=candidate.side,
        symbol=symbol,
        data_valid=True,
        funding_coverage=funding_coverage,
        trade_count=len(disc_trades),
        net_cagr=metrics.cagr,
        mdd=metrics.mdd,
        lcb90=gate.lcb90_cagr,
        t_stat=gate.t_stat,
        fold_score=folds.max_period_contribution,
        stress_verdict="PENDING",
        p_negative=gate.p_negative,
        fingerprint=fingerprint,
        discovery_pass=False,
        rejected_reason=None,
        policy_lcb90=policy_gate.lcb90_cagr,
        policy_cagr=policy_metrics.cagr,
        policy_mdd=policy_metrics.mdd,
        policy_trade_count=len(disc_trades),
        policy_schedule_hash=_schedule_hash(policy_schedule),
        kelly_fraction=kelly_spec.fraction,
        kelly_lookback_days=kelly_spec.lookback_days,
        allocation_cost=allocation_cost,
    )
    run = _CellRun(cell, result)
    run.disc_equity = disc_equity
    run.policy_equity = policy_disc_equity
    run.policy_schedule = policy_schedule
    run.allocation_cost = allocation_cost
    return run


def _invalid_cell(
    candidate: TechnicalCandidate,
    symbol: str,
    fingerprint: dict[str, str],
    funding_coverage: float,
    reason: str,
) -> _CellRun:
    cell = TrendScreenCell(
        return_source=candidate.return_source,
        family=candidate.family,
        side=candidate.side,
        symbol=symbol,
        data_valid=False,
        funding_coverage=funding_coverage,
        trade_count=0,
        net_cagr=0.0,
        mdd=0.0,
        lcb90=0.0,
        t_stat=0.0,
        fold_score=0.0,
        stress_verdict="PENDING",
        p_negative=1.0,
        fingerprint=fingerprint,
        discovery_pass=False,
        rejected_reason=reason,
    )
    return _CellRun(cell)


def _run_symbol_cells_worker(
    symbol: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> list[_CellRun]:
    """Load one symbol once and execute all 30 base discovery candidates.

    A process owns its input frame, avoiding 30 repeated parquet/funding loads
    while allowing the Python-bar execution loop to run outside the GIL.
    """
    try:
        frame, funding, fingerprint, coverage = _load_symbol_data(symbol, start, end)
    except (DataIntegrityError, FileNotFoundError) as exc:
        _logger.info("[TREND] symbol=%s data invalid reason=%s", symbol, exc)
        reason = f"data_invalid:{type(exc).__name__}:{exc}"
        return [
            _invalid_cell(candidate, symbol, {}, 0.0, reason)
            for candidate in TREND_SCREEN_CANDIDATES
        ]

    runs: list[_CellRun] = []
    for candidate in TREND_SCREEN_CANDIDATES:
        try:
            runs.append(_run_cell(
                frame, funding, candidate, symbol, fingerprint, coverage,
            ))
        except (DataIntegrityError, ValueError) as exc:
            _logger.info(
                "[TREND] cell=%s/%s failed reason=%s",
                candidate.return_source, symbol, exc,
            )
            runs.append(_invalid_cell(
                candidate, symbol, fingerprint, coverage,
                f"cell_failed:{type(exc).__name__}",
            ))
    return runs


def _discovery_duration_months(
    start: pd.Timestamp, end: pd.Timestamp,
) -> int:
    """Number of complete calendar months spanned by the inclusive window."""
    return int(
        (end.year - start.year) * 12 + (end.month - start.month) + 1
    )


def _apply_discovery_requirements(
    runs: list[_CellRun],
    config: ReliabilityGateConfig,
) -> None:
    """Gate discovery eligibility by data validity, causal lookback, trade
    coverage, and positive unit-risk LCB90.

    The required close count is duration-derived (one closed trade per complete
    discovery month), replacing the fixed 30-close cliff. Holm, bootstrap
    negative fraction, raw CAGR, and IID t-stat are reported diagnostics, never
    hard gates here.
    """
    required_trades = _discovery_duration_months(DISCOVERY_START, DISCOVERY_END)
    for run in runs:
        cell = run.cell
        if not cell.data_valid:
            continue
        reasons: list[str] = []
        if cell.trade_count < required_trades:
            reasons.append(f"min_trades:{cell.trade_count}<{required_trades}")
        if cell.lcb90 <= 0.0:
            reasons.append(f"lcb90:{cell.lcb90:.6f}<=0")
        if not _policy_lookback_complete(run):
            reasons.append("incomplete_causal_lookback")
        if reasons:
            run.cell = dataclasses.replace(
                cell, discovery_pass=False, rejected_reason=";".join(reasons),
            )
        else:
            run.cell = dataclasses.replace(
                cell, discovery_pass=True, rejected_reason=None,
            )


def _policy_lookback_complete(run: _CellRun) -> bool:
    """True when the policy schedule reaches non-zero exposure in discovery.

    The schedule only turns non-zero after a complete causal lookback, so any
    positive discovery bar certifies the required lookback was available.
    """
    schedule = run.policy_schedule
    if schedule is None or len(schedule) == 0:
        return False
    active = schedule[
        (schedule.index >= DISCOVERY_START) & (schedule.index <= DISCOVERY_END)
    ]
    if len(active) == 0:
        return False
    return bool((active > 0.0).any())


def _apply_policy_schedule(
    unit_equity: pd.Series,
    schedule: pd.Series,
    costs: CostModel,
) -> tuple[pd.Series, float]:
    """Apply a frozen leverage schedule to a unit ledger with turnover cost.

    On each causal bar the leveraged marked return is the unit simple return
    times the applied leverage minus the allocation turnover cost
    ``0.5 * abs(L_t - L_{t-1}) * (fee_rate + slippage_rate)``, then compounded
    into the total-equity ledger. The schedule is reused verbatim (aligned by
    timestamp, missing rows default to zero exposure). Returns the ledger and
    the cumulative allocation cost.
    """
    aligned = schedule.reindex(unit_equity.index).fillna(0.0).to_numpy(dtype=np.float64)
    returns = unit_equity.pct_change().fillna(0.0).to_numpy(dtype=np.float64)
    prev = np.concatenate([[0.0], aligned[:-1]])
    turnover = 0.5 * np.abs(aligned - prev) * (costs.fee_rate + costs.slippage_rate)
    net_returns = returns * aligned - turnover
    equity = (1.0 + net_returns).cumprod() * _INITIAL_EQUITY
    ledger = pd.Series(equity, index=unit_equity.index, name="equity", dtype=np.float64)
    return ledger, float(turnover.sum())


def _select_sleeves(runs: list[_CellRun]) -> list[_CellRun]:
    """At most one side per family and at most one symbol per retained identity.

    Ranking uses the policy ledger (policy LCB90, policy MDD) with unit-risk
    LCB90 and lexical identity as tie-breakers.
    """
    survivors = [run for run in runs if run.cell.discovery_pass]
    if not survivors:
        return []

    by_identity: dict[tuple[str, str], list[_CellRun]] = {}
    for run in survivors:
        by_identity.setdefault((run.cell.family, run.cell.side), []).append(run)
    identity_best: list[_CellRun] = [
        max(members, key=lambda r: (
            r.cell.policy_lcb90, r.cell.policy_mdd, r.cell.lcb90, r.cell.symbol,
        ))
        for members in by_identity.values()
    ]

    by_family: dict[str, list[_CellRun]] = {}
    for run in identity_best:
        by_family.setdefault(run.cell.family, []).append(run)
    return [
        max(members, key=lambda r: (
            r.cell.policy_lcb90, r.cell.policy_mdd, r.cell.lcb90,
            r.cell.return_source,
        ))
        for members in by_family.values()
    ]


def _discovery_log_returns(run: _CellRun) -> pd.Series:
    assert run.policy_equity is not None
    equity = run.policy_equity
    returns = np.log(equity / equity.shift()).dropna()
    returns.name = f"{run.cell.return_source}|{run.cell.symbol}"
    return returns


def _mean_abs_corr(candidate_run: _CellRun, selected: list[_CellRun]) -> float:
    if not selected:
        return 0.0
    target = _discovery_log_returns(candidate_run)
    values: list[float] = []
    for other in selected:
        other_returns = _discovery_log_returns(other)
        aligned = pd.concat(
            (target.rename("a"), other_returns.rename("b")), axis=1,
        ).dropna()
        if len(aligned) < 2:
            continue
        values.append(float(aligned["a"].corr(aligned["b"])))
    if not values:
        return 0.0
    return float(np.mean(np.abs(values)))


def _greedy_portfolio_selection(runs: list[_CellRun], max_sleeves: int = 5) -> list[_CellRun]:
    """Rank by policy discovery LCB90/MDD; break ties by |pairwise policy
    log-return corr| then lexical."""
    selected: list[_CellRun] = []
    remaining = list(runs)
    while len(selected) < max_sleeves and remaining:
        best = max(
            remaining,
            key=lambda r: (
                r.cell.policy_lcb90,
                r.cell.policy_mdd,
                r.cell.lcb90,
                -_mean_abs_corr(r, selected),
                r.cell.return_source,
                r.cell.symbol,
            ),
        )
        selected.append(best)
        remaining = [r for r in remaining if r is not best]
    return selected


def _equal_risk_weights(selected: list[_CellRun]) -> list[float]:
    """Inverse-discovery-policy-volatility equal-risk weights frozen at the boundary."""
    inv: list[float] = []
    for run in selected:
        assert run.policy_equity is not None
        returns = run.policy_equity.pct_change().dropna()
        vol = float(returns.std() * np.sqrt(_BARS_PER_YEAR)) if len(returns) > 1 else 0.0
        inv.append(1.0 / vol if vol > 0.0 else 0.0)
    total = sum(inv)
    if total <= 0.0:
        return [1.0 / len(selected)] * len(selected)
    return [w / total for w in inv]


def _unit_equity(result: BacktestResult) -> pd.Series:
    return (result.equity / result.equity.iloc[0]).rename("equity")


def _blend_unit_equities(
    unit_equities: list[pd.Series],
    weights: list[float],
) -> pd.Series:
    common = sorted(set.intersection(*(set(eq.index) for eq in unit_equities)))
    common_idx = pd.DatetimeIndex(common)
    blended = sum(
        w * eq.loc[common_idx] for w, eq in zip(weights, unit_equities, strict=True)
    )
    return pd.Series(blended, index=common_idx, name="equity", dtype=np.float64)


def _schedule_hash(schedule: pd.Series) -> str:
    payload = json.dumps(schedule.round(8).tolist(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _qualify(
    selected: list[_CellRun],
    weights: list[float],
    data: dict[str, tuple[pd.DataFrame, pd.Series]],
    *,
    unseal_holdout: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, TrendScreenQualification]:
    """Build the frozen-schedule qualification ledger and compose its gates.

    The causal fractional-Kelly/MDD schedule is built once from the base unit
    blend (prior marks only) and applied verbatim to the stressed rerun, so both
    ledgers share the exact allocation. Full-window realized-MDD scalar
    calibration is never used.
    """
    base_unit: list[pd.Series] = []
    for run in selected:
        assert run.result is not None
        base_unit.append(_unit_equity(run.result))
    blend = _blend_unit_equities(base_unit, weights)
    schedule = build_causal_fractional_kelly_schedule(
        blend, CausalLeverageSpec(), CausalFractionalKellySpec(),
    )
    mdd_cap = build_causal_leverage_schedule(blend, CausalLeverageSpec())
    if not (schedule.to_numpy() <= mdd_cap.to_numpy() + 1e-12).all():
        raise DataIntegrityError("policy schedule exceeds the causal MDD cap")
    scheduled, _base_cost = _apply_policy_schedule(blend, schedule, CostModel())
    trades = _portfolio_trades(
        [(run.result.trades, run.result.equity.index) for run in selected if run.result is not None],
    )

    qual_equity = scheduled[
        (scheduled.index >= QUALIFICATION_START) & (scheduled.index <= QUALIFICATION_END)
    ]
    qual_trades = _trades_in_window(
        trades, scheduled.index, QUALIFICATION_START, QUALIFICATION_END,
    )
    observation = compute_equity_reliability_gate(qual_equity, len(qual_trades))
    folds = compute_fold_distribution(
        BacktestResult(equity=qual_equity, trades=qual_trades, signals=pd.DataFrame()),
    )

    stressed_costs = CostModel(
        fee_rate=0.0005 * _STRESS_FEE_MULT,
        slippage_rate=0.0003 * _STRESS_SLIPPAGE_MULT,
    )
    stress_unit: list[pd.Series] = []
    stress_trade_rows: list[tuple[pd.DataFrame, pd.DatetimeIndex]] = []
    for run in selected:
        frame, funding = data[run.cell.symbol]
        stress_result = run_technical_expert_backtest(
            frame, _candidate_by_source(run.cell.return_source), stressed_costs, funding,
            initial_equity=_INITIAL_EQUITY, signal_delay_bars=_STRESS_DELAY_BARS,
        )
        stress_unit.append(_unit_equity(stress_result))
        stress_trade_rows.append((stress_result.trades, stress_result.equity.index))
    stress_blend = _blend_unit_equities(stress_unit, weights)
    stress_scheduled, _stress_cost = _apply_policy_schedule(
        stress_blend, schedule, stressed_costs,
    )
    stress_qual_equity = stress_scheduled[
        (stress_scheduled.index >= QUALIFICATION_START)
        & (stress_scheduled.index <= QUALIFICATION_END)
    ]
    stress_trades = _portfolio_trades(stress_trade_rows)
    stress_trades = _trades_in_window(
        stress_trades, stress_scheduled.index, QUALIFICATION_START, QUALIFICATION_END,
    )
    stress_gate = compute_equity_reliability_gate(
        stress_qual_equity, len(stress_trades),
        dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )

    holdout_gate: ReliabilityGateResult | None = None
    if unseal_holdout and scheduled.index[-1] > HOLDOUT_CUTOFF:
        segment = split_holdout_segment(
            BacktestResult(equity=scheduled, trades=trades, signals=pd.DataFrame()),
            HOLDOUT_CUTOFF,
        )
        holdout_gate = compute_equity_reliability_gate(
            segment.holdout_equity, len(segment.holdout_trades),
        )

    constraints: list[str] = []
    if observation.verdict != "PASS":
        constraints.append(f"observation:{observation.verdict}")
    if not folds.gate_pass:
        constraints.append("fold:gate_pass=False")
    if stress_gate.verdict != "PASS":
        constraints.append(f"stress:{stress_gate.verdict}")
    if holdout_gate is not None and holdout_gate.verdict != "PASS":
        constraints.append(f"holdout:{holdout_gate.verdict}")

    qualification = TrendScreenQualification(
        admitted=not constraints,
        observation_verdict=observation.verdict,
        fold_gate_pass=folds.gate_pass,
        stress_verdict=stress_gate.verdict,
        holdout_verdict=holdout_gate.verdict if holdout_gate is not None else None,
        binding_constraint=";".join(constraints) if constraints else None,
    )
    return schedule, scheduled, stress_scheduled, qualification


def run_trend_screen(
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    unseal_holdout: bool = False,
    max_workers: int | None = None,
) -> TrendScreenReport:
    """Execute one sealed baseline-gate trend screen profile.

    Runs exactly 450 cells (30 identities x 15 symbols) on funding-complete 4h
    data, screens discovery, forms the maximum-five-sleeve equal-risk portfolio,
    and evaluates qualification under the frozen causal schedule. Returns the
    deterministic report; nothing is registered. ``end`` defaults to the sealed
    cutoff unless ``unseal_holdout`` is set.
    """
    end = resolve_evaluation_end(end, unseal_holdout=unseal_holdout)
    workers = effective_worker_count(len(TREND_SCREEN_SYMBOLS), requested=max_workers)
    if workers == 1:
        per_symbol = [
            _run_symbol_cells_worker(symbol, start, end)
            for symbol in TREND_SCREEN_SYMBOLS
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_symbol_cells_worker, symbol, start, end)
                for symbol in TREND_SCREEN_SYMBOLS
            ]
            try:
                per_symbol = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    runs = [run for symbol_runs in per_symbol for run in symbol_runs]

    config = ReliabilityGateConfig()
    _apply_discovery_requirements(runs, config)

    selected_runs = _greedy_portfolio_selection(_select_sleeves(runs))
    if selected_runs:
        weights = _equal_risk_weights(selected_runs)
        selection = TrendScreenSelection(
            return_sources=tuple(run.cell.return_source for run in selected_runs),
            symbols=tuple(run.cell.symbol for run in selected_runs),
            weights=tuple(weights),
            discovery_lcb90=tuple(run.cell.policy_lcb90 for run in selected_runs),
        )
        selected_symbols = {run.cell.symbol for run in selected_runs}
        data = {
            symbol: _load_symbol_data(symbol, start, end)[:2]
            for symbol in selected_symbols
        }
        schedule, _scheduled, _stress_scheduled, qualification = _qualify(
            selected_runs, weights, data, unseal_holdout=unseal_holdout,
        )
    else:
        selection = None
        schedule = pd.Series(dtype="float64", name="leverage")
        qualification = TrendScreenQualification(
            admitted=False,
            observation_verdict="PENDING",
            fold_gate_pass=True,
            stress_verdict="PENDING",
            holdout_verdict=None,
            binding_constraint="no_discovery_eligible_cells",
        )

    return TrendScreenReport(
        profile=TREND_SCREEN_PROFILE_ID,
        universe=TREND_SCREEN_SYMBOLS,
        cells=tuple(run.cell for run in runs),
        selection=selection,
        schedule_hash=_schedule_hash(schedule),
        qualification=qualification,
    )


def persist_trend_screen_report(report: TrendScreenReport, path: Path) -> None:
    """Write the byte-deterministic report payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")


def trend_screen_report_path() -> Path:
    """Default persistence location for the pre-registered screen profile."""
    return Path("docs/results") / f"trend_screen_{TREND_SCREEN_PROFILE_ID}.json"


def _check_contract() -> None:
    """Executable assertions locking the screen surface."""
    assert run_trend_screen.__name__ == "run_trend_screen"
    assert len(TREND_SCREEN_CANDIDATES) == 30
    assert len(TREND_SCREEN_SYMBOLS) == 15
    assert TREND_SCREEN_PROFILE_ID == "baseline_gate_performance_v1"


_check_contract()
