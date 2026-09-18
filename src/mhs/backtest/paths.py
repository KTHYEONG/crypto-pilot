"""Shared process decision paths evaluated across cost tiers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.backtest.contracts import (
    PROCESS_CERTIFICATION_LEVEL,
    ProcessBacktestReport,
    ProcessMarketData,
    ProcessPath,
    RefitRecord,
)
from src.mhs.backtest.market_data import (
    _admit_process_stage,
    _estimate_panel_bytes,
    _process_memory_budget,
    apply_process_execution_availability,
    load_process_market_data,
)
from src.mhs.books import scale_book_to_target_gross
from src.mhs.deploy_gate import evaluate_deploy_gate
from src.mhs.execution.pnl import mhs_ledger_pnl
from src.mhs.params import (
    CLI_GROWTH_ENVELOPE_DEFAULT,
    DISCOVERY_START,
    GROWTH_RISK_ENVELOPES,
    PROCESS_EVALUATION_CEILING,
    PROCESS_SMOOTHING_HALFLIFE_DAYS,
    STRESS_COST_MULTIPLIER,
)
from src.mhs.process import (
    ProcessExecutionPolicy,
    ProcessRiskSizingSpec,
    RefitPoint,
    apply_process_execution_policy,
    causal_volatility_scaled_exposure,
    ema_smoothing_rate,
    estimation_adjusted_mean,
    ledoit_wolf_covariance,
    long_only_growth_weights,
    monthly_refit_schedule,
    smoothed_book_path,
    step_proxy_net_returns,
    volatility_scaled_exposure,
)
from src.mhs.resources import MhsMemoryBudget, _current_tree_swap_bytes
from src.mhs.types import ExecutionSpec


def run_process_paths(
    data: ProcessMarketData, schedule: tuple[RefitPoint, ...], *,
    decision_bps: float, evaluation_bps: tuple[float, ...], leverage_cap: float,
    execution_policy: ProcessExecutionPolicy | None = None,
    risk_sizing: ProcessRiskSizingSpec | None = None,
    memory_budget: MhsMemoryBudget | None = None,
) -> tuple[ProcessPath, ...]:
    """Construct identical process decisions with interval-local refit buffers.

    Args:
        data: Corrected causal features and registered candidate books.
        schedule: Chronological refits with purged training cutoffs.
        decision_bps: Finite nonnegative decision friction.
        evaluation_bps: Nonempty ordered evaluation tiers.
        leverage_cap: Positive finite exposure ceiling.
        execution_policy: Existing adoption policy or baseline.
        risk_sizing: Causal volatility sizing contract or legacy behavior.
        memory_budget: Explicit limits or validated defaults.

    Returns:
        Ordered paths preserving targets, proxy evidence and refit records.

    Raises:
        ValueError: Schedule or process controls are invalid.
        DataIntegrityError: Financial inputs or allocation admission fail.
    """
    budget = _process_memory_budget(memory_budget)
    initial_swap_bytes = _current_tree_swap_bytes()
    if not schedule:
        raise ValueError("schedule must not be empty")
    if not data.member_books:
        raise ValueError("no candidate books")
    if not evaluation_bps:
        raise ValueError("evaluation_bps must not be empty")
    try:
        decision_value = float(decision_bps)
    except (TypeError, ValueError):
        raise ValueError(f"decision_bps must be finite, got {decision_bps}") from None
    if not np.isfinite(decision_value) or decision_value < 0.0:
        raise ValueError(f"decision_bps must be finite and >= 0, got {decision_bps}")
    tier_values: list[float] = []
    for bps in evaluation_bps:
        try:
            value = float(bps)
        except (TypeError, ValueError):
            raise ValueError(f"evaluation_bps must be finite, got {bps}") from None
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"evaluation_bps must be finite and >= 0, got {bps}")
        tier_values.append(value)
    try:
        cap_value = float(leverage_cap)
    except (TypeError, ValueError):
        raise ValueError(f"leverage_cap must be finite, got {leverage_cap}") from None
    if not np.isfinite(cap_value) or cap_value <= 0.0:
        raise ValueError(f"leverage_cap must be finite and > 0, got {leverage_cap}")
    if execution_policy is not None and not isinstance(execution_policy, ProcessExecutionPolicy):
        raise ValueError("execution_policy must be a ProcessExecutionPolicy or None")
    if risk_sizing is not None and not isinstance(risk_sizing, ProcessRiskSizingSpec):
        raise ValueError("risk_sizing must be a ProcessRiskSizingSpec or None")
    policy = execution_policy if execution_policy is not None else ProcessExecutionPolicy()
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
    del smoothed_members
    step = member_net.index[1] - member_net.index[0]
    oos_start = schedule[0].effective_from
    oos_end = schedule[-1].effective_to
    oos_days = data.decision_grid[(data.decision_grid >= oos_start) & (data.decision_grid < oos_end)]
    n_symbols = len(data.member_books[names[0]].columns)
    _admit_process_stage(
        stage="process_refit_combination", estimated_bytes=_estimate_panel_bytes(len(oos_days), n_symbols, len(names) + 1),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    refit_targets: list[pd.DataFrame] = []
    records: list[RefitRecord] = []
    for point in schedule:
        train_rows = member_net.index[member_net.index + step <= point.train_end]
        train = member_net.loc[train_rows]
        weights = long_only_growth_weights(
            estimation_adjusted_mean(train), ledoit_wolf_covariance(train)
        )
        effective_index = data.member_books[names[0]].index[
            (data.member_books[names[0]].index >= point.effective_from)
            & (data.member_books[names[0]].index < point.effective_to)
        ]
        combined = sum(
            (weights[name] * data.member_books[name].loc[effective_index] for name in names),
            start=data.member_books[names[0]].loc[effective_index] * 0.0,
        )
        target = scale_book_to_target_gross(combined, 1.0)
        del combined
        refit_targets.append(target)
        del target
        records.append(
            RefitRecord(
                point=point,
                member_weights={n: float(weights[n]) for n in names if float(weights[n]) != 0.0},
                smoothing_halflife_days=PROCESS_SMOOTHING_HALFLIFE_DAYS,
            )
        )
    targets_oos = pd.concat(refit_targets)
    del refit_targets
    smoothed_targets = smoothed_book_path(targets_oos, pd.Series(rate, index=oos_days))
    del targets_oos
    sized_targets = apply_process_execution_policy(smoothed_targets, policy)
    del smoothed_targets
    execution_mask = data.execution_mask
    sized_targets = apply_process_execution_availability(
        sized_targets, execution_mask.reindex(sized_targets.index)
    )
    _admit_process_stage(
        stage="process_hourly_ledger", estimated_bytes=_estimate_panel_bytes(len(data.grid_1h), n_symbols, 3),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    unit_1h = sized_targets.reindex(data.grid_1h, method="ffill").fillna(0.0)
    unit_net_1h, _ = mhs_ledger_pnl(unit_1h, data.opens_1h, data.bar_funding_1h, decision_value)
    _reject_invalid_ledger_returns(unit_net_1h)
    decision_unit_daily = (
        ((1.0 + unit_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
    )
    _reject_invalid_ledger_returns(decision_unit_daily)
    if risk_sizing is not None and risk_sizing.leverage_cap != cap_value:
        raise ValueError(
            f"risk_sizing leverage_cap {risk_sizing.leverage_cap} must equal leverage_cap {cap_value}"
        )
    if risk_sizing is None:
        exposure = volatility_scaled_exposure(decision_unit_daily, cap=cap_value)
    else:
        exposure = causal_volatility_scaled_exposure(decision_unit_daily, risk_sizing)
    sized = sized_targets.mul(exposure.reindex(sized_targets.index).fillna(0.0), axis=0)
    sized_1h = sized.reindex(data.grid_1h, method="ffill").fillna(0.0)
    paths: list[ProcessPath] = []
    for bps in tier_values:
        net_1h, turnover_1h = mhs_ledger_pnl(sized_1h, data.opens_1h, data.bar_funding_1h, bps)
        _reject_invalid_ledger_returns(net_1h)
        daily_returns = ((1.0 + net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
        _reject_invalid_ledger_returns(daily_returns)
        if bps == decision_value:
            unit_daily_returns = decision_unit_daily
        else:
            alt_net_1h, _ = mhs_ledger_pnl(unit_1h, data.opens_1h, data.bar_funding_1h, bps)
            _reject_invalid_ledger_returns(alt_net_1h)
            unit_daily_returns = (
                ((1.0 + alt_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
            )
            _reject_invalid_ledger_returns(unit_daily_returns)
        paths.append(
            ProcessPath(
                one_way_bps=bps,
                daily_returns=daily_returns,
                unit_daily_returns=unit_daily_returns,
                exposure=exposure,
                refits=tuple(records),
                leverage_cap=cap_value,
                execution_policy=policy,
                unit_target_weights=sized_targets,
                target_weights=sized,
                turnover_1h=turnover_1h,
                risk_sizing=risk_sizing,
            )
        )
    del unit_1h, sized_1h
    return tuple(paths)


def _reject_invalid_ledger_returns(returns: pd.Series) -> None:
    values = returns.to_numpy(dtype="float64")
    if values.size and not bool(np.isfinite(values).all()):
        raise DataIntegrityError("ledger net return is non-finite")
    if values.size and bool((values <= -1.0).any()):
        raise DataIntegrityError("ledger net return does not exceed minus one")


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
    end: pd.Timestamp = PROCESS_EVALUATION_CEILING, *,
    data_root: str | None = None,
    execution_policy: ProcessExecutionPolicy | None = None,
    risk_sizing: ProcessRiskSizingSpec | None = None,
    memory_budget: MhsMemoryBudget | None = None,
) -> ProcessBacktestReport:
    """Evaluate an explicitly configured process as hourly proxy evidence.

    The fixed evaluation ceiling preserves the unused forward interval.
    Passing a research control does not certify inventory execution,
    capacity, or eligibility for deployment.

    Args:
        start: Timezone-aware input start.
        end: Timezone-aware input end within the registered ceiling.
        data_root: Optional hourly OHLCV root using the existing dev loader.
        execution_policy: Explicit adoption control; None preserves baseline.
        risk_sizing: Research-only causal volatility sizing; None preserves baseline.
        memory_budget: Explicit process-tree limits or validated stage defaults.

    Returns:
        Same-decision base and stress evidence with the existing proxy gate.

    Raises:
        ValueError: Timestamps or process controls are invalid.
        DataIntegrityError: The evaluation exceeds the registered ceiling
            or market inputs violate a process invariant.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    if start >= end:
        raise ValueError("start must be before end")
    if end > PROCESS_EVALUATION_CEILING:
        raise DataIntegrityError(f"end {end} exceeds PROCESS_EVALUATION_CEILING")
    if execution_policy is not None and not isinstance(execution_policy, ProcessExecutionPolicy):
        raise ValueError("execution_policy must be a ProcessExecutionPolicy or None")
    if risk_sizing is not None and not isinstance(risk_sizing, ProcessRiskSizingSpec):
        raise ValueError("risk_sizing must be a ProcessRiskSizingSpec or None")
    envelope = GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT]
    base_bps = ExecutionSpec().one_way_taker_bps()
    stress_bps = base_bps * STRESS_COST_MULTIPLIER
    data = load_process_market_data(start, end, data_root=data_root, memory_budget=memory_budget)
    schedule = monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])
    if execution_policy is None:
        base, stress = run_process_paths(
            data, schedule, decision_bps=base_bps,
            evaluation_bps=(base_bps, stress_bps), leverage_cap=envelope.leverage_ceiling,
            execution_policy=execution_policy, risk_sizing=risk_sizing, memory_budget=memory_budget,
        )
    else:
        base, stress = run_process_paths(
            data, schedule, decision_bps=base_bps,
            evaluation_bps=(base_bps, stress_bps), leverage_cap=envelope.leverage_ceiling,
            execution_policy=execution_policy, risk_sizing=risk_sizing, memory_budget=memory_budget,
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
