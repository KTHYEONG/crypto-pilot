"""Shared process decision paths evaluated across cost tiers."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

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
from src.mhs.backtest.labels import (
    MaturedMemberReturns,
    ProcessClockSpec,
    build_proxy_member_returns,
    select_matured_training_returns,
)
from src.mhs.backtest.market_data import (
    _admit_process_stage,
    _estimate_panel_bytes,
    _process_memory_budget,
    apply_process_execution_availability,
    load_process_market_data,
)
from src.mhs.backtest.selection import (
    InnerFitAudit,
    InnerPolicyEvidence,
    NestedSelectionSpec,
    RefitPolicyChoice,
    TrainingWindowSpec,
    choose_refit_policy,
)
from src.mhs.books import scale_book_to_target_gross
from src.mhs.deploy_gate import evaluate_deploy_gate
from src.mhs.execution.pnl import mhs_ledger_pnl
from src.mhs.params import (
    CLI_GROWTH_ENVELOPE_DEFAULT,
    DISCOVERY_START,
    EVIDENCE_GATE_ALPHA,
    GROWTH_RISK_ENVELOPES,
    NULL_BOOTSTRAP_SEED,
    PROCESS_EVALUATION_CEILING,
    PROCESS_FEATURE_CANDIDATES,
    PROCESS_FUNDING_CARRY_CANDIDATES_HOURS,
    PROCESS_MIN_TRAIN_DAYS,
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
    matured_monthly_refit_schedule,
    smoothed_book_path,
    step_proxy_net_returns,
    volatility_scaled_exposure,
)
from src.mhs.resources import MhsMemoryBudget, _current_tree_swap_bytes
from src.mhs.types import ExecutionSpec

if TYPE_CHECKING:
    from src.mhs.backtest.journal import ProcessProcedureDefinition

_logger = logging.getLogger(__name__)


def _windowed_weights(
    member_evidence: MaturedMemberReturns,
    point: RefitPoint,
    window: TrainingWindowSpec,
    names: list[str],
) -> tuple[pd.Series, pd.Timestamp | None, int, pd.Timestamp | None, pd.Timestamp | None]:
    if window.kind == "equal_member":
        train = select_matured_training_returns(member_evidence, fit_cutoff=point.train_end)
        weights = pd.Series(1.0 / len(names), index=names)
        return (weights, None, len(train), None, None)
    train_start: pd.Timestamp | None = None
    if window.kind == "rolling" and window.months is not None:
        train_start = pd.Timestamp(point.train_end - pd.DateOffset(months=int(window.months)))
    train = select_matured_training_returns(member_evidence, fit_cutoff=point.train_end, train_start=train_start)
    if len(train) < 2:
        return (pd.Series(0.0, index=names), train_start, len(train), None, None)
    weights = long_only_growth_weights(estimation_adjusted_mean(train), ledoit_wolf_covariance(train))
    last = train.index[-1]
    pos = int(member_evidence.returns.index.get_loc(last))
    return (
        weights,
        train_start,
        len(train),
        pd.Timestamp(member_evidence.label_end[pos]),
        pd.Timestamp(member_evidence.available_at[pos]),
    )


def run_process_paths(
    data: ProcessMarketData,
    schedule: tuple[RefitPoint, ...],
    *,
    decision_bps: float,
    evaluation_bps: tuple[float, ...],
    leverage_cap: float,
    execution_policy: ProcessExecutionPolicy | None = None,
    risk_sizing: ProcessRiskSizingSpec | None = None,
    memory_budget: MhsMemoryBudget | None = None,
    clock: ProcessClockSpec | None = None,
    member_evidence: MaturedMemberReturns | None = None,
    training_window: TrainingWindowSpec | None = None,
    selection_spec: NestedSelectionSpec | None = None,
    inner_evidence: Mapping[str, InnerPolicyEvidence] | None = None,
) -> tuple[ProcessPath, ...]:
    """Construct identical process decisions with interval-local refit buffers.

    Fit only complete labels available at each refit information cutoff and carry
    the exact signal-publication clock with the resulting path. Legacy omissions
    are explicitly provisional and cannot establish label-maturity certification.

    Args:
        data: Corrected causal features and registered candidate books.
        schedule: Chronological refits with purged training cutoffs.
        decision_bps: Finite nonnegative decision friction.
        evaluation_bps: Nonempty ordered evaluation tiers.
        leverage_cap: Positive finite exposure ceiling.
        execution_policy: Existing adoption policy or baseline.
        risk_sizing: Causal volatility sizing contract or legacy behavior.
        memory_budget: Explicit limits or validated defaults.
        clock: Registered signal-publication clock for matured-label fitting.
        member_evidence: Complete-interval member labels with knowledge masks.
        training_window: Single registered estimator window without nested selection.
        selection_spec: Frozen comparison pool, control and uncertainty rules.
        inner_evidence: Fixed-pool causal combined-policy evidence for selection.

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
    if (clock is None) != (member_evidence is None):
        raise ValueError("clock and member_evidence must be supplied together")
    if training_window is not None and selection_spec is not None:
        raise ValueError("training_window and selection_spec must not be supplied together")
    if selection_spec is not None and (
        inner_evidence is None or set(inner_evidence.keys()) != {p.policy_id for p in selection_spec.policies}
    ):
        raise DataIntegrityError("selection_spec requires exact declared inner_evidence membership")
    policy = execution_policy if execution_policy is not None else ProcessExecutionPolicy()
    rate = ema_smoothing_rate(PROCESS_SMOOTHING_HALFLIFE_DAYS)
    names = list(data.member_books.keys())
    if member_evidence is not None and list(member_evidence.returns.columns) != names:
        raise DataIntegrityError("member evidence must cover the declared candidate books exactly")
    smoothed_members = {
        name: smoothed_book_path(data.member_books[name], pd.Series(rate, index=data.member_books[name].index))
        for name in names
    }
    member_net = pd.DataFrame(
        {
            name: step_proxy_net_returns(smoothed_members[name], data.log_close_step, data.funding_step, decision_bps)
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
        stage="process_refit_combination",
        estimated_bytes=_estimate_panel_bytes(len(oos_days), n_symbols, len(names) + 1),
        budget=budget,
        replay=False,
        initial_swap_bytes=initial_swap_bytes,
    )
    refit_targets: list[pd.DataFrame] = []
    records: list[RefitRecord] = []
    policy_choices: list[RefitPolicyChoice] = []
    windows: dict[str, TrainingWindowSpec] = {}
    if selection_spec is not None:
        windows = {p.policy_id: p for p in selection_spec.policies}
    for point in schedule:
        choice: RefitPolicyChoice | None = None
        active_window: TrainingWindowSpec | None = training_window
        if selection_spec is not None and inner_evidence is not None:
            choice = choose_refit_policy(inner_evidence, point, spec=selection_spec)
            policy_choices.append(choice)
            active_window = windows.get(choice.policy_id) if choice.policy_id is not None else None
        if member_evidence is not None and clock is not None:
            if active_window is not None:
                weights, w_start, n_labels, _, _ = _windowed_weights(member_evidence, point, active_window, names)
                rec_policy = active_window.policy_id
                rec_start = w_start
                rec_count: int | None = n_labels
            else:
                if choice is not None and choice.policy_id is None:
                    weights = pd.Series(0.0, index=names)
                    rec_policy = None
                    rec_start = None
                    rec_count = 0
                else:
                    train = select_matured_training_returns(member_evidence, fit_cutoff=point.train_end)
                    rec_policy = None
                    rec_start = None
                    rec_count = len(train)
                    if len(train) < 2:
                        _logger.warning(
                            "[DATA] stage=refit_no_trade reason=insufficient_mature_labels cutoff=%s rows=%d",
                            point.train_end,
                            len(train),
                        )
                        weights = pd.Series(0.0, index=names)
                    else:
                        weights = long_only_growth_weights(
                            estimation_adjusted_mean(train), ledoit_wolf_covariance(train)
                        )
        else:
            train_rows = member_net.index[member_net.index + step <= point.train_end]
            train = member_net.loc[train_rows]
            weights = long_only_growth_weights(estimation_adjusted_mean(train), ledoit_wolf_covariance(train))
            rec_policy = None
            rec_start = None
            rec_count = None
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
                policy_id=rec_policy,
                train_start=rec_start,
                n_train_labels=rec_count,
            )
        )
    targets_oos = pd.concat(refit_targets)
    del refit_targets
    smoothed_targets = smoothed_book_path(targets_oos, pd.Series(rate, index=oos_days))
    del targets_oos
    sized_targets = apply_process_execution_policy(smoothed_targets, policy)
    del smoothed_targets
    execution_mask = data.execution_mask
    sized_targets = apply_process_execution_availability(sized_targets, execution_mask.reindex(sized_targets.index))
    _admit_process_stage(
        stage="process_hourly_ledger",
        estimated_bytes=_estimate_panel_bytes(len(data.grid_1h), n_symbols, 3),
        budget=budget,
        replay=False,
        initial_swap_bytes=initial_swap_bytes,
    )
    unit_1h = sized_targets.reindex(data.grid_1h, method="ffill").fillna(0.0)
    unit_net_1h, _ = mhs_ledger_pnl(unit_1h, data.opens_1h, data.bar_funding_1h, decision_value)
    _reject_invalid_ledger_returns(unit_net_1h)
    decision_unit_daily = ((1.0 + unit_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
    _reject_invalid_ledger_returns(decision_unit_daily)
    if risk_sizing is not None and risk_sizing.leverage_cap != cap_value:
        raise ValueError(f"risk_sizing leverage_cap {risk_sizing.leverage_cap} must equal leverage_cap {cap_value}")
    if risk_sizing is None:
        exposure = volatility_scaled_exposure(decision_unit_daily, cap=cap_value)
    else:
        exposure = causal_volatility_scaled_exposure(decision_unit_daily, risk_sizing)
    sized = sized_targets.mul(exposure.reindex(sized_targets.index).fillna(0.0), axis=0)
    sized_1h = sized.reindex(data.grid_1h, method="ffill").fillna(0.0)
    if member_evidence is not None and clock is not None:
        signal_available_at: pd.DatetimeIndex | None = pd.DatetimeIndex(sized_targets.index + clock.bar_completion_lag)
        clock_mode: Literal["legacy_purge", "matured_labels"] = "matured_labels"
    else:
        signal_available_at = None
        clock_mode = "legacy_purge"
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
            unit_daily_returns = ((1.0 + alt_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
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
                signal_available_at=signal_available_at,
                clock_mode=clock_mode,
                policy_choices=tuple(policy_choices),
            )
        )
    del unit_1h, sized_1h
    return tuple(paths)


def build_inner_policy_evidence(
    data: ProcessMarketData,
    member_evidence: MaturedMemberReturns,
    *,
    clock: ProcessClockSpec,
    selection_spec: NestedSelectionSpec,
    decision_bps: float,
    leverage_cap: float,
    execution_policy: ProcessExecutionPolicy | None,
    risk_sizing: ProcessRiskSizingSpec | None,
    memory_budget: MhsMemoryBudget | None,
) -> dict[str, InnerPolicyEvidence]:
    """Materialize each registered chronological policy trajectory once for later prefix-only nested comparisons.

    Args:
        data: Canonical causal books and publication clocks.
        member_evidence: Complete maturity-tagged member training labels.
        clock: Frozen decision and fitting clock.
        selection_spec: Fixed estimator/control pool and evidence requirements.
        decision_bps: Registered screening friction used by every trial.
        leverage_cap: Existing registered exposure ceiling.
        execution_policy: Same registered adoption policy for every trial.
        risk_sizing: Explicit named sizing mode shared by every trial.
        memory_budget: Existing admitted working-set controls.
    Returns:
        Fixed-pool causal combined-policy evidence, explicitly hourly-proxy until
        execution-native equivalents with matching identities are available.
    Raises:
        DataIntegrityError: Chronology, provenance, financial state or resources fail.
    """
    del decision_bps, leverage_cap, execution_policy, risk_sizing
    budget = _process_memory_budget(memory_budget)
    initial_swap_bytes = _current_tree_swap_bytes()
    names = list(data.member_books.keys())
    if list(member_evidence.returns.columns) != names:
        raise DataIntegrityError("member evidence must cover the declared candidate books exactly")
    decisions = member_evidence.returns.index
    _admit_process_stage(
        stage="process_inner_evidence",
        estimated_bytes=_estimate_panel_bytes(len(decisions), len(names), len(selection_spec.policies)),
        budget=budget,
        replay=False,
        initial_swap_bytes=initial_swap_bytes,
    )
    schedule = matured_monthly_refit_schedule(
        member_evidence,
        data.decision_grid[-1],
        min_train_days=PROCESS_MIN_TRAIN_DAYS,
        fit_latency=clock.fit_latency,
    )
    first_label = pd.Timestamp(member_evidence.label_start[0])
    out: dict[str, InnerPolicyEvidence] = {}
    for window in selection_spec.policies:
        if window.kind == "rolling" and window.months is not None:
            ready = pd.Timestamp(first_label + pd.DateOffset(months=int(window.months)))
        else:
            ready = pd.Timestamp(first_label + pd.Timedelta(days=PROCESS_MIN_TRAIN_DAYS))
        rets: list[float] = []
        turns: list[float] = []
        valids: list[bool] = []
        stamps: list[pd.Timestamp] = []
        audits: list[InnerFitAudit] = []
        prev_w: pd.Series | None = None
        for point in schedule:
            weights, w_start, n_labels, last_end, last_avail = _windowed_weights(member_evidence, point, window, names)
            audits.append(InnerFitAudit(point, w_start, n_labels, last_end, last_avail))
            step_turn = 0.0 if prev_w is None else float((weights - prev_w).abs().sum())
            prev_w = weights
            span = decisions[(decisions >= point.effective_from) & (decisions < point.effective_to)]
            for stamp in span:
                row = member_evidence.returns.loc[stamp].to_numpy(dtype="float64")
                known = bool(member_evidence.known.loc[stamp].to_numpy(dtype=bool).all())
                if known and bool(np.isfinite(row).all()):
                    rets.append(float(weights.to_numpy(dtype="float64") @ row))
                    valids.append(True)
                else:
                    rets.append(float("nan"))
                    valids.append(False)
                turns.append(step_turn)
                stamps.append(pd.Timestamp(stamp))
                step_turn = 0.0
        idx = pd.DatetimeIndex(stamps)
        avail_pos = member_evidence.returns.index.get_indexer(idx)
        avail = pd.DatetimeIndex(member_evidence.available_at[avail_pos])
        out[window.policy_id] = InnerPolicyEvidence(
            window.policy_id,
            pd.Series(rets, index=idx),
            avail,
            pd.Series(turns, index=idx),
            pd.Series(valids, index=idx),
            member_evidence.procedure_digest,
            member_evidence.input_manifest_digest,
            "hourly_proxy",
            ready,
            tuple(audits),
        )
    return out


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


def baseline_process_procedure(*, code_digest: str) -> ProcessProcedureDefinition:
    """Resolve the established MHS research controls into an explicit immutable baseline.

    Args:
        code_digest: Actual current source/dependency identity, never a display label.
    Returns:
        A historical research definition with the declared members, clock, control,
        estimator comparisons, proxy limitations and all approval requirements.
    Raises:
        DataIntegrityError: Existing controls cannot produce a valid typed definition.
    """
    from src.mhs.backtest.certification import REQUIRED_CHECK_NAMES, ValidationInferenceSpec
    from src.mhs.backtest.journal import PROCEDURE_SCHEMA_VERSION, ProcessProcedureDefinition
    from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT

    if not isinstance(code_digest, str) or not code_digest:
        raise DataIntegrityError("code_digest must be a nonempty identity")
    member_ids = tuple(PROCESS_FEATURE_CANDIDATES) + tuple(
        f"funding_carry_{h}h" for h in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS
    )
    clock = ProcessClockSpec(
        decision_period=pd.Timedelta(hours=24),
        bar_completion_lag=pd.Timedelta(hours=1),
        fit_latency=pd.Timedelta(0),
    )
    selection = NestedSelectionSpec(
        policies=(
            TrainingWindowSpec(policy_id="expanding", kind="expanding", months=None),
            TrainingWindowSpec(policy_id="rolling_12m", kind="rolling", months=12),
            TrainingWindowSpec(policy_id="rolling_24m", kind="rolling", months=24),
            TrainingWindowSpec(policy_id="equal_member", kind="equal_member", months=None),
        ),
        control_policy_id="equal_member",
        minimum_inner_labels=PROCESS_MIN_TRAIN_DAYS,
        alpha=float(EVIDENCE_GATE_ALPHA),
        bootstrap_paths=500,
        seed=int(NULL_BOOTSTRAP_SEED),
        fit_latency=pd.Timedelta(0),
    )
    family_alpha = float(EVIDENCE_GATE_ALPHA)
    procedure_budget = 4
    look_budget = 1
    endpoint_budget = 4
    local_alpha = family_alpha / (procedure_budget * look_budget * endpoint_budget)
    required_paths = math.ceil(20.0 / local_alpha)
    inference = ValidationInferenceSpec(
        family_alpha=family_alpha,
        procedure_budget=procedure_budget,
        look_budget=look_budget,
        endpoint_budget=endpoint_budget,
        bootstrap_paths=required_paths,
        seed=int(NULL_BOOTSTRAP_SEED),
        resample_batch_paths=500,
        minimum_block_days=1,
    )
    procedure = ProcessProcedureDefinition(
        schema_version=PROCEDURE_SCHEMA_VERSION,
        code_digest=code_digest,
        data_policy=str(MHS_DATA_POLICY_DEFAULT),
        universe_partition="dev",
        member_ids=member_ids,
        clock=clock,
        selection=selection,
        inference=inference,
        execution_policy=ProcessExecutionPolicy(),
        risk_sizing=None,
        sizing_source="hourly_proxy",
        member_evidence_source="daily_step_proxy",
        execution_spec=ExecutionSpec(),
        envelope=GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT],
        initial_equity=1.0,
        smoothing_halflife_days=float(PROCESS_SMOOTHING_HALFLIFE_DAYS),
        required_checks=tuple(REQUIRED_CHECK_NAMES),
    )
    return procedure


def evaluate_process_backtest(
    start: pd.Timestamp = DISCOVERY_START,
    end: pd.Timestamp = PROCESS_EVALUATION_CEILING,
    *,
    data_root: str | None = None,
    execution_policy: ProcessExecutionPolicy | None = None,
    risk_sizing: ProcessRiskSizingSpec | None = None,
    memory_budget: MhsMemoryBudget | None = None,
    procedure: ProcessProcedureDefinition | None = None,
    member_evidence: MaturedMemberReturns | None = None,
) -> ProcessBacktestReport:
    """Evaluate an explicitly configured process as hourly proxy evidence.

    Evaluate the exact defined historical procedure with maturity-tagged nested
    selection. Proxy evidence is comparative research state and never a deployment verdict.

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
        procedure: Frozen historical definition; None preserves legacy research.
        member_evidence: Inventory-native labels admitted only on identity match.

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
    if procedure is None and member_evidence is not None:
        raise DataIntegrityError("member_evidence requires a typed procedure for admission")
    if procedure is not None:
        from src.mhs.backtest.journal import ProcessProcedureDefinition as _Procedure
        from src.mhs.backtest.journal import process_procedure_digest as _digest
        from src.mhs.deploy_gate import DeployGateResult as _Gate

        if not isinstance(procedure, _Procedure):
            raise DataIntegrityError("procedure must be a ProcessProcedureDefinition or None")
        if member_evidence is not None and not isinstance(member_evidence, MaturedMemberReturns):
            raise DataIntegrityError("member_evidence must be MaturedMemberReturns or None")
        if execution_policy is not None and execution_policy != procedure.execution_policy:
            raise DataIntegrityError("execution_policy must match the frozen procedure")
        if risk_sizing is not None and risk_sizing != procedure.risk_sizing:
            raise DataIntegrityError("risk_sizing must match the frozen procedure")
        if member_evidence is not None:
            if member_evidence.source != procedure.member_evidence_source:
                raise DataIntegrityError("member evidence source does not match the procedure")
            if list(member_evidence.returns.columns) != list(procedure.member_ids):
                raise DataIntegrityError("member evidence must cover the declared procedure members exactly")
            if member_evidence.procedure_digest != _digest(procedure):
                raise DataIntegrityError("member evidence procedure identity does not match")
        digest = _digest(procedure)
        envelope = procedure.envelope
        base_bps = procedure.execution_spec.one_way_taker_bps()
        stress_bps = base_bps * STRESS_COST_MULTIPLIER
        data = load_process_market_data(start, end, data_root=data_root, memory_budget=memory_budget)
        clock = procedure.clock
        resolved = member_evidence
        if resolved is None:
            resolved = build_proxy_member_returns(
                data,
                clock=clock,
                one_way_bps=base_bps,
                procedure_digest=digest,
                input_manifest_digest=None,
            )
        data = dataclasses.replace(data, member_evidence=resolved)
        schedule = matured_monthly_refit_schedule(
            resolved,
            data.decision_grid[-1],
            min_train_days=PROCESS_MIN_TRAIN_DAYS,
            fit_latency=clock.fit_latency,
        )
        inner_evidence = build_inner_policy_evidence(
            data,
            resolved,
            clock=clock,
            selection_spec=procedure.selection,
            decision_bps=base_bps,
            leverage_cap=envelope.leverage_ceiling,
            execution_policy=procedure.execution_policy,
            risk_sizing=procedure.risk_sizing,
            memory_budget=memory_budget,
        )
        base, stress = run_process_paths(
            data,
            schedule,
            decision_bps=base_bps,
            evaluation_bps=(base_bps, stress_bps),
            leverage_cap=envelope.leverage_ceiling,
            execution_policy=procedure.execution_policy,
            risk_sizing=procedure.risk_sizing,
            memory_budget=memory_budget,
            clock=clock,
            member_evidence=resolved,
            selection_spec=procedure.selection,
            inner_evidence=inner_evidence,
        )
        provisional = evaluate_deploy_gate(
            fold_returns=quarter_fold_returns(base.daily_returns),
            fold_stress_returns=quarter_fold_returns(stress.daily_returns),
            integrity_reasons=(),
            envelope=envelope,
        )
        gate = _Gate(
            go=False,
            reason_codes=tuple(sorted(set(provisional.reason_codes) | {"PROXY_NEVER_DEPLOYS"})),
            metrics=dict(provisional.metrics),
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
    envelope = GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT]
    base_bps = ExecutionSpec().one_way_taker_bps()
    stress_bps = base_bps * STRESS_COST_MULTIPLIER
    data = load_process_market_data(start, end, data_root=data_root, memory_budget=memory_budget)
    clock = ProcessClockSpec(
        decision_period=pd.Timedelta(hours=24),
        bar_completion_lag=data.grid_1h[1] - data.grid_1h[0],
        fit_latency=pd.Timedelta(0),
    )
    procedure_digest = hashlib.sha256(
        "|".join(
            [
                ",".join(data.member_books.keys()),
                str(PROCESS_SMOOTHING_HALFLIFE_DAYS),
                str(clock.decision_period),
                str(clock.bar_completion_lag),
                str(clock.fit_latency),
                str(base_bps),
            ]
        ).encode("utf-8")
    ).hexdigest()
    member_evidence = build_proxy_member_returns(
        data,
        clock=clock,
        one_way_bps=base_bps,
        procedure_digest=procedure_digest,
        input_manifest_digest=None,
    )
    data = dataclasses.replace(data, member_evidence=member_evidence)
    schedule = matured_monthly_refit_schedule(
        member_evidence,
        data.decision_grid[-1],
        min_train_days=PROCESS_MIN_TRAIN_DAYS,
        fit_latency=clock.fit_latency,
    )
    if execution_policy is None:
        base, stress = run_process_paths(
            data,
            schedule,
            decision_bps=base_bps,
            evaluation_bps=(base_bps, stress_bps),
            leverage_cap=envelope.leverage_ceiling,
            execution_policy=execution_policy,
            risk_sizing=risk_sizing,
            memory_budget=memory_budget,
            clock=clock,
            member_evidence=member_evidence,
        )
    else:
        base, stress = run_process_paths(
            data,
            schedule,
            decision_bps=base_bps,
            evaluation_bps=(base_bps, stress_bps),
            leverage_cap=envelope.leverage_ceiling,
            execution_policy=execution_policy,
            risk_sizing=risk_sizing,
            memory_budget=memory_budget,
            clock=clock,
            member_evidence=member_evidence,
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
