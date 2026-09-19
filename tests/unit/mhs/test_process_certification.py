"""Invariant guards for dependence-aware process validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.backtest.certification import (
    EvaluationContext,
    EvidenceCheck,
    ValidationInferenceSpec,
    assess_process_validation,
    inventory_daily_evidence,
)
from src.mhs.backtest.contracts import ProcessInventoryReport
from src.mhs.deploy_gate import DeployGateResult
from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult
from src.mhs.params import GrowthRiskEnvelope
from src.mhs.resources import MhsMemoryBudget

NAMES = (
    "input_seal",
    "availability",
    "pit_universe",
    "label_maturity",
    "selection_independence",
    "execution_consistency",
    "live_parity",
    "margin_survival",
    "capital_capacity",
    "joint_stress",
    "forward_independence",
)


def _grid(days: int = 60) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=days * 48, freq="30min", tz="UTC")


def _replay(grid: pd.DatetimeIndex, equity: pd.Series, *, valid: bool = True) -> StrategyExecutionReplayResult:
    n = len(equity)
    zeros = pd.Series([0.0] * n, index=equity.index, dtype="float64")
    ledger = SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().dropna(),
        simulated_units=None,
        mark_to_market_pnl=zeros,
        funding_charge=zeros,
        fee_charge=zeros,
        fill_turnover=zeros,
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="OHLCV_CLOSE_FALLBACK",
        primary_valid=valid,
        invalid_reasons=() if valid else ("MISSING_DATA",),
        data_gaps=(),
    )
    avail = pd.DatetimeIndex(grid + pd.Timedelta(minutes=30), tz="UTC")
    return StrategyExecutionReplayResult(
        simulated_fills=pd.DataFrame(),
        ledger=ledger,
        simulated_units=pd.DataFrame(),
        simulated_notional_weights=pd.DataFrame(),
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="OHLCV_CLOSE_FALLBACK",
        submit_times=pd.Series(dtype="datetime64[ns, UTC]"),
        fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
        fill_count=0,
        unfilled_count=0,
        fallback_count=0,
        all_intent_shortfall_bps=0.0,
        forced_exit_count=0,
        forced_exit_notional=0.0,
        termination_counts={},
        unsupported_assumptions=(),
        elapsed_seconds=0.0,
        ledger_available_at=avail,
    )


def _growing_replays(days: int = 60, daily: float = 0.002) -> tuple[StrategyExecutionReplayResult, StrategyExecutionReplayResult]:
    grid = _grid(days)
    levels = pd.Series(1.0 + 0.00004 * np.arange(len(grid)), index=grid, dtype="float64")
    base = _replay(grid, levels, valid=True)
    stress = _replay(grid, levels * 0.999999, valid=True)
    assert daily is not None
    return base, stress


def _envelope() -> GrowthRiskEnvelope:
    return GrowthRiskEnvelope(
        name="test",
        max_drawdown=0.6,
        max_drawdown_prob=0.5,
        ruin_fraction=0.6,
        max_ruin_prob=0.5,
        horizon_years=0.25,
        leverage_ceiling=1.0,
    )


def _spec(paths: int = 400) -> ValidationInferenceSpec:
    return ValidationInferenceSpec(
        family_alpha=0.2,
        procedure_budget=1,
        look_budget=1,
        endpoint_budget=4,
        bootstrap_paths=paths,
        seed=7,
        resample_batch_paths=200,
        minimum_block_days=1,
    )


def _context(
    grid_days: int = 60, *, role: str = "historical", spec: ValidationInferenceSpec | None = None
) -> EvaluationContext:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=grid_days)
    return EvaluationContext(
        role=role,  # type: ignore[arg-type]
        procedure_digest="proc",
        code_digest="code",
        input_manifest_digest="manifest",
        interval_start=start,
        interval_end=end,
        registered_at=start - pd.Timedelta(days=1) if role == "forward" else None,
        consulted_through=start - pd.Timedelta(days=1) if role == "forward" else None,
        family_id="fam",
        look_ordinal=1,
        inference_spec=spec,
        journal_complete=True,
        observed_through=end + pd.Timedelta(hours=1),
    )


def _passed_checks(ctx: EvaluationContext) -> tuple[EvidenceCheck, ...]:
    return tuple(
        EvidenceCheck(
            requirement=name,
            status="passed",
            procedure_digest=ctx.procedure_digest,
            input_manifest_digest=ctx.input_manifest_digest,
            code_digest=ctx.code_digest,
            interval_start=ctx.interval_start - pd.Timedelta(days=1),
            interval_end=ctx.interval_end + pd.Timedelta(days=1),
            artifact_digest="artifact",
            reason_codes=(),
        )
        for name in NAMES
    )


def test_assess_blocks_approval_on_invalid_ledger() -> None:
    grid = _grid(10)
    levels = pd.Series([1.0] * len(grid), index=grid, dtype="float64")
    base = _replay(grid, levels, valid=False)
    stress = _replay(grid, levels, valid=True)
    ctx = _context(10)
    result = assess_process_validation(
        base, stress, context=ctx, checks=(), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.accounting_valid is False
    assert result.historical_acceptance != "passed"
    assert result.forward_acceptance != "passed"
    assert result.gate.go is False


def test_assess_leaves_requirements_unverified_without_evidence() -> None:
    base, stress = _growing_replays(30)
    ctx = _context(30)
    result = assess_process_validation(
        base, stress, context=ctx, checks=(), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.accounting_valid is True
    assert result.gate.go is False
    assert any(r.startswith("REQUIREMENT_UNVERIFIED:") for r in result.reason_codes)


def test_assess_rejects_mismatched_attestation_identities() -> None:
    base, stress = _growing_replays(30)
    ctx = _context(30, spec=_spec())
    checks = tuple(
        EvidenceCheck(
            requirement=name,
            status="passed",
            procedure_digest="other-proc",
            input_manifest_digest="manifest",
            code_digest="code",
            interval_start=ctx.interval_start - pd.Timedelta(days=1),
            interval_end=ctx.interval_end + pd.Timedelta(days=1),
            artifact_digest="artifact",
            reason_codes=(),
        )
        for name in NAMES
    )
    result = assess_process_validation(
        base, stress, context=ctx, checks=checks, envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.gate.go is False
    assert "PROCEDURE_IDENTITY_MISMATCH" in result.reason_codes
    assert all(c.status != "passed" for c in result.requirements)


def test_historical_acceptance_without_forward_approval() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="historical", spec=_spec())
    result = assess_process_validation(
        base,
        stress,
        context=ctx,
        checks=_passed_checks(ctx)[:6],
        envelope=_envelope(),
        memory_budget=MhsMemoryBudget(),
    )
    assert result.historical_acceptance == "passed"
    assert result.forward_acceptance == "unverified"
    assert result.gate.go is False
    assert "HISTORICAL_ONLY" in result.reason_codes


def test_forward_independence_fails_on_consulted_interval() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="forward", spec=_spec())
    ctx = EvaluationContext(
        role="forward",
        procedure_digest=ctx.procedure_digest,
        code_digest=ctx.code_digest,
        input_manifest_digest=ctx.input_manifest_digest,
        interval_start=ctx.interval_start,
        interval_end=ctx.interval_end,
        registered_at=ctx.registered_at,
        consulted_through=ctx.interval_start + pd.Timedelta(days=1),
        family_id=ctx.family_id,
        look_ordinal=1,
        inference_spec=ctx.inference_spec,
        journal_complete=True,
        observed_through=ctx.observed_through,
    )
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.forward_acceptance != "passed"
    assert result.gate.go is False
    assert "FORWARD_INTERVAL_CONSULTED" in result.reason_codes


def test_assess_blocks_recycled_alpha_on_exhausted_look() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="forward", spec=_spec())
    ctx = EvaluationContext(
        role="forward",
        procedure_digest=ctx.procedure_digest,
        code_digest=ctx.code_digest,
        input_manifest_digest=ctx.input_manifest_digest,
        interval_start=ctx.interval_start,
        interval_end=ctx.interval_end,
        registered_at=ctx.registered_at,
        consulted_through=ctx.consulted_through,
        family_id=ctx.family_id,
        look_ordinal=5,
        inference_spec=ctx.inference_spec,
        journal_complete=True,
        observed_through=ctx.observed_through,
    )
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.forward_acceptance != "passed"
    assert result.gate.go is False
    assert "LOOK_BUDGET_EXHAUSTED" in result.reason_codes


def test_invalid_accounting_reports_labeled_diagnostics() -> None:
    grid = _grid(10)
    up = pd.Series(1.0 + 0.001 * np.arange(len(grid)), index=grid, dtype="float64")
    base = _replay(grid, up, valid=False)
    stress = _replay(grid, up, valid=False)
    ctx = _context(10)
    result = assess_process_validation(
        base, stress, context=ctx, checks=(), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.diagnostic_metrics["base_ann_log_growth_lcb"] is None
    assert result.diagnostic_metrics["p_mdd_breach_ucb"] is None
    assert float(result.diagnostic_metrics["observed_equity_rows"] or 0.0) > 0.0
    evidence = inventory_daily_evidence(_replay(grid, up, valid=True))
    assert len(evidence.returns) > 0


def test_report_carries_single_authoritative_verdict() -> None:
    base, stress = _growing_replays(10)
    ctx = _context(10)
    result = assess_process_validation(
        base, stress, context=ctx, checks=(), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    gate: DeployGateResult = result.gate
    report = ProcessInventoryReport(
        proxy=_proxy_report(base, stress, gate),
        base=base,
        stress=stress,
        gate=gate,
        resource_measurements=(),
        memory_stats=_memory_stats(),
        validation=result,
    )
    assert report.gate is gate
    assert report.validation is result
    assert report.gate.go is False


def _proxy_report(base: StrategyExecutionReplayResult, stress: StrategyExecutionReplayResult, gate: DeployGateResult):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessBacktestReport
    from src.mhs.backtest.inventory import _inventory_daily_returns

    daily = _inventory_daily_returns(base)
    from src.mhs.backtest.paths import quarter_fold_returns  # noqa: F401
    from src.mhs.process import ProcessExecutionPolicy

    import pandas as pd

    frame = pd.DataFrame({"AUSDT": [0.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]))
    from src.mhs.backtest.contracts import ProcessPath

    path = ProcessPath(
        one_way_bps=1.0,
        daily_returns=daily,
        unit_daily_returns=daily,
        exposure=daily * 0.0,
        refits=(),
        leverage_cap=1.0,
        execution_policy=ProcessExecutionPolicy(),
        unit_target_weights=frame,
        target_weights=frame,
        turnover_1h=daily * 0.0,
    )
    return ProcessBacktestReport(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-11", tz="UTC"),
        certification_level="process_proxy_1h_ledger",
        n_candidates=1,
        base=path,
        stress=path,
        gate=gate,
    )


def _memory_stats():  # type: ignore[no-untyped-def]
    from src.mhs.resources import ProcessTreeMemoryStats

    return ProcessTreeMemoryStats(
        tree_pss_peak_bytes=1,
        tree_uss_peak_bytes=1,
        min_system_available_bytes=1,
        max_concurrent_procs=1,
        samples_taken=0,
    )


def test_assess_rejects_duplicate_requirement_names() -> None:
    base, stress = _growing_replays(10)
    ctx = _context(10)
    check = EvidenceCheck(
        requirement="input_seal",
        status="passed",
        procedure_digest="proc",
        input_manifest_digest="manifest",
        code_digest="code",
        interval_start=ctx.interval_start - pd.Timedelta(days=1),
        interval_end=ctx.interval_end + pd.Timedelta(days=1),
        artifact_digest="artifact",
        reason_codes=(),
    )
    try:
        assess_process_validation(
            base, stress, context=ctx, checks=(check, check), envelope=_envelope(), memory_budget=MhsMemoryBudget()
        )
    except DataIntegrityError:
        pass
    else:
        raise AssertionError("duplicate requirements must raise")


def test_failed_prerequisite_blocks_historical_inference() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="historical", spec=_spec())
    checks = list(_passed_checks(ctx)[:6])
    failed = EvidenceCheck(
        requirement=checks[0].requirement,
        status="failed",
        procedure_digest=checks[0].procedure_digest,
        input_manifest_digest=checks[0].input_manifest_digest,
        code_digest=checks[0].code_digest,
        interval_start=checks[0].interval_start,
        interval_end=checks[0].interval_end,
        artifact_digest=checks[0].artifact_digest,
        reason_codes=("SEAL_BROKEN",),
    )
    checks[0] = failed
    result = assess_process_validation(
        base, stress, context=ctx, checks=tuple(checks), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.historical_acceptance == "failed"
    assert result.gate.go is False
    assert f"REQUIREMENT_FAILED:{failed.requirement}" in result.reason_codes


def test_passed_prerequisites_without_spec_stay_unverified() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="historical", spec=None)
    result = assess_process_validation(
        base,
        stress,
        context=ctx,
        checks=_passed_checks(ctx)[:6],
        envelope=_envelope(),
        memory_budget=MhsMemoryBudget(),
    )
    assert result.historical_acceptance == "unverified"
    assert result.gate.go is False


def test_unknown_requirement_name_raises() -> None:
    from types import SimpleNamespace

    base, stress = _growing_replays(10)
    ctx = _context(10)
    foreign = SimpleNamespace(
        requirement="not_a_requirement",
        status="passed",
        procedure_digest="proc",
        input_manifest_digest="manifest",
        code_digest="code",
        interval_start=ctx.interval_start,
        interval_end=ctx.interval_end,
        artifact_digest="artifact",
        reason_codes=(),
    )
    try:
        assess_process_validation(
            base, stress, context=ctx, checks=(foreign,), envelope=_envelope(), memory_budget=MhsMemoryBudget()  # type: ignore[arg-type]
        )
    except DataIntegrityError:
        pass
    else:
        raise AssertionError("unknown requirement must raise")


def test_provided_unverified_check_keeps_go_closed() -> None:
    base, stress = _growing_replays(10)
    ctx = _context(10)
    check = EvidenceCheck(
        requirement="input_seal",
        status="unverified",
        procedure_digest="proc",
        input_manifest_digest="manifest",
        code_digest="code",
        interval_start=ctx.interval_start - pd.Timedelta(days=1),
        interval_end=ctx.interval_end + pd.Timedelta(days=1),
        artifact_digest=None,
        reason_codes=("SEAL_MISSING",),
    )
    result = assess_process_validation(
        base, stress, context=ctx, checks=(check,), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.gate.go is False
    assert "REQUIREMENT_UNVERIFIED:input_seal" in result.reason_codes


def test_partial_coverage_leaves_inference_unverified() -> None:
    from src.mhs.backtest.certification import _select_formal_returns, inventory_daily_evidence

    base, stress = _growing_replays(4)
    ctx = _context(10, role="historical", spec=_spec())
    base_evidence = inventory_daily_evidence(base)
    stress_evidence = inventory_daily_evidence(stress)
    try:
        _select_formal_returns(base_evidence, stress_evidence, ctx)
    except DataIntegrityError:
        pass
    else:
        raise AssertionError("partial coverage must raise")
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx)[:6], envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.historical_acceptance == "unverified"
    assert "EVIDENCE_COVERAGE_INCOMPLETE" in result.reason_codes


def test_insufficient_paths_yield_tail_unresolved() -> None:
    base, stress = _growing_replays(60)
    thin = ValidationInferenceSpec(
        family_alpha=0.2,
        procedure_budget=1,
        look_budget=1,
        endpoint_budget=4,
        bootstrap_paths=10,
        seed=7,
        resample_batch_paths=10,
        minimum_block_days=1,
    )
    ctx = _context(60, role="historical", spec=thin)
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx)[:6], envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.historical_acceptance == "unverified"
    assert "INFERENCE_TAIL_UNRESOLVED" in result.reason_codes


def test_rejected_working_memory_leaves_inference_unverified() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="historical", spec=_spec())
    tiny = MhsMemoryBudget(total_tree_pss_bytes=1, replay_tree_pss_bytes=1, min_available_bytes=1)
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx)[:6], envelope=_envelope(), memory_budget=tiny
    )
    assert result.historical_acceptance == "unverified"
    assert "EVIDENCE_COVERAGE_INCOMPLETE" in result.reason_codes


def _falling_replays(days: int = 60) -> tuple[StrategyExecutionReplayResult, StrategyExecutionReplayResult]:
    grid = _grid(days)
    levels = pd.Series(1.0 - 0.00004 * np.arange(len(grid)), index=grid, dtype="float64")
    return _replay(grid, levels, valid=True), _replay(grid, levels * 0.999999, valid=True)


def test_negative_drift_fails_formal_statistics() -> None:
    base, stress = _falling_replays(60)
    ctx = _context(60, role="historical", spec=_spec())
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx)[:6], envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.historical_acceptance == "failed"
    assert result.gate.go is False


def test_forward_pass_opens_deployment() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="forward", spec=_spec())
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.historical_acceptance == "passed"
    assert result.forward_acceptance == "passed"
    assert result.gate.go is True


def test_forward_with_partial_checks_stays_unverified() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="forward", spec=_spec())
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx)[:6], envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.forward_acceptance == "unverified"
    assert result.gate.go is False


def test_forward_without_registration_stays_unverified() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="forward", spec=_spec())
    ctx = EvaluationContext(
        role="forward",
        procedure_digest=ctx.procedure_digest,
        code_digest=ctx.code_digest,
        input_manifest_digest=ctx.input_manifest_digest,
        interval_start=ctx.interval_start,
        interval_end=ctx.interval_end,
        registered_at=None,
        consulted_through=ctx.consulted_through,
        family_id=ctx.family_id,
        look_ordinal=ctx.look_ordinal,
        inference_spec=ctx.inference_spec,
        journal_complete=True,
        observed_through=ctx.observed_through,
    )
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.forward_acceptance != "passed"
    assert result.gate.go is False


def test_forward_without_journal_stays_unverified() -> None:
    base, stress = _growing_replays(60)
    ctx = _context(60, role="forward", spec=_spec())
    ctx = EvaluationContext(
        role="forward",
        procedure_digest=ctx.procedure_digest,
        code_digest=ctx.code_digest,
        input_manifest_digest=ctx.input_manifest_digest,
        interval_start=ctx.interval_start,
        interval_end=ctx.interval_end,
        registered_at=ctx.registered_at,
        consulted_through=ctx.consulted_through,
        family_id=ctx.family_id,
        look_ordinal=ctx.look_ordinal,
        inference_spec=ctx.inference_spec,
        journal_complete=False,
        observed_through=ctx.observed_through,
    )
    result = assess_process_validation(
        base, stress, context=ctx, checks=_passed_checks(ctx), envelope=_envelope(), memory_budget=MhsMemoryBudget()
    )
    assert result.forward_acceptance != "passed"
    assert result.gate.go is False
    assert "ACCESS_HISTORY_INCOMPLETE" in result.reason_codes
