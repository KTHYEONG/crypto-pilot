"""Invariant guards for continuous process orchestration and evidence producers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError


def _procedure(member_ids: tuple[str, ...] = ("AAA", "BBB")):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.certification import REQUIRED_CHECK_NAMES, ValidationInferenceSpec
    from src.mhs.backtest.journal import PROCEDURE_SCHEMA_VERSION, ProcessProcedureDefinition
    from src.mhs.backtest.labels import ProcessClockSpec
    from src.mhs.backtest.selection import NestedSelectionSpec, TrainingWindowSpec
    from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
    from src.mhs.params import (
        CLI_GROWTH_ENVELOPE_DEFAULT,
        EVIDENCE_GATE_ALPHA,
        GROWTH_RISK_ENVELOPES,
        NULL_BOOTSTRAP_SEED,
        PROCESS_MIN_TRAIN_DAYS,
        PROCESS_SMOOTHING_HALFLIFE_DAYS,
    )
    from src.mhs.process import ProcessExecutionPolicy
    from src.mhs.types import ExecutionSpec

    return ProcessProcedureDefinition(
        schema_version=PROCEDURE_SCHEMA_VERSION,
        code_digest="ab" * 32,
        data_policy=str(MHS_DATA_POLICY_DEFAULT),
        universe_partition="dev",
        member_ids=tuple(member_ids),
        clock=ProcessClockSpec(
            decision_period=pd.Timedelta(hours=24),
            bar_completion_lag=pd.Timedelta(hours=1),
            fit_latency=pd.Timedelta(0),
        ),
        selection=NestedSelectionSpec(
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
        ),
        inference=ValidationInferenceSpec(
            family_alpha=float(EVIDENCE_GATE_ALPHA),
            procedure_budget=4,
            look_budget=1,
            endpoint_budget=4,
            bootstrap_paths=6400,
            seed=int(NULL_BOOTSTRAP_SEED),
            resample_batch_paths=500,
            minimum_block_days=1,
        ),
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


def _context_for(procedure):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.journal import process_procedure_digest

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-03-01", tz="UTC")
    return __import__("src.mhs.backtest.certification", fromlist=["EvaluationContext"]).EvaluationContext(
        role="historical",
        procedure_digest=process_procedure_digest(procedure),
        code_digest=procedure.code_digest,
        input_manifest_digest=None,
        interval_start=start,
        interval_end=end,
        registered_at=None,
        consulted_through=None,
        family_id=None,
        look_ordinal=None,
        inference_spec=procedure.inference,
        journal_complete=False,
        observed_through=end,
    )


def _replay_pair():  # type: ignore[no-untyped-def]
    import numpy as np
    import pandas as pd

    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    grid = pd.date_range("2024-01-01", periods=96, freq="30min", tz="UTC")
    levels = pd.Series(1.0 + 0.00004 * np.arange(len(grid)), index=grid, dtype="float64")
    zeros = pd.Series([0.0] * len(grid), index=grid, dtype="float64")
    ledger = SimulatedInventoryLedgerResult(
        equity=levels,
        net_returns=levels.pct_change().dropna(),
        simulated_units=None,
        mark_to_market_pnl=zeros,
        funding_charge=zeros,
        fee_charge=zeros,
        fill_turnover=zeros,
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="OHLCV_CLOSE_FALLBACK",
        primary_valid=True,
        invalid_reasons=(),
        data_gaps=(),
    )
    avail = pd.DatetimeIndex(grid + pd.Timedelta(minutes=30), tz="UTC")

    def _one(eq):  # type: ignore[no-untyped-def]
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

    return _one(levels), _one(levels)


def _proxy():  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessBacktestReport
    from src.mhs.deploy_gate import DeployGateResult

    return ProcessBacktestReport(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-03-01", tz="UTC"),
        certification_level="process_proxy_1h_ledger",
        n_candidates=2,
        base=None,  # type: ignore[arg-type]
        stress=None,  # type: ignore[arg-type]
        gate=DeployGateResult(go=False, reason_codes=("PROXY_NEVER_DEPLOYS",), metrics={}),
    )


def _budget():  # type: ignore[no-untyped-def]
    from src.mhs.resources import resolve_mhs_memory_budget

    return resolve_mhs_memory_budget(None)


def test_held_symbol_window_carries_funding_knowledge_source(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    import pandas as pd

    import src.mhs.execution.window_stream as ws

    grid = pd.date_range("2024-01-01", periods=8, freq="3min", tz="UTC")
    weights = pd.DataFrame({"AAA": [1.0, 0.0], "BBB": [0.0, 0.0]}, index=pd.DatetimeIndex([grid[0], grid[2]], tz="UTC"))
    signals = pd.DatetimeIndex([grid[0] + pd.Timedelta(minutes=3), grid[2] + pd.Timedelta(minutes=3)], tz="UTC")
    funding = {
        "AAA": pd.Series([0.0001, 0.0002], index=pd.DatetimeIndex([grid[0], grid[4]], tz="UTC")),
        "BBB": pd.Series([0.0], index=pd.DatetimeIndex([grid[0]], tz="UTC")),
    }
    frames = {
        "AAA": pd.DataFrame({"quote_vol": np.full(len(grid), 100.0)}, index=grid),
        "BBB": pd.DataFrame({"quote_vol": np.full(len(grid), 100.0)}, index=grid),
    }
    monkeypatch.setattr(ws, "_load_window_minute_frames", lambda *a, **k: frames)
    monkeypatch.setattr(
        ws,
        "_build_window_frames",
        lambda *a, **k: (
            pd.DataFrame(1.0, index=grid, columns=["AAA", "BBB"]),
            pd.DataFrame(1.0, index=grid, columns=["AAA", "BBB"]),
            pd.DataFrame(1.0, index=grid, columns=["AAA", "BBB"]),
        ),
    )
    monkeypatch.setattr(ws, "_cached_mark_panel", lambda *a, **k: pd.DataFrame(1.0, index=grid, columns=["AAA", "BBB"]))
    import src.mhs.evaluation.integrity as integ

    monkeypatch.setattr(integ, "_assert_cache_required_marks", lambda *a, **k: None)
    allocation = ws._estimate_mhs_execution_allocation(n_symbols=2, n_columns=2, bound_count=2)
    window = ws._materialize_execution_piece(
        piece_grid=grid,
        piece_weights=weights,
        piece_signals=signals,
        roster=["AAA", "BBB"],
        columns=("AAA", "BBB"),
        root="ohlcv-root",
        timeframe="3m",
        funding_by_symbol=funding,
        mark_mode="cache_required",
        funding_failures={},
        allocation=allocation,
        budget_bytes=None,
        reserve_bytes=None,
        window_start=grid[0],
        window_end=grid[-1] + pd.Timedelta(minutes=3),
        logical_partition=(0, 2),
    )
    assert window.funding_knowledge_source == "archive_recency_proxy"
    assert window.funding_known is not None
    assert window.bar_funding is not None
    assert list(window.funding_known.columns) == ["AAA", "BBB"]


def test_manifest_mutation_fails_provenance() -> None:
    import dataclasses

    from src.mhs.backtest.inventory import build_process_evidence_checks

    procedure = _procedure()
    context = _context_for(procedure)
    bad = dataclasses.replace(context, procedure_digest="00" * 32)
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(procedure, bad, _proxy(), *_replay_pair(), memory_budget=_budget())
    bad_code = dataclasses.replace(context, code_digest="ff" * 32)
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(procedure, bad_code, _proxy(), *_replay_pair(), memory_budget=_budget())


def test_mixed_source_roots_stay_unverified() -> None:
    from src.mhs.backtest.inventory import build_process_evidence_checks

    procedure = _procedure()
    context = _context_for(procedure)
    checks = build_process_evidence_checks(procedure, context, _proxy(), *_replay_pair(), memory_budget=_budget())
    assert len(checks) == len(procedure.required_checks)
    assert {c.requirement for c in checks} == set(procedure.required_checks)
    assert all(c.status == "unverified" for c in checks)
    assert all(c.procedure_digest == context.procedure_digest for c in checks)


def test_unsupported_survival_claims_stay_unverified() -> None:
    from src.mhs.backtest.certification import assess_process_validation

    procedure = _procedure()
    context = _context_for(procedure)
    from src.mhs.backtest.inventory import build_process_evidence_checks

    base, stress = _replay_pair()
    checks = build_process_evidence_checks(procedure, context, _proxy(), base, stress, memory_budget=_budget())
    validation = assess_process_validation(
        base, stress, context=context, checks=checks, envelope=procedure.envelope, memory_budget=_budget()
    )
    by_name = {c.requirement: c for c in validation.requirements}
    assert by_name["margin_survival"].status == "unverified"
    assert by_name["capital_capacity"].status == "unverified"
    assert by_name["joint_stress"].status == "unverified"
    assert validation.gate.go is False


def test_explicit_submission_chronology_uses_path_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    import src.mhs.backtest.inventory as inv
    from src.mhs.backtest.contracts import ProcessBacktestReport
    from src.mhs.deploy_gate import DeployGateResult
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    captured: dict[str, object] = {}
    targets = pd.DataFrame(
        {"AAA": [1.0, 0.5]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
    )
    signals = pd.DatetimeIndex(targets.index + pd.Timedelta(hours=2))

    class _Path:
        target_weights = targets
        signal_available_at = signals
        execution_policy = __import__("src.mhs.process", fromlist=["ProcessExecutionPolicy"]).ProcessExecutionPolicy()

    fake_proxy = ProcessBacktestReport(
        start=DISCOVERY_START,
        end=PROCESS_EVALUATION_CEILING,
        certification_level="process_proxy_1h_ledger",
        n_candidates=1,
        base=_Path(),  # type: ignore[arg-type]
        stress=_Path(),  # type: ignore[arg-type]
        gate=DeployGateResult(go=False, reason_codes=(), metrics={}),
    )
    base, stress = _replay_pair()

    def _fake_eval(*a, **k):  # type: ignore[no-untyped-def]
        captured.update(k)
        return fake_proxy

    def _fake_stream(path, signal_available_at, *a, **k):  # type: ignore[no-untyped-def]
        captured["signal_available_at"] = signal_available_at
        captured["path"] = path
        return iter(())

    monkeypatch.setattr(inv, "evaluate_process_backtest", _fake_eval)
    monkeypatch.setattr(inv, "_inventory_window_stream", _fake_stream)
    monkeypatch.setattr(inv, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    monkeypatch.setattr(inv, "_assert_stage_rss_budget", lambda *a, **k: None)
    procedure = _procedure()
    report = inv.evaluate_process_inventory_backtest(
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        procedure=procedure,
    )
    assert list(captured["signal_available_at"]) == list(signals)
    assert captured.get("procedure") is procedure
    assert report.gate.go is False
    assert report.validation is not None


def test_journal_free_direct_evaluation_cannot_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.inventory as inv
    from src.mhs.backtest.contracts import ProcessBacktestReport
    from src.mhs.deploy_gate import DeployGateResult
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    targets = __import__("pandas").DataFrame(
        {"AAA": [1.0]},
        index=__import__("pandas").DatetimeIndex(["2024-01-01"], tz="UTC"),
    )

    class _Path:
        target_weights = targets
        signal_available_at = None
        execution_policy = __import__("src.mhs.process", fromlist=["ProcessExecutionPolicy"]).ProcessExecutionPolicy()

    fake_proxy = ProcessBacktestReport(
        start=DISCOVERY_START,
        end=PROCESS_EVALUATION_CEILING,
        certification_level="process_proxy_1h_ledger",
        n_candidates=1,
        base=_Path(),  # type: ignore[arg-type]
        stress=_Path(),  # type: ignore[arg-type]
        gate=DeployGateResult(go=False, reason_codes=(), metrics={}),
    )
    base, stress = _replay_pair()
    monkeypatch.setattr(inv, "evaluate_process_backtest", lambda *a, **k: fake_proxy)
    monkeypatch.setattr(
        inv,
        "_inventory_window_stream",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unused")) if False else iter(()),
    )
    monkeypatch.setattr(inv, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    monkeypatch.setattr(inv, "_assert_stage_rss_budget", lambda *a, **k: None)

    def _fake_stream(path, signal_available_at, *a, **k):  # type: ignore[no-untyped-def]
        assert signal_available_at is not None
        return iter(())

    monkeypatch.setattr(inv, "_inventory_window_stream", _fake_stream)
    procedure = _procedure()
    report = inv.evaluate_process_inventory_backtest(
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        procedure=procedure,
    )
    assert report.gate.go is False
    assert report.validation is not None
    assert report.validation.forward_acceptance == "unverified"


def test_future_source_perturbation_keeps_procedure_identity() -> None:
    from src.mhs.backtest.journal import process_procedure_digest
    from src.mhs.backtest.paths import baseline_process_procedure

    first = baseline_process_procedure(code_digest="ab" * 32)
    second = baseline_process_procedure(code_digest="ab" * 32)
    assert process_procedure_digest(first) == process_procedure_digest(second)
    assert first.clock.fit_latency == pd.Timedelta(0)
    assert first.inference.minimum_block_days == 1


def test_evidence_producer_input_conflicts_rejected() -> None:
    import dataclasses

    from src.mhs.backtest.inventory import build_process_evidence_checks

    procedure = _procedure()
    context = _context_for(procedure)
    base, stress = _replay_pair()
    proxy = _proxy()
    budget = _budget()
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks("bad", context, proxy, base, stress, memory_budget=budget)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(procedure, "bad", proxy, base, stress, memory_budget=budget)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(procedure, context, "bad", base, stress, memory_budget=budget)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(procedure, context, proxy, "bad", stress, memory_budget=budget)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(procedure, context, proxy, base, "bad", memory_budget=budget)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(procedure, context, proxy, base, stress, memory_budget="bad")  # type: ignore[arg-type]
    dup = dataclasses.replace(procedure, required_checks=("input_seal", "input_seal"))
    dup_context = _context_for(dup)
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(dup, dup_context, proxy, base, stress, memory_budget=budget)
    unknown = dataclasses.replace(procedure, required_checks=("not_a_check",))
    unknown_context = _context_for(unknown)
    with pytest.raises(DataIntegrityError):
        build_process_evidence_checks(unknown, unknown_context, proxy, base, stress, memory_budget=budget)


def test_inventory_entry_validates_typed_overrides() -> None:
    import src.mhs.backtest.inventory as inv
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    with pytest.raises(DataIntegrityError):
        inv.evaluate_process_inventory_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        inv.evaluate_process_inventory_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            evaluation_context="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        inv.evaluate_process_inventory_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            member_evidence="bad",  # type: ignore[arg-type]
        )


def test_reserved_context_is_used_for_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.inventory as inv
    from src.mhs.backtest.contracts import ProcessBacktestReport
    from src.mhs.deploy_gate import DeployGateResult
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    targets = __import__("pandas").DataFrame(
        {"AAA": [1.0]},
        index=__import__("pandas").DatetimeIndex(["2024-01-01"], tz="UTC"),
    )

    class _Path:
        target_weights = targets
        signal_available_at = None
        execution_policy = __import__("src.mhs.process", fromlist=["ProcessExecutionPolicy"]).ProcessExecutionPolicy()

    fake_proxy = ProcessBacktestReport(
        start=DISCOVERY_START,
        end=PROCESS_EVALUATION_CEILING,
        certification_level="process_proxy_1h_ledger",
        n_candidates=1,
        base=_Path(),  # type: ignore[arg-type]
        stress=_Path(),  # type: ignore[arg-type]
        gate=DeployGateResult(go=False, reason_codes=(), metrics={}),
    )
    base, stress = _replay_pair()
    monkeypatch.setattr(inv, "evaluate_process_backtest", lambda *a, **k: fake_proxy)
    monkeypatch.setattr(inv, "_inventory_window_stream", lambda *a, **k: iter(()))
    monkeypatch.setattr(inv, "replay_execution_window_batch", lambda *a, **k: (base, stress))
    monkeypatch.setattr(inv, "_assert_stage_rss_budget", lambda *a, **k: None)
    procedure = _procedure()
    supplied = _context_for(procedure)
    seen: dict[str, object] = {}
    real_assess = inv.assess_process_validation

    def _spy(b, s, *, context, checks, envelope, memory_budget):  # type: ignore[no-untyped-def]
        seen["context"] = context
        seen["checks"] = checks
        return real_assess(b, s, context=context, checks=checks, envelope=envelope, memory_budget=memory_budget)

    monkeypatch.setattr(inv, "assess_process_validation", _spy)
    report = inv.evaluate_process_inventory_backtest(
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        procedure=procedure,
        evaluation_context=supplied,
    )
    assert seen["context"] is supplied
    assert len(seen["checks"]) == len(procedure.required_checks)
    assert report.gate.go is False
