"""Invariant guards for the durable consultation and attempt journal."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.certification import EvaluationContext, ValidationInferenceSpec
from src.mhs.backtest.journal import (
    ProcessEvaluationPlan,
    ProcessProcedureDefinition,
    consulted_process_horizon,
    context_from_json,
    finish_research_attempt,
    initialize_research_journal,
    load_process_evaluation_plan,
    load_research_attempt,
    persist_process_registration,
    plan_from_json,
    procedure_from_json,
    procedure_to_json,
    process_procedure_digest,
    record_research_consultation,
    reserve_research_attempt,
)
from src.mhs.backtest.journal import _endpoint_schedule
from src.mhs.backtest.labels import ProcessClockSpec
from src.mhs.backtest.selection import NestedSelectionSpec, TrainingWindowSpec
from src.mhs.params import MHS_FINAL_OOS_CUTOFF_2026H1, GrowthRiskEnvelope
from src.mhs.process import ProcessExecutionPolicy, ProcessRiskSizingSpec
from src.mhs.types import ExecutionSpec

INIT_NOW = pd.Timestamp("2026-08-01", tz="UTC")
LEGACY_OLD = pd.Timestamp("2024-01-01", tz="UTC")
REG_NOW = pd.Timestamp("2026-08-02", tz="UTC")
JUDGE = pd.Timestamp("2026-09-01", tz="UTC")
E1 = pd.Timestamp("2026-09-30", tz="UTC")
E2 = pd.Timestamp("2026-10-31", tz="UTC")

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


def _clock() -> ProcessClockSpec:
    return ProcessClockSpec(
        decision_period=pd.Timedelta(hours=1),
        bar_completion_lag=pd.Timedelta(hours=1),
        fit_latency=pd.Timedelta(0),
    )


def _selection() -> NestedSelectionSpec:
    return NestedSelectionSpec(
        policies=(
            TrainingWindowSpec(policy_id="expanding", kind="expanding", months=None),
            TrainingWindowSpec(policy_id="rolling_12m", kind="rolling", months=12),
            TrainingWindowSpec(policy_id="rolling_24m", kind="rolling", months=24),
            TrainingWindowSpec(policy_id="equal_member", kind="equal_member", months=None),
        ),
        control_policy_id="equal_member",
        minimum_inner_labels=12,
        alpha=0.05,
        bootstrap_paths=500,
        seed=11,
        fit_latency=pd.Timedelta(0),
    )


def _inference(*, look_budget: int = 2) -> ValidationInferenceSpec:
    return ValidationInferenceSpec(
        family_alpha=0.05,
        procedure_budget=2,
        look_budget=look_budget,
        endpoint_budget=4,
        bootstrap_paths=2000,
        seed=7,
        resample_batch_paths=500,
        minimum_block_days=5,
    )


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


def _procedure(**overrides: object) -> ProcessProcedureDefinition:
    fields: dict[str, object] = {
        "schema_version": 1,
        "code_digest": "ab" * 32,
        "data_policy": "policy-v1",
        "universe_partition": "all",
        "member_ids": ("AAA", "BBB"),
        "clock": _clock(),
        "selection": _selection(),
        "inference": _inference(),
        "execution_policy": ProcessExecutionPolicy(),
        "risk_sizing": ProcessRiskSizingSpec(
            annual_volatility_target=0.4,
            ewma_halflife_days=90,
            minimum_observations=45,
            leverage_cap=1.0,
        ),
        "sizing_source": "hourly_proxy",
        "member_evidence_source": "daily_step_proxy",
        "execution_spec": ExecutionSpec(),
        "envelope": _envelope(),
        "initial_equity": 1000.0,
        "smoothing_halflife_days": 8.0,
        "required_checks": NAMES,
    }
    fields.update(overrides)
    return ProcessProcedureDefinition(**fields)  # type: ignore[arg-type]


def _plan(
    procedure: ProcessProcedureDefinition | None = None,
    *,
    role: str = "forward",
    family_id: str = "fam-a",
    endpoints: tuple[pd.Timestamp, ...] = (E1, E2),
    judging_start: pd.Timestamp | None = JUDGE,
) -> ProcessEvaluationPlan:
    procedure = procedure if procedure is not None else _procedure()
    return ProcessEvaluationPlan(
        role=role,  # type: ignore[arg-type]
        procedure=procedure,
        procedure_digest=process_procedure_digest(procedure),
        family_id=family_id,
        look_endpoints=endpoints,
        registration_digest=None,
        judging_start=judging_start,
    )


def _journal(path: Path, *, complete: bool = False) -> Path:
    initialize_research_journal(
        path, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=complete
    )
    return path


def _registered(tmp_path: Path, *, role: str = "forward", look_budget: int = 2) -> tuple[Path, ProcessEvaluationPlan]:
    from src.mhs.preregistration import register_process_procedure

    journal = _journal(tmp_path / "j.db")
    procedure = _procedure(inference=_inference(look_budget=look_budget))
    plan = _plan(procedure, role=role)
    registered = register_process_procedure(
        plan,
        now=REG_NOW,
        journal_path=journal,
        legacy_history_dir=tmp_path / "hist",
        legacy_registry_path=tmp_path / "reg.jsonl",
    )
    return journal, registered


def test_reserve_before_read_marks_interval_consulted(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    attempt = reserve_research_attempt(
        journal,
        plan,
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=E1,
        now=REG_NOW + pd.Timedelta(hours=1),
        attempt_id="a1",
    )
    with pytest.raises(RuntimeError):
        raise RuntimeError("source loader failed")
    assert consulted_process_horizon(journal) >= E1
    assert attempt.pre_read_consulted_through < JUDGE
    assert attempt.context.interval_start == JUDGE
    assert attempt.context.look_ordinal == 1


def test_failed_attempt_retains_slot_and_horizon(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    first = reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    finish_research_attempt(journal, first, status="failed", result_digest=None, now=REG_NOW + pd.Timedelta(hours=1))
    assert consulted_process_horizon(journal) >= E1
    second = reserve_research_attempt(journal, plan, start=JUDGE, end=E2, now=REG_NOW, attempt_id="a2")
    assert second.context.look_ordinal == 2
    assert second.context.interval_start == E1.normalize() + pd.Timedelta(days=1)
    finish_research_attempt(journal, second, status="failed", result_digest=None, now=REG_NOW + pd.Timedelta(hours=2))
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, plan, start=JUDGE, end=E2, now=REG_NOW, attempt_id="a3")
    assert consulted_process_horizon(journal) >= E2


def test_concurrent_reservation_grants_single_look(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path, look_budget=1)
    barrier = threading.Barrier(2)
    outcomes: dict[str, str] = {}

    def _reserve(attempt_id: str) -> None:
        barrier.wait(timeout=10.0)
        try:
            reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id=attempt_id)
        except DataIntegrityError:
            outcomes[attempt_id] = "rejected"
        else:
            outcomes[attempt_id] = "reserved"

    threads = [threading.Thread(target=_reserve, args=(aid,)) for aid in ("a1", "a2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert sorted(outcomes.values()) == ["rejected", "reserved"]
    conn = sqlite3.connect(str(journal))
    try:
        count = conn.execute("SELECT COUNT(*) FROM attempts WHERE family_id='fam-a'").fetchone()[0]
    finally:
        conn.close()
    assert int(count) == 1


def test_family_budget_is_immutable_after_first_look(tmp_path: Path) -> None:
    from src.mhs.preregistration import register_process_procedure

    journal, plan = _registered(tmp_path)
    identical = register_process_procedure(
        _plan(),
        now=REG_NOW,
        journal_path=journal,
        legacy_history_dir=tmp_path / "hist",
        legacy_registry_path=tmp_path / "reg.jsonl",
    )
    assert identical.registration_digest == plan.registration_digest
    reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    enlarged = _plan(_procedure(inference=_inference(look_budget=3)))
    with pytest.raises(DataIntegrityError):
        register_process_procedure(
            enlarged,
            now=REG_NOW,
            journal_path=journal,
            legacy_history_dir=tmp_path / "hist",
            legacy_registry_path=tmp_path / "reg.jsonl",
        )
    sibling = _plan(_procedure(member_ids=("AAA", "CCC")))
    with pytest.raises(DataIntegrityError):
        register_process_procedure(
            sibling,
            now=REG_NOW,
            journal_path=journal,
            legacy_history_dir=tmp_path / "hist",
            legacy_registry_path=tmp_path / "reg.jsonl",
        )


def test_outcome_recording_is_idempotent_for_identical_digest(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    attempt = reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    finish_research_attempt(journal, attempt, status="completed", result_digest="d" * 64, now=REG_NOW)
    finish_research_attempt(journal, attempt, status="completed", result_digest="d" * 64, now=REG_NOW)
    with pytest.raises(DataIntegrityError):
        finish_research_attempt(journal, attempt, status="completed", result_digest="e" * 64, now=REG_NOW)
    loaded = load_research_attempt(journal, "a1")
    assert loaded == attempt


def test_pruned_legacy_history_stays_consulted(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "j.db")
    assert consulted_process_horizon(journal) >= MHS_FINAL_OOS_CUTOFF_2026H1
    assert consulted_process_horizon(journal) >= INIT_NOW
    historical = _plan(role="historical", judging_start=None)
    attempt = reserve_research_attempt(
        journal, historical, start=LEGACY_OLD, end=LEGACY_OLD + pd.Timedelta(days=30), now=INIT_NOW, attempt_id="h1"
    )
    assert attempt.context.journal_complete is False
    assert attempt.context.look_ordinal is None
    journal2 = tmp_path / "j2.db"
    initialize_research_journal(
        journal2, now=LEGACY_OLD, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=True
    )
    assert consulted_process_horizon(journal2) == LEGACY_OLD


def test_current_reservation_excluded_from_prior_consultation(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    before = consulted_process_horizon(journal)
    attempt = reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    assert attempt.pre_read_consulted_through == before
    assert attempt.context.consulted_through == before
    assert before < JUDGE
    assert attempt.context.observed_through == REG_NOW
    assert attempt.context.registered_at == REG_NOW


def test_cache_retrieval_is_consultation_not_evidence(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "j.db")
    record_research_consultation(
        journal,
        procedure_digest=None,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-06-01", tz="UTC"),
        now=INIT_NOW,
        source="cache",
        result_digest="d" * 64,
    )
    assert consulted_process_horizon(journal) >= pd.Timestamp("2025-06-01", tz="UTC")
    with pytest.raises(DataIntegrityError):
        load_research_attempt(journal, "never-reserved")


def test_corrupt_journal_fails_closed(tmp_path: Path) -> None:
    garbage = tmp_path / "bad.db"
    garbage.write_bytes(b"not a sqlite database")
    with pytest.raises(DataIntegrityError):
        initialize_research_journal(garbage, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=False)
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(garbage, _plan(), start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    with pytest.raises(DataIntegrityError):
        load_process_evaluation_plan(garbage)
    with pytest.raises(DataIntegrityError):
        consulted_process_horizon(tmp_path / "missing.db")
    journal = _journal(tmp_path / "j.db")
    conn = sqlite3.connect(str(journal))
    try:
        conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, _plan(), start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    with pytest.raises(DataIntegrityError):
        initialize_research_journal(journal, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=False)


def test_procedure_identity_covers_all_economic_controls(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    baseline = process_procedure_digest(_procedure())
    variants = [
        _procedure(smoothing_halflife_days=9.0),
        _procedure(execution_spec=dataclasses.replace(ExecutionSpec(), taker_fee_bps=9.0)),
        _procedure(member_ids=("BBB", "AAA")),
        _procedure(required_checks=NAMES[:10]),
        _procedure(initial_equity=2000.0),
    ]
    digests = {process_procedure_digest(variant) for variant in variants}
    assert len(digests) == len(variants)
    assert baseline not in digests
    tampered = dataclasses.replace(plan, procedure_digest="00" * 32)
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, tampered, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    encoded = procedure_to_json(_procedure())
    encoded["unknown_field"] = "x"
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    conn = sqlite3.connect(str(journal))
    try:
        row = conn.execute("SELECT canonical FROM plans ORDER BY rowid DESC LIMIT 1").fetchone()
        tampered_plan = json.loads(str(row[0]))
        tampered_plan["look_endpoints"] = "not-a-list"
        conn.execute("UPDATE plans SET canonical=?", (json.dumps(tampered_plan),))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        load_process_evaluation_plan(journal)


def test_reservation_validates_request_identities(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, plan, start=E1, end=E1, now=REG_NOW, attempt_id="a1")
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="")
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, plan, start=pd.Timestamp("2026-09-01"), end=E1, now=REG_NOW, attempt_id="a1")
    reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    fresh = _plan(family_id="fam-b")
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, fresh, start=JUDGE, end=E1, now=REG_NOW, attempt_id="b1")
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, plan, start=JUDGE, end=E2 + pd.Timedelta(days=1), now=REG_NOW, attempt_id="a2")
    record_research_consultation(
        journal, procedure_digest=None, start=JUDGE, end=E2, now=REG_NOW, source="external"
    )
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="late")


def test_historical_plan_with_judging_start_binds_assessment(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "j.db")
    historical = _plan(role="historical", judging_start=pd.Timestamp("2026-02-01", tz="UTC"))
    attempt = reserve_research_attempt(
        journal, historical, start=LEGACY_OLD, end=pd.Timestamp("2026-06-01", tz="UTC"), now=INIT_NOW, attempt_id="h1"
    )
    assert attempt.context.interval_start == pd.Timestamp("2026-02-01", tz="UTC")
    late = _plan(role="historical", judging_start=pd.Timestamp("2026-06-01", tz="UTC"))
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(
            journal, late, start=LEGACY_OLD, end=pd.Timestamp("2026-06-01", tz="UTC"), now=INIT_NOW, attempt_id="h2"
        )


def test_finish_attempt_validates_identity_and_status(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    attempt = reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    with pytest.raises(DataIntegrityError):
        finish_research_attempt(journal, attempt, status="done", result_digest="d" * 64, now=REG_NOW)  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        finish_research_attempt(journal, attempt, status="failed", result_digest=None, now=pd.Timestamp("2026-08-02"))
    with pytest.raises(DataIntegrityError):
        finish_research_attempt(journal, attempt, status="failed", result_digest="", now=REG_NOW)
    with pytest.raises(DataIntegrityError):
        finish_research_attempt(journal, attempt, status="completed", result_digest=None, now=REG_NOW)
    forged = dataclasses.replace(attempt, requested_end=E2)
    with pytest.raises(DataIntegrityError):
        finish_research_attempt(journal, forged, status="failed", result_digest=None, now=REG_NOW)
    ghost = dataclasses.replace(attempt, attempt_id="ghost")
    with pytest.raises(DataIntegrityError):
        finish_research_attempt(journal, ghost, status="failed", result_digest=None, now=REG_NOW)
    finish_research_attempt(journal, attempt, status="interrupted", result_digest=None, now=REG_NOW)
    second = reserve_research_attempt(journal, plan, start=JUDGE, end=E2, now=REG_NOW, attempt_id="a2")
    finish_research_attempt(journal, second, status="completed", result_digest="d" * 64, now=REG_NOW)


def test_malformed_encodings_are_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        procedure_from_json({"__type__": "ProcessProcedureDefinition"})
    encoded = procedure_to_json(_procedure())
    encoded["schema_version"] = "1"
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["initial_equity"] = float("inf")
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["member_ids"] = "AAA"
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["clock"] = {**encoded["clock"], "decision_period": 3600}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["selection"] = {**encoded["selection"], "policies": "nope"}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["inference"] = {**encoded["inference"], "family_alpha": "high"}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["inference"] = {**encoded["inference"], "bootstrap_paths": "many"}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["execution_policy"] = {"__type__": "ProcessExecutionPolicy", "tracking_error_threshold": "x"}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["risk_sizing"] = {**encoded["risk_sizing"], "annual_volatility_target": -1.0}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["execution_spec"] = {**encoded["execution_spec"], "maker_fee_bps": -5.0}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["execution_spec"] = {**encoded["execution_spec"], "require_trade_through": "yes"}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded = procedure_to_json(_procedure())
    encoded["envelope"] = {**encoded["envelope"], "max_drawdown": -1.0}
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    sized = procedure_from_json(procedure_to_json(_procedure(risk_sizing=None)))
    assert sized.risk_sizing is None
    assert process_procedure_digest(sized) == process_procedure_digest(_procedure(risk_sizing=None))
    encoded = procedure_to_json(_procedure())
    encoded["selection"]["policies"] = "nope"
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    encoded["selection"] = procedure_to_json(_procedure())["selection"]
    encoded["selection"]["policies"][1]["months"] = None
    with pytest.raises(DataIntegrityError):
        procedure_from_json(encoded)
    from src.mhs.backtest.journal import plan_to_json

    plan_encoded = plan_to_json(_plan())
    plan_encoded["look_endpoints"] = "yesterday"
    with pytest.raises(DataIntegrityError):
        plan_from_json(plan_encoded)
    plan_encoded = plan_to_json(_plan())
    plan_encoded["role"] = 5
    with pytest.raises(DataIntegrityError):
        plan_from_json(plan_encoded)
    with pytest.raises(DataIntegrityError):
        _endpoint_schedule("{corrupt")
    with pytest.raises(DataIntegrityError):
        _endpoint_schedule("5")


def test_invalid_constructions_are_rejected() -> None:
    with pytest.raises(DataIntegrityError):
        _procedure(schema_version=2)
    with pytest.raises(DataIntegrityError):
        _procedure(code_digest="")
    with pytest.raises(DataIntegrityError):
        _procedure(data_policy="")
    with pytest.raises(DataIntegrityError):
        _procedure(universe_partition="paper")
    with pytest.raises(DataIntegrityError):
        _procedure(member_ids=())
    with pytest.raises(DataIntegrityError):
        _procedure(clock="clock")
    with pytest.raises(DataIntegrityError):
        _procedure(selection="selection")
    with pytest.raises(DataIntegrityError):
        _procedure(inference="inference")
    with pytest.raises(DataIntegrityError):
        _procedure(execution_policy="policy")
    with pytest.raises(DataIntegrityError):
        _procedure(risk_sizing="sizing")
    with pytest.raises(DataIntegrityError):
        _procedure(sizing_source="inventory")
    with pytest.raises(DataIntegrityError):
        _procedure(member_evidence_source="tape")
    with pytest.raises(DataIntegrityError):
        _procedure(execution_spec="spec")
    with pytest.raises(DataIntegrityError):
        _procedure(envelope="envelope")
    with pytest.raises(DataIntegrityError):
        _procedure(initial_equity=0.0)
    with pytest.raises(DataIntegrityError):
        _procedure(initial_equity=True)
    with pytest.raises(DataIntegrityError):
        _procedure(initial_equity=float("nan"))
    with pytest.raises(DataIntegrityError):
        _procedure(smoothing_halflife_days=-1.0)
    with pytest.raises(DataIntegrityError):
        _procedure(smoothing_halflife_days="week")
    with pytest.raises(DataIntegrityError):
        _procedure(required_checks=())
    with pytest.raises(DataIntegrityError):
        _plan(role="paper")
    with pytest.raises(DataIntegrityError):
        _plan(family_id="")
    with pytest.raises(DataIntegrityError):
        dataclasses.replace(_plan(), procedure_digest="short")
    with pytest.raises(DataIntegrityError):
        dataclasses.replace(_plan(), registration_digest="short")
    with pytest.raises(DataIntegrityError):
        _plan(endpoints=())
    with pytest.raises(DataIntegrityError):
        _plan(endpoints=(E2, E1))
    with pytest.raises(DataIntegrityError):
        _plan(endpoints=(pd.Timestamp("2026-09-30"), E2))
    with pytest.raises(DataIntegrityError):
        _plan(judging_start=pd.Timestamp("2026-01-01"))
    with pytest.raises(DataIntegrityError):
        _plan(judging_start=E2)
    with pytest.raises(DataIntegrityError):
        _plan(role="forward", judging_start=None)
    with pytest.raises(DataIntegrityError):
        ProcessEvaluationPlan(
            role="forward",
            procedure="procedure",  # type: ignore[arg-type]
            procedure_digest="x",
            family_id="fam",
            look_endpoints=(E1,),
            registration_digest=None,
            judging_start=JUDGE,
        )


def test_attempt_construction_validates_context_and_clocks() -> None:
    from src.mhs.backtest.journal import ResearchAttempt

    context = EvaluationContext(
        role="historical",
        procedure_digest="p",
        code_digest="c",
        input_manifest_digest=None,
        interval_start=JUDGE,
        interval_end=E1,
        registered_at=None,
        consulted_through=INIT_NOW,
        family_id="fam",
        look_ordinal=None,
        inference_spec=None,
        journal_complete=False,
        observed_through=INIT_NOW,
    )
    with pytest.raises(DataIntegrityError):
        ResearchAttempt(
            attempt_id="",
            context=context,
            reserved_at=INIT_NOW,
            pre_read_consulted_through=INIT_NOW,
            requested_start=JUDGE,
            requested_end=E1,
        )
    with pytest.raises(DataIntegrityError):
        ResearchAttempt(
            attempt_id="a1",
            context="context",  # type: ignore[arg-type]
            reserved_at=INIT_NOW,
            pre_read_consulted_through=INIT_NOW,
            requested_start=JUDGE,
            requested_end=E1,
        )
    with pytest.raises(DataIntegrityError):
        ResearchAttempt(
            attempt_id="a1",
            context=context,
            reserved_at=pd.Timestamp("2026-08-01"),
            pre_read_consulted_through=INIT_NOW,
            requested_start=JUDGE,
            requested_end=E1,
        )
    with pytest.raises(DataIntegrityError):
        ResearchAttempt(
            attempt_id="a1",
            context=context,
            reserved_at=INIT_NOW,
            pre_read_consulted_through=INIT_NOW,
            requested_start=pd.Timestamp("2026-09-01"),
            requested_end=E1,
        )
    with pytest.raises(DataIntegrityError):
        ResearchAttempt(
            attempt_id="a1",
            context=context,
            reserved_at=INIT_NOW,
            pre_read_consulted_through=INIT_NOW,
            requested_start=E1,
            requested_end=E1,
        )


def test_initialize_and_record_validate_inputs(tmp_path: Path) -> None:
    journal = tmp_path / "j.db"
    with pytest.raises(DataIntegrityError):
        initialize_research_journal(
            journal, now=pd.Timestamp("2026-08-01"), legacy_consulted_through=LEGACY_OLD, legacy_history_complete=False
        )
    with pytest.raises(DataIntegrityError):
        initialize_research_journal(
            journal, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete="yes"  # type: ignore[arg-type]
        )
    _journal(journal)
    initialize_research_journal(journal, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=False)
    initialize_research_journal(
        journal, now=INIT_NOW, legacy_consulted_through=pd.Timestamp("2026-08-05", tz="UTC"), legacy_history_complete=False
    )
    assert consulted_process_horizon(journal) == pd.Timestamp("2026-08-05", tz="UTC")
    with pytest.raises(DataIntegrityError):
        record_research_consultation(
            journal, procedure_digest=None, start=E1, end=E1, now=INIT_NOW, source="cache"
        )
    with pytest.raises(DataIntegrityError):
        record_research_consultation(
            journal, procedure_digest=None, start=JUDGE, end=E1, now=INIT_NOW, source="tape"  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        record_research_consultation(
            journal, procedure_digest="", start=JUDGE, end=E1, now=INIT_NOW, source="cache"
        )
    with pytest.raises(DataIntegrityError):
        record_research_consultation(
            journal, procedure_digest=None, start=JUDGE, end=E1, now=INIT_NOW, source="cache", result_digest=""
        )
    with pytest.raises(DataIntegrityError):
        record_research_consultation(
            tmp_path / "missing.db", procedure_digest=None, start=JUDGE, end=E1, now=INIT_NOW, source="cache"
        )
    with pytest.raises(DataIntegrityError):
        load_research_attempt(journal, "")
    conn = sqlite3.connect(str(journal))
    try:
        conn.execute("UPDATE meta SET value='garbage' WHERE key='consulted_through'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        consulted_process_horizon(journal)
    conn = sqlite3.connect(str(journal))
    try:
        conn.execute("UPDATE meta SET value='2024-01-01T00:00:00' WHERE key='consulted_through'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        consulted_process_horizon(journal)
    with pytest.raises(DataIntegrityError):
        initialize_research_journal(journal, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=False)
    conn = sqlite3.connect(str(journal))
    try:
        conn.execute("DELETE FROM meta WHERE key='consulted_through'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        consulted_process_horizon(journal)
    with pytest.raises(DataIntegrityError):
        initialize_research_journal(journal, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=False)


def test_load_plan_requires_registered_identity(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "j.db")
    with pytest.raises(DataIntegrityError):
        load_process_evaluation_plan(journal)
    _, plan = _registered(tmp_path)
    loaded = load_process_evaluation_plan(tmp_path / "j.db")
    assert loaded == plan
    conn = sqlite3.connect(str(tmp_path / "j.db"))
    try:
        conn.execute("UPDATE plans SET procedure_digest=?", ("ff" * 32,))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        load_process_evaluation_plan(tmp_path / "j.db")


def test_context_round_trip_preserves_meaning(tmp_path: Path) -> None:
    from src.mhs.backtest.journal import context_to_json

    journal, plan = _registered(tmp_path)
    attempt = reserve_research_attempt(journal, plan, start=JUDGE, end=E1, now=REG_NOW, attempt_id="a1")
    assert context_from_json(context_to_json(attempt.context)) == attempt.context
    conn = sqlite3.connect(str(journal))
    try:
        original = json.loads(conn.execute("SELECT context FROM attempts WHERE attempt_id='a1'").fetchone()[0])

        def _store(payload: object) -> None:
            conn.execute("UPDATE attempts SET context=? WHERE attempt_id='a1'", (json.dumps(payload),))
            conn.commit()

        tampered = dict(original)
        tampered["role"] = "bogus"
        _store(tampered)
        with pytest.raises(DataIntegrityError):
            load_research_attempt(journal, "a1")
        tampered = dict(original)
        tampered["interval_start"] = "2024-01-01"
        _store(tampered)
        with pytest.raises(DataIntegrityError):
            load_research_attempt(journal, "a1")
        tampered = dict(original)
        tampered["interval_start"] = {"__timestamp__": "garbage"}
        _store(tampered)
        with pytest.raises(DataIntegrityError):
            load_research_attempt(journal, "a1")
        tampered = dict(original)
        tampered["interval_start"] = {"__timestamp__": "2024-01-01T00:00:00"}
        _store(tampered)
        with pytest.raises(DataIntegrityError):
            load_research_attempt(journal, "a1")
    finally:
        conn.close()


def test_canonical_values_reject_non_json_meanings() -> None:
    from src.mhs.backtest.journal import _canonical

    with pytest.raises(DataIntegrityError):
        _canonical(float("inf"))
    with pytest.raises(DataIntegrityError):
        _canonical(pd.Timestamp("2024-01-01"))
    with pytest.raises(DataIntegrityError):
        _canonical(object())


def test_registration_rejects_stale_or_fresh_identities(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "j.db")
    registered = persist_process_registration(journal, _plan(), now=REG_NOW)
    assert registered.registration_digest is not None
    with pytest.raises(DataIntegrityError):
        persist_process_registration(journal, registered, now=REG_NOW)
    with pytest.raises(DataIntegrityError):
        persist_process_registration(journal, _plan(endpoints=(E1,)), now=REG_NOW)
    with pytest.raises(DataIntegrityError):
        persist_process_registration(
            journal, dataclasses.replace(_plan(), procedure_digest="00" * 32), now=REG_NOW
        )
    with pytest.raises(DataIntegrityError):
        persist_process_registration(journal, _plan(), now=pd.Timestamp("2026-08-02"))
    with pytest.raises(DataIntegrityError):
        persist_process_registration(tmp_path / "missing.db", _plan(), now=REG_NOW)


def test_forward_family_mismatch_and_short_schedule_rejected(tmp_path: Path) -> None:
    journal, plan = _registered(tmp_path)
    mismatched = _plan(endpoints=(E1,))
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, mismatched, start=JUDGE, end=E1, now=REG_NOW, attempt_id="m1")
    single = _plan(family_id="fam-single", endpoints=(E2,))
    registered_single = persist_process_registration(journal, single, now=REG_NOW)
    first = reserve_research_attempt(journal, registered_single, start=JUDGE, end=E2, now=REG_NOW, attempt_id="s1")
    assert first.context.look_ordinal == 1
    with pytest.raises(DataIntegrityError):
        reserve_research_attempt(journal, registered_single, start=JUDGE, end=E2, now=REG_NOW, attempt_id="s2")
