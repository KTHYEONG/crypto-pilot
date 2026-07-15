from __future__ import annotations

from src.domain.futures.strategy.tiered_workflow.outcomes import (
    TieredRunFailure,
    TieredRunOutcome,
)


def test_tiered_run_outcome_constructed() -> None:
    outcome = TieredRunOutcome(
        status="completed",
        l1_result=None,
        l2_result=None,
        l3_result=None,
        per_tf_l1=(),
        failure=None,
        policy_fingerprint="fp",
        diagnostic_complete=True,
    )
    assert outcome.status == "completed"
    assert outcome.failure is None
    assert outcome.diagnostic_complete


def test_tiered_run_failed_outcome() -> None:
    failure = TieredRunFailure(
        code="native_event_contract",
        timeframe="4h",
        message="timeframe=4h event_id=7 entry_idx mismatch",
    )
    outcome = TieredRunOutcome(
        status="failed",
        l1_result=None,
        l2_result=None,
        l3_result=None,
        per_tf_l1=(),
        failure=failure,
        policy_fingerprint="fp",
        diagnostic_complete=False,
    )
    assert outcome.status == "failed"
    assert outcome.failure is not None
    assert outcome.failure.code == "native_event_contract"
    assert "event_id=7" in outcome.failure.message
    assert not outcome.diagnostic_complete
