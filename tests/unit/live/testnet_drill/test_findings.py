"""Hypothesis pipeline invariants."""

from __future__ import annotations

from src.live.testnet_drill.findings import Finding, HypothesisStatus, derive_findings
from src.live.testnet_drill.report import ScenarioResult, Verdict


def test_one_finding_per_hypothesis() -> None:
    findings = derive_findings(())
    assert [f.hypothesis_id for f in findings] == [f"H{i}" for i in range(1, 12)]
    assert all(f.status is HypothesisStatus.INCONCLUSIVE for f in findings)


def test_confirmed_needs_evidence() -> None:
    touched = ScenarioResult("D8", Verdict.FAIL, expectations=(("corrupt_body_queried", False),),
                             hypothesis_ids=("H11",), hypothesis_outcome="INCONCLUSIVE", evidence_path="d8.json")
    findings = {f.hypothesis_id: f for f in derive_findings((touched,))}
    assert findings["H11"].status is HypothesisStatus.INCONCLUSIVE
    confirmed = ScenarioResult("D8", Verdict.FAIL, expectations=(("corrupt_body_queried", False),),
                                hypothesis_ids=("H11",), hypothesis_outcome="CONFIRMED", evidence_path="d8.json")
    findings = {f.hypothesis_id: f for f in derive_findings((confirmed,))}
    assert findings["H11"].status is HypothesisStatus.CONFIRMED
    assert findings["H11"].evidence == ("D8:d8.json",)
    unrecorded = ScenarioResult("D8", Verdict.FAIL, expectations=(("complete_evidence", False),),
                                hypothesis_ids=("H11",), hypothesis_outcome="CONFIRMED", evidence_path="d8.json")
    assert derive_findings((unrecorded,))[10].status is HypothesisStatus.INCONCLUSIVE
    no_record = ScenarioResult("D8", Verdict.FAIL, expectations=(("corrupt_body_queried", False),),
                               hypothesis_ids=("H11",), hypothesis_outcome="CONFIRMED")
    assert derive_findings((no_record,))[10].status is HypothesisStatus.INCONCLUSIVE


def test_refuted_when_deciding_scenario_shows_safe_behavior() -> None:
    safe = ScenarioResult("D8", Verdict.PASS, expectations=(("corrupt_body_queried", True),),
                          hypothesis_ids=("H11",), hypothesis_outcome="REFUTED", evidence_path="d8.json",
                          conservation_passed=True)
    findings = {f.hypothesis_id: f for f in derive_findings((safe,))}
    assert findings["H11"].status is HypothesisStatus.REFUTED
    assert findings["H11"].evidence == ("D8:d8.json",)
    skipped = ScenarioResult("D8", Verdict.SKIPPED, expectations=(("corrupt_body_queried", True),),
                             hypothesis_ids=("H11",), hypothesis_outcome="REFUTED", evidence_path="d8.json")
    assert derive_findings((skipped,))[10].status is HypothesisStatus.INCONCLUSIVE


def test_per_hypothesis_outcomes_do_not_cross_confirm() -> None:
    mixed = ScenarioResult("D5", Verdict.PASS, expectations=(("rate_limit_contained", True),),
                           hypothesis_ids=("H3", "H10"), hypothesis_outcomes=(("H3", "REFUTED"), ("H10", "CONFIRMED")),
                           evidence_path="d5.json", conservation_passed=True)
    findings = {f.hypothesis_id: f.status for f in derive_findings((mixed,))}
    assert findings["H3"] is HypothesisStatus.REFUTED
    assert findings["H10"] is HypothesisStatus.CONFIRMED


def test_format_findings_md_renders_table() -> None:
    from src.live.testnet_drill.findings import format_findings_md

    findings = derive_findings((ScenarioResult("D6", Verdict.PASS, expectations=(("ban_halted", True),),
                                               hypothesis_ids=("H4",), hypothesis_outcome="CONFIRMED",
                                               evidence_path="d6.json"),))
    table = format_findings_md(findings)
    assert "| hypothesis_id | status | evidence | remediation | 설명 |" in table
    assert "| H4 | confirmed | D6:d6.json | LOGIC_CHANGE |" in table


def test_remediation_is_advisory(tmp_path) -> None:
    import src.live.errors as errors

    before = dict(errors.BINANCE_ERROR_POLICY)
    results = (ScenarioResult("D6", Verdict.PASS, expectations=(("ban_halted", True),),
                              hypothesis_ids=("H4",), hypothesis_outcome="CONFIRMED"),)
    findings = derive_findings(results)
    assert dict(errors.BINANCE_ERROR_POLICY) == before
    assert isinstance(findings[0], Finding)
    assert all(f.remediation in ("REGISTRY_ADD", "LOGIC_CHANGE", "DOC_ONLY", "NONE") for f in findings)
