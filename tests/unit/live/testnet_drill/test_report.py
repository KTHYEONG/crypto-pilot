"""Drill report, verdict and registry-proposal invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.errors import BINANCE_ERROR_POLICY
from src.live.testnet_drill.report import (
    ScenarioResult,
    VenueObservation,
    Verdict,
    apply_conservation,
    build_registry_proposals,
    summarize_results,
    write_report,
)


def _observation(code: int | None, scenario: str = "B9") -> VenueObservation:
    return VenueObservation(code=code, http_status=400, message=f"venue msg {code} signature=abc", path="/fapi/v1/order", registry_action="fail_closed")


def _result(scenario: str, verdict: Verdict, **overrides) -> ScenarioResult:
    base = {"scenario_id": scenario, "verdict": verdict, "expectations": (("accepted", True),), "observations": (), "hypothesis_ids": ("H1",), "hypothesis_outcome": "CONFIRMED", "evidence_path": f"{scenario}.jsonl", "conservation_passed": True, "proposed_action": "fail_closed"}
    base.update(overrides)
    return ScenarioResult(**base)


def test_proposal_only_for_unregistered_codes() -> None:
    results = [
        _result("B5", Verdict.OBSERVED, observations=(_observation(-2019),), proposed_action="margin_wait", proposal_rationale="wait"),
        _result("B9", Verdict.OBSERVED, observations=(_observation(-4061),), proposed_action="fail_closed", proposal_rationale="halt"),
    ]
    (proposal,) = build_registry_proposals(results)
    assert proposal.code == -4061
    assert proposal.observed_in == ("B9",)
    assert proposal.proposed_action == "fail_closed"
    assert proposal.rationale == "halt"


def test_duplicate_code_merges_observed_in() -> None:
    results = [_result("B9", Verdict.OBSERVED, observations=(_observation(-4061),)), _result("B10", Verdict.OBSERVED, observations=(_observation(-4061),))]
    (proposal,) = build_registry_proposals(results)
    assert proposal.observed_in == ("B9", "B10")


def test_none_code_skipped() -> None:
    assert build_registry_proposals([_result("D5", Verdict.OBSERVED, observations=(_observation(None),))]) == ()


def test_observed_never_passes() -> None:
    summary = summarize_results([_result("B9", Verdict.OBSERVED)])
    assert summary == {"pass": 0, "fail": 0, "observed": 1, "skipped": 0}


def test_summarize_counts_every_verdict() -> None:
    results = [_result("B1", Verdict.PASS), _result("B2", Verdict.FAIL), _result("B3", Verdict.OBSERVED), _result("B4", Verdict.SKIPPED)]
    assert summarize_results(results) == {"pass": 1, "fail": 1, "observed": 1, "skipped": 1}


def test_conservation_failure_downgrades_pass() -> None:
    downgraded = apply_conservation(_result("C3", Verdict.PASS), False)
    assert downgraded.verdict is Verdict.FAIL
    assert downgraded.conservation_passed is False


def test_conservation_success_keeps_pass() -> None:
    intact = _result("C3", Verdict.PASS)
    assert apply_conservation(intact, True) is intact


def test_conservation_flag_syncs_on_observed() -> None:
    synced = apply_conservation(_result("B9", Verdict.OBSERVED), False)
    assert synced.verdict is Verdict.OBSERVED
    assert synced.conservation_passed is False


def test_rejects_unknown_hypothesis_outcome() -> None:
    with pytest.raises(ValueError, match="hypothesis_outcome"):
        _result("B9", Verdict.OBSERVED, hypothesis_outcome="MAYBE")


def test_report_is_write_only_to_workspace(tmp_path: Path) -> None:
    policy_before = dict(BINANCE_ERROR_POLICY)
    workspace = tmp_path / "ws"
    paths = write_report([_result("B9", Verdict.PASS, observations=(_observation(-4061),))], workspace)
    assert paths.json_path.is_relative_to(workspace)
    assert paths.md_path.is_relative_to(workspace)
    assert paths.evidence_path.is_relative_to(workspace)
    assert dict(BINANCE_ERROR_POLICY) == policy_before
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["pass"] == 1
    assert payload["hypotheses"][0]["id"] == "H1"
    assert "signature=abc" not in paths.json_path.read_text(encoding="utf-8")
    assert "signature=<stripped>" in paths.evidence_path.read_text(encoding="utf-8")
    assert "합계" in paths.md_path.read_text(encoding="utf-8")


def test_empty_report(tmp_path: Path) -> None:
    paths = write_report([], tmp_path / "ws")
    assert json.loads(paths.json_path.read_text(encoding="utf-8"))["summary"]["pass"] == 0
    assert paths.md_path.exists()
    assert paths.evidence_path.exists()


def test_pass_requires_expectations_and_explicit_conservation() -> None:
    assert ScenarioResult("C1", Verdict.PASS).verdict is Verdict.FAIL
    assert _result("C1", Verdict.PASS, expectations=(("filled", False),)).verdict is Verdict.FAIL
    assert _result("C1", Verdict.PASS, conservation_passed=False).verdict is Verdict.FAIL


def test_undeclared_and_conflicting_proposals_refused() -> None:
    observation = (_observation(-4061),)
    with pytest.raises(ValueError, match="declared"):
        build_registry_proposals([_result("B1", Verdict.OBSERVED, observations=observation, proposed_action=None)])
    with pytest.raises(ValueError, match="conflicting"):
        build_registry_proposals([
            _result("B1", Verdict.OBSERVED, observations=observation, proposed_action="intent_reject"),
            _result("B2", Verdict.OBSERVED, observations=observation, proposed_action="fail_closed"),
        ])


def test_report_symlink_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.json"
    (workspace / "report.json").symlink_to(outside)
    with pytest.raises(ValueError, match="inside workspace"):
        write_report([], workspace)
    assert not outside.exists()


def test_per_hypothesis_outcomes_must_be_declared_and_valid() -> None:
    import pytest

    from src.live.testnet_drill.report import ScenarioResult, Verdict

    with pytest.raises(ValueError, match="declared"):
        ScenarioResult("D5", Verdict.OBSERVED, hypothesis_ids=("H3",), hypothesis_outcomes=(("H10", "CONFIRMED"),))
    with pytest.raises(ValueError, match="hypothesis_outcome"):
        ScenarioResult("D5", Verdict.OBSERVED, hypothesis_ids=("H3",), hypothesis_outcomes=(("H3", "MAYBE"),))
    result = ScenarioResult("D5", Verdict.OBSERVED, hypothesis_ids=("H3", "H10"), hypothesis_outcomes=(("H3", "REFUTED"),))
    assert (result.outcome_for("H3"), result.outcome_for("H10")) == ("REFUTED", "INCONCLUSIVE")
