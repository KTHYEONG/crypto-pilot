"""Selection order and complete-evidence gates for drill scenarios."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from src.live.testnet_drill import registry
from src.live.testnet_drill.report import ScenarioResult, Verdict
from src.live.testnet_drill.scenarios.base import ev


@pytest.mark.parametrize("scenario_id", list(registry.SCENARIOS))
def test_missing_execution_evidence_never_passes(scenario_id):
    result = registry.SCENARIOS[scenario_id].evaluate(ev(scenario_id, note="unexecuted"))
    assert result.verdict is Verdict.FAIL
    assert result.conservation_passed is False
    assert not result.observations


def test_valid_expectations_cannot_self_certify_conservation():
    result = registry.SCENARIOS["B8"].evaluate(
        ev("B8", outcomes=["FILLED", "REJECTED", "FILLED"], escaped_exception=False, terminal_once=True)
    )
    assert result.verdict is Verdict.FAIL
    assert result.conservation_passed is False


def test_numeric_selection_order_and_deduplication(monkeypatch, tmp_path):
    seen = []

    def spec(key):
        return SimpleNamespace(
            group="C",
            run=lambda session, config: ev(key),
            evaluate=lambda evidence: seen.append(evidence.scenario_id) or ScenarioResult(key, Verdict.OBSERVED),
        )

    monkeypatch.setattr(registry, "SCENARIOS", MappingProxyType({key: spec(key) for key in ("C1", "C2", "C10")}))
    registry.run_selected(lambda: SimpleNamespace(workspace=tmp_path), ["C10", "C2", "C1", "C2"], None)
    assert seen == ["C1", "C2", "C10"]
    assert (tmp_path / "scenario_evidence.json").is_file()
    seen.clear()
    registry.run_selected(lambda: SimpleNamespace(workspace=tmp_path), "C", None)
    assert seen == ["C1", "C2", "C10"]
    seen.clear()
    registry.run_selected(lambda: SimpleNamespace(workspace=tmp_path), "all", None)
    assert seen == ["C1", "C2", "C10"]


def test_invalid_selection_is_rejected_before_session_contact():
    def refuse_contact():
        raise AssertionError("must not open a session")

    with pytest.raises(ValueError, match="unknown scenarios"):
        registry.run_selected(refuse_contact, ["C1", "A999"], None)


def test_duplicate_registration_is_refused(monkeypatch):
    first = registry.SCENARIOS["A1"]
    monkeypatch.setattr(registry.account_config, "scenario_specs", lambda: (first, first))
    with pytest.raises(ValueError, match="duplicate"):
        registry._collect()


def test_account_failure_stops_later_groups(monkeypatch, tmp_path):
    seen = []

    def fake_spec(key, group):
        def run(session, config):
            seen.append(key)
            return ev(key)

        return SimpleNamespace(group=group, run=run, evaluate=lambda evidence: ScenarioResult(key, Verdict.FAIL))

    monkeypatch.setattr(registry, "SCENARIOS", {"A1": fake_spec("A1", "A"), "B1": fake_spec("B1", "B")})
    registry.run_selected(lambda: SimpleNamespace(workspace=tmp_path), "all", None)
    assert seen == ["A1"]


def test_wrong_scenario_evidence_is_rejected():
    assert registry.SCENARIOS["C1"].evaluate(ev("C2")).verdict is Verdict.FAIL


def test_malformed_complete_evidence_is_a_failure():
    from src.live.testnet_drill.scenarios.base import REQUIRED_EVIDENCE

    evidence = ev("C4", **dict.fromkeys(REQUIRED_EVIDENCE["C4"], None))
    assert registry.SCENARIOS["C4"].evaluate(evidence).verdict is Verdict.FAIL


def test_failed_observation_is_not_reported_as_success():
    evidence = ev("B10", code=-4116, query_returned="", submit_seq_journal_derived=False)
    assert registry.SCENARIOS["B10"].evaluate(evidence).verdict is Verdict.FAIL


def _raising_spec(key, exc):
    def run(session, config):
        raise exc

    return SimpleNamespace(group=key[0], run=run, evaluate=lambda evidence: ScenarioResult(key, Verdict.PASS))


def test_scenario_error_is_redacted_failure_and_stops_run(monkeypatch, tmp_path):
    later = SimpleNamespace(group="C", run=lambda s, c: (_ for _ in ()).throw(AssertionError("must stop")), evaluate=None)
    monkeypatch.setattr(
        registry, "SCENARIOS", {"C1": _raising_spec("C1", ValueError("bad signature=abc")), "C2": later}
    )
    results = registry.run_selected(lambda: SimpleNamespace(workspace=tmp_path), "all", None)
    assert [(r.scenario_id, r.verdict) for r in results] == [("C1", Verdict.FAIL)]
    assert "abc" not in results[0].proposal_rationale


def test_refused_precondition_is_skipped_and_run_continues(monkeypatch, tmp_path):
    from src.live.testnet_drill.guard import DrillRefused

    later = SimpleNamespace(group="C", run=lambda s, c: ev("C2"), evaluate=lambda e: ScenarioResult("C2", Verdict.OBSERVED))
    monkeypatch.setattr(
        registry, "SCENARIOS", {"C1": _raising_spec("C1", DrillRefused("drill refused: intent_notional_cap")), "C2": later}
    )
    results = registry.run_selected(lambda: SimpleNamespace(workspace=tmp_path), "all", None)
    assert [(r.scenario_id, r.verdict) for r in results] == [("C1", Verdict.SKIPPED), ("C2", Verdict.OBSERVED)]


def test_dirty_session_aborts_whole_run(monkeypatch, tmp_path):
    from src.live.testnet_drill.session import DirtySessionError

    monkeypatch.setattr(registry, "SCENARIOS", {"A4": _raising_spec("A4", DirtySessionError("DIRTY"))})
    with pytest.raises(DirtySessionError):
        registry.run_selected(lambda: SimpleNamespace(workspace=tmp_path), "all", None)


def test_evidence_symlink_outside_workspace_is_refused(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "scenario_evidence.json").symlink_to(tmp_path / "outside.json")
    spec = SimpleNamespace(group="C", run=lambda s, c: ev("C1"), evaluate=lambda e: ScenarioResult("C1", Verdict.OBSERVED))
    monkeypatch.setattr(registry, "SCENARIOS", {"C1": spec})
    with pytest.raises(ValueError, match="inside workspace"):
        registry.run_selected(lambda: SimpleNamespace(workspace=workspace), "all", None)
    assert not (tmp_path / "outside.json").exists()
