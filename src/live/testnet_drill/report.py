"""Drill verdicts, registry proposals and workspace reports."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.live.errors import BINANCE_ERROR_POLICY
from src.live.testnet_drill.faults import redact

_HYPOTHESIS_OUTCOMES = frozenset({"CONFIRMED", "REFUTED", "INCONCLUSIVE"})


class Verdict(str, Enum):  # noqa: UP042 - contract pins the (str, Enum) base
    """Closed set of scenario verdicts."""

    PASS = "pass"  # noqa: S105 - enum member name, not a credential
    FAIL = "fail"
    OBSERVED = "observed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class VenueObservation:
    """One venue error response seen in a scenario."""

    code: int | None
    http_status: int
    message: str
    path: str
    registry_action: str


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Verdict plus named expectation outcomes, observations and hypothesis evidence."""

    scenario_id: str
    verdict: Verdict
    expectations: tuple[tuple[str, bool], ...] = ()
    observations: tuple[VenueObservation, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    hypothesis_outcome: str = "INCONCLUSIVE"
    evidence_path: str = ""
    conservation_passed: bool = False
    proposed_action: str | None = None
    proposal_rationale: str = ""

    def __post_init__(self) -> None:
        if self.hypothesis_outcome not in _HYPOTHESIS_OUTCOMES:
            raise ValueError(f"hypothesis_outcome must be one of {sorted(_HYPOTHESIS_OUTCOMES)}")
        if self.verdict is Verdict.PASS and (
            not self.expectations or not all(passed for _, passed in self.expectations) or not self.conservation_passed
        ):
            object.__setattr__(self, "verdict", Verdict.FAIL)


@dataclass(frozen=True, slots=True)
class RegistryProposal:
    """A candidate error-policy entry for an unregistered venue code."""

    code: int
    venue_msg: str
    http_status: int
    observed_in: tuple[str, ...]
    proposed_action: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """Workspace report artifacts written by write_report."""

    json_path: Path
    md_path: Path
    evidence_path: Path


def apply_conservation(result: ScenarioResult, passed: bool) -> ScenarioResult:
    """Downgrade PASS to FAIL when the post-scenario conservation check failed."""
    if result.verdict is Verdict.PASS and not passed:
        return dataclasses.replace(result, verdict=Verdict.FAIL, conservation_passed=False)
    if result.conservation_passed is not passed:
        return dataclasses.replace(result, conservation_passed=passed)
    return result


def summarize_results(results: Sequence[ScenarioResult]) -> dict[str, int]:
    """Count verdicts; OBSERVED never counts as PASS or FAIL."""
    summary = {"pass": 0, "fail": 0, "observed": 0, "skipped": 0}
    for result in results:
        if result.verdict is Verdict.PASS:
            summary["pass"] += 1
        elif result.verdict is Verdict.FAIL:
            summary["fail"] += 1
        elif result.verdict is Verdict.OBSERVED:
            summary["observed"] += 1
        else:
            summary["skipped"] += 1
    return summary


def build_registry_proposals(results: Sequence[ScenarioResult]) -> tuple[RegistryProposal, ...]:
    """Propose registry entries for observed codes absent from the error policy."""
    seen: dict[int, RegistryProposal] = {}
    observed_in: dict[int, list[str]] = {}
    for result in results:
        for observation in result.observations:
            if observation.code is None or observation.code in BINANCE_ERROR_POLICY:
                continue
            if result.proposed_action is None:
                raise ValueError("unregistered code requires a declared proposed_action")
            if observation.code in seen and seen[observation.code].proposed_action != result.proposed_action:
                raise ValueError("conflicting declared proposed_action")
            observed_in.setdefault(observation.code, []).append(result.scenario_id)
            if observation.code not in seen:
                seen[observation.code] = RegistryProposal(
                    code=observation.code,
                    venue_msg=_strip_secrets(observation.message),
                    http_status=observation.http_status,
                    observed_in=(),
                    proposed_action=result.proposed_action,
                    rationale=_strip_secrets(result.proposal_rationale),
                )
    return tuple(
        dataclasses.replace(proposal, observed_in=tuple(observed_in[proposal.code]))
        for proposal in (seen[code] for code in sorted(seen))
    )


def _strip_secrets(text: str) -> str:
    return redact(text)


def _result_payload(result: ScenarioResult) -> dict[str, object]:
    return {
        "scenario_id": result.scenario_id,
        "verdict": result.verdict.value,
        "expectations": [{"name": name, "passed": passed} for name, passed in result.expectations],
        "observations": [
            {
                "code": observation.code,
                "http_status": observation.http_status,
                "message": _strip_secrets(observation.message),
                "path": observation.path,
                "registry_action": observation.registry_action,
            }
            for observation in result.observations
        ],
        "hypothesis_ids": list(result.hypothesis_ids),
        "hypothesis_outcome": result.hypothesis_outcome,
        "evidence_path": result.evidence_path,
        "conservation_passed": result.conservation_passed,
    }


def write_report(results: Sequence[ScenarioResult], workspace: Path) -> ReportPaths:
    """Write report.json, report.md and evidence.jsonl under the workspace."""
    workspace.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    summary = summarize_results(rows)
    hypotheses = [
        {"id": hypothesis, "outcome": result.hypothesis_outcome, "evidence": result.evidence_path}
        for result in rows
        for hypothesis in result.hypothesis_ids
    ]
    json_path = workspace / "report.json"
    md_path = workspace / "report.md"
    evidence_path = workspace / "evidence.jsonl"
    if any(not path.resolve().is_relative_to(workspace.resolve()) for path in (json_path, md_path, evidence_path)):
        raise ValueError("report paths must stay inside workspace")
    with evidence_path.open("w", encoding="utf-8") as handle:
        for result in rows:
            for observation in result.observations:
                handle.write(
                    json.dumps(
                        {
                            "scenario_id": result.scenario_id,
                            "code": observation.code,
                            "http_status": observation.http_status,
                            "message": _strip_secrets(observation.message),
                            "path": observation.path,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [_result_payload(result) for result in rows],
                "hypotheses": hypotheses,
                "registry_proposals": [dataclasses.asdict(p) for p in build_registry_proposals(rows)],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Testnet Drill Report",
        "",
        f"합계: PASS {summary['pass']} / FAIL {summary['fail']} / OBSERVED {summary['observed']} / SKIPPED {summary['skipped']}",
        "",
        "| scenario_id | verdict | expectations | hypotheses |",
        "| --- | --- | --- | --- |",
    ]
    for result in rows:
        expectations = (
            ", ".join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in result.expectations) or "-"
        )
        hypotheses_cell = (
            ", ".join(f"{hypothesis}={result.hypothesis_outcome}" for hypothesis in result.hypothesis_ids) or "-"
        )
        lines.append(f"| {result.scenario_id} | {result.verdict.value} | {expectations} | {hypotheses_cell} |")
    lines.extend(["", "| hypothesis_id | outcome | evidence |", "| --- | --- | --- |"])
    lines.extend(f"| {h['id']} | {h['outcome']} | {h['evidence']} |" for h in hypotheses)
    md_path.write_text(_strip_secrets("\n".join(lines)) + "\n", encoding="utf-8")
    return ReportPaths(json_path=json_path, md_path=md_path, evidence_path=evidence_path)
