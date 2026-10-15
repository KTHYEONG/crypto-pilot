"""Hypothesis outcomes (H1..H11) derived from drill scenario results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class HypothesisStatus(str, Enum):  # noqa: UP042 - contract pins the (str, Enum) base
    """Closed set of hypothesis outcomes."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class Finding:
    """One hypothesis outcome with evidence record ids and a proposed remediation class."""

    hypothesis_id: str
    status: HypothesisStatus
    evidence: tuple[str, ...] = ()
    remediation: str = "NONE"
    detail: str = ""


_REMEDIATION: dict[str, tuple[str, str]] = {
    "H1": ("REGISTRY_ADD", "등록된 코드에 명시적 조치를 추가하고 미등록 코드는 scoped fail-closed를 검토한다."),
    "H2": ("LOGIC_CHANGE", "누적 체결값 차이로 델타가를 도출하고 userTrades 수수료·메이커 표시를 기록한다."),
    "H3": ("LOGIC_CHANGE", "멱등 취소를 유한 백오프로 재시도하고 조회로 확인한 뒤 중단한다."),
    "H4": ("LOGIC_CHANGE", "418 Retry-After를 포착해 venue_ban_until을 유지하고 호출을 중단한다."),
    "H5": ("DOC_ONLY", "totalMaintMargin 출처를 채우거나 미사용 필드를 제거하고 MMR 게이트를 명시한다."),
    "H6": ("LOGIC_CHANGE", "채택·복구 경로에서 주문별 단일 워터마크를 유지한다."),
    "H7": ("LOGIC_CHANGE", "축소 전용 다리가 끝난 뒤 개시 다리를 내거나 잔여분을 명시한다."),
    "H8": ("LOGIC_CHANGE", "시작 시 베뉴 최근 주문에서 submit_seq 하한을 도출한다."),
    "H9": ("LOGIC_CHANGE", "포지션을 포함한 단일 계정 엔드포인트를 쓰거나 갱신 시 재조회한다."),
    "H10": ("LOGIC_CHANGE", "Retry-After 대기를 남은 실행 창으로 제한하고 초과 시 틱을 건너뛴다."),
    "H11": ("LOGIC_CHANGE", "변이 2xx의 빈·파손 본문이나 식별자 누락을 OrderStatusUnknown으로 처리한다."),
}

_HYPOTHESES: tuple[str, ...] = tuple(f"H{i}" for i in range(1, 12))


def _decisive(result: Any) -> bool:
    """A result can decide a hypothesis only if its scenario ran and recorded complete evidence."""
    verdict = getattr(result, "verdict", None)
    if getattr(verdict, "value", verdict) == "skipped" or not getattr(result, "evidence_path", ""):
        return False
    return not any(name in _INCOMPLETE for name, _ in getattr(result, "expectations", ()))


_INCOMPLETE = frozenset({"complete_evidence", "valid_evidence", "scenario_executed"})


def _status_for(hypothesis_id: str, results: Sequence[Any]) -> tuple[HypothesisStatus, tuple[str, ...]]:
    confirmed: list[str] = []
    refuted: list[str] = []
    for result in results:
        if hypothesis_id not in tuple(getattr(result, "hypothesis_ids", ())) or not _decisive(result):
            continue
        outcome = result.outcome_for(hypothesis_id)
        record = f"{result.scenario_id}:{result.evidence_path}"
        if outcome == "CONFIRMED":
            confirmed.append(record)
        elif outcome == "REFUTED":
            refuted.append(record)
    if confirmed:
        return HypothesisStatus.CONFIRMED, tuple(confirmed)
    if refuted:
        return HypothesisStatus.REFUTED, tuple(refuted)
    return HypothesisStatus.INCONCLUSIVE, ()


def derive_findings(results: Sequence[Any]) -> tuple[Finding, ...]:
    """Derive one finding per hypothesis without mutating production state."""
    findings: list[Finding] = []
    for hypothesis_id in _HYPOTHESES:
        status, evidence = _status_for(hypothesis_id, results)
        remediation, detail = _REMEDIATION[hypothesis_id]
        findings.append(Finding(hypothesis_id=hypothesis_id, status=status, evidence=evidence,
                                remediation=remediation if status is not HypothesisStatus.INCONCLUSIVE else "NONE",
                                detail=detail if status is not HypothesisStatus.INCONCLUSIVE else ""))
    return tuple(findings)


def format_findings_md(findings: Sequence[Finding]) -> str:
    """Render the Korean-language hypothesis table with English keys."""
    lines = [
        "| hypothesis_id | status | evidence | remediation | 설명 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for finding in findings:
        evidence = ", ".join(finding.evidence) or "-"
        detail = finding.detail or "근거 부족: 결정 시나리오가 실행되지 않았거나 증거가 불완전하다."
        lines.append(
            f"| {finding.hypothesis_id} | {finding.status.value} | {evidence} | {finding.remediation} | {detail} |"
        )
    return "\n".join(lines) + "\n"
