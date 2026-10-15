"""Shared scenario contract for testnet drill groups."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import wraps
from typing import Any

from src.live.testnet_drill.report import ScenarioResult, VenueObservation, Verdict


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    """Recorded venue truth for one scenario; evaluate is pure over this."""

    scenario_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Executable scenario with a pure evaluator."""

    scenario_id: str
    group: str
    title: str
    hypotheses: tuple[str, ...]
    run: Callable[[Any, Any], ScenarioEvidence]
    evaluate: Callable[[ScenarioEvidence], ScenarioResult]

    def __post_init__(self) -> None:
        evaluator = self.evaluate

        @wraps(evaluator)
        def checked(evidence: ScenarioEvidence) -> ScenarioResult:
            required = REQUIRED_EVIDENCE[self.scenario_id]
            missing = sorted(required - evidence.payload.keys())
            invalid_flags = [
                key
                for key in required & BOOLEAN_FIELDS
                if key in evidence.payload and not isinstance(evidence.payload[key], bool)
            ]
            if evidence.scenario_id != self.scenario_id or missing or invalid_flags or evidence.payload.get("error"):
                return ScenarioResult(
                    self.scenario_id,
                    Verdict.FAIL,
                    expectations=(("complete_evidence", False),),
                    proposal_rationale=f"missing={missing}; {evidence.payload.get('error', '')}",
                )
            try:
                result = evaluator(evidence)
            except (TypeError, ValueError, ArithmeticError, KeyError):
                return ScenarioResult(self.scenario_id, Verdict.FAIL, expectations=(("valid_evidence", False),))
            conservation = evidence.payload.get("conservation_passed") is True
            verdict = result.verdict
            if (result.verdict is Verdict.PASS and not conservation) or any(
                not passed for _, passed in result.expectations
            ):
                verdict = Verdict.FAIL
            return replace(result, verdict=verdict, conservation_passed=conservation)

        object.__setattr__(self, "evaluate", checked)


REQUIRED_EVIDENCE: dict[str, frozenset[str]] = {
    key: frozenset(fields.split())
    for key, fields in {
        "A1": "testnet_accepted mainnet_code redacted",
        "A2": "accepted resynced order_accepted offset_ms",
        "A3": "sends resyncs max_sends accepted first_code",
        "A4": "gate raw_code restored",
        "A5": "gate raw_code restored",
        "A6": "code blocked gate restored",
        "A7": "code action silent_success restored",
        "A8": "keys total_maint total_initial snapshot_total_maint",
        "B1": "per_symbol raw_codes",
        "B2": "level client_decision venue_decision agreed",
        "B3": "level client_decision venue_decision agreed",
        "B4": "code chases bounded interval_respected",
        "B5": "code planner_emits_above",
        "B6": "code retries max_margin_rejects reduce_only_waited status",
        "B7": "code others_unaffected",
        "B8": "outcomes escaped_exception terminal_once",
        "B9": "code cycle_halted",
        "B10": "code query_returned submit_seq_journal_derived",
        "B11": "ok_36 rejected_37 venue_accepted non_ascii_accepted legacy_accepted",
        "C1": "status_path closed fills journal_submit journal_terminal",
        "C2": "code chases window no_resting_taker",
        "C3": "status qty_exact vwap_within_tolerance maker_false",
        "C4": "status executed_qty requested_qty ioc_attempts max_ioc_attempts residual_exact booked_once",
        "C5": "cumulative_monotonic deltas_sum_to_venue delta_price user_trade_price price_rel_tolerance uses_cumulative_avg",
        "C6": "repetitions",
        "C7": "never_flips contained_to_intent codes",
        "C8": "final_equals_target residual_explicit has_obsolete_leg",
        "C9": "open_order_snapshots cumulative_filled intent_qty interval_respected",
        "C10": "slice_sum intent_qty residual_explicit slice_ids_unique final_equals_target",
        "C11": "r1_r6_pass symbol_count",
        "C12": "escalated_to_ioc price_within_cap open_empty residual_exact",
        "C13": "no_429 intent_count symbol_count throttle_respected reducing_first",
        "C14": "reducing_first no_spurious_2019 margin_near_limit",
    }.items()
}

BOOLEAN_FIELDS = frozenset(
    [
        "testnet_accepted",
        "accepted",
        "resynced",
        "order_accepted",
        "restored",
        "silent_success",
        "agreed",
        "bounded",
        "interval_respected",
        "planner_emits_above",
        "reduce_only_waited",
        "others_unaffected",
        "escaped_exception",
        "terminal_once",
        "cycle_halted",
        "submit_seq_journal_derived",
        "ok_36",
        "rejected_37",
        "venue_accepted",
        "non_ascii_accepted",
        "legacy_accepted",
        "closed",
        "journal_submit",
        "journal_terminal",
        "no_resting_taker",
        "qty_exact",
        "vwap_within_tolerance",
        "maker_false",
        "residual_exact",
        "booked_once",
        "cumulative_monotonic",
        "deltas_sum_to_venue",
        "uses_cumulative_avg",
        "never_flips",
        "contained_to_intent",
        "final_equals_target",
        "residual_explicit",
        "has_obsolete_leg",
        "slice_ids_unique",
        "r1_r6_pass",
        "escalated_to_ioc",
        "price_within_cap",
        "open_empty",
        "no_429",
        "throttle_respected",
        "reducing_first",
        "no_spurious_2019",
        "margin_near_limit",
    ]
)


def isolated_run(
    body: Callable[..., ScenarioEvidence], *, with_slate: bool = False
) -> Callable[[Any, Any], ScenarioEvidence]:
    """Collect teardown and conservation evidence before closing session handles."""

    def run(session: Any, config: Any) -> ScenarioEvidence:
        from src.live.testnet_drill.reconcile import evaluate_reconciliation, reconcile_session
        from src.live.testnet_drill.session import (
            _read_positions,
            close_session_handles,
            prepare_clean_slate,
            teardown_session,
        )

        since_ms = int(session.opened_at * 1000)
        try:
            prepare_clean_slate(session)
            baseline = Decimal(str(session.client.account()["totalWalletBalance"]))
            session = replace(session, starting_wallet_balance=baseline)
            since_ms = int(time.time() * 1000)
            evidence = body(session, config, None) if with_slate else body(session, config)
        finally:
            try:
                teardown_session(session, close_handles=False)
                reconciled = reconcile_session(session, since_ms=since_ms, settle_wait_s=config.settle_wait_s)
                checks = evaluate_reconciliation(reconciled, config)
                aux_clean = all(
                    not _read_positions(aux.client) and not aux.client.open_orders() for aux in session.counterparties
                )
                primary_clean = not _read_positions(session.client) and not session.client.open_orders()
            finally:
                close_session_handles(session)
        return replace(
            evidence,
            payload={
                **evidence.payload,
                **({"r1_r6_pass": all(check.passed for check in checks)} if evidence.scenario_id == "C11" else {}),
                "conservation_passed": primary_clean and aux_clean and all(check.passed for check in checks),
                "reconciliation": checks,
                "counterparty_clean": aux_clean,
                "primary_clean": primary_clean,
            },
        )

    return run


def ev(scenario_id: str, **payload: Any) -> ScenarioEvidence:
    """Build evidence without venue contact."""
    return ScenarioEvidence(scenario_id=scenario_id, payload=dict(payload))


def observation(
    code: int | None,
    registry_action: str,
    *,
    http_status: int = 400,
    path: str = "/fapi/v1/order",
    message: str = "",
) -> VenueObservation:
    """One venue observation row carrying our resolved action."""
    return VenueObservation(
        code=code,
        http_status=http_status,
        message=message or f"venue code {code}",
        path=path,
        registry_action=registry_action,
    )


def result_of(
    evidence: ScenarioEvidence,
    scenario_id: str,
    verdict: Verdict,
    expectations: tuple[tuple[str, bool], ...] = (),
    *,
    observations: tuple[VenueObservation, ...] = (),
    hypotheses: tuple[str, ...] = (),
    hypothesis_outcome: str = "INCONCLUSIVE",
    proposed_action: str | None = None,
    proposal_rationale: str = "",
) -> ScenarioResult:
    """Build a result honoring the PASS gate in ScenarioResult."""
    return ScenarioResult(
        scenario_id=scenario_id,
        verdict=verdict,
        expectations=expectations,
        observations=observations,
        hypothesis_ids=hypotheses,
        hypothesis_outcome=hypothesis_outcome,
        conservation_passed=evidence.payload.get("conservation_passed") is True,
        proposed_action=proposed_action,
        proposal_rationale=proposal_rationale,
    )
