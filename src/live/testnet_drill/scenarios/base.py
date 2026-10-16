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
        "B4": "code",
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
        "C5": "cumulative_monotonic deltas_sum_to_venue multi_step_observed delta_price user_trade_price price_rel_tolerance uses_cumulative_avg",
        "C6": "repetitions",
        "C7": "never_flips contained_to_intent codes",
        "C8": "final_equals_target residual_explicit has_obsolete_leg open_posted_after_close_filled blocked_open_leg open_leg_posts final_position target_qty position_never_crossed_zero",
        "C9": "open_order_snapshots cumulative_filled intent_qty interval_respected repegs",
        "C10": "slice_sum intent_qty residual_explicit slice_ids_unique final_equals_target",
        "C11": "r1_r6_pass symbol_count",
        "C12": "escalated_to_ioc price_within_cap open_empty residual_exact",
        "C13": "no_429 intent_count symbol_count throttle_respected reducing_first",
        "C14": "reducing_first no_spurious_2019 margin_near_limit",
        "D1": "unresolved_adopted single_post booked_once",
        "D2": "booked_from_query position_parity no_repost",
        "D3": "reposted_early reposted_after_horizon miss_limit_reached journal_not_placed never_two_live post_count",
        "D4": "delete_count confirmed_by_query terminal_once",
        "D5": "not_placed retried_later no_halt cancel_outcome resting_left rate_limited_read retry_after_s sleep_s remaining_window_s sleep_exceeds_window",
        "D6": "records_after_ban ban_horizon_captured halted",
        "D7": "get_retries_bounded post_unknown_no_repost post_fail_closed no_duplicate_exposure post_resent_after_5xx",
        "D8": "resolved_by_query treated_as_active_without_query",
        "D9": "post_count first_code resync_count no_duplicate",
        "D10": "tick_skipped counters_reset aborted_with_cleanup no_live_order max_failures_exceeded order_read_contained",
        "E1": "exit_code query_code terminal_status venue_exposure orphan_count",
        "E2": "settled_terminal no_fills no_orphan",
        "E3": "recovered_qty venue_qty fee_taker single_booking",
        "E4": "recovered_once no_double_count",
        "E5": "journal_ahead_before ledger_equal_after single_commit",
        "E6": "confirmed_or_recancelled fill_booked",
        "E7": "no_live_order no_double_exposure next_cycle_posts",
        "E8": "recovery_passes recovered_fill_count single_pass_fill_count idempotent",
        "E9": "unsettled_within_budget settled_after_restart decision_executed",
        "E10": "tail_corrupt_at truncated_tail_ok mid_file_fails_closed",
        "F1": "foreign_audited foreign_cancelled symbol_excluded halt_when_derisk_off derisk_when_enabled",
        "F2": "breach_detected derisk_or_halt adoption_listed breach_cleared",
        "F3": "orphan_cancelled orphan_settled symbol_withheld_on_unconfirmed",
        "F4": "inconsistency_rate breach_would_exceed",
        "F5": "funding_events funding_parity sign_convention_ok skipped",
        "F6": "limits_match throttle_reacts no_429",
        "F7": "clean_pass removal_flips warning_only",
        "G1": "cycles_completed r1_r6_pass fills_agree final_flat",
        "G2": "new_orders decision_guard_honored no_duplicate_fills",
        "G3": "converged no_double_execution silent_shortfall",
        "G4": "guard_tripped no_phantom_fill",
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
        "open_posted_after_close_filled",
        "blocked_open_leg",
        "position_never_crossed_zero",
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
        "unresolved_adopted",
        "single_post",
        "booked_from_query",
        "position_parity",
        "no_repost",
        "reposted_early",
        "reposted_after_horizon",
        "miss_limit_reached",
        "confirmed_by_query",
        "not_placed",
        "retried_later",
        "no_halt",
        "sleep_exceeds_window",
        "ban_horizon_captured",
        "get_retries_bounded",
        "post_unknown_no_repost",
        "post_fail_closed",
        "no_duplicate_exposure",
        "post_resent_after_5xx",
        "resolved_by_query",
        "treated_as_active_without_query",
        "no_duplicate",
        "tick_skipped",
        "counters_reset",
        "aborted_with_cleanup",
        "no_live_order",
        "max_failures_exceeded",
        "settled_terminal",
        "no_fills",
        "no_orphan",
        "fee_taker",
        "single_booking",
        "recovered_once",
        "no_double_count",
        "journal_ahead_before",
        "ledger_equal_after",
        "single_commit",
        "confirmed_or_recancelled",
        "fill_booked",
        "no_double_exposure",
        "idempotent",
        "unsettled_within_budget",
        "settled_after_restart",
        "decision_executed",
        "truncated_tail_ok",
        "mid_file_fails_closed",
        "foreign_audited",
        "foreign_cancelled",
        "symbol_excluded",
        "halt_when_derisk_off",
        "breach_detected",
        "derisk_or_halt",
        "adoption_listed",
        "breach_cleared",
        "orphan_cancelled",
        "orphan_settled",
        "symbol_withheld_on_unconfirmed",
        "breach_would_exceed",
        "funding_parity",
        "skipped",
        "limits_match",
        "throttle_reacts",
        "clean_pass",
        "removal_flips",
        "warning_only",
        "fills_agree",
        "final_flat",
        "decision_guard_honored",
        "no_duplicate_fills",
        "converged",
        "no_double_execution",
        "silent_shortfall",
        "guard_tripped",
        "no_phantom_fill",
        "journal_not_placed",
        "never_two_live",
        "resting_left",
        "rate_limited_read",
        "halted",
        "order_read_contained",
        "next_cycle_posts",
        "derisk_when_enabled",
        "sign_convention_ok",
    ]
)


def isolated_run(
    body: Callable[..., ScenarioEvidence], *, with_slate: bool = False
) -> Callable[[Any, Any], ScenarioEvidence]:
    """Collect teardown and conservation evidence before closing session handles."""

    def run(session: Any, config: Any) -> ScenarioEvidence:
        from src.live.order_journal import OrderJournal
        from src.live.testnet_drill.reconcile import evaluate_reconciliation, reconcile_session
        from src.live.testnet_drill.session import (
            _read_positions,
            close_session_handles,
            prepare_clean_slate,
            teardown_session,
        )

        since_ms = session.client.server_time_ms()
        reset_suspected = False
        try:
            reset_suspected = prepare_clean_slate(session).testnet_reset_suspected
            # Venue timestamps and the learned clock offset disagree by up to a round trip, so keep slate
            # cleanup trades and the scenario's first trade at least one guard interval away from the window edge.
            time.sleep(config.window_guard_s)
            baseline = Decimal(str(session.client.account()["totalWalletBalance"]))
            session = replace(session, starting_wallet_balance=baseline)
            since_ms = session.client.server_time_ms()
            time.sleep(config.window_guard_s)
            evidence = body(session, config, None) if with_slate else body(session, config)
        finally:
            # Child workers (group E) append to the same journal file; re-read it before judging conservation.
            session = replace(session, journal=OrderJournal(session.journal.path))
            try:
                teardown_session(session, close_handles=False)
                reconciled = reconcile_session(session, since_ms=since_ms, settle_wait_s=config.settle_wait_s)
                checks = evaluate_reconciliation(reconciled, config)
                primary_clean = not _read_positions(session.client) and not session.client.open_orders()
            finally:
                close_session_handles(session)
        return replace(
            evidence,
            payload={
                **evidence.payload,
                **({"r1_r6_pass": all(check.passed for check in checks)} if evidence.scenario_id == "C11" else {}),
                "conservation_passed": primary_clean and all(check.passed for check in checks),
                "reconciliation": checks,
                "primary_clean": primary_clean,
                "testnet_reset_suspected": reset_suspected,
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
    hypothesis_outcomes: tuple[tuple[str, str], ...] = (),
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
        hypothesis_outcomes=hypothesis_outcomes,
        conservation_passed=evidence.payload.get("conservation_passed") is True,
        proposed_action=proposed_action,
        proposal_rationale=proposal_rationale,
    )
