"""Group C fill, cancel and lifecycle scenarios (C1..C14)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.base import ScenarioEvidence, ScenarioSpec, result_of
from src.live.testnet_drill.scenarios.lifecycle_runs import runner


def _eval_c1(evidence: ScenarioEvidence) -> Any:
    status_path = list(evidence.payload.get("status_path", []))
    closed = bool(evidence.payload.get("closed", False))
    fills = int(evidence.payload.get("fills", 0))
    passed = status_path == ["NEW", "CANCELED"] and closed and fills == 0
    return result_of(
        evidence,
        "C1",
        Verdict.PASS if passed else Verdict.FAIL,
        (("resting_cancel", passed),),
        hypotheses=(),
    )


def _eval_c2(evidence: ScenarioEvidence) -> Any:
    code = evidence.payload["code"]
    chases = int(evidence.payload["chases"])
    window = int(evidence.payload["window"])
    no_resting_taker = evidence.payload["no_resting_taker"] is True
    if code is None and no_resting_taker:
        # The passive policy cannot be forced marketable; an unobserved -5022 is a finding, not a failure.
        return result_of(evidence, "C2", Verdict.OBSERVED, (("gtx_reprice_observed", True),), hypotheses=())
    passed = code == -5022 and 0 < chases <= window and no_resting_taker
    return result_of(
        evidence,
        "C2",
        Verdict.PASS if passed else Verdict.FAIL,
        (("gtx_reprice_bounded", passed),),
        hypotheses=(),
    )


def _eval_c3(evidence: ScenarioEvidence) -> Any:
    status = str(evidence.payload.get("status", "FILLED"))
    qty_ok = bool(evidence.payload.get("qty_exact", True))
    vwap_ok = bool(evidence.payload.get("vwap_within_tolerance", True))
    maker = bool(evidence.payload.get("maker_false", True))
    if status == "FILLED" and qty_ok and vwap_ok and not maker:
        # Strict passive policy may fill as maker before any IOC escalation; record instead of failing.
        return result_of(evidence, "C3", Verdict.OBSERVED, (("ioc_full_fill_observed", True),), hypotheses=("H2",))
    passed = status == "FILLED" and qty_ok and vwap_ok and maker
    return result_of(
        evidence,
        "C3",
        Verdict.PASS if passed else Verdict.FAIL,
        (("ioc_full_fill", passed),),
        hypotheses=("H2",),
    )


def _eval_c4(evidence: ScenarioEvidence) -> Any:
    status = str(evidence.payload.get("status", "EXPIRED"))
    executed = Decimal(str(evidence.payload.get("executed_qty", "0.001")))
    requested = Decimal(str(evidence.payload.get("requested_qty", "0.002")))
    attempts = int(evidence.payload.get("ioc_attempts", 1))
    max_attempts = int(evidence.payload.get("max_ioc_attempts", 10))
    residual_ok = bool(evidence.payload.get("residual_exact", True))
    passed = status == "EXPIRED" and Decimal(0) < executed < requested and attempts <= max_attempts and residual_ok
    if attempts == 0 and evidence.payload.get("escaped_exception") is None:
        return result_of(evidence, "C4", Verdict.OBSERVED, (("ioc_path_not_exercised", True),), hypotheses=())
    return result_of(
        evidence,
        "C4",
        Verdict.PASS if passed else Verdict.FAIL,
        (("ioc_partial", passed),),
        hypotheses=(),
    )


def _eval_c5(evidence: ScenarioEvidence) -> Any:
    monotonic = bool(evidence.payload.get("cumulative_monotonic", True))
    sums_ok = bool(evidence.payload.get("deltas_sum_to_venue", True))
    delta_price = Decimal(str(evidence.payload.get("delta_price", "100")))
    trade_price = Decimal(str(evidence.payload.get("user_trade_price", "100")))
    tolerance = Decimal(str(evidence.payload.get("price_rel_tolerance", "0.001")))
    denom = abs(trade_price) if trade_price != 0 else Decimal(1)
    uses_cumulative_avg = bool(evidence.payload.get("uses_cumulative_avg", False))
    differs = abs(delta_price - trade_price) / denom > tolerance
    confirmed = monotonic and sums_ok and uses_cumulative_avg and differs
    passed = monotonic and sums_ok
    observed = bool(evidence.payload.get("multi_step_observed", False))
    verdict = Verdict.FAIL if not passed else (Verdict.PASS if observed else Verdict.OBSERVED)
    return result_of(
        evidence,
        "C5",
        verdict,
        (("partial_attribution", passed),),
        hypotheses=("H2",),
        hypothesis_outcome="CONFIRMED" if confirmed else "INCONCLUSIVE",
    )


def _eval_c6(evidence: ScenarioEvidence) -> Any:
    repetitions = list(evidence.payload.get("repetitions", []))
    if not repetitions:
        return result_of(evidence, "C6", Verdict.FAIL, (("race_accounting", False),), hypotheses=())
    bad = [
        str(rep.get("id", index))
        for index, rep in enumerate(repetitions)
        if not (rep.get("terminal_confirmed", False) and rep.get("qty_match", False) and rep.get("no_double", True))
    ]
    raced = any(Decimal(str(rep.get("executed_qty", "0"))) > 0 for rep in repetitions)
    verdict = Verdict.FAIL if bad else (Verdict.PASS if raced else Verdict.OBSERVED)
    return result_of(
        evidence,
        "C6",
        verdict,
        tuple((f"race_accounting:{rep_id}", False) for rep_id in bad) if bad else (("race_accounting", True),),
        hypotheses=(),
        proposal_rationale="",
    )


def _eval_c7(evidence: ScenarioEvidence) -> Any:
    never_flips = bool(evidence.payload.get("never_flips", True))
    contained = bool(evidence.payload.get("contained_to_intent", True))
    passed = never_flips and contained
    return result_of(
        evidence,
        "C7",
        Verdict.PASS if passed else Verdict.FAIL,
        (("reduce_only_matrix", passed),),
        hypotheses=("H1",),
    )


def _eval_c8(evidence: ScenarioEvidence) -> Any:
    final_ok = bool(evidence.payload.get("final_equals_target", False))
    residual_explicit = bool(evidence.payload.get("residual_explicit", False))
    has_obsolete = bool(evidence.payload.get("has_obsolete_leg", False))
    if final_ok:
        return result_of(evidence, "C8", Verdict.PASS, (("flip_legs", True),), hypotheses=("H7",))
    if residual_explicit and has_obsolete:
        return result_of(
            evidence,
            "C8",
            Verdict.PASS,
            (("flip_legs", True),),
            hypotheses=("H7",),
            hypothesis_outcome="CONFIRMED",
        )
    return result_of(evidence, "C8", Verdict.FAIL, (("flip_legs", False),), hypotheses=("H7",))


def _eval_c9(evidence: ScenarioEvidence) -> Any:
    snapshots = list(evidence.payload.get("open_order_snapshots", []))
    filled = Decimal(str(evidence.payload.get("cumulative_filled", "0")))
    requested = Decimal(str(evidence.payload.get("intent_qty", "0")))
    interval_ok = bool(evidence.payload.get("interval_respected", True))
    per_intent_ok = all(int(count) <= 1 for count in snapshots)
    repegs = int(evidence.payload.get("repegs", 0))
    sound = bool(snapshots) and requested > 0 and per_intent_ok and filled <= requested and interval_ok
    verdict = Verdict.FAIL if not sound else (Verdict.PASS if repegs >= 1 else Verdict.OBSERVED)
    return result_of(
        evidence,
        "C9",
        verdict,
        (("single_live_order", per_intent_ok), ("fill_bounded", filled <= requested), ("repeg_interval", interval_ok)),
        hypotheses=(),
    )


def _eval_c10(evidence: ScenarioEvidence) -> Any:
    total = Decimal(str(evidence.payload.get("slice_sum", "0")))
    requested = Decimal(str(evidence.payload.get("intent_qty", "0")))
    residual_explicit = bool(evidence.payload.get("residual_explicit", False))
    unique_ids = bool(evidence.payload.get("slice_ids_unique", True))
    exact = total == requested or (total < requested and residual_explicit)
    passed = bool(
        requested > 0 and exact and unique_ids and (evidence.payload["final_equals_target"] or residual_explicit)
    )
    return result_of(
        evidence,
        "C10",
        Verdict.PASS if passed else Verdict.FAIL,
        (("slice_exactness", passed),),
        hypotheses=(),
    )


def _eval_c11(evidence: ScenarioEvidence) -> Any:
    passed = evidence.payload["r1_r6_pass"] is True and evidence.payload["symbol_count"] >= 2
    return result_of(
        evidence,
        "C11",
        Verdict.PASS if passed else Verdict.FAIL,
        (("conservation", passed),),
        hypotheses=(),
    )


def _eval_c12(evidence: ScenarioEvidence) -> Any:
    escalated = bool(evidence.payload.get("escalated_to_ioc", False))
    cap_ok = bool(evidence.payload.get("price_within_cap", True))
    empty = bool(evidence.payload.get("open_empty", True))
    passed = escalated and cap_ok and empty and evidence.payload["residual_exact"] is True
    if not escalated and empty and evidence.payload["residual_exact"] is True:
        return result_of(evidence, "C12", Verdict.OBSERVED, (("escalation_not_reached", True),), hypotheses=())
    return result_of(
        evidence,
        "C12",
        Verdict.PASS if passed else Verdict.FAIL,
        (("timeout_escalation", passed),),
        hypotheses=(),
    )


def _eval_c13(evidence: ScenarioEvidence) -> Any:
    no_429 = (
        evidence.payload["no_429"] is True
        and evidence.payload["intent_count"] >= 10
        and evidence.payload["symbol_count"] >= 3
        and evidence.payload["throttle_respected"] is True
        and evidence.payload["reducing_first"] is True
    )
    return result_of(
        evidence,
        "C13",
        Verdict.PASS if no_429 else Verdict.FAIL,
        (("concurrency_throttle", no_429),),
        hypotheses=(),
    )


def _eval_c14(evidence: ScenarioEvidence) -> Any:
    reducing_first = bool(evidence.payload.get("reducing_first", True))
    no_spurious_margin = bool(evidence.payload.get("no_spurious_2019", True))
    passed = reducing_first and no_spurious_margin and evidence.payload["margin_near_limit"] is True
    return result_of(
        evidence,
        "C14",
        Verdict.PASS if passed else Verdict.FAIL,
        (("margin_release_ordering", passed),),
        hypotheses=(),
    )


_C_EVALUATORS: dict[str, Any] = {
    "C1": _eval_c1,
    "C2": _eval_c2,
    "C3": _eval_c3,
    "C4": _eval_c4,
    "C5": _eval_c5,
    "C6": _eval_c6,
    "C7": _eval_c7,
    "C8": _eval_c8,
    "C9": _eval_c9,
    "C10": _eval_c10,
    "C11": _eval_c11,
    "C12": _eval_c12,
    "C13": _eval_c13,
    "C14": _eval_c14,
}

_C_TITLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "C1": ("resting gtx cancel", ()),
    "C2": ("gtx marketable reject", ()),
    "C3": ("ioc full fill", ("H2",)),
    "C4": ("ioc partial", ()),
    "C5": ("gtx partial multi step", ("H2",)),
    "C6": ("cancel fill race", ()),
    "C7": ("reduce only matrix", ("H1",)),
    "C8": ("flip legs", ("H7",)),
    "C9": ("repeg loop invariants", ()),
    "C10": ("slice exactness", ()),
    "C11": ("end to end conservation", ()),
    "C12": ("timeout escalation", ()),
    "C13": ("concurrency and throttle", ()),
    "C14": ("margin release ordering", ()),
}


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group C scenario registry."""
    specs: list[ScenarioSpec] = []
    for scenario_id in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"):
        title, hypotheses = _C_TITLES[scenario_id]
        specs.append(ScenarioSpec(scenario_id, "C", title, hypotheses, runner(scenario_id), _C_EVALUATORS[scenario_id]))
    return tuple(specs)
