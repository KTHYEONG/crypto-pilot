"""Lifecycle group (C) pure-evaluate invariants."""

from __future__ import annotations

import pytest

from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios import lifecycle
from src.live.testnet_drill.scenarios.base import ev

_SPECS = {spec.scenario_id: spec for spec in lifecycle.scenario_specs()}


def test_single_live_order_per_intent() -> None:
    failing = ev(
        "C9", open_order_snapshots=[1, 2, 1], cumulative_filled="0.1", intent_qty="0.2", interval_respected=True, repegs=1
    )
    assert _SPECS["C9"].evaluate(failing).verdict is Verdict.FAIL
    passing = ev(
        "C9",
        open_order_snapshots=[1, 1, 0],
        cumulative_filled="0.1",
        intent_qty="0.2",
        interval_respected=True,
        repegs=2,
        conservation_passed=True,
    )
    assert _SPECS["C9"].evaluate(passing).verdict is Verdict.PASS
    unexercised = ev(
        "C9",
        open_order_snapshots=[1, 0],
        cumulative_filled="0.2",
        intent_qty="0.2",
        interval_respected=True,
        repegs=0,
        conservation_passed=True,
    )
    assert _SPECS["C9"].evaluate(unexercised).verdict is Verdict.OBSERVED


def test_partial_price_attribution_confirms_h2() -> None:
    evidence = ev(
        "C5",
        cumulative_monotonic=True,
        deltas_sum_to_venue=True,
        multi_step_observed=True,
        delta_price="100",
        user_trade_price="101",
        price_rel_tolerance="0.001",
        uses_cumulative_avg=True,
        conservation_passed=True,
    )
    result = _SPECS["C5"].evaluate(evidence)
    assert result.verdict is Verdict.PASS
    assert result.hypothesis_outcome == "CONFIRMED"


def test_race_accounting() -> None:
    reps = [
        {"id": f"r{i}", "terminal_confirmed": True, "qty_match": True, "no_double": True, "executed_qty": "0.1"}
        for i in range(20)
    ]
    assert _SPECS["C6"].evaluate(ev("C6", repetitions=reps, conservation_passed=True)).verdict is Verdict.PASS
    quiet = [{**rep, "executed_qty": "0"} for rep in reps]
    assert _SPECS["C6"].evaluate(ev("C6", repetitions=quiet, conservation_passed=True)).verdict is Verdict.OBSERVED
    bad = list(reps)
    bad[7] = {"id": "r7", "terminal_confirmed": True, "qty_match": False, "no_double": True, "executed_qty": "0.1"}
    result = _SPECS["C6"].evaluate(ev("C6", repetitions=bad))
    assert result.verdict is Verdict.FAIL


def test_flip_residual_explicit() -> None:
    sequenced = ev(
        "C8",
        final_equals_target=True,
        residual_explicit=False,
        has_obsolete_leg=False,
        open_posted_after_close_filled=True,
        blocked_open_leg=False,
        open_leg_posts=1,
        position_never_crossed_zero=False,
        final_position="0.01",
        target_qty="0.01",
        conservation_passed=True,
    )
    result = _SPECS["C8"].evaluate(sequenced)
    assert result.verdict is Verdict.PASS
    assert result.outcome_for("H7") == "REFUTED"
    legacy = ev(
        "C8",
        final_equals_target=False,
        residual_explicit=True,
        has_obsolete_leg=True,
        open_posted_after_close_filled=False,
        blocked_open_leg=False,
        open_leg_posts=1,
        position_never_crossed_zero=False,
        final_position="0.005",
        target_qty="0.01",
        conservation_passed=True,
    )
    assert _SPECS["C8"].evaluate(legacy).verdict is Verdict.FAIL
    silent = ev(
        "C8",
        final_equals_target=False,
        residual_explicit=False,
        has_obsolete_leg=True,
        open_posted_after_close_filled=False,
        blocked_open_leg=False,
        open_leg_posts=1,
        position_never_crossed_zero=False,
        final_position="0.005",
        target_qty="0.01",
    )
    assert _SPECS["C8"].evaluate(silent).verdict is Verdict.FAIL


def test_c8_blocked_flip_passes_only_when_explicit() -> None:
    blocked = ev(
        "C8",
        final_equals_target=False,
        residual_explicit=True,
        has_obsolete_leg=False,
        open_posted_after_close_filled=False,
        blocked_open_leg=True,
        open_leg_posts=0,
        position_never_crossed_zero=True,
        final_position="0.01",
        target_qty="-0.01",
        conservation_passed=True,
    )
    assert _SPECS["C8"].evaluate(blocked).verdict is Verdict.PASS
    implicit = ev(
        "C8",
        final_equals_target=False,
        residual_explicit=True,
        has_obsolete_leg=False,
        open_posted_after_close_filled=False,
        blocked_open_leg=False,
        open_leg_posts=0,
        position_never_crossed_zero=True,
        final_position="0.01",
        target_qty="-0.01",
        conservation_passed=True,
    )
    result = _SPECS["C8"].evaluate(implicit)
    assert result.verdict is Verdict.FAIL
    assert result.outcome_for("H7") == "CONFIRMED"


@pytest.mark.parametrize(
    ("final", "never_crossed", "posts", "passed"),
    [("0", True, 0, True), ("0.01", False, 0, False), ("0.01", True, 1, False)],
)
def test_c8_blocked_position_path(final, never_crossed, posts, passed) -> None:
    evidence = ev(
        "C8", final_equals_target=False, residual_explicit=True, has_obsolete_leg=False,
        open_posted_after_close_filled=False, blocked_open_leg=True, open_leg_posts=posts,
        final_position=final, target_qty="-0.01", position_never_crossed_zero=never_crossed,
        conservation_passed=True,
    )
    assert (_SPECS["C8"].evaluate(evidence).verdict is Verdict.PASS) is passed


def test_slice_sum_exactness() -> None:
    failing = ev("C10", slice_sum="0.09", intent_qty="0.1", residual_explicit=False, slice_ids_unique=True)
    assert _SPECS["C10"].evaluate(failing).verdict is Verdict.FAIL
    passing = ev(
        "C10",
        slice_sum="0.1",
        intent_qty="0.1",
        residual_explicit=False,
        slice_ids_unique=True,
        final_equals_target=True,
        conservation_passed=True,
    )
    assert _SPECS["C10"].evaluate(passing).verdict is Verdict.PASS


def test_window_closure() -> None:
    failing = ev("C12", escalated_to_ioc=True, price_within_cap=True, open_empty=False)
    assert _SPECS["C12"].evaluate(failing).verdict is Verdict.FAIL
    passing = ev(
        "C12",
        escalated_to_ioc=True,
        price_within_cap=True,
        open_empty=True,
        residual_exact=True,
        conservation_passed=True,
    )
    assert _SPECS["C12"].evaluate(passing).verdict is Verdict.PASS


def test_registry_covers_all_c_scenarios() -> None:
    assert set(_SPECS) == {f"C{i}" for i in range(1, 15)}


def test_gtx_reprice_never_rests_as_taker() -> None:
    base = {"code": -5022, "chases": 2, "window": 3, "conservation_passed": True}
    assert _SPECS["C2"].evaluate(ev("C2", **base, no_resting_taker=True)).verdict is Verdict.PASS
    assert _SPECS["C2"].evaluate(ev("C2", **base, no_resting_taker=False)).verdict is Verdict.FAIL
    assert _SPECS["C2"].evaluate(ev("C2", **{**base, "code": None}, no_resting_taker=True)).verdict is Verdict.OBSERVED
    assert _SPECS["C2"].evaluate(ev("C2", **{**base, "code": None}, no_resting_taker=False)).verdict is Verdict.FAIL


def test_passive_maker_fill_in_c3_is_observed_not_passed() -> None:
    base = {"status": "FILLED", "qty_exact": True, "vwap_within_tolerance": True, "conservation_passed": True}
    assert _SPECS["C3"].evaluate(ev("C3", **base, maker_false=True)).verdict is Verdict.PASS
    assert _SPECS["C3"].evaluate(ev("C3", **base, maker_false=False)).verdict is Verdict.OBSERVED
    assert _SPECS["C3"].evaluate(ev("C3", **{**base, "qty_exact": False}, maker_false=False)).verdict is Verdict.FAIL
