"""Lifecycle group (C) pure-evaluate invariants."""

from __future__ import annotations

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
    explicit = ev(
        "C8", final_equals_target=False, residual_explicit=True, has_obsolete_leg=True, conservation_passed=True
    )
    result = _SPECS["C8"].evaluate(explicit)
    assert result.verdict is Verdict.PASS
    assert "H7" in result.hypothesis_ids
    silent = ev("C8", final_equals_target=False, residual_explicit=False, has_obsolete_leg=True)
    assert _SPECS["C8"].evaluate(silent).verdict is Verdict.FAIL


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
