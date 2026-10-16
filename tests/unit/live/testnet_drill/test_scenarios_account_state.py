"""Group F/G account-state and cycle pure-evaluate invariants."""

from __future__ import annotations

from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios import account_state, cycle_e2e
from src.live.testnet_drill.scenarios.base import ev

_F = {spec.scenario_id: spec for spec in account_state.scenario_specs()}
_G = {spec.scenario_id: spec for spec in cycle_e2e.scenario_specs()}


def test_foreign_order_de_risk() -> None:
    base = {"foreign_audited": True, "symbol_excluded": True, "halt_when_derisk_off": True, "derisk_when_enabled": True}
    cancelled = ev("F1", **base, foreign_cancelled=True, conservation_passed=True)
    assert _F["F1"].evaluate(cancelled).verdict is Verdict.FAIL
    passing = ev("F1", **base, foreign_cancelled=False, conservation_passed=True)
    assert _F["F1"].evaluate(passing).verdict is Verdict.PASS


def test_snapshot_atomicity_confirms_h9() -> None:
    result = _F["F4"].evaluate(ev("F4", inconsistency_rate=0.2, breach_would_exceed=True, conservation_passed=True))
    assert result.verdict is Verdict.OBSERVED
    assert result.outcome_for("H9") == "CONFIRMED"
    calm = _F["F4"].evaluate(ev("F4", inconsistency_rate=0.0, breach_would_exceed=False, conservation_passed=True))
    assert calm.outcome_for("H9") == "INCONCLUSIVE"


def test_funding_soak_skip() -> None:
    skipped = ev("F5", funding_events=0, funding_parity=False, sign_convention_ok=False, skipped=True)
    assert _F["F5"].evaluate(skipped).verdict is Verdict.SKIPPED
    wrong_sign = ev("F5", funding_events=3, funding_parity=True, sign_convention_ok=False, skipped=False,
                    conservation_passed=True)
    assert _F["F5"].evaluate(wrong_sign).verdict is Verdict.FAIL


def test_three_cycle_and_idempotent_rerun() -> None:
    passing_g1 = ev("G1", cycles_completed=3, r1_r6_pass=True, fills_agree=True, final_flat=True,
                    conservation_passed=True)
    assert _G["G1"].evaluate(passing_g1).verdict is Verdict.PASS
    failing_g2 = ev("G2", new_orders=1, decision_guard_honored=True, no_duplicate_fills=True)
    assert _G["G2"].evaluate(failing_g2).verdict is Verdict.FAIL
    passing_g2 = ev("G2", new_orders=0, decision_guard_honored=True, no_duplicate_fills=True,
                    conservation_passed=True)
    assert _G["G2"].evaluate(passing_g2).verdict is Verdict.PASS
    assert _G["G3"].evaluate(ev("G3", converged=True, no_double_execution=True,
                                conservation_passed=True)).verdict is Verdict.PASS


def test_registry_covers_f_and_g() -> None:
    assert set(_F) == {f"F{i}" for i in range(1, 8)}
    assert set(_G) == {f"G{i}" for i in range(1, 5)}
