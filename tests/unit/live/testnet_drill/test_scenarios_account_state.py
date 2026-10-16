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
    assert _G["G3"].evaluate(ev("G3", converged=True, no_double_execution=True, silent_shortfall=False,
                                conservation_passed=True)).verdict is Verdict.PASS
    assert _G["G3"].evaluate(ev("G3", converged=False, no_double_execution=True, silent_shortfall=True,
                                conservation_passed=True)).verdict is Verdict.FAIL


def test_registry_covers_f_and_g() -> None:
    assert set(_F) == {f"F{i}" for i in range(1, 8)}
    assert set(_G) == {f"G{i}" for i in range(1, 5)}


def test_g3_silent_shortfall_fails_without_explicit_residual() -> None:
    silent = ev("G3", converged=False, no_double_execution=True, silent_shortfall=True,
                conservation_passed=True)
    assert _G["G3"].evaluate(silent).verdict is Verdict.FAIL


def test_g3_converged_with_blocked_outcome() -> None:
    from decimal import Decimal

    from src.live.executor import ExecutionOutcome
    from src.live.testnet_drill.scenarios.cycle_e2e import _converged

    class _Session:
        def __init__(self) -> None:
            self.filters = {"AAAUSDT": type("_F", (), {"step_size": Decimal("0.001")})()}

    target = {"AAAUSDT": Decimal("-0.245")}
    venue = {"AAAUSDT": Decimal("-0.123")}
    blocked = type("_R", (), {"outcomes": (
        ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("0"), unfilled_qty=Decimal("0.122"),
                         avg_fill_price=None, chases=0, status="RESIDUAL",
                         block_reason="blocked_by_close_leg"),
    )})()
    assert _converged(_Session(), target, venue, blocked) is True
    plain = type("_R", (), {"outcomes": (
        ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("0.1"), unfilled_qty=Decimal("0"),
                         avg_fill_price=Decimal("100"), chases=0, status="FILLED"),
    )})()
    assert _converged(_Session(), target, venue, plain) is False
    obsolete = type("_R", (), {"outcomes": (
        ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("0"), unfilled_qty=Decimal("0.122"),
                         avg_fill_price=None, chases=0, status="OBSOLETE"),
    )})()
    assert _converged(_Session(), target, venue, obsolete) is False
    residual = type("_R", (), {"outcomes": (
        ExecutionOutcome(symbol="BBB", filled_qty=Decimal("1"), unfilled_qty=Decimal("1"),
                         avg_fill_price=None, chases=0, status="RESIDUAL"),
        ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("0.1"), unfilled_qty=Decimal("0.022"),
                         avg_fill_price=None, chases=0, status="RESIDUAL"),
    )})()
    from types import SimpleNamespace

    residual.intents = (SimpleNamespace(symbol="BBB", quantity=Decimal("2")),
                        SimpleNamespace(symbol="AAAUSDT", quantity=Decimal("0.122")))
    assert _converged(_Session(), target, venue, residual) is True
    residual.intents[1].quantity = Decimal("0.123")
    assert _converged(_Session(), target, venue, residual) is False


def test_g3_run_records_silent_shortfall(monkeypatch) -> None:
    from decimal import Decimal
    from pathlib import Path

    import pandas as pd

    import src.live.testnet_drill.scenarios.cycle_e2e as g3
    from src.live.executor import ExecutionOutcome
    from src.live.testnet_drill.worker import KILL_EXIT_CODE

    step = type("_F", (), {"step_size": Decimal("0.001")})()

    class _Client:
        pass

    class _Session:
        def __init__(self, outcomes) -> None:
            self.filters = {"AAAUSDT": step}
            self.client = _Client()
            self._outcomes = outcomes

    target = {"AAAUSDT": Decimal("-0.245")}
    venue = {"AAAUSDT": Decimal("-0.123")}
    days = [pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-02", tz="UTC"),
            pd.Timestamp("2020-01-03", tz="UTC")]

    def _run_case(outcomes, *, expect_converged: bool, expect_silent: bool):
        rerun = type("_R", (), {"status": "COMPLETE", "outcomes": outcomes})()
        first = type("_R", (), {"status": "COMPLETE"})()
        again = type("_R", (), {"reason": "already_executed"})()
        calls = {"n": 0}

        def _fake_cycle(session, weights_path, day):
            calls["n"] += 1
            if calls["n"] == 1:
                return first, None
            if calls["n"] == 2:
                return rerun, None
            return again, None

        monkeypatch.setattr(g3, "_script", lambda session: (Path("w.parquet"), days))
        monkeypatch.setattr(g3, "_cycle", _fake_cycle)
        monkeypatch.setattr(g3, "_fresh", lambda session: session)
        monkeypatch.setattr(g3, "_target_positions", lambda session, path, day: dict(target))
        monkeypatch.setattr("src.live.testnet_drill.session._read_positions", lambda client: dict(venue))
        monkeypatch.setattr("src.live.testnet_drill.scenarios.restart_recovery.spawn",
                            lambda session, payload, extra: type("_Run", (), {"exit_code": KILL_EXIT_CODE})())
        evidence = g3._run_g3(_Session(outcomes), object())
        assert evidence.payload["converged"] is expect_converged
        assert evidence.payload["silent_shortfall"] is expect_silent

    _run_case((), expect_converged=False, expect_silent=True)
    _run_case((ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("0"), unfilled_qty=Decimal("0.122"),
                                avg_fill_price=None, chases=0, status="RESIDUAL",
                                block_reason="blocked_by_close_leg"),),
              expect_converged=True, expect_silent=False)
