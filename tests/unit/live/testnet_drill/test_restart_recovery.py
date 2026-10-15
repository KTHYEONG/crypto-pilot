"""Group E restart-recovery pure-evaluate invariants."""

from __future__ import annotations

import pytest

from src.live.testnet_drill.faults import FaultAction, FaultRule
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios import restart_recovery
from src.live.testnet_drill.scenarios.base import ev

_SPECS = {spec.scenario_id: spec for spec in restart_recovery.scenario_specs()}


def test_idempotent_recovery() -> None:
    passing = ev("E8", recovery_passes=2, recovered_fill_count=1, single_pass_fill_count=1,
                 idempotent=True, conservation_passed=True)
    assert _SPECS["E8"].evaluate(passing).verdict is Verdict.PASS
    failing = ev("E8", recovery_passes=2, recovered_fill_count=2, single_pass_fill_count=1, idempotent=False)
    assert _SPECS["E8"].evaluate(failing).verdict is Verdict.FAIL


def test_single_booking() -> None:
    failing = ev("E4", recovered_once=True, no_double_count=False,
                 kinds="execution+recovered", as_execution=True, as_recovered=True)
    assert _SPECS["E4"].evaluate(failing).verdict is Verdict.FAIL


def test_not_found_terminal() -> None:
    passing = ev("E1", exit_code=137, query_code=-2013, terminal_status="NOT_FOUND", venue_exposure=0,
                 orphan_count=0, conservation_passed=True)
    assert _SPECS["E1"].evaluate(passing).verdict is Verdict.PASS


def test_ledger_watermark_commit() -> None:
    passing = ev("E5", journal_ahead_before=True, ledger_equal_after=True, single_commit=True,
                 conservation_passed=True)
    assert _SPECS["E5"].evaluate(passing).verdict is Verdict.PASS


def test_tail_corruption_handled() -> None:
    tail = ev("E10", tail_corrupt_at="tail", truncated_tail_ok=True, mid_file_fails_closed=True,
              conservation_passed=True)
    assert _SPECS["E10"].evaluate(tail).verdict is Verdict.PASS
    lost_tail = ev("E10", tail_corrupt_at="tail", truncated_tail_ok=False, mid_file_fails_closed=True,
                   conservation_passed=True)
    assert _SPECS["E10"].evaluate(lost_tail).verdict is Verdict.FAIL
    mid_open = ev("E10", tail_corrupt_at="tail", truncated_tail_ok=True, mid_file_fails_closed=False,
                  conservation_passed=True)
    assert _SPECS["E10"].evaluate(mid_open).verdict is Verdict.FAIL


def test_parent_never_kills_itself(tmp_path, monkeypatch) -> None:
    from src.live.testnet_drill.guard import DrillRefused
    from src.live.testnet_drill.session import open_drill_session
    from tests.unit.live.testnet_drill._stubs import make_config, make_settings, patch_drill_root

    root = patch_drill_root(monkeypatch, tmp_path)
    rules = (FaultRule(method="POST", path="/fapi/v1/order", action=FaultAction.KILL_AFTER_DELIVERY),)
    with pytest.raises(DrillRefused, match="parent_kill"):
        open_drill_session(make_settings(), make_config(), run_id="e9x", rules=rules, workspace=root / "ws")
    assert not (root / "ws").exists()


def test_runtime_injection_refuses_kill_rules(exchange_session) -> None:
    from src.live.testnet_drill.guard import DrillRefused

    session, _, _ = exchange_session()
    with pytest.raises(DrillRefused, match="parent_kill"), session.transport.inject(
        [FaultRule(method="POST", path="/fapi/v1/order", action=FaultAction.KILL_BEFORE_DELIVERY)]
    ):
        pass


def test_registry_covers_all_e_scenarios() -> None:
    assert set(_SPECS) == {f"E{i}" for i in range(1, 11)}


def test_restart_anchor_is_production_entry() -> None:
    assert restart_recovery.RESTART_ANCHOR == "src.live.runner:_reconcile_pre_trade"
    import src.live.runner as runner

    assert hasattr(runner, "_reconcile_pre_trade")
