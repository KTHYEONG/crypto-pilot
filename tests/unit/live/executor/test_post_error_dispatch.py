"""Diff-coverage for spec 18 post-error dispatch and unknown-submission primitives."""

from __future__ import annotations

import json
from decimal import Decimal
from http import HTTPStatus

from src.live.audit import AuditLog
from src.live.errors import TransientReadError, VenueError
from src.live.executor import (
    UNKNOWN_SUBMISSION_MISS_LIMIT,
    FeeSchedule,
    PassiveExecutionPolicy,
    _CycleFlags,
    _IntentRuntime,
    _adopt_unknown_submission,
    _declare_not_placed,
    _handle_post_error,
    _lookup_unknown_submission,
    _mark_posted,
    _record_order_posted,
    _resolve_unknown_at_exit,
    _resolve_unknown_submission,
    _settle_paper_post,
)
from src.live.filters import SymbolFilters
from src.live.order_journal import OrderJournal
from src.live.planner import OrderIntent


def _filters(symbol="AAAUSDT"):
    return SymbolFilters(
        symbol=symbol,
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("1"),
        max_qty=Decimal("100000"),
        quantity_precision=3,
        price_precision=2,
    )


def _intent(symbol="AAAUSDT", qty="1", reduce_only=False):
    return OrderIntent(
        symbol=symbol,
        side="BUY",
        quantity=Decimal(qty),
        reduce_only=reduce_only,
        target_qty=Decimal(qty),
        current_qty=Decimal("0"),
        client_order_prefix="20260914",
        leg_index=0,
        decision_price=Decimal("100.10"),
    )


def _policy(**kw):
    base = {"poll_interval_s": 1.0, "passive_deadline_s": 10.0, "window_deadline_s": 20.0}
    base.update(kw)
    return PassiveExecutionPolicy(**base)


def _terminals(tmp_path):
    path = tmp_path / "journal.jsonl"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("event") == "terminal":
            out[rec["client_order_id"]] = rec["status"]
    return out


def _terminal_lines(tmp_path, oid):
    path = tmp_path / "journal.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("event") == "terminal" and rec.get("client_order_id") == oid:
            rows.append(rec)
    return rows


def _wired_rt(tmp_path, order_id, *, reduce_only=False, qty="1", audit_name="audit.jsonl"):
    import pandas as pd

    journal = OrderJournal(tmp_path / "journal.jsonl")
    now = pd.Timestamp("2026-09-14T00:00:00Z")
    attempt = journal.begin_attempt(
        decision_time=now,
        run_id="test",
        mode="LIVE",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={},
        started_at=now,
    )
    journal.record_submit(
        order_id, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal(qty), reduce_only=reduce_only, leg_index=0,
    )
    audit = AuditLog(tmp_path / audit_name)
    rt = _IntentRuntime(
        intent=_intent(reduce_only=reduce_only, qty=qty),
        filters=_filters(),
        fee_schedule=FeeSchedule(2.0, 5.0),
        journal=journal,
    )
    rt.attempt_seq = attempt.attempt_seq
    rt.journal_enabled = True
    rt.journal_submitted.add(order_id)
    return rt, audit, journal


def _events(tmp_path, name="audit.jsonl"):
    path = tmp_path / name
    if not path.exists():
        return []
    return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]


def _verr(code, http_status=400):
    return VenueError("v", code=code, http_status=http_status, path="/fapi/v1/order", payload_digest="0" * 12)


def test_rate_limit_without_code(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-1")
    assert _handle_post_error(
        _verr(None, HTTPStatus.TOO_MANY_REQUESTS), rt, order_id="oid-1", now=10.0,
        policy=_policy(), audit=audit, cycle_flags=_CycleFlags(),
    ) is True
    assert _terminals(tmp_path)["oid-1"] == "NOT_PLACED"
    assert "order_rate_limited" in _events(tmp_path)
    assert rt.terminal_status is None


def test_long_backoff_code_uses_registry(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-1")
    assert _handle_post_error(
        _verr(-1003, 400), rt, order_id="oid-1", now=10.0,
        policy=_policy(), audit=audit, cycle_flags=None,
    ) is True
    assert _terminals(tmp_path)["oid-1"] == "NOT_PLACED"
    assert "order_rate_limited" in _events(tmp_path)


def test_reprice_chase_via_registry(tmp_path):
    for i, code in enumerate((-5022, -4131)):
        oid = f"oid-rp-{i}"
        rt, audit, _ = _wired_rt(tmp_path, oid, audit_name=f"audit-{i}.jsonl")
        assert _handle_post_error(
            _verr(code), rt, order_id=oid, now=10.0,
            policy=_policy(), audit=audit, cycle_flags=None,
        ) is True
        assert _terminals(tmp_path)[oid] == "NOT_PLACED"
        assert rt.chases == 1
        assert _events(tmp_path, f"audit-{i}.jsonl") == []


def test_intent_reject(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-1")
    assert _handle_post_error(
        _verr(-1013), rt, order_id="oid-1", now=10.0,
        policy=_policy(), audit=audit, cycle_flags=None,
    ) is True
    assert rt.terminal_status == "REJECTED"
    assert rt.reject_code == -1013
    assert _terminals(tmp_path)["oid-1"] == "REJECTED"


def test_margin_wait_reduce_only_rejects(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-1", reduce_only=True)
    assert _handle_post_error(
        _verr(-2019), rt, order_id="oid-1", now=10.0,
        policy=_policy(), audit=audit, cycle_flags=None,
    ) is True
    assert rt.terminal_status == "REJECTED"
    assert rt.margin_rejects == 0
    assert _terminals(tmp_path)["oid-1"] == "REJECTED"
    assert "intent_margin_wait" not in _events(tmp_path)


def test_margin_wait_bounded(tmp_path):
    rt, audit, journal = _wired_rt(tmp_path, "oid-w1", audit_name="a1.jsonl")
    policy = _policy(max_margin_rejects=1)
    assert _handle_post_error(
        _verr(-2019), rt, order_id="oid-w1", now=10.0,
        policy=policy, audit=audit, cycle_flags=None,
    ) is True
    assert rt.margin_wait_until == 10.0 + policy.margin_retry_s
    assert _terminals(tmp_path)["oid-w1"] == "NOT_PLACED"
    assert rt.margin_rejects == 1
    assert _events(tmp_path, "a1.jsonl") == ["intent_margin_wait"]
    journal.record_submit(
        "oid-w2", "AAAUSDT", journal.next_submit_seq(), attempt_seq=rt.attempt_seq,
        side="BUY", quantity=rt.intent.quantity, reduce_only=False, leg_index=0,
    )
    rt.journal_submitted.add("oid-w2")
    assert _handle_post_error(
        _verr(-2019), rt, order_id="oid-w2", now=rt.margin_wait_until,
        policy=policy, audit=audit, cycle_flags=None,
    ) is True
    assert rt.terminal_status == "REJECTED"
    assert rt.margin_rejects == 2
    assert _terminals(tmp_path)["oid-w2"] == "REJECTED"
    assert _events(tmp_path, "a1.jsonl") == ["intent_margin_wait", "intent_rejected"]


def test_freeze_sets_flag_before_reject(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-f1", audit_name="a1.jsonl")
    flags = _CycleFlags()
    assert _handle_post_error(
        _verr(-4400), rt, order_id="oid-f1", now=10.0,
        policy=_policy(), audit=audit, cycle_flags=flags,
    ) is True
    assert flags.risk_increase_frozen is True
    assert flags.freeze_code == -4400
    events = _events(tmp_path, "a1.jsonl")
    assert events.index("risk_increase_frozen") < events.index("intent_rejected")
    rt2, audit2, _ = _wired_rt(tmp_path, "oid-f2", audit_name="a2.jsonl")
    assert _handle_post_error(
        _verr(-4400), rt2, order_id="oid-f2", now=10.0,
        policy=_policy(), audit=audit2, cycle_flags=None,
    ) is True
    assert rt2.terminal_status == "REJECTED"


def test_unscoped_returns_false(tmp_path):
    cases = [(-1022, 400), (-9999, 400), (-1000, 400), (-2011, 400), (None, 0)]
    for i, (code, http) in enumerate(cases):
        oid = f"oid-u-{i}"
        rt, audit, _ = _wired_rt(tmp_path, oid, audit_name=f"au-{i}.jsonl")
        before = (rt.chases, rt.margin_rejects, rt.terminal_status)
        assert _handle_post_error(
            _verr(code, http), rt, order_id=oid, now=10.0,
            policy=_policy(), audit=audit, cycle_flags=_CycleFlags(),
        ) is False
        assert oid not in _terminals(tmp_path)
        assert (rt.chases, rt.margin_rejects, rt.terminal_status) == before
        assert _events(tmp_path, f"au-{i}.jsonl") == []


def test_mark_and_record_live_ioc(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-1")
    _mark_posted(rt, order_id="oid-1", price=Decimal("100"), post_qty=Decimal("1"), now=5.0, simulated=False)
    assert rt.paper_active is False
    assert rt.reported_executed == Decimal("0")
    assert rt.active_id == "oid-1"
    _record_order_posted(
        rt, audit, order_id="oid-1", time_in_force="IOC", price=Decimal("100"),
        post_qty=Decimal("1"), touch=(Decimal("99"), Decimal("101")), simulated=False,
    )
    assert rt.ioc_attempts == 1
    rec = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec["event"] == "order_posted"
    assert "simulated" not in rec
    assert rec["phase"] == "ioc"


def test_settle_paper_ioc_full_fill(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-1")
    _settle_paper_post(
        rt, audit, order_id="oid-1", time_in_force="IOC", price=Decimal("99"),
        post_qty=Decimal("1"), touch=(Decimal("99"), Decimal("101")), now=5.0,
    )
    assert rt.terminal_status == "FILLED"
    assert rt.active_id is None
    assert rt.ioc_attempts == 0
    assert _events(tmp_path).count("order_posted") == 0
    assert "paper_filled" in _events(tmp_path)


def test_settle_paper_gtx_rests(tmp_path):
    rt, audit, _ = _wired_rt(tmp_path, "oid-1")
    _settle_paper_post(
        rt, audit, order_id="oid-1", time_in_force="GTX", price=Decimal("99"),
        post_qty=Decimal("1"), touch=(Decimal("99"), Decimal("101")), now=5.0,
    )
    assert rt.active_id == "oid-1"
    assert rt.paper_active is True
    rec = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec["event"] == "order_posted"
    assert rec.get("simulated") is True
    assert "fill" not in _events(tmp_path)


class _QueryClient:
    def __init__(self, script, horizon=35.0):
        self.script = list(script)
        self.calls = 0
        self.unknown_outcome_horizon_s = horizon

    def query_order(self, symbol, oid):
        self.calls += 1
        outcome = self.script.pop(0)
        if outcome == "found":
            return {"status": "NEW", "executedQty": "0"}
        if outcome == "transient":
            raise TransientReadError("t", path="/fapi/v1/order", http_status=503, code=None, attempts=4)
        if outcome == "other":
            raise _verr(-1100)
        raise _verr(-2013)


def _unknown_rt(tmp_path, oid="u-1", misses=0, unresolved_at=0.0, tag="audit.jsonl"):
    import pandas as pd

    journal = OrderJournal(tmp_path / "journal.jsonl")
    now = pd.Timestamp("2026-09-14T00:00:00Z")
    attempt = journal.begin_attempt(
        decision_time=now, run_id="test", mode="LIVE", pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity", decision_marks={}, started_at=now,
    )
    journal.record_submit(
        oid, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("1"), reduce_only=False, leg_index=0,
    )
    audit = AuditLog(tmp_path / tag)
    rt = _IntentRuntime(intent=_intent(), filters=_filters(), fee_schedule=FeeSchedule(2.0, 5.0), journal=journal)
    rt.attempt_seq = attempt.attempt_seq
    rt.journal_enabled = True
    rt.journal_submitted.add(oid)
    rt.unresolved_id = oid
    rt.unresolved_price = Decimal("100")
    rt.unresolved_post_qty = Decimal("1")
    rt.unresolved_tif = "GTX"
    rt.unresolved_at = unresolved_at
    rt.unknown_misses = misses
    return rt, audit, journal


def test_lookup_branches(tmp_path):
    import pytest

    rt, _, _ = _unknown_rt(tmp_path, oid="u-l1", tag="al1.jsonl")
    assert _lookup_unknown_submission(_QueryClient(["found"]), rt) is True
    rt2, _, _ = _unknown_rt(tmp_path, oid="u-l2", tag="al2.jsonl")
    assert _lookup_unknown_submission(_QueryClient(["gone"]), rt2) is False
    rt3, _, _ = _unknown_rt(tmp_path, oid="u-l3", tag="al3.jsonl")
    with pytest.raises(VenueError) as excinfo:
        _lookup_unknown_submission(_QueryClient(["other"]), rt3)
    assert excinfo.value.code == -1100
    rt4, _, _ = _unknown_rt(tmp_path, oid="u-l4", tag="al4.jsonl")
    with pytest.raises(TransientReadError):
        _lookup_unknown_submission(_QueryClient(["transient"]), rt4)


def test_adopt_and_declare(tmp_path):
    rt, audit, _ = _unknown_rt(tmp_path, oid="u-a1", misses=1)
    _adopt_unknown_submission(rt, 9.0, audit)
    assert rt.active_id == "u-a1"
    assert rt.unresolved_id is None
    rt2, audit2, _ = _unknown_rt(tmp_path, oid="u-a2", tag="a2.jsonl")
    _declare_not_placed(rt2, audit2)
    assert _terminals(tmp_path)["u-a2"] == "NOT_PLACED"
    assert rt2.unresolved_id is None
    assert rt2.unknown_misses == 0
    rt2.unresolved_id = "u-a2"
    _declare_not_placed(rt2, audit2)
    assert len(_terminal_lines(tmp_path, "u-a2")) == 1


def test_tick_resolution_paths(tmp_path):
    rt, audit, _ = _unknown_rt(tmp_path, oid="u-t1")
    assert _resolve_unknown_submission(_QueryClient(["transient"]), rt, 100.0, audit) is False
    assert rt.unknown_misses == 0
    rt2, audit2, _ = _unknown_rt(tmp_path, oid="u-t2", tag="a2.jsonl")
    assert _resolve_unknown_submission(_QueryClient(["found"]), rt2, 100.0, audit2) is True
    assert rt2.active_id == "u-t2"
    rt3, audit3, _ = _unknown_rt(tmp_path, oid="u-t3", unresolved_at=0.0, tag="a3.jsonl")
    assert _resolve_unknown_submission(_QueryClient(["gone"], horizon=35.0), rt3, 40.0, audit3) is False
    assert rt3.unknown_misses == 1
    assert "u-t3" not in _terminals(tmp_path)
    assert UNKNOWN_SUBMISSION_MISS_LIMIT == 2
    assert _resolve_unknown_submission(_QueryClient(["gone"], horizon=35.0), rt3, 40.0, audit3) is True
    assert _terminals(tmp_path)["u-t3"] == "NOT_PLACED"


def test_exit_horizon_only_and_left_unresolved(tmp_path):
    rt, audit, _ = _unknown_rt(tmp_path, oid="u-e1", unresolved_at=0.0)
    client = _QueryClient(["gone"], horizon=35.0)
    out = _resolve_unknown_at_exit(client, rt, audit, lambda: 99.0, lambda s: None, 40.0)
    assert out == 40.0
    assert client.calls == 1
    assert _terminals(tmp_path)["u-e1"] == "NOT_PLACED"
    assert rt.unknown_misses == 0
    rt2, audit2, _ = _unknown_rt(tmp_path, oid="u-e2", unresolved_at=0.0, tag="a2.jsonl")
    out2 = _resolve_unknown_at_exit(
        _QueryClient(["gone", "gone"], horizon=30.0), rt2, audit2, lambda: 1.0, lambda s: None, 1.0,
    )
    assert out2 is None
    assert "order_unknown_left_unresolved" in _events(tmp_path, "a2.jsonl")
    assert rt2.unresolved_id == "u-e2"
    assert "u-e2" not in _terminals(tmp_path)
    rt3, audit3, _ = _unknown_rt(tmp_path, oid="u-e3", unresolved_at=0.0, tag="a3.jsonl")
    out3 = _resolve_unknown_at_exit(_QueryClient(["found"], horizon=35.0), rt3, audit3, lambda: 99.0, lambda s: None, 40.0)
    assert out3 == 40.0
    assert rt3.active_id == "u-e3"


def test_tick_miss_limit_without_horizon_stays_undecided(tmp_path):
    rt, audit, _ = _unknown_rt(tmp_path, oid="u-t4", misses=1, unresolved_at=0.0)
    assert _resolve_unknown_submission(_QueryClient(["gone"], horizon=35.0), rt, 10.0, audit) is False
    assert rt.unknown_misses == 2
    assert "u-t4" not in _terminals(tmp_path)


def test_exit_second_lookup_branches(tmp_path):
    clock_state = [1.0]

    def clock():
        return clock_state[0]

    def sleep_fn(seconds):
        clock_state[0] += seconds

    rt, audit, _ = _unknown_rt(tmp_path, oid="u-s1", unresolved_at=0.0)
    out = _resolve_unknown_at_exit(_QueryClient(["gone", "found"], horizon=30.0), rt, audit, clock, sleep_fn, 1.0)
    assert out == 30.0
    assert rt.active_id == "u-s1"
    rt2, audit2, _ = _unknown_rt(tmp_path, oid="u-s2", unresolved_at=0.0, tag="a2.jsonl")

    def jump(_seconds):
        clock_state[0] = 101.0

    out2 = _resolve_unknown_at_exit(
        _QueryClient(["gone", "gone"], horizon=30.0), rt2, audit2, lambda: 101.0, jump, 1.0,
    )
    assert out2 == 101.0
    assert _terminals(tmp_path)["u-s2"] == "NOT_PLACED"


def test_finalize_unknown_paths(tmp_path):
    from src.live.executor import _finalize

    rt, audit, _ = _unknown_rt(tmp_path, oid="u-f1", unresolved_at=0.0)
    _finalize(_QueryClient(["gone"], horizon=35.0), [rt], audit, lambda: 40.0, lambda s: None)
    assert rt.unresolved_id is None
    assert rt.terminal_status == "RESIDUAL"
    assert _terminals(tmp_path)["u-f1"] == "NOT_PLACED"
    rt2, audit2, _ = _unknown_rt(tmp_path, oid="u-f2", unresolved_at=0.0, tag="a2.jsonl")
    _finalize(
        _QueryClient(["gone", "gone"], horizon=30.0), [rt2], audit2, lambda: 1.0, lambda s: None,
    )
    assert rt2.unresolved_id == "u-f2"
    assert rt2.terminal_status is None


def test_post_error_dispatch_has_no_venue_code_literals():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(_handle_post_error))
    assert not any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "HTTPStatus"
        and node.attr == "TOO_MANY_REQUESTS"
        for node in ast.walk(tree)
    )


def test_finalize_unknown_transient_propagates(tmp_path):
    import pytest

    from src.live.executor import _finalize

    rt, audit, _ = _unknown_rt(tmp_path)
    with pytest.raises(TransientReadError):
        _finalize(_QueryClient(["transient"]), [rt], audit, lambda: 40.0, lambda s: None)
    assert rt.unresolved_id == "u-1"
    assert rt.terminal_status is None
    assert _terminals(tmp_path) == {}


def test_finalize_adopted_order_requires_confirmed_cancel(tmp_path):
    import pytest

    from src.live.executor import CancelNotConfirmed, _finalize

    class Client(_QueryClient):
        def cancel_order(self, symbol, oid):
            assert symbol == "AAAUSDT"
            assert oid == "u-1"
            raise _verr(-2011)

    rt, audit, _ = _unknown_rt(tmp_path)
    client = Client(["found", "found"])
    with pytest.raises(CancelNotConfirmed):
        _finalize(client, [rt], audit, lambda: 40.0, lambda s: None)
    assert client.calls == 2
    assert rt.active_id == "u-1"
    assert rt.unresolved_id is None
    assert rt.terminal_status is None
    assert _events(tmp_path) == ["order_unknown_adopted", "order_cancel_unconfirmed"]
    assert _terminals(tmp_path) == {}
