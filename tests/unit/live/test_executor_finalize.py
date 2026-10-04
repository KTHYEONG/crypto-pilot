"""Unknown submission resolution and execution finalization horizon guards."""

from __future__ import annotations

from src.live.executor import FeeSchedule

def _test_attempt(journal):
    """Create one v2 attempt on ``journal`` for tests (Spec 03 both-or-none contract)."""
    import pandas as pd
    from decimal import Decimal

    now = pd.Timestamp("2026-09-14T00:00:00Z")
    return journal.begin_attempt(
        decision_time=now,
        run_id="test",
        mode="LIVE",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={},
        started_at=now,
    )

def _exec_filters(symbol):
    from decimal import Decimal
    from src.live.filters import SymbolFilters

    return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                         min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                         quantity_precision=3, price_precision=2)

def _exec_intent(symbol):
    from decimal import Decimal
    from src.live.planner import OrderIntent

    return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                       target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                       leg_index=0, decision_price=Decimal("100.10"))


def test_execute_intents_unknown_submission_is_adopted_not_resent(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent
    from src.live.rest import OrderStatusUnknown

    def _filters(symbol):
        return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                             min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                             quantity_precision=3, price_precision=2)

    def _intent(symbol):
        return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                           target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                           leg_index=0, decision_price=Decimal("100.10"))

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=10.0, window_deadline_s=20.0)
    audit_path = tmp_path / "exec_audit.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(seconds):
        clock_state[0] += seconds

    def _events():
        return [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]

    class _Client:
        def __init__(self):
            self.posted: list[dict] = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.posted.append(params)
            raise OrderStatusUnknown("timeout", path="/fapi/v1/order", http_status=0, code=None)

        def cancel_order(self, symbol, orig_client_order_id):
            return {}

        def query_order(self, symbol, orig_client_order_id):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()

    outcomes = execute_intents(client, [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, policy, audit,
                               clock, sleep_fn, journal=journal, attempt=attempt)

    order_id = client.posted[0]["newClientOrderId"]
    assert len(client.posted) == 1
    assert order_id.startswith("mh20260914-")
    assert order_id.endswith("-0-0")
    assert outcomes[0].status == "FILLED"
    assert outcomes[0].filled_qty == Decimal("1")
    reloaded = OrderJournal(tmp_path / "journal.jsonl")
    assert reloaded.next_submit_seq() == 1
    assert reloaded.observed_qty(order_id) == Decimal("1")
    assert "order_status_unknown" in _events()
    assert "order_unknown_adopted" in _events()


def test_execute_intents_unknown_submission_confirmed_absent_posts_new_seq(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, UNKNOWN_SUBMISSION_MISS_LIMIT, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent
    from src.live.rest import OrderStatusUnknown

    def _filters(symbol):
        return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                             min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                             quantity_precision=3, price_precision=2)

    def _intent(symbol):
        return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                           target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                           leg_index=0, decision_price=Decimal("100.10"))

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=50.0, window_deadline_s=200.0)
    audit_path = tmp_path / "exec_audit.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(seconds):
        clock_state[0] += seconds

    def _events():
        return [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]

    class _Client:
        unknown_outcome_horizon_s = 35.0

        def __init__(self):
            self.posted: list[dict] = []
            self.lookups: list[str] = []
            self.post_times: list[float] = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.post_times.append(clock_state[0])
            self.posted.append(params)
            if len(self.posted) == 1:
                raise OrderStatusUnknown("503", path="/fapi/v1/order", http_status=503, code=None)
            return {"orderId": 2}

        def cancel_order(self, symbol, orig_client_order_id):
            return {}

        def query_order(self, symbol, orig_client_order_id):
            self.lookups.append(orig_client_order_id)
            if orig_client_order_id == self.posted[0]["newClientOrderId"]:
                raise VenueError("missing", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()

    outcomes = execute_intents(client, [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, policy, audit,
                               clock, sleep_fn, journal=journal, attempt=attempt)

    first_id = client.posted[0]["newClientOrderId"]
    second_id = client.posted[1]["newClientOrderId"]
    assert UNKNOWN_SUBMISSION_MISS_LIMIT == 2
    assert len(client.posted) == 2
    assert first_id.endswith("-0-0")
    assert second_id.endswith("-0-1")
    # Not placed only after the horizon: second post at least 35 s after the first.
    assert client.post_times[1] - client.post_times[0] >= 35.0
    assert client.lookups.count(first_id) >= 2
    assert outcomes[0].status == "FILLED"
    assert "order_unknown_not_placed" in _events()


def test_execute_intents_abort_resolves_unknown_submission_at_exit(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent
    from src.live.rest import OrderStatusUnknown

    def _filters(symbol):
        return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                             min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                             quantity_precision=3, price_precision=2)

    def _intent(symbol):
        return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                           target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                           leg_index=0, decision_price=Decimal("100.10"))

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=10.0, window_deadline_s=20.0)
    audit_path = tmp_path / "exec_audit.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(seconds):
        clock_state[0] += seconds

    def _events():
        return [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]

    import pytest

    class _Client:
        def __init__(self):
            self.posted: list[dict] = []
            self.cancels: list[str] = []

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            self.posted.append(params)
            if params["symbol"] == "AAAUSDT":
                raise OrderStatusUnknown("timeout", path="/fapi/v1/order", http_status=0, code=None)
            raise VenueError("unknown rejection", code=-9999, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            return {"status": "PARTIALLY_FILLED", "executedQty": "0.25", "avgPrice": "100.00"}

    client = _Client()
    filters = {"AAAUSDT": _filters("AAAUSDT"), "BBBUSDT": _filters("BBBUSDT")}

    with pytest.raises(VenueError) as exc_info:
        execute_intents(client, [_intent("AAAUSDT"), _intent("BBBUSDT")], filters, policy, audit,
                        clock, sleep_fn, journal=journal, attempt=attempt)

    aaa_id = client.posted[0]["newClientOrderId"]
    assert client.cancels == [aaa_id]
    partial = {o.symbol: o for o in exc_info.value.partial_outcomes}
    assert partial["AAAUSDT"].filled_qty == Decimal("0.25")


def test_execute_intents_window_end_drops_unknown_submission_confirmed_absent(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent
    from src.live.rest import OrderStatusUnknown

    def _filters(symbol):
        return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                             min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                             quantity_precision=3, price_precision=2)

    def _intent(symbol):
        return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                           target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                           leg_index=0, decision_price=Decimal("100.10"))

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=10.0, window_deadline_s=20.0)
    audit_path = tmp_path / "exec_audit.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(seconds):
        clock_state[0] += seconds

    def _events():
        return [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]

    short_policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=0.5, window_deadline_s=1.0)

    class _Client:
        def __init__(self):
            self.posted: list[dict] = []
            self.cancels: list[str] = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.posted.append(params)
            raise OrderStatusUnknown("timeout", path="/fapi/v1/order", http_status=0, code=None)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            raise VenueError("missing", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    client = _Client()

    outcomes = execute_intents(client, [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, short_policy, audit,
                               clock, sleep_fn, journal=journal, attempt=attempt)

    assert len(client.posted) == 1
    assert client.cancels == []
    assert outcomes[0].status == "RESIDUAL"


def test_execute_intents_unknown_lookup_failure_propagates(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent
    from src.live.rest import OrderStatusUnknown

    def _filters(symbol):
        return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                             min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                             quantity_precision=3, price_precision=2)

    def _intent(symbol):
        return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                           target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                           leg_index=0, decision_price=Decimal("100.10"))

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=10.0, window_deadline_s=20.0)
    audit_path = tmp_path / "exec_audit.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(seconds):
        clock_state[0] += seconds

    def _events():
        return [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]

    import pytest

    class _Client:
        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            raise OrderStatusUnknown("timeout", path="/fapi/v1/order", http_status=0, code=None)

        def cancel_order(self, symbol, orig_client_order_id):
            return {}

        def query_order(self, symbol, orig_client_order_id):
            raise VenueError("bad signature", code=-1022, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    with pytest.raises(VenueError) as exc_info:
        execute_intents(_Client(), [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, policy, audit,
                        clock, sleep_fn, journal=journal, attempt=attempt)

    assert exc_info.value.code == -1022


def test_execute_intents_window_end_unknown_lookup_failure_propagates(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent
    from src.live.rest import OrderStatusUnknown

    def _filters(symbol):
        return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                             min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                             quantity_precision=3, price_precision=2)

    def _intent(symbol):
        return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                           target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                           leg_index=0, decision_price=Decimal("100.10"))

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=10.0, window_deadline_s=20.0)
    audit_path = tmp_path / "exec_audit.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(seconds):
        clock_state[0] += seconds

    def _events():
        return [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]

    import pytest

    short_policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=0.5, window_deadline_s=1.0)

    class _Client:
        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            raise OrderStatusUnknown("timeout", path="/fapi/v1/order", http_status=0, code=None)

        def cancel_order(self, symbol, orig_client_order_id):
            return {}

        def query_order(self, symbol, orig_client_order_id):
            raise VenueError("bad signature", code=-1022, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    with pytest.raises(VenueError) as exc_info:
        execute_intents(_Client(), [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, short_policy, audit,
                        clock, sleep_fn, journal=journal, attempt=attempt)

    assert exc_info.value.code == -1022
    assert "abort_cleanup_failed" in _events()


def test_unknown_submission_not_placed_before_horizon(tmp_path) -> None:
    """Unknown submission is not declared not placed before the horizon."""
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=50.0, window_deadline_s=200.0)
    audit_path = tmp_path / "h.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "j.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    class _Client:
        unknown_outcome_horizon_s = 35.0

        def __init__(self):
            self.posted: list = []
            self.post_times: list = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.post_times.append(clock_state[0])
            self.posted.append(params)
            if len(self.posted) == 1:
                raise OrderStatusUnknown("x", path="/fapi/v1/order", http_status=0, code=None)
            return {"orderId": 2}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            if oid == self.posted[0]["newClientOrderId"]:
                raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()
    execute_intents(client, [_exec_intent("AAAUSDT")], {"AAAUSDT": _exec_filters("AAAUSDT")}, policy, audit,
                    lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s), journal=journal, attempt=attempt)
    assert len(client.posted) == 2
    assert client.post_times[1] - client.post_times[0] >= 35.0


def test_unknown_submission_found_late_adopted(tmp_path) -> None:
    """An unknown submission found late is adopted, not duplicated."""
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=50.0, window_deadline_s=200.0)
    audit_path = tmp_path / "adopt.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "j.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]

    class _Client:
        unknown_outcome_horizon_s = 35.0

        def __init__(self):
            self.posted: list = []
            self.post_times: list = []
            self.cancels: list = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.post_times.append(clock_state[0])
            self.posted.append(params)
            if len(self.posted) == 1:
                raise OrderStatusUnknown("x", path="/fapi/v1/order", http_status=0, code=None)
            return {"orderId": len(self.posted)}

        def cancel_order(self, symbol, oid):
            self.cancels.append(oid)
            return {}

        def query_order(self, s, oid):
            if oid in self.cancels:
                return {"status": "CANCELED", "executedQty": "0", "avgPrice": "0"}
            if clock_state[0] < 10.0:
                raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"status": "NEW", "executedQty": "0", "avgPrice": "0"}

    client = _Client()
    import json
    execute_intents(client, [_exec_intent("AAAUSDT")], {"AAAUSDT": _exec_filters("AAAUSDT")}, policy, audit,
                    lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s), journal=journal, attempt=attempt)
    assert len(client.posted) >= 1
    events = [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert "order_unknown_adopted" in events
    # No duplicate post for the same submission: any repost happens only via the IOC backstop.
    if len(client.post_times) > 1:
        assert client.post_times[1] >= 50.0


def test_window_end_finalize_waits_for_horizon(tmp_path) -> None:
    """Window-end finalize sleeps until the horizon before the final lookup."""
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, _finalize, _IntentRuntime

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=5.0, window_deadline_s=600.0)
    audit = AuditLog(tmp_path / "f.jsonl")

    class _Client:
        unknown_outcome_horizon_s = 35.0

        def __init__(self):
            self.lookups = 0

        def query_order(self, s, oid):
            self.lookups += 1
            raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

        def cancel_order(self, *a, **k):
            return {}

    rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"), fee_schedule=FeeSchedule(2.0, 5.0))
    rt.unresolved_id = "oid-1"
    rt.unresolved_at = 0.0
    clock_state = [5.0]
    sleeps: list = []

    def sleep_fn(s):
        sleeps.append(s)
        clock_state[0] += s

    _finalize(_Client(), [rt], audit, lambda: clock_state[0], sleep_fn)
    assert sleeps == [30.0]
    assert rt.unresolved_id is None


def test_unknown_lookup_transient_is_undecidable(tmp_path) -> None:
    """A TransientReadError during unknown lookup neither adopts nor counts a miss."""
    from src.live.audit import AuditLog
    from src.live.errors import TransientReadError
    from src.live.executor import _resolve_unknown_submission, _IntentRuntime

    rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"), fee_schedule=FeeSchedule(2.0, 5.0))
    rt.unresolved_id = "oid-1"
    rt.unresolved_at = 100.0

    class _C:
        def query_order(self, s, oid):
            raise TransientReadError("blip", path="/fapi/v1/order", http_status=503, code=None, attempts=4)

    assert _resolve_unknown_submission(_C(), rt, 101.0, AuditLog(tmp_path / "u.jsonl")) is False
    assert rt.unresolved_id == "oid-1"
    assert rt.unknown_misses == 0


def test_finalize_horizon_branches(tmp_path) -> None:
    """Finalize covers adopted-now, adopted-after-sleep and already-elapsed paths."""
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, _finalize, _IntentRuntime

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=5.0, window_deadline_s=600.0)
    assert policy.poll_interval_s == 1.0

    def _rt(oid, at):
        rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"), fee_schedule=FeeSchedule(2.0, 5.0))
        rt.unresolved_id = oid
        rt.unresolved_at = at
        return rt

    # First lookup success: adopted immediately, no sleep.
    class _Found:
        unknown_outcome_horizon_s = 35.0

        def __init__(self):
            self.cancels: list = []

        def query_order(self, s, oid):
            if oid in self.cancels:
                return {"status": "CANCELED", "executedQty": "0"}
            return {"status": "NEW", "executedQty": "0"}

        def cancel_order(self, symbol, oid):
            self.cancels.append(oid)
            return {}

    rt = _rt("a", 0.0)
    sleeps: list = []
    _finalize(_Found(), [rt], AuditLog(tmp_path / "f1.jsonl"), lambda: 5.0, sleeps.append)
    assert sleeps == []
    assert rt.unresolved_id is None

    # First miss then found after the horizon sleep: adopted.
    class _Late:
        unknown_outcome_horizon_s = 35.0

        def __init__(self):
            self.n = 0
            self.cancels: list = []

        def query_order(self, s, oid):
            self.n += 1
            if oid in self.cancels:
                return {"status": "CANCELED", "executedQty": "0"}
            if self.n == 1:
                raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"status": "NEW", "executedQty": "0"}

        def cancel_order(self, symbol, oid):
            self.cancels.append(oid)
            return {}

    rt2 = _rt("b", 0.0)
    clock_state = [5.0]
    sleeps2: list = []

    def _sleep(s):
        sleeps2.append(s)
        clock_state[0] += s

    _finalize(_Late(), [rt2], AuditLog(tmp_path / "f2.jsonl"), lambda: clock_state[0], _sleep)
    assert sleeps2 == [30.0]
    assert rt2.unresolved_id is None

    # Already past the horizon: no sleep, declared not placed.
    class _Gone:
        unknown_outcome_horizon_s = 35.0

        def query_order(self, s, oid):
            raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

        def cancel_order(self, *a, **k):
            return {}

    rt3 = _rt("c", 0.0)
    sleeps3: list = []
    _finalize(_Gone(), [rt3], AuditLog(tmp_path / "f3.jsonl"), lambda: 100.0, sleeps3.append)
    assert sleeps3 == []
    assert rt3.unresolved_id is None

    # Second lookup with a non-2013 error propagates.
    import pytest

    class _BadSecond:
        unknown_outcome_horizon_s = 35.0

        def __init__(self):
            self.n = 0

        def query_order(self, s, oid):
            self.n += 1
            if self.n == 1:
                raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            raise VenueError("bad", code=-1022, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

        def cancel_order(self, *a, **k):
            return {}

    with pytest.raises(VenueError, match="bad"):
        _finalize(_BadSecond(), [_rt("d", 0.0)], AuditLog(tmp_path / "f4.jsonl"), lambda: 5.0, lambda s: None)


def test_finalize_no_wait_path_leaves_unknown_order_unresolved(tmp_path) -> None:
    """A no-op sleeper (shutdown/abort cleanup) never declares an unknown order NOT_PLACED early."""
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import _finalize, _IntentRuntime

    class _Missing:
        unknown_outcome_horizon_s = 30.0

        def __init__(self) -> None:
            self.queries = 0

        def query_order(self, s, oid):
            self.queries += 1
            raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"), fee_schedule=FeeSchedule(2.0, 5.0))
    rt.unresolved_id = "u1"
    rt.unresolved_at = 0.0
    client = _Missing()
    _finalize(client, [rt], AuditLog(tmp_path / "a.jsonl"), lambda: 1.0, lambda _s: None)
    assert rt.unresolved_id == "u1"
    assert client.queries == 2
    assert rt.terminal_status is None


def test_default_unknown_outcome_horizon_ignores_environment(monkeypatch) -> None:
    """The module-level default derives from field defaults, never from LIVE_* env at import time."""
    from src.live.executor import _default_unknown_outcome_horizon_s
    from src.live.settings import LiveSettings

    monkeypatch.setenv("LIVE_RECV_WINDOW_MS", "not-a-number")
    expected = LiveSettings.model_fields["recv_window_ms"].default / 1000
    assert _default_unknown_outcome_horizon_s() > expected
