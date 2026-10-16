"""SCENARIO_LIVE_02: 오류 정책 레지스트리는 미등록 코드에 fail-closed다."""

from __future__ import annotations

from collections.abc import Mapping

from src.live.errors import BINANCE_ERROR_POLICY, ErrorAction, resolve_error_action

def _venue_now_ms() -> int:
    import time

    return int(time.time() * 1000)



def test_SCENARIO_LIVE_02_error_registry_fails_closed() -> None:
    assert resolve_error_action(-1021) is ErrorAction.RESYNC_CLOCK
    assert resolve_error_action(-1013) is ErrorAction.INTENT_REJECT
    assert resolve_error_action(-5022) is ErrorAction.BENIGN_REPRICE
    assert resolve_error_action(-2011) is ErrorAction.BENIGN

    for unregistered in (-9999, 12345, 0):
        assert resolve_error_action(unregistered) is ErrorAction.FAIL_CLOSED
    assert resolve_error_action(None) is ErrorAction.FAIL_CLOSED

    assert isinstance(BINANCE_ERROR_POLICY, Mapping)
    assert all(isinstance(action, ErrorAction) for action in BINANCE_ERROR_POLICY.values())


def test_scoped_rejection_codes_resolve_to_scoped_actions() -> None:
    """Spec 02: symbol-scoped codes map to scoped actions, not FAIL_CLOSED."""
    for code in (-4164, -4140, -2027, -1111, -4014, -4023, -4003, -1013):
        assert resolve_error_action(code) is ErrorAction.INTENT_REJECT
    assert resolve_error_action(-2019) is ErrorAction.MARGIN_WAIT
    assert resolve_error_action(-4400) is ErrorAction.RISK_INCREASE_FREEZE
    assert resolve_error_action(-9999) is ErrorAction.FAIL_CLOSED
    assert resolve_error_action(None) is ErrorAction.FAIL_CLOSED


def test_observed_venue_codes_are_intent_rejects() -> None:
    for code in (-4016, -4005, -4118):
        assert resolve_error_action(code) is ErrorAction.INTENT_REJECT


def test_duplicate_client_order_id_halts_closed() -> None:
    assert BINANCE_ERROR_POLICY[-4116] is ErrorAction.FAIL_CLOSED
    assert resolve_error_action(-4116) is ErrorAction.FAIL_CLOSED


def test_unobserved_scoped_codes_stay_fail_closed() -> None:
    for code in (-2025, -4061, -2024, -4015, -2018):
        assert code not in BINANCE_ERROR_POLICY
        assert resolve_error_action(code) is ErrorAction.FAIL_CLOSED


def test_max_qty_reject_isolates_one_intent(tmp_path) -> None:
    """A quantity rejection terminates one intent while the next intent fills."""
    from decimal import Decimal

    import pandas as pd

    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import (
        PassiveExecutionPolicy,
        execute_intents,
    )
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent

    filters = SymbolFilters(
        symbol="AAAUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("1"),
        max_qty=Decimal("100000"),
        quantity_precision=3,
        price_precision=2,
    )
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
    audit = AuditLog(tmp_path / "audit.jsonl")
    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=10.0, window_deadline_s=20.0)

    def _intent(leg_index: int) -> OrderIntent:
        return OrderIntent(
                symbol="AAAUSDT",
                side="BUY",
                quantity=Decimal("1"),
                reduce_only=False,
                target_qty=Decimal("1"),
                current_qty=Decimal("0"),
                client_order_prefix="20260914",
                leg_index=leg_index,
                decision_price=Decimal("100.10"),
        )

    class Client:
        def __init__(self):
            self.posts = []

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.posts.append(params)
            if len(self.posts) == 1:
                raise VenueError(
                    "quantity exceeds maximum", code=-4005, http_status=400,
                    path="/fapi/v1/order", payload_digest="0" * 12,
                )
            return {"time": _venue_now_ms(), "status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

        def query_order(self, symbol, order_id):
            return {"time": _venue_now_ms(), "status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    clock_state = [0.0]

    def sleep(seconds):
        clock_state[0] += seconds

    client = Client()
    outcomes = execute_intents(
        client, [_intent(0), _intent(1)], {"AAAUSDT": filters}, policy, audit,
        lambda: clock_state[0], sleep, journal=journal, attempt=attempt,
    )
    assert [outcome.status for outcome in outcomes] == ["REJECTED", "FILLED"]
    assert outcomes[0].reject_code == -4005
    assert outcomes[0].unfilled_qty == Decimal("1")
    assert outcomes[1].filled_qty == Decimal("1")
    assert len(client.posts) == 2

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_02_ERROR_REGISTRY_FAILS_CLOSED",
)
