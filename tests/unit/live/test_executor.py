"""SCENARIO_LIVE_09: 패시브 체이스 -> IOC 백스톱. MARKET은 절대 존재하지 않는다."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


from src.live.audit import AuditLog
from src.live.errors import VenueError
from src.live.executor import PassiveExecutionPolicy, execute_intent
from src.live.filters import SymbolFilters
from src.live.planner import OrderIntent


class StubClient:
    def __init__(
        self,
        *,
        executed_sequence: list[str] | None = None,
        gtx_rejects: int = 0,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancels: list[str] = []
        self.queries = 0
        self._executed_sequence = list(executed_sequence or [])
        self._gtx_rejects = gtx_rejects

    def book_ticker(self, symbol: str) -> dict[str, str]:
        return {"bidPrice": "100.00", "askPrice": "100.20"}

    def new_order(self, params: dict[str, Any]) -> dict[str, Any]:
        if params["timeInForce"] == "GTX" and self._gtx_rejects > 0:
            self._gtx_rejects -= 1
            raise VenueError(
                "post-only would trade",
                code=-5022,
                http_status=400,
                path="/fapi/v1/order",
                payload_digest="0" * 12,
            )
        self.orders.append(params)
        return {"orderId": len(self.orders)}

    def cancel_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        self.cancels.append(orig_client_order_id)
        return {}

    def query_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        self.queries += 1
        executed = (
            self._executed_sequence.pop(0)
            if self._executed_sequence
            else self._executed_sequence_default
        )
        return {"status": "NEW", "executedQty": executed}

    _executed_sequence_default = "0"


class SteppingClock:
    """호출마다 step 초씩 진행되는 가짜 시계."""

    def __init__(self, step: float) -> None:
        self._step = step
        self._t = -step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


def _intent() -> OrderIntent:
    return OrderIntent(
        symbol="AAAUSDT",
        side="BUY",
        quantity=Decimal("1.000"),
        reduce_only=False,
        target_qty=Decimal("1.000"),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
    )


def _filters() -> SymbolFilters:
    return SymbolFilters(
        symbol="AAAUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )


def _policy(**overrides) -> PassiveExecutionPolicy:
    base = {
        "reprice_interval_s": 10.0,
        "chase_ticks": 2,
        "max_chases": 3,
        "passive_deadline_s": 50.0,
        "window_deadline_s": 600.0,
        "taker_cap_bps": 15.0,
        "max_slices": 1,
    }
    base.update(overrides)
    return PassiveExecutionPolicy(**base)


def test_SCENARIO_LIVE_09_passive_chase_then_ioc_never_market(tmp_path) -> None:
    # (a) 고정 호가 + 미체결: reprice_interval마다 cancel+재게시, chases <= max_chases.
    client = StubClient()
    outcome = execute_intent(
        client, _intent(), _filters(), _policy(), AuditLog(tmp_path / "a.jsonl"), SteppingClock(15.0)
    )
    gtx_posts = [o for o in client.orders if o["timeInForce"] == "GTX"]
    assert len(client.cancels) >= 1
    reposts_after_initial = len(gtx_posts) - 1
    assert 0 < reposts_after_initial <= _policy().max_chases
    assert outcome.unfilled_qty > 0  # 아무것도 체결되지 않음

    # (b) 모든 주문은 LIMIT + GTX/IOC 이며 MARKET은 없다.
    all_orders = client.orders
    assert all(o["type"] == "LIMIT" for o in all_orders)
    assert all(o["timeInForce"] in ("GTX", "IOC") for o in all_orders)

    # (c) -5022(GTX 거절)는 예외로 실패하지 않고 재호가로 이어진다.
    rejecty_client = StubClient(gtx_rejects=2)
    execute_intent(
        rejecty_client,
        _intent(),
        _filters(),
        _policy(max_chases=8),
        AuditLog(tmp_path / "c.jsonl"),
        SteppingClock(5.0),
    )
    assert len(rejecty_client.orders) >= 1

    # (d) passive_deadline 경과 후 잔여는 IOC + 가격 상한(ask*(1+15bps))으로 게시된다.
    deadline_client = StubClient()
    execute_intent(
        deadline_client,
        _intent(),
        _filters(),
        _policy(passive_deadline_s=20.0),
        AuditLog(tmp_path / "d.jsonl"),
        SteppingClock(15.0),
    )
    ioc_orders = [o for o in deadline_client.orders if o["timeInForce"] == "IOC"]
    assert len(ioc_orders) >= 1
    ask = Decimal("100.20")
    cap = ask * (Decimal(1) + Decimal("15") / Decimal(10_000))
    for order in ioc_orders:
        assert Decimal(order["price"]) <= cap

    # (e) window_deadline 경과: 잔여 취소 + unfilled 보고 + 추가 주문 없음.
    class ScriptedClock:
        def __init__(self) -> None:
            self._times = [0.0, 0.0, 0.0]

        def __call__(self) -> float:
            if self._times:
                return self._times.pop(0)
            return 8000.0

    windowed_client = StubClient()
    windowed_outcome = execute_intent(
        windowed_client,
        _intent(),
        _filters(),
        _policy(window_deadline_s=7200.0),
        AuditLog(tmp_path / "e.jsonl"),
        ScriptedClock(),
    )
    assert len(windowed_client.orders) == 1  # 초기 GTX 한 건뿐
    assert windowed_client.cancels == [
        o["newClientOrderId"] for o in windowed_client.orders
    ]
    assert windowed_outcome.unfilled_qty > 0
    assert windowed_outcome.status == "RESIDUAL"

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_09_PASSIVE_CHASE_THEN_IOC_NEVER_MARKET",
)
