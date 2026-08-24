"""SCENARIO_LIVE_09/15/16/17/23: 패시브 체이스, bounded 폴링, 정확 양자화, 슬라이스 보존,
SHADOW 숏서킷과 체이스 밴드. MARKET은 절대 존재하지 않는다."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from src.live.audit import AuditLog
from src.live.errors import VenueError
from src.live.executor import (
    PassiveExecutionPolicy,
    _capped_ioc_price,
    _slice_quantities,
    execute_intent,
    execute_intents,
)
from src.live.filters import SymbolFilters
from src.live.planner import OrderIntent
from src.live.rest import ShadowResponse


class StubClient:
    def __init__(
        self,
        *,
        executed_sequence: list[str] | None = None,
        gtx_rejects: int = 0,
        touches: list[tuple[str, str]] | None = None,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancels: list[str] = []
        self.queries = 0
        self._executed_sequence = list(executed_sequence or [])
        self._gtx_rejects = gtx_rejects
        self._touches = list(touches or [("100.00", "100.20")])

    def book_ticker(self, symbol: str) -> dict[str, str]:
        if len(self._touches) > 1:
            # 호가가 움직이는 시나리오: 호출마다 시퀀스를 전진한다(마지막 값 고정).
            idx = getattr(self, "_touch_cursor", 0)
            bid, ask = self._touches[min(idx, len(self._touches) - 1)]
            self._touch_cursor = idx + 1
        else:
            bid, ask = self._touches[0]
        return {"bidPrice": bid, "askPrice": ask}

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
        leg_index=0,
        decision_price=Decimal("100"),
    )


def _filters(tick_size: str = "0.10") -> SymbolFilters:
    return SymbolFilters(
        symbol="AAAUSDT",
        tick_size=Decimal(tick_size),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )


def _policy(**overrides) -> PassiveExecutionPolicy:
    base = {
        "poll_interval_s": 3.0,
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
    # (a') 고정 호가 + 미체결: passive_deadline 경과 후 IOC 백스톱으로 전이한다.
    # I-CHASE-BAND 하에서 정적 호가는 재호가를 유발하지 않는다(밴드 안 유지).
    client = StubClient()
    outcome = execute_intent(
        client, _intent(), _filters(), _policy(), AuditLog(tmp_path / "a.jsonl"), SteppingClock(15.0)
    )
    gtx_posts = [o for o in client.orders if o["timeInForce"] == "GTX"]
    assert len(client.cancels) >= 1
    reposts_after_initial = len(gtx_posts) - 1
    assert 0 <= reposts_after_initial <= _policy().max_chases
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


def test_SCENARIO_LIVE_15_poll_loop_is_sleep_bounded(tmp_path) -> None:
    """R1: 매 tick 은 sleep 으로 끝나고 루프 상한은 ceil(window/poll)+1 로 유도된다."""
    policy = _policy(poll_interval_s=3.0, window_deadline_s=600.0)
    expected_ticks = math.ceil(policy.window_deadline_s / policy.poll_interval_s) + 1
    client = StubClient()
    audit = AuditLog(tmp_path / "15.jsonl")
    sleeps: list[float] = []

    outcomes = execute_intents(
        client,
        [_intent()],
        {"AAAUSDT": _filters()},
        policy,
        audit,
        lambda: 0.0,
        sleeps.append,
    )

    assert len(sleeps) >= 1
    assert len(sleeps) == expected_ticks
    assert client.queries <= expected_ticks
    assert outcomes[0].status == "RESIDUAL"


def test_SCENARIO_LIVE_16_ioc_price_is_tick_multiple_and_capped() -> None:
    """D3 수정: IOC 가격은 지수가 아니라 틱 '배수'로 양자화된다."""
    price = _capped_ioc_price(
        Decimal("100.37"), is_buy=True, taker_cap_bps=8.0, tick_size=Decimal("0.5")
    )
    assert price is not None
    assert price % Decimal("0.5") == 0
    assert price <= Decimal("100.37") * Decimal("1.0008")

    sell_price = _capped_ioc_price(
        Decimal("100.37"), is_buy=False, taker_cap_bps=8.0, tick_size=Decimal("0.5")
    )
    assert sell_price is not None
    assert sell_price % Decimal("0.5") == 0
    assert sell_price >= Decimal("100.37") * Decimal("0.9992")


def test_SCENARIO_LIVE_17_slice_quantities_conserve_total() -> None:
    """D4/I-SLICE-EXACT: 합은 total 과 정확히 같고 각 원소는 step 의 배수다."""
    slices = _slice_quantities(Decimal("0.007"), 4, Decimal("0.001"))
    assert sum(slices) == Decimal("0.007")
    assert len(slices) <= 4
    assert all(q % Decimal("0.001") == 0 for q in slices)

    tiny = _slice_quantities(Decimal("0.002"), 4, Decimal("0.001"))
    assert tiny == [Decimal("0.001"), Decimal("0.001")]
    assert sum(tiny) == Decimal("0.002")


class ShadowStubClient:
    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []
        self.queries = 0

    def book_ticker(self, symbol: str) -> dict[str, str]:
        return {"bidPrice": "100.00", "askPrice": "100.20"}

    def new_order(self, params: dict[str, Any]) -> Any:
        self.orders.append(params)
        return ShadowResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

    def query_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        self.queries += 1
        return {"status": "NEW", "executedQty": "0"}


def test_SCENARIO_LIVE_23_shadow_shortcircuit_and_chase_band(tmp_path) -> None:
    """R8: ShadowResponse 면 query_order 0회. I-CHASE-BAND: 밴드 밖 재호가 금지."""
    shadow_client = ShadowStubClient()
    outcomes = execute_intents(
        shadow_client,
        [_intent()],
        {"AAAUSDT": _filters()},
        _policy(),
        AuditLog(tmp_path / "shadow.jsonl"),
        lambda: 0.0,
        lambda _seconds: None,
    )
    assert shadow_client.queries == 0
    assert outcomes[0].status == "SHADOW"
    assert outcomes[0].filled_qty == 0

    # 호가가 decision_price 대비 +50bps 로 이동하면 GTX 는 밴드 상한을 넘지 못한다.
    moving_client = StubClient(touches=[("100.00", "100.20"), ("100.50", "100.70")])
    moving_outcomes = execute_intents(
        moving_client,
        [_intent()],
        {"AAAUSDT": _filters()},
        _policy(poll_interval_s=3.0, window_deadline_s=60.0),
        AuditLog(tmp_path / "band.jsonl"),
        lambda: 0.0,
        lambda _seconds: None,
    )
    band_high = Decimal("100") * (Decimal(1) + Decimal("10") / Decimal(10_000))
    posted_prices = [Decimal(o["price"]) for o in moving_client.orders]
    assert posted_prices, "initial GTX must be posted"
    assert all(price <= band_high for price in posted_prices)
    assert len(moving_client.orders) == 1  # 밴드 이탈 구간에서 신규 게시 없음
    assert moving_outcomes[0].unfilled_qty > 0


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_09_PASSIVE_CHASE_THEN_IOC_NEVER_MARKET",
    "SCENARIO_LIVE_15",  # POLL_LOOP_IS_SLEEP_BOUNDED
    "SCENARIO_LIVE_16",  # IOC_PRICE_IS_TICK_MULTIPLE_AND_CAPPED
    "SCENARIO_LIVE_17",  # SLICE_QUANTITIES_CONSERVE_TOTAL
    "SCENARIO_LIVE_23",  # SHADOW_SHORTCIRCUIT_AND_CHASE_BAND
)
