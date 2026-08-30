"""SCENARIO_LIVE_09/15/16/17/23: 패시브 체이스, bounded 폴링, 정확 양자화, 슬라이스 보존,
SHADOW 숏서킷과 체이스 밴드. MARKET은 절대 존재하지 않는다."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import pytest

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
from src.live.rest import PaperResponse, ShadowResponse


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

    # (d) passive_deadline 경과 후 잔여는 IOC 백스톱으로 전이하며, 게시 가격은
    # ask*(1+15bps) 상한을 준수하고 재게시는 max_ioc_attempts 로 상한 종결된다.
    deadline_client = StubClient(touches=[("100.00", "100.00")])
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
    assert len(ioc_orders) <= _policy().max_ioc_attempts
    ask = Decimal("100.00")
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


def test_SCENARIO_LIVE_20_PASSIVE_DEADLINE_STRICTLY_BELOW_WINDOW(tmp_path) -> None:
    """passive_deadline_s 기본값 1080.0 은 window_deadline_s 보다 엄격히 작고,
    위반 구성은 fail-closed 한다. 정적 호가 런은 t>=1080s 첫 폴에서 IOC 로
    전이한다(L1 사분기 죽은 분기 제거)."""
    policy = PassiveExecutionPolicy()
    assert policy.passive_deadline_s == 1080.0
    assert policy.passive_deadline_s < policy.window_deadline_s == 1800.0
    with pytest.raises(ValueError, match="passive_deadline_s"):
        PassiveExecutionPolicy(passive_deadline_s=1800.0)
    with pytest.raises(ValueError, match="poll_interval_s"):
        PassiveExecutionPolicy(poll_interval_s=0.0)
    with pytest.raises(ValueError, match="max_ioc_attempts"):
        PassiveExecutionPolicy(max_ioc_attempts=0)

    client = StubClient(touches=[("100.00", "100.05")])
    rclock = SteppingClock(3.0)
    now_value = 0.0
    post_times: list[float] = []
    orig_new_order = client.new_order

    def _recording_new_order(params: dict[str, Any]) -> dict[str, Any]:
        post_times.append(now_value)
        return orig_new_order(params)

    class _NowClock:
        def __call__(self) -> float:
            nonlocal now_value
            now_value = rclock()
            return now_value

    client.new_order = _recording_new_order  # type: ignore[method-assign]
    outcome = execute_intent(
        client, _intent(), _filters(tick_size="0.01"), policy, AuditLog(tmp_path / "20.jsonl"), _NowClock()
    )
    assert len(post_times) >= 2
    assert client.orders[0]["timeInForce"] == "GTX"
    assert client.orders[1]["timeInForce"] == "IOC"
    assert post_times[0] == 3.0  # 첫 폴에서 초기 GTX 게시
    # timed_out 이 유일한 전이 트리거: 게시 나이가 정확히 passive_deadline 에
    # 도달하는 첫 폴에서 즉시 전이한다.
    assert post_times[1] == post_times[0] + policy.passive_deadline_s
    assert post_times[1] >= 1080.0
    ioc_price = Decimal(client.orders[1]["price"])
    # S7: IOC 캡은 리스크 레일(max_cross_bps)이다 -- chase_band_bps 가 아니다.
    assert ioc_price <= Decimal("100") * (
        Decimal(1) + Decimal(str(policy.max_cross_bps)) / Decimal(10_000)
    )
    assert ioc_price >= Decimal("100.05")  # ask 이상: 마케터블
    assert outcome.status == "RESIDUAL"


def test_SCENARIO_LIVE_21_CHASE_EXHAUSTION_HOLDS_INSTEAD_OF_CROSSING(tmp_path) -> None:
    """호가가 매 폴 움직여도 chases 소진은 재페그 중단일 뿐이다: GTX 게시는
    정확히 max_chases+1 회, IOC 전이 없음(t=1080s 이전 무조건)."""
    touches = [(f"{100 + 0.3 * i:.2f}", f"{100.20 + 0.3 * i:.2f}") for i in range(16)]
    client = StubClient(touches=touches)
    # 리스크 레일은 알파 레일(chase_band)을 초과해야 한다. 최종 호가
    # (104.50/104.70)는 400bps 레일 상한 104.00 밖이라 IOC 거부가 유지된다.
    policy = _policy(max_chases=8, chase_band_bps=300.0, max_cross_bps=400.0)
    assert policy.passive_deadline_s == 50.0
    assert policy.window_deadline_s == 600.0
    outcomes = execute_intents(
        client,
        [_intent()],
        {"AAAUSDT": _filters()},
        policy,
        AuditLog(tmp_path / "21.jsonl"),
        SteppingClock(3.0),
        lambda _seconds: None,
    )
    gtx_posts = [o for o in client.orders if o["timeInForce"] == "GTX"]
    ioc_posts = [o for o in client.orders if o["timeInForce"] == "IOC"]
    assert all(o["type"] == "LIMIT" for o in client.orders)
    assert len(gtx_posts) == policy.max_chases + 1 == 9
    assert not ioc_posts
    assert outcomes[0].chases == policy.max_chases
    assert outcomes[0].status == "RESIDUAL"


def test_SCENARIO_LIVE_22_FAVOURABLE_TOUCH_STILL_POSTS(tmp_path) -> None:
    """유리 방향 밴드 이탈(매수 중 100bps 하락 bid=99.00)은 HOLD 가 아니라
    게시다. 매도 거울(ask=101.00 상승)도 동일하다."""
    buy_client = StubClient(touches=[("99.00", "99.20")])
    buy_outcome = execute_intent(
        buy_client, _intent(), _filters(), _policy(), AuditLog(tmp_path / "22b.jsonl"), SteppingClock(3.0)
    )
    buy_gtx = [o for o in buy_client.orders if o["timeInForce"] == "GTX"]
    assert buy_gtx
    assert Decimal(buy_gtx[0]["price"]) == Decimal("99.00")
    assert buy_outcome.unfilled_qty > 0

    sell_intent = OrderIntent(
        symbol="AAAUSDT",
        side="SELL",
        quantity=Decimal("1.000"),
        reduce_only=False,
        target_qty=Decimal("0"),
        current_qty=Decimal("1.000"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("100"),
    )
    sell_client = StubClient(touches=[("100.80", "101.00")])
    sell_outcome = execute_intent(
        sell_client, sell_intent, _filters(), _policy(), AuditLog(tmp_path / "22s.jsonl"), SteppingClock(3.0)
    )
    sell_gtx = [o for o in sell_client.orders if o["timeInForce"] == "GTX"]
    assert sell_gtx
    assert Decimal(sell_gtx[0]["price"]) == Decimal("101.00")
    assert sell_outcome.unfilled_qty > 0


def test_SCENARIO_LIVE_23_NON_MARKETABLE_IOC_IS_BOUNDED(tmp_path) -> None:
    """클램프 후 비마케터블 IOC 는 None 이고, 시장가보다 낮은 가격 재게시는
    max_ioc_attempts 회로 상한 종결된다(창 끝까지 ~600회 스팸 금지)."""
    capped = _capped_ioc_price(
        Decimal("100.50"), is_buy=True, taker_cap_bps=15.0,
        tick_size=Decimal("0.10"), band_low=Decimal("99.90"), band_high=Decimal("100.10"),
    )
    assert capped is None  # 클램프 100.10 < ask 100.50
    capped_sell = _capped_ioc_price(
        Decimal("99.50"), is_buy=False, taker_cap_bps=15.0,
        tick_size=Decimal("0.10"), band_low=Decimal("99.90"), band_high=Decimal("100.10"),
    )
    assert capped_sell is None  # 클램프 99.90 > bid 99.50
    marketable_sell = _capped_ioc_price(
        Decimal("100.00"), is_buy=False, taker_cap_bps=15.0,
        tick_size=Decimal("0.10"), band_low=Decimal("99.90"), band_high=Decimal("100.10"),
    )
    assert marketable_sell is not None

    client = StubClient(touches=[("100.00", "100.00")])
    policy = _policy(passive_deadline_s=20.0, window_deadline_s=600.0)
    outcome = execute_intent(
        client, _intent(), _filters(), policy, AuditLog(tmp_path / "23.jsonl"), SteppingClock(3.0)
    )
    ioc_orders = [o for o in client.orders if o["timeInForce"] == "IOC"]
    assert 1 <= len(ioc_orders) <= policy.max_ioc_attempts == 10
    assert outcome.filled_qty == 0
    assert outcome.status == "RESIDUAL"


class CancelFillStubClient(StubClient):
    """cancel 직후 같은 주문 id 의 executedQty 만 '0.5'를 반환하는 fake."""

    def __init__(self, touches: list[tuple[str, str]] | None = None) -> None:
        super().__init__(touches=touches)
        self.cancelled_ids: set[str] = set()

    def cancel_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        self.cancelled_ids.add(orig_client_order_id)
        return {}

    def query_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        self.queries += 1
        executed = "0.5" if orig_client_order_id in self.cancelled_ids else "0"
        return {"status": "NEW", "executedQty": executed}


def test_SCENARIO_LIVE_24_CANCEL_SETTLES_PARTIAL_FILL(tmp_path) -> None:
    """취소 시점 부분체결은 cancel 직후 재조회로 정산된다. 수정 전 경로
    (취소 응답 폐기)라면 filled_qty 는 0 으로 수렴해 이 테스트는 실패한다.
    ask(101.00)가 리스크 레일 밖이라 IOC 백스톱은 게시되지 않고, 잔량 0.5 는
    순수하게 취소 정산만으로 확정된다."""
    client = CancelFillStubClient(touches=[("100.00", "101.00")])
    policy = _policy(passive_deadline_s=20.0, window_deadline_s=600.0)
    outcome = execute_intent(
        client, _intent(), _filters(), policy, AuditLog(tmp_path / "24.jsonl"), SteppingClock(3.0)
    )
    assert outcome.filled_qty == Decimal("0.5")
    assert outcome.unfilled_qty == Decimal("0.5")


def test_SCENARIO_LIVE_25_IOC_ALWAYS_MARKETABLE_INSIDE_RISK_RAIL(tmp_path) -> None:
    """S7: chase_band(10bps)는 알파 레일, max_cross_bps(50bps)는 리스크 레일.
    ask 가 chase 밴드 밖이지만 리스크 레일 안이면 IOC 는 반드시 마케터블
    가격을 반환한다(수정 전에는 밴드 클램프 때문에 None)."""
    assert PassiveExecutionPolicy().max_cross_bps == 50.0
    with pytest.raises(ValueError, match="max_cross_bps"):
        PassiveExecutionPolicy(max_cross_bps=10.0)

    price = _capped_ioc_price(
        Decimal("100.30"), is_buy=True, taker_cap_bps=8.0,
        tick_size=Decimal("0.10"), band_low=Decimal("99.50"), band_high=Decimal("100.50"),
    )
    assert price is not None
    assert price >= Decimal("100.30")
    assert price % Decimal("0.10") == 0

    # 엔드투엔드: passive 마감 후 IOC 가 실제로 게시된다(밴드 거부 없음).
    client = StubClient(touches=[("100.00", "100.30")])
    policy = _policy(passive_deadline_s=20.0, window_deadline_s=600.0)
    outcome = execute_intent(
        client, _intent(), _filters(tick_size="0.10"), policy,
        AuditLog(tmp_path / "25.jsonl"), SteppingClock(3.0),
    )
    ioc_orders = [o for o in client.orders if o["timeInForce"] == "IOC"]
    assert len(ioc_orders) >= 1
    assert all(
        Decimal(o["price"]) >= Decimal("100.30") and o["type"] == "LIMIT"
        for o in ioc_orders
    )


def test_SCENARIO_LIVE_26_RISK_RAIL_STILL_REFUSES_ANOMALY(tmp_path) -> None:
    """S7: ask 가 max_cross_bps 레일 밖(100bps)이면 None 이고, execute_intents
    는 해당 intent 에 IOC 0건으로 RESIDUAL 로 종결한다."""
    assert _capped_ioc_price(
        Decimal("101.00"), is_buy=True, taker_cap_bps=8.0,
        tick_size=Decimal("0.10"), band_low=Decimal("99.50"), band_high=Decimal("100.50"),
    ) is None

    client = StubClient(touches=[("100.00", "101.00")])
    policy = _policy(passive_deadline_s=20.0, window_deadline_s=600.0)
    outcome = execute_intent(
        client, _intent(), _filters(tick_size="0.10"), policy,
        AuditLog(tmp_path / "26.jsonl"), SteppingClock(3.0),
    )
    ioc_orders = [o for o in client.orders if o["timeInForce"] == "IOC"]
    assert not ioc_orders
    assert outcome.status == "RESIDUAL"
    assert outcome.unfilled_qty > 0


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_09_PASSIVE_CHASE_THEN_IOC_NEVER_MARKET",
    "SCENARIO_LIVE_15",  # POLL_LOOP_IS_SLEEP_BOUNDED
    "SCENARIO_LIVE_16",  # IOC_PRICE_IS_TICK_MULTIPLE_AND_CAPPED
    "SCENARIO_LIVE_17",  # SLICE_QUANTITIES_CONSERVE_TOTAL
    "SCENARIO_LIVE_23",  # SHADOW_SHORTCIRCUIT_AND_CHASE_BAND
    "SCENARIO_LIVE_20_PASSIVE_DEADLINE_STRICTLY_BELOW_WINDOW",
    "SCENARIO_LIVE_21_CHASE_EXHAUSTION_HOLDS_INSTEAD_OF_CROSSING",
    "SCENARIO_LIVE_22_FAVOURABLE_TOUCH_STILL_POSTS",
    "SCENARIO_LIVE_23_NON_MARKETABLE_IOC_IS_BOUNDED",
    "SCENARIO_LIVE_24_CANCEL_SETTLES_PARTIAL_FILL",
    "SCENARIO_LIVE_25_IOC_ALWAYS_MARKETABLE_INSIDE_RISK_RAIL",
    "SCENARIO_LIVE_26_RISK_RAIL_STILL_REFUSES_ANOMALY",
    "SCENARIO_LIVE_27_PAPER_FILLS_WITHOUT_SENDING_ORDERS",
    "SCENARIO_LIVE_28_PAPER_EXERCISES_IOC_BACKSTOP",
)


class PaperStubClient:
    """PAPER 전송 초크포인트를 흉내내는 스텁.

    변이 시도는 기록만 하고 실제 네트워크 전송은 절대 없다(sent_* 는 항상
    비어 있다). 주문 조회는 PAPER 에서 발생하지 않는다는 불변을 단언한다.
    """

    def __init__(self, touches: list[tuple[str, str]]) -> None:
        self.suppressed_attempts: list[dict[str, Any]] = []
        self.sent_orders: list[dict[str, Any]] = []
        self.sent_cancels: list[str] = []
        self._touches = list(touches)
        self._cursor = 0

    def book_ticker(self, symbol: str) -> dict[str, str]:
        idx = min(self._cursor, len(self._touches) - 1)
        self._cursor += 1
        bid, ask = self._touches[idx]
        return {"bidPrice": bid, "askPrice": ask}

    def new_order(self, params: dict[str, Any]) -> Any:
        self.suppressed_attempts.append(dict(params))
        return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

    def cancel_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        return {}

    def query_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        raise AssertionError("PAPER must never poll a venue order")


def test_SCENARIO_LIVE_27_PAPER_FILLS_WITHOUT_SENDING_ORDERS(tmp_path) -> None:
    """SCENARIO_LIVE_27_PAPER_FILLS_WITHOUT_SENDING_ORDERS: a PAPER run whose
    observed touch trades through the posted GTX price fills locally (non-zero
    filled_qty, status FILLED) while zero mutating requests reach the network;
    the equivalent SHADOW run stays at filled_qty == 0 / status SHADOW."""
    # ask(99.50) < 게시 GTX 가격(bid 100.00 양자화): 엄밀 trade-through.
    paper_client = PaperStubClient(touches=[("100.00", "99.50")])
    outcome = execute_intent(
        paper_client,
        _intent(),
        _filters(tick_size="0.01"),
        _policy(),
        AuditLog(tmp_path / "paper27.jsonl"),
        SteppingClock(3.0),
    )
    assert outcome.status == "FILLED"
    assert outcome.filled_qty == Decimal("1.000")
    assert outcome.avg_fill_price is not None
    assert outcome.avg_fill_price > 0
    assert paper_client.sent_orders == []
    assert paper_client.sent_cancels == []
    assert len(paper_client.suppressed_attempts) >= 1
    assert all(o["type"] == "LIMIT" for o in paper_client.suppressed_attempts)
    assert all(o["timeInForce"] in ("GTX", "IOC") for o in paper_client.suppressed_attempts)

    shadow_client = ShadowStubClient()
    shadow_outcomes = execute_intents(
        shadow_client,
        [_intent()],
        {"AAAUSDT": _filters()},
        _policy(),
        AuditLog(tmp_path / "shadow27.jsonl"),
        lambda: 0.0,
        lambda _seconds: None,
    )
    assert shadow_outcomes[0].status == "SHADOW"
    assert shadow_outcomes[0].filled_qty == 0


def test_SCENARIO_LIVE_28_PAPER_EXERCISES_IOC_BACKSTOP(tmp_path) -> None:
    """SCENARIO_LIVE_28_PAPER_EXERCISES_IOC_BACKSTOP: when the touch never
    trades through the GTX price, the run still transitions to the IOC phase
    after passive_deadline_s and terminates FILLED via the capped IOC -- chase
    and backstop paths that SHADOW never reaches."""
    # 정적 균형 호가: ask(100.00) < 게시가(100.00) 거짓 -> GTX 미체결 유지.
    client = PaperStubClient(touches=[("100.00", "100.00")])
    policy = _policy(passive_deadline_s=20.0)
    outcome = execute_intent(
        client,
        _intent(),
        _filters(tick_size="0.01"),
        policy,
        AuditLog(tmp_path / "paper28.jsonl"),
        SteppingClock(15.0),
    )
    assert outcome.status == "FILLED"
    assert outcome.filled_qty == Decimal("1.000")
    ioc_posts = [
        o for o in client.suppressed_attempts if o["timeInForce"] == "IOC"
    ]
    assert len(ioc_posts) >= 1  # 백스톱 경로가 실제로 실행됐다
    ask = Decimal("100.00")
    cap = ask * (Decimal(1) + Decimal(str(policy.taker_cap_bps)) / Decimal(10_000))
    for order in ioc_posts:
        assert Decimal(order["price"]) <= cap
    assert client.sent_orders == []
    assert client.sent_cancels == []


class OffsetSteppingClock:
    """0이 아닌 값에서 시작하는 단조 증가 시계.

    posted_at/finalized_at 는 0.0 을 '미설정' 센티널로 쓰므로(I-LATENCY-NONE-WHEN-UNPOSTED),
    실제로 게시가 일어난 tick 이 0.0 과 우연히 겹치지 않도록 양수 오프셋에서 시작한다.
    """

    def __init__(self, start: float, step: float) -> None:
        self._t = start - step
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


def test_SCENARIO_EXECUTOR_LATENCY_NONNEGATIVE_ON_FILL(tmp_path) -> None:
    """PAPER 체결 시 latency_seconds 가 채워지고 항상 0 이상이다."""
    paper_client = PaperStubClient(touches=[("100.00", "99.50")])
    outcome = execute_intent(
        paper_client,
        _intent(),
        _filters(tick_size="0.01"),
        _policy(),
        AuditLog(tmp_path / "latency_fill.jsonl"),
        OffsetSteppingClock(1000.0, 3.0),
    )
    assert outcome.status == "FILLED"
    assert outcome.latency_seconds is not None
    assert outcome.latency_seconds >= 0.0


def test_SCENARIO_EXECUTOR_LATENCY_NONE_WHEN_NEVER_POSTED(tmp_path) -> None:
    """filters 맵에 심볼 엔트리가 없으면 즉시 RESIDUAL 확정되고 posted_at 은 결코
    설정되지 않으므로 latency_seconds 는 반드시 None 이다."""
    client = StubClient()
    outcomes = execute_intents(
        client,
        [_intent()],
        {},  # AAAUSDT 엔트리 부재 -> rt.filters is None -> 즉시 RESIDUAL
        _policy(),
        AuditLog(tmp_path / "latency_unposted.jsonl"),
        lambda: 0.0,
        lambda _seconds: None,
    )
    assert outcomes[0].status == "RESIDUAL"
    assert outcomes[0].filled_qty == Decimal(0)
    assert outcomes[0].latency_seconds is None
    assert client.orders == []


def test_SCENARIO_EXECUTOR_FINALIZE_RESIDUAL_HAS_LATENCY(tmp_path) -> None:
    """게시는 됐지만(posted_at > 0) window_deadline_s 안에 체결되지 않아 _finalize 가
    RESIDUAL 로 정산하는 경로: _finalize 에 새로 배선된 clock 이 finalized_at 을
    찍어 latency_seconds 가 관측 가능해야 한다."""
    client = StubClient()
    outcome = execute_intent(
        client,
        _intent(),
        _filters(),
        _policy(window_deadline_s=7200.0),
        AuditLog(tmp_path / "latency_finalize.jsonl"),
        OffsetSteppingClock(1000.0, 2500.0),
    )
    assert outcome.status == "RESIDUAL"
    assert outcome.unfilled_qty > 0
    assert outcome.latency_seconds is not None
    assert outcome.latency_seconds >= 0.0


COVERED_SCENARIOS = (
    *COVERED_SCENARIOS,
    "SCENARIO_LIVE_27_PAPER_FILLS_WITHOUT_SENDING_ORDERS",
    "SCENARIO_LIVE_28_PAPER_EXERCISES_IOC_BACKSTOP",
    "SCENARIO_EXECUTOR_LATENCY_NONNEGATIVE_ON_FILL",
    "SCENARIO_EXECUTOR_LATENCY_NONE_WHEN_NEVER_POSTED",
    "SCENARIO_EXECUTOR_FINALIZE_RESIDUAL_HAS_LATENCY",
)

# SCENARIO_RESIL_01-sink-survives-non-live-error
def test_SCENARIO_RESIL_01_sink_survives_non_live_error(tmp_path):  # noqa: D103
    """SCENARIO_RESIL_01-sink-survives-non-live-error"""
    import http.client
    from decimal import Decimal

    from src.live.audit import AuditLog
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.planner import OrderIntent
    from src.live.rest import PaperResponse

    def make_intent(symbol: str) -> OrderIntent:
        return OrderIntent(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("1.000"),
            reduce_only=False,
            target_qty=Decimal("1.000"),
            current_qty=Decimal("0"),
            client_order_prefix="run1",
            leg_index=0,
            decision_price=Decimal("100"),
        )

    filt_a = SymbolFilters(
        symbol="AAAUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )
    filt_b = SymbolFilters(
        symbol="BBBUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )
    policy = PassiveExecutionPolicy(
        poll_interval_s=3.0,
        chase_ticks=2,
        max_chases=3,
        passive_deadline_s=50.0,
        window_deadline_s=600.0,
        taker_cap_bps=15.0,
        max_slices=1,
    )

    class FillThenThrowClient:
        def __init__(self, exc_type):
            self.tick = 0
            self.exc_type = exc_type

        def book_tickers(self):
            self.tick += 1
            if self.tick == 2:
                raise self.exc_type("boom")
            return {
                "AAAUSDT": {"bidPrice": "100.00", "askPrice": "99.50"},
                "BBBUSDT": {"bidPrice": "100.00", "askPrice": "100.20"},
            }

        def book_ticker(self, symbol):  # noqa: ARG002
            return {"bidPrice": "100.00", "askPrice": "99.50"}

        def new_order(self, params):  # noqa: ARG002
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

        def cancel_order(self, *a, **k):  # noqa: ARG002
            return {}

        def query_order(self, *a, **k):  # noqa: ARG002
            return {"executedQty": "0"}

    intents = [make_intent("AAAUSDT"), make_intent("BBBUSDT")]
    filters = {"AAAUSDT": filt_a, "BBBUSDT": filt_b}
    for exc in (http.client.HTTPException, OSError, KeyboardInterrupt):
        client = FillThenThrowClient(exc)
        sink: list = []
        try:
            execute_intents(
                client,
                intents,
                filters,
                policy,
                AuditLog(tmp_path / f"sink_{exc.__name__}.jsonl"),
                lambda: 0.0,
                lambda _s: None,
                outcome_sink=sink,
            )
            raise AssertionError("should have raised")
        except exc:
            assert len(sink) == len(intents)
            assert sink[0].filled_qty > 0


# SCENARIO_RESIL_07-order-budget-throttle
def test_SCENARIO_RESIL_07_order_budget_throttle(tmp_path):  # noqa: D103
    """SCENARIO_RESIL_07-order-budget-throttle"""
    from decimal import Decimal

    from src.live.executor import PassiveExecutionPolicy, _order_budget_exceeded, _throttled_interval
    from src.live.filters import SymbolFilters
    from src.live.planner import OrderIntent
    from src.live.rest import RateLimits

    filt = SymbolFilters(
        symbol="AAAUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )
    intent = OrderIntent(
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
    policy = PassiveExecutionPolicy()
    rate_limits = RateLimits(request_weight_1m=2400, orders_1m=1200, orders_10s=300)

    class BudgetClient:
        def __init__(self, order_10s):
            self.rate_state = type("S", (), {"order_count_10s": order_10s, "order_count_1m": 0, "used_weight_1m": 0})()
            self.orders: list = []

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"}}

        def book_ticker(self, s):  # noqa: ARG002
            return {"bidPrice": "100.00", "askPrice": "100.20"}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": 1}

        def cancel_order(self, *a, **k):  # noqa: ARG002
            return {}

        def query_order(self, *a, **k):  # noqa: ARG002
            return {"executedQty": "0"}

    client_high = BudgetClient(order_10s=250)
    assert _order_budget_exceeded(client_high, policy, rate_limits) is True
    assert _throttled_interval(client_high, 3.0, policy, rate_limits) > 3.0
    assert _order_budget_exceeded(client_high, policy, None) is False

    client_low = BudgetClient(order_10s=100)
    assert _order_budget_exceeded(client_low, policy, rate_limits) is False
    assert _throttled_interval(client_low, 3.0, policy, rate_limits) == 3.0


def test_SCENARIO_PARITY_01_slice_progress_no_stall(tmp_path):
    """SCENARIO_PARITY_01-slice-progress-no-stall"""
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.audit import AuditLog
    # Use 20 qty, price 100 => 2000 notional -> 4 slices (500 each) => slices 5 each?
    # Stub that instantly fills posted slice (GTX trade-through)
    from decimal import Decimal
    from src.live.filters import SymbolFilters
    from src.live.planner import OrderIntent
    filters = {
        "AAAUSDT": SymbolFilters(symbol="AAAUSDT", tick_size=Decimal("0.10"), step_size=Decimal("0.001"), min_qty=Decimal("0.001"), min_notional=Decimal("5"), max_qty=Decimal("1000000"), quantity_precision=3, price_precision=2)
    }
    intent = OrderIntent(symbol="AAAUSDT", side="BUY", quantity=Decimal("20"), reduce_only=False, target_qty=Decimal("20"), current_qty=Decimal("0"), client_order_prefix="20260101", leg_index=0, decision_price=Decimal("100"))
    # Touch where ask < price for GTX to fill instantly (trade-through)
    class SliceClient:
        def __init__(self):
            self.orders=[]
            self._tick=0
        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "99.00"}}
        def book_ticker(self, s):
            return {"bidPrice": "100.00", "askPrice": "99.00"}
        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}
        def cancel_order(self, *a, **k):
            return {}
        def query_order(self, symbol, oid):
            # find order quantity
            for o in self.orders:
                if o["newClientOrderId"]==oid:
                    return {"executedQty": str(o["quantity"]), "avgPrice": o["price"]}
            return {"executedQty": "0", "avgPrice": "0"}
        def open_orders(self):
            return []
    client = SliceClient()
    policy = PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=1000, window_deadline_s=6000, max_slices=4)
    clock_val = [0.0]
    def clock():
        v=clock_val[0]
        clock_val[0]+=3.0
        return v
    times=[]
    orig_new=client.new_order
    def rec(params):
        times.append(clock_val[0])
        return orig_new(params)
    client.new_order=rec
    audit=AuditLog(tmp_path/"p1.jsonl")
    outcomes=execute_intents(client,[intent],filters,policy,audit,clock,lambda s: None)
    # check intervals <=3.0 and all GTX
    assert len(client.orders)>=2
    for i in range(1,len(times)):
        assert times[i]-times[i-1] <= 3.0+1e-9
    assert all(o["timeInForce"]=="GTX" for o in client.orders)
    assert not any(o["timeInForce"]=="IOC" for o in client.orders)
    assert outcomes[0].status=="FILLED"
    assert outcomes[0].maker_qty==Decimal("20")
    assert outcomes[0].taker_qty==Decimal("0")

def test_SCENARIO_PARITY_02_avg_price_preferred(tmp_path):
    """SCENARIO_PARITY_02-avg-price-preferred"""
    from decimal import Decimal
    from src.live.filters import SymbolFilters
    from src.live.planner import OrderIntent
    from src.live.executor import PassiveExecutionPolicy
    from src.live.audit import AuditLog
    filters={"AAAUSDT": SymbolFilters(symbol="AAAUSDT", tick_size=Decimal("0.10"), step_size=Decimal("0.001"), min_qty=Decimal("0.001"), min_notional=Decimal("5"), max_qty=Decimal("1000000"), quantity_precision=3, price_precision=2)}
    intent=OrderIntent(symbol="AAAUSDT", side="BUY", quantity=Decimal("1"), reduce_only=False, target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="run1", leg_index=0, decision_price=Decimal("100"))
    class AvgClient:
        def __init__(self, with_avg):
            self.with_avg=with_avg
            self.orders=[]
        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "99.00", "askPrice": "100.00"}}
        def book_ticker(self,s):
            return {"bidPrice": "99.00", "askPrice": "100.00"}
        def new_order(self, p):
            # force IOC phase by using short passive_deadline and window
            self.orders.append(p)
            return {"orderId":1}
        def cancel_order(self,*a,**k):
            return {}
        def query_order(self,s,oid):
            if self.with_avg:
                return {"executedQty":"1", "avgPrice":"99.50"}
            else:
                return {"executedQty":"1"}
        def open_orders(self):
            return []
    # Test with avgPrice
    client=AvgClient(True)
    # Need to force IOC: set passive_deadline very small and make GTX not fill then IOC
    # Simplify: directly test _record_fill via execute_intents with paper? Instead test execute path with stub that returns IOC
    # We'll use a client that posts IOC immediately (phase ioc)
    policy=PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=0.1, window_deadline_s=600, taker_cap_bps=50, max_slices=1)
    # To get IOC, we need to let first poll timeout -> phase ioc
    clock_vals=[0,0,3]
    idx=[0]
    def clock():
        v=clock_vals[idx[0]] if idx[0]<len(clock_vals) else clock_vals[-1]
        idx[0]+=1
        return v
    audit=AuditLog(tmp_path/"p2.jsonl")
    # Use stub that simulates IOC filled with avgPrice
    # Our SliceClient logic needs to support IOC price not GTX band?
    # Instead we test _record_fill directly
    from src.live.executor import _IntentRuntime, _record_fill
    rt=_IntentRuntime(intent=intent, filters=filters["AAAUSDT"])
    rt.active_id="oid1"
    rt.active_price=Decimal("100.00")
    rt.phase="ioc"
    _record_fill(rt, Decimal("1"), avg_price=Decimal("99.50"))
    assert rt.fill_notional==Decimal("99.50")
    # fallback without avgPrice
    rt2=_IntentRuntime(intent=intent, filters=filters["AAAUSDT"])
    rt2.active_id="oid2"
    rt2.active_price=Decimal("100.00")
    rt2.phase="passive"
    _record_fill(rt2, Decimal("1"), avg_price=None)
    assert rt2.fill_notional==Decimal("100.00")


def test_simulate_immediate_taker_fills_fills_full_quantity_at_mid() -> None:
    from src.live.executor import FeeSchedule, PassiveExecutionPolicy, simulate_immediate_taker_fills
    from src.live.filters import _ZERO
    from src.live.planner import OrderIntent

    intent = OrderIntent(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("2"),
        reduce_only=False,
        target_qty=Decimal("2"),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("101"),
    )
    books = {"BTCUSDT": (Decimal("100"), Decimal("102"))}
    policy = PassiveExecutionPolicy(
        fee_schedule=FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0),
        taker_slippage_bps=3.0,
    )
    outcomes = simulate_immediate_taker_fills([intent], books, policy)
    assert len(outcomes) == 1
    oc = outcomes[0]
    assert oc.status == "FILLED"
    assert oc.filled_qty == Decimal("2")
    assert oc.unfilled_qty == _ZERO
    assert oc.avg_fill_price == Decimal("101")
    assert oc.chases == 0
    assert oc.latency_seconds == 0.0
    assert len(oc.fills) == 1
    assert oc.fills[0] == (Decimal("2"), Decimal("101"), 8.0, "immediate_taker", "taker")
    assert oc.taker_qty == Decimal("2")
    assert oc.maker_qty == _ZERO


def test_simulate_immediate_taker_fills_missing_book_returns_residual() -> None:
    from src.live.executor import FeeSchedule, PassiveExecutionPolicy, simulate_immediate_taker_fills
    from src.live.filters import _ZERO
    from src.live.planner import OrderIntent

    intent = OrderIntent(
        symbol="ETHUSDT",
        side="BUY",
        quantity=Decimal("1.5"),
        reduce_only=False,
        target_qty=Decimal("1.5"),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("100"),
    )
    policy = PassiveExecutionPolicy(
        fee_schedule=FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0),
        taker_slippage_bps=3.0,
    )
    outcomes = simulate_immediate_taker_fills([intent], {}, policy)
    oc = outcomes[0]
    assert oc.status == "RESIDUAL"
    assert oc.filled_qty == _ZERO
    assert oc.avg_fill_price is None
    assert oc.unfilled_qty == intent.quantity
    assert oc.fills == ()


def test_execute_intents_immediate_taker_bypasses_peg_chase_loop(tmp_path) -> None:
    from src.live.audit import AuditLog
    from src.live.executor import FeeSchedule, PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.planner import OrderIntent
    import json

    filt = SymbolFilters(
        symbol="BTCUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("1"),
        max_qty=Decimal("100000"),
        quantity_precision=3,
        price_precision=2,
    )
    intent = OrderIntent(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("1.0"),
        reduce_only=False,
        target_qty=Decimal("1.0"),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("100"),
    )
    policy = PassiveExecutionPolicy(
        fee_schedule=FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0),
        taker_slippage_bps=3.0,
    )

    class NoMutClient:
        def book_tickers(self):
            return {"BTCUSDT": {"bidPrice": "100", "askPrice": "102"}}

        def book_ticker(self, symbol):
            return {"bidPrice": "100", "askPrice": "102"}

        def new_order(self, params):
            raise AssertionError("new_order must not be called in immediate_taker")

        def query_order(self, *a, **k):
            raise AssertionError("query_order must not be called in immediate_taker")

        def cancel_order(self, *a, **k):
            raise AssertionError("cancel_order must not be called in immediate_taker")

    audit_path = tmp_path / "audit_immediate.jsonl"
    audit = AuditLog(audit_path)
    sleeps: list[float] = []

    def sleep_fn(s):
        sleeps.append(s)

    sink: list = []
    client = NoMutClient()
    outcomes = execute_intents(
        client,
        [intent],
        {"BTCUSDT": filt},
        policy,
        audit,
        lambda: 0.0,
        sleep_fn,
        outcome_sink=sink,
        paper_fill_model="immediate_taker",
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "FILLED"
    assert sleeps == []
    assert sink == list(outcomes)
    events = [json.loads(l) for l in audit_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert any(e.get("event") == "intent_outcome" for e in events)


def test_execute_intents_default_model_preserves_peg_chase_path(tmp_path) -> None:
    from src.live.audit import AuditLog
    from src.live.executor import FeeSchedule, PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.planner import OrderIntent

    filt = SymbolFilters(
        symbol="AAAUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("1"),
        max_qty=Decimal("100000"),
        quantity_precision=3,
        price_precision=2,
    )
    intent = OrderIntent(
        symbol="AAAUSDT",
        side="BUY",
        quantity=Decimal("1.0"),
        reduce_only=False,
        target_qty=Decimal("1.0"),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("100"),
    )
    policy = PassiveExecutionPolicy(
        poll_interval_s=3.0,
        passive_deadline_s=1000.0,
        window_deadline_s=6000.0,
        max_slices=1,
        fee_schedule=FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0),
        taker_slippage_bps=3.0,
    )

    class FillClient:
        def __init__(self):
            self.orders: list[dict] = []

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "99.00"}}

        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "99.00"}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, symbol, oid):
            # return filled quantity as posted
            for o in self.orders:
                if o["newClientOrderId"] == oid:
                    return {"executedQty": str(o["quantity"]), "avgPrice": o["price"]}
            return {"executedQty": "0", "avgPrice": "0"}

    audit = AuditLog(tmp_path / "peg.jsonl")
    client = FillClient()
    clock_val = [0.0]

    def clock():
        v = clock_val[0]
        clock_val[0] += 3.0
        return v

    outcomes = execute_intents(
        client,
        [intent],
        {"AAAUSDT": filt},
        policy,
        audit,
        clock,
        lambda s: None,
    )
    assert outcomes[0].status == "FILLED"
    # reasons should be maker_fill or backstop_taker/timeout_taker
    reasons = {r for _, _, _, r, _ in outcomes[0].fills}
    assert reasons.issubset({"maker_fill", "timeout_taker", "backstop_taker"})
    assert len(client.orders) >= 1
