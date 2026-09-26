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
    EXECUTION_BAR_SECONDS,
    FeeSchedule,
    PassiveExecutionPolicy,
    _capped_ioc_price,
    _slice_quantities,
    backtest_parity_execution_policy,
    execute_intent,
    execute_intents,
    strict_passive_execution_policy,
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
    deadline_client = StubClient(touches=[("99.90", "100.00")])
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
    # Fresh-book rail: the IOC crosses a genuine trend -- capped at the current
    # far touch plus taker cap, never at the stale decision price.
    ask = Decimal("100.05")
    assert ioc_price <= ask * (
        Decimal(1) + Decimal(str(policy.taker_cap_bps)) / Decimal(10_000)
    )
    assert ioc_price >= ask  # 마케터블
    assert outcome.status == "RESIDUAL"


def test_SCENARIO_LIVE_21_CHASE_EXHAUSTION_HOLDS_INSTEAD_OF_CROSSING(tmp_path) -> None:
    """호가가 매 폴 움직여도 chases 소진은 재페그 중단일 뿐이다: GTX 게시는
    정확히 max_chases+1 회, IOC 전이 없음(t=1080s 이전 무조건)."""
    touches = [(f"{100 + 0.3 * i:.2f}", f"{100.20 + 0.3 * i:.2f}") for i in range(15)]
    # IOC 시점의 호가는 스프레드 붕괴(own mid 대비 far touch +800bp 초과)라
    # fresh-book 레일이 거부한다: 점진적 추세는 크로싱되고, 깨진 호가만 대기한다.
    touches.append(("104.50", "114.00"))
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

    client = StubClient(touches=[("99.90", "100.00")])
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
    far touch(102.00)가 own-mid 레일 밖 스프레드 붕괴라 IOC 백스톱은 게시되지
    않고, 잔량 0.5 는 순수하게 취소 정산만으로 확정된다."""
    client = CancelFillStubClient(touches=[("100.00", "102.00")])
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
    """S7: far touch 가 own-mid ± max_cross_bps 레일 밖(스프레드 붕괴)이면
    None 이고, execute_intents 는 해당 intent 에 IOC 0건으로 RESIDUAL 로 종결한다."""
    assert _capped_ioc_price(
        Decimal("101.00"), is_buy=True, taker_cap_bps=8.0,
        tick_size=Decimal("0.10"), band_low=Decimal("99.50"), band_high=Decimal("100.50"),
    ) is None

    client = StubClient(touches=[("100.00", "102.00")])
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
    IOC backstop fills locally (non-zero filled_qty, status FILLED) while zero
    mutating requests reach the network; the equivalent SHADOW run stays at
    filled_qty == 0 / status SHADOW."""
    # 유효한 정적 호가: GTX는 미체결로 휴지하다 passive 마감 후 IOC 백스톱으로 체결된다.
    paper_client = PaperStubClient(touches=[("100.00", "100.20")])
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
    # 정적 유효 호가: ask(100.00) > 게시가(100.00 전후) -> GTX 미체결 유지.
    client = PaperStubClient(touches=[("99.90", "100.00")])
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
    paper_client = PaperStubClient(touches=[("100.00", "100.20")])
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
        passive_pricing="anchored",
    )

    class FillThenThrowClient:
        def __init__(self, exc_type):
            self.tick = 0
            self.exc_type = exc_type

        def book_tickers(self):
            self.tick += 1
            if self.tick == 3:
                raise self.exc_type("boom")
            if self.tick == 2:
                return {
                    "AAAUSDT": {"bidPrice": "99.40", "askPrice": "99.50"},
                    "BBBUSDT": {"bidPrice": "99.40", "askPrice": "99.50"},
                    "CCCUSDT": {"bidPrice": "100.00", "askPrice": "100.20"},
                }
            return {
                "AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"},
                "BBBUSDT": {"bidPrice": "100.00", "askPrice": "100.20"},
                "CCCUSDT": {"bidPrice": "100.00", "askPrice": "100.20"},
            }

        def book_ticker(self, symbol):  # noqa: ARG002
            return {"bidPrice": "100.00", "askPrice": "100.20"}

        def new_order(self, params):  # noqa: ARG002
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

        def cancel_order(self, *a, **k):  # noqa: ARG002
            return {}

        def query_order(self, *a, **k):  # noqa: ARG002
            return {"executedQty": "0"}

    intents = [make_intent("AAAUSDT"), make_intent("BBBUSDT"), make_intent("CCCUSDT")]
    filters = {"AAAUSDT": filt_a, "BBBUSDT": filt_b, "CCCUSDT": filt_a}
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
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"}}
        def book_ticker(self, s):
            return {"bidPrice": "100.00", "askPrice": "100.20"}
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
    _record_fill(rt, Decimal("1"), now=3.0, avg_price=Decimal("99.50"))
    assert rt.fill_notional==Decimal("99.50")
    # fallback without avgPrice
    rt2=_IntentRuntime(intent=intent, filters=filters["AAAUSDT"])
    rt2.active_id="oid2"
    rt2.active_price=Decimal("100.00")
    rt2.phase="passive"
    _record_fill(rt2, Decimal("1"), now=3.0, avg_price=None)
    assert rt2.fill_notional==Decimal("100.00")


def test_simulate_immediate_taker_fills_fills_full_quantity_at_mid() -> None:
    import pandas as pd

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
    outcomes = simulate_immediate_taker_fills([intent], books, policy, now=12.0)
    assert len(outcomes) == 1
    oc = outcomes[0]
    assert oc.status == "FILLED"
    assert oc.filled_qty == Decimal("2")
    assert oc.unfilled_qty == _ZERO
    assert oc.avg_fill_price == Decimal("101")
    assert oc.chases == 0
    assert oc.latency_seconds == 0.0
    assert len(oc.fills) == 1
    assert oc.fills[0] == (Decimal("2"), Decimal("101"), 8.0, "immediate_taker", "taker", pd.Timestamp(12.0, unit="s", tz="UTC"))
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
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"}}

        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "100.20"}

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
    reasons = {r for _, _, _, r, _, _ in outcomes[0].fills}
    assert reasons.issubset({"maker_fill", "timeout_taker", "backstop_taker"})
    assert len(client.orders) >= 1


def test_backtest_parity_policy_bounds_passive_phase_without_resting_order(tmp_path) -> None:
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.executor import EXECUTION_BAR_SECONDS, FeeSchedule, PassiveExecutionPolicy, backtest_parity_execution_policy, execute_intent

    policy = backtest_parity_execution_policy(FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0), 3.0)
    assert EXECUTION_BAR_SECONDS == 180.0
    assert policy.passive_deadline_s == EXECUTION_BAR_SECONDS
    assert policy.window_deadline_s == 2 * EXECUTION_BAR_SECONDS
    assert policy.taker_cap_bps == 8.0
    assert policy.taker_slippage_bps == 3.0

    client = StubClient(touches=[("100.20", "100.30")])
    outcome = execute_intent(
        client,
        _intent(),
        _filters(),
        PassiveExecutionPolicy(poll_interval_s=3.0, max_chases=3, passive_deadline_s=20.0, window_deadline_s=600.0, taker_cap_bps=15.0, max_slices=1),
        AuditLog(tmp_path / "parity.jsonl"),
        SteppingClock(5.0),
    )
    gtx_orders = [o for o in client.orders if o["timeInForce"] == "GTX"]
    ioc_orders = [o for o in client.orders if o["timeInForce"] == "IOC"]
    assert gtx_orders == []
    assert len(ioc_orders) >= 1
    assert all(Decimal(o["price"]) <= Decimal("100.50") for o in ioc_orders)
    assert outcome.status == "RESIDUAL"


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

def test_execute_intents_rate_limited_submission_retries_next_tick_with_new_seq(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent

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
            if len(self.posted) == 1:
                raise VenueError("rate limited", code=None, http_status=429, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"orderId": 2}

        def cancel_order(self, symbol, orig_client_order_id):
            return {}

        def query_order(self, symbol, orig_client_order_id):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()

    outcomes = execute_intents(client, [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, policy, audit,
                               clock, sleep_fn, journal=journal, attempt=attempt)

    assert [p["newClientOrderId"][-4:] for p in client.posted] == ["-0-0", "-0-1"]
    assert outcomes[0].status == "FILLED"
    assert "order_rate_limited" in _events()

def test_execute_intents_abort_cancels_and_settles_active_orders(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent

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
            if params["symbol"] == "BBBUSDT":
                raise VenueError("unknown rejection", code=-9999, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"orderId": 1}

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            return {"status": "CANCELED", "executedQty": "0.4", "avgPrice": "100.00"}

    client = _Client()
    filters = {"AAAUSDT": _filters("AAAUSDT"), "BBBUSDT": _filters("BBBUSDT")}

    with pytest.raises(VenueError) as exc_info:
        execute_intents(client, [_intent("AAAUSDT"), _intent("BBBUSDT")], filters, policy, audit,
                        clock, sleep_fn, journal=journal, attempt=attempt)

    aaa_id = client.posted[0]["newClientOrderId"]
    assert client.cancels == [aaa_id]
    partial = {o.symbol: o for o in exc_info.value.partial_outcomes}
    assert partial["AAAUSDT"].filled_qty == Decimal("0.4")
    assert partial["BBBUSDT"].filled_qty == Decimal("0")
    assert OrderJournal(tmp_path / "journal.jsonl").observed_qty(aaa_id) == Decimal("0.4")

def test_execute_intents_abort_cleanup_failure_does_not_mask_original_error(tmp_path) -> None:
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent

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
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            if params["symbol"] == "BBBUSDT":
                raise VenueError("unknown rejection", code=-9999, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"orderId": 1}

        def cancel_order(self, symbol, orig_client_order_id):
            raise ConnectionError("network down")

        def query_order(self, symbol, orig_client_order_id):
            return {"status": "NEW", "executedQty": "0"}

    filters = {"AAAUSDT": _filters("AAAUSDT"), "BBBUSDT": _filters("BBBUSDT")}

    with pytest.raises(VenueError) as exc_info:
        execute_intents(_Client(), [_intent("AAAUSDT"), _intent("BBBUSDT")], filters, policy, audit,
                        clock, sleep_fn, journal=journal, attempt=attempt)

    assert exc_info.value.code == -9999
    assert "abort_cleanup_failed" in _events()

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

def test_execute_intents_suppressed_paper_client_journals_submits(tmp_path) -> None:
    """Spec 03 INV-FILL-WAL: a mutation-suppressed PAPER client still journals its submissions (and fills),
    because the runner rebuilds the PAPER ledger from the journal alone; ids come from the journal sequence."""
    import json
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.filters import SymbolFilters
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent

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

    from src.live.rest import PaperResponse
    from src.live.settings import ExecutionMode

    short_policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=1.5, window_deadline_s=3.0)

    class _Client:
        mode = ExecutionMode.PAPER

        def __init__(self):
            self.posted: list[dict] = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.posted.append(params)
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "")

    client = _Client()

    execute_intents(client, [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, short_policy, audit,
                    clock, sleep_fn, journal=journal, attempt=attempt)

    assert client.posted[0]["newClientOrderId"].endswith("-0-0")
    assert journal.next_submit_seq() == len(client.posted) >= 1
    journal_text = (tmp_path / "journal.jsonl").read_text(encoding="utf-8")
    assert all(p["newClientOrderId"] in journal_text for p in client.posted)

def test_cancel_orphan_orders_settles_prior_day_and_legacy_orders_by_journal_delta(tmp_path) -> None:
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown

    audit = AuditLog(tmp_path / "orphan_audit.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    class _Client:
        def __init__(self, open_orders, executed, *, mode=None, cancel_unknown_status=None):
            self._open = open_orders
            self._executed = executed
            self.cancels: list[str] = []
            self._cancel_unknown_status = cancel_unknown_status
            if mode is not None:
                self.mode = mode

        def open_orders(self):
            return list(self._open)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            if self._cancel_unknown_status is not None:
                raise OrderStatusUnknown("cancel unknown", path="/fapi/v1/order", http_status=503, code=None)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            status = self._cancel_unknown_status or "CANCELED"
            return {"status": status, "side": "BUY", "avgPrice": "100",
                    "executedQty": self._executed.get(orig_client_order_id, "0")}

    prior = "mh20260913-ABCDEFGHIJ-0-0-7"
    booked = "mh20260914-KLMNOPQRST-1-0-8"
    legacy = "20260912-BBBUSDT-0-0-1"
    import pandas as pd

    now = pd.Timestamp("2026-09-14T00:00:00Z")
    journal.record_submit(
        prior, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("4"), reduce_only=False, leg_index=0,
    )
    journal.record_fill(
        kind="execution", attempt_seq=attempt.attempt_seq, symbol="AAAUSDT", side="BUY",
        quantity=Decimal("3"), price=Decimal("100"), fee_bps=2.0, liquidity="maker",
        reason="maker_fill", filled_at=now, client_order_id=prior, leg_index=0,
        cumulative_executed_qty=Decimal("3"), simulated=False,
    )
    journal.record_submit(
        booked, "CCCUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("4"), reduce_only=False, leg_index=1,
    )
    journal.record_fill(
        kind="execution", attempt_seq=attempt.attempt_seq, symbol="CCCUSDT", side="BUY",
        quantity=Decimal("4"), price=Decimal("100"), fee_bps=2.0, liquidity="maker",
        reason="maker_fill", filled_at=now, client_order_id=booked, leg_index=1,
        cumulative_executed_qty=Decimal("4"), simulated=False,
    )
    client = _Client(
        [{"symbol": "AAAUSDT", "clientOrderId": prior}, {"symbol": "CCCUSDT", "clientOrderId": booked},
         {"symbol": "BBBUSDT", "clientOrderId": legacy}],
        {prior: "4", booked: "4", legacy: "2"},
    )

    settlements = cancel_orphan_orders(client, "20260914", audit, journal=journal, now=now, taker_fee_bps=4.5)

    assert client.cancels == [prior, booked, legacy]
    assert [(s.symbol, s.quantity) for s in settlements] == [("AAAUSDT", Decimal("1")), ("BBBUSDT", Decimal("2"))]
    assert all(s.kind == "orphan_settlement" and s.simulated is False for s in settlements)
    assert (settlements[0].side, settlements[0].price) == ("BUY", Decimal("100"))
    assert settlements[0].cumulative_executed_qty == Decimal("4")
    # Both submitted ids are terminal now, so restart recovery has nothing to query.
    assert journal.unresolved_submits(since=now - pd.Timedelta(hours=72)) == ()

def test_cancel_orphan_orders_reports_foreign_symbols_without_cancelling_them(tmp_path) -> None:
    from src.live.audit import AuditLog
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown
    from src.live.settings import ExecutionMode

    audit = AuditLog(tmp_path / "orphan_audit.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    class _Client:
        def __init__(self, open_orders, executed, *, mode=None, cancel_unknown_status=None):
            self._open = open_orders
            self._executed = executed
            self.cancels: list[str] = []
            self._cancel_unknown_status = cancel_unknown_status
            if mode is not None:
                self.mode = mode

        def open_orders(self):
            return list(self._open)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            if self._cancel_unknown_status is not None:
                raise OrderStatusUnknown("cancel unknown", path="/fapi/v1/order", http_status=503, code=None)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            status = self._cancel_unknown_status or "CANCELED"
            return {"status": status, "side": "BUY", "avgPrice": "100",
                    "executedQty": self._executed.get(orig_client_order_id, "0")}

    import pandas as pd

    now = pd.Timestamp("2026-09-14T00:00:00Z")
    client = _Client(
        [{"symbol": "AAAUSDT", "clientOrderId": "mh20260914-ABCDEFGHIJ-0-0-1"},
         {"symbol": "BBBUSDT", "clientOrderId": "web_manual_123"}],
        {},
        mode=ExecutionMode.LIVE_TESTNET,
    )

    sweep = cancel_orphan_orders(client, "20260914", audit, journal=journal, now=now, taker_fee_bps=4.5)

    assert sweep.foreign_symbols == ("BBBUSDT",)
    assert client.cancels == ["mh20260914-ABCDEFGHIJ-0-0-1"]

def test_cancel_orphan_orders_ignores_foreign_orders_in_suppressed_mode(tmp_path) -> None:
    from src.live.audit import AuditLog
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown
    from src.live.settings import ExecutionMode

    audit = AuditLog(tmp_path / "orphan_audit.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    class _Client:
        def __init__(self, open_orders, executed, *, mode=None, cancel_unknown_status=None):
            self._open = open_orders
            self._executed = executed
            self.cancels: list[str] = []
            self._cancel_unknown_status = cancel_unknown_status
            if mode is not None:
                self.mode = mode

        def open_orders(self):
            return list(self._open)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            if self._cancel_unknown_status is not None:
                raise OrderStatusUnknown("cancel unknown", path="/fapi/v1/order", http_status=503, code=None)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            status = self._cancel_unknown_status or "CANCELED"
            return {"status": status, "side": "BUY", "avgPrice": "100",
                    "executedQty": self._executed.get(orig_client_order_id, "0")}

    client = _Client([{"symbol": "BBBUSDT", "clientOrderId": "web_manual_123"}], {}, mode=ExecutionMode.PAPER)

    import pandas as pd

    sweep = cancel_orphan_orders(client, "20260914", audit, journal=journal, now=pd.Timestamp("2026-09-14T00:00:00Z"), taker_fee_bps=4.5)

    assert sweep == []
    assert sweep.foreign_symbols == ("BBBUSDT",)
    assert client.cancels == []
    assert "foreign_open_order" in (tmp_path / "orphan_audit.jsonl").read_text(encoding="utf-8")

def test_cancel_orphan_orders_cancel_status_unknown_resolved_by_lookup(tmp_path) -> None:
    import pandas as pd
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown

    audit = AuditLog(tmp_path / "orphan_audit.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    class _Client:
        def __init__(self, open_orders, executed, *, mode=None, cancel_unknown_status=None):
            self._open = open_orders
            self._executed = executed
            self.cancels: list[str] = []
            self._cancel_unknown_status = cancel_unknown_status
            if mode is not None:
                self.mode = mode

        def open_orders(self):
            return list(self._open)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            if self._cancel_unknown_status is not None:
                raise OrderStatusUnknown("cancel unknown", path="/fapi/v1/order", http_status=503, code=None)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            status = self._cancel_unknown_status or "CANCELED"
            return {"status": status, "side": "BUY", "avgPrice": "100",
                    "executedQty": self._executed.get(orig_client_order_id, "0")}

    import pytest

    order = "mh20260914-ABCDEFGHIJ-0-0-3"
    closed = _Client([{"symbol": "AAAUSDT", "clientOrderId": order}], {order: "1"}, cancel_unknown_status="CANCELED")

    settlements = cancel_orphan_orders(closed, "20260914", audit, journal=journal, now=pd.Timestamp("2026-09-14T00:00:00Z"), taker_fee_bps=4.5)

    assert closed.cancels == [order]
    assert [s.quantity for s in settlements] == [Decimal("1")]

    still_open = _Client([{"symbol": "AAAUSDT", "clientOrderId": order}], {order: "0"}, cancel_unknown_status="NEW")
    with pytest.raises(OrderStatusUnknown):
        cancel_orphan_orders(still_open, "20260914", audit, journal=journal, now=pd.Timestamp("2026-09-14T00:00:00Z"), taker_fee_bps=4.5)
    assert still_open.cancels == [order]

def test_cancel_orphan_orders_tolerates_order_gone_on_lookup(tmp_path) -> None:
    import pandas as pd
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown

    audit = AuditLog(tmp_path / "orphan_audit.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    class _Client:
        def __init__(self, open_orders, executed, *, mode=None, cancel_unknown_status=None):
            self._open = open_orders
            self._executed = executed
            self.cancels: list[str] = []
            self._cancel_unknown_status = cancel_unknown_status
            if mode is not None:
                self.mode = mode

        def open_orders(self):
            return list(self._open)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            if self._cancel_unknown_status is not None:
                raise OrderStatusUnknown("cancel unknown", path="/fapi/v1/order", http_status=503, code=None)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            status = self._cancel_unknown_status or "CANCELED"
            return {"status": status, "side": "BUY", "avgPrice": "100",
                    "executedQty": self._executed.get(orig_client_order_id, "0")}

    order = "mh20260914-ABCDEFGHIJ-0-0-3"

    class _GoneClient(_Client):
        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            raise VenueError("unknown order", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

        def query_order(self, symbol, orig_client_order_id):
            raise VenueError("unknown order", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    client = _GoneClient([{"symbol": "AAAUSDT", "clientOrderId": order}], {})

    assert cancel_orphan_orders(client, "20260914", audit, journal=journal, now=pd.Timestamp("2026-09-14T00:00:00Z"), taker_fee_bps=4.5) == []
    assert client.cancels == [order]

def test_cancel_orphan_orders_propagates_non_benign_cancel_rejection(tmp_path) -> None:
    import pandas as pd
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown

    audit = AuditLog(tmp_path / "orphan_audit.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    class _Client:
        def __init__(self, open_orders, executed, *, mode=None, cancel_unknown_status=None):
            self._open = open_orders
            self._executed = executed
            self.cancels: list[str] = []
            self._cancel_unknown_status = cancel_unknown_status
            if mode is not None:
                self.mode = mode

        def open_orders(self):
            return list(self._open)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            if self._cancel_unknown_status is not None:
                raise OrderStatusUnknown("cancel unknown", path="/fapi/v1/order", http_status=503, code=None)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            status = self._cancel_unknown_status or "CANCELED"
            return {"status": status, "side": "BUY", "avgPrice": "100",
                    "executedQty": self._executed.get(orig_client_order_id, "0")}

    import pytest

    order = "mh20260914-ABCDEFGHIJ-0-0-3"

    class _RejectingClient(_Client):
        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            raise VenueError("bad signature", code=-1022, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    client = _RejectingClient([{"symbol": "AAAUSDT", "clientOrderId": order}], {order: "1"})

    with pytest.raises(VenueError) as exc_info:
        cancel_orphan_orders(client, "20260914", audit, journal=journal, now=pd.Timestamp("2026-09-14T00:00:00Z"), taker_fee_bps=4.5)

    assert exc_info.value.code == -1022

def test_cancel_orphan_orders_falls_back_to_open_order_avg_price(tmp_path) -> None:
    import pandas as pd
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal
    from src.live.rest import OrderStatusUnknown

    audit = AuditLog(tmp_path / "orphan_audit.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    class _Client:
        def __init__(self, open_orders, executed, *, mode=None, cancel_unknown_status=None):
            self._open = open_orders
            self._executed = executed
            self.cancels: list[str] = []
            self._cancel_unknown_status = cancel_unknown_status
            if mode is not None:
                self.mode = mode

        def open_orders(self):
            return list(self._open)

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            if self._cancel_unknown_status is not None:
                raise OrderStatusUnknown("cancel unknown", path="/fapi/v1/order", http_status=503, code=None)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            status = self._cancel_unknown_status or "CANCELED"
            return {"status": status, "side": "BUY", "avgPrice": "100",
                    "executedQty": self._executed.get(orig_client_order_id, "0")}

    order = "mh20260914-ABCDEFGHIJ-0-0-3"

    class _NoAvgClient(_Client):
        def query_order(self, symbol, orig_client_order_id):
            return {"status": "CANCELED", "side": "SELL", "executedQty": "2"}

    client = _NoAvgClient([{"symbol": "AAAUSDT", "clientOrderId": order, "avgPrice": "99.5"}], {})

    settlements = cancel_orphan_orders(client, "20260914", audit, journal=journal, now=pd.Timestamp("2026-09-14T00:00:00Z"), taker_fee_bps=4.5)

    assert [(s.side, s.quantity, s.price) for s in settlements] == [("SELL", Decimal("2"), Decimal("99.5"))]

def test_execute_intents_unknown_ioc_submission_adopted_without_second_send(tmp_path) -> None:
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

    ioc_policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=0.0, window_deadline_s=20.0)

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
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.20"}

    client = _Client()

    outcomes = execute_intents(client, [_intent("AAAUSDT")], {"AAAUSDT": _filters("AAAUSDT")}, ioc_policy, audit,
                               clock, sleep_fn, journal=journal, attempt=attempt)

    assert [p["timeInForce"] for p in client.posted] == ["IOC"]
    assert outcomes[0].status == "FILLED"

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


def _anchored_policy(**overrides: object) -> PassiveExecutionPolicy:
    base: dict[str, object] = {
        "poll_interval_s": 3.0,
        "passive_deadline_s": 500.0,
        "window_deadline_s": 600.0,
        "taker_cap_bps": 8.0,
        "max_slices": 1,
        "passive_pricing": "anchored",
    }
    base.update(overrides)
    return PassiveExecutionPolicy(**base)  # type: ignore[arg-type]


def _sell_intent() -> OrderIntent:
    return OrderIntent(
        symbol="AAAUSDT",
        side="SELL",
        quantity=Decimal("1.000"),
        reduce_only=False,
        target_qty=Decimal("1.000"),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("100"),
    )


def test_strict_passive_policy_derives_timing_from_timeout() -> None:
    """Timeout 30 gives a 1800s passive phase, a two-bar window, and the taker cap."""
    schedule = FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0)
    policy = strict_passive_execution_policy(schedule, 3.0, 30)

    assert policy.passive_deadline_s == 1800.0
    assert policy.window_deadline_s == 1800.0 + 2 * EXECUTION_BAR_SECONDS
    assert policy.taker_cap_bps == 8.0
    assert policy.passive_pricing == "anchored"


def test_strict_passive_policy_rejects_non_positive_timeout() -> None:
    """A zero or negative passive timeout fails closed."""
    schedule = FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0)
    with pytest.raises(ValueError, match="passive_timeout_minutes"):
        strict_passive_execution_policy(schedule, 3.0, 0)
    with pytest.raises(ValueError, match="passive_timeout_minutes"):
        strict_passive_execution_policy(schedule, 3.0, -5)


def test_passive_execution_policy_rejects_unknown_pricing() -> None:
    """A passive_pricing outside the closed set fails closed."""
    with pytest.raises(ValueError, match="passive_pricing"):
        PassiveExecutionPolicy(passive_pricing="mid")  # type: ignore[arg-type]


def test_anchored_buy_rests_at_decision_price_inside_spread(tmp_path) -> None:
    """An anchor inside the spread rests at the decision price as GTX."""
    client = StubClient(touches=[("99.00", "101.00")])
    execute_intent(
        client, _intent(), _filters(tick_size="0.10"), _anchored_policy(),
        AuditLog(tmp_path / "anchored_buy.jsonl"), SteppingClock(15.0),
    )

    assert Decimal(client.orders[0]["price"]) == Decimal("100.0")
    assert client.orders[0]["timeInForce"] == "GTX"


def test_anchored_buy_clamps_below_ask_when_anchor_would_cross(tmp_path) -> None:
    """An anchor above the ask rests one tick below the ask, still post-only."""
    client = StubClient(touches=[("99.00", "99.50")])
    execute_intent(
        client, _intent(), _filters(tick_size="0.10"), _anchored_policy(),
        AuditLog(tmp_path / "anchored_clamp.jsonl"), SteppingClock(15.0),
    )

    assert Decimal(client.orders[0]["price"]) == Decimal("99.4")
    assert client.orders[0]["timeInForce"] == "GTX"


def test_anchored_sell_mirrors(tmp_path) -> None:
    """A sell anchor below the bid rests one tick above the bid."""
    client = StubClient(touches=[("100.50", "101.00")])
    execute_intent(
        client, _sell_intent(), _filters(tick_size="0.10"), _anchored_policy(),
        AuditLog(tmp_path / "anchored_sell.jsonl"), SteppingClock(15.0),
    )

    assert Decimal(client.orders[0]["price"]) == Decimal("100.6")
    assert client.orders[0]["timeInForce"] == "GTX"


def test_anchored_skips_non_positive_price(tmp_path) -> None:
    """An anchor derived at or below zero is never posted; a dust book that far
    from its own mid is a broken quote, so the IOC backstop refuses too."""
    client = StubClient(touches=[("0.01", "0.05")])
    outcome = execute_intent(
        client, _intent(), _filters(tick_size="0.10"), _anchored_policy(),
        AuditLog(tmp_path / "anchored_zero.jsonl"), SteppingClock(15.0),
    )

    assert client.orders == []
    assert outcome.status == "RESIDUAL"


def test_anchored_never_chases_book_moves(tmp_path) -> None:
    """Rising quotes before the deadline cause no cancel, no repost, and no chase."""
    touches = [(f"{100.00 + 0.10 * i:.2f}", f"{101.20 + 0.10 * i:.2f}") for i in range(12)]
    client = StubClient(touches=touches)
    outcome = execute_intent(
        client, _intent(), _filters(tick_size="0.10"), _anchored_policy(),
        AuditLog(tmp_path / "anchored_nochase.jsonl"), SteppingClock(15.0),
    )

    gtx_posts = [o for o in client.orders if o["timeInForce"] == "GTX"]
    assert len(gtx_posts) == 1
    assert len(client.cancels) == 1
    assert outcome.chases == 0


def test_anchored_escalates_to_capped_ioc_at_deadline(tmp_path) -> None:
    """Past the passive deadline the remainder crosses once via the capped IOC backstop."""
    client = StubClient(touches=[("99.90", "100.00")])
    policy = _anchored_policy(passive_deadline_s=20.0)
    execute_intent(
        client, _intent(), _filters(tick_size="0.10"), policy,
        AuditLog(tmp_path / "anchored_ioc.jsonl"), SteppingClock(15.0),
    )

    ioc_orders = [o for o in client.orders if o["timeInForce"] == "IOC"]
    assert len(ioc_orders) >= 1
    cap = Decimal("100.00") * (Decimal(1) + Decimal(str(policy.taker_cap_bps)) / Decimal(10_000))
    for order in ioc_orders:
        assert Decimal(order["price"]) <= cap
    assert all(o["type"] == "LIMIT" for o in client.orders)


def test_anchored_paper_fill_is_maker_on_trade_through(tmp_path) -> None:
    """A resting anchored GTX fills as maker at its own price when the ask trades through."""
    client = PaperStubClient(touches=[("99.00", "101.00"), ("99.00", "99.90")])
    outcome = execute_intent(
        client, _intent(), _filters(tick_size="0.10"), _anchored_policy(),
        AuditLog(tmp_path / "anchored_paper.jsonl"), SteppingClock(3.0),
    )

    assert outcome.status == "FILLED"
    assert outcome.filled_qty == Decimal("1.000")
    assert outcome.avg_fill_price == Decimal("100.0")
    assert outcome.fills[0][4] == "maker"
    assert outcome.fills[0][1] == Decimal("100.0")


def test_touch_chase_behaviour_unchanged(tmp_path) -> None:
    """The default touch-chase policy keeps its chase-then-IOC shape bit-identically."""
    client = StubClient()
    outcome = execute_intent(
        client, _intent(), _filters(), _policy(), AuditLog(tmp_path / "parity09.jsonl"), SteppingClock(15.0)
    )
    gtx_posts = [o for o in client.orders if o["timeInForce"] == "GTX"]
    assert len(client.cancels) >= 1
    assert 0 <= len(gtx_posts) - 1 <= _policy().max_chases
    assert outcome.unfilled_qty > 0
    assert all(o["type"] == "LIMIT" for o in client.orders)

    schedule = FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0)
    parity = backtest_parity_execution_policy(schedule, 3.0)
    assert parity.passive_pricing == "touch_chase"
    assert parity.passive_deadline_s == EXECUTION_BAR_SECONDS



def _fill_events(audit_path):
    import json as _json

    return [
        _json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
        if _json.loads(line)["event"] == "fill"
    ]


def _cancel_events(audit_path):
    import json as _json

    return [
        _json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
        if _json.loads(line)["event"] == "order_cancelled"
    ]


def _posted_events(audit_path):
    import json as _json

    return [
        _json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
        if _json.loads(line)["event"] == "order_posted"
    ]


def _intent_for(symbol: str, qty: str = "1.000") -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side="BUY",
        quantity=Decimal(qty),
        reduce_only=False,
        target_qty=Decimal(qty),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("100"),
    )


def _filters_for(symbol: str) -> SymbolFilters:
    return SymbolFilters(
        symbol=symbol,
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )


def test_order_fill_events_reconcile_with_outcome(tmp_path) -> None:
    """Per-intent fill quantities and VWAP reconcile exactly with the outcome."""
    from src.live.rest import PaperResponse
    from src.live.settings import ExecutionMode

    audit_path = tmp_path / "fills_audit.jsonl"
    audit = AuditLog(audit_path)
    books_by_tick = [
        {"AAAUSDT": ("100.00", "100.20"), "BBBUSDT": ("100.00", "100.20")},
        {"AAAUSDT": ("99.40", "99.50"), "BBBUSDT": ("100.30", "100.50")},
        {"AAAUSDT": ("99.40", "99.50"), "BBBUSDT": ("99.40", "99.50")},
    ]

    class _PaperBookClient:
        mode = ExecutionMode.PAPER

        def __init__(self):
            self.tick = -1

        def book_tickers(self):
            self.tick += 1
            snap = books_by_tick[min(self.tick, len(books_by_tick) - 1)]
            return {s: {"bidPrice": b, "askPrice": a} for s, (b, a) in snap.items()}

        def new_order(self, params):
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, *a, **k):
            raise AssertionError("paper path never queries")

    policy = PassiveExecutionPolicy(
        poll_interval_s=1.0, chase_ticks=2, max_chases=8,
        passive_deadline_s=600.0, window_deadline_s=3600.0,
        taker_cap_bps=15.0, max_slices=1,
        chase_band_bps=100.0, max_cross_bps=200.0,
        passive_pricing="anchored",
    )
    intents = [_intent_for("AAAUSDT"), _intent_for("BBBUSDT")]
    filters = {"AAAUSDT": _filters_for("AAAUSDT"), "BBBUSDT": _filters_for("BBBUSDT")}
    outcomes = execute_intents(
        _PaperBookClient(), intents, filters, policy, audit, SteppingClock(5.0), lambda _s: None,
    )
    by_symbol = {oc.symbol: oc for oc in outcomes}
    assert by_symbol["AAAUSDT"].filled_qty == Decimal("1.000")
    assert by_symbol["BBBUSDT"].filled_qty == Decimal("1.000")
    fills = _fill_events(audit_path)
    assert len(fills) == 2
    for record in fills:
        assert record["simulated"] is True
        assert "bid" in record
        assert "ask" in record
        outcome = by_symbol[record["symbol"]]
        sym_fills = [r for r in fills if r["symbol"] == record["symbol"]]
        assert sum(Decimal(r["qty"]) for r in sym_fills) == outcome.filled_qty
        vwap = sum(Decimal(r["qty"]) * Decimal(r["price"]) for r in sym_fills) / sum(
            Decimal(r["qty"]) for r in sym_fills
        )
        assert vwap == outcome.avg_fill_price
    aaa = next(r for r in fills if r["symbol"] == "AAAUSDT")
    assert (aaa["bid"], aaa["ask"], aaa["liquidity"]) == ("99.40", "99.50", "maker")
    bbb = next(r for r in fills if r["symbol"] == "BBBUSDT")
    assert (bbb["bid"], bbb["ask"], bbb["price"]) == ("99.40", "99.50", "100")


def test_order_cancelled_chase_then_passive_timeout(tmp_path) -> None:
    """Chase and timeout cancels are recorded in time order."""
    audit_path = tmp_path / "cancel_audit.jsonl"
    audit = AuditLog(audit_path)
    touches = [("100.00", "100.20"), ("100.30", "100.50")]

    class _MovingClient(StubClient):
        def __init__(self):
            super().__init__(touches=[touches[0]])
            self._tick = -1

        def book_tickers(self):
            self._tick += 1
            bid, ask = touches[1] if self._tick >= 1 else touches[0]
            return {"AAAUSDT": {"bidPrice": bid, "askPrice": ask}}

    policy = _policy(
        poll_interval_s=10.0, passive_deadline_s=15.0, window_deadline_s=60.0,
        chase_band_bps=50.0, max_cross_bps=100.0,
    )
    outcome = execute_intent(
        _MovingClient(), _intent(), _filters(), policy, audit, SteppingClock(10.0),
    )
    reasons = [record["reason"] for record in _cancel_events(audit_path)]
    assert "chase" in reasons
    assert "passive_timeout" in reasons
    assert reasons.index("chase") < reasons.index("passive_timeout")
    assert reasons[-1] == "window_end"
    assert outcome.status == "RESIDUAL"


def test_order_posted_carries_pricing_touch(tmp_path) -> None:
    """The posted order records the touch it priced against."""
    from src.live.rest import PaperResponse
    from src.live.settings import ExecutionMode

    audit_path = tmp_path / "posted_audit.jsonl"
    audit = AuditLog(audit_path)

    class _StaticPaperClient:
        mode = ExecutionMode.PAPER

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, *a, **k):
            raise AssertionError("paper path never queries")

    policy = _policy(poll_interval_s=15.0, passive_deadline_s=45.0, window_deadline_s=60.0)
    outcome = execute_intent(
        _StaticPaperClient(), _intent(), _filters(), policy, audit, SteppingClock(15.0),
    )
    assert outcome.status == "RESIDUAL"
    posted = _posted_events(audit_path)
    assert len(posted) == 1
    assert (posted[0]["bid"], posted[0]["ask"], posted[0]["phase"]) == ("100.00", "100.20", "passive")


def test_live_partial_fills_emit_deltas(tmp_path) -> None:
    """Live executedQty increases emit one fill per delta."""
    audit_path = tmp_path / "partial_audit.jsonl"
    audit = AuditLog(audit_path)
    touches = [("100.00", "100.20")] * 3 + [("100.30", "100.50")] * 100

    class _PartialClient:
        def __init__(self):
            self.tick = -1
            self.orders: list[dict] = []
            self._executed = ["3", "5", "6"]

        def book_tickers(self):
            self.tick += 1
            bid, ask = touches[min(self.tick, len(touches) - 1)]
            return {"AAAUSDT": {"bidPrice": bid, "askPrice": ask}}

        def book_ticker(self, symbol):
            bid, ask = touches[min(max(self.tick, 0), len(touches) - 1)]
            return {"bidPrice": bid, "askPrice": ask}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, symbol, orig_client_order_id):
            return {}

        def query_order(self, symbol, orig_client_order_id):
            executed = self._executed.pop(0) if self._executed else "0"
            return {"status": "NEW", "executedQty": executed}

    policy = PassiveExecutionPolicy(
        poll_interval_s=1.0, chase_ticks=2, max_chases=8,
        passive_deadline_s=30.0, window_deadline_s=120.0,
        taker_cap_bps=15.0, max_slices=1, max_ioc_attempts=1000,
    )
    outcome = execute_intent(
        _PartialClient(), _intent_for("AAAUSDT", "10.000"), _filters(), policy, audit, SteppingClock(1.0),
    )
    assert outcome.filled_qty == Decimal("6")
    fills = _fill_events(audit_path)
    assert [r["qty"] for r in fills] == ["3", "2", "1"]
    assert all(r["simulated"] is False for r in fills)
    assert all("bid" in r and "ask" in r for r in fills)
    assert sum(Decimal(r["qty"]) for r in fills) == outcome.filled_qty
    cancels = _cancel_events(audit_path)
    assert cancels[-1]["reason"] == "window_end"
    assert "bid" not in cancels[-1]
    assert "ask" not in cancels[-1]


def test_trend_during_passive_window_is_still_crossed(tmp_path) -> None:
    """Fresh-book rail: decision 100에서 +80bp 추세도 IOC로 크로싱된다."""
    from src.live.executor import PassiveExecutionPolicy

    client = PaperStubClient(
        touches=[("100.00", "100.05")] * 3 + [("100.80", "100.82")] * 50
    )
    policy = _policy(passive_deadline_s=9.0, window_deadline_s=600.0)
    outcome = execute_intent(
        client, _intent(), _filters(), policy,
        AuditLog(tmp_path / "trend.jsonl"), SteppingClock(3.0),
    )
    ioc_posts = [o for o in client.suppressed_attempts if o["timeInForce"] == "IOC"]
    assert ioc_posts
    ask = Decimal("100.82")
    cap = ask * (Decimal(1) + Decimal(str(policy.taker_cap_bps)) / Decimal(10_000))
    for order in ioc_posts:
        assert Decimal(order["price"]) >= ask
        assert Decimal(order["price"]) <= cap
    assert outcome.status == "FILLED"
    assert isinstance(policy, PassiveExecutionPolicy)


def test_broken_book_is_refused(tmp_path) -> None:
    """Own-mid 대비 far touch +125bp 스프레드 붕괴는 IOC 없이 RESIDUAL이다."""
    client = PaperStubClient(touches=[("99.00", "101.50")])
    policy = _policy(passive_deadline_s=9.0, window_deadline_s=120.0)
    outcome = execute_intent(
        client, _intent(), _filters(), policy,
        AuditLog(tmp_path / "broken.jsonl"), SteppingClock(3.0),
    )
    assert [o for o in client.suppressed_attempts if o["timeInForce"] == "IOC"] == []
    assert outcome.status == "RESIDUAL"
    assert outcome.unfilled_qty > 0


def test_sell_side_mirrors_trend_crossing(tmp_path) -> None:
    """매도: 결정가 대비 -60bp 타이트 북도 IOC로 크로싱된다."""
    from src.live.planner import OrderIntent

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
    client = PaperStubClient(touches=[("99.38", "99.40")] * 50)
    policy = _policy(passive_deadline_s=9.0, window_deadline_s=600.0)
    outcome = execute_intent(
        client, sell_intent, _filters(), policy,
        AuditLog(tmp_path / "sell_trend.jsonl"), SteppingClock(3.0),
    )
    assert [o for o in client.suppressed_attempts if o["timeInForce"] == "IOC"]
    assert outcome.status == "FILLED"


def test_fills_carry_confirmation_time(tmp_path) -> None:
    """3초 폴 시계에서 5번째 폴의 메이커 체결 filled_at은 start+15s다."""
    import pandas as pd

    from src.live.executor import PassiveExecutionPolicy

    client = PaperStubClient(
        touches=[("100.00", "100.05")] * 4 + [("99.40", "99.50")] * 50
    )
    policy = PassiveExecutionPolicy(
        poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0,
        taker_cap_bps=15.0, max_slices=1, passive_pricing="anchored",
    )
    outcome = execute_intent(
        client, _intent(), _filters(tick_size="0.01"), policy,
        AuditLog(tmp_path / "fill_time.jsonl"), SteppingClock(3.0),
    )
    assert outcome.status == "FILLED"
    assert len(outcome.fills) == 1
    filled_at = outcome.fills[0][5]
    assert filled_at == pd.Timestamp(15.0, unit="s", tz="UTC")
    assert filled_at != pd.Timestamp("2026-08-24 00:00", tz="UTC")
    assert filled_at.tzinfo is not None


def test_every_fill_path_stamps_time(tmp_path) -> None:
    """resting-maker, partial-on-cancel, IOC, immediate-taker 모두 6필드 UTC 스탬프다."""
    import pandas as pd

    from src.live.executor import (
        FeeSchedule,
        PassiveExecutionPolicy,
        simulate_immediate_taker_fills,
    )

    anchored = PassiveExecutionPolicy(
        poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0,
        taker_cap_bps=15.0, max_slices=1, passive_pricing="anchored",
    )
    maker_client = PaperStubClient(
        touches=[("100.00", "100.05")] * 4 + [("99.40", "99.50")] * 50
    )
    maker_outcome = execute_intent(
        maker_client, _intent(), _filters(tick_size="0.01"), anchored,
        AuditLog(tmp_path / "stamp_maker.jsonl"), SteppingClock(3.0),
    )
    cancel_client = CancelFillStubClient(touches=[("100.00", "102.00")])
    cancel_outcome = execute_intent(
        cancel_client, _intent(), _filters(), _policy(passive_deadline_s=20.0, window_deadline_s=600.0),
        AuditLog(tmp_path / "stamp_cancel.jsonl"), SteppingClock(3.0),
    )
    trend_client = PaperStubClient(
        touches=[("100.00", "100.05")] * 3 + [("100.80", "100.82")] * 50
    )
    trend_outcome = execute_intent(
        trend_client, _intent(), _filters(), _policy(passive_deadline_s=9.0, window_deadline_s=600.0),
        AuditLog(tmp_path / "stamp_ioc.jsonl"), SteppingClock(3.0),
    )
    immediate = simulate_immediate_taker_fills(
        [_intent()], {"AAAUSDT": (Decimal("100"), Decimal("102"))},
        PassiveExecutionPolicy(
            fee_schedule=FeeSchedule(maker_fee_bps=2.0, taker_fee_bps=5.0),
            taker_slippage_bps=3.0,
        ),
        now=12.0,
    )[0]
    for outcome in (maker_outcome, cancel_outcome, trend_outcome, immediate):
        assert outcome.fills
        for fill in outcome.fills:
            assert len(fill) == 6
            stamped = fill[5]
            assert isinstance(stamped, pd.Timestamp)
            assert stamped.tzinfo is not None
            assert str(stamped.tzinfo) == "UTC"


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


def _exec_events(audit_path):
    import json
    return [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]


def test_book_fetch_transient_skips_single_tick(tmp_path) -> None:
    """A single book-fetch transient failure skips one tick only."""
    from src.live.audit import AuditLog
    from src.live.errors import TransientReadError
    from src.live.executor import PassiveExecutionPolicy, execute_intents

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=50.0, window_deadline_s=200.0)
    audit_path = tmp_path / "tick.jsonl"
    audit = AuditLog(audit_path)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(s):
        clock_state[0] += s

    class _Client:
        def __init__(self):
            self.n = 0
            self.orders: list = []

        def book_tickers(self):
            self.n += 1
            if self.n == 2:
                raise TransientReadError("blip", path="/fapi/v1/ticker/bookTicker", http_status=503, code=None, attempts=4)
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, symbol, oid):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    outcomes = execute_intents(_Client(), [_exec_intent("AAAUSDT")], {"AAAUSDT": _exec_filters("AAAUSDT")},
                               policy, audit, clock, sleep_fn)
    assert outcomes[0].status == "FILLED"
    skips = [e for e in _exec_events(audit_path) if e["event"] == "tick_skipped_read_failure" and e.get("scope") == "books"]
    assert len(skips) == 1


def test_consecutive_book_failures_abort_fail_closed(tmp_path) -> None:
    """Consecutive book failures beyond the cap abort fail-closed."""
    import pytest
    from src.live.audit import AuditLog
    from src.live.errors import TransientReadError
    from src.live.executor import PassiveExecutionPolicy, execute_intents

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=50.0, window_deadline_s=200.0,
                                    max_consecutive_read_failures=2)
    audit_path = tmp_path / "abort.jsonl"
    audit = AuditLog(audit_path)
    clock_state = [0.0]

    class _Client:
        def __init__(self):
            self.cancels: list = []
            self.posted = 0

        def book_tickers(self):
            self.posted += 1
            if self.posted == 1:
                return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}
            raise TransientReadError("down", path="/fapi/v1/ticker/bookTicker", http_status=503, code=None, attempts=4)

        def new_order(self, params):
            return {"orderId": 1}

        def cancel_order(self, symbol, oid):
            self.cancels.append(oid)
            return {}

        def query_order(self, symbol, oid):
            return {"status": "NEW", "executedQty": "0"}

    client = _Client()
    with pytest.raises(TransientReadError) as exc_info:
        execute_intents(client, [_exec_intent("AAAUSDT")], {"AAAUSDT": _exec_filters("AAAUSDT")},
                        policy, audit, lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s))
    assert exc_info.value.partial_outcomes is not None
    assert client.cancels != []


def test_order_query_transient_isolates_intent(tmp_path) -> None:
    """An order-query transient failure isolates to its intent."""
    from src.live.audit import AuditLog
    from src.live.errors import TransientReadError
    from src.live.executor import PassiveExecutionPolicy, execute_intents

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=10.0, window_deadline_s=30.0)
    audit_path = tmp_path / "iso.jsonl"
    audit = AuditLog(audit_path)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(s):
        clock_state[0] += s

    class _Client:
        def __init__(self):
            self.orders: list = []
            self.post_times: list = []
            self.calls = 0

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            self.orders.append(params)
            self.post_times.append((params["symbol"], clock_state[0]))
            return {"orderId": len(self.orders)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, symbol, oid):
            if symbol == "AAAUSDT":
                self.calls += 1
                if self.calls == 1:
                    raise TransientReadError("blip", path="/fapi/v1/order", http_status=503, code=None, attempts=4)
            if symbol == "AAAUSDT":
                return {"status": "NEW", "executedQty": "0"}
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()
    outcomes = execute_intents(client, [_exec_intent("AAAUSDT"), _exec_intent("BBBUSDT")],
                               {"AAAUSDT": _exec_filters("AAAUSDT"), "BBBUSDT": _exec_filters("BBBUSDT")},
                               policy, audit, clock, sleep_fn)
    by = {o.symbol: o for o in outcomes}
    assert by["BBBUSDT"].status == "FILLED"
    passive_posts = [(sym, t) for sym, t in client.post_times if t < policy.passive_deadline_s]
    # 조회 실패 틱은 AAA 의 활성 주문을 유지할 뿐 재게시하지 않고, BBB 는 같은 틱에 정상 게시된다.
    assert [t for sym, t in passive_posts if sym == "AAAUSDT"] == [0.0]
    assert [t for sym, t in passive_posts if sym == "BBBUSDT"] == [0.0]
    skips = [e for e in _exec_events(audit_path) if e["event"] == "tick_skipped_read_failure" and e.get("scope") == "order"]
    assert skips


def test_repro_b_single_503_no_abort(tmp_path, monkeypatch) -> None:
    """One 503 on an order-status GET skips a tick: no abort, both orders keep resting until the window ends."""
    from pydantic import SecretStr
    from src.live.audit import AuditLog
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.rest import BinanceFuturesRestClient, HttpResponse
    from src.live.settings import ExecutionMode

    monkeypatch.setattr("src.live.rest.time.sleep", lambda s: None)
    clock_state = [0.0]
    calls: list[tuple[str, str, float]] = []
    served_503: list[tuple[str, str]] = []

    def _responder(method, url):
        from urllib.parse import urlparse
        path = urlparse(url).path
        calls.append((method, path, clock_state[0]))
        if path == "/fapi/v1/ticker/bookTicker":
            return HttpResponse(status_code=200, headers={}, body=b'[{"symbol":"AAAUSDT","bidPrice":"100.00","askPrice":"100.20"},{"symbol":"BBBUSDT","bidPrice":"100.00","askPrice":"100.20"}]')
        if method == "GET" and path == "/fapi/v1/order" and not served_503:
            # 모든 재시도를 소진시키도록 한 틱 동안 5xx 를 유지한다.
            if sum(1 for m, p, t in calls if m == "GET" and p == "/fapi/v1/order" and t == clock_state[0]) >= 4:
                served_503.append((method, path))
            return HttpResponse(status_code=503, headers={}, body=b"")
        return HttpResponse(status_code=200, headers={}, body=b'{"status":"NEW","executedQty":"0"}')

    class _Transport:
        def call(self, method, url, headers):
            return _responder(method, url)

    audit = AuditLog(tmp_path / "repro.jsonl")
    client = BinanceFuturesRestClient("https://fapi.binance.com", SecretStr("k"), SecretStr("s"),
                                      ExecutionMode.LIVE_TESTNET, audit, session=_Transport())
    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=5.0, window_deadline_s=10.0)
    outcomes = execute_intents(client, [_exec_intent("AAAUSDT"), _exec_intent("BBBUSDT")],
                               {"AAAUSDT": _exec_filters("AAAUSDT"), "BBBUSDT": _exec_filters("BBBUSDT")},
                               policy, audit, lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s))

    assert served_503 == [("GET", "/fapi/v1/order")]
    assert sorted(o.symbol for o in outcomes) == ["AAAUSDT", "BBBUSDT"]
    first_503_at = next(t for m, p, t in calls if m == "GET" and p == "/fapi/v1/order")
    cancels = [t for m, p, t in calls if m == "DELETE"]
    # 중단(abort) 정리가 아니라 창 종료 시점에만 취소가 일어난다.
    assert cancels
    assert min(cancels) > first_503_at
    assert all(t >= policy.passive_deadline_s for t in cancels)
    assert [p for m, p, t in calls if m == "POST" and t <= first_503_at].count("/fapi/v1/order") == 2


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

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.post_times.append(clock_state[0])
            self.posted.append(params)
            if len(self.posted) == 1:
                raise OrderStatusUnknown("x", path="/fapi/v1/order", http_status=0, code=None)
            return {"orderId": len(self.posted)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
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

    rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"))
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


def test_throttle_recovers_below_recovery_floor(tmp_path) -> None:
    """Throttle doubles above budget then halves back once below the recovery floor."""
    from src.live.executor import PassiveExecutionPolicy, _throttled_interval
    from src.live.rest import RateLimits

    policy = PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0)
    limits = RateLimits(request_weight_1m=2400, orders_1m=1200, orders_10s=300)

    class _C:
        def __init__(self, w):
            self.rate_state = type("S", (), {"used_weight_1m": w, "order_count_10s": 0, "order_count_1m": 0})()

    interval = 3.0
    high = _C(1500)
    for _ in range(3):
        interval = _throttled_interval(high, interval, policy, limits)
    assert interval > 3.0
    low = _C(500)
    for _ in range(10):
        interval = _throttled_interval(low, interval, policy, limits)
    assert interval == policy.poll_interval_s
    assert interval >= policy.poll_interval_s


def test_throttle_holds_inside_hysteresis_band() -> None:
    """Inside the hysteresis band the interval is unchanged."""
    from src.live.executor import PassiveExecutionPolicy, _throttled_interval
    from src.live.rest import RateLimits

    policy = PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0)
    limits = RateLimits(request_weight_1m=2400, orders_1m=1200, orders_10s=300)

    class _C:
        rate_state = type("S", (), {"used_weight_1m": 1000, "order_count_10s": 0, "order_count_1m": 0})()

    assert _throttled_interval(_C(), 6.0, policy, limits) == 6.0


def test_policy_validates_new_fields() -> None:
    """Policy validates the new read/throttle fields."""
    import pytest
    from src.live.executor import PassiveExecutionPolicy

    with pytest.raises(ValueError, match="max_consecutive_read_failures"):
        PassiveExecutionPolicy(max_consecutive_read_failures=0)
    with pytest.raises(ValueError, match="rate_weight_recover_fraction"):
        PassiveExecutionPolicy(rate_weight_recover_fraction=0.5)
    with pytest.raises(ValueError, match="rate_weight_recover_fraction"):
        PassiveExecutionPolicy(rate_weight_recover_fraction=0.0)


def test_throttled_interval_without_rate_state_returns_floor() -> None:
    """No rate state (or all-None counters) cannot retain a stale enlarged interval."""
    from src.live.executor import PassiveExecutionPolicy, _throttled_interval
    from src.live.rest import RateLimits

    policy = PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=5.0, window_deadline_s=600.0)
    limits = RateLimits(request_weight_1m=2400, orders_1m=1200, orders_10s=300)

    class _NoState:
        pass

    assert _throttled_interval(_NoState(), 9.0, policy, limits) == 3.0
    assert _throttled_interval(_NoState(), 9.0, policy, None) == 3.0

    class _Empty:
        rate_state = type("S", (), {"used_weight_1m": None, "order_count_10s": None, "order_count_1m": None})()

    assert _throttled_interval(_Empty(), 9.0, policy, limits) == 3.0


def test_throttled_interval_order_counts_drive_hysteresis() -> None:
    """Order-count counters double above budget and recover below the floor."""
    from src.live.executor import PassiveExecutionPolicy, _throttled_interval
    from src.live.rest import RateLimits

    policy = PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=5.0, window_deadline_s=600.0)
    limits = RateLimits(request_weight_1m=2400, orders_1m=1200, orders_10s=300)

    class _C:
        def __init__(self, w, c10, c1m):
            self.rate_state = type("S", (), {"used_weight_1m": w, "order_count_10s": c10, "order_count_1m": c1m})()

    assert _throttled_interval(_C(0, 250, 0), 3.0, policy, limits) == 6.0
    assert _throttled_interval(_C(0, 0, 900), 3.0, policy, limits) == 6.0
    # Above the recover floor but below budget: hold.
    assert _throttled_interval(_C(0, 120, 0), 6.0, policy, limits) == 6.0
    assert _throttled_interval(_C(900, 0, 0), 6.0, policy, limits) == 6.0
    assert _throttled_interval(_C(0, 0, 500), 6.0, policy, limits) == 6.0


def test_unknown_lookup_transient_is_undecidable(tmp_path) -> None:
    """A TransientReadError during unknown lookup neither adopts nor counts a miss."""
    from src.live.audit import AuditLog
    from src.live.errors import TransientReadError
    from src.live.executor import _resolve_unknown_submission, _IntentRuntime

    rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"))
    rt.unresolved_id = "oid-1"
    rt.unresolved_at = 100.0

    class _C:
        def query_order(self, s, oid):
            raise TransientReadError("blip", path="/fapi/v1/order", http_status=503, code=None, attempts=4)

    assert _resolve_unknown_submission(_C(), rt, 101.0, AuditLog(tmp_path / "u.jsonl")) is False
    assert rt.unresolved_id == "oid-1"
    assert rt.unknown_misses == 0


def test_per_rt_consecutive_read_failures_abort(tmp_path) -> None:
    """Per-rt transient reads beyond the cap re-raise fail-closed."""
    import pytest
    from src.live.audit import AuditLog
    from src.live.errors import TransientReadError
    from src.live.executor import PassiveExecutionPolicy, execute_intents

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=5.0, window_deadline_s=50.0,
                                    max_consecutive_read_failures=1)
    audit = AuditLog(tmp_path / "r.jsonl")
    clock_state = [0.0]

    class _C:
        def __init__(self):
            self.orders = 0

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.orders += 1
            return {"orderId": self.orders}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            raise TransientReadError("down", path="/fapi/v1/order", http_status=503, code=None, attempts=4)

    with pytest.raises(TransientReadError):
        execute_intents(_C(), [_exec_intent("AAAUSDT")], {"AAAUSDT": _exec_filters("AAAUSDT")}, policy, audit,
                        lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s))


def test_poll_active_avg_price_branches(tmp_path) -> None:
    """Active poll tolerates zero and malformed avgPrice without recording fills."""
    from decimal import Decimal
    from src.live.audit import AuditLog
    from src.live.executor import PassiveExecutionPolicy, _IntentRuntime, _poll_active

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=50.0, window_deadline_s=600.0)
    audit = AuditLog(tmp_path / "avg.jsonl")

    class _ZeroAvg:
        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0", "avgPrice": "0.000"}

        def cancel_order(self, *a, **k):
            return {}

    rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"))
    rt.active_id = "oid-1"
    rt.active_price = Decimal("100")
    _poll_active(_ZeroAvg(), rt, (Decimal("100"), Decimal("100.20")), 1.0, policy, audit)
    assert rt.filled_total == Decimal("0")

    class _BadAvg:
        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0", "avgPrice": "nan-x"}

        def cancel_order(self, *a, **k):
            return {}

    rt2 = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"))
    rt2.active_id = "oid-2"
    rt2.active_price = Decimal("100")
    _poll_active(_BadAvg(), rt2, (Decimal("100"), Decimal("100.20")), 1.0, policy, audit)
    assert rt2.filled_total == Decimal("0")


def test_finalize_horizon_branches(tmp_path) -> None:
    """Finalize covers adopted-now, adopted-after-sleep and already-elapsed paths."""
    from src.live.audit import AuditLog
    from src.live.errors import VenueError
    from src.live.executor import PassiveExecutionPolicy, _finalize, _IntentRuntime

    policy = PassiveExecutionPolicy(poll_interval_s=1.0, passive_deadline_s=5.0, window_deadline_s=600.0)
    assert policy.poll_interval_s == 1.0

    def _rt(oid, at):
        rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"))
        rt.unresolved_id = oid
        rt.unresolved_at = at
        return rt

    # First lookup success: adopted immediately, no sleep.
    class _Found:
        unknown_outcome_horizon_s = 35.0

        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0"}

        def cancel_order(self, *a, **k):
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

        def query_order(self, s, oid):
            self.n += 1
            if self.n == 1:
                raise VenueError("m", code=-2013, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"status": "NEW", "executedQty": "0"}

        def cancel_order(self, *a, **k):
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


def _iso_filters(symbol, min_qty="0.001", min_notional="1"):
    from decimal import Decimal
    from src.live.filters import SymbolFilters

    return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                         min_qty=Decimal(min_qty), min_notional=Decimal(min_notional), max_qty=Decimal("100000"),
                         quantity_precision=3, price_precision=2)


def _iso_intent(symbol, qty="1", reduce_only=False):
    from decimal import Decimal
    from src.live.planner import OrderIntent

    return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal(qty), reduce_only=reduce_only,
                       target_qty=Decimal(qty), current_qty=Decimal("0"), client_order_prefix="20260914",
                       leg_index=0, decision_price=Decimal("100.10"))


def _iso_policy(**overrides):
    from src.live.executor import PassiveExecutionPolicy

    base = {"poll_interval_s": 1.0, "passive_deadline_s": 50.0, "window_deadline_s": 200.0}
    base.update(overrides)
    return PassiveExecutionPolicy(**base)


def _iso_run(client, intents, filters, policy, tmp_path, name="iso.jsonl"):
    from src.live.audit import AuditLog

    audit_path = tmp_path / name
    audit = AuditLog(audit_path)
    clock_state = [0.0]

    def clock():
        return clock_state[0]

    def sleep_fn(s):
        clock_state[0] += s

    from src.live.executor import execute_intents

    outcomes = execute_intents(client, intents, filters, policy, audit, clock, sleep_fn)
    import json

    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    return outcomes, events, client


def test_sub_minimum_residual_after_partial_fill(tmp_path) -> None:
    """Spec 02 Repro A: a 0.04 remainder below minNotional is never posted."""
    from decimal import Decimal

    class _Client:
        def __init__(self):
            self.orders: list = []

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, symbol, oid):
            if symbol == "AAAUSDT":
                return {"status": "NEW", "executedQty": "0.96", "avgPrice": "100.00"}
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()
    policy = _iso_policy(passive_deadline_s=20.0, window_deadline_s=600.0)
    filters = {"AAAUSDT": _iso_filters("AAAUSDT", min_notional="5"), "BBBUSDT": _iso_filters("BBBUSDT")}
    outcomes, events, _ = _iso_run(client, [_iso_intent("AAAUSDT"), _iso_intent("BBBUSDT")], filters, policy, tmp_path)
    by = {o.symbol: o for o in outcomes}
    assert by["AAAUSDT"].status == "RESIDUAL_SUB_MINIMUM"
    assert by["AAAUSDT"].filled_qty == Decimal("0.96")
    assert by["BBBUSDT"].status == "FILLED"
    assert all(Decimal(o["quantity"]) != Decimal("0.04") for o in client.orders)
    assert any(e["event"] == "intent_sub_minimum" for e in events)


def test_sub_minimum_reduce_only_residual_is_posted(tmp_path) -> None:
    """Spec 02: a reduce-only remainder below minNotional is posted reduce-only."""
    class _Client:
        def __init__(self):
            self.orders: list = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0"}

    client = _Client()
    policy = _iso_policy(window_deadline_s=6.0, passive_deadline_s=5.0)
    outcomes, _, _ = _iso_run(client, [_iso_intent("AAAUSDT", qty="0.04", reduce_only=True)],
                               {"AAAUSDT": _iso_filters("AAAUSDT", min_notional="5")}, policy, tmp_path)
    assert client.orders
    assert all(o.get("reduceOnly") == "true" and o["quantity"] == "0.04" for o in client.orders)


def test_remainder_below_min_qty_never_posted(tmp_path) -> None:
    """Spec 02: a remainder below minQty ends quietly with zero posts."""
    class _Client:
        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            raise AssertionError("must never post")

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0"}

    policy = _iso_policy(window_deadline_s=6.0, passive_deadline_s=5.0)
    outcomes, events, _ = _iso_run(_Client(), [_iso_intent("AAAUSDT", qty="0.0005")],
                                   {"AAAUSDT": _iso_filters("AAAUSDT")}, policy, tmp_path)
    assert outcomes[0].status == "RESIDUAL_SUB_MINIMUM"
    assert any(e["event"] == "intent_sub_minimum" for e in events)


def test_undersized_head_slice_posts_whole_remainder(tmp_path) -> None:
    """Spec 02: slicing must never create an unpostable head order."""
    from decimal import Decimal

    class _Client:
        def __init__(self):
            self.orders: list = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0"}

    client = _Client()
    policy = _iso_policy(window_deadline_s=6.0, passive_deadline_s=5.0)
    _iso_run(client, [_iso_intent("AAAUSDT", qty="10")],
             {"AAAUSDT": _iso_filters("AAAUSDT", min_notional="600")}, policy, tmp_path)
    assert client.orders
    assert all(Decimal(o["quantity"]) == Decimal("10") for o in client.orders)


def test_symbol_scoped_rejection_ends_only_that_intent(tmp_path) -> None:
    """Spec 02: BBB -4140 ends REJECTED while AAA fills normally."""
    from src.live.errors import VenueError

    class _Client:
        def __init__(self):
            self.cancels: list = []

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            if params["symbol"] == "BBBUSDT":
                raise VenueError("invalid status", code=-4140, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"orderId": 1}

        def cancel_order(self, symbol, oid):
            self.cancels.append(oid)
            return {}

        def query_order(self, s, oid):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()
    filters = {"AAAUSDT": _iso_filters("AAAUSDT"), "BBBUSDT": _iso_filters("BBBUSDT")}
    outcomes, events, _ = _iso_run(client, [_iso_intent("AAAUSDT"), _iso_intent("BBBUSDT")], filters, _iso_policy(), tmp_path)
    by = {o.symbol: o for o in outcomes}
    assert by["BBBUSDT"].status == "REJECTED"
    assert by["BBBUSDT"].reject_code == -4140
    assert by["AAAUSDT"].status == "FILLED"
    assert client.cancels == []
    rejected = [e for e in events if e["event"] == "intent_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["code"] == -4140
    assert any(e["event"] == "intent_outcome" and e.get("reject_code") == -4140 for e in events)


def test_margin_rejection_waits_then_retries(tmp_path) -> None:
    """Spec 02: one -2019 waits margin_retry_s before reposting, then fills."""
    from src.live.errors import VenueError

    clock_state = [0.0]

    class _Client:
        def __init__(self):
            self.post_times: list = []

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.post_times.append(clock_state[0])
            if len(self.post_times) == 1:
                raise VenueError("margin", code=-2019, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"orderId": 2}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    from src.live.audit import AuditLog
    from src.live.executor import execute_intents
    import json

    audit_path = tmp_path / "margin.jsonl"
    audit = AuditLog(audit_path)
    client = _Client()
    policy = _iso_policy(passive_deadline_s=500.0, window_deadline_s=600.0)
    outcomes = execute_intents(client, [_iso_intent("AAAUSDT")], {"AAAUSDT": _iso_filters("AAAUSDT")},
                               policy, audit, lambda: clock_state[0],
                               lambda s: clock_state.__setitem__(0, clock_state[0] + s))
    assert outcomes[0].status == "FILLED"
    assert len(client.post_times) == 2
    assert client.post_times[1] - client.post_times[0] >= 60.0
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert sum(1 for e in events if e["event"] == "intent_margin_wait") == 1


def test_margin_rejections_bounded(tmp_path) -> None:
    """Spec 02: perpetual -2019 ends REJECTED after max_margin_rejects+1 attempts."""
    from src.live.errors import VenueError

    class _Client:
        def __init__(self):
            self.posts = 0

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            if params["symbol"] == "AAAUSDT":
                self.posts += 1
                raise VenueError("margin", code=-2019, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"orderId": 99}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()
    policy = _iso_policy(passive_deadline_s=500.0, window_deadline_s=600.0, max_margin_rejects=2)
    filters = {"AAAUSDT": _iso_filters("AAAUSDT"), "BBBUSDT": _iso_filters("BBBUSDT")}
    outcomes, _, _ = _iso_run(client, [_iso_intent("AAAUSDT"), _iso_intent("BBBUSDT")], filters, policy, tmp_path)
    by = {o.symbol: o for o in outcomes}
    assert by["AAAUSDT"].status == "REJECTED"
    assert by["AAAUSDT"].reject_code == -2019
    assert client.posts == 3
    assert by["BBBUSDT"].status == "FILLED"


def test_margin_rejection_on_reduce_only_rejects_immediately(tmp_path) -> None:
    """Spec 02: reduce-only needs no margin, so its -2019 is anomalous REJECTED."""
    from src.live.errors import VenueError

    class _Client:
        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            raise VenueError("margin", code=-2019, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0"}

    policy = _iso_policy(window_deadline_s=6.0, passive_deadline_s=5.0)
    outcomes, _, _ = _iso_run(_Client(), [_iso_intent("AAAUSDT", reduce_only=True)],
                               {"AAAUSDT": _iso_filters("AAAUSDT")}, policy, tmp_path)
    assert outcomes[0].status == "REJECTED"
    assert outcomes[0].reject_code == -2019


def test_risk_increase_freeze_stops_only_increases(tmp_path) -> None:
    """Spec 02: -4400 freezes increases while reduce-only fills."""
    from src.live.errors import VenueError

    class _Client:
        def __init__(self):
            self.posted: list = []

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT")}

        def new_order(self, params):
            if params["symbol"] == "AAAUSDT":
                raise VenueError("frozen", code=-4400, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            self.posted.append(params)
            return {"orderId": len(self.posted)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    client = _Client()
    filters = {s: _iso_filters(s) for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT")}
    intents = [_iso_intent("AAAUSDT"), _iso_intent("BBBUSDT"), _iso_intent("CCCUSDT", reduce_only=True)]
    outcomes, events, _ = _iso_run(client, intents, filters, _iso_policy(), tmp_path)
    by = {o.symbol: o for o in outcomes}
    assert by["AAAUSDT"].status == "REJECTED"
    assert by["AAAUSDT"].reject_code == -4400
    assert by["BBBUSDT"].status == "REJECTED"
    assert by["CCCUSDT"].status == "FILLED"
    assert all(p["symbol"] != "BBBUSDT" for p in client.posted)
    assert any(e["event"] == "risk_increase_frozen" and e["code"] == -4400 for e in events)


def test_reductions_posted_before_increases(tmp_path) -> None:
    """Spec 02: within a tick reduce-only intents post first."""
    class _Client:
        def __init__(self):
            self.posted: list = []

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            self.posted.append(params)
            return {"orderId": len(self.posted)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0"}

    client = _Client()
    policy = _iso_policy(window_deadline_s=4.0, passive_deadline_s=3.0)
    filters = {"AAAUSDT": _iso_filters("AAAUSDT"), "BBBUSDT": _iso_filters("BBBUSDT")}
    _iso_run(client, [_iso_intent("AAAUSDT"), _iso_intent("BBBUSDT", reduce_only=True)], filters, policy, tmp_path)
    assert client.posted
    assert client.posted[0]["symbol"] == "BBBUSDT"


def test_unregistered_rejection_still_aborts(tmp_path) -> None:
    """Spec 02: an unregistered code fails closed with cleanup and partials."""
    from src.live.errors import VenueError
    from src.live.order_journal import OrderJournal

    class _Client:
        def __init__(self):
            self.posted: list = []
            self.cancels: list = []

        def book_tickers(self):
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            self.posted.append(params)
            if params["symbol"] == "BBBUSDT":
                raise VenueError("unknown", code=-9999, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
            return {"orderId": 1}

        def cancel_order(self, symbol, oid):
            self.cancels.append(oid)
            return {}

        def query_order(self, s, oid):
            return {"status": "CANCELED", "executedQty": "0.4", "avgPrice": "100.00"}

    import pytest

    from src.live.audit import AuditLog
    from src.live.executor import execute_intents

    audit = AuditLog(tmp_path / "a.jsonl")
    journal = OrderJournal(tmp_path / "j.jsonl")
    attempt = _test_attempt(journal)
    clock_state = [0.0]
    client = _Client()
    filters = {"AAAUSDT": _iso_filters("AAAUSDT"), "BBBUSDT": _iso_filters("BBBUSDT")}
    with pytest.raises(VenueError):
        execute_intents(client, [_iso_intent("AAAUSDT"), _iso_intent("BBBUSDT")], filters, _iso_policy(),
                        audit, lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s),
                        journal=journal, attempt=attempt)
    assert client.cancels == [client.posted[0]["newClientOrderId"]]


def test_outcome_statuses_closed_set() -> None:
    """Spec 02: OUTCOME_STATUSES pins the six terminal states."""
    from src.live.executor import OUTCOME_STATUSES

    assert frozenset({"FILLED", "RESIDUAL", "RESIDUAL_SUB_MINIMUM", "REJECTED", "SHADOW", "OBSOLETE"}) == OUTCOME_STATUSES


def test_policy_validates_margin_fields() -> None:
    """Spec 02: margin wait knobs fail closed on nonsense values."""
    import pytest

    from src.live.executor import PassiveExecutionPolicy

    with pytest.raises(ValueError, match="margin_retry_s"):
        PassiveExecutionPolicy(margin_retry_s=0.0)
    with pytest.raises(ValueError, match="max_margin_rejects"):
        PassiveExecutionPolicy(max_margin_rejects=0)


def test_paper_invalid_quote_never_fills(tmp_path) -> None:
    """Spec 02: an ask of 0 cannot trade through a resting PAPER BUY."""
    from src.live.rest import PaperResponse

    class _Client:
        def __init__(self):
            self.orders: list = []
            self.cancels: list = []
            self.tick = 0

        def book_tickers(self):
            self.tick += 1
            if self.tick >= 2:
                return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "0"}}
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.orders.append(params)
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

        def cancel_order(self, symbol, oid):
            self.cancels.append(oid)
            return {}

        def query_order(self, s, oid):
            raise AssertionError("PAPER must never poll a venue order")

    client = _Client()
    policy = _iso_policy(window_deadline_s=3.0, passive_deadline_s=1.5)
    outcomes, events, _ = _iso_run(client, [_iso_intent("AAAUSDT")], {"AAAUSDT": _iso_filters("AAAUSDT")},
                                   policy, tmp_path, name="paper.jsonl")
    assert len(client.orders) == 1
    assert client.cancels == []
    assert outcomes[0].fills == ()
    assert outcomes[0].status == "RESIDUAL"
    invalid = [e for e in events if e["event"] == "quote_invalid"]
    assert len(invalid) == 1


def test_immediate_taker_skips_invalid_quotes(tmp_path) -> None:
    """Spec 02: immediate-taker simulation leaves invalid-quote symbols RESIDUAL."""
    from decimal import Decimal

    from src.live.audit import AuditLog
    from src.live.executor import execute_intents

    class _Client:
        def book_tickers(self):
            return {
                "AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "0", "askPrice": "100.20"},
                "BBBUSDT": {"symbol": "BBBUSDT", "bidPrice": "100.00", "askPrice": "100.20"},
            }

    audit = AuditLog(tmp_path / "imm.jsonl")
    filters = {"AAAUSDT": _iso_filters("AAAUSDT"), "BBBUSDT": _iso_filters("BBBUSDT")}
    outcomes = execute_intents(_Client(), [_iso_intent("AAAUSDT"), _iso_intent("BBBUSDT")], filters,
                               _iso_policy(), audit, lambda: 0.0, lambda s: None,
                               paper_fill_model="immediate_taker")
    by = {o.symbol: o for o in outcomes}
    assert by["AAAUSDT"].status == "RESIDUAL"
    assert by["AAAUSDT"].filled_qty == Decimal("0")
    assert by["BBBUSDT"].status == "FILLED"


def test_fetch_books_reports_invalid_quotes() -> None:
    """Spec 02: _fetch_books omits invalid quotes with audit, and tolerates gaps."""
    from src.live.executor import _fetch_books

    class _BatchMissing:
        def book_tickers(self):
            return {"OTHERUSDT": {"bidPrice": "1", "askPrice": "2"}}

    assert _fetch_books(_BatchMissing(), ["AAAUSDT"]) == {}

    class _LegacyInvalid:
        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "0"}

    assert _fetch_books(_LegacyInvalid(), ["AAAUSDT"]) == {}


def test_fetch_books_legacy_invalid_audited(tmp_path) -> None:
    """Spec 02: legacy per-symbol fallback validates through the same gate."""
    import json

    from src.live.audit import AuditLog
    from src.live.executor import _fetch_books

    class _LegacyInvalid:
        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "0"}

    audit_path = tmp_path / "books.jsonl"
    books = _fetch_books(_LegacyInvalid(), ["AAAUSDT"], audit=AuditLog(audit_path))
    assert books == {}
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [(e["event"], e["symbol"]) for e in events] == [("quote_invalid", "AAAUSDT")]


def test_order_budget_exceeded_posts_nothing(tmp_path) -> None:
    """Spec 02: an exhausted order budget skips posting without ending the intent."""
    from src.live.audit import AuditLog
    from src.live.executor import execute_intents
    from src.live.rest import RateLimits

    class _Client:
        def __init__(self):
            self.orders: list = []
            self.rate_state = type("S", (), {"used_weight_1m": 0, "order_count_10s": 290, "order_count_1m": 0})()

        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, *a, **k):
            return {}

        def query_order(self, s, oid):
            return {"status": "NEW", "executedQty": "0"}

    client = _Client()
    limits = RateLimits(request_weight_1m=2400, orders_1m=1200, orders_10s=300)
    audit = AuditLog(tmp_path / "budget.jsonl")
    clock_state = [0.0]
    outcomes = execute_intents(client, [_iso_intent("AAAUSDT")], {"AAAUSDT": _iso_filters("AAAUSDT")},
                               _iso_policy(window_deadline_s=4.0, passive_deadline_s=3.0), audit,
                               lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s),
                               rate_limits=limits)
    assert client.orders == []
    assert outcomes[0].status == "RESIDUAL"


# ---- Spec 03: execution interruption & journal-backed fill durability ----

def _shutdown_policy(**overrides):
    base = {"poll_interval_s": 1.0, "passive_deadline_s": 50.0, "window_deadline_s": 200.0,
            "taker_cap_bps": 15.0, "max_slices": 1}
    base.update(overrides)
    from src.live.executor import PassiveExecutionPolicy

    return PassiveExecutionPolicy(**base)


def _shutdown_intent(symbol):
    from decimal import Decimal

    from src.live.planner import OrderIntent

    return OrderIntent(symbol=symbol, side="BUY", quantity=Decimal("1"), reduce_only=False,
                       target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260914",
                       leg_index=0, decision_price=Decimal("100.10"))


def _shutdown_filters(symbol):
    from decimal import Decimal

    from src.live.filters import SymbolFilters

    return SymbolFilters(symbol=symbol, tick_size=Decimal("0.01"), step_size=Decimal("0.001"),
                         min_qty=Decimal("0.001"), min_notional=Decimal("1"), max_qty=Decimal("100000"),
                         quantity_precision=3, price_precision=2)


class _ShutdownLiveClient:
    """Two resting GTX orders; AAA partially fills; shutdown flag trips on the Nth book fetch."""

    def __init__(self, flag, trip_at=2):
        self.orders: list = []
        self.cancels: list = []
        self.flag = flag
        self.trip_at = trip_at
        self.books = 0

    def book_tickers(self):
        self.books += 1
        if self.books >= self.trip_at:
            self.flag.requested = True
        return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"}
                for s in ("AAAUSDT", "BBBUSDT")}

    def new_order(self, params):
        self.orders.append(params)
        return {"orderId": len(self.orders)}

    def cancel_order(self, symbol, orig_client_order_id):
        self.cancels.append(orig_client_order_id)
        return {}

    def query_order(self, symbol, orig_client_order_id):
        for order in self.orders:
            if order["newClientOrderId"] == orig_client_order_id:
                if orig_client_order_id in self.cancels:
                    return {"status": "CANCELED", "executedQty": "0.4" if symbol == "AAAUSDT" else "0",
                            "avgPrice": "100.00"}
                return {"status": "NEW", "executedQty": "0.4" if symbol == "AAAUSDT" else "0",
                        "avgPrice": "100.00"}
        return {"status": "NEW", "executedQty": "0"}


def test_shutdown_during_window_settles_and_raises_interrupted(tmp_path) -> None:
    """Shutdown mid-window cancels, settles, journals, then raises ExecutionInterrupted."""
    import json
    from decimal import Decimal
    from types import SimpleNamespace

    import pytest

    from src.live.audit import AuditLog
    from src.live.executor import ExecutionInterrupted, execute_intents
    from src.live.order_journal import OrderJournal

    audit_path = tmp_path / "shutdown.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    flag = SimpleNamespace(requested=False)
    client = _ShutdownLiveClient(flag)
    clock_state = [0.0]
    intents = [_shutdown_intent("AAAUSDT"), _shutdown_intent("BBBUSDT")]
    filters = {"AAAUSDT": _shutdown_filters("AAAUSDT"), "BBBUSDT": _shutdown_filters("BBBUSDT")}

    with pytest.raises(ExecutionInterrupted) as exc_info:
        execute_intents(client, intents, filters, _shutdown_policy(), audit,
                        lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s),
                        shutdown=flag, journal=journal, attempt=attempt)

    posted_ids = [o["newClientOrderId"] for o in client.orders]
    assert sorted(client.cancels) == sorted(posted_ids)
    partial = {o.symbol: o for o in exc_info.value.partial_outcomes}
    assert partial["AAAUSDT"].filled_qty == Decimal("0.4")
    assert partial["BBBUSDT"].filled_qty == Decimal("0")
    fills = journal.fills_after(-1)
    assert len(fills) == 1
    assert fills[0].kind == "execution"
    assert fills[0].quantity == Decimal("0.4")
    assert fills[0].cumulative_executed_qty == Decimal("0.4")
    assert fills[0].simulated is False
    import pandas as pd

    assert journal.unresolved_submits(since=pd.Timestamp("2026-09-14T00:00:00Z") - pd.Timedelta(hours=72)) == ()
    events = [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert "execution_interrupted" in events


def test_shutdown_before_first_tick_posts_nothing(tmp_path) -> None:
    """A pre-set shutdown flag raises with empty fills and no order posted."""
    from types import SimpleNamespace

    import pytest

    from src.live.audit import AuditLog
    from src.live.executor import ExecutionInterrupted, execute_intents
    from src.live.order_journal import OrderJournal

    audit = AuditLog(tmp_path / "pre.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    flag = SimpleNamespace(requested=True)
    client = _ShutdownLiveClient(flag, trip_at=10**9)
    clock_state = [0.0]
    intents = [_shutdown_intent("AAAUSDT"), _shutdown_intent("BBBUSDT")]
    filters = {"AAAUSDT": _shutdown_filters("AAAUSDT"), "BBBUSDT": _shutdown_filters("BBBUSDT")}

    with pytest.raises(ExecutionInterrupted) as exc_info:
        execute_intents(client, intents, filters, _shutdown_policy(), audit,
                        lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s),
                        shutdown=flag, journal=journal, attempt=attempt)

    assert client.orders == []
    assert all(o.fills == () and o.filled_qty == 0 for o in exc_info.value.partial_outcomes)
    assert journal.fills_after(-1) == ()


def test_shutdown_cleanup_budget_bounds_venue_calls(tmp_path) -> None:
    """With a 20 s budget and 15 s per cancel, at most two of three orders are settled."""
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    from src.live.audit import AuditLog
    from src.live.executor import ExecutionInterrupted, execute_intents
    from src.live.order_journal import OrderJournal

    audit = AuditLog(tmp_path / "budget.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    flag = SimpleNamespace(requested=False)
    clock_state = [0.0]

    class _SlowCancelClient(_ShutdownLiveClient):
        def cancel_order(self, symbol, orig_client_order_id):
            clock_state[0] += 15.0
            return super().cancel_order(symbol, orig_client_order_id)

    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")

    class _ThreeClient(_SlowCancelClient):
        def book_tickers(self):
            self.books += 1
            if self.books >= self.trip_at:
                self.flag.requested = True
            return {s: {"symbol": s, "bidPrice": "100.00", "askPrice": "100.20"} for s in symbols}

        def query_order(self, symbol, orig_client_order_id):
            return {"status": "CANCELED" if orig_client_order_id in self.cancels else "NEW",
                    "executedQty": "0", "avgPrice": "100.00"}

    client = _ThreeClient(flag)
    intents = [_shutdown_intent(s) for s in symbols]
    filters = {s: _shutdown_filters(s) for s in symbols}

    with pytest.raises(ExecutionInterrupted):
        execute_intents(client, intents, filters, _shutdown_policy(), audit,
                        lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s),
                        shutdown=flag, journal=journal, attempt=attempt,
                        shutdown_cleanup_budget_s=20.0)

    assert len(client.cancels) == 2
    posted_ids = [o["newClientOrderId"] for o in client.orders]
    leftover = [i for i in posted_ids if i not in client.cancels]
    assert len(leftover) == 1
    pending = journal.unresolved_submits(
        since=pd.Timestamp("2026-09-14T00:00:00Z") - pd.Timedelta(hours=72))
    assert [s.client_order_id for s in pending] == leftover


def test_every_fill_is_journaled_before_visible(tmp_path) -> None:
    """PAPER trade-through and live partial fills each have exactly one journal fill; a failing journal aborts with no in-memory fill."""
    from decimal import Decimal

    import pytest

    from src.live.audit import AuditLog
    from src.live.executor import execute_intents
    from src.live.order_journal import OrderJournal
    from src.live.rest import PaperResponse

    # PAPER GTX trade-through.
    class _PaperThrough:
        def __init__(self, touches):
            self._touches = list(touches)
            self._cursor = 0

        def book_tickers(self):
            idx = min(self._cursor, len(self._touches) - 1)
            self._cursor += 1
            bid, ask = self._touches[idx]
            return {"AAAUSDT": {"bidPrice": bid, "askPrice": ask}}

        def new_order(self, params):
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

    from src.live.executor import PassiveExecutionPolicy

    anchored = PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0,
                                      taker_cap_bps=15.0, max_slices=1, passive_pricing="anchored")
    paper_client = _PaperThrough([("100.00", "100.05")] * 4 + [("99.40", "99.50")] * 50)
    paper_journal = OrderJournal(tmp_path / "paper_journal.jsonl")
    paper_attempt = _test_attempt(paper_journal)
    clock_state = [0.0]

    def _clock():
        value = clock_state[0]
        clock_state[0] += 3.0
        return value

    outcomes = execute_intents(paper_client, [_shutdown_intent("AAAUSDT")],
                               {"AAAUSDT": _shutdown_filters("AAAUSDT")}, anchored,
                               AuditLog(tmp_path / "paper.jsonl"), _clock, lambda s: None,
                               journal=paper_journal, attempt=paper_attempt)
    assert outcomes[0].status == "FILLED"
    paper_fills = paper_journal.fills_after(-1)
    assert len(paper_fills) == 1
    assert paper_fills[0].simulated is True
    assert paper_fills[0].cumulative_executed_qty is None
    assert paper_fills[0].quantity == outcomes[0].filled_qty == Decimal("1")
    assert paper_fills[0].price == outcomes[0].avg_fill_price
    assert paper_fills[0].attempt_seq == paper_attempt.attempt_seq

    # Live partial fill: one journal execution fill mirrors the outcome delta.
    live_journal = OrderJournal(tmp_path / "live_journal.jsonl")
    live_attempt = _test_attempt(live_journal)
    live_client = _ShutdownLiveClient(__import__("types").SimpleNamespace(requested=False), trip_at=10**9)
    first_id: dict = {}

    def _once_partial(symbol, oid):
        # A constant venue answer would re-fill on every repost; only the first order may fill.
        first_id.setdefault("id", oid)
        executed = "0.4" if oid == first_id["id"] else "0"
        return {"status": "NEW", "executedQty": executed, "avgPrice": "100.00"}

    live_client.query_order = _once_partial
    short = _shutdown_policy(passive_deadline_s=20.0, window_deadline_s=600.0)
    live_outcomes = execute_intents(live_client, [_shutdown_intent("AAAUSDT")],
                                    {"AAAUSDT": _shutdown_filters("AAAUSDT")}, short,
                                    AuditLog(tmp_path / "live.jsonl"), SteppingClock(3.0), lambda s: None,
                                    journal=live_journal, attempt=live_attempt)
    assert live_outcomes[0].filled_qty == Decimal("0.4")
    journaled = live_journal.fills_after(-1)
    assert [(f.quantity, f.price) for f in journaled] == [(Decimal("0.4"), live_outcomes[0].avg_fill_price)]

    # A journal that raises on write aborts the attempt with no in-memory fill.
    boom_journal = OrderJournal(tmp_path / "boom_journal.jsonl")
    boom_attempt = _test_attempt(boom_journal)
    boom_client = _ShutdownLiveClient(__import__("types").SimpleNamespace(requested=False), trip_at=10**9)
    boom_client.query_order = lambda s, oid: {"status": "NEW", "executedQty": "1", "avgPrice": "100.00"}

    def _boom(**kwargs):
        raise RuntimeError("journal down")

    boom_journal.record_fill = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="journal down"):
        execute_intents(boom_client, [_shutdown_intent("AAAUSDT")],
                        {"AAAUSDT": _shutdown_filters("AAAUSDT")}, short,
                        AuditLog(tmp_path / "boom.jsonl"), SteppingClock(3.0), lambda s: None,
                        journal=boom_journal, attempt=boom_attempt)


def test_orphan_settlement_journaled_not_applied(tmp_path) -> None:
    """Venue executedQty 1.5 over observed 1.0 yields one orphan_settlement fill plus terminal."""
    import pandas as pd
    from decimal import Decimal

    from src.live.audit import AuditLog
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal

    audit_path = tmp_path / "orphan2.jsonl"
    audit = AuditLog(audit_path)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    now = pd.Timestamp("2026-09-14T00:00:00Z")
    order_id = "mh20260914-ABCDEFGHIJ-0-0-3"
    journal.record_submit(order_id, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
                          side="BUY", quantity=Decimal("2"), reduce_only=False, leg_index=0)
    journal.record_fill(kind="execution", attempt_seq=attempt.attempt_seq, symbol="AAAUSDT", side="BUY",
                        quantity=Decimal("1"), price=Decimal("99"), fee_bps=2.0, liquidity="maker",
                        reason="maker_fill", filled_at=now, client_order_id=order_id, leg_index=0,
                        cumulative_executed_qty=Decimal("1"), simulated=False)

    class _Client:
        def open_orders(self):
            return [{"symbol": "AAAUSDT", "clientOrderId": order_id}]

        def cancel_order(self, symbol, oid):
            return {}

        def query_order(self, symbol, oid):
            return {"status": "CANCELED", "side": "SELL", "avgPrice": "101", "executedQty": "1.5"}

    fills = cancel_orphan_orders(_Client(), "20260914", audit, journal=journal, now=now, taker_fee_bps=4.5)

    assert len(fills) == 1
    assert fills[0].kind == "orphan_settlement"
    assert fills[0].quantity == Decimal("0.5")
    assert (fills[0].side, fills[0].price) == ("SELL", Decimal("101"))
    assert fills[0].cumulative_executed_qty == Decimal("1.5")
    assert fills[0].fee_bps == 4.5
    assert journal.unresolved_submits(since=now - pd.Timedelta(hours=72)) == ()


@pytest.mark.parametrize(
    "answer",
    [
        {"status": "CANCELED", "side": "BUY", "avgPrice": "0", "executedQty": "1.5"},
        {"status": "CANCELED", "side": "BUY", "avgPrice": "abc", "executedQty": "1.5"},
        {"status": "CANCELED", "avgPrice": "101", "executedQty": "1.5"},
        {"status": "CANCELED", "side": "BUY", "avgPrice": "101"},
        {"status": "CANCELED", "side": "BUY", "avgPrice": "101", "executedQty": "x"},
        {"status": "CANCELED", "side": "BUY", "avgPrice": "101", "executedQty": "NaN"},
    ],
    ids=["zero_price", "unparseable_price", "no_side", "no_executed", "bad_executed", "nan_executed"],
)
def test_orphan_settlement_without_price_fails_closed(tmp_path, answer) -> None:
    """A settled orphan with a missing venue side, executedQty or positive price raises and journals nothing."""
    import pandas as pd
    from decimal import Decimal

    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.audit import AuditLog
    from src.live.executor import cancel_orphan_orders
    from src.live.order_journal import OrderJournal

    audit = AuditLog(tmp_path / "orphan3.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    now = pd.Timestamp("2026-09-14T00:00:00Z")
    order_id = "mh20260914-ABCDEFGHIJ-0-0-3"
    journal.record_submit(order_id, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
                          side="BUY", quantity=Decimal("2"), reduce_only=False, leg_index=0)

    class _Client:
        def open_orders(self):
            return [{"symbol": "AAAUSDT", "clientOrderId": order_id}]

        def cancel_order(self, symbol, oid):
            return {}

        def query_order(self, symbol, oid):
            return dict(answer)

    with pytest.raises(DataIntegrityError):
        cancel_orphan_orders(_Client(), "20260914", audit, journal=journal, now=now, taker_fee_bps=4.5)

    assert journal.fills_after(-1) == ()
    assert [s.client_order_id for s in journal.unresolved_submits(since=now - pd.Timedelta(hours=72))] == [order_id]


def test_orphan_sweep_equality_branches(tmp_path) -> None:
    """OrphanSweep compares by fills against lists and by value against sweeps."""
    import pandas as pd
    from src.live.audit import AuditLog
    from src.live.executor import OrphanSweep, cancel_orphan_orders
    from src.live.order_journal import OrderJournal

    audit = AuditLog(tmp_path / "sweep_eq.jsonl")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    now = pd.Timestamp("2026-09-14T00:00:00Z")

    class _Client:
        def open_orders(self):
            return []

    empty = cancel_orphan_orders(_Client(), "20260914", audit, journal=journal, now=now, taker_fee_bps=4.5)
    assert empty == []
    assert empty == ()
    assert empty == OrphanSweep(fills=(), foreign_symbols=())
    assert (empty == object()) is False


@pytest.mark.parametrize("paper_fill_model", ["immediate_taker", None], ids=["immediate_taker", "peg_chase"])
def test_mutation_suppressed_paper_client_still_journals_fills(tmp_path, paper_fill_model) -> None:
    """Production PAPER clients report mode=PAPER (mutations suppressed); their simulated fills must still be
    journaled, because the runner commits the ledger from journal fills only (INV-FILL-WAL)."""
    from src.live.audit import AuditLog
    from src.live.executor import PassiveExecutionPolicy, execute_intents
    from src.live.order_journal import OrderJournal
    from src.live.rest import PaperResponse
    from src.live.settings import ExecutionMode

    class _SuppressedPaperClient:
        mode = ExecutionMode.PAPER

        def __init__(self) -> None:
            self._ticks = 0

        def book_tickers(self):
            self._ticks += 1
            bid, ask = ("100.00", "100.05") if self._ticks <= 4 else ("99.40", "99.50")
            return {"AAAUSDT": {"bidPrice": bid, "askPrice": ask}}

        def new_order(self, params):
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

    policy = PassiveExecutionPolicy(poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0,
                                    taker_cap_bps=15.0, max_slices=1, passive_pricing="anchored")
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    outcomes = execute_intents(_SuppressedPaperClient(), [_shutdown_intent("AAAUSDT")],
                               {"AAAUSDT": _shutdown_filters("AAAUSDT")}, policy,
                               AuditLog(tmp_path / "audit.jsonl"), SteppingClock(3.0), lambda s: None,
                               paper_fill_model=paper_fill_model, journal=journal, attempt=attempt)

    assert outcomes[0].status == "FILLED"
    fills = journal.fills_after(-1)
    assert [(f.quantity, f.simulated) for f in fills] == [(outcomes[0].filled_qty, True)]


def test_paper_ioc_backstop_fill_is_journaled_and_terminal(tmp_path) -> None:
    """A PAPER IOC backstop that fills at posting journals one simulated fill and marks the order FILLED."""
    import pandas as pd

    from src.live.order_journal import OrderJournal
    from src.live.settings import ExecutionMode

    class _StaticPaper(PaperStubClient):
        mode = ExecutionMode.PAPER

        def book_tickers(self):
            return {"AAAUSDT": self.book_ticker("AAAUSDT")}

    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    client = _StaticPaper(touches=[("100.00", "100.20")])

    outcomes = execute_intents(client, [_intent()], {"AAAUSDT": _filters(tick_size="0.01")}, _policy(),
                               AuditLog(tmp_path / "audit.jsonl"), SteppingClock(3.0), lambda _s: None,
                               journal=journal, attempt=attempt)

    assert outcomes[0].status == "FILLED"
    fills = journal.fills_after(-1)
    assert sum(f.quantity for f in fills) == outcomes[0].filled_qty
    assert all(f.simulated and f.reason == "timeout_taker" for f in fills)
    assert journal.unresolved_submits(since=pd.Timestamp("2020-01-01", tz="UTC")) == ()


def test_shutdown_cleanup_failure_is_audited_and_left_for_recovery(tmp_path) -> None:
    """A venue error while settling on shutdown is audited; the order stays unresolved (restart recovery) and
    ExecutionInterrupted is still raised with the fills observed so far."""
    import json
    from types import SimpleNamespace

    import pandas as pd

    from src.live.errors import VenueError
    from src.live.executor import ExecutionInterrupted
    from src.live.order_journal import OrderJournal

    class _CancelFails(_ShutdownLiveClient):
        def cancel_order(self, symbol, orig_client_order_id):
            raise VenueError("venue", code=-1022, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    audit_path = tmp_path / "audit.jsonl"
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    flag = SimpleNamespace(requested=False)
    clock_state = [0.0]

    with pytest.raises(ExecutionInterrupted) as exc_info:
        execute_intents(_CancelFails(flag), [_shutdown_intent("AAAUSDT")], {"AAAUSDT": _shutdown_filters("AAAUSDT")},
                        _shutdown_policy(), AuditLog(audit_path),
                        lambda: clock_state[0], lambda s: clock_state.__setitem__(0, clock_state[0] + s),
                        shutdown=flag, journal=journal, attempt=attempt)

    assert exc_info.value.partial_outcomes[0].filled_qty == Decimal("0.4")
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "abort_cleanup_failed" for e in events)
    unresolved = journal.unresolved_submits(since=pd.Timestamp("2026-01-01", tz="UTC"))
    assert len(unresolved) == 1


def test_execute_intents_requires_journal_and_attempt_together(tmp_path) -> None:
    from src.live.order_journal import OrderJournal

    with pytest.raises(ValueError, match="together"):
        execute_intents(object(), [], {}, _shutdown_policy(), AuditLog(tmp_path / "a.jsonl"), lambda: 0.0,
                        lambda s: None, journal=OrderJournal(tmp_path / "j.jsonl"))


def test_immediate_taker_shutdown_before_first_tick_raises_interrupted(tmp_path) -> None:
    """immediate_taker honours a shutdown requested before any fill: nothing is fetched, simulated or journaled."""
    from types import SimpleNamespace

    from src.live.executor import ExecutionInterrupted
    from src.live.order_journal import OrderJournal

    class _NoCalls:
        def book_tickers(self):
            raise AssertionError("no book fetch after shutdown")

    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    with pytest.raises(ExecutionInterrupted) as exc_info:
        execute_intents(_NoCalls(), [_shutdown_intent("AAAUSDT")], {"AAAUSDT": _shutdown_filters("AAAUSDT")},
                        _shutdown_policy(), AuditLog(tmp_path / "a.jsonl"), lambda: 0.0, lambda s: None,
                        shutdown=SimpleNamespace(requested=True), paper_fill_model="immediate_taker",
                        journal=journal, attempt=attempt)

    assert exc_info.value.partial_outcomes == ()
    assert journal.fills_after(-1) == ()


def test_immediate_taker_journal_failure_clears_sink(tmp_path) -> None:
    """If the WAL write fails, no simulated fill is exposed through the outcome sink (the journal is the only truth)."""
    from src.live.order_journal import OrderJournal
    from src.live.settings import ExecutionMode

    class _DiskFull(OrderJournal):
        def record_fill(self, **kwargs):
            raise OSError(28, "No space left on device")

    class _Paper:
        mode = ExecutionMode.PAPER

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"}}

    journal = _DiskFull(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    sink: list = ["stale"]

    with pytest.raises(OSError, match="No space left"):
        execute_intents(_Paper(), [_shutdown_intent("AAAUSDT")], {"AAAUSDT": _shutdown_filters("AAAUSDT")},
                        _shutdown_policy(), AuditLog(tmp_path / "a.jsonl"), lambda: 0.0, lambda s: None,
                        outcome_sink=sink, paper_fill_model="immediate_taker", journal=journal, attempt=attempt)

    assert sink == []


def test_cancel_fill_race_at_passive_timeout_completes_intent(tmp_path) -> None:
    """The passive-timeout cancel settles a full fill that raced the cancel: the intent ends FILLED (journaled terminal)
    and no IOC backstop is posted."""
    import pandas as pd

    from src.live.order_journal import OrderJournal

    class _RaceClient:
        def __init__(self) -> None:
            self.orders: list = []
            self.cancelled: set = set()

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "100.20"}}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

        def cancel_order(self, symbol, oid):
            self.cancelled.add(oid)
            return {}

        def query_order(self, symbol, oid):
            if oid in self.cancelled:
                return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}
            return {"status": "NEW", "executedQty": "0", "avgPrice": "0"}

    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)
    client = _RaceClient()

    outcomes = execute_intents(client, [_shutdown_intent("AAAUSDT")], {"AAAUSDT": _shutdown_filters("AAAUSDT")},
                               _shutdown_policy(passive_deadline_s=5.0), AuditLog(tmp_path / "a.jsonl"),
                               SteppingClock(1.0), lambda s: None, journal=journal, attempt=attempt)

    assert outcomes[0].status == "FILLED"
    assert outcomes[0].filled_qty == Decimal("1")
    assert len(client.orders) == 1
    assert journal.unresolved_submits(since=pd.Timestamp("2026-01-01", tz="UTC")) == ()


def test_reduce_only_obsolete_ends_only_that_intent_and_journals_terminal(tmp_path) -> None:
    """-2022 (position already closed) on a reduce-only post ends that intent OBSOLETE, journals the id terminal,
    and the other symbol keeps executing."""
    import pandas as pd

    from src.live.errors import OrderObsolete
    from src.live.order_journal import OrderJournal
    from src.live.planner import OrderIntent

    class _Client:
        def __init__(self) -> None:
            self.orders: list = []

        def book_tickers(self):
            return {s: {"bidPrice": "100.00", "askPrice": "100.20"} for s in ("AAAUSDT", "BBBUSDT")}

        def new_order(self, params):
            self.orders.append(params)
            if params["symbol"] == "AAAUSDT":
                raise OrderObsolete("reduce only rejected")
            return {"orderId": len(self.orders)}

        def cancel_order(self, symbol, oid):
            return {}

        def query_order(self, symbol, oid):
            return {"status": "FILLED", "executedQty": "1", "avgPrice": "100.00"}

    reduce = OrderIntent(symbol="AAAUSDT", side="SELL", quantity=Decimal("1"), reduce_only=True,
                         target_qty=Decimal("0"), current_qty=Decimal("1"), client_order_prefix="20260914",
                         leg_index=0, decision_price=Decimal("100.10"))
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _test_attempt(journal)

    outcomes = execute_intents(_Client(), [reduce, _shutdown_intent("BBBUSDT")],
                               {"AAAUSDT": _shutdown_filters("AAAUSDT"), "BBBUSDT": _shutdown_filters("BBBUSDT")},
                               _shutdown_policy(), AuditLog(tmp_path / "a.jsonl"), SteppingClock(1.0), lambda s: None,
                               journal=journal, attempt=attempt)

    by_symbol = {o.symbol: o for o in outcomes}
    assert by_symbol["AAAUSDT"].status == "OBSOLETE"
    assert by_symbol["BBBUSDT"].status == "FILLED"
    assert journal.unresolved_submits(since=pd.Timestamp("2026-01-01", tz="UTC")) == ()


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

    rt = _IntentRuntime(intent=_exec_intent("AAAUSDT"), filters=_exec_filters("AAAUSDT"))
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
