"""SCENARIO_LIVE_08: 재조정은 불일치 시 HALT 하고 절대 자동 보정하지 않는다."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.live.account import (
    AccountSnapshot,
    assert_venue_configuration,
    reconcile_or_halt,
)
from src.live.errors import ReconciliationBreach, RiskGateBreach


def _snapshot(positions: dict[str, Decimal], **overrides) -> AccountSnapshot:
    defaults = {
        "taken_at": pd.Timestamp("2026-08-24 01:00Z"),
        "wallet_balance": Decimal("1000"),
        "available_balance": Decimal("900"),
        "total_maint_margin": Decimal("10"),
        "unrealized_pnl": Decimal("0"),
        "positions": positions,
        "dual_side_position": False,
        "multi_assets_margin": False,
    }
    defaults.update(overrides)
    return AccountSnapshot(**defaults)


def test_SCENARIO_LIVE_08_reconcile_halts_on_divergence() -> None:
    ledger = {"AAAUSDT": Decimal("10")}

    within_tolerance = _snapshot({"AAAUSDT": Decimal("10.001")})
    reconcile_or_halt(within_tolerance, ledger, qty_tolerance_fraction=0.01)

    divergent_ledger = {"AAAUSDT": Decimal("10")}
    divergent_snapshot = _snapshot({"AAAUSDT": Decimal("12")})
    with pytest.raises(ReconciliationBreach):
        reconcile_or_halt(divergent_snapshot, divergent_ledger, qty_tolerance_fraction=0.01)
    # 자동 보정 금지: 원장 객체는 호출 전후로 동일하다.
    assert divergent_ledger == {"AAAUSDT": Decimal("10")}

    ghost_snapshot = _snapshot({"GHOSTUSDT": Decimal("5")})
    with pytest.raises(ReconciliationBreach):
        reconcile_or_halt(ghost_snapshot, {}, qty_tolerance_fraction=0.01)


def test_venue_configuration_guard() -> None:
    assert_venue_configuration(_snapshot({}))

    with pytest.raises(RiskGateBreach):
        assert_venue_configuration(_snapshot({}, multi_assets_margin=True))
    with pytest.raises(RiskGateBreach):
        assert_venue_configuration(_snapshot({}, dual_side_position=True))

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_08_RECONCILE_HALTS_ON_DIVERGENCE",
)

def test_SCENARIO_PARITY_06_paper_virtual_mtm_equity():
    """SCENARIO_PARITY_06-paper-virtual-mtm-equity"""
    from decimal import Decimal
    import pandas as pd
    from src.live.account import AccountSnapshot, resolve_sizing_equity
    from src.live.settings import ExecutionMode
    from src.live.errors import RiskGateBreach
    snapshot = AccountSnapshot(taken_at=pd.Timestamp("2026-01-01", tz="UTC"), wallet_balance=Decimal("0"), available_balance=Decimal("0"), total_maint_margin=Decimal("0"), unrealized_pnl=Decimal("0"), positions={}, dual_side_position=False, multi_assets_margin=False)
    # PAPER virtual MTM: cash 1500 + positions 5*100=500 => min(2000, cap 2000)=2000
    assert resolve_sizing_equity(snapshot, Decimal("2000"), mode=ExecutionMode.PAPER, cash_usdt=Decimal("1500"), positions={"BTCUSDT": Decimal("5")}, marks={"BTCUSDT": Decimal("100")}) == Decimal("2000")
    # LIVE_TESTNET should breach because wallet 0 -> equity 0 -> RiskGateBreach
    try:
        resolve_sizing_equity(snapshot, Decimal("2000"), mode=ExecutionMode.LIVE_TESTNET, cash_usdt=Decimal("1500"), positions={"BTCUSDT": Decimal("5")}, marks={"BTCUSDT": Decimal("100")})
        pytest.fail("should have raised")
    except RiskGateBreach:
        pass
    # cash None seeds with cap -> no breach
    assert resolve_sizing_equity(snapshot, Decimal("2000"), mode=ExecutionMode.PAPER, cash_usdt=None, positions={}, marks={}) == Decimal("2000")


def test_fetch_account_snapshot_pulls_dual_side_from_dedicated_endpoint() -> None:
    """Real Binance /fapi/v2/account omits dualSidePosition; it must come from
    GET /fapi/v1/positionSide/dual. multiAssetsMargin stays in the account
    payload when present."""
    from src.live.account import fetch_account_snapshot

    calls: list[str] = []

    class _Client:
        def request(self, method, path, params=None, *, signed=False):
            calls.append(path)
            if path == "/fapi/v2/account":
                return {
                    "totalWalletBalance": "2000",
                    "availableBalance": "1900",
                    "totalInitialMargin": "10",
                    "totalUnrealizedProfit": "0",
                    "multiAssetsMargin": False,  # present, dualSidePosition absent
                }
            if path == "/fapi/v2/positionRisk":
                return []
            if path == "/fapi/v1/positionSide/dual":
                return {"dualSidePosition": False}
            raise AssertionError(f"unexpected path {path}")

    snap = fetch_account_snapshot(_Client(), now=pd.Timestamp("2026-08-30 00:00Z"))
    assert snap.dual_side_position is False
    assert snap.multi_assets_margin is False
    assert "/fapi/v1/positionSide/dual" in calls


def test_fetch_account_snapshot_falls_back_for_multi_assets_on_v3_shape() -> None:
    from src.live.account import fetch_account_snapshot

    class _Client:
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {
                    "totalWalletBalance": "2000",
                    "availableBalance": "1900",
                    "totalInitialMargin": "10",
                    "totalUnrealizedProfit": "0",
                }  # neither flag present (v3-like)
            if path == "/fapi/v2/positionRisk":
                return []
            if path == "/fapi/v1/positionSide/dual":
                return {"dualSidePosition": True}
            if path == "/fapi/v1/multiAssetsMargin":
                return {"multiAssetsMargin": False}
            raise AssertionError(f"unexpected path {path}")

    snap = fetch_account_snapshot(_Client(), now=pd.Timestamp("2026-08-30 00:00Z"))
    assert snap.dual_side_position is True
    assert snap.multi_assets_margin is False



def test_resolve_sizing_equity_paper_uncapped() -> None:
    from decimal import Decimal

    from src.live.account import AccountSnapshot, resolve_sizing_equity
    from src.live.settings import ExecutionMode
    import pandas as pd

    snap = AccountSnapshot(
        taken_at=pd.Timestamp("2026-08-30", tz="UTC"), wallet_balance=Decimal(0), available_balance=Decimal(0),
        total_maint_margin=Decimal(0), unrealized_pnl=Decimal(0), positions={"BTCUSDT": Decimal("1")},
        dual_side_position=False, multi_assets_margin=False,
    )
    marks = {"BTCUSDT": Decimal("5000")}
    eq = resolve_sizing_equity(snap, Decimal("2000"), mode=ExecutionMode.PAPER, cash_usdt=Decimal("0"), positions={"BTCUSDT": Decimal("1")}, marks=marks)
    assert eq == Decimal("5000")

    live_snap = AccountSnapshot(
        taken_at=pd.Timestamp("2026-08-30", tz="UTC"), wallet_balance=Decimal("5000"), available_balance=Decimal("5000"),
        total_maint_margin=Decimal(0), unrealized_pnl=Decimal(0), positions={}, dual_side_position=False, multi_assets_margin=False,
    )
    live_eq = resolve_sizing_equity(live_snap, Decimal("2000"), mode=ExecutionMode.LIVE_MAINNET)
    assert live_eq == Decimal("5000")

def test_synthetic_flat_snapshot_passes_guards() -> None:
    import pandas as pd

    from src.live.account import assert_suppressed_venue_flat, assert_venue_configuration, synthetic_flat_snapshot

    snap = synthetic_flat_snapshot(pd.Timestamp("2026-08-30 01:00", tz="UTC"))
    assert snap.positions == {}
    assert snap.dual_side_position is False
    assert snap.multi_assets_margin is False
    assert_venue_configuration(snap)
    assert_suppressed_venue_flat(snap)


def test_parse_leverage_brackets_sorts_by_bracket_and_parses_decimals() -> None:
    from decimal import Decimal
    from src.live.account import LeverageBracket, parse_leverage_brackets

    payload = [{"symbol": "BTCUSDT", "notionalCoef": 1.0, "brackets": [
        {"bracket": 2, "initialLeverage": 25, "notionalCap": 250000, "notionalFloor": 50000, "maintMarginRatio": 0.02, "cum": 750},
        {"bracket": 1, "initialLeverage": 125, "notionalCap": 50000, "notionalFloor": 0, "maintMarginRatio": 0.004, "cum": 0},
    ]}]

    parsed = parse_leverage_brackets(payload)

    assert parsed == {"BTCUSDT": (
        LeverageBracket(bracket=1, initial_leverage=125, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),
        LeverageBracket(bracket=2, initial_leverage=25, notional_cap=Decimal("250000"), notional_floor=Decimal("50000")),
    )}




@pytest.mark.parametrize("payload", [
    {"symbol": "BTCUSDT"},
    [{"symbol": "BTCUSDT", "brackets": []}],
    [{"brackets": [{"bracket": 1, "initialLeverage": 20, "notionalCap": 1, "notionalFloor": 0}]}],
    [{"symbol": "BTCUSDT", "brackets": [{"bracket": 1, "initialLeverage": 20, "notionalFloor": 0}]}],
])
def test_parse_leverage_brackets_rejects_malformed_payload(payload) -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.live.account import parse_leverage_brackets

    with pytest.raises(DataIntegrityError, match="leverageBracket"):
        parse_leverage_brackets(payload)


def test_parse_position_config_reads_margin_type_and_leverage() -> None:
    from src.live.account import VenueSymbolConfig, parse_position_config

    payload = [
        {"symbol": "BTCUSDT", "positionAmt": "0", "marginType": "cross", "leverage": "20"},
        {"symbol": "ETHUSDT", "positionAmt": "0.1"},
    ]

    parsed = parse_position_config(payload)

    assert parsed == {
        "BTCUSDT": VenueSymbolConfig(symbol="BTCUSDT", margin_type="cross", leverage=20),
        "ETHUSDT": VenueSymbolConfig(symbol="ETHUSDT", margin_type=None, leverage=None),
    }


def test_parse_position_config_rejects_unexpected_schema() -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.live.account import parse_position_config

    with pytest.raises(DataIntegrityError, match="positionRisk"):
        parse_position_config({"symbol": "BTCUSDT"})
    with pytest.raises(DataIntegrityError, match="positionRisk"):
        parse_position_config([{"positionAmt": "0"}])


def test_required_leverage_applies_buffer_and_bracket_ceiling() -> None:
    from src.live.account import required_leverage

    assert required_leverage(3.0, 0.25, 125) == 4
    assert required_leverage(3.0, 0.25, 3) == 3
    assert required_leverage(0.5, 0.0, 20) == 1
    assert required_leverage(2.2, 0.0, 50) == 3


def test_max_notional_at_leverage_picks_highest_eligible_cap() -> None:
    from decimal import Decimal
    from src.live.account import LeverageBracket, max_notional_at_leverage

    brackets = (
        LeverageBracket(bracket=1, initial_leverage=50, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),
        LeverageBracket(bracket=2, initial_leverage=25, notional_cap=Decimal("250000"), notional_floor=Decimal("50000")),
        LeverageBracket(bracket=3, initial_leverage=10, notional_cap=Decimal("1000000"), notional_floor=Decimal("250000")),
    )

    assert max_notional_at_leverage(brackets, 4) == Decimal("1000000")
    assert max_notional_at_leverage(brackets, 20) == Decimal("250000")
    assert max_notional_at_leverage(brackets, 50) == Decimal("50000")
    assert max_notional_at_leverage(brackets, 60) == Decimal("0")


def test_plan_venue_leverage_lists_only_mismatched_symbols() -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    class _Audit:
        def __init__(self) -> None:
            self.records: list[tuple[str, dict]] = []

        def record(self, event: str, **fields) -> None:
            self.records.append((event, fields))

    class _FakeVenue:
        def __init__(self, position_rows, failures=None) -> None:
            self.calls: list[tuple[str, str, dict]] = []
            self._rows = position_rows
            self._failures = failures or {}

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path, dict(params or {})))
            if path in self._failures:
                raise self._failures[path]
            if path == "/fapi/v2/positionRisk":
                return self._rows
            return {"code": 200, "msg": "success"}

    def _venue_error(code: int, path: str) -> VenueError:
        return VenueError("venue rejected request", code=code, http_status=400, path=path, payload_digest="000000000000")

    from src.live.account import LeverageBracket

    brackets = {
        "AAAUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
        "BUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
    }
    from src.live.account import VenueSymbolConfig, plan_venue_leverage

    brackets["CUSDT"] = brackets["AAAUSDT"]
    configs = {
        "AAAUSDT": VenueSymbolConfig(symbol="AAAUSDT", margin_type="cross", leverage=4),
        "BUSDT": VenueSymbolConfig(symbol="BUSDT", margin_type="isolated", leverage=10),
    }

    plan = plan_venue_leverage(["CUSDT", "BUSDT", "AAAUSDT"], brackets, configs, max_gross_leverage=3.0, buffer_fraction=0.25)

    assert dict(plan.target_leverage) == {"AAAUSDT": 4, "BUSDT": 4, "CUSDT": 4}
    assert plan.margin_type_changes == ("BUSDT", "CUSDT")
    assert plan.leverage_changes == ("BUSDT", "CUSDT")


def test_plan_venue_leverage_fails_closed_without_bracket() -> None:
    import pytest
    from src.live.account import plan_venue_leverage
    from src.live.errors import RiskGateBreach

    with pytest.raises(RiskGateBreach, match="no leverage bracket for ZZZUSDT"):
        plan_venue_leverage(["ZZZUSDT"], {}, {}, max_gross_leverage=3.0, buffer_fraction=0.25)


def test_ensure_venue_leverage_sets_cross_and_leverage_only_when_needed() -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    class _Audit:
        def __init__(self) -> None:
            self.records: list[tuple[str, dict]] = []

        def record(self, event: str, **fields) -> None:
            self.records.append((event, fields))

    class _FakeVenue:
        def __init__(self, position_rows, failures=None) -> None:
            self.calls: list[tuple[str, str, dict]] = []
            self._rows = position_rows
            self._failures = failures or {}

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path, dict(params or {})))
            if path in self._failures:
                raise self._failures[path]
            if path == "/fapi/v2/positionRisk":
                return self._rows
            return {"code": 200, "msg": "success"}

    def _venue_error(code: int, path: str) -> VenueError:
        return VenueError("venue rejected request", code=code, http_status=400, path=path, payload_digest="000000000000")

    from src.live.account import LeverageBracket

    brackets = {
        "AAAUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
        "BUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
    }
    from src.live.account import ensure_venue_leverage

    venue = _FakeVenue([
        {"symbol": "AAAUSDT", "positionAmt": "0", "marginType": "cross", "leverage": "4"},
        {"symbol": "BUSDT", "positionAmt": "0", "marginType": "isolated", "leverage": "20"},
    ])
    audit = _Audit()

    leverages = ensure_venue_leverage(venue, ["AAAUSDT", "BUSDT"], brackets, max_gross_leverage=3.0, buffer_fraction=0.25, audit=audit)

    assert leverages == {"AAAUSDT": 4, "BUSDT": 4}
    assert venue.calls == [
        ("GET", "/fapi/v2/positionRisk", {}),
        ("POST", "/fapi/v1/marginType", {"symbol": "BUSDT", "marginType": "CROSSED"}),
        ("POST", "/fapi/v1/leverage", {"symbol": "BUSDT", "leverage": 4}),
    ]
    assert audit.records == [
        ("venue_margin_type_set", {"symbol": "BUSDT", "margin_type": "CROSSED"}),
        ("venue_leverage_set", {"symbol": "BUSDT", "leverage": 4}),
    ]




@pytest.mark.parametrize("code", [-4047, -4048])
def test_ensure_venue_leverage_fails_closed_when_margin_change_blocked(code) -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    class _Audit:
        def __init__(self) -> None:
            self.records: list[tuple[str, dict]] = []

        def record(self, event: str, **fields) -> None:
            self.records.append((event, fields))

    class _FakeVenue:
        def __init__(self, position_rows, failures=None) -> None:
            self.calls: list[tuple[str, str, dict]] = []
            self._rows = position_rows
            self._failures = failures or {}

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path, dict(params or {})))
            if path in self._failures:
                raise self._failures[path]
            if path == "/fapi/v2/positionRisk":
                return self._rows
            return {"code": 200, "msg": "success"}

    def _venue_error(code: int, path: str) -> VenueError:
        return VenueError("venue rejected request", code=code, http_status=400, path=path, payload_digest="000000000000")

    from src.live.account import LeverageBracket

    brackets = {
        "AAAUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
        "BUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
    }
    import pytest
    from src.live.account import ensure_venue_leverage
    from src.live.errors import RiskGateBreach

    venue = _FakeVenue([], failures={"/fapi/v1/marginType": _venue_error(code, "/fapi/v1/marginType")})

    with pytest.raises(RiskGateBreach, match=f"AAAUSDT \\(code={code}\\)"):
        ensure_venue_leverage(venue, ["AAAUSDT"], brackets, max_gross_leverage=3.0, buffer_fraction=0.25, audit=_Audit())

    assert [path for _, path, _ in venue.calls] == ["/fapi/v2/positionRisk", "/fapi/v1/marginType"]


def test_ensure_venue_leverage_fails_closed_when_leverage_change_blocked() -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    class _Audit:
        def __init__(self) -> None:
            self.records: list[tuple[str, dict]] = []

        def record(self, event: str, **fields) -> None:
            self.records.append((event, fields))

    class _FakeVenue:
        def __init__(self, position_rows, failures=None) -> None:
            self.calls: list[tuple[str, str, dict]] = []
            self._rows = position_rows
            self._failures = failures or {}

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path, dict(params or {})))
            if path in self._failures:
                raise self._failures[path]
            if path == "/fapi/v2/positionRisk":
                return self._rows
            return {"code": 200, "msg": "success"}

    def _venue_error(code: int, path: str) -> VenueError:
        return VenueError("venue rejected request", code=code, http_status=400, path=path, payload_digest="000000000000")

    from src.live.account import LeverageBracket

    brackets = {
        "AAAUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
        "BUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
    }
    import pytest
    from src.live.account import ensure_venue_leverage
    from src.live.errors import RiskGateBreach

    venue = _FakeVenue(
        [{"symbol": "AAAUSDT", "positionAmt": "1", "marginType": "cross", "leverage": "20"}],
        failures={"/fapi/v1/leverage": _venue_error(-4161, "/fapi/v1/leverage")},
    )

    with pytest.raises(RiskGateBreach, match=r"leverage change to 4 blocked for AAAUSDT \(code=-4161\)"):
        ensure_venue_leverage(venue, ["AAAUSDT"], brackets, max_gross_leverage=3.0, buffer_fraction=0.25, audit=_Audit())


def test_ensure_venue_leverage_propagates_unrelated_venue_errors() -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    class _Audit:
        def __init__(self) -> None:
            self.records: list[tuple[str, dict]] = []

        def record(self, event: str, **fields) -> None:
            self.records.append((event, fields))

    class _FakeVenue:
        def __init__(self, position_rows, failures=None) -> None:
            self.calls: list[tuple[str, str, dict]] = []
            self._rows = position_rows
            self._failures = failures or {}

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path, dict(params or {})))
            if path in self._failures:
                raise self._failures[path]
            if path == "/fapi/v2/positionRisk":
                return self._rows
            return {"code": 200, "msg": "success"}

    def _venue_error(code: int, path: str) -> VenueError:
        return VenueError("venue rejected request", code=code, http_status=400, path=path, payload_digest="000000000000")

    from src.live.account import LeverageBracket

    brackets = {
        "AAAUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
        "BUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
    }
    import pytest
    from src.live.account import ensure_venue_leverage

    margin_failure = _FakeVenue([], failures={"/fapi/v1/marginType": _venue_error(-1022, "/fapi/v1/marginType")})
    with pytest.raises(VenueError) as margin_info:
        ensure_venue_leverage(margin_failure, ["AAAUSDT"], brackets, max_gross_leverage=3.0, buffer_fraction=0.25, audit=_Audit())
    assert margin_info.value.code == -1022

    leverage_failure = _FakeVenue(
        [{"symbol": "AAAUSDT", "positionAmt": "0", "marginType": "cross", "leverage": "20"}],
        failures={"/fapi/v1/leverage": _venue_error(-1022, "/fapi/v1/leverage")},
    )
    with pytest.raises(VenueError) as leverage_info:
        ensure_venue_leverage(leverage_failure, ["AAAUSDT"], brackets, max_gross_leverage=3.0, buffer_fraction=0.25, audit=_Audit())
    assert leverage_info.value.code == -1022


def test_reject_intents_over_notional_cap_keeps_reduce_only_and_audits_rejections() -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    class _Audit:
        def __init__(self) -> None:
            self.records: list[tuple[str, dict]] = []

        def record(self, event: str, **fields) -> None:
            self.records.append((event, fields))

    class _FakeVenue:
        def __init__(self, position_rows, failures=None) -> None:
            self.calls: list[tuple[str, str, dict]] = []
            self._rows = position_rows
            self._failures = failures or {}

        def request(self, method, path, params=None, *, signed=False):
            self.calls.append((method, path, dict(params or {})))
            if path in self._failures:
                raise self._failures[path]
            if path == "/fapi/v2/positionRisk":
                return self._rows
            return {"code": 200, "msg": "success"}

    def _venue_error(code: int, path: str) -> VenueError:
        return VenueError("venue rejected request", code=code, http_status=400, path=path, payload_digest="000000000000")

    from src.live.account import LeverageBracket

    brackets = {
        "AAAUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
        "BUSDT": (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("50000"), notional_floor=Decimal("0")),),
    }
    from src.live.account import reject_intents_over_notional_cap
    from src.live.planner import OrderIntent

    brackets["AAAUSDT"] = (LeverageBracket(bracket=1, initial_leverage=20, notional_cap=Decimal("30"), notional_floor=Decimal("0")),)

    def _intent(symbol: str, target: str, reduce_only: bool, leg: int) -> OrderIntent:
        return OrderIntent(
            symbol=symbol, side="BUY", quantity=Decimal("0.4"), reduce_only=reduce_only,
            target_qty=Decimal(target), current_qty=Decimal("-0.1"), client_order_prefix="20260824",
            leg_index=leg, decision_price=Decimal("100"),
        )

    close_leg = _intent("AAAUSDT", "0.4", True, 0)
    open_leg = _intent("AAAUSDT", "0.4", False, 1)
    small = _intent("BUSDT", "0.2", False, 0)
    audit = _Audit()

    kept = reject_intents_over_notional_cap([close_leg, open_leg, small], brackets, {"AAAUSDT": 4, "BUSDT": 4}, audit)

    assert kept == [close_leg, small]
    assert audit.records == [("notional_cap_rejected", {
        "symbol": "AAAUSDT", "target_notional": "40.0", "notional_cap": "30", "leverage": 4,
    })]


def test_settled_delisting_symbols_requires_delisted_status_and_elapsed_delivery() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.account import settled_delisting_symbols

    now = pd.Timestamp("2026-09-14 01:00Z")
    past = int((now - pd.Timedelta(days=1)).value // 1_000_000)
    future = int((now + pd.Timedelta(days=1)).value // 1_000_000)
    exchange_info = {"symbols": [
        {"symbol": "SETUSDT", "status": "SETTLING", "deliveryDate": past},
        {"symbol": "CLOUSDT", "status": "CLOSE", "deliveryDate": past},
        {"symbol": "SOONUSDT", "status": "SETTLING", "deliveryDate": future},
        {"symbol": "LIVEUSDT", "status": "TRADING", "deliveryDate": past},
        {"symbol": "HELDUSDT", "status": "SETTLING", "deliveryDate": past},
        {"symbol": "NODATEUSDT", "status": "SETTLING"},
    ]}
    ledger = {
        "SETUSDT": Decimal("1"), "CLOUSDT": Decimal("-2"), "SOONUSDT": Decimal("1"), "LIVEUSDT": Decimal("1"),
        "HELDUSDT": Decimal("1"), "NODATEUSDT": Decimal("1"), "GONEUSDT": Decimal("1"), "ZEROUSDT": Decimal("0"),
    }
    venue = {"HELDUSDT": Decimal("1")}

    assert settled_delisting_symbols(exchange_info, venue, ledger, now=now) == ("CLOUSDT", "SETUSDT")


def test_reconcile_or_halt_accepts_settled_symbols_only_when_venue_flat() -> None:
    from decimal import Decimal
    import pandas as pd
    import pytest
    from src.live.account import AccountSnapshot, reconcile_or_halt
    from src.live.errors import ReconciliationBreach

    def _snap(positions):
        return AccountSnapshot(
            taken_at=pd.Timestamp("2026-09-14 01:00Z"), wallet_balance=Decimal("1000"), available_balance=Decimal("900"),
            total_maint_margin=Decimal("0"), unrealized_pnl=Decimal("0"), positions=positions,
            dual_side_position=False, multi_assets_margin=False,
        )

    reconcile_or_halt(_snap({}), {"SETUSDT": Decimal("1")}, qty_tolerance_fraction=0.0, settled_symbols=("SETUSDT",))

    with pytest.raises(ReconciliationBreach, match="SETUSDT"):
        reconcile_or_halt(_snap({"SETUSDT": Decimal("0.5")}), {"SETUSDT": Decimal("1")}, qty_tolerance_fraction=0.0, settled_symbols=("SETUSDT",))
    with pytest.raises(ReconciliationBreach, match="OTHERUSDT"):
        reconcile_or_halt(_snap({}), {"OTHERUSDT": Decimal("1")}, qty_tolerance_fraction=0.0, settled_symbols=("SETUSDT",))



def test_live_equity_is_uncapped() -> None:
    from src.live.account import resolve_sizing_equity
    from src.live.settings import ExecutionMode

    snap = _snapshot({}, wallet_balance=Decimal("5000"), unrealized_pnl=Decimal("250"))
    assert resolve_sizing_equity(snap, Decimal("2100"), mode=ExecutionMode.LIVE_MAINNET) == Decimal("5250")


def test_live_non_positive_equity_fails_closed() -> None:
    import pytest
    from src.live.account import resolve_sizing_equity
    from src.live.errors import RiskGateBreach
    from src.live.settings import ExecutionMode

    snap = _snapshot({}, wallet_balance=Decimal("0"), unrealized_pnl=Decimal("-1"))
    with pytest.raises(RiskGateBreach):
        resolve_sizing_equity(snap, Decimal("2100"), mode=ExecutionMode.LIVE_MAINNET)


def test_paper_equity_uses_virtual_mtm() -> None:
    from src.live.account import resolve_sizing_equity
    from src.live.settings import ExecutionMode

    snap = _snapshot({})
    eq = resolve_sizing_equity(
        snap, Decimal("2000"), mode=ExecutionMode.PAPER,
        cash_usdt=Decimal("2739.16"), positions={}, marks={},
    )
    assert eq == Decimal("2739.16")
