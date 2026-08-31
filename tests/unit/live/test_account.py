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
    assert live_eq == Decimal("2000")

def test_synthetic_flat_snapshot_passes_guards() -> None:
    import pandas as pd

    from src.live.account import assert_suppressed_venue_flat, assert_venue_configuration, synthetic_flat_snapshot

    snap = synthetic_flat_snapshot(pd.Timestamp("2026-08-30 01:00", tz="UTC"))
    assert snap.positions == {}
    assert snap.dual_side_position is False
    assert snap.multi_assets_margin is False
    assert_venue_configuration(snap)
    assert_suppressed_venue_flat(snap)
