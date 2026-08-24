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
