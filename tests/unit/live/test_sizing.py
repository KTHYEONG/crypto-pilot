"""SCENARIO_LIVE_04: min_notional 미달 종목은 드롭되고 gross는 재분배되지 않는다."""

from __future__ import annotations

import pandas as pd
from decimal import Decimal

from src.live.filters import SymbolFilters
from src.live.sizing import target_quantities


def _filters(symbol: str) -> SymbolFilters:
    return SymbolFilters(
        symbol=symbol,
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )


def test_SCENARIO_LIVE_04_min_notional_drop_no_redistribution() -> None:
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]
    weights = pd.Series(
        [0.30, 0.25, 0.20, 0.002, 0.001], index=symbols, dtype="float64"
    )
    marks = {s: Decimal("100") for s in symbols}
    filters = {s: _filters(s) for s in symbols}
    equity = Decimal("1000")

    targets, dropped = target_quantities(weights, marks, filters, equity)

    assert sorted(targets) == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    reasons = {d.symbol: d.reason for d in dropped}
    assert reasons == {
        "DDDUSDT": "MIN_NOTIONAL",
        "EEEUSDT": "MIN_NOTIONAL",
    }

    # 재분배 금지: 남은 3종목 목표 노셔널 합은 드롭 전 그 3종목 합과 정확히 같다.
    kept_before = sum(
        (equity * Decimal(str(float(weights[s])))) for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    )
    kept_after = sum((targets[s] * marks[s]) for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT"))
    relative_error = abs(kept_after - kept_before) / kept_before
    assert relative_error < Decimal("1e-12")


def test_missing_filter_or_mark_dropped_as_not_tradable() -> None:
    weights = pd.Series([0.5, 0.5], index=["XXXUSDT", "YYYUSDT"], dtype="float64")
    marks = {"XXXUSDT": Decimal("100")}
    filters = {"XXXUSDT": _filters("XXXUSDT")}
    targets, dropped = target_quantities(weights, marks, filters, Decimal("1000"))
    assert targets == {} or list(targets) == ["XXXUSDT"]
    assert all(d.reason == "NOT_TRADABLE" for d in dropped)

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_04_MIN_NOTIONAL_DROP_NO_REDISTRIBUTION",
)
