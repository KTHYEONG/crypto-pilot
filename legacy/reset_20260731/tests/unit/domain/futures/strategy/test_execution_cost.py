from __future__ import annotations

import pytest

from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def test_default_one_way_bps() -> None:
    m = ExecutionCostModel()
    # 0.75*2 + 0.25*5 + 1 = 1.5 + 1.25 + 1 = 3.75
    assert m.one_way_bps() == pytest.approx(3.75, rel=1e-6)


def test_default_round_trip_bps() -> None:
    m = ExecutionCostModel()
    assert m.round_trip_bps() == pytest.approx(7.5, rel=1e-6)


def test_default_stress_round_trip_bps() -> None:
    m = ExecutionCostModel()
    assert m.stress_round_trip_bps() == pytest.approx(11.25, rel=1e-6)


def test_taker_only_round_trip() -> None:
    m = ExecutionCostModel(maker_ratio=0.0, slippage_bps=0.0, impact_coeff_bps=0.0)
    assert m.round_trip_bps() == pytest.approx(10.0, rel=1e-6)


def test_maker_only_round_trip() -> None:
    m = ExecutionCostModel(maker_ratio=1.0, slippage_bps=0.0, impact_coeff_bps=0.0)
    assert m.round_trip_bps() == pytest.approx(4.0, rel=1e-6)


def test_stress_multiplier_scales_rt() -> None:
    m = ExecutionCostModel(stress_multiplier=2.0)
    assert m.stress_round_trip_bps() == pytest.approx(m.round_trip_bps() * 2.0, rel=1e-6)


def test_invalid_maker_ratio_raises() -> None:
    with pytest.raises(ValueError, match="maker_ratio"):
        ExecutionCostModel(maker_ratio=1.5)


def test_negative_fee_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ExecutionCostModel(maker_fee_bps=-1.0)


def test_stress_multiplier_below_one_raises() -> None:
    with pytest.raises(ValueError, match="stress_multiplier"):
        ExecutionCostModel(stress_multiplier=0.9)


def test_impact_coeff_adds_to_one_way() -> None:
    m = ExecutionCostModel(impact_coeff_bps=2.0, slippage_bps=0.0, maker_ratio=1.0)
    assert m.one_way_bps() == pytest.approx(4.0, rel=1e-6)
