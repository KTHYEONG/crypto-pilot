"""Re-export contract for src.mhs.types: params-owned tunables must stay importable."""

from __future__ import annotations

import pytest

import src.mhs.types as types
from src.mhs.params import (
    EXPOSURE_DRAWDOWN_BRAKE_FLOOR,
    EXPOSURE_DRAWDOWN_BRAKE_K,
)
from src.mhs.types import ExecutionSpec


def test_brake_constants_reexported_from_params() -> None:
    """types 재수출 경로가 params 단일 소스와 동일 객체를 노출한다(I2)."""
    assert types.EXPOSURE_DRAWDOWN_BRAKE_K is EXPOSURE_DRAWDOWN_BRAKE_K
    assert types.EXPOSURE_DRAWDOWN_BRAKE_FLOOR is EXPOSURE_DRAWDOWN_BRAKE_FLOOR


def test_all_covers_reexported_brake_constants() -> None:
    assert "EXPOSURE_DRAWDOWN_BRAKE_K" in types.__all__
    assert "EXPOSURE_DRAWDOWN_BRAKE_FLOOR" in types.__all__


def test_SCENARIO_MHS_PEG_CHASE_06_SPEC_DEFAULTS_ARE_BIT_IDENTICAL() -> None:
    """SCENARIO_MHS_PEG_CHASE_06_SPEC_DEFAULTS_ARE_BIT_IDENTICAL: new peg-chase
    fields freeze the pre-change behaviour: the bare default equals an explicit
    legacy-value construction, and invalid domains fail closed in __post_init__."""
    assert ExecutionSpec() == ExecutionSpec(
        decision_anchor="decision_bar",
        peg_passive_fraction=0.6,
        peg_chase_band_bps=10.0,
    )
    with pytest.raises(ValueError, match="peg_passive_fraction"):
        ExecutionSpec(peg_passive_fraction=0.0)
    with pytest.raises(ValueError, match="peg_passive_fraction"):
        ExecutionSpec(peg_passive_fraction=1.5)
    with pytest.raises(ValueError, match="peg_chase_band_bps"):
        ExecutionSpec(peg_chase_band_bps=0.0)
