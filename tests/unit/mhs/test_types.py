"""Re-export contract for src.mhs.types: params-owned tunables must stay importable."""

from __future__ import annotations

import src.mhs.types as types
from src.mhs.params import (
    EXPOSURE_DRAWDOWN_BRAKE_FLOOR,
    EXPOSURE_DRAWDOWN_BRAKE_K,
)


def test_brake_constants_reexported_from_params() -> None:
    """types 재수출 경로가 params 단일 소스와 동일 객체를 노출한다(I2)."""
    assert types.EXPOSURE_DRAWDOWN_BRAKE_K is EXPOSURE_DRAWDOWN_BRAKE_K
    assert types.EXPOSURE_DRAWDOWN_BRAKE_FLOOR is EXPOSURE_DRAWDOWN_BRAKE_FLOOR


def test_all_covers_reexported_brake_constants() -> None:
    assert "EXPOSURE_DRAWDOWN_BRAKE_K" in types.__all__
    assert "EXPOSURE_DRAWDOWN_BRAKE_FLOOR" in types.__all__
