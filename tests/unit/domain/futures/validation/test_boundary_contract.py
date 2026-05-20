"""Phase 6: PurgeBarsRegistry boundary contract 테스트.

사양서 §8.2 기준.
"""

from __future__ import annotations

import pytest

from src.domain.futures.validation.boundary_contract import (
    ModulePurgeBarsMeta,
    PurgeBarsRegistry,
)


class TestBoundaryContract:
    """PurgeBarsRegistry fail-fast 및 max 반환 검증."""

    def test_empty_registry_raises_runtime_error(self) -> None:
        """빈 레지스트리에서 get_boundary_purge_bars → RuntimeError."""
        reg = PurgeBarsRegistry()
        with pytest.raises(RuntimeError, match="Fail-fast"):
            reg.get_boundary_purge_bars()

    def test_multiple_modules_returns_max(self) -> None:
        """여러 모듈 등록 시 max purge_bars 반환."""
        reg = PurgeBarsRegistry()
        reg.register(ModulePurgeBarsMeta("scaler", 24, "fit_window=24bars"))
        reg.register(ModulePurgeBarsMeta("label", 6, "label_horizon=6bars"))

        result = reg.get_boundary_purge_bars()
        assert result == 24, f"max purge_bars=24이어야 함: {result}"

    def test_single_module_returns_its_purge_bars(self) -> None:
        """단일 모듈 등록 → 해당 모듈의 purge_bars 반환."""
        reg = PurgeBarsRegistry()
        reg.register(ModulePurgeBarsMeta("hmm_regime", 48, "fit_window=48bars"))

        result = reg.get_boundary_purge_bars()
        assert result == 48

    def test_validate_empty_raises(self) -> None:
        """미등록 상태 validate() → RuntimeError."""
        reg = PurgeBarsRegistry()
        with pytest.raises(RuntimeError):
            reg.validate()

    def test_validate_with_modules_does_not_raise(self) -> None:
        """모듈 등록 후 validate() → 오류 없음."""
        reg = PurgeBarsRegistry()
        reg.register(ModulePurgeBarsMeta("feature_eng", 12, "rolling=12bars"))
        # Should not raise
        reg.validate()

    def test_oos_start_bar_is_after_purge(self) -> None:
        """purge 적용 후 IS 말단 ~ OOS 시작 사이 데이터 사용 불가 검증.

        IS 마지막 bar = T, purge_bars = 24 → OOS 첫 사용 가능 bar = T + 24.
        """
        reg = PurgeBarsRegistry()
        reg.register(ModulePurgeBarsMeta("scaler", 24, "fit_window=24bars"))
        reg.register(ModulePurgeBarsMeta("label", 8, "label_horizon=8bars"))

        purge_bars = reg.get_boundary_purge_bars()  # max = 24
        is_end_idx = 1000  # IS 마지막 bar 인덱스

        oos_first_usable = is_end_idx + purge_bars
        assert oos_first_usable == 1024, (
            f"OOS 첫 사용 가능 bar index = {oos_first_usable}, expected 1024"
        )

    def test_register_overwrite_same_module(self) -> None:
        """동일 모듈명 재등록 시 최신값으로 갱신."""
        reg = PurgeBarsRegistry()
        reg.register(ModulePurgeBarsMeta("scaler", 12, "old value"))
        reg.register(ModulePurgeBarsMeta("scaler", 36, "new value"))

        result = reg.get_boundary_purge_bars()
        assert result == 36, f"재등록 후 purge_bars=36이어야 함: {result}"

    def test_purge_bars_zero_valid(self) -> None:
        """purge_bars=0인 모듈도 등록 가능 (단순 rolling indicator)."""
        reg = PurgeBarsRegistry()
        reg.register(ModulePurgeBarsMeta("simple_ema", 0, "rolling=50bars, no purge"))
        reg.register(ModulePurgeBarsMeta("feature", 10, "label_horizon=10bars"))

        result = reg.get_boundary_purge_bars()
        assert result == 10, f"max purge_bars=10이어야 함: {result}"
