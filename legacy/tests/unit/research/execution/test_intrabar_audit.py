from __future__ import annotations

import pytest

from src.research.execution.intrabar_audit import intrabar_audit_required


class TestIntrabarAuditRequired:
    # GEV2-12-INTRABAR-TRIGGER
    def test_current_single_exit_design_does_not_require_audit(self) -> None:
        assert intrabar_audit_required(competing_intrabar_exits=1, stop_atr_mult=3.0) is False

    def test_second_competing_exit_requires_audit(self) -> None:
        assert intrabar_audit_required(competing_intrabar_exits=2, stop_atr_mult=3.0) is True

    def test_stop_inside_gap_through_zone_requires_audit(self) -> None:
        assert intrabar_audit_required(competing_intrabar_exits=1, stop_atr_mult=1.0) is True

    def test_boundary_at_gap_free_atr_mult_is_safe(self) -> None:
        assert intrabar_audit_required(
            competing_intrabar_exits=1, stop_atr_mult=1.5, gap_free_atr_mult=1.5,
        ) is False

    def test_zero_exits_never_requires_audit(self) -> None:
        assert intrabar_audit_required(competing_intrabar_exits=0, stop_atr_mult=3.0) is False

    def test_rejects_negative_exit_count(self) -> None:
        with pytest.raises(ValueError, match="competing_intrabar_exits"):
            intrabar_audit_required(competing_intrabar_exits=-1, stop_atr_mult=3.0)

    def test_rejects_non_positive_stop(self) -> None:
        with pytest.raises(ValueError, match="stop_atr_mult"):
            intrabar_audit_required(competing_intrabar_exits=1, stop_atr_mult=0.0)
