from tests.fixtures.mhs_requests import research_baseline
# ruff: noqa
def test_sizing_policy_rejects_invalid_kelly_lcb_z() -> None:
    import pytest
    from src.lab.mhs.deployment_policy import SizingPolicy

    with pytest.raises(ValueError, match="kelly_lcb_z"):
        SizingPolicy(mode="growth_budget", target_annual_vol=0.35, exposure_cap=3.0, scale_floor=0.2, kelly_enabled=True, kelly_window_days=42, kelly_fraction=0.5, kelly_lcb_z=-0.1, kelly_blend_weight=0.5, drawdown_brake=False)



def test_sizing_policy_validates_all_bounds() -> None:
    import pytest
    from src.lab.mhs.deployment_policy import SizingPolicy

    good = dict(mode="growth_budget", target_annual_vol=0.35, exposure_cap=3.0, scale_floor=0.2, kelly_enabled=True, kelly_window_days=42, kelly_fraction=0.5, kelly_lcb_z=0.0, kelly_blend_weight=0.5, drawdown_brake=False)
    for key, bad in [("target_annual_vol", 0.0), ("exposure_cap", 0.5), ("scale_floor", 0.0), ("kelly_window_days", 0), ("kelly_fraction", 0.75), ("kelly_blend_weight", 1.5)]:
        with pytest.raises(ValueError, match=key):
            SizingPolicy(**{**good, key: bad})


def test_deleted_policy_types_stay_deleted() -> None:
    import src.lab.mhs.deployment_policy as deployment_policy

    for dead in ("build_deployment_policy", "TargetWeightPolicy", "SignalWindowPolicy", "MhsDeploymentPolicy"):
        assert not hasattr(deployment_policy, dead), dead
    assert hasattr(deployment_policy, "SizingPolicy")
    assert hasattr(deployment_policy, "live_parity_blockers")
    assert hasattr(deployment_policy, "LIVE_UNSUPPORTED_REQUEST_FLAGS")


def test_live_parity_blockers_is_single_registry_seam() -> None:
    import dataclasses
    from dataclasses import fields

    from src.lab.mhs.contracts import MhsDiagnosticRequest
    from src.lab.mhs.deployment_policy import (
        LIVE_UNSUPPORTED_REQUEST_FLAGS,
        live_parity_blockers,
    )

    # Given: 등록부의 모든 키는 실제 요청 필드여야 한다(오탈자 fail-closed)
    request_fields = {f.name for f in fields(MhsDiagnosticRequest)}
    assert set(LIVE_UNSUPPORTED_REQUEST_FLAGS) <= request_fields
    assert "name_drift_trim" in LIVE_UNSUPPORTED_REQUEST_FLAGS

    # When / Then: 기본값은 차단 없음
    assert live_parity_blockers(MhsDiagnosticRequest()) == ()
    assert live_parity_blockers(None) == ()

    # When / Then: trim ON 차단
    trim_request = MhsDiagnosticRequest(name_drift_trim=True)
    assert live_parity_blockers(trim_request) == ("name_drift_trim",)


def test_request_parser_rejects_retired_parity_flags() -> None:
    """Retired parity flags are rejected rather than ignored."""
    import pytest

    from src.lab.mhs.contracts import MhsDiagnosticRequest

    with pytest.raises(TypeError):
        research_baseline(fill_mark_parity_gate=True)  # type: ignore[call-arg]
