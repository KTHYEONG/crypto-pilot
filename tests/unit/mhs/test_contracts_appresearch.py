"""Request contract pins for src.mhs.contracts."""

from __future__ import annotations

import dataclasses

from src.mhs.contracts import MhsDiagnosticRequest


def test_exposure_drawdown_brake_default_inert_and_declared_once() -> None:
    """기본값 False(비트 동일 보장)이며 CLI 플래그 메타데이터와 정확히 대응한다."""
    field = next(
        f for f in dataclasses.fields(MhsDiagnosticRequest)
        if f.name == "exposure_drawdown_brake"
    )
    assert field.default is False
    assert field.metadata["flag"] == "--exposure-drawdown-brake"
    assert field.metadata.get("negate_flag") is None

    request = MhsDiagnosticRequest()
    assert request.exposure_drawdown_brake is False
