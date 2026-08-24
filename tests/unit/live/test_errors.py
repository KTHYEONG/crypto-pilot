"""SCENARIO_LIVE_02: 오류 정책 레지스트리는 미등록 코드에 fail-closed다."""

from __future__ import annotations

from collections.abc import Mapping

from src.live.errors import BINANCE_ERROR_POLICY, ErrorAction, resolve_error_action


def test_SCENARIO_LIVE_02_error_registry_fails_closed() -> None:
    assert resolve_error_action(-1021) is ErrorAction.RESYNC_CLOCK
    assert resolve_error_action(-1013) is ErrorAction.FAIL_CLOSED
    assert resolve_error_action(-5022) is ErrorAction.BENIGN_REPRICE
    assert resolve_error_action(-2011) is ErrorAction.BENIGN

    for unregistered in (-9999, 12345, 0):
        assert resolve_error_action(unregistered) is ErrorAction.FAIL_CLOSED
    assert resolve_error_action(None) is ErrorAction.FAIL_CLOSED

    assert isinstance(BINANCE_ERROR_POLICY, Mapping)
    assert all(isinstance(action, ErrorAction) for action in BINANCE_ERROR_POLICY.values())

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_02_ERROR_REGISTRY_FAILS_CLOSED",
)
