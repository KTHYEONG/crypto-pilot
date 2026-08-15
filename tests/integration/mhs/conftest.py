"""MHS 파이프라인 통합 테스트 전용 fixture."""

from __future__ import annotations

from types import SimpleNamespace

import psutil
import pytest

# fork-admission 게이트(assert_fork_admission/plan_worker_count)는 실측
# psutil.virtual_memory()를 참조한다. 기본값을 넉넉하게 고정해 xdist 동시
# 워커의 메모리 경합에 따라 게이트가 우연히 발동하는 것을 막는다(테스트
# 로직이 아니라 동시 실행 중인 다른 워커의 부하에 결과가 좌우되는 플레이키를
# 방지). RAM 가드 자체를 검증하는 테스트는 자신의 monkeypatch로 이 기본값을
# 이후에 덮어써 정상적으로 오버라이드한다.
_AMPLE_MEMORY = SimpleNamespace(total=64 * 2**30, available=60 * 2**30)


@pytest.fixture(autouse=True)
def _mhs_ample_virtual_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _AMPLE_MEMORY)
