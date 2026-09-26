"""Stage names shared by the live daemon and the CI deploy gate.

Kept stdlib-only and dependency-free: the deploy gate runs on a bare CI runner without the project
environment, and the daemon must agree with it on which stages make a container replacement unsafe.
"""

from __future__ import annotations

# 이 단계 중에는 컨테이너를 교체하면 신호 지연/집행 절단이 생기므로 배포를 유예한다.
BUSY_STAGES: frozenset[str] = frozenset({"refresh", "signal", "execute"})
