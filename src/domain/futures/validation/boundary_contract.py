"""Purge bars seam 등록 계약 — boundary contract.

모든 signal/feature 모듈이 purge_bars를 중앙 레지스트리에 등록하며,
백테스트 진입 전 Fail-fast 검증을 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModulePurgeBarsMeta:
    """모듈별 purge_bars 메타데이터.

    Attributes:
        module_name: 모듈 식별자.
        purge_bars: IS/OOS 경계 purge 길이 (바 단위).
        reason: purge_bars 값의 근거 (label_horizon / fit_window 등).
    """

    module_name: str
    purge_bars: int
    reason: str


class PurgeBarsRegistry:
    """모든 signal/feature 모듈이 purge_bars를 등록하는 중앙 레지스트리.

    빈 레지스트리에서 get_boundary_purge_bars() 호출 시 RuntimeError를 발생시켜
    미등록 상태의 백테스트 진입을 Fail-fast로 차단한다.
    """

    def __init__(self) -> None:
        self._registry: dict[str, ModulePurgeBarsMeta] = {}

    def register(self, meta: ModulePurgeBarsMeta) -> None:
        """모듈 purge_bars 등록. 동일 모듈명 재등록 시 최신값 갱신.

        Args:
            meta: ModulePurgeBarsMeta 인스턴스.
        """
        self._registry[meta.module_name] = meta

    def get_boundary_purge_bars(self) -> int:
        """등록된 모든 모듈의 purge_bars 중 최대값 반환.

        Returns:
            max(all registered purge_bars).

        Raises:
            RuntimeError: 등록된 모듈이 없을 경우 Fail-fast.
        """
        if not self._registry:
            raise RuntimeError(
                "No modules registered purge_bars. Fail-fast. "
                "백테스트 진입 전 모든 signal/feature 모듈이 purge_bars를 등록해야 합니다."
            )
        return max(m.purge_bars for m in self._registry.values())

    def validate(self) -> None:
        """미등록 상태면 backtest 진입 거부.

        Raises:
            RuntimeError: 등록된 모듈이 없을 경우.
        """
        _ = self.get_boundary_purge_bars()  # RuntimeError 전파

    def list_modules(self) -> list[ModulePurgeBarsMeta]:
        """등록된 모든 모듈 목록 반환."""
        return list(self._registry.values())
