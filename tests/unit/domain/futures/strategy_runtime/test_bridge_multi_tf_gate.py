from __future__ import annotations

from typing import Any

import pytest

from src.domain.futures.strategy_runtime.bridge import (
    build_multi_tf_panels,
)


def test_build_multi_tf_panels_wrapper_calls_sub_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """S1-6: build_multi_tf_panels thin wrapper delegates to sub-functions."""
    native_called: list[bool] = []
    project_called: list[bool] = []

    def fake_native(**kwargs: Any) -> dict[str, Any]:
        native_called.append(True)
        return {}

    def fake_project(**kwargs: Any) -> tuple:
        project_called.append(True)
        return ()

    monkeypatch.setattr("src.domain.futures.strategy_runtime.bridge.build_native_htf_panels", fake_native)
    monkeypatch.setattr("src.domain.futures.strategy_runtime.bridge.project_htf_panels_to_base", fake_project)

    result = build_multi_tf_panels(
        data_maps={},
        symbols=[],
        aligned_base=None,  # type: ignore[arg-type]
        base_cfg=None,  # type: ignore[arg-type]
        base_tf="4h",
        tfs=("6h", "8h"),
        family_pool=lambda x: (),
        htf_only=True,
    )
    assert len(native_called) == 1, "build_native_htf_panels should be called once"
    assert len(project_called) == 1, "project_htf_panels_to_base should be called once"
    assert result == ()
