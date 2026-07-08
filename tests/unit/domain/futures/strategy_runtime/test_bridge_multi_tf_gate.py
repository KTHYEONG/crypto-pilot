from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy_runtime.bridge import (
    build_multi_tf_panels,
    build_native_htf_panels,
)


def _mock_build_native_panels_ltf(
    monkeypatch: pytest.MonkeyPatch,
    tfs: tuple[str, ...],
) -> None:
    """Helper: stub bridge internals so build_native_htf_panels returns one entry per TF."""

    def fake_virtual_maps(
        data_maps: dict[str, Any], symbols: list[str], target_tf: str
    ) -> dict[str, Any]:
        return {s: {target_tf: MagicMock()} for s in symbols[:1]}

    def fake_align(
        data_maps: dict[str, Any], symbols: list[str], tf: str
    ) -> MagicMock:
        aligned = MagicMock()
        aligned.datetimes = MagicMock()
        return aligned

    def fake_build_panels(
        aligned: Any, cfg: Any, family_filter: tuple[str, ...] | None = None, **kwargs: Any
    ) -> tuple[MagicMock, ...]:
        return (MagicMock(),)

    monkeypatch.setattr(
        "src.domain.futures.strategy_runtime.bridge._build_virtual_probe_tf_maps",
        fake_virtual_maps,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.common.alignment.align_data_maps",
        fake_align,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
        fake_build_panels,
    )


def _base_cfg() -> CandidateStrategyConfig:
    """Return a minimal CandidateStrategyConfig for bridge tests."""
    return CandidateStrategyConfig()


def test_build_native_htf_panels_includes_ltf_when_htf_only_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: htf_only=False allows 1h (LTF) into eligible TFs."""
    _mock_build_native_panels_ltf(monkeypatch, ("4h", "1h"))

    result = build_native_htf_panels(
        data_maps={},
        symbols=["BTCUSDT"],
        aligned_base=MagicMock(),
        base_cfg=_base_cfg(),
        base_tf="4h",
        tfs=("4h", "1h"),
        family_pool=lambda tf: ("trend_ma",),
        htf_only=False,
    )

    assert "1h" in result


def test_build_native_htf_panels_htf_only_true_still_excludes_ltf_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: htf_only=True still excludes LTF even when explicitly in tfs (backward compat)."""
    _mock_build_native_panels_ltf(monkeypatch, ("4h", "1h"))

    result = build_native_htf_panels(
        data_maps={},
        symbols=["BTCUSDT"],
        aligned_base=MagicMock(),
        base_cfg=_base_cfg(),
        base_tf="4h",
        tfs=("4h", "1h"),
        family_pool=lambda tf: ("trend_ma",),
        htf_only=True,
    )

    assert "1h" not in result


def test_build_native_htf_panels_hpb_i_gte_hpb_base_unaffected_by_htf_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2 regression: HTF (6h/8h/12h) keys identical regardless of htf_only flag."""
    _mock_build_native_panels_ltf(monkeypatch, ("6h", "8h", "12h"))

    common_kwargs = {
        "data_maps": {},
        "symbols": ["BTCUSDT"],
        "aligned_base": MagicMock(),
        "base_cfg": _base_cfg(),
        "base_tf": "4h",
        "tfs": ("6h", "8h", "12h"),
        "family_pool": lambda tf: ("trend_ma",),
    }
    result_true = build_native_htf_panels(**common_kwargs, htf_only=True)
    result_false = build_native_htf_panels(**common_kwargs, htf_only=False)

    assert set(result_true.keys()) == set(result_false.keys())


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
