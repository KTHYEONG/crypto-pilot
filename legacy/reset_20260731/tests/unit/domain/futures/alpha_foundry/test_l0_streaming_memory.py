from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.futures.alpha_foundry.bridge_helpers import (
    L0StreamingContractError,
    L0StreamingResult,
    TfPanelBundle,
    run_alpha_foundry_l0_gate_streaming,
)


def test_streaming_releases_each_bundle_and_checks_fingerprint() -> None:
    calls: list[str] = []
    generations: dict[str, int] = {}

    def factory(tf: str) -> TfPanelBundle:
        generation = generations.get(tf, 0)
        generations[tf] = generation + 1
        calls.append(f"{generation}:{tf}")
        return TfPanelBundle(
            timeframe=tf,
            panels=(MagicMock(),),
            bindings=(MagicMock(panel_index=0, recipe_id=f"{tf}-r1"),),
            recipes={f"{tf}-r1": MagicMock()},
            aligned=MagicMock(),
            fingerprint=f"stable-{tf}",
        )

    config = MagicMock()
    config.cheap_gate = MagicMock()
    with (
        patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=(),
        ),
        patch(
            "src.domain.futures.alpha_foundry.bridge_helpers.run_alpha_foundry_l0_gate",
            return_value=MagicMock(panels_for_l1=()),
        ),
    ):
        result = run_alpha_foundry_l0_gate_streaming(
            timeframes=("4h", "8h"),
            build_bundle=factory,  # type: ignore[arg-type]
            cost_model=MagicMock(),
            runtime_config=config,
            run_id_prefix="test",
            total_l1_verification_budget=2,
        )

    assert calls == ["0:4h", "0:8h", "1:4h", "1:8h"]
    assert set(result.results_by_tf) == {"4h", "8h"}


def test_streaming_rejects_non_deterministic_rebuild() -> None:
    generations: dict[str, int] = {}

    def factory(tf: str) -> TfPanelBundle:
        generation = generations.get(tf, 0)
        generations[tf] = generation + 1
        return TfPanelBundle(
            timeframe=tf,
            panels=(),
            bindings=(),
            recipes={},
            aligned=MagicMock(),
            fingerprint=f"{tf}-{generation}",
        )

    with pytest.raises(L0StreamingContractError, match="fingerprint"):
        run_alpha_foundry_l0_gate_streaming(
            timeframes=("4h",),
            build_bundle=factory,  # type: ignore[arg-type]
            cost_model=MagicMock(),
            runtime_config=MagicMock(),
            run_id_prefix="test",
            total_l1_verification_budget=1,
        )


def test_streaming_budget_limits_selected_panels() -> None:
    calls: list[str] = []

    def factory(tf: str) -> TfPanelBundle:
        calls.append(tf)
        return TfPanelBundle(
            timeframe=tf,
            panels=(MagicMock(), MagicMock()),
            bindings=(MagicMock(panel_index=0, recipe_id=f"{tf}-r1"), MagicMock(panel_index=1, recipe_id=f"{tf}-r2")),
            recipes={f"{tf}-r1": MagicMock(), f"{tf}-r2": MagicMock()},
            aligned=MagicMock(),
            fingerprint=f"stable-{tf}",
        )

    config = MagicMock()
    config.cheap_gate = MagicMock()

    mock_panels_4h = (MagicMock(recipe_id="4h-r1"), MagicMock(recipe_id="4h-r2"))
    mock_panels_8h = (MagicMock(recipe_id="8h-r1"), MagicMock(recipe_id="8h-r2"))

    with (
        patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=(),
        ),
        patch(
            "src.domain.futures.alpha_foundry.bridge_helpers.run_alpha_foundry_l0_gate",
            side_effect=[
                MagicMock(panels_for_l1=mock_panels_4h),
                MagicMock(panels_for_l1=mock_panels_8h),
            ],
        ),
    ):
        result = run_alpha_foundry_l0_gate_streaming(
            timeframes=("4h", "8h"),
            build_bundle=factory,  # type: ignore[arg-type]
            cost_model=MagicMock(),
            runtime_config=config,
            run_id_prefix="test",
            total_l1_verification_budget=3,
        )

    assert len(result.final_selected_recipe_ids) <= 3


def test_streaming_empty_tf_typed_empty_result() -> None:
    calls: list[str] = []
    generations: dict[str, int] = {}

    def factory(tf: str) -> TfPanelBundle:
        generation = generations.get(tf, 0)
        generations[tf] = generation + 1
        calls.append(f"{generation}:{tf}")
        return TfPanelBundle(
            timeframe=tf,
            panels=(),
            bindings=(),
            recipes={},
            aligned=MagicMock(),
            fingerprint=f"empty-{tf}",
        )

    config = MagicMock()
    config.cheap_gate = MagicMock()

    with (
        patch(
            "src.domain.futures.alpha_foundry.cheap_gate.evaluate_alpha_cheap_gate_batch",
            return_value=(),
        ),
        patch(
            "src.domain.futures.alpha_foundry.bridge_helpers.run_alpha_foundry_l0_gate",
            return_value=MagicMock(panels_for_l1=()),
        ),
    ):
        result = run_alpha_foundry_l0_gate_streaming(
            timeframes=("4h", "8h"),
            build_bundle=factory,  # type: ignore[arg-type]
            cost_model=MagicMock(),
            runtime_config=config,
            run_id_prefix="test",
            total_l1_verification_budget=2,
        )

    assert calls == ["0:4h", "0:8h", "1:4h", "1:8h"]
    assert set(result.results_by_tf) == {"4h", "8h"}
    assert result.selected_panels_by_tf["4h"] == ()


def test_l0_streaming_result_dataclass() -> None:
    result = L0StreamingResult(
        results_by_tf={},
        selected_panels_by_tf={},
        alignment_views_by_tf={},
        final_selected_recipe_ids=(),
    )
    assert result.final_selected_recipe_ids == ()
