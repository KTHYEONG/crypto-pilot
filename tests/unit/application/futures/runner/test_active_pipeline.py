from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd

from src.application.futures.runner.active_pipeline import (
    _has_l1_delivery_candidates,
    _run_strategy_stage,
)


def test_strategy_stage_wires_causal_cutoff_and_delivery_manifest() -> None:
    """The runner must forward both sides of the L0→L1 delivery contract."""
    source = inspect.getsource(_run_strategy_stage)

    assert "l0_evidence_end=getattr(tiered_window, \"l1_start\", None)" in source
    assert "l0_delivery_manifest=ml_out.l0_delivery_manifest" in source


def test_has_l1_delivery_candidates_uses_multi_tf_manifest_not_base_report() -> None:
    """HTF L0 candidates must not be discarded when the base-TF report is empty."""
    output = SimpleNamespace(
        l0_delivery_manifest=SimpleNamespace(final_selected_recipe_ids=("recipe:12h",)),
    )

    assert _has_l1_delivery_candidates(output)
    assert not _has_l1_delivery_candidates(SimpleNamespace(l0_delivery_manifest=None))


def test_tiered_labeled_events_marks_unrouted_events_with_empty_l0_recipe_id() -> None:
    """Unrouted events must be filtered by the L0 manifest, not crash L1."""
    from src.application.futures.runner.active_pipeline import _tiered_labeled_events

    source = SimpleNamespace(labeled_unfiltered=pd.DataFrame({"native_tf": ["4h"]}))

    labeled = _tiered_labeled_events(source)

    assert labeled["l0_recipe_id"].tolist() == [""]
