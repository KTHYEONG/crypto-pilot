from __future__ import annotations

import inspect

from src.application.futures.runner.active_pipeline import _run_strategy_stage


def test_active_pipeline_contains_explicit_causal_delivery_wiring() -> None:
    source = inspect.getsource(_run_strategy_stage)

    assert 'l0_evidence_end=getattr(tiered_window, "l1_start", None)' in source
    assert "l0_delivery_manifest=ml_out.l0_delivery_manifest" in source
