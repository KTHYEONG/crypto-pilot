from __future__ import annotations

from unittest.mock import MagicMock

from src.application.futures.runner.compound_pipeline import run_compound_pipeline


def test_compound_pipeline_has_no_order_side_effect_by_default() -> None:
    outcome = run_compound_pipeline(aligned=MagicMock(), universe=MagicMock(), settings=MagicMock(mode="shadow"))
    assert outcome.order_routed is False
