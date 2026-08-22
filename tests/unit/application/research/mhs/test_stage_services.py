"""Contract tests for the stage_services transitional seam (P4).

The seam re-exports the private helpers the pipeline stages consume; these
tests pin the re-export surface so an extraction that drops a symbol fails
loudly here instead of as a deep import error inside a stage module.
"""

from __future__ import annotations

import importlib

import pytest

SEAM = "src.application.research.mhs.stage_services"

EXPECTED: tuple[str, ...] = (
    "_active_blend_book_and_grid",
    "_apply_trend_sleeve",
    "_book_weights",
    "_candidate_weight_books",
    "_committee_diagnostic",
    "_committee_evidence_weights_by_boundary",
    "_committee_execution_book",
    "_committee_member_attribution",
    "_committee_member_books",
    "_fold_blend_parity",
    "_fold_growth_concentration",
    "_guard_stage_or_breach",
    "_horizon_ensemble_execution_weights",
    "_load_feature_panels",
    "_multi_feature_diagnostic",
    "_phase_diagnostics",
    "_prefer_funding_carry_selection",
    "_run_books_concurrent",
    "_run_fold_safe_discovery_parallel",
    "_run_post_book_concurrently",
    "_signal_ema_span",
    "_trend_sleeve_diagnostic",
    "_trend_sleeve_position",
)


def test_seam_reexports_every_expected_symbol() -> None:
    module = importlib.import_module(SEAM)
    missing = [name for name in EXPECTED if not hasattr(module, name)]
    assert missing == []


@pytest.mark.parametrize("name", EXPECTED)
def test_seam_symbol_is_the_evaluation_original(name: str) -> None:
    """Each seam symbol is identical to the evaluation definition (no copy)."""
    import src.application.research.mhs.evaluation as evaluation

    module = importlib.import_module(SEAM)
    assert getattr(module, name) is getattr(evaluation, name)
