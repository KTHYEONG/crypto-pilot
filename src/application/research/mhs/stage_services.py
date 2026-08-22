"""Domain-service seam consumed exclusively by the pipeline stages.

Blueprint P4 target architecture: pipeline stages depend on this explicit seam
instead of reaching into the evaluation composition monolith for private
helpers directly (by ``from evaluation import _foo`` or by attribute access on
an evaluation-aliased module object). Every name below is the SAME function
object as its ``evaluation`` counterpart (an identity re-export, verified by
``tests/unit/application/research/mhs/test_stage_services.py``), so existing
``monkeypatch.setattr(evaluation, "_foo", ...)`` call sites keep working
whenever a stage's own import happens to bind before the patch; test call
sites that need the injection to hold regardless of import order patch both
``evaluation`` and this module explicitly (see
``tests/unit/application/research/mhs/test_realistic_execution_primary_swap.py``).

Stages import names from here, never from ``evaluation`` directly and never
via attribute access on an evaluation-aliased module object; enforced by
``tests/contract/test_module_boundaries.py``.
"""

from __future__ import annotations

from src.application.research.mhs.evaluation import (
    _active_blend_book_and_grid,
    _apply_trend_sleeve,
    _book_weights,
    _candidate_weight_books,
    _committee_diagnostic,
    _committee_evidence_weights_by_boundary,
    _committee_execution_book,
    _committee_member_attribution,
    _committee_member_books,
    _fold_blend_parity,
    _fold_growth_concentration,
    _guard_stage_or_breach,
    _horizon_ensemble_execution_weights,
    _load_feature_panels,
    _multi_feature_diagnostic,
    _phase_diagnostics,
    _prefer_funding_carry_selection,
    _run_books_concurrent,
    _run_fold_safe_discovery_parallel,
    _run_post_book_concurrently,
    _signal_ema_span,
    _trend_sleeve_diagnostic,
    _trend_sleeve_position,
)

__all__ = [
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
]
