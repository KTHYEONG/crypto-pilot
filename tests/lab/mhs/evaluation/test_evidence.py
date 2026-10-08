"""Invariant tests for book-structure trace numerics in blend parity."""

from __future__ import annotations

from src.lab.mhs.contracts import MhsFoldReport
from src.lab.mhs.evaluation.evidence import _fold_blend_parity, _trace_number


def _report(fold_index: int, book_structure: dict | None) -> MhsFoldReport:
    return MhsFoldReport(
        fold_index=fold_index,
        validation_start="2022-01-08",
        validation_end="2022-12-31",
        strict=None,
        stress=None,
        primary_valid=False,
        primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0,
        primary_net_ann=0.0,
        primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0,
        stress_naive_sharpe=0.0,
        decision_intents=0,
        termination_counts={},
        failures=(),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
        book_structure=book_structure,
    )


def test_trace_number_ignores_provenance_strings() -> None:
    trace = {"gross_mean": 0.5, "sizing_reference_start": "2021-02-08T00:00:00+00:00"}
    assert _trace_number(trace, "gross_mean") == 0.5
    assert _trace_number(trace, "sizing_reference_start") is None
    assert _trace_number(trace, "missing_key") is None


def test_blend_parity_unchanged_with_mixed_traces() -> None:
    fold_float = {"holdings_mean": 42.0, "gross_mean": 0.84, "exposure_scale_mean": 0.63}
    blend_float = {"holdings_mean": 40.0, "gross_mean": 0.80, "exposure_scale_mean": 1.0}
    payload_float, _ = _fold_blend_parity({0: dict(blend_float)}, (_report(0, dict(fold_float)),))
    fold_mixed = dict(fold_float, sizing_reference_start="2021-02-08T00:00:00+00:00", sizing_reference_end="2021-05-08T00:00:00+00:00")
    payload_mixed, _ = _fold_blend_parity({0: dict(blend_float)}, (_report(0, fold_mixed),))
    assert payload_mixed["folds"][0]["holdings_log_ratio"] == payload_float["folds"][0]["holdings_log_ratio"]
    assert payload_mixed["folds"][0]["gross_log_ratio"] == payload_float["folds"][0]["gross_log_ratio"]
    assert payload_mixed["folds"][0]["deployed_gross_log_ratio"] == payload_float["folds"][0]["deployed_gross_log_ratio"]
    assert 0 not in payload_mixed["unmeasured"]


def test_string_valued_holdings_is_unmeasured() -> None:
    from src.lab.mhs import research_go as research_go_mod

    fold_trace = {"holdings_mean": "n/a", "gross_mean": 0.84, "exposure_scale_mean": 1.0}
    blend_trace = {"holdings_mean": 42.0, "gross_mean": 0.80, "exposure_scale_mean": 1.0}
    payload, reasons = _fold_blend_parity({0: blend_trace}, (_report(0, fold_trace),))
    assert 0 in payload["unmeasured"]
    assert payload["folds"][0]["holdings_log_ratio"] is None
    assert research_go_mod.GO_REASON_PATH_DIVERGENCE not in reasons
