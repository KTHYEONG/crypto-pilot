"""src.mhs.report.schema: MhsHorizonDiagnosticReport field contract.

Golden byte-exactness of the full assembled payload is covered by
tests/integration/mhs/test_golden_identity.py; this module targets the
schema's own defaults and its to_payload dispatch, which no other test names
directly.
"""

from __future__ import annotations

from src.mhs.report.schema import RENAME_REGISTRY, MhsHorizonDiagnosticReport


def _minimal_report(**overrides: object) -> MhsHorizonDiagnosticReport:
    defaults: dict[str, object] = {
        "feature": "mhs_phase1",
        "status": "COMPLETE",
        "start": "2021-01-01",
        "end": "2021-01-02",
        "resolved_end": "2021-01-02",
        "partition": "dev",
        "execution_tiers_bps": (),
        "books": {},
        "blend": None,
        "blend_target_gross": 0.0,
        "blend_cash_fraction": 0.0,
        "eligible_symbols": 0,
        "trials_attempted": 0,
        "deflated_sharpe_ratio": None,
        "xs_rank_ic": {},
        "date_clustered_regression": {},
        "horizon_diagnostics": {},
        "bootstrap_ci": None,
        "placebo_sharpe_percentile": None,
        "deployment_readiness": None,
        "synthetic_stress": {},
        "participation_warnings": {},
        "termination_counts": {},
        "unsupported_assumptions": (),
        "anchored_folds": (),
        "folds": (),
        "research_go": None,
        "fill_source": "NOT_RUN_NO_EXECUTION_DATA",
        "mark_source": "NOT_RUN_NO_EXECUTION_DATA",
        "execution_timeframe": "3m",
        "execution_universe_size": 0,
        "execution_symbols": (),
        "run_elapsed_seconds": 0.0,
    }
    defaults.update(overrides)
    return MhsHorizonDiagnosticReport(**defaults)  # type: ignore[arg-type]


def test_worker_plan_and_tree_memory_default_empty() -> None:
    """New measurement-correctness fields (P0) default to empty/None so every
    pre-existing construction site stays valid."""
    report = _minimal_report()
    assert report.worker_plan == {}
    assert report.tree_memory is None


def test_to_payload_dispatches_through_jsonable() -> None:
    report = _minimal_report(worker_plan={"books": 3})
    payload = report.to_payload()
    assert payload["worker_plan"] == {"books": 3}
    assert payload["tree_memory"] is None
    assert payload["status"] == "COMPLETE"


def test_rename_registry_only_covers_stated_migrations() -> None:
    """Frozen key set: an accidental addition here silently changes which
    old keys the golden comparison remaps."""
    assert RENAME_REGISTRY == {
        "MHS_COMMITTEE_TARGET_GROSS": "COMMITTEE_TARGET_GROSS",
        "MHS_COMMITTEE_TARGET_VOL": "COMMITTEE_TARGET_VOL",
        "PHASE_1_BOOK_SPECS": "BOOK_SPECS",
        "PHASE_1_BOOK_BLEND_WEIGHTS": "BOOK_BLEND_WEIGHTS",
    }
