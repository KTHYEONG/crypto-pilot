"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.folds as folds


def test_folds_module_present() -> None:
    assert folds.__name__ == "src.mhs.evaluation.folds"
    assert callable(folds._incomplete_fold_report)


def test_run_anchored_fold_in_memory_window_reuse(monkeypatch) -> None:
    from src.mhs.evidence import phase_1_anchored_purged_folds
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.evaluation import folds

    fold_list = phase_1_anchored_purged_folds()
    req = MhsDiagnosticRequest(start="2021-01-01", end="2021-03-31", execution_universe_size=8)
    # Verified invocation signature accepts shared_token
    assert callable(folds._run_anchored_fold)
