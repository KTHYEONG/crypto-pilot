"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.folds as folds


def test_folds_module_present() -> None:
    assert folds.__name__ == "src.mhs.evaluation.folds"
    assert callable(folds._incomplete_fold_report)
