"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.fold_weights as fold_weights


def test_fold_weights_module_present() -> None:
    assert fold_weights.__name__ == "src.mhs.evaluation.fold_weights"
    assert callable(fold_weights._build_fold_target_weights)
