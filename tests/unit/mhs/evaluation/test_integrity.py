"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.integrity as integrity


def test_integrity_module_present() -> None:
    assert integrity.__name__ == "src.mhs.evaluation.integrity"
    assert callable(integrity._assert_cache_required_ledger_valid)
