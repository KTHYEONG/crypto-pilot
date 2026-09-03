"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.specs as specs


def test_specs_module_present() -> None:
    assert specs.__name__ == "src.mhs.evaluation.specs"
    assert callable(specs._resolved_base_execution_spec)
