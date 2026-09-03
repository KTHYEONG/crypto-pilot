"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.diagnostics as diagnostics


def test_diagnostics_module_present() -> None:
    assert diagnostics.__name__ == "src.mhs.evaluation.diagnostics"
    assert callable(diagnostics._phase_diagnostics)
