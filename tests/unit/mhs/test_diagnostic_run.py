"""P4 path-presence pin for the MHS composition root.

Behavioral coverage lives in the moved evaluation suite
(``tests/unit/mhs/test_evaluation_appresearch.py``) and
``tests/integration/mhs/test_mhs_horizon_diagnostic.py``.
"""

from __future__ import annotations

from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic


def test_diagnostic_run_module_present() -> None:
    assert callable(run_mhs_horizon_diagnostic)
