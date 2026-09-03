"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.regime as regime


def test_regime_module_present() -> None:
    assert regime.__name__ == "src.mhs.evaluation.regime"
    assert callable(regime._regime_reference_characterization)
