"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/lab/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.lab.mhs.evaluation.participation as participation


def test_participation_module_present() -> None:
    assert participation.__name__ == "src.lab.mhs.evaluation.participation"
    assert callable(participation._load_symbol_quote_volume)
