"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.windows as windows


def test_windows_module_present() -> None:
    assert windows.__name__ == "src.mhs.evaluation.windows"
    assert callable(windows._resolve_ns_vectorized)
