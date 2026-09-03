"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.guards as guards


def test_guards_module_present() -> None:
    assert guards.__name__ == "src.mhs.evaluation.guards"
    assert callable(guards._guard_stage_or_breach)
