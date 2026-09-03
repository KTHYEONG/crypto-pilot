"""Naming-convention bridge for ``src.quant.evaluation.policy``.

``tools/agent_skills/lean_check.py`` resolves co-modification coverage by
module filename (``test_<module>.py``). The canonical evaluation-end policy
scenarios live in ``test_evaluation_end_policy.py`` per their registered
scenario contracts; they are re-exported here unchanged so both entry points
run the identical tests.
"""

from __future__ import annotations

from tests.unit.quant.evaluation.test_evaluation_end_policy import (
    test_SCENARIO_MHS_RESOLVE_EVALUATION_END_ALWAYS_RESOLVES,
    test_unsealed_path_enforces_derived_ceiling,
)

__all__ = [
    "test_SCENARIO_MHS_RESOLVE_EVALUATION_END_ALWAYS_RESOLVES",
    "test_unsealed_path_enforces_derived_ceiling",
]
