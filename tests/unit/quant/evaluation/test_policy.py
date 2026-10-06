"""Naming-convention bridge for ``src.quant.evaluation.policy``.

Co-modification coverage is resolved by module filename
(``test_<module>.py``). The canonical evaluation-end policy scenarios live
in ``test_evaluation_end_policy.py`` per their registered scenario
contracts; they are re-exported here unchanged so both entry points run the
identical tests.
"""

from __future__ import annotations

import pandas as pd

from src.quant.evaluation.policy import DISCOVERY_END, HOLDOUT_CUTOFF
from tests.unit.quant.evaluation.test_evaluation_end_policy import (
    test_SCENARIO_MHS_RESOLVE_EVALUATION_END_ALWAYS_RESOLVES,
    test_unsealed_path_enforces_derived_ceiling,
)

__all__ = [
    "test_SCENARIO_MHS_RESOLVE_EVALUATION_END_ALWAYS_RESOLVES",
    "test_unsealed_path_enforces_derived_ceiling",
]


def test_discovery_end_is_frozen_inclusive_boundary() -> None:
    assert pd.Timestamp("2023-12-31 23:59:59", tz="UTC") == DISCOVERY_END
    assert DISCOVERY_END.tz is not None
    assert str(DISCOVERY_END.tz) == "UTC"
    assert DISCOVERY_END < HOLDOUT_CUTOFF


def test_committee_stage_binds_policy_window_objects() -> None:
    import src.mhs.pipeline.stages.committee as committee

    assert committee.DISCOVERY_END is DISCOVERY_END
    assert committee.HOLDOUT_CUTOFF is HOLDOUT_CUTOFF
    assert not hasattr(committee, "QUALIFICATION_END")
