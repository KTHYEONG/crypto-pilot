"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.specs as specs


def test_specs_module_present() -> None:
    assert specs.__name__ == "src.mhs.evaluation.specs"
    assert callable(specs._resolved_base_execution_spec)

def test_resolved_base_execution_spec_threads_name_drift_trim() -> None:
    import dataclasses

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.evaluation.specs import _resolved_base_execution_spec, _stress_cost_execution_spec
    from src.mhs.params import NAME_DRIFT_TRIM_INTERVAL_HOURS, NAME_DRIFT_TRIM_MAX_WEIGHT

    default_spec = _resolved_base_execution_spec(MhsDiagnosticRequest())
    assert default_spec.name_drift_trim_max_weight is None

    request = dataclasses.replace(MhsDiagnosticRequest(), name_drift_trim=True)
    base = _resolved_base_execution_spec(request)
    assert base.name_drift_trim_max_weight == NAME_DRIFT_TRIM_MAX_WEIGHT
    assert base.name_drift_trim_interval_hours == NAME_DRIFT_TRIM_INTERVAL_HOURS

    stress = _stress_cost_execution_spec(base)
    assert stress.name_drift_trim_max_weight == NAME_DRIFT_TRIM_MAX_WEIGHT
    assert stress.name_drift_trim_interval_hours == NAME_DRIFT_TRIM_INTERVAL_HOURS
    assert stress.taker_fee_bps == base.taker_fee_bps * 3.0

