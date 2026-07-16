"""Measure per-TF effective_symbol_n distribution at the final outer-fold snapshot
and propose (not apply) recalibrated l1_min_effective_sym_n values.

[ADR_20260716_L1_SLOW_TF_GATE_RECALIBRATION] Measurement and adoption are
deliberately separate: this script only writes a proposal artifact
(logs/futures/diagnostics/l1_symbol_breadth_calibration.json); config.py's
_DEFAULT_PER_TF_GATE_OVERRIDES must be updated by hand after review.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.domain.futures.strategy.run_l1_cross_tf_replay import _CONTROL_TFS, run_once  # noqa: E402
from src.domain.futures.strategy.tiered_workflow.cross_tf_diagnostics import STAGE_ORDER  # noqa: E402
from src.domain.futures.strategy.tiered_workflow.signal_selection import _compute_effective_sym_n  # noqa: E402

_FLOOR = 1.0
_CEIL = 3.0


def measure_effective_sym_n_by_tf() -> dict[str, float]:
    """Run control replay once; capture _compute_effective_sym_n(fold_reports) per TF
    via a pipeline.evaluate_layer1_readiness monkeypatch."""
    from src.domain.futures.strategy.tiered_workflow import pipeline as _pipeline_mod

    _pipeline = cast(Any, _pipeline_mod)
    captured: dict[str, list[float]] = defaultdict(list)

    _orig_evaluate = _pipeline.evaluate_layer1_readiness
    _orig_per_tf = _pipeline.run_per_tf_l1

    _current_tf: str = ""

    def _evaluate_wrapper(**kwargs: Any) -> Any:
        result = _orig_evaluate(**kwargs)
        if _current_tf:
            fold_reports = kwargs.get("fold_reports", ())
            if fold_reports:
                eff_n = float(_compute_effective_sym_n(tuple(fold_reports)))
                captured[_current_tf].append(eff_n)
        return result

    def _per_tf_wrapper(**kwargs: Any) -> Any:
        nonlocal _current_tf
        _current_tf = str(kwargs.get("tf", ""))
        return _orig_per_tf(**kwargs)

    _pipeline.evaluate_layer1_readiness = _evaluate_wrapper
    _pipeline.run_per_tf_l1 = _per_tf_wrapper
    try:
        trace: dict[str, dict[str, dict[str, object]]] = {stage: {} for stage in STAGE_ORDER}
        run_once(label="control", tfs=_CONTROL_TFS, ablate_1h_fusion=False, trace=trace)
    finally:
        _pipeline.evaluate_layer1_readiness = _orig_evaluate
        _pipeline.run_per_tf_l1 = _orig_per_tf

    return {tf: float(np.percentile(np.asarray(v, dtype=np.float64), 10)) for tf, v in captured.items() if v}


def propose_thresholds(measured: dict[str, float]) -> dict[str, float]:
    proposals: dict[str, float] = {}
    for tf, value in measured.items():
        if value is None or not np.isfinite(value):
            continue
        proposals[tf] = float(np.clip(value, _FLOOR, _CEIL))
    return proposals


def main() -> int:
    measured = measure_effective_sym_n_by_tf()
    proposals = propose_thresholds(measured)
    artifact = {
        "measured_p10_effective_sym_n_by_tf": {tf: float(v) for tf, v in measured.items()},
        "proposed_l1_min_effective_sym_n": proposals,
        "floor": _FLOOR,
        "ceiling": _CEIL,
        "note": "NOT auto-applied -- human review required before updating config.py _DEFAULT_PER_TF_GATE_OVERRIDES",
    }
    out_path = Path("logs/futures/diagnostics/l1_symbol_breadth_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
