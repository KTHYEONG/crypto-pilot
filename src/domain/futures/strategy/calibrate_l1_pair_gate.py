"""Measure per-TF effective_n distribution at the final outer-fold snapshot
and propose (not apply) recalibrated l1_pair_min_effective_obs values.

[ADR_20260715_L1_PAIR_GATE_TF_DENSITY_CALIBRATION] Measurement and adoption are
deliberately separate: this script only writes a proposal artifact
(logs/futures/diagnostics/l1_pair_gate_calibration.json); config.py's
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

_FLOOR = 2.0
_CEIL = 4.0


def measure_effective_n_by_tf(*, final_snapshot_index: int = 3) -> dict[str, list[float]]:
    """Run control replay once, capturing effective_n at the final evidence snapshot per TF.

    Injects one sink per TF via monkeypatching:
      - patches ``run_per_tf_l1`` to track the current TF being processed
      - patches ``compute_symbol_strategy_evidence`` to wire ``effective_n_sink``
    """
    # NOTE: pipeline.py does `from ... import compute_symbol_strategy_evidence`, which
    # creates its OWN name binding separate from signal_selection's. run_l1_nested_swf's
    # _build_snapshot() resolves the bare name via pipeline.py's module globals, so the
    # patch target must be `pipeline.compute_symbol_strategy_evidence`, NOT
    # `signal_selection.compute_symbol_strategy_evidence` (patching the latter alone is a
    # silent no-op here).
    from src.domain.futures.strategy.tiered_workflow import pipeline as _pipeline_mod

    _pipeline = cast(Any, _pipeline_mod)
    captured: dict[str, list[float]] = defaultdict(list)

    _orig_evidence = _pipeline.compute_symbol_strategy_evidence
    _orig_per_tf = _pipeline.run_per_tf_l1

    _current_tf: str = ""

    def _evidence_wrapper(*args: Any, **kwargs: Any) -> Any:
        if "effective_n_sink" not in kwargs:
            kwargs["effective_n_sink"] = (
                lambda snap_idx, eff_n: captured[_current_tf].append(eff_n)
                if snap_idx == final_snapshot_index
                else None
            )
        return _orig_evidence(*args, **kwargs)

    def _per_tf_wrapper(**kwargs: Any) -> Any:
        nonlocal _current_tf
        _current_tf = str(kwargs.get("tf", ""))
        return _orig_per_tf(**kwargs)

    _pipeline.compute_symbol_strategy_evidence = _evidence_wrapper
    _pipeline.run_per_tf_l1 = _per_tf_wrapper
    try:
        trace: dict[str, dict[str, dict[str, object]]] = {stage: {} for stage in STAGE_ORDER}
        run_once(label="control", tfs=_CONTROL_TFS, ablate_1h_fusion=False, trace=trace)
    finally:
        _pipeline.compute_symbol_strategy_evidence = _orig_evidence
        _pipeline.run_per_tf_l1 = _orig_per_tf

    return dict(captured)


def propose_thresholds(measured: dict[str, list[float]]) -> dict[str, float]:
    proposals: dict[str, float] = {}
    for tf, values in measured.items():
        if not values:
            continue
        p10 = float(np.percentile(np.asarray(values, dtype=np.float64), 10))
        proposals[tf] = float(min(_CEIL, max(_FLOOR, np.floor(p10))))
    return proposals


def main() -> int:
    measured = measure_effective_n_by_tf()
    proposals = propose_thresholds(measured)
    artifact = {
        "measured_effective_n_p10_by_tf": {
            tf: float(np.percentile(np.asarray(v), 10)) for tf, v in measured.items() if v
        },
        "measured_sample_count_by_tf": {tf: len(v) for tf, v in measured.items()},
        "proposed_l1_pair_min_effective_obs": proposals,
        "floor": _FLOOR,
        "ceiling": _CEIL,
        "note": "NOT auto-applied -- human review required before updating config.py _DEFAULT_PER_TF_GATE_OVERRIDES",
    }
    out_path = Path("logs/futures/diagnostics/l1_pair_gate_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
