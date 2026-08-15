"""Measure per-TF pooled effective_symbol_n and fold-level raw ready_symbols
distributions, and propose (not apply) recalibrated l1_min_effective_sym_n /
l1_min_cross_section values.

[ADR_20260716_L1_SLOW_TF_GATE_RECALIBRATION] Measurement and adoption are
deliberately separate: this script only writes a proposal artifact
(logs/futures/diagnostics/l1_symbol_breadth_calibration.json); config.py's
_DEFAULT_PER_TF_GATE_OVERRIDES must be updated by hand after review.

[ADR pending: L1_REGISTRY_ADMISSION_RECALIBRATION Phase A] measure_fold_min_ready_symbols_by_tf
extends this file to the fold-level raw symbol-count gate (l1_min_cross_section),
which the pooled effective_sym_n fix above does not reach.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable
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
_CROSS_SECTION_FLOOR = 1.0
_CROSS_SECTION_CEIL = 2.0
_REGISTRY_EMPTY_BLOCKER = "empty_opportunities:registry_empty"


def _run_control_replay_with_evidence_capture(
    on_fold_reports: Callable[[str, tuple[Any, ...]], None],
) -> None:
    """Run control replay once, invoking on_fold_reports(tf, fold_reports) each time
    pipeline.evaluate_layer1_readiness is called for the currently-processing TF.

    Shared harness for both measure_effective_sym_n_by_tf and
    measure_fold_min_ready_symbols_by_tf -- both need the same
    run_per_tf_l1/evaluate_layer1_readiness monkeypatch wiring, differing only
    in what they extract from fold_reports.
    """
    from src.domain.futures.strategy.tiered_workflow import pipeline as _pipeline_mod

    _pipeline = cast(Any, _pipeline_mod)
    _orig_evaluate = _pipeline.evaluate_layer1_readiness
    _orig_per_tf = _pipeline.run_per_tf_l1

    _current_tf: str = ""

    def _evaluate_wrapper(**kwargs: Any) -> Any:
        result = _orig_evaluate(**kwargs)
        if _current_tf:
            fold_reports = kwargs.get("fold_reports", ())
            if fold_reports:
                on_fold_reports(_current_tf, tuple(fold_reports))
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


def measure_effective_sym_n_by_tf() -> dict[str, float]:
    """Run control replay once; capture _compute_effective_sym_n(fold_reports) per TF."""
    captured: dict[str, list[float]] = defaultdict(list)

    def _capture(tf: str, fold_reports: tuple[Any, ...]) -> None:
        eff_n = float(_compute_effective_sym_n(fold_reports))
        captured[tf].append(eff_n)

    _run_control_replay_with_evidence_capture(_capture)
    return {tf: float(np.percentile(np.asarray(v, dtype=np.float64), 10)) for tf, v in captured.items() if v}


def measure_fold_min_ready_symbols_by_tf() -> dict[str, float]:
    """Run control replay once; capture p10 of raw len(ready_symbols) across
    non-registry_empty Layer1FoldReadiness objects per TF.

    Folds carrying the "empty_opportunities:registry_empty" blocker are excluded
    -- a fold with zero admitted candidates says nothing about the raw
    symbol-count floor and would bias the p10 toward zero for no reason.
    """
    captured: dict[str, list[float]] = defaultdict(list)

    def _capture(tf: str, fold_reports: tuple[Any, ...]) -> None:
        for report in fold_reports:
            if _REGISTRY_EMPTY_BLOCKER in report.blockers:
                continue
            captured[tf].append(float(len(report.ready_symbols)))

    _run_control_replay_with_evidence_capture(_capture)
    return {tf: float(np.percentile(np.asarray(v, dtype=np.float64), 10)) for tf, v in captured.items() if v}


def propose_thresholds(measured: dict[str, float]) -> dict[str, float]:
    proposals: dict[str, float] = {}
    for tf, value in measured.items():
        if value is None or not np.isfinite(value):
            continue
        proposals[tf] = float(np.clip(value, _FLOOR, _CEIL))
    return proposals


def propose_cross_section_thresholds(measured: dict[str, float]) -> dict[str, float]:
    """Clip each measured value to [_CROSS_SECTION_FLOOR, _CROSS_SECTION_CEIL].

    floor=1: a non-registry_empty fold has >=1 ready symbol by construction.
    ceiling=2: today's flat l1_min_cross_section default -- calibration may
    only relax slower TFs, never tighten them.
    """
    proposals: dict[str, float] = {}
    for tf, value in measured.items():
        if value is None or not np.isfinite(value):
            continue
        proposals[tf] = float(np.clip(value, _CROSS_SECTION_FLOOR, _CROSS_SECTION_CEIL))
    return proposals


def main() -> int:
    measured = measure_effective_sym_n_by_tf()
    proposals = propose_thresholds(measured)
    cross_section_measured = measure_fold_min_ready_symbols_by_tf()
    cross_section_proposals = propose_cross_section_thresholds(cross_section_measured)
    artifact = {
        "measured_p10_effective_sym_n_by_tf": {tf: float(v) for tf, v in measured.items()},
        "proposed_l1_min_effective_sym_n": proposals,
        "floor": _FLOOR,
        "ceiling": _CEIL,
        "measured_p10_fold_ready_symbols_by_tf": {tf: float(v) for tf, v in cross_section_measured.items()},
        "proposed_l1_min_cross_section": cross_section_proposals,
        "cross_section_floor": _CROSS_SECTION_FLOOR,
        "cross_section_ceiling": _CROSS_SECTION_CEIL,
        "note": "NOT auto-applied -- human review required before updating config.py _DEFAULT_PER_TF_GATE_OVERRIDES",
    }
    out_path = Path("logs/futures/diagnostics/l1_symbol_breadth_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
