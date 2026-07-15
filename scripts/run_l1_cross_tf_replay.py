"""Run exactly one isolated L1 cross-timeframe diagnostic replay."""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.application.futures.optimization import strategy_service
from src.application.futures.optimization.config import FuturesRunConfig
from src.application.futures.runner import active_pipeline
from src.application.futures.runner.config import build_l0_runtime_config
from src.domain.futures.alpha_foundry import bridge_helpers
from src.domain.futures.alpha_foundry.multi_tf_fusion import fuse_multi_timeframe_evidence
from src.domain.futures.strategy.tiered_workflow import pipeline as tiered_pipeline
from src.domain.futures.strategy_runtime import bridge

_LOGGER = logging.getLogger(__name__)
_CONTROL_TFS = ("2h", "4h", "6h", "8h", "12h", "1d")
_TREATMENT_TFS = ("1h", *_CONTROL_TFS)
_RUN_SPECS: dict[str, tuple[tuple[str, ...], bool]] = {
    "control": (_CONTROL_TFS, False),
    "control_repeat": (_CONTROL_TFS, False),
    "treatment": (_TREATMENT_TFS, False),
    "fusion_ablation": (_TREATMENT_TFS, True),
}


def _digest(value: object) -> str:
    hasher = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, np.ndarray):
            hasher.update(f"array:{item.dtype}:{item.shape}".encode())
            hasher.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, pd.DataFrame):
            ordered = item.reindex(sorted(item.columns), axis=1)
            hasher.update(str(tuple(ordered.columns)).encode())
            hasher.update(pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            hasher.update(repr(item).encode())

    update(value)
    return hasher.hexdigest()


def _frame_snapshot(frame: pd.DataFrame) -> dict[str, object]:
    return {"count": len(frame), "digest": _digest(frame)}


def _l1_snapshot(result: object) -> dict[str, object]:
    l1 = getattr(result, "l1_result", result)
    report = getattr(l1, "gate_report", None)
    blockers = tuple(getattr(report, "blockers", ()) if report is not None else ())
    return {
        "gate_passed": bool(getattr(l1, "gate_passed", False)),
        "n_valid": int(getattr(l1, "n_valid", 0)),
        "n_total": int(getattr(l1, "n_total", 0)),
        "n_winning_signals": int(getattr(result, "n_winning_signals", 0)),
        "fold_pass_ratio": float(getattr(l1, "fold_pass_ratio", 0.0)),
        "blockers": blockers,
        "digest": _digest(
            (
                getattr(l1, "gate_passed", False),
                getattr(l1, "n_valid", 0),
                getattr(l1, "n_total", 0),
                getattr(l1, "fold_pass_ratio", 0.0),
                blockers,
            )
        ),
    }


def run_once(*, label: str, tfs: tuple[str, ...], ablate_1h_fusion: bool) -> dict[str, dict[str, dict[str, object]]]:
    """Run one process-local replay and restore every patched callable."""
    trace: dict[str, dict[str, dict[str, object]]] = {
        stage: {}
        for stage in (
            "native_panels",
            "cheap_evidence",
            "fusion_evidence",
            "canonical_l0",
            "manifest_route",
            "native_labeled_events",
            "l1_delivery_events",
            "l1_result",
        )
    }
    active_pipeline_module = cast(Any, active_pipeline)
    strategy_service_module = cast(Any, strategy_service)
    original_build = strategy_service.build_candidate_strategy_config
    original_active_build = active_pipeline_module.build_candidate_strategy_config
    original_multi = bridge_helpers.run_alpha_foundry_l0_gate_multi_tf
    original_phase3 = bridge_helpers._run_phase3_sequential
    original_bridge = strategy_service_module.run_candidate_strategy_for_universe
    original_select = tiered_pipeline.select_l1_delivery_events
    original_per_tf = tiered_pipeline.run_per_tf_l1

    def build_override(**kwargs: Any) -> Any:
        config = original_build(**kwargs)
        return replace(config, candidate=replace(config.candidate, l1_tfs=tfs))

    def phase3_capture(**kwargs: Any) -> dict[str, Any]:
        evidence_by_tf = kwargs["evidence_by_tf"]
        exclusions = kwargs.get("diagnostic_fusion_exclusions") or {}
        for timeframe, frame in evidence_by_tf.items():
            trace["cheap_evidence"][timeframe] = _frame_snapshot(frame)
        for target_tf in evidence_by_tf:
            excluded = exclusions.get(target_tf, frozenset())
            fusion = fuse_multi_timeframe_evidence(
                evidence_by_tf={tf: frame for tf, frame in evidence_by_tf.items() if tf not in excluded}
            )
            rows = tuple(
                (row.native_recipe_id, row.tf_coverage_count, row.sign_agreement_ratio, row.corroboration_tier)
                for row in fusion
                if row.native_timeframe == target_tf
            )
            trace["fusion_evidence"][target_tf] = {"count": len(rows), "digest": _digest(rows)}
        results = original_phase3(**kwargs)
        for timeframe, result in results.items():
            rows = tuple(
                (
                    candidate.recipe_id,
                    candidate.corroboration_tier,
                    candidate.l1_priority_score,
                    candidate.hard_reject_reasons,
                )
                for candidate in result.candidates_for_l1
            )
            trace["canonical_l0"][timeframe] = {"count": len(rows), "digest": _digest(rows)}
        return results

    def multi_override(**kwargs: Any) -> dict[str, Any]:
        if ablate_1h_fusion:
            kwargs["diagnostic_fusion_exclusions"] = {tf: frozenset({"1h"}) for tf in _CONTROL_TFS}
        return original_multi(**kwargs)

    def bridge_override(**kwargs: Any) -> Any:
        original_single = bridge._build_single_tf_panels

        def single_capture(*args: Any, **inner_kwargs: Any) -> tuple[Any, Any, Any]:
            result = original_single(*args, **inner_kwargs)
            panels = result[2]
            trace["native_panels"][str(inner_kwargs["tf_i"])] = {
                "count": len(panels),
                "digest": _digest(
                    tuple(
                        (
                            panel.family,
                            panel.variant,
                            panel.expected_holding_bars,
                            panel.signed_score_2d,
                            panel.valid_mask_2d,
                        )
                        for panel in panels
                    )
                ),
            }
            return result

        bridge._build_single_tf_panels = single_capture
        try:
            output = original_bridge(**kwargs)
        finally:
            bridge._build_single_tf_panels = original_single
        for timeframe, events in getattr(output, "labeled_events_by_tf", {}).items():
            trace["native_labeled_events"][timeframe] = _frame_snapshot(events)
        manifest = getattr(output, "l0_delivery_manifest", None)
        for route in getattr(manifest, "routes", ()):
            trace["manifest_route"][route.timeframe] = {
                "count": len(route.selected_recipe_ids),
                "digest": _digest((route.selected_recipe_ids, route.allocated_budget_units)),
            }
        return output

    def select_capture(**kwargs: Any) -> pd.DataFrame:
        events = original_select(**kwargs)
        trace["l1_delivery_events"][str(kwargs["tf"])] = _frame_snapshot(events)
        return events

    def per_tf_capture(**kwargs: Any) -> Any:
        result = original_per_tf(**kwargs)
        trace["l1_result"][str(kwargs["tf"])] = _l1_snapshot(result)
        return result

    strategy_service.build_candidate_strategy_config = build_override
    active_pipeline_module.build_candidate_strategy_config = build_override
    bridge_helpers._run_phase3_sequential = phase3_capture
    bridge_helpers.run_alpha_foundry_l0_gate_multi_tf = multi_override
    strategy_service_module.run_candidate_strategy_for_universe = bridge_override
    tiered_pipeline.select_l1_delivery_events = select_capture
    tiered_pipeline.run_per_tf_l1 = per_tf_capture
    try:
        active_pipeline.run_pipeline(
            FuturesRunConfig(
                timeframe="4h",
                date="2026-05-01",
                trials=1,
                phase="l1",
                sync="skip",
                refresh_universe=False,
                sync_metrics=False,
                l0_runtime=build_l0_runtime_config(phase="l1", settings={}),
            ),
            seed=42,
        )
    finally:
        strategy_service.build_candidate_strategy_config = original_build
        active_pipeline_module.build_candidate_strategy_config = original_active_build
        bridge_helpers.run_alpha_foundry_l0_gate_multi_tf = original_multi
        bridge_helpers._run_phase3_sequential = original_phase3
        strategy_service_module.run_candidate_strategy_for_universe = original_bridge
        tiered_pipeline.select_l1_delivery_events = original_select
        tiered_pipeline.run_per_tf_l1 = original_per_tf
    return trace


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in _RUN_SPECS:
        _LOGGER.error("[DATA] usage: %s <control|control_repeat|treatment|fusion_ablation>", Path(sys.argv[0]).name)
        return 2
    label = sys.argv[1]
    tfs, ablate_1h_fusion = _RUN_SPECS[label]
    output_path = Path("logs/futures/diagnostics/l1_cross_tf") / f"{label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        trace = run_once(label=label, tfs=tfs, ablate_1h_fusion=ablate_1h_fusion)
    except BaseException as exc:
        output_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), encoding="utf-8")
        raise
    output_path.write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
