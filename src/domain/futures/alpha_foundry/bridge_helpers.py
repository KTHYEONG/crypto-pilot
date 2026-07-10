"""Alpha Foundry L0 gate bridge helpers for the strategy runtime bridge.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
[ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
[ADR_20260707_L0_MULTI_TF_GATE_REDESIGN]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
[ADR_20260710_L0_TERMINAL_DEBUG_OBSERVABILITY]
[ADR_20260710_L0_TF_CORROBORATION_WIRING_FIX]
"""

from __future__ import annotations

import json
import logging
import time as _time_module
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    pass
from src.core.utils.utils import setup_logger
from src.domain.futures.alpha_foundry.contracts import AlphaRecipe, CandidateFeatureFamily
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe as _AlphaRecipe,
)
from src.domain.futures.alpha_foundry.contracts import (
    PanelRecipeBinding as _PanelRecipeBinding,
)
from src.domain.futures.observability import emit_csv_artifact_debug, emit_json_artifact_debug
from src.domain.futures.signals.contracts import CandidateSignalPanel

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AlphaFoundryL0Result:
    panels_for_l1: tuple[Any, ...]
    summary_report: Any | None
    gate_results: tuple[Any, ...]
    panel_bindings: tuple[Any, ...]
    artifact_rows: tuple[Any, ...] = ()
    discovery_units_for_l1: tuple[Any, ...] = ()

    @property
    def report(self) -> Any | None:
        return self.summary_report

    @property
    def evidences(self) -> tuple[Any, ...]:
        return self.gate_results

    @property
    def bindings(self) -> tuple[Any, ...]:
        return self.panel_bindings

    @property
    def evidence_rows(self) -> tuple[Any, ...]:
        return self.artifact_rows


def _bind_panels_to_recipe_ids(
    panels: Sequence[Any],
    bindings: Sequence[Any],
) -> tuple[Any, ...]:
    """Attach metadata['recipe_id'] to each panel per its binding record.

    Pure transform shared by the multi-TF cheap-evidence fan-out (Phase 1)
    and the canonical per-TF gate (`run_alpha_foundry_l0_gate`) — previously
    duplicated inline only in the latter, leaving Phase 1's panels unbound
    and `evidence_by_tf` structurally empty. [LIMIT-01][LIMIT-02]
    """
    binding_by_index = {b.panel_index: b for b in bindings}
    return tuple(
        replace(
            panel,
            metadata={
                **dict(getattr(panel, "metadata", {}) or {}),
                "recipe_id": binding_by_index[i].recipe_id,
            },
        )
        for i, panel in enumerate(panels)
        if i in binding_by_index
    )


def _normalize_variant(variant: str, timeframe: str) -> str:
    suffix = f"_{timeframe}"
    if variant.endswith(suffix):
        return variant[: -len(suffix)]
    return variant


def bind_panels_to_alpha_recipes(
    *,
    panels: Sequence[Any],
    recipes: MutableMapping[str, Any],
    timeframe: str,
    max_recipes_per_family: int,
    include_families: tuple[str, ...],
    exclude_families: tuple[str, ...],
    enable_synthetic_recipes: bool = True,
    feature_family_by_family: Mapping[str, CandidateFeatureFamily] | None = None,
) -> tuple[Any, ...]:

    include_set = set(include_families) if include_families else None
    exclude_set = set(exclude_families) if exclude_families else None

    bindings: list[Any] = []
    family_count: dict[str, int] = {}

    for i, panel in enumerate(panels):
        family = panel.family if hasattr(panel, "family") else ""
        variant = panel.variant if hasattr(panel, "variant") else ""

        if include_set and family not in include_set:
            continue
        if exclude_set and family in exclude_set:
            continue

        count = family_count.get(family, 0)
        if count >= max_recipes_per_family:
            continue
        family_count[family] = count + 1

        matched_recipe_id: str | None = None
        matched_source: str = "synthetic_recipe"
        normalized_variant = _normalize_variant(variant, timeframe)

        for rid, recipe in recipes.items():
            rv = recipe.variant if hasattr(recipe, "variant") else ""
            rf = recipe.family if hasattr(recipe, "family") else ""
            if rv == variant and rf == family:
                matched_recipe_id = rid
                matched_source = "catalog_exact"
                break
            if rv == normalized_variant and rf == family:
                matched_recipe_id = rid
                matched_source = "catalog_family_variant"
                break

        if matched_recipe_id is None:
            if not enable_synthetic_recipes:
                continue
            from src.domain.futures.alpha_foundry.recipes import (
                _make_recipe_id,
                map_signal_archetype_to_alpha_archetype,
            )

            panel_params = dict(getattr(panel, "params", {})) or {}
            synth_recipe_id = _make_recipe_id(family, variant, timeframe, panel_params)
            if synth_recipe_id in recipes:
                matched_recipe_id = synth_recipe_id
                matched_source = "synthetic_recipe"
            else:
                synth_params = dict(getattr(panel, "params", {}))
                panel_archetype = str(getattr(panel, "archetype", ""))
                recipes[synth_recipe_id] = _AlphaRecipe(
                    recipe_id=synth_recipe_id,
                    family=family,
                    variant=variant,
                    timeframe=timeframe,
                    archetype=map_signal_archetype_to_alpha_archetype(panel_archetype),
                    indicator_params=synth_params,
                    side_rule_id=f"synthetic:{family}",
                    exit_policy_id=f"synthetic:{family}",
                    required_fields=(),
                    causal_lag_bars=1,
                    max_turnover_per_year=365.0,
                )
                matched_recipe_id = synth_recipe_id
                matched_source = "synthetic_recipe"

        src = cast(
            Literal["catalog_exact", "catalog_family_variant", "synthetic_recipe"],
            matched_source,
        )
        bindings.append(
            _PanelRecipeBinding(
                panel_index=i,
                recipe_id=matched_recipe_id,
                family=family,
                variant=variant,
                source=src,
            )
        )

    return tuple(bindings)


def synthesize_recipe_from_panel(
    *,
    panel: CandidateSignalPanel,
    timeframe: str,
    catalog_recipes: Mapping[str, AlphaRecipe],
) -> AlphaRecipe:
    family = panel.family if hasattr(panel, "family") else ""
    variant = panel.variant if hasattr(panel, "variant") else ""
    panel_archetype = str(getattr(panel, "archetype", ""))

    from src.domain.futures.alpha_foundry.recipes import (
        FAMILY_ARCHETYPE,
        FAMILY_EXIT_POLICY,
        FAMILY_MAX_TURNOVER,
        FAMILY_SIDE_RULE,
        _make_recipe_id,
        map_signal_archetype_to_alpha_archetype,
    )

    params = dict(getattr(panel, "params", {})) or {}
    recipe_id = _make_recipe_id(family, variant, timeframe, params)

    if recipe_id in catalog_recipes:
        return catalog_recipes[recipe_id]

    # Synthesize from family template or archetype defaults
    alpha_arch = FAMILY_ARCHETYPE.get(family, map_signal_archetype_to_alpha_archetype(panel_archetype))
    side_rule = FAMILY_SIDE_RULE.get(family, f"synthetic:{family}")
    exit_policy = FAMILY_EXIT_POLICY.get(family, f"synthetic:{family}")
    max_turnover = FAMILY_MAX_TURNOVER.get(family, 365.0)
    required = ("close",)
    causal_lag = 1

    return AlphaRecipe(
        recipe_id=recipe_id,
        family=family,
        variant=variant,
        timeframe=timeframe,
        archetype=alpha_arch,
        indicator_params=params,
        side_rule_id=side_rule,
        exit_policy_id=exit_policy,
        required_fields=required,
        causal_lag_bars=causal_lag,
        max_turnover_per_year=max_turnover,
    )


def _write_alpha_foundry_report(
    report: Any,
    evidence_rows: Sequence[Any],
    report_dir: Path,
    run_id: str,
) -> tuple[str, str]:
    import pandas as pd

    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = str(report_dir / f"{run_id}_report.json")
    parquet_path = str(report_dir / f"{run_id}_evidence.parquet")

    report_dict = {
        "run_id": report.run_id,
        "mode": report.mode,
        "timeframe": report.timeframe,
        "symbols": list(report.symbols),
        "n_bars": report.n_bars,
        "n_panels_in": report.n_panels_in,
        "n_bound_panels": report.n_bound_panels,
        "n_evidence": report.n_evidence,
        "n_passed": report.n_passed,
        "n_rejected": report.n_rejected,
        "reject_reason_counts": dict(report.reject_reason_counts),
        "elapsed_sec": report.elapsed_sec,
    }
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    if evidence_rows:
        df = pd.DataFrame([asdict(r) for r in evidence_rows])
    else:
        df = pd.DataFrame(
            {
                "run_id": pd.Series(dtype="str"),
                "timeframe": pd.Series(dtype="str"),
                "family": pd.Series(dtype="str"),
                "variant": pd.Series(dtype="str"),
                "recipe_id": pd.Series(dtype="str"),
                "archetype": pd.Series(dtype="str"),
                "n_events": pd.Series(dtype="int64"),
                "effective_n": pd.Series(dtype="float64"),
                "mean_net_bps": pd.Series(dtype="float64"),
                "nw_tstat": pd.Series(dtype="float64"),
                "block_lcb_bps": pd.Series(dtype="float64"),
                "rank_ic": pd.Series(dtype="float64"),
                "incremental_rank_ic": pd.Series(dtype="float64"),
                "cost_drag_ratio": pd.Series(dtype="float64"),
                "turnover_per_year": pd.Series(dtype="float64"),
                "compute_cost_score": pd.Series(dtype="float64"),
                "bootstrap_lcb_bps": pd.Series(dtype="float64"),
                "bootstrap_agree": pd.Series(dtype="bool"),
                "gate_passed": pd.Series(dtype="bool"),
                "reject_reasons": pd.Series(dtype="str"),
                "bucket_key": pd.Series(dtype="str"),
                "bucket_rank": pd.Series(dtype="int64"),
                "selected_for_l1": pd.Series(dtype="bool"),
                "redundant_with": pd.Series(dtype="str"),
                "bucket_eff_test_count": pd.Series(dtype="float64"),
                "global_eff_test_count": pd.Series(dtype="float64"),
                "created_at_ms": pd.Series(dtype="int64"),
            }
        )
    df.to_parquet(parquet_path, index=False)

    return json_path, parquet_path


def run_alpha_foundry_l0_gate(
    *,
    panels: Sequence[Any],
    bindings: Sequence[Any],
    recipes: Mapping[str, Any],
    aligned: Any,
    cost_model: Any,
    runtime_config: Any,
    run_id: str,
    timeframe: str,
    evidence_by_tf: Mapping[str, Any] | None = None,
) -> AlphaFoundryL0Result:
    from src.domain.futures.alpha_foundry.contracts import (
        AlphaFoundryBridgeReport,
    )

    raw_mode = runtime_config.mode if hasattr(runtime_config, "mode") else "off"
    if raw_mode == "off":
        return AlphaFoundryL0Result(
            panels_for_l1=tuple(panels),
            summary_report=None,
            gate_results=(),
            panel_bindings=tuple(bindings),
        )

    mode: Literal["audit", "gate"] = cast(Literal["audit", "gate"], raw_mode)

    bound_panels = list(_bind_panels_to_recipe_ids(panels, bindings))

    cheap_config = runtime_config.cheap_gate if hasattr(runtime_config, "cheap_gate") else None
    l0_start = _time_module.perf_counter()

    evidence_rows: tuple[Any, ...] = ()
    l0_artifacts: Any = None

    if cheap_config is not None and bound_panels:
        from src.domain.futures.alpha_foundry.pipeline import (
            run_alpha_foundry_l0_pipeline,
        )

        l0_artifacts = run_alpha_foundry_l0_pipeline(
            panels=bound_panels,
            recipes=recipes,
            aligned=aligned,
            cost_model=cost_model,
            cheap_gate_config=cheap_config,
            run_id=run_id,
            top_k_per_family_tf=(
                runtime_config.top_k_per_family_tf if hasattr(runtime_config, "top_k_per_family_tf") else 5
            ),
            min_conviction_lcb_bps=(
                runtime_config.min_conviction_lcb_bps if hasattr(runtime_config, "min_conviction_lcb_bps") else 5.0
            ),
            total_l1_verification_budget=(
                runtime_config.total_l1_verification_budget
                if hasattr(runtime_config, "total_l1_verification_budget")
                else 30
            ),
            runtime_config=runtime_config,
            evidence_by_tf=evidence_by_tf,
        )
        evidences = l0_artifacts.evidences
        evidence_rows = l0_artifacts.evidence_rows
    else:
        evidences = ()
        evidence_rows = ()

    # Gate mode: forward only panels matching passed_recipe_ids
    if mode == "gate" and l0_artifacts is not None:
        passed_ids = set(l0_artifacts.passed_recipe_ids)
        passed_panel_indices = {b.panel_index for b in bindings if b.recipe_id in passed_ids}
        panels_for_l1 = tuple(p for i, p in enumerate(panels) if i in passed_panel_indices)
    elif mode == "audit":
        panels_for_l1 = tuple(panels)
    else:
        panels_for_l1 = tuple(panels)

    elapsed_sec = _time_module.perf_counter() - l0_start

    n_panels_in = len(panels)
    n_bound = len(bindings)
    n_evidence = len(evidences)
    n_passed = sum(1 for row in evidence_rows if bool(getattr(row, "gate_passed", False)))
    n_rejected = n_evidence - n_passed

    from src.domain.futures.alpha_foundry.diversity import estimate_distinct_thesis_count
    passed_families = [
        str(getattr(row, "family", ""))
        for row in evidence_rows
        if bool(getattr(row, "gate_passed", False))
    ]
    n_distinct_thesis_ids_passed = estimate_distinct_thesis_count(passed_families)

    symbols = aligned.symbols if hasattr(aligned, "symbols") else ()
    n_bars = aligned.close_2d.shape[0] if hasattr(aligned, "close_2d") else 0

    reject_reason_counts: dict[str, int] = {}
    if l0_artifacts is not None:
        reject_reason_counts = l0_artifacts.reject_reason_counts

    report = AlphaFoundryBridgeReport(
        run_id=run_id,
        mode=mode,
        timeframe=timeframe,
        symbols=symbols,
        n_bars=n_bars,
        n_panels_in=n_panels_in,
        n_bound_panels=n_bound,
        n_evidence=n_evidence,
        n_passed=n_passed,
        n_rejected=n_rejected,
        reject_reason_counts=reject_reason_counts,
        elapsed_sec=elapsed_sec,
        n_distinct_thesis_ids_passed=n_distinct_thesis_ids_passed,
        json_path="",
        parquet_path="",
    )

    jp, pp = maybe_write_alpha_foundry_report(
        report=report,
        evidence_rows=evidence_rows,
        runtime_config=runtime_config,
    )
    json_path = jp if jp is not None else ""
    parquet_path = pp if pp is not None else ""
    if json_path or parquet_path:
        report = AlphaFoundryBridgeReport(
            run_id=run_id,
            mode=mode,
            timeframe=timeframe,
            symbols=symbols,
            n_bars=n_bars,
            n_panels_in=n_panels_in,
            n_bound_panels=n_bound,
            n_evidence=n_evidence,
            n_passed=n_passed,
            n_rejected=n_rejected,
            reject_reason_counts=reject_reason_counts,
            elapsed_sec=elapsed_sec,
            n_distinct_thesis_ids_passed=n_distinct_thesis_ids_passed,
            json_path=json_path,
            parquet_path=parquet_path,
        )

    return AlphaFoundryL0Result(
        panels_for_l1=panels_for_l1,
        summary_report=report,
        gate_results=evidences,
        panel_bindings=tuple(bindings),
        artifact_rows=evidence_rows,
    )


def maybe_write_alpha_foundry_report(
    *,
    report: Any,
    evidence_rows: Sequence[Any],
    runtime_config: Any,
) -> tuple[str | None, str | None]:
    artifact_enabled = getattr(runtime_config, "artifact_write_enabled", False)
    observability = getattr(runtime_config, "observability_mode", "debug_log")

    if not artifact_enabled:
        if observability == "debug_log":
            logger = setup_logger("opt_main_futures", write_file=False)
            capture_logger = logging.getLogger(__name__)
            report_payload = {
                "run_id": getattr(report, "run_id", ""),
                "mode": getattr(report, "mode", ""),
                "timeframe": getattr(report, "timeframe", ""),
                "symbols": list(getattr(report, "symbols", ()) or ()),
                "n_bars": getattr(report, "n_bars", 0),
                "n_panels_in": getattr(report, "n_panels_in", 0),
                "n_bound_panels": getattr(report, "n_bound_panels", 0),
                "n_evidence": getattr(report, "n_evidence", 0),
                "n_passed": getattr(report, "n_passed", 0),
                "n_rejected": getattr(report, "n_rejected", 0),
                "reject_reason_counts": dict(getattr(report, "reject_reason_counts", {}) or {}),
                "elapsed_sec": getattr(report, "elapsed_sec", 0.0),
            }
            evidence_payload = [asdict(row) for row in evidence_rows]
            for debug_logger in (logger, capture_logger):
                emit_json_artifact_debug(
                    logger=debug_logger,
                    artifact_name="alpha_foundry_report",
                    run_id=report_payload["run_id"],
                    payload=report_payload,
                )
                emit_csv_artifact_debug(
                    logger=debug_logger,
                    artifact_name="alpha_foundry_evidence",
                    run_id=report_payload["run_id"],
                    rows=evidence_payload,
                )
        return (None, None)

    report_dir = getattr(runtime_config, "report_dir", Path("logs/futures/alpha_foundry"))
    run_id = getattr(report, "run_id", "unknown")
    return _write_alpha_foundry_report(
        report=report,
        evidence_rows=evidence_rows,
        report_dir=report_dir,
        run_id=run_id,
    )


def build_cheap_gate_evidence_frame(
    *,
    panels: Sequence[Any],
    recipes: Mapping[str, Any],
    aligned: Any,
    cost_model: Any,
    cheap_gate_config: Any,
    timeframe: str,
) -> Any:
    import pandas as pd

    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_cheap_gate_batch

    cheap_evidences = evaluate_alpha_cheap_gate_batch(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        config=cheap_gate_config,
    )
    rows: list[dict[str, Any]] = []
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        rows.append(
            {
                "family": recipe.family,
                "variant": recipe.variant,
                "timeframe": recipe.timeframe,
                "recipe_id": ev.recipe_id,
                "reject_reasons": "|".join(ev.reject_reasons),
                "mean_net_bps": ev.mean_net_bps,
                "block_lcb_bps": ev.block_lcb_bps,
            }
        )
    if not rows:
        return pd.DataFrame(
            {
                "family": pd.Series(dtype=str),
                "variant": pd.Series(dtype=str),
                "timeframe": pd.Series(dtype=str),
                "recipe_id": pd.Series(dtype=str),
                "reject_reasons": pd.Series(dtype=str),
                "mean_net_bps": pd.Series(dtype=float),
                "block_lcb_bps": pd.Series(dtype=float),
            }
        )
    return pd.DataFrame(rows)


def run_alpha_foundry_l0_gate_multi_tf(
    *,
    panels_by_tf: Mapping[str, Sequence[Any]],
    bindings_by_tf: Mapping[str, Sequence[Any]],
    recipes_by_tf: Mapping[str, MutableMapping[str, Any]],
    aligned_by_tf: Mapping[str, Any],
    cost_model: Any,
    runtime_config: Any,
    run_id_prefix: str,
) -> dict[str, AlphaFoundryL0Result]:
    missing = set(panels_by_tf) - set(aligned_by_tf)
    if missing:
        raise ValueError(f"aligned_by_tf missing timeframe: {next(iter(missing))}")

    import pandas as pd

    # Phase 1: bind panels to recipe_id, then cheap-gate per TF -> evidence_by_tf [LIMIT-01][LIMIT-03]
    evidence_by_tf: dict[str, Any] = {}
    for tf, tf_panels in panels_by_tf.items():
        tf_recipes = recipes_by_tf.get(tf, {})
        tf_bindings = bindings_by_tf.get(tf, [])
        if not tf_panels or not tf_recipes:
            evidence_by_tf[tf] = pd.DataFrame(
                {
                    "family": pd.Series(dtype=str),
                    "variant": pd.Series(dtype=str),
                    "timeframe": pd.Series(dtype=str),
                    "recipe_id": pd.Series(dtype=str),
                    "reject_reasons": pd.Series(dtype=str),
                    "mean_net_bps": pd.Series(dtype=float),
                    "block_lcb_bps": pd.Series(dtype=float),
                }
            )
            continue
        bound_tf_panels = _bind_panels_to_recipe_ids(tf_panels, tf_bindings)
        evidence_by_tf[tf] = build_cheap_gate_evidence_frame(
            panels=bound_tf_panels,
            recipes=tf_recipes,
            aligned=aligned_by_tf[tf],
            cost_model=cost_model,
            cheap_gate_config=runtime_config.cheap_gate,
            timeframe=tf,
        )
        n_evidence_rows = len(evidence_by_tf[tf])
        _log_fn = _logger.warning if (tf_bindings and n_evidence_rows == 0) else _logger.debug
        _log_fn(
            "[SYS] stage=multi_tf_cheap_evidence tf=%s n_panels_in=%d n_bindings=%d n_evidence_rows=%d",
            tf, len(tf_panels), len(tf_bindings), n_evidence_rows,
        )  # [LIMIT-08][LIMIT-09]

    # Phase 2: fuse_multi_timeframe_evidence is called inside pipeline
    # Phase 3: canonical gate per TF
    results: dict[str, AlphaFoundryL0Result] = {}
    for tf in panels_by_tf:
        tf_panels = panels_by_tf[tf]
        tf_bindings = bindings_by_tf.get(tf, [])
        tf_recipes = recipes_by_tf.get(tf, {})
        tf_aligned = aligned_by_tf[tf]
        _run_id = f"{run_id_prefix}_{tf}"
        results[tf] = run_alpha_foundry_l0_gate(
            panels=tf_panels,
            bindings=tf_bindings,
            recipes=tf_recipes,
            aligned=tf_aligned,
            cost_model=cost_model,
            runtime_config=runtime_config,
            run_id=_run_id,
            timeframe=tf,
            evidence_by_tf=evidence_by_tf,
        )

    # [EVAL] tf_corroboration distribution summary — once per run [LIMIT-08]
    _all_rows: list[Any] = []
    for r in results.values():
        _all_rows.extend(r.evidence_rows)
    if _all_rows:
        _n_total = len(_all_rows)
        _n_gt0 = 0
        _cov_vals: list[int] = []
        for row in _all_rows:
            cov: int = int(getattr(row, "tf_coverage_count", 0) or 0)
            if cov > 0:
                _n_gt0 += 1
                _cov_vals.append(cov)
        _cov_vals.sort()
        _median_cov = _cov_vals[len(_cov_vals) // 2] if _cov_vals else 0
        _logger.debug(
            "[EVAL] stage=tf_corroboration_summary"
            " n_rows_total=%d n_rows_tf_coverage_gt0=%d median_tf_coverage_count=%d",
            _n_total, _n_gt0, _median_cov,
        )
    return results
