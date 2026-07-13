"""Alpha Foundry L0 gate bridge helpers for the strategy runtime bridge.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
[ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
[ADR_20260707_L0_MULTI_TF_GATE_REDESIGN]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
[ADR_20260710_L0_TERMINAL_DEBUG_OBSERVABILITY]
[ADR_20260710_L0_TF_CORROBORATION_WIRING_FIX]
[ADR_20260711_L0_STRATEGY_DELIVERY_HARDENING]
[ADR_20260711_L0_CROSS_TF_PRUNING_ADMISSION]
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
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.core.utils.utils import setup_logger
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CandidateFeatureFamily,
    L0StrategyDeliveryManifest,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe as _AlphaRecipe,
)
from src.domain.futures.alpha_foundry.contracts import (
    PanelRecipeBinding as _PanelRecipeBinding,
)
from src.domain.futures.observability import emit_csv_artifact_debug, emit_json_artifact_debug
from src.domain.futures.signals.contracts import CandidateSignalPanel

_logger = logging.getLogger(__name__)

_L0_TF_INPUT_CACHE: dict[str, tuple[Any, ...]] = {}
_L0_PHASE1_INPUT_CACHE: dict[str, tuple[Any, ...]] = {}
"""Module-level prefork cache: tf -> (panels, bindings, recipes, aligned, cost_model, runtime_config,
evidence_by_tf, cheap_evidences_for_tf).
Populated in the parent process before ProcessPoolExecutor creation; child
workers (fork mp_context) inherit this via copy-on-write and look up their
own slice by `tf` key, avoiding per-task pickling of large AlignedMarketData
arrays. [LIMIT-02]
"""


@dataclass(slots=True, frozen=True)
class AlphaFoundryL0Result:
    panels_for_l1: tuple[Any, ...]
    summary_report: Any | None
    gate_results: tuple[Any, ...]
    panel_bindings: tuple[Any, ...]
    artifact_rows: tuple[Any, ...] = ()
    discovery_units_for_l1: tuple[Any, ...] = ()
    candidates_for_l1: tuple[Any, ...] = ()

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
    precomputed_cheap_evidences: tuple[Any, ...] | None = None,
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
            precomputed_cheap_evidences=precomputed_cheap_evidences,
        )
        evidences = l0_artifacts.evidences
        evidence_rows = l0_artifacts.evidence_rows
    else:
        evidences = ()
        evidence_rows = ()

    # Gate mode: forward only panels matching passed_recipe_ids
    candidates_for_l1: tuple[Any, ...] = ()
    if mode == "gate" and l0_artifacts is not None:
        passed_ids = set(l0_artifacts.passed_recipe_ids)
        binding_by_panel_index = {b.panel_index: b for b in bindings}
        # Stamp metadata["recipe_id"] onto forwarded panels (mirrors
        # _bind_panels_to_recipe_ids) so downstream consumers — e.g. the
        # cross-TF independence audit — can key panels by recipe_id without
        # needing the discarded `bound_panels` local. [ADR_20260711_L0_STRATEGY_DELIVERY_HARDENING]
        panels_for_l1 = tuple(
            replace(
                p,
                metadata={
                    **dict(getattr(p, "metadata", {}) or {}),
                    "recipe_id": binding_by_panel_index[i].recipe_id,
                },
            )
            for i, p in enumerate(panels)
            if i in binding_by_panel_index and binding_by_panel_index[i].recipe_id in passed_ids
        )
        candidates_for_l1 = tuple(c for c in l0_artifacts.candidates if c.recipe_id in passed_ids)
    elif mode == "audit":
        panels_for_l1 = tuple(panels)
    else:
        panels_for_l1 = tuple(panels)

    elapsed_sec = _time_module.perf_counter() - l0_start

    n_panels_in = len(panels)
    n_bound = len(bindings)
    n_evidence = len(evidences)
    passed_ids_for_report = (
        set(l0_artifacts.passed_recipe_ids)
        if mode == "gate" and l0_artifacts is not None
        else None
    )
    n_passed = sum(
        1
        for row in evidence_rows
        if bool(getattr(row, "gate_passed", False))
        and (
            passed_ids_for_report is None
            or str(getattr(row, "recipe_id", "")) in passed_ids_for_report
        )
    )
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

    _t_report_write = _time_module.perf_counter()
    jp, pp = maybe_write_alpha_foundry_report(
        report=report,
        evidence_rows=evidence_rows,
        runtime_config=runtime_config,
    )
    setup_logger("opt_main_futures", write_file=False).debug(
        "[SYS] stage=report_write tf=%s n_rows=%d took=%.4fs",
        timeframe, len(evidence_rows), _time_module.perf_counter() - _t_report_write,
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
        candidates_for_l1=candidates_for_l1,
    )


def maybe_write_alpha_foundry_report(
    *,
    report: Any,
    evidence_rows: Sequence[Any],
    runtime_config: Any,
) -> tuple[str | None, str | None]:
    # [ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION] Log L0 Gate survivors summary
    if getattr(report, "mode", "") == "gate":
        passed_rows = [r for r in evidence_rows if bool(getattr(r, "gate_passed", False))]
        if passed_rows:
            _logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            _logger.info("🟢 [L0 GATE SURVIVORS SUMMARY]")
            _logger.info("--------------------------------------------------------------")
            _logger.info("  %-32s | %-5s | %-12s | %-8s", "Recipe ID", "TF", "Net LCB (bps)", "t-stat")
            _logger.info("--------------------------------------------------------------")
            for r in passed_rows[:20]:
                recipe_id = getattr(r, "recipe_id", "")
                tf = getattr(r, "timeframe", "")
                net_lcb = getattr(r, "net_lcb_bps", 0.0)
                t_stat = getattr(r, "nw_tstat", 0.0)
                _logger.info("  %-32s | %-5s | %12.2f | %8.2f", recipe_id[:32], tf, net_lcb, t_stat)
            if len(passed_rows) > 20:
                _logger.info("  ... and %d more recipes.", len(passed_rows) - 20)
            _logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

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
                    run_id=str(report_payload["run_id"]),
                    payload=report_payload,
                )
                emit_csv_artifact_debug(
                    logger=debug_logger,
                    artifact_name="alpha_foundry_evidence",
                    run_id=str(report_payload["run_id"]),
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


def build_cheap_gate_evidence_frame_from_evidences(
    *,
    cheap_evidences: Sequence[Any],
    recipes: Mapping[str, Any],
) -> Any:
    """Pure DataFrame-projection extracted from build_cheap_gate_evidence_frame.
    [LIMIT-05] Identical row-building logic; callable directly when evidences
    are already computed, avoiding a second evaluate_alpha_cheap_gate_batch call.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        data_support_tier = str(getattr(ev, "data_support_tier", "full_support"))
        rows.append(
            {
                "family": recipe.family,
                "variant": recipe.variant,
                "timeframe": recipe.timeframe,
                "recipe_id": ev.recipe_id,
                "reject_reasons": "|".join(ev.reject_reasons),
                "mean_net_bps": ev.mean_net_bps,
                "block_lcb_bps": ev.block_lcb_bps,
                "data_support_tier": data_support_tier,
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
                "data_support_tier": pd.Series(dtype=str),
            }
        )
    return pd.DataFrame(rows)


def build_cheap_gate_evidence_frame(
    *,
    panels: Sequence[Any],
    recipes: Mapping[str, Any],
    aligned: Any,
    cost_model: Any,
    cheap_gate_config: Any,
    timeframe: str,
) -> Any:
    """[LIMIT-05] UNCHANGED signature/behavior -- now a thin wrapper:
    evaluate_alpha_cheap_gate_batch(...) then
    build_cheap_gate_evidence_frame_from_evidences(...).
    """
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_cheap_gate_batch

    cheap_evidences = evaluate_alpha_cheap_gate_batch(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        config=cheap_gate_config,
    )
    return build_cheap_gate_evidence_frame_from_evidences(
        cheap_evidences=cheap_evidences, recipes=recipes,
    )


def _prime_l0_phase1_input_cache(
    *,
    panels_by_tf: Mapping[str, Sequence[Any]],
    recipes_by_tf: Mapping[str, MutableMapping[str, Any]],
    bindings_by_tf: Mapping[str, Sequence[Any]],
    aligned_by_tf: Mapping[str, Any],
    cost_model: Any,
    runtime_config: Any,
) -> None:
    _L0_PHASE1_INPUT_CACHE.clear()
    for tf in panels_by_tf:
        _L0_PHASE1_INPUT_CACHE[tf] = (
            panels_by_tf[tf],
            recipes_by_tf.get(tf, {}),
            bindings_by_tf.get(tf, []),
            aligned_by_tf[tf],
            cost_model,
            runtime_config,
        )


def _run_l0_cheap_evidence_worker(tf: str) -> tuple[str, tuple[Any, ...], pd.DataFrame]:
    panels, recipes, bindings, aligned, cost_model, runtime_config = (
        _L0_PHASE1_INPUT_CACHE[tf]
    )
    bound_tf_panels = _bind_panels_to_recipe_ids(panels, bindings)
    from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_cheap_gate_batch
    cheap_evidences = evaluate_alpha_cheap_gate_batch(
        panels=bound_tf_panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        config=runtime_config.cheap_gate,
    )
    evidence_df = build_cheap_gate_evidence_frame_from_evidences(
        cheap_evidences=cheap_evidences, recipes=recipes,
    )
    return tf, cheap_evidences, evidence_df


def _prime_l0_tf_input_cache(
    *,
    panels_by_tf: Mapping[str, Sequence[Any]],
    bindings_by_tf: Mapping[str, Sequence[Any]],
    recipes_by_tf: Mapping[str, MutableMapping[str, Any]],
    aligned_by_tf: Mapping[str, Any],
    cost_model: Any,
    runtime_config: Any,
    evidence_by_tf: Mapping[str, Any],
    cheap_evidences_by_tf: Mapping[str, tuple[Any, ...]],
) -> None:
    """Populate _L0_TF_INPUT_CACHE. Must be called in the parent process
    strictly before the ProcessPoolExecutor is created. [LIMIT-02]
    """
    _L0_TF_INPUT_CACHE.clear()
    for tf in panels_by_tf:
        _L0_TF_INPUT_CACHE[tf] = (
            panels_by_tf[tf],
            bindings_by_tf.get(tf, []),
            recipes_by_tf.get(tf, {}),
            aligned_by_tf[tf],
            cost_model,
            runtime_config,
            evidence_by_tf,
            cheap_evidences_by_tf.get(tf, ()),
        )


def _run_l0_gate_worker(tf: str, run_id: str) -> tuple[str, AlphaFoundryL0Result]:
    """Worker entrypoint: looks up its inputs from _L0_TF_INPUT_CACHE[tf]
    (inherited via fork COW, never passed as a submit() argument) and calls
    run_alpha_foundry_l0_gate(...) exactly as the sequential path does.
    Picklable return: (tf, AlphaFoundryL0Result) -- small, no raw market data.
    """
    panels, bindings, recipes, aligned, cost_model, runtime_config, evidence_by_tf, cheap_evidences = (
        _L0_TF_INPUT_CACHE[tf]
    )
    result = run_alpha_foundry_l0_gate(
        panels=panels,
        bindings=bindings,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        runtime_config=runtime_config,
        run_id=run_id,
        timeframe=tf,
        evidence_by_tf=evidence_by_tf,
        precomputed_cheap_evidences=cheap_evidences,
    )
    return tf, result


def _run_phase3_sequential(
    *,
    panels_by_tf: Mapping[str, Sequence[Any]],
    bindings_by_tf: Mapping[str, Sequence[Any]],
    recipes_by_tf: Mapping[str, MutableMapping[str, Any]],
    aligned_by_tf: Mapping[str, Any],
    cost_model: Any,
    runtime_config: Any,
    run_id_prefix: str,
    evidence_by_tf: Mapping[str, Any],
    cheap_evidences_by_tf: Mapping[str, tuple[Any, ...]],
) -> dict[str, AlphaFoundryL0Result]:
    """Run Phase 3 canonical gate sequentially (the original behavior)."""
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
            precomputed_cheap_evidences=cheap_evidences_by_tf.get(tf),
        )
    return results


def _run_phase3_parallel(
    *,
    panels_by_tf: Mapping[str, Sequence[Any]],
    bindings_by_tf: Mapping[str, Sequence[Any]],
    recipes_by_tf: Mapping[str, MutableMapping[str, Any]],
    aligned_by_tf: Mapping[str, Any],
    cost_model: Any,
    runtime_config: Any,
    run_id_prefix: str,
    evidence_by_tf: Mapping[str, Any],
    cheap_evidences_by_tf: Mapping[str, tuple[Any, ...]],
    max_workers: int,
) -> dict[str, AlphaFoundryL0Result]:
    """Run Phase 3 canonical gate in parallel via fork-based ProcessPoolExecutor
    with prefork COW cache. [LIMIT-02][LIMIT-03]
    """
    _prime_l0_tf_input_cache(
        panels_by_tf=panels_by_tf,
        bindings_by_tf=bindings_by_tf,
        recipes_by_tf=recipes_by_tf,
        aligned_by_tf=aligned_by_tf,
        cost_model=cost_model,
        runtime_config=runtime_config,
        evidence_by_tf=evidence_by_tf,
        cheap_evidences_by_tf=cheap_evidences_by_tf,
    )
    import multiprocessing as _mp

    tfs = list(panels_by_tf)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=_mp.get_context("fork")) as executor:
        futures = {
            executor.submit(_run_l0_gate_worker, tf, f"{run_id_prefix}_{tf}"): tf
            for tf in tfs
        }
        results: dict[str, AlphaFoundryL0Result] = {}
        for future in futures:
            tf_key = futures[future]
            _tf, result = future.result()
            results[tf_key] = result
    return results


def run_alpha_foundry_l0_gate_multi_tf(
    *,
    panels_by_tf: Mapping[str, Sequence[Any]],
    bindings_by_tf: Mapping[str, Sequence[Any]],
    recipes_by_tf: Mapping[str, MutableMapping[str, Any]],
    aligned_by_tf: Mapping[str, Any],
    cost_model: Any,
    runtime_config: Any,
    run_id_prefix: str,
    parallel_max_workers: int = 1,
) -> dict[str, AlphaFoundryL0Result]:
    """[LIMIT-05] Signature/return type UNCHANGED except the additive
    keyword-only `parallel_max_workers` (default 1 = today's exact
    sequential behavior, zero call-site changes required).

    Raises:
        ValueError: if `parallel_max_workers` not in [1, 4]. [LIMIT-03]
        ValueError: (unchanged) if aligned_by_tf missing a timeframe present
            in panels_by_tf.
    """
    if not (1 <= parallel_max_workers <= 4):
        raise ValueError(
            f"parallel_max_workers must be in [1,4], got {parallel_max_workers}"
        )

    missing = set(panels_by_tf) - set(aligned_by_tf)
    if missing:
        raise ValueError(f"aligned_by_tf missing timeframe: {next(iter(missing))}")

    import pandas as pd

    # Phase 1: bind panels to recipe_id, then cheap-gate per TF -> evidence_by_tf [LIMIT-01][LIMIT-03]
    # Stash raw CheapGateEvidence tuples for Phase 3 reuse (avoids redundant recomputation)
    _t_phase1 = _time_module.perf_counter()
    evidence_by_tf: dict[str, Any] = {}
    cheap_evidences_by_tf: dict[str, tuple[Any, ...]] = {}
    if parallel_max_workers <= 1:
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
                        "data_support_tier": pd.Series(dtype=str),
                    }
                )
                cheap_evidences_by_tf[tf] = ()
                continue
            bound_tf_panels = _bind_panels_to_recipe_ids(tf_panels, tf_bindings)
            from src.domain.futures.alpha_foundry.cheap_gate import evaluate_alpha_cheap_gate_batch

            cheap_evidences = evaluate_alpha_cheap_gate_batch(
                panels=bound_tf_panels,
                recipes=tf_recipes,
                aligned=aligned_by_tf[tf],
                cost_model=cost_model,
                config=runtime_config.cheap_gate,
            )
            cheap_evidences_by_tf[tf] = cheap_evidences
            evidence_by_tf[tf] = build_cheap_gate_evidence_frame_from_evidences(
                cheap_evidences=cheap_evidences, recipes=tf_recipes,
            )
            n_evidence_rows = len(evidence_by_tf[tf])
            _log_fn = _logger.warning if (tf_bindings and n_evidence_rows == 0) else _logger.debug
            _log_fn(
                "[SYS] stage=multi_tf_cheap_evidence tf=%s n_panels_in=%d n_bindings=%d n_evidence_rows=%d",
                tf, len(tf_panels), len(tf_bindings), n_evidence_rows,
            )  # [LIMIT-08][LIMIT-09]
    else:
        _prime_l0_phase1_input_cache(
            panels_by_tf=panels_by_tf,
            recipes_by_tf=recipes_by_tf,
            bindings_by_tf=bindings_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=cost_model,
            runtime_config=runtime_config,
        )
        import multiprocessing as _mp
        with ProcessPoolExecutor(max_workers=parallel_max_workers, mp_context=_mp.get_context("fork")) as executor:
            futures = {
                executor.submit(_run_l0_cheap_evidence_worker, tf): tf
                for tf in panels_by_tf
                if panels_by_tf[tf] and recipes_by_tf.get(tf)
            }
            # Populate empty results for those without panels or recipes
            for tf in panels_by_tf:
                if not panels_by_tf[tf] or not recipes_by_tf.get(tf):
                    evidence_by_tf[tf] = pd.DataFrame(
                        {
                            "family": pd.Series(dtype=str),
                            "variant": pd.Series(dtype=str),
                            "timeframe": pd.Series(dtype=str),
                            "recipe_id": pd.Series(dtype=str),
                            "reject_reasons": pd.Series(dtype=str),
                            "mean_net_bps": pd.Series(dtype=float),
                            "block_lcb_bps": pd.Series(dtype=float),
                            "data_support_tier": pd.Series(dtype=str),
                        }
                    )
                    cheap_evidences_by_tf[tf] = ()

            for future in futures:
                tf_key = futures[future]
                _, cheap_evidences, evidence_df = future.result()
                cheap_evidences_by_tf[tf_key] = cheap_evidences
                evidence_by_tf[tf_key] = evidence_df
                n_evidence_rows = len(evidence_df)
                tf_bindings = bindings_by_tf.get(tf_key, [])
                _log_fn = _logger.warning if (tf_bindings and n_evidence_rows == 0) else _logger.debug
                _log_fn(
                    "[SYS] stage=multi_tf_cheap_evidence tf=%s n_panels_in=%d n_bindings=%d n_evidence_rows=%d",
                    tf_key, len(panels_by_tf[tf_key]), len(tf_bindings), n_evidence_rows,
                )
        _L0_PHASE1_INPUT_CACHE.clear()

    setup_logger("opt_main_futures", write_file=False).debug(
        "[SYS] stage=l0_phase1_cheap_evidence took=%.4fs n_tfs=%d",
        _time_module.perf_counter() - _t_phase1, len(panels_by_tf),
    )

    # Phase 2: fuse_multi_timeframe_evidence is called inside pipeline
    # Phase 3: canonical gate per TF
    _t_phase3 = _time_module.perf_counter()
    if parallel_max_workers <= 1:
        results = _run_phase3_sequential(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=cost_model,
            runtime_config=runtime_config,
            run_id_prefix=run_id_prefix,
            evidence_by_tf=evidence_by_tf,
            cheap_evidences_by_tf=cheap_evidences_by_tf,
        )
    else:
        results = _run_phase3_parallel(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=cost_model,
            runtime_config=runtime_config,
            run_id_prefix=run_id_prefix,
            evidence_by_tf=evidence_by_tf,
            cheap_evidences_by_tf=cheap_evidences_by_tf,
            max_workers=parallel_max_workers,
        )
    setup_logger("opt_main_futures", write_file=False).debug(
        "[SYS] stage=l0_phase3_canonical_gate took=%.4fs parallel_max_workers=%d",
        _time_module.perf_counter() - _t_phase3, parallel_max_workers,
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


def assemble_l0_strategy_delivery_manifest(
    *,
    multi_results: Mapping[str, AlphaFoundryL0Result],
    aligned_by_tf: Mapping[str, Any],
    run_id_prefix: str,
    enable_audit: bool,
    enable_pruning: bool,
    total_l1_verification_budget: int = 30,
    max_novelty_corr: float = 0.70,
    min_survivors_per_archetype: int = 1,
    min_survivors_per_tf: int = 0,
    min_common_active_bars: int = 480,
    min_directional_entry_jaccard: float = 0.50,
    min_shared_directional_entries: int = 12,
) -> tuple[dict[str, AlphaFoundryL0Result], L0StrategyDeliveryManifest]:
    """Cross-TF post-processing over an already-completed multi-TF L0 gate
    run. [ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION]

    Resolves its own canonical context from candidates, removing the
    caller-provided `canonical_tf`. Pure function: multi_results and
    aligned_by_tf are never mutated.

    Returns (possibly-pruned multi_results, manifest). When both
    enable_audit and enable_pruning are False, returns
    (dict(multi_results), manifest-with-Nones) — a cheap passthrough.
    """
    manifest_audit: Any = None
    manifest_final_ids: tuple[str, ...] = ()
    manifest_status: str = "disabled"
    manifest_reason: str = ""
    pruned_multi_results: dict[str, AlphaFoundryL0Result] = dict(multi_results)

    selected_by_tf: dict[str, Any] = {}
    panel_by_recipe_id: dict[str, Any] = {}
    all_tfs_with_candidates = False

    for tf_k, res in multi_results.items():
        cands = getattr(res, "candidates_for_l1", ())
        if cands:
            selected_by_tf[tf_k] = cands

    for res in multi_results.values():
        for p in res.panels_for_l1:
            rid = getattr(p, "metadata", {}).get("recipe_id", "")
            if rid:
                panel_by_recipe_id[rid] = p

    all_tfs_with_candidates = bool(selected_by_tf and panel_by_recipe_id)

    shared_context: Any = None
    if enable_audit and enable_pruning and all_tfs_with_candidates:
        from src.domain.futures.alpha_foundry.diversity import resolve_cross_tf_shared_context

        try:
            shared_context = resolve_cross_tf_shared_context(
                selected_by_tf=selected_by_tf,
                panel_by_recipe_id=panel_by_recipe_id,
                aligned_by_tf=aligned_by_tf,
                min_common_active_bars=min_common_active_bars,
            )
        except ValueError as exc:
            manifest_status = "fail_open"
            manifest_reason = str(exc)
            _fallback_logger = setup_logger("opt_main_futures", write_file=False)
            for _l in (_fallback_logger, _logger):
                _l.warning(
                    "[EVAL] stage=l0_cross_tf_shared_context run_id=%s failed: %s,"
                    " falling back to unpruned multi_results",
                    run_id_prefix, exc,
                )
            pruned_multi_results = dict(multi_results)
            manifest_final_ids = tuple(
                c.recipe_id
                for res in multi_results.values()
                for c in res.candidates_for_l1
            )
            _early_reports: dict[str, Any] = {}
            for _tf_k, _res in multi_results.items():
                _r = getattr(_res, "summary_report", None)
                if _r is not None:
                    _early_reports[_tf_k] = _r
            return pruned_multi_results, L0StrategyDeliveryManifest(
                run_id_prefix=run_id_prefix,
                reports_by_tf=_early_reports,
                independence_audit=None,
                final_selected_recipe_ids=manifest_final_ids,
                total_l1_verification_budget=total_l1_verification_budget,
                pruning_status="fail_open",
                pruning_reason=manifest_reason,
            )

    if enable_audit and all_tfs_with_candidates:
        from src.domain.futures.alpha_foundry.diversity import audit_l0_selected_recipe_independence

        _t_audit_start = _time_module.perf_counter()
        _audit_status = "success"
        try:
            manifest_audit = audit_l0_selected_recipe_independence(
                selected_by_tf=selected_by_tf,
                panel_by_recipe_id=panel_by_recipe_id,
                aligned_by_tf=aligned_by_tf,
                min_common_active_bars=min_common_active_bars,
                max_corr=max_novelty_corr,
                precomputed_shared_context=shared_context,
            )
        except ValueError as exc:
            _audit_status = "failed"
            _fallback_logger = setup_logger("opt_main_futures", write_file=False)
            for _l in (_fallback_logger, _logger):
                _l.warning(
                    "[EVAL] stage=l0_cross_tf_audit run_id=%s failed: %s",
                    run_id_prefix, exc,
                )
        finally:
            setup_logger("opt_main_futures", write_file=False).info(
                "[SYS] stage=l0_cross_tf_audit took=%.4fs enabled=True status=%s",
                _time_module.perf_counter() - _t_audit_start, _audit_status,
            )

    if enable_pruning and all_tfs_with_candidates:
        from src.domain.futures.alpha_foundry.diversity import (
            apply_cross_tf_survival_floor,
            compute_cross_tf_redundancy,
        )

        _t_prune_start = _time_module.perf_counter()
        try:
            cross_tf_result = compute_cross_tf_redundancy(
                selected_by_tf=selected_by_tf,
                panel_by_recipe_id=panel_by_recipe_id,
                aligned_by_tf=aligned_by_tf,
                min_common_active_bars=min_common_active_bars,
                max_novelty_corr=max_novelty_corr,
                min_directional_entry_jaccard=min_directional_entry_jaccard,
                min_shared_directional_entries=min_shared_directional_entries,
                precomputed_shared_context=shared_context,
            )

            floor_result = apply_cross_tf_survival_floor(
                cross_tf_result=cross_tf_result,
                candidate_by_recipe_id={
                    c.recipe_id: c
                    for cands in selected_by_tf.values()
                    for c in cands
                    if hasattr(c, "recipe_id")
                },
                min_survivors_per_archetype=min_survivors_per_archetype,
                min_survivors_per_tf=min_survivors_per_tf,
            )

            manifest_final_ids = floor_result.final_selected_recipe_ids

            n_demoted = len(cross_tf_result.demoted_recipe_ids)
            if n_demoted > 0:
                manifest_status = "applied"
                manifest_reason = (
                    f"demoted={n_demoted} canonical_tf={cross_tf_result.canonical_tf}"
                    f" n_common_active_bars={cross_tf_result.n_common_active_bars}"
                )
            else:
                manifest_status = "audit_only"
                manifest_reason = "no redundant pairs found"
            setup_logger("opt_main_futures", write_file=False).info(
                "[SYS] stage=l0_cross_tf_pruning took=%.4fs enabled=True status=%s",
                _time_module.perf_counter() - _t_prune_start, manifest_status,
            )

            final_set = set(manifest_final_ids)
            pruned_multi_results = {}
            for tf_k, res in multi_results.items():
                kept_candidates = tuple(
                    c for c in res.candidates_for_l1
                    if c.recipe_id in final_set
                )
                kept_panels = tuple(
                    p for p in res.panels_for_l1
                    if getattr(p, "metadata", {}).get("recipe_id", "") in final_set
                )
                if len(kept_candidates) == len(res.candidates_for_l1) and len(kept_panels) == len(res.panels_for_l1):
                    pruned_multi_results[tf_k] = res
                else:
                    pruned_multi_results[tf_k] = replace(
                        res,
                        candidates_for_l1=kept_candidates,
                        panels_for_l1=kept_panels,
                    )

        except ValueError as exc:
            manifest_status = "fail_open"
            manifest_reason = str(exc)
            _fallback_logger = setup_logger("opt_main_futures", write_file=False)
            for _l in (_fallback_logger, _logger):
                _l.warning(
                    "[SYS] stage=l0_cross_tf_pruning took=%.4fs enabled=True status=fail_open",
                    _time_module.perf_counter() - _t_prune_start,
                )
                _l.warning(
                    "[EVAL] stage=l0_cross_tf_pruning run_id=%s failed: %s,"
                    " falling back to unpruned multi_results",
                    run_id_prefix, exc,
                )
            pruned_multi_results = dict(multi_results)
            manifest_final_ids = ()

    if not manifest_final_ids:
        manifest_final_ids = tuple(
            c.recipe_id
            for res in multi_results.values()
            for c in res.candidates_for_l1
        )

    reports_by_tf: dict[str, Any] = {}
    for tf_k, res in multi_results.items():
        r = getattr(res, "summary_report", None)
        if r is not None:
            reports_by_tf[tf_k] = r

    manifest = L0StrategyDeliveryManifest(
        run_id_prefix=run_id_prefix,
        reports_by_tf=reports_by_tf,
        independence_audit=manifest_audit,
        final_selected_recipe_ids=manifest_final_ids,
        total_l1_verification_budget=total_l1_verification_budget,
        pruning_status=manifest_status,  # type: ignore[arg-type]
        pruning_reason=manifest_reason,
    )

    return pruned_multi_results, manifest
