"""Alpha Foundry L0 gate bridge helpers for the strategy runtime bridge.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
[ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

import json
import time as _time_module
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    pass

from src.domain.futures.alpha_foundry.contracts import AlphaRecipe, CandidateFeatureFamily
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe as _AlphaRecipe,
)
from src.domain.futures.alpha_foundry.contracts import (
    PanelRecipeBinding as _PanelRecipeBinding,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel


@dataclass(slots=True, frozen=True)
class AlphaFoundryL0Result:
    panels_for_l1: tuple[Any, ...]
    report: Any | None
    evidences: tuple[Any, ...]
    bindings: tuple[Any, ...]


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
        from dataclasses import asdict

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
) -> AlphaFoundryL0Result:
    from src.domain.futures.alpha_foundry.contracts import (
        AlphaFoundryBridgeReport,
    )

    raw_mode = runtime_config.mode if hasattr(runtime_config, "mode") else "off"
    if raw_mode == "off":
        return AlphaFoundryL0Result(
            panels_for_l1=tuple(panels),
            report=None,
            evidences=(),
            bindings=tuple(bindings),
        )

    mode: Literal["audit", "gate"] = cast(Literal["audit", "gate"], raw_mode)

    binding_by_index = {b.panel_index: b for b in bindings}
    bound_panel_indices = set(binding_by_index)
    bound_panels = [
        replace(
            panel,
            metadata={
                **dict(getattr(panel, "metadata", {}) or {}),
                "recipe_id": binding_by_index[i].recipe_id,
            },
        )
        for i, panel in enumerate(panels)
        if i in bound_panel_indices
    ]

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
    n_passed = len(l0_artifacts.passed_recipe_ids) if l0_artifacts is not None else 0
    n_rejected = n_evidence - n_passed

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
            json_path=json_path,
            parquet_path=parquet_path,
        )

    return AlphaFoundryL0Result(
        panels_for_l1=panels_for_l1,
        report=report,
        evidences=evidences,
        bindings=tuple(bindings),
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
        # DEBUG mode: log only, no file write
        if observability == "debug_log":
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                "[REPORT] stage=alpha_foundry run_id=%s mode=%s n_evidence=%d n_passed=%d",
                getattr(report, "run_id", ""),
                getattr(report, "mode", ""),
                getattr(report, "n_evidence", 0),
                getattr(report, "n_passed", 0),
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
