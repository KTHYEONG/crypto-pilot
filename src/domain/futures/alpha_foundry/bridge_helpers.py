"""Alpha Foundry L0 gate bridge helpers for the strategy runtime bridge. [ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING]"""

from __future__ import annotations

import json
import time as _time_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    pass


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
    recipes: Mapping[str, Any],
    timeframe: str,
    max_recipes_per_family: int,
    include_families: tuple[str, ...],
    exclude_families: tuple[str, ...],
) -> tuple[Any, ...]:
    from src.domain.futures.alpha_foundry.contracts import (
        PanelRecipeBinding as _PanelRecipeBinding,
    )

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
            continue

        src = cast(
            Literal["catalog_exact", "catalog_family_variant", "synthetic_recipe"],
            matched_source,
        )
        bindings.append(_PanelRecipeBinding(
            panel_index=i,
            recipe_id=matched_recipe_id,
            family=family,
            variant=variant,
            source=src,
        ))

    return tuple(bindings)


def _write_alpha_foundry_report(
    report: Any,
    report_dir: Path,
    run_id: str,
) -> tuple[str, str]:
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
    from src.domain.futures.alpha_foundry.cheap_gate import (
        evaluate_alpha_cheap_gate_batch,
    )
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

    if cheap_config is not None and bound_panels:
        evidences = evaluate_alpha_cheap_gate_batch(
            panels=bound_panels,
            recipes=recipes,
            aligned=aligned,
            cost_model=cost_model,
            config=cheap_config,
        )
    else:
        evidences = ()

    passed_ids = {ev.recipe_id for ev in evidences if ev.gate_passed}
    reject_reason_counts: dict[str, int] = {}
    for ev in evidences:
        for reason in ev.reject_reasons:
            reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

    if mode == "gate":
        passed_panel_indices = {
            b.panel_index
            for b in bindings
            if b.recipe_id in passed_ids
        }
        panels_for_l1 = tuple(
            p for i, p in enumerate(panels) if i in passed_panel_indices
        )
    else:
        panels_for_l1 = tuple(panels)

    elapsed_sec = _time_module.perf_counter() - l0_start

    n_panels_in = len(panels)
    n_bound = len(bindings)
    n_evidence = len(evidences)
    n_passed = len(passed_ids)
    n_rejected = n_evidence - n_passed

    symbols = aligned.symbols if hasattr(aligned, "symbols") else ()
    n_bars = aligned.close_2d.shape[0] if hasattr(aligned, "close_2d") else 0

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

    try:
        report_dir = (
            runtime_config.report_dir
            if hasattr(runtime_config, "report_dir")
            else Path("logs/futures/alpha_foundry")
        )
        json_path, parquet_path = _write_alpha_foundry_report(
            report, report_dir, run_id,
        )
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
    except OSError:
        raise

    return AlphaFoundryL0Result(
        panels_for_l1=panels_for_l1,
        report=report,
        evidences=evidences,
        bindings=tuple(bindings),
    )
