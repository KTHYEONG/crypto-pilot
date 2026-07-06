"""Alpha Foundry L0-to-L2 orchestration bridge. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.budget import build_l1_verification_units
from src.domain.futures.alpha_foundry.cheap_gate import (
    evaluate_alpha_cheap_gate_batch,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateConfig,
    CheapGateEvidence,
    L1PosteriorEvidence,
    L1VerificationUnit,
    L2PosteriorPolicyConfig,
    L2PosteriorSleeve,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.l2_policy import (
    convert_posterior_to_l2_sleeves,
)
from src.domain.futures.alpha_foundry.posterior import shrink_l1_evidence_hierarchical
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


@dataclass(slots=True, frozen=True)
class AlphaFoundryL0Artifacts:
    evidences: tuple[CheapGateEvidence, ...]
    passed_recipe_ids: tuple[str, ...]
    reject_reason_counts: dict[str, int]


def run_alpha_foundry_l0_pipeline(
    *,
    panels: Sequence[CandidateSignalPanel],
    recipes: Mapping[str, AlphaRecipe],
    aligned: AlignedMarketData,
    cost_model: ExecutionCostModel,
    cheap_gate_config: CheapGateConfig,
    regime_code_1d: NDArray[np.int8] | None = None,
) -> AlphaFoundryL0Artifacts:
    cheap_evidences = evaluate_alpha_cheap_gate_batch(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        config=cheap_gate_config,
        regime_code_1d=regime_code_1d,
    )
    passed_ids = tuple(
        ev.recipe_id for ev in cheap_evidences if ev.gate_passed
    )
    reject_reason_counts: dict[str, int] = {}
    for ev in cheap_evidences:
        for reason in ev.reject_reasons:
            reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1
    return AlphaFoundryL0Artifacts(
        evidences=cheap_evidences,
        passed_recipe_ids=passed_ids,
        reject_reason_counts=reject_reason_counts,
    )


def build_posterior_from_l1_fold_rows(
    *,
    raw_rows: pd.DataFrame,
    cost_model: ExecutionCostModel,
    config: PosteriorGateConfig,
) -> tuple[L1PosteriorEvidence, ...]:
    if not isinstance(raw_rows, pd.DataFrame):
        raise ValueError("raw_rows must be a DataFrame")
    required = {"net_bps", "symbol"}
    missing = required - set(raw_rows.columns)
    if missing:
        raise ValueError(f"raw_rows missing required columns: {missing}")
    if raw_rows.empty:
        return ()
    return shrink_l1_evidence_hierarchical(
        raw_rows=raw_rows,
        cost_model=cost_model,
        config=config,
    )


def build_l2_sleeves_from_posterior(
    *,
    posterior: Sequence[L1PosteriorEvidence],
    cost_model: ExecutionCostModel,
    config: L2PosteriorPolicyConfig,
) -> tuple[L2PosteriorSleeve, ...]:
    return convert_posterior_to_l2_sleeves(
        posterior=posterior,
        cost_model=cost_model,
        config=config,
    )


def run_alpha_foundry_pipeline(
    *,
    panels: Sequence[CandidateSignalPanel],
    recipes: Mapping[str, AlphaRecipe],
    aligned: AlignedMarketData,
    cost_model: ExecutionCostModel,
    cheap_gate_config: CheapGateConfig,
    posterior_gate_config: PosteriorGateConfig,
    l2_config: L2PosteriorPolicyConfig,
    symbols: tuple[str, ...],
    top_k_per_family_tf: int = 5,
    initial_fold_budget: int = 3,
    regime_code_1d: NDArray[np.int8] | None = None,
) -> tuple[
    tuple[CheapGateEvidence, ...],
    tuple[L1VerificationUnit, ...],
    tuple[L1PosteriorEvidence, ...],
    tuple[L2PosteriorSleeve, ...],
]:
    """Legacy wrapper — prefer the split API (L0 / posterior / L2)."""
    l0 = run_alpha_foundry_l0_pipeline(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        cheap_gate_config=cheap_gate_config,
        regime_code_1d=regime_code_1d,
    )

    l1_units = build_l1_verification_units(
        evidences=l0.evidences,
        recipes=recipes,
        symbols=symbols,
        top_k_per_family_tf=top_k_per_family_tf,
        initial_fold_budget=initial_fold_budget,
    )

    l1_posterior: tuple[L1PosteriorEvidence, ...] = ()
    l2_sleeves: tuple[L2PosteriorSleeve, ...] = ()

    return l0.evidences, l1_units, l1_posterior, l2_sleeves
