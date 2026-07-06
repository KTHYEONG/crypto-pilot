"""Alpha Foundry L0-to-L2 orchestration bridge. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    """Run L0 → L1 → L2 foundry pipeline.

    Returns:
        (cheap_gate_evidences, l1_units, posterior_evidences, l2_sleeves)
    """
    cheap_evidences = evaluate_alpha_cheap_gate_batch(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        config=cheap_gate_config,
        regime_code_1d=regime_code_1d,
    )

    l1_units = build_l1_verification_units(
        evidences=cheap_evidences,
        recipes=recipes,
        symbols=symbols,
        top_k_per_family_tf=top_k_per_family_tf,
        initial_fold_budget=initial_fold_budget,
    )

    raw_rows = _build_dummy_fold_rows(cheap_evidences)
    posterior_evidences: tuple[L1PosteriorEvidence, ...] = ()
    if len(raw_rows) > 0:
        posterior_evidences = shrink_l1_evidence_hierarchical(
            raw_rows=raw_rows,
            cost_model=cost_model,
            config=posterior_gate_config,
        )

    l2_sleeves = convert_posterior_to_l2_sleeves(
        posterior=posterior_evidences,
        cost_model=cost_model,
        config=l2_config,
    )

    return cheap_evidences, l1_units, posterior_evidences, l2_sleeves


def _build_dummy_fold_rows(
    evidences: Sequence[CheapGateEvidence],
) -> pd.DataFrame:
    rows = [
        {
            "symbol": "BTCUSDT",
            "recipe_id": ev.recipe_id,
            "family": "foundry",
            "timeframe": ev.timeframe,
            "activation_context": "pooled",
            "net_bps": ev.mean_net_bps,
            "fold_id": 0,
            "effective_weight": 1.0,
        }
        for ev in evidences
    ]
    return pd.DataFrame(rows)
