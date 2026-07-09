"""Alpha Foundry cross-timeframe corroboration fusion.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.domain.futures.alpha_foundry.contracts import (
    CorroborationTier,
    MultiTimeframeEvidence,
)

_CORROBORATION_BOOST = 1.15
_REJECT_REASON_INSUFFICIENT = ("insufficient_events", "insufficient_effective_n")


def _strip_tf_suffix(variant: str, tf: str) -> str:
    """Strip a trailing '_{tf}' suffix from variant (mirrors
    bridge_helpers._normalize_variant). Panel variants are conventionally
    suffixed with their own timeframe (e.g. 'ema_12_72_8h') — without this,
    the same conceptual signal never groups across timeframes."""
    suffix = f"_{tf}"
    return variant[: -len(suffix)] if variant.endswith(suffix) else variant


def fuse_multi_timeframe_evidence(
    *,
    evidence_by_tf: Mapping[str, pd.DataFrame],
    min_coverage_for_corroboration: int = 2,
    min_sign_agreement_ratio: float = 0.66,
    max_sign_agreement_ratio_for_contradiction: float = 0.50,
) -> tuple[MultiTimeframeEvidence, ...]:
    if not evidence_by_tf:
        return ()

    combined = pd.concat(evidence_by_tf.values(), ignore_index=True)
    combined["variant"] = [
        _strip_tf_suffix(v, tf) for v, tf in zip(combined["variant"], combined["timeframe"], strict=True)
    ]
    results: list[MultiTimeframeEvidence] = []

    for (family, variant), group in combined.groupby(["family", "variant"]):
        for tf, tf_group in group.groupby("timeframe"):
            if len(tf_group) > 1:
                raise ValueError(f"duplicate (family,variant,timeframe) rows: {family}/{variant}/{tf}")
            native_row = tf_group.iloc[0]

            others = group[group.timeframe != tf]
            reject_mask = others["reject_reasons"].str.contains(
                "|".join(_REJECT_REASON_INSUFFICIENT), na=False, regex=True
            )
            covered = others[~reject_mask]
            tf_coverage_count = len(covered)

            if tf_coverage_count == 0:
                tier = "insufficient_coverage"
                sign_agreement_ratio = 0.0
            else:
                native_sign = np.sign(native_row["mean_net_bps"])
                agree_count = (np.sign(covered["mean_net_bps"]) == native_sign).sum()
                sign_agreement_ratio = agree_count / tf_coverage_count

                if (
                    tf_coverage_count >= min_coverage_for_corroboration
                    and sign_agreement_ratio >= min_sign_agreement_ratio
                ):
                    tier = "corroborated"
                elif (
                    tf_coverage_count >= min_coverage_for_corroboration
                    and sign_agreement_ratio <= max_sign_agreement_ratio_for_contradiction
                ):
                    tier = "contradicted"
                else:
                    tier = "single_tf_strict"

            base = float(native_row["block_lcb_bps"])
            if tier == "corroborated":
                fused_conviction_score = base * _CORROBORATION_BOOST
            elif tier == "contradicted":
                fused_conviction_score = -abs(base)
            else:
                fused_conviction_score = base * 1.0

            results.append(
                MultiTimeframeEvidence(
                    family=str(family),
                    variant=str(variant),
                    native_timeframe=str(tf),
                    native_recipe_id=str(native_row["recipe_id"]),
                    tf_coverage_count=tf_coverage_count,
                    sign_agreement_ratio=sign_agreement_ratio,
                    corroboration_tier=cast(CorroborationTier, tier),
                    fused_conviction_score=fused_conviction_score,
                )
            )

    return tuple(results)


def fuse_multi_timeframe_evidence_weighted(
    *,
    evidence_by_tf: Mapping[str, pd.DataFrame],
    min_coverage_mass_for_corroboration: float = 1.0,
    min_partial_coverage_mass: float = 0.25,
    min_sign_agreement_ratio: float = 0.66,
    max_sign_agreement_ratio_for_contradiction: float = 0.50,
) -> tuple[MultiTimeframeEvidence, ...]:
    """Fuse evidence with sample-size weights instead of binary insufficient-sample exclusion."""
    if not evidence_by_tf:
        return ()

    combined = pd.concat(evidence_by_tf.values(), ignore_index=True)
    combined["variant"] = [
        _strip_tf_suffix(v, tf) for v, tf in zip(combined["variant"], combined["timeframe"], strict=True)
    ]
    results: list[MultiTimeframeEvidence] = []

    for (family, variant), group in combined.groupby(["family", "variant"]):
        for tf, tf_group in group.groupby("timeframe"):
            if len(tf_group) > 1:
                raise ValueError(f"duplicate (family,variant,timeframe) rows: {family}/{variant}/{tf}")
            native_row = tf_group.iloc[0]

            others = group[group.timeframe != tf]
            coverage_mass = 0.0
            weighted_agree = 0.0

            if not others.empty:
                n_events_arr = others["n_events"].values.astype(np.float64)
                eff_n_arr = others["effective_n"].values.astype(np.float64)

                weight_n = np.minimum(1.0, n_events_arr / float(combined.get("min_events", 40)))
                weight_eff = np.minimum(1.0, eff_n_arr / float(combined.get("min_effective_n", 20.0)))
                weights = weight_n * weight_eff

                native_sign = np.sign(native_row["mean_net_bps"])
                other_signs = np.sign(others["mean_net_bps"].values.astype(np.float64))
                agrees = (other_signs == native_sign).astype(np.float64)

                coverage_mass = float(np.sum(weights))
                weighted_agree = float(np.sum(weights * agrees)) / coverage_mass if coverage_mass > 0.0 else 0.0

            if coverage_mass >= min_coverage_mass_for_corroboration:
                if weighted_agree >= min_sign_agreement_ratio:
                    tier = "corroborated"
                elif weighted_agree <= max_sign_agreement_ratio_for_contradiction:
                    tier = "contradicted"
                else:
                    tier = "partial_support"
            elif coverage_mass >= min_partial_coverage_mass:
                tier = "partial_support"
            else:
                tier = "insufficient_coverage"

            base = float(native_row["block_lcb_bps"])
            if tier == "corroborated":
                fused_conviction_score = base * _CORROBORATION_BOOST
            elif tier == "contradicted":
                fused_conviction_score = -abs(base)
            else:
                fused_conviction_score = base * 1.0

            results.append(
                MultiTimeframeEvidence(
                    family=str(family),
                    variant=str(variant),
                    native_timeframe=str(tf),
                    native_recipe_id=str(native_row["recipe_id"]),
                    tf_coverage_count=int(coverage_mass),
                    sign_agreement_ratio=weighted_agree,
                    corroboration_tier=cast(CorroborationTier, tier),
                    fused_conviction_score=fused_conviction_score,
                )
            )

    return tuple(results)


def index_multi_timeframe_evidence(
    fusion_rows: Sequence[MultiTimeframeEvidence],
) -> dict[tuple[str, str, str], MultiTimeframeEvidence]:
    result: dict[tuple[str, str, str], MultiTimeframeEvidence] = {}
    for row in fusion_rows:
        key = (row.family, row.variant, row.native_timeframe)
        result[key] = row
    return result
