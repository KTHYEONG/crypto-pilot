"""Alpha Foundry hierarchical posterior shrinkage. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.domain.futures.alpha_foundry.contracts import (
    ActivationContract,
    L1PosteriorEvidence,
    PosteriorGateConfig,
)
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_REQUIRED_COLS: tuple[str, ...] = (
    "symbol",
    "recipe_id",
    "family",
    "timeframe",
    "activation_context",
    "net_bps",
    "fold_id",
    "effective_weight",
)


def shrink_l1_evidence_hierarchical(
    *,
    raw_rows: pd.DataFrame,
    cost_model: ExecutionCostModel,
    config: PosteriorGateConfig,
) -> tuple[L1PosteriorEvidence, ...]:
    for col in _REQUIRED_COLS:
        if col not in raw_rows.columns:
            raise ValueError(f"missing required column: {col}")

    results: list[L1PosteriorEvidence] = []
    cost_bps = cost_model.stress_round_trip_bps()

    groups = raw_rows.groupby(["family", "timeframe", "activation_context"], dropna=False)
    family_stats: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, grp in groups:
        mu = float(grp["net_bps"].mean())
        sigma = float(grp["net_bps"].std()) if len(grp) > 1 else 1.0
        family_stats[key] = (mu, sigma)

    symbol_groups: pd.core.groupby.DataFrameGroupBy = raw_rows.groupby(
        ["symbol", "recipe_id", "family", "timeframe", "activation_context"], dropna=False
    )
    for key, grp in symbol_groups:
        symbol, recipe_id, family, timeframe, activation_context = key
        cell_mu = float(grp["net_bps"].mean())
        n_eff_cell = float(grp["effective_weight"].sum())

        family_key = (family, timeframe, activation_context)
        fam_mu, fam_sigma = family_stats.get(family_key, (0.0, 1.0))

        w = n_eff_cell / (n_eff_cell + config.prior_effective_n)
        mu_post = w * cell_mu + (1 - w) * fam_mu
        sigma_post = np.sqrt(
            w ** 2 * max(grp["net_bps"].std(), 1e-6) ** 2
            + (1 - w) ** 2 * fam_sigma ** 2
        )

        prob_mu_gt_cost = 1.0 - float(scipy_stats.norm.cdf(
            (cost_bps - mu_post) / max(sigma_post, 1e-10)
        ))

        lcb_net_bps = mu_post - 1.0 * sigma_post
        fold_pass_ratio = float(
            (grp["net_bps"] > cost_bps).mean()
        )
        regime_stability = 0.0
        q_value = 1.0 - prob_mu_gt_cost
        quality_weight = max(0.0, min(1.0, 2.0 * prob_mu_gt_cost - 1.0)) * fold_pass_ratio

        if lcb_net_bps <= cost_bps:
            activation_contract: ActivationContract = "observe"
            quality_weight = 0.0
        elif prob_mu_gt_cost >= config.promote_prob_min:
            activation_contract = "hard"
        elif prob_mu_gt_cost >= config.drop_prob_max:
            activation_contract = "soft"
        else:
            activation_contract = "observe"
            quality_weight = 0.0

        results.append(L1PosteriorEvidence(
            symbol=symbol,
            recipe_id=recipe_id,
            family=family,
            timeframe=timeframe,
            activation_context=activation_context,
            posterior_mu_bps=float(mu_post),
            posterior_sigma_bps=float(sigma_post),
            prob_mu_gt_cost=float(prob_mu_gt_cost),
            lcb_net_bps=float(lcb_net_bps),
            q_value=float(q_value),
            fold_pass_ratio=fold_pass_ratio,
            regime_stability=regime_stability,
            quality_weight=quality_weight,
            activation_contract=activation_contract,
        ))
    return tuple(results)
