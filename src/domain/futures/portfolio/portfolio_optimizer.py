"""Portfolio policy clipping and explicit input contracts for strategy sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PortfolioPolicyConfig:
    target_ann_vol: float
    gross_exposure_cap: float
    per_symbol_cap: float
    top_k_long: int
    top_k_short: int
    entry_edge_threshold: float
    rebalance_bars: int
    min_long_pf: float
    min_short_pf: float
    min_is_net_alpha_pct: float


@dataclass(frozen=True)
class PortfolioPolicyInputs:
    """Explicit portfolio policy input contract.

    Attributes:
        mu_long_2d: Expected long return tensor, shape [T, N], per-bar simple return.
        mu_short_2d: Expected short return tensor, shape [T, N], per-bar simple return.
        risk_sigma_3d: Optional covariance cube shape [T, N, N].
        risk_beta_2d: Optional beta matrix shape [T, N].
        risk_residual_var_2d: Optional residual variance matrix shape [T, N].
        cost_fraction_2d: Optional execution cost tensor in fraction units, shape [T, N].
        cost_bps_2d: Optional execution cost tensor in bps units, shape [T, N].
        cost_source: Optional canonical cost source metadata.
        capacity_notional_2d: Optional per-bar capacity tensor shape [T, N].

    """

    mu_long_2d: np.ndarray | None = None
    mu_short_2d: np.ndarray | None = None
    risk_sigma_3d: np.ndarray | None = None
    risk_beta_2d: np.ndarray | None = None
    risk_residual_var_2d: np.ndarray | None = None
    cost_fraction_2d: np.ndarray | None = None
    cost_bps_2d: np.ndarray | None = None
    cost_source: str | None = None
    capacity_notional_2d: np.ndarray | None = None


def load_portfolio_policy_config(cfg: dict[str, Any]) -> PortfolioPolicyConfig:
    block = cfg.get("FUTURES_PORTFOLIO_POLICY", {})
    return PortfolioPolicyConfig(
        target_ann_vol=float(block.get("target_ann_vol", 0.45)),
        gross_exposure_cap=float(block.get("gross_exposure_cap", 1.2)),
        per_symbol_cap=float(block.get("per_symbol_cap", 0.25)),
        top_k_long=int(block.get("top_k_long", 3)),
        top_k_short=int(block.get("top_k_short", 3)),
        entry_edge_threshold=float(block.get("entry_edge_threshold", 0.15)),
        rebalance_bars=int(block.get("rebalance_bars", 3)),
        min_long_pf=float(block.get("min_long_pf", 1.05)),
        min_short_pf=float(block.get("min_short_pf", 1.05)),
        min_is_net_alpha_pct=float(block.get("min_is_net_alpha_pct", 0.0)),
    )


def apply_policy_constraints(
    params: dict[str, Any],
    policy: PortfolioPolicyConfig,
) -> dict[str, Any]:
    out = dict(params)
    out["K_LONG"] = int(max(1, min(int(out.get("K_LONG", policy.top_k_long)), policy.top_k_long)))
    out["K_SHORT"] = int(
        max(1, min(int(out.get("K_SHORT", policy.top_k_short)), policy.top_k_short))
    )
    out["REBALANCE_BARS"] = int(max(1, out.get("REBALANCE_BARS", policy.rebalance_bars)))
    out["MAX_EXPOSURE_PER_COIN"] = float(
        min(
            max(float(out.get("MAX_EXPOSURE_PER_COIN", policy.per_symbol_cap)), 0.05),
            policy.per_symbol_cap,
        )
    )
    out["MAX_EXPOSURE"] = float(
        min(
            max(float(out.get("MAX_EXPOSURE", policy.gross_exposure_cap)), 0.20),
            policy.gross_exposure_cap,
        )
    )
    out["TARGET_ANN_VOL"] = float(
        min(
            max(float(out.get("TARGET_ANN_VOL", policy.target_ann_vol)), 0.05),
            policy.target_ann_vol,
        )
    )
    return out


def finalize_strategy_portfolio_params(
    raw_params: dict[str, Any],
    policy: PortfolioPolicyConfig | None = None,
    *,
    futures_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single funnel for applying policy caps to trial/deploy params."""
    pol = (
        policy
        if policy is not None
        else load_portfolio_policy_config(futures_config if futures_config is not None else {})
    )
    return apply_policy_constraints(dict(raw_params), pol)
