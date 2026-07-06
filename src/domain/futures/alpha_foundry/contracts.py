"""Alpha Foundry contracts and stage state machines. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

AlphaArchetype: TypeAlias = Literal[
    "trend",
    "mean_reversion",
    "carry",
    "flow",
    "cross_sectional",
    "hedge",
]

SymbolScope: TypeAlias = Literal["global", "cluster", "symbol"]

ActivationContract: TypeAlias = Literal["hard", "soft", "observe"]

CheapGateRejectReason: TypeAlias = Literal[
    "insufficient_events",
    "insufficient_effective_n",
    "non_positive_lcb",
    "weak_tstat",
    "excess_cost_drag",
    "excess_turnover",
    "duplicate_signal",
    "invalid_shape",
    "lookahead_risk",
    "missing_required_field",
]


def _validate_non_empty_keys(mapping: Mapping[str, object], name: str) -> None:
    if not mapping:
        raise ValueError(f"{name} must not be empty")


@dataclass(slots=True, frozen=True)
class AlphaRecipe:
    recipe_id: str
    family: str
    variant: str
    timeframe: str
    archetype: AlphaArchetype
    indicator_params: Mapping[str, float | int | str]
    side_rule_id: str
    exit_policy_id: str
    required_fields: tuple[str, ...]
    causal_lag_bars: int
    max_turnover_per_year: float

    def __post_init__(self) -> None:
        if self.causal_lag_bars < 1:
            raise ValueError("causal_lag_bars must be >= 1")
        if self.max_turnover_per_year < 0.0:
            raise ValueError("max_turnover_per_year must be non-negative")
        if not self.recipe_id:
            raise ValueError("recipe_id must not be empty")
        if not self.family:
            raise ValueError("family must not be empty")
        if not self.variant:
            raise ValueError("variant must not be empty")


@dataclass(slots=True, frozen=True)
class CheapGateConfig:
    min_events: int = 40
    min_effective_n: float = 20.0
    min_lcb_net_bps: float = 0.0
    min_nw_tstat: float = 1.25
    max_cost_drag_ratio: float = 0.60
    max_turnover_per_year: float = 365.0
    max_novelty_corr: float = 0.85
    min_incremental_rank_ic: float = 0.01
    block_bars: int = 6
    bootstrap_samples: int = 200
    fdr_alpha: float = 0.10


@dataclass(slots=True, frozen=True)
class CheapGateEvidence:
    recipe_id: str
    timeframe: str
    symbol_scope: SymbolScope
    n_events: int
    effective_n: float
    mean_net_bps: float
    nw_tstat: float
    block_lcb_bps: float
    rank_ic: float
    monotonic_bucket_score: float
    regime_edges_bps: Mapping[str, float]
    cost_drag_ratio: float
    turnover_per_year: float
    novelty_corr_max: float
    incremental_rank_ic: float
    compute_cost_score: float
    gate_passed: bool
    reject_reasons: tuple[CheapGateRejectReason, ...]


@dataclass(slots=True, frozen=True)
class L1VerificationUnit:
    unit_id: str
    recipe_id: str
    timeframe: str
    scope_symbols: tuple[str, ...]
    prior_mu_bps: float
    prior_sigma_bps: float
    allocated_fold_budget: int
    early_stop_state: Literal["pending", "drop", "promote", "continue"]


@dataclass(slots=True, frozen=True)
class L1PosteriorEvidence:
    symbol: str
    recipe_id: str
    family: str
    timeframe: str
    activation_context: str
    posterior_mu_bps: float
    posterior_sigma_bps: float
    prob_mu_gt_cost: float
    lcb_net_bps: float
    q_value: float
    fold_pass_ratio: float
    regime_stability: float
    quality_weight: float
    activation_contract: ActivationContract


@dataclass(slots=True, frozen=True)
class PosteriorGateConfig:
    prior_effective_n: float = 30.0
    promote_prob_min: float = 0.70
    drop_prob_max: float = 0.45
    min_lcb_net_bps: float = 0.0
    min_fold_pass_ratio: float = 0.50
    min_regime_stability: float = 0.0
    fdr_alpha: float = 0.10


@dataclass(slots=True, frozen=True)
class L2PosteriorPolicyConfig:
    k_rank: int = 3
    rebalance_bars: int = 3
    kelly_fraction: float = 0.25
    posterior_z: float = 0.50
    risk_budget_target: float = 0.50
    gross_cap_by_regime: Mapping[str, float] = field(
        default_factory=lambda: {"bull": 1.0, "bear": 0.35, "crisis": 0.25}
    )
    cov_mode: Literal["diagonal", "shrinkage"] = "diagonal"
    cost_safety_mult: float = 1.25
    turnover_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.kelly_fraction <= 0.0:
            raise ValueError("kelly_fraction must be positive")
        if self.cost_safety_mult < 1.0:
            raise ValueError("cost_safety_mult must be >= 1.0")
        for regime, cap in self.gross_cap_by_regime.items():
            if not (0.0 <= cap <= 1.0):
                raise ValueError(f"regime cap for '{regime}' must be in [0, 1], got {cap}")


@dataclass(slots=True, frozen=True)
class L2PosteriorSleeve:
    symbol: str
    recipe_id: str
    family: str
    timeframe: str
    activation_context: str
    mu_eff_bps: float
    sigma_bps: float
    quality_weight: float
    side: Literal[-1, 0, 1]
    disabled_reason: str = ""


@dataclass(slots=True, frozen=True)
class StagedSearchBudget:
    stage: Literal["signal", "risk", "regime", "deployment"]
    n_trials: int
    min_feasible_eff: float
    patience: int
    seed_count: int

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
