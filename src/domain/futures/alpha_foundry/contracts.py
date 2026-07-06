"""Alpha Foundry contracts and stage state machines.

[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

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
    "invalid_shape",
    "lookahead_risk",
    "missing_required_field",
]

BucketKey: TypeAlias = tuple[str, str]  # (family, timeframe)


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
    bootstrap_seed: int = 42
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
    cost_drag_ratio: float
    turnover_per_year: float
    novelty_corr_max: float
    incremental_rank_ic: float
    compute_cost_score: float
    bootstrap_lcb_bps: float
    bootstrap_agree: bool
    gate_passed: bool
    reject_reasons: tuple[CheapGateRejectReason, ...]


@dataclass(slots=True, frozen=True)
class DiversitySelectionResult:
    bucket_key: BucketKey
    ranked_recipe_ids: tuple[str, ...]
    selected_recipe_ids: tuple[str, ...]
    redundant_recipe_ids: tuple[str, ...]
    redundant_reason_by_id: Mapping[str, str]  # recipe_id -> 최대상관 상대 recipe_id
    bucket_corr: NDArray[np.float64]
    bucket_eff_test_count: float


@dataclass(slots=True, frozen=True)
class CrossBucketDiversityResult:
    final_selected_recipe_ids: tuple[str, ...]
    demoted_recipe_ids: tuple[str, ...]  # Stage2 selected였으나 Stage3에서 강등
    demoted_reason_by_id: Mapping[str, str]
    cross_bucket_corr: NDArray[np.float64]
    global_eff_test_count: float


@dataclass(slots=True, frozen=True)
class AlphaFoundryEvidenceRow:
    run_id: str
    timeframe: str
    family: str
    variant: str
    recipe_id: str
    archetype: str
    n_events: int
    effective_n: float
    mean_net_bps: float
    nw_tstat: float
    block_lcb_bps: float
    rank_ic: float
    incremental_rank_ic: float
    cost_drag_ratio: float
    turnover_per_year: float
    compute_cost_score: float
    bootstrap_lcb_bps: float
    bootstrap_agree: bool
    gate_passed: bool
    reject_reasons: str  # "|".join(reject_reasons)
    bucket_key: str  # f"{family}:{timeframe}"
    bucket_rank: int
    selected_for_l1: bool
    redundant_with: str  # "" if none
    bucket_eff_test_count: float
    global_eff_test_count: float
    created_at_ms: int


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


AlphaFoundryMode: TypeAlias = Literal["off", "audit", "gate"]


@dataclass(slots=True, frozen=True)
class AlphaFoundryRuntimeConfig:
    mode: AlphaFoundryMode = "off"
    report_dir: Path = Path("logs/futures/alpha_foundry")
    max_recipes_per_family: int = 64
    include_families: tuple[str, ...] = ()
    exclude_families: tuple[str, ...] = ()
    top_k_per_family_tf: int = 5
    initial_fold_budget: int = 3
    enable_synthetic_recipes: bool = True
    min_conviction_lcb_bps: float = 5.0
    total_l1_verification_budget: int = 30
    cheap_gate: CheapGateConfig = field(default_factory=CheapGateConfig)
    posterior_gate: PosteriorGateConfig = field(default_factory=PosteriorGateConfig)
    l2_policy: L2PosteriorPolicyConfig = field(default_factory=L2PosteriorPolicyConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"off", "audit", "gate"}:
            raise ValueError(f"invalid alpha_foundry mode: {self.mode!r}")
        if self.max_recipes_per_family < 1:
            raise ValueError(f"max_recipes_per_family must be >= 1, got {self.max_recipes_per_family}")
        if self.top_k_per_family_tf < 1:
            raise ValueError(f"top_k_per_family_tf must be >= 1, got {self.top_k_per_family_tf}")
        if self.initial_fold_budget < 1:
            raise ValueError(f"initial_fold_budget must be >= 1, got {self.initial_fold_budget}")
        for f in self.include_families:
            if not f.strip():
                raise ValueError("include_families contains empty string after trim")
        for f in self.exclude_families:
            if not f.strip():
                raise ValueError("exclude_families contains empty string after trim")
        if self.min_conviction_lcb_bps < 0.0:
            raise ValueError(f"min_conviction_lcb_bps must be >= 0.0, got {self.min_conviction_lcb_bps}")
        if self.total_l1_verification_budget < 1:
            raise ValueError(f"total_l1_verification_budget must be >= 1, got {self.total_l1_verification_budget}")


@dataclass(slots=True, frozen=True)
class PanelRecipeBinding:
    panel_index: int
    recipe_id: str
    family: str
    variant: str
    source: Literal["catalog_exact", "catalog_family_variant", "synthetic_recipe"]


@dataclass(slots=True, frozen=True)
class AlphaFoundryBridgeReport:
    run_id: str
    mode: Literal["audit", "gate"]
    timeframe: str
    symbols: tuple[str, ...]
    n_bars: int
    n_panels_in: int
    n_bound_panels: int
    n_evidence: int
    n_passed: int
    n_rejected: int
    reject_reason_counts: dict[str, int]
    elapsed_sec: float
    json_path: str
    parquet_path: str


@dataclass(slots=True, frozen=True)
class AlphaFoundryFoldRow:
    symbol: str
    recipe_id: str
    family: str
    timeframe: str
    activation_context: str
    net_bps: float
    fold_id: int
    effective_weight: float


CorroborationTier: TypeAlias = Literal[
    "corroborated",
    "single_tf_strict",
    "contradicted",
    "insufficient_coverage",
]


@dataclass(slots=True, frozen=True)
class MultiTimeframeEvidence:
    family: str
    variant: str
    native_timeframe: str
    native_recipe_id: str
    tf_coverage_count: int
    sign_agreement_ratio: float
    corroboration_tier: CorroborationTier
    fused_conviction_score: float
