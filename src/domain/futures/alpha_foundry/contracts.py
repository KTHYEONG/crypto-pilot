"""Alpha Foundry contracts and stage state machines.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
[ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.signals.contracts import CandidateSignalPanel

# ── Alpha Gate type aliases ────────────────────────────────────────────
CandidateFeatureFamily: TypeAlias = Literal[
    "price_structure",
    "positioning",
    "flow",
    "liquidity",
    "relative_value",
    "regime",
]

FeatureDirectionRule: TypeAlias = Literal[
    "trend_follow",
    "fade_crowding",
    "breakout_retest",
    "xs_neutral",
    "carry_filter",
]

SearchRetireReason: TypeAlias = Literal[
    "no_generator",
    "missing_required_field",
    "cost_prior_failed",
    "repeated_hard_reject",
    "budget_exhausted",
]

AlphaGateSchema: TypeAlias = Literal["unified"]
AlphaGateObservabilityMode: TypeAlias = Literal["debug_log", "artifact", "off"]
AlphaGateHandoffTier: TypeAlias = Literal["blocked", "seed", "candidate"]

AlphaEntryMode: TypeAlias = Literal["sparse", "continuous", "cross_sectional_rank"]
AlphaSearchStatus: TypeAlias = Literal["pending", "screened", "gated", "l1_queued", "retired"]
AlphaTimeframe: TypeAlias = Literal["30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d"]

# ── L0/L1 Signal Discovery types ──────────────────────────────────────
DiscoveryTier: TypeAlias = Literal["seed", "candidate", "verified", "blocked"]

L0HardRejectReason: TypeAlias = Literal[
    "insufficient_events",
    "insufficient_effective_n",
    "excess_cost_drag",
    "excess_turnover",
    "invalid_shape",
    "lookahead_risk",
    "missing_required_field",
    "deep_negative_lcb",
    "tf_contradicted",
]

L0SoftFlag: TypeAlias = Literal[
    "weak_tstat",
    "fdr_rejected",
    "bootstrap_disagree",
    "below_conviction_floor",
    "insufficient_tf_coverage",
    "high_bucket_corr",
    "weak_rank_ic",
]

L0HandoffExclusionReason: TypeAlias = Literal[
    "",
    "hard_reject",
    "bucket_redundant",
    "cross_bucket_redundant",
    "budget_exhausted",
    "non_positive_priority",
    "missing_panel",
]


@dataclass(slots=True, frozen=True)
class L0HandoffDecision:
    recipe_id: str
    bucket_key: BucketKey
    candidate_tier: DiscoveryTier
    eligible_for_diversity: bool
    eligible_for_budget: bool
    selected_for_l1: bool
    budget_units: int
    exclusion_reason: L0HandoffExclusionReason


@dataclass(slots=True, frozen=True)
class AlphaHypothesis:
    hypothesis_id: str
    family: str
    variant: str
    archetype: AlphaArchetype
    timeframe: AlphaTimeframe
    data_scope: tuple[str, ...]
    entry_mode: AlphaEntryMode
    causal_lag_bars: int
    holding_bars: int
    turnover_budget_per_year: float
    prior_score: float
    status: AlphaSearchStatus = "pending"

    def __post_init__(self) -> None:
        if self.causal_lag_bars < 1:
            raise ValueError("causal_lag_bars must be >= 1")
        if self.holding_bars < 1:
            raise ValueError("holding_bars must be >= 1")
        if self.turnover_budget_per_year < 0.0:
            raise ValueError("turnover_budget_per_year must be >= 0.0")
        if not np.isfinite(self.prior_score):
            raise ValueError("prior_score must be finite")


@dataclass(slots=True, frozen=True)
class AlphaFeatureBlueprint:
    blueprint_id: str
    hypothesis_id: str
    feature_family: CandidateFeatureFamily
    lookback_bars: tuple[int, ...]
    thresholds: Mapping[str, float]
    direction_rule: FeatureDirectionRule
    required_fields: tuple[str, ...]
    validity_mask_name: str
    max_compute_cost_score: float

    def __post_init__(self) -> None:
        if not self.blueprint_id:
            raise ValueError("blueprint_id must not be empty")
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must not be empty")
        for lb in self.lookback_bars:
            if lb < 1:
                raise ValueError("lookback_bars must be >= 1")
        if self.max_compute_cost_score <= 0.0:
            raise ValueError("max_compute_cost_score must be > 0.0")


@dataclass(slots=True, frozen=True)
class AlphaSearchPolicyState:
    family: str
    timeframe: str
    tested_count: int = 0
    survivor_count: int = 0
    retired_count: int = 0
    posterior_pass_rate: float = 0.0
    posterior_edge_bps: float = 0.0
    next_budget: int = 1
    retire_reason: SearchRetireReason | None = None

    def __post_init__(self) -> None:
        if self.tested_count < 0:
            raise ValueError("tested_count must be >= 0")
        if self.survivor_count < 0:
            raise ValueError("survivor_count must be >= 0")
        if self.retired_count < 0:
            raise ValueError("retired_count must be >= 0")
        if self.next_budget < 1:
            raise ValueError("next_budget must be >= 1")
        if not (0.0 <= self.posterior_pass_rate <= 1.0):
            raise ValueError("posterior_pass_rate must be in [0.0, 1.0]")


@dataclass(slots=True, frozen=True)
class FamilyTimeframeGatePolicy:
    archetype: AlphaArchetype
    min_events: int
    min_effective_n: float
    target_effective_n: float
    max_cost_drag_ratio: float
    max_turnover_per_year: float
    deep_negative_lcb_bps: float
    min_seed_slots: int = 1


@dataclass(slots=True, frozen=True)
class L0PriorityWeights:
    edge_mean_weight: float = 0.25
    corroborated_multiplier: float = 1.15
    single_tf_multiplier: float = 1.00
    insufficient_coverage_multiplier: float = 0.70
    contradicted_multiplier: float = 0.00
    corr_soft_floor: float = 0.85
    weak_rank_ic_multiplier: float = 0.70


@dataclass(slots=True, frozen=True)
class L0SignalCandidate:
    run_id: str
    timeframe: str
    family: str
    variant: str
    recipe_id: str
    archetype: AlphaArchetype
    source: Literal["catalog_exact", "catalog_family_variant", "synthetic_recipe"]
    n_events: int
    effective_n: float
    mean_net_bps: float
    block_lcb_bps: float
    nw_tstat: float
    bootstrap_lcb_bps: float
    bootstrap_agree: bool
    cost_drag_ratio: float
    turnover_per_year: float
    max_abs_corr_in_bucket: float
    tf_coverage_count: int
    sign_agreement_ratio: float
    corroboration_tier: Literal[
        "corroborated",
        "single_tf_strict",
        "contradicted",
        "insufficient_coverage",
    ]
    discovery_tier: DiscoveryTier
    l1_priority_score: float
    l1_budget_units: int
    hard_reject_reasons: tuple[L0HardRejectReason, ...]
    soft_flags: tuple[L0SoftFlag, ...]


@dataclass(slots=True, frozen=True)
class L0BucketBudget:
    bucket_key: BucketKey
    archetype: AlphaArchetype
    candidate_count: int
    selected_count: int
    min_seed_slots: int
    max_slots: int
    allocated_slots: int
    bucket_quality: float
    effective_test_count: float


@dataclass(slots=True, frozen=True)
class L0ReportStageCounts:
    hard_reject: int
    soft_reject: int
    seeded: int
    budget_exhausted: int
    tf_contradicted: int
    l1_queued: int
    viable_candidates: int = 0


@dataclass(slots=True, frozen=True)
class L0L1Handoff:
    panels_for_l1: tuple[CandidateSignalPanel, ...]
    candidates: tuple[L0SignalCandidate, ...]
    budget_by_bucket: dict[BucketKey, int]
    audit_rows_path: str
    mode: Literal["audit", "gate"]


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
class AlphaSignalBlueprint:
    family: str
    variant: str
    archetype: AlphaArchetype
    timeframe: str
    required_fields: tuple[str, ...]
    causal_lag_bars: int
    lookback_bars: tuple[int, ...]
    holding_bars: int
    max_turnover_per_year: float
    entry_mode: AlphaEntryMode
    side_rule_id: str
    exit_policy_id: str

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("family must not be empty")
        if not self.variant:
            raise ValueError("variant must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if not self.side_rule_id:
            raise ValueError("side_rule_id must not be empty")
        if not self.exit_policy_id:
            raise ValueError("exit_policy_id must not be empty")
        if self.causal_lag_bars < 1:
            raise ValueError("causal_lag_bars must be >= 1")
        if self.holding_bars < 1:
            raise ValueError("holding_bars must be >= 1")
        for lb in self.lookback_bars:
            if lb < 1:
                raise ValueError("lookback_bars must be >= 1")
        if self.max_turnover_per_year < 0.0:
            raise ValueError("max_turnover_per_year must be >= 0.0")
        if self.entry_mode == "continuous" and self.max_turnover_per_year > 365.0:
            raise ValueError("continuous mode requires max_turnover_per_year <= 365.0")


@dataclass(slots=True, frozen=True)
class L0SearchCell:
    blueprint_id: str
    family: str
    variant: str
    timeframe: str
    tf_minutes: int
    symbol_scope: SymbolScope
    cost_floor_bps: float
    expected_event_rate: float
    family_prior_score: float
    status: AlphaSearchStatus = "pending"
    retire_reason: str | None = None
    feature_family: CandidateFeatureFamily = "price_structure"
    turnover_budget_per_year: float = 365.0
    max_compute_cost_score: float = 1.0
    tested_count: int = 0
    survivor_count: int = 0
    posterior_pass_rate: float = 0.0
    posterior_edge_bps: float = 0.0

    def __post_init__(self) -> None:
        if not self.blueprint_id:
            raise ValueError("blueprint_id must not be empty")
        if not self.family:
            raise ValueError("family must not be empty")
        if not self.variant:
            raise ValueError("variant must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.tf_minutes <= 0:
            raise ValueError("tf_minutes must be positive")
        if self.cost_floor_bps < 0.0:
            raise ValueError("cost_floor_bps must be >= 0.0")
        if self.expected_event_rate < 0.0:
            raise ValueError("expected_event_rate must be >= 0.0")
        if not np.isfinite(self.family_prior_score):
            raise ValueError("family_prior_score must be finite")
        if self.tested_count < 0:
            raise ValueError("tested_count must be >= 0")
        if self.survivor_count < 0:
            raise ValueError("survivor_count must be >= 0")
        if self.posterior_pass_rate < 0.0 or self.posterior_pass_rate > 1.0:
            raise ValueError("posterior_pass_rate must be in [0.0, 1.0]")
        if not np.isfinite(self.posterior_edge_bps):
            raise ValueError("posterior_edge_bps must be finite")


@dataclass(slots=True, frozen=True)
class AlphaGateEvidence:
    schema_version: AlphaGateSchema
    run_id: str
    timeframe: str
    family: str
    variant: str
    recipe_id: str
    archetype: AlphaArchetype
    symbol_scope: SymbolScope
    n_events: int
    effective_n: float
    mean_gross_bps: float
    mean_cost_bps: float
    mean_net_bps: float
    gross_lcb_bps: float
    net_lcb_bps: float
    nw_tstat: float
    rank_ic: float
    rank_ic_tstat: float
    cost_drag_ratio: float
    turnover_per_year: float
    novelty_corr_max: float
    incremental_rank_ic: float
    compute_cost_score: float
    event_hit_rate: float
    payoff_skew: float
    xs_spread_lcb_bps: float | None
    liquidity_cost_stress_bps: float
    bootstrap_lcb_bps: float
    bootstrap_agree: bool
    gate_passed: bool
    handoff_tier: AlphaGateHandoffTier
    selected_for_l1: bool
    reject_reasons: tuple[str, ...]
    soft_flags: tuple[str, ...]
    capacity_score: float = 0.0
    regime_stability: float = 0.0
    tf_corroboration: float = 0.0
    entry_mode: AlphaEntryMode = "sparse"

    def __post_init__(self) -> None:
        if self.n_events < 0:
            raise ValueError("n_events must be >= 0")
        if self.effective_n < 0.0:
            raise ValueError("effective_n must be >= 0.0")
        if self.cost_drag_ratio < 0.0:
            raise ValueError("cost_drag_ratio must be >= 0.0")
        if not (0.0 <= self.event_hit_rate <= 1.0):
            raise ValueError("event_hit_rate must be in [0.0, 1.0]")
        numeric_fields = [
            self.mean_gross_bps, self.mean_cost_bps, self.mean_net_bps,
            self.gross_lcb_bps, self.net_lcb_bps, self.nw_tstat,
            self.rank_ic, self.rank_ic_tstat, self.cost_drag_ratio,
            self.turnover_per_year, self.event_hit_rate, self.payoff_skew,
            self.liquidity_cost_stress_bps, self.bootstrap_lcb_bps,
        ]
        for v in numeric_fields:
            if not np.isfinite(v):
                raise ValueError(f"numeric field must be finite, got {v}")
        if self.xs_spread_lcb_bps is not None and not np.isfinite(self.xs_spread_lcb_bps):
            raise ValueError("xs_spread_lcb_bps must be finite if not None")
        if not (0.0 <= self.capacity_score <= 1.0):
            raise ValueError("capacity_score must be in [0.0, 1.0]")
        if not (0.0 <= self.regime_stability <= 1.0):
            raise ValueError("regime_stability must be in [0.0, 1.0]")
        if not (0.0 <= self.tf_corroboration <= 1.0):
            raise ValueError("tf_corroboration must be in [0.0, 1.0]")


@dataclass(slots=True, frozen=True)
class AlphaGateConfig:
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
    archetype_event_floors: Mapping[AlphaArchetype, int] = field(
        default_factory=lambda: {
            "trend": 30,
            "mean_reversion": 40,
            "carry": 20,
            "flow": 12,
            "cross_sectional": 30,
            "hedge": 10,
        }
    )
    family_event_floors: Mapping[str, int] = field(default_factory=dict)
    min_seed_slots_per_archetype: int = 1
    min_seed_slots_per_timeframe: int = 1
    allow_soft_seed_when_only_soft_failures: bool = True
    priority_weights: L0PriorityWeights = field(default_factory=L0PriorityWeights)
    min_candidate_rank_ic_tstat: float = 2.0
    min_xs_symbols_per_bar: int = 5
    max_abs_btc_beta: float = 0.80
    high_turnover_per_year: float = 180.0
    liquidity_cost_stress_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.min_candidate_rank_ic_tstat < 0.0:
            raise ValueError("min_candidate_rank_ic_tstat must be >= 0.0")
        if self.min_xs_symbols_per_bar < 2:
            raise ValueError("min_xs_symbols_per_bar must be >= 2")
        if self.max_abs_btc_beta < 0.0:
            raise ValueError("max_abs_btc_beta must be >= 0.0")
        if self.high_turnover_per_year < 0.0:
            raise ValueError("high_turnover_per_year must be >= 0.0")
        if self.liquidity_cost_stress_mult < 0.0:
            raise ValueError("liquidity_cost_stress_mult must be >= 0.0")


# Backward-compat alias
CheapGateConfig = AlphaGateConfig


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
    mean_gross_bps: float
    mean_cost_bps: float


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
    mean_gross_bps: float
    mean_cost_bps: float
    mean_net_bps: float
    gross_lcb_bps: float
    net_lcb_bps: float
    nw_tstat: float
    rank_ic: float
    rank_ic_tstat: float
    cost_drag_ratio: float
    turnover_per_year: float
    novelty_corr_max: float
    incremental_rank_ic: float
    compute_cost_score: float
    event_hit_rate: float
    payoff_skew: float
    xs_spread_lcb_bps: float | None
    liquidity_cost_stress_bps: float
    bootstrap_lcb_bps: float
    bootstrap_agree: bool
    gate_passed: bool
    handoff_tier: str
    selected_for_l1: bool
    reject_reasons: str
    soft_flags: str
    bucket_key: str
    bucket_rank: int
    redundant_with: str
    bucket_eff_test_count: float
    global_eff_test_count: float
    l1_priority_score: float
    l1_budget_units: int
    tf_coverage_count: int
    sign_agreement_ratio: float
    corroboration_tier: str
    stage_label: str
    created_at_ms: int
    source: str = ""
    capacity_score: float = 0.0
    regime_stability: float = 0.0
    tf_corroboration: float = 0.0
    entry_mode: str = "sparse"


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
    cheap_gate: AlphaGateConfig = field(default_factory=AlphaGateConfig)
    posterior_gate: PosteriorGateConfig = field(default_factory=PosteriorGateConfig)
    l2_policy: L2PosteriorPolicyConfig = field(default_factory=L2PosteriorPolicyConfig)
    enable_fast_discovery_timeframes: bool = False
    fast_discovery_timeframes: tuple[str, ...] = ("1h", "2h")
    enable_correlation_audit: bool = False
    observability_mode: AlphaGateObservabilityMode = "debug_log"
    debug_top_k_rows: int = 10
    artifact_write_enabled: bool = False
    gate_schema: AlphaGateSchema = "unified"
    enable_cost_aware_generation: bool = True
    exploration_budget_fraction: float = 0.15
    cost_prior_floor_by_tf: Mapping[str, float] = field(default_factory=dict)
    use_all_timeframes_in_l0: bool = True
    debug_reject_bucket_rows: int = 5

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
        if self.observability_mode not in ("debug_log", "artifact", "off"):
            raise ValueError(f"invalid observability_mode: {self.observability_mode!r}")
        if self.gate_schema != "unified":
            raise ValueError(f"gate_schema must be 'unified', got {self.gate_schema!r}")
        if self.debug_top_k_rows < 1:
            raise ValueError(f"debug_top_k_rows must be >= 1, got {self.debug_top_k_rows}")
        if not (0.0 < self.exploration_budget_fraction < 1.0):
            raise ValueError(
                f"exploration_budget_fraction must be in (0.0, 1.0), got {self.exploration_budget_fraction}"
            )
        if self.debug_reject_bucket_rows < 1:
            raise ValueError(f"debug_reject_bucket_rows must be >= 1, got {self.debug_reject_bucket_rows}")


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
