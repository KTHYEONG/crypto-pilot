"""Alpha Foundry contracts and stage state machines.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
[ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
[ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.edge_failure import EdgeFailureAxis
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


CrossTFPruningStatus: TypeAlias = Literal["disabled", "applied", "audit_only", "fail_open"]

CrossTfDiagnosticRun: TypeAlias = Literal[
    "control",
    "control_repeat",
    "treatment",
    "fusion_ablation",
]
CrossTfDiagnosticStage: TypeAlias = Literal[
    "native_panels",
    "cheap_evidence",
    "fusion_evidence",
    "canonical_l0",
    "manifest_route",
    "native_labeled_events",
    "l1_delivery_events",
    "outer_folds",
    "l1_result",
]
DiagnosticScalar: TypeAlias = int | float | str | bool


@dataclass(slots=True, frozen=True)
class CrossTfStageSnapshot:
    """Compact immutable record of one cross-timeframe diagnostic stage."""

    schema_version: int
    run: CrossTfDiagnosticRun
    stage: CrossTfDiagnosticStage
    timeframe: str
    digest_sha256: str
    item_count: int
    identity_keys: tuple[str, ...]
    metrics: tuple[tuple[str, DiagnosticScalar], ...]


class CrossTfDiagnosticSink(Protocol):
    """Receive an opaque stage payload without creating reverse imports."""

    def __call__(
        self,
        *,
        run: CrossTfDiagnosticRun,
        stage: CrossTfDiagnosticStage,
        timeframe: str,
        payload: object,
    ) -> None:
        """Record one diagnostic payload."""


@dataclass(slots=True, frozen=True)
class CrossTFSharedContext:
    """    [ADR_20260712_L0_CROSS_TF_PRUNING_PERFORMANCE] Precomputed inputs shared
    between audit_l0_selected_recipe_independence() and compute_cross_tf_redundancy()
    to eliminate duplicate canonical-context/projection/correlation work when both
    are invoked in the same manifest call.

    [ADR_20260712_L0_CROSS_TF_BATCH_ACCELERATION] entry_pos_flat/entry_neg_flat/
    n_entries added for batch jaccard matmul. dict[str, "NDArray[np.int8]"] string
    annotation avoids numpy-eager resolution in slots=True runtime."""
    canonical_context: CrossTFCanonicalContext
    proj_cache: dict[str, tuple[NDArray[np.float32], NDArray[np.bool_]]]
    side_entry_cache: dict[str, tuple[NDArray[np.int8], NDArray[np.bool_]]]
    corr: NDArray[np.float64]
    recipe_order: tuple[str, ...]
    entry_pos_flat: dict[str, "NDArray[np.int8]"] = field(default_factory=dict)  # noqa: UP037
    entry_neg_flat: dict[str, "NDArray[np.int8]"] = field(default_factory=dict)  # noqa: UP037
    n_entries: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CrossTFCanonicalContext:
    canonical_tf: str
    canonical_datetimes_ns: NDArray[np.int64]
    active_mask_2d: NDArray[np.bool_]
    common_start_ns: int
    common_end_ns: int
    n_common_active_bars: int


@dataclass(slots=True, frozen=True)
class CrossTFPairEvidence:
    recipe_id_a: str
    recipe_id_b: str
    score_corr: float
    shared_directional_entries: int
    directional_entry_jaccard: float
    is_redundant: bool


DataSupportTier: TypeAlias = Literal["full_support", "partial_support"]


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
    corroboration_tier: CorroborationTier
    discovery_tier: DiscoveryTier
    l1_priority_score: float
    l1_budget_units: int
    hard_reject_reasons: tuple[L0HardRejectReason, ...]
    soft_flags: tuple[L0SoftFlag, ...]
    data_support_tier: DataSupportTier = "full_support"


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
    family_event_floors: Mapping[str, int] = field(
        default_factory=lambda: {"funding_flow_carry": 200}
    )
    min_seed_slots_per_archetype: int = 1
    min_seed_slots_per_timeframe: int = 1
    allow_soft_seed_when_only_soft_failures: bool = True
    priority_weights: L0PriorityWeights = field(default_factory=L0PriorityWeights)
    min_candidate_rank_ic_tstat: float = 2.0
    min_xs_symbols_per_bar: int = 5
    max_abs_btc_beta: float = 0.80
    high_turnover_per_year: float = 180.0
    liquidity_cost_stress_mult: float = 1.0
    l0_cost_diagnostics_enabled: bool = False  # opt-in, log-only [ADR_20260711_L0_NAN_COST_HTF_BLIND_REJECTION]

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
    """[ADR_20260712_L0_GATE_PIPELINE_OPT] 3 optional cache fields added:
    cheap_event_arrays (event_mask for canonical), cheap_block_stats (gross_block_means),
    cheap_meta_stats (rank_ic) — populated by evaluate_panel_cheap_gate, consumed by
    evaluate_panel_gate cache path to skip redundant Phase 3 computation."""

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
    data_support_tier: DataSupportTier = "full_support"
    cheap_event_arrays: dict[str, NDArray[np.float64]] | None = field(default=None)
    cheap_block_stats: dict[str, Any] | None = field(default=None)
    cheap_meta_stats: dict[str, Any] | None = field(default=None)


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
    pair_evidence: tuple[CrossTFPairEvidence, ...] = ()
    canonical_tf: str = ""
    common_start_ns: int = 0
    common_end_ns: int = 0
    n_common_active_bars: int = 0


@dataclass(slots=True, frozen=True)
class AlphaFoundryEvidenceRow:
    """[ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION]"""

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

    # new backward-compatible optional fields
    cell_id: str = ""
    cell_axes: str = ""
    cell_values: str = ""
    execution_style: str = "taker_now"
    fill_probability: float = 1.0
    adverse_selection_bps: float = 0.0
    tested_horizons: str = ""
    selected_horizon: int = 0
    failure_axis: str = ""
    failure_axes: str = ""


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

ConditionalAxis = Literal[
    "symbol_liquidity",
    "symbol_cluster",
    "market_regime",
    "volatility_regime",
    "funding_polarity",
    "score_quantile",
    "event_hour_utc",
    "source_tf",
]

ExecutionStyle = Literal["taker_now", "maker_retest", "maker_or_cancel", "hybrid"]


@dataclass(slots=True, frozen=True)
class ConditionalCellGateConfig:
    """Controls conditional L0 cell search. [ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION]"""

    enabled: bool = False
    axes: tuple[ConditionalAxis, ...] = ("score_quantile", "symbol_liquidity", "volatility_regime")
    min_cell_events: int = 80
    min_cell_effective_n: float = 40.0
    max_cells_per_recipe: int = 24
    max_axes_per_cell: int = 2
    min_symbols_per_cell: int = 5
    quantile_bins: tuple[float, ...] = (0.70, 0.85, 0.95)
    allow_single_symbol_cells: bool = False
    selection_lcb_penalty_bps: float = 2.0
    dedupe_selected_events: bool = True

    def __post_init__(self) -> None:
        if self.min_cell_events < 1:
            raise ValueError("min_cell_events must be >= 1")
        if self.min_cell_effective_n < 1.0:
            raise ValueError("min_cell_effective_n must be >= 1.0")
        if self.max_cells_per_recipe < 1:
            raise ValueError("max_cells_per_recipe must be >= 1")
        if self.max_axes_per_cell < 1:
            raise ValueError("max_axes_per_cell must be >= 1")
        if self.min_symbols_per_cell < 1:
            raise ValueError("min_symbols_per_cell must be >= 1")
        for q in self.quantile_bins:
            if not (0.0 < q < 1.0):
                raise ValueError(f"quantile_bins values must be in (0.0, 1.0), got {q}")
        if self.selection_lcb_penalty_bps < 0.0:
            raise ValueError("selection_lcb_penalty_bps must be >= 0.0")


@dataclass(slots=True, frozen=True)
class ExecutionArmConfig:
    """Controls execution arm exploration. [ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION]"""

    enabled: bool = False
    styles: tuple[ExecutionStyle, ...] = ("taker_now",)
    min_fill_probability: float = 0.35
    maker_retest_window_bars: int = 1
    min_adverse_selection_bps: float = 1.0
    max_arm_count_per_cell: int = 3

    _VALID_STYLES: frozenset[str] = frozenset({
        "taker_now", "maker_retest", "maker_or_cancel", "hybrid",
    })

    def __post_init__(self) -> None:
        for s in self.styles:
            if s not in self._VALID_STYLES:
                raise ValueError(f"unsupported execution style: {s!r}")
        if not (0.0 <= self.min_fill_probability <= 1.0):
            raise ValueError("min_fill_probability must be in [0.0, 1.0]")
        if self.maker_retest_window_bars < 1:
            raise ValueError("maker_retest_window_bars must be >= 1")
        if self.max_arm_count_per_cell < 1:
            raise ValueError("max_arm_count_per_cell must be >= 1")


@dataclass(slots=True, frozen=True)
class L0DiagnosticConfig:
    """Controls diagnostic-only conditional-cell / execution-arm pass.

    [ADR_20260709_L0_CONDITIONAL_DIAGNOSTIC_WIRING]
    """

    failure_axes_for_cell_search: tuple[EdgeFailureAxis, ...] = ("weak_gross_edge", "cost_dominated")
    failure_axes_for_arm_search: tuple[EdgeFailureAxis, ...] = ("cost_dominated",)
    calibration_fraction: float = 0.70
    max_diagnostic_recipes: int = 50

    def __post_init__(self) -> None:
        if not (0.0 < self.calibration_fraction < 1.0):
            raise ValueError(f"calibration_fraction must be in (0,1), got {self.calibration_fraction}")
        if self.max_diagnostic_recipes < 1:
            raise ValueError(f"max_diagnostic_recipes must be >= 1, got {self.max_diagnostic_recipes}")


@dataclass(slots=True, frozen=True)
class AlphaFoundryRuntimeConfig:
    """[ADR_20260715_L0_L1_NATIVE_CONTRACT] Runtime mode and L0 gate settings."""
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

    # new feature flags default off
    enable_failure_attribution: bool = False
    enable_conditional_l0_cells: bool = False
    enable_execution_arms: bool = False
    enable_horizon_sweep: bool = False
    conditional_cell: ConditionalCellGateConfig = field(default_factory=ConditionalCellGateConfig)
    execution_arm: ExecutionArmConfig = field(default_factory=ExecutionArmConfig)
    horizon_sweep_bars: tuple[int, ...] = (1, 2, 3, 6)
    diagnostic: L0DiagnosticConfig = field(default_factory=L0DiagnosticConfig)
    enable_discovery_unit_handoff: bool = False
    max_discovery_units_for_l1: int = 12
    max_discovery_event_jaccard: float = 0.80
    min_discovery_unit_lcb_bps: float = 0.0

    # Cross-TF diversity audit [LIMIT-04][LIMIT-09]
    enable_cross_tf_diversity_audit: bool = False
    cross_tf_diversity_canonical_tf: str = "1h"

    # Cross-TF pruning enforcement [ADR_20260711_L0_CROSS_TF_PRUNING_ADMISSION]
    enable_cross_tf_pruning: bool = False
    cross_tf_pruning_min_survivors_per_archetype: int = 1
    cross_tf_pruning_min_survivors_per_tf: int = 0

    # Cross-TF evidence-conditioned admission [ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION]
    cross_tf_min_common_active_bars: int = 480
    cross_tf_min_directional_entry_jaccard: float = 0.50
    cross_tf_min_shared_directional_entries: int = 12
    ltf_exec_1m_min_coverage: float = 0.80
    enable_ltf_family_pool_experiment: bool = False

    # L0 memory-bound dataflow [ADR_20260712_L0_MEMORY_BOUND_DATAFLOW]
    l0_max_rss_mb: int = 10_240
    l0_memory_fraction_cap: float = 0.60
    l0_memory_safety_margin_mb: int = 512
    ltf_exec_1m_max_symbols: int = 64
    ltf_exec_1m_max_workers: int = 1
    ltf_streaming_enabled: bool = True
    l0_native_tf_max_workers: int = 1

    # L0 parallel gate execution [LIMIT-03] 1=sequential (current behavior), 2-4=parallel
    l0_parallel_max_workers: int = 1
    # TF-probe skip capability [LIMIT-07] default unchanged (still runs)
    enable_tf_probe_scoped: bool = True

    # Cross-TF corroboration reference set [LIMIT-12]
    corroboration_reference_tfs: tuple[str, ...] = (
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    )

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
        if self.max_discovery_units_for_l1 < 1:
            raise ValueError(f"max_discovery_units_for_l1 must be >= 1, got {self.max_discovery_units_for_l1}")
        if not (0.0 <= self.max_discovery_event_jaccard <= 1.0):
            raise ValueError(
                f"max_discovery_event_jaccard must be in [0.0, 1.0], got {self.max_discovery_event_jaccard}"
            )
        if self.min_discovery_unit_lcb_bps < 0.0:
            raise ValueError(
                f"min_discovery_unit_lcb_bps must be >= 0.0, got {self.min_discovery_unit_lcb_bps}"
            )
        if self.cross_tf_pruning_min_survivors_per_archetype < 1:
            raise ValueError(
                f"cross_tf_pruning_min_survivors_per_archetype must be >= 1, "
                f"got {self.cross_tf_pruning_min_survivors_per_archetype}"
            )
        if self.cross_tf_pruning_min_survivors_per_tf < 0:
            raise ValueError(
                f"cross_tf_pruning_min_survivors_per_tf must be >= 0, "
                f"got {self.cross_tf_pruning_min_survivors_per_tf}"
            )
        if self.cross_tf_min_common_active_bars < 1:
            raise ValueError(
                f"cross_tf_min_common_active_bars must be >= 1, "
                f"got {self.cross_tf_min_common_active_bars}"
            )
        if not (0.0 <= self.cross_tf_min_directional_entry_jaccard <= 1.0):
            raise ValueError(
                f"cross_tf_min_directional_entry_jaccard must be in [0.0, 1.0], "
                f"got {self.cross_tf_min_directional_entry_jaccard}"
            )
        if self.cross_tf_min_shared_directional_entries < 1:
            raise ValueError(
                f"cross_tf_min_shared_directional_entries must be >= 1, "
                f"got {self.cross_tf_min_shared_directional_entries}"
            )
        if not (0.0 <= self.ltf_exec_1m_min_coverage <= 1.0):
            raise ValueError(
                f"ltf_exec_1m_min_coverage must be in [0.0, 1.0], "
                f"got {self.ltf_exec_1m_min_coverage}"
            )
        if self.l0_max_rss_mb < 1:
            raise ValueError(f"l0_max_rss_mb must be >= 1, got {self.l0_max_rss_mb}")
        if not (0.0 < self.l0_memory_fraction_cap <= 1.0):
            raise ValueError(
                f"l0_memory_fraction_cap must be in (0.0, 1.0], got {self.l0_memory_fraction_cap}"
            )
        if self.l0_memory_safety_margin_mb < 0:
            raise ValueError(
                f"l0_memory_safety_margin_mb must be >= 0, got {self.l0_memory_safety_margin_mb}"
            )
        if self.ltf_exec_1m_max_symbols < 1:
            raise ValueError(
                f"ltf_exec_1m_max_symbols must be >= 1, got {self.ltf_exec_1m_max_symbols}"
            )
        if not (1 <= self.ltf_exec_1m_max_workers <= 2):
            raise ValueError(
                f"ltf_exec_1m_max_workers must be in [1,2], got {self.ltf_exec_1m_max_workers}"
            )
        if not (1 <= self.l0_native_tf_max_workers <= 2):
            raise ValueError(
                f"l0_native_tf_max_workers must be in [1,2], got {self.l0_native_tf_max_workers}"
            )
        if not (1 <= self.l0_parallel_max_workers <= 4):
            raise ValueError(
                f"l0_parallel_max_workers must be in [1,4], got {self.l0_parallel_max_workers}"
            )

        # ── corroboration_reference_tfs contract [LIMIT-12] ──
        _valid_tfs = frozenset({"1h", "2h", "4h", "6h", "8h", "12h", "1d"})
        if len(set(self.corroboration_reference_tfs)) != len(self.corroboration_reference_tfs):
            raise ValueError("corroboration_reference_tfs contains duplicates")
        if any(tf not in _valid_tfs for tf in self.corroboration_reference_tfs):
            raise ValueError("corroboration_reference_tfs contains unsupported timeframe")


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
    n_distinct_thesis_ids_passed: int = 0  # additive, default preserves old fixtures [LIMIT-10]
    json_path: str = ""
    parquet_path: str = ""


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
    "partial_support",
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


@dataclass(slots=True, frozen=True)
class EntryConfluenceSnapshot:
    """Per (symbol, ltf_bar) entry-timing feature snapshot. Look-ahead safe (닫힌 봉만).

    [ADR_20260707_LTF_ENTRY_TIMING_LAYER]
    """

    symbol: str
    ltf: str
    bar_ts: np.int64
    cvd_delta_z: float
    cvd_aligned_with_bias: bool
    vwap_dev_sigma: float
    trend_quality_pass: bool
    confluence_score: float
    confluence_direction: int


@dataclass(slots=True, frozen=True)
class HtfDirectionalEpisode:
    """Existing L0-gated directional event 참조용 링크. 신규 저장소 아님 — raw_events 행에서 파생."""

    episode_id: str
    symbol: str
    family: str
    variant: str
    timeframe: str
    htf_bias: int
    base_entry_idx: int
    expected_holding_bars: int
    handoff_tier: str

    def __post_init__(self) -> None:
        if self.handoff_tier not in {"seed", "candidate"}:
            raise ValueError(f"handoff_tier must be 'seed' or 'candidate', got {self.handoff_tier!r}")


@dataclass(slots=True, frozen=True)
class EntryTimingWindow:
    episode_id: str
    ltf: str
    max_wait_bars_base: int
    triggered: bool
    refined_entry_idx: int
    price_improvement_bps: float
    opportunity_cost_bps: float
    net_timing_edge_bps: float
    coverage_status: Literal["covered", "uncovered_fallback"] = "covered"

    def __post_init__(self) -> None:
        if not self.triggered and self.net_timing_edge_bps != 0.0:
            raise ValueError("non-triggered window must have net_timing_edge_bps == 0.0 (fail-closed invariant)")


@dataclass(slots=True, frozen=True)
class EntryTimingGateConfig:
    enabled: bool = False
    ltf_grid: tuple[str, ...] = ("5m", "15m", "30m", "1h")
    max_wait_bars_ratio: float = 0.25
    min_net_timing_edge_lcb_bps: float = 1.0
    cvd_lookback_bars: int = 96
    vwap_anchor_max_bars: int = 288
    confluence_weights: Mapping[str, float] = field(
        default_factory=lambda: {"cvd": 0.5, "vwap": 0.5}
    )
    enabled_combos: frozenset[tuple[str, str, str]] = frozenset()

    def __post_init__(self) -> None:
        if not (0.0 < self.max_wait_bars_ratio <= 1.0):
            raise ValueError(f"max_wait_bars_ratio must be in (0, 1], got {self.max_wait_bars_ratio}")
        if self.min_net_timing_edge_lcb_bps < 0.0:
            raise ValueError("min_net_timing_edge_lcb_bps must be >= 0")


@dataclass(slots=True, frozen=True)
class Universe1mCoverageTier:
    """Tier classifying which universe symbols have 1m data available.

    Attributes:
        covered_symbols: Symbols with {symbol}_1m.parquet present.
        universe_symbols: Full active universe set.

    [ADR_20260708_LTF_NATIVE_DIRECTIONAL_SEARCH]
    """

    covered_symbols: frozenset[str]
    universe_symbols: frozenset[str]

    @property
    def coverage_ratio(self) -> float:
        """Fraction of universe_symbols that are covered."""
        if not self.universe_symbols:
            return 0.0
        return len(self.covered_symbols) / len(self.universe_symbols)

    def is_covered(self, symbol: str) -> bool:
        """True if symbol is among covered_symbols."""
        return symbol in self.covered_symbols


@dataclass(slots=True, frozen=True)
class L0IndependenceAudit:
    n_selected_total: int
    n_distinct_thesis_ids: int
    n_independent_clusters: int
    cluster_members: dict[int, tuple[str, ...]]
    demoted_recipe_ids: tuple[str, ...]
    demoted_reason_by_id: dict[str, str]
    canonical_tf: str
    max_corr_threshold: float


class L0DeliveryContractError(ValueError):
    """Raised when a gate-mode L0 delivery cannot be consumed exactly by L1."""


@dataclass(slots=True, frozen=True)
class L0TfDeliveryRoute:
    timeframe: str
    selected_recipe_ids: tuple[str, ...]
    allocated_budget_units: int
    evidence_end_ns: int

    def __post_init__(self) -> None:
        if not self.timeframe:
            raise L0DeliveryContractError("delivery route timeframe must not be empty")
        if self.allocated_budget_units < 0:
            raise L0DeliveryContractError("allocated_budget_units must be non-negative")
        if len(set(self.selected_recipe_ids)) != len(self.selected_recipe_ids):
            raise L0DeliveryContractError("delivery route contains duplicate recipe_id")
        if self.evidence_end_ns <= 0:
            raise L0DeliveryContractError("evidence_end_ns must be positive")


@dataclass(slots=True, frozen=True)
class L0StrategyDeliveryManifest:
    run_id_prefix: str
    reports_by_tf: dict[str, AlphaFoundryBridgeReport]
    independence_audit: L0IndependenceAudit | None
    final_selected_recipe_ids: tuple[str, ...]
    total_l1_verification_budget: int
    pruning_status: CrossTFPruningStatus = "disabled"
    pruning_reason: str = ""
    routes: tuple[L0TfDeliveryRoute, ...] = field(default_factory=tuple)


_L0_VALID_TFS = frozenset({"1h", "2h", "4h", "6h", "8h", "12h", "1d"})


def resolve_corroboration_evidence_for_target(
    *,
    target_tf: str,
    evidence_by_tf: Mapping[str, Any],
    reference_tfs: tuple[str, ...],
) -> dict[str, Any]:
    if target_tf not in evidence_by_tf:
        raise L0DeliveryContractError(f"missing native evidence for target_tf={target_tf}")
    allowed = set(reference_tfs)
    allowed.add(target_tf)
    return {tf: evidence_by_tf[tf] for tf in evidence_by_tf if tf in allowed}
