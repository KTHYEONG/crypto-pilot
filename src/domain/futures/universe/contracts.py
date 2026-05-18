"""Typed contracts for futures universe selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RejectCode(StrEnum):
    """Canonical reject reason codes across stages."""

    NOT_TRADING = "NOT_TRADING"
    NOT_LISTED = "NOT_LISTED"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"
    LOW_COVERAGE = "LOW_COVERAGE"
    TOO_MANY_ZERO_VOLUME_BARS = "TOO_MANY_ZERO_VOLUME_BARS"
    EXCESSIVE_GAPS = "EXCESSIVE_GAPS"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    HIGH_EXECUTION_COST = "HIGH_EXECUTION_COST"
    FUNDING_ANOMALY = "FUNDING_ANOMALY"
    BASIS_ANOMALY = "BASIS_ANOMALY"
    RISK_EVENT_OVERRIDE = "RISK_EVENT_OVERRIDE"
    LISTING_TOO_YOUNG = "LISTING_TOO_YOUNG"
    VOL_BAND_VIOLATION = "VOL_BAND_VIOLATION"
    RANKED_OUT = "RANKED_OUT"


class EventType(StrEnum):
    """Manual risk-event categories."""

    SCHEDULED_UNLOCK = "SCHEDULED_UNLOCK"
    EXCHANGE_HALT = "EXCHANGE_HALT"
    REGULATORY = "REGULATORY"
    SECURITY_INCIDENT = "SECURITY_INCIDENT"


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """Daily append-only universe ledger row."""

    symbol: str
    date: str
    knowledge_date: str
    is_listed: bool
    is_trading: bool
    status: str
    first_kline_date: str
    delist_date: str | None
    delist_announcement: str | None
    adv_usdt_median: float
    adv_usdt_mean: float
    has_kline: bool
    has_funding: bool
    n_bar_gaps: int
    max_gap_bars: int
    frozen_bars: int
    last_60d_coverage: float
    n_zero_volume_bars_60d: int
    funding_rate_8h: float
    open_interest_usdt: float
    oi_usdt_median: float
    oi_change_30d: float
    listing_age_days: int
    vol_30d: float
    basis_z_score: float | None
    basis_annualized_mean: float | None
    basis_vol: float | None
    risk_event_override: str | None
    updated_at_utc: str


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Input data lock record for reproducibility."""

    symbol: str
    period: str
    source: str
    sha256: str
    is_final: bool
    updated_at_utc: str
    tf: str = ""
    url: str = ""
    bytes: int = 0
    fetched_at_utc: str = ""


@dataclass(frozen=True, slots=True)
class ManualEventRow:
    """Manual risk-event input with PIT-safe knowledge-date."""

    symbol: str
    event_type: EventType
    event_date: str
    knowledge_date: str
    severity: str
    action: str
    source_url: str
    recorded_at_utc: str


@dataclass(frozen=True, slots=True)
class SymbolMeta:
    """Per-symbol metadata carried by a universe snapshot."""

    symbol: str
    role: str
    adv_usdt: float
    execution_cost_bps: float
    funding_carry_8h: float
    beta_vs_market: float
    cluster_id: int
    tradeable_rank: int
    basis_annualized_mean: float | None
    basis_vol: float | None
    oi_usdt_median: float
    oi_to_adv: float
    oi_change_30d: float
    capacity_clip_usdt_list: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FilterReport:
    """Audit report for stage-level pass/fail reasons and metrics."""

    symbol: str
    stage0_pass: bool
    stage1_reason: RejectCode | None
    stage1_metrics: dict[str, float]
    stage2_reason: RejectCode | None
    stage2_metrics: dict[str, float]
    stage3_reason: RejectCode | None
    stage3_metrics: dict[str, float]
    stage4_reason: RejectCode | None
    stage4_metrics: dict[str, float]
    stage5_reason: RejectCode | None
    stage5_metrics: dict[str, float]
    stage6_reason: RejectCode | None
    stage6_metrics: dict[str, float]
    final_rank: int | None
    final_cluster_id: int | None
    audit_trail: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """Frozen snapshot for replayable universe membership."""

    as_of: str
    tf: str
    schema_version: int
    config_hash: str
    data_manifest_hash: str
    basket_ref: tuple[str, ...]
    basket_weights: tuple[float, ...]
    selected: tuple[SymbolMeta, ...]
    rejected: dict[str, FilterReport]
    generated_at_utc: str
    ledger_confidence: str
    n_stage0: int
    n_stage1_pass: int
    n_stage2_pass: int
    n_stage3_pass: int
    n_stage4_pass: int
    n_stage5_pass: int
    n_stage6_selected: int
