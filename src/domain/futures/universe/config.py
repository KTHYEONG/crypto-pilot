"""Universe configuration and deterministic hashing utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field  # field kept for PITUniverseConfig default
from typing import Any, Literal


def _canonicalize(value: Any) -> Any:
    """Convert nested values into deterministic JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _sha256_json(payload: dict[str, Any]) -> str:
    canonical_payload = _canonicalize(payload)
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()



@dataclass(frozen=True, slots=True)
class PITUniverseConfig:
    """PIT-compliant universe configuration replacing Stage2-6 gates.

    Replaces Stage2Config…Stage6Config for the new PIT eligibility path.
    k_in, k_out, anchor_symbols, min_dwell_days, max_per_cluster are absent
    by design — eligibility is determined per-bar by execution rules and
    market observations, not by a fixed ranked membership list.

    Attributes:
        schema_version: Bumped on breaking schema changes; differs from
            UniverseConfig.schema_version to prevent cache collisions.
        decision_timeframe: Resolution of the decision calendar (e.g. "4h").
        contract_market: Exchange+market discriminator.
        metric_lookback_days: Trailing window for ADV/Amihud/vol metrics.
        quality_lookback_days: Trailing window for data-quality (coverage/gaps).
        max_market_data_staleness_bars: How many bars without fresh close
            before the instrument is considered stale and ineligible.
        min_metric_observations: Minimum valid daily observations before a
            trailing metric is considered non-NaN.
        max_round_trip_cost_bps: Hard execution-cost ceiling (round-trip).
        max_participation_rate: Maximum fraction of ADV for a single order.
        min_data_confidence: Minimum DataConfidence label accepted; stricter
            policies reject RECONSTRUCTED data and fail closed.
        default_intended_notional_usdt: Bootstrap notional for eligibility
            checks; L2 overrides with actual target order size.
    """

    schema_version: int = 2
    decision_timeframe: str = "4h"
    contract_market: Literal["binance_usdt_perpetual"] = "binance_usdt_perpetual"
    metric_lookback_days: int = 30
    quality_lookback_days: int = 60
    max_market_data_staleness_bars: int = 1
    min_metric_observations: int = 20
    max_round_trip_cost_bps: float = 50.0
    max_participation_rate: float = 0.01
    min_data_confidence: str = "reconstructed"  # DataConfidence value
    default_intended_notional_usdt: float = 10_000.0
    k_in: int = 0  # 0 = no hard top-N (capacity_coverage_target governs); >0 = legacy fixed-N
    capacity_coverage_target: float = 0.90  # cumulative capacity_usdt fraction to retain
    k_max: int = 100  # compute ceiling (hard upper bound)

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if not (0 < self.capacity_coverage_target <= 1.0):
            raise ValueError(
                f"capacity_coverage_target must be in (0, 1]; got {self.capacity_coverage_target}"
            )
        if self.k_max < 1:
            raise ValueError(f"k_max must be >= 1; got {self.k_max}")

    def to_payload(self) -> dict[str, Any]:
        """Return canonical payload for persistence and hashing."""
        return asdict(self)

    def config_hash(self) -> str:
        """Compute deterministic SHA256 hash for reproducibility."""
        return _sha256_json(self.to_payload())


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """Top-level universe configuration with deterministic hash."""

    schema_version: int = 1
    timeframe: str = "4h"
    ledger_confidence: str = "reconstructed"
    universe_engine: Literal["stage6", "pit"] = "pit"
    pit_config: PITUniverseConfig = field(default_factory=PITUniverseConfig)

    def to_payload(self) -> dict[str, Any]:
        """Return canonical payload for persistence and hashing."""
        return asdict(self)

    def config_hash(self) -> str:
        """Compute deterministic SHA256 hash for reproducibility."""
        return _sha256_json(self.to_payload())


def hash_config(config: UniverseConfig) -> str:
    """Return deterministic config hash."""
    return config.config_hash()
