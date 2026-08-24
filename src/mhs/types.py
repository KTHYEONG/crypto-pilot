"""Frozen Phase 1 MHS contracts (types only).

All domain tunables moved to ``src.mhs.params`` (I2 single source); this module
keeps only the type definitions and the derived book/band specs that depend on
them. ``HorizonBand``/``ExecutionSpec``/``BookSpec`` are the types; ``_FAST_BAND``/
``_SLOW_BAND``/``BOOK_SPECS`` are derived from the params-owned horizon
candidates. The tunables are re-exported here so existing ``from src.mhs.types
import X`` import paths keep working.

All literals here are preregistered measurement outputs or frozen architecture
decisions (``docs/architecture/multi-horizon-market-state.md``). A change to any of
them is a new contract revision, never an inline edit at a call site.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.mhs.params import (
    BOOK_BLEND_WEIGHTS,
    COMMITTEE_GROWTH_BARS_PER_YEAR,
    COMMITTEE_GROWTH_HORIZON_YEARS,
    COMMITTEE_GROWTH_MAX_DRAWDOWN,
    COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
    COMMITTEE_GROWTH_MAX_RUIN_PROB,
    COMMITTEE_GROWTH_N_PATHS,
    COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS,
    COMMITTEE_GROWTH_RUIN_FRACTION,
    COMMITTEE_MEMBERS,
    COMMITTEE_OOS_START,
    COMMITTEE_PURGE_HOURS,
    COMMITTEE_REGIME_ADAPTIVE_WINDOW,
    COMMITTEE_TARGET_GROSS,
    COMMITTEE_TARGET_VOL,
    COMMITTEE_TRANCHE_COUNT,
    CRASH_REGIME_REFERENCE_SYMBOLS,
    DISCOVERY_START,
    EXPOSURE_DRAWDOWN_BRAKE_FLOOR,
    EXPOSURE_DRAWDOWN_BRAKE_K,
    FEATURE_MIN_COVERAGE,
    FILL_MARK_MAX_LOG_DIVERGENCE,
    FILL_MARK_PRICE_PROTECTION_BAND,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS,
    FUNDING_CARRY_SLEEVE_WEIGHT,
    GROWTH_ENVELOPE_DEFAULT,
    GROWTH_RISK_ENVELOPES,
    MEASURED_EXECUTION_COST_TIERS_BPS,
    MOMENTUM_HORIZON_CANDIDATES_HOURS,
    PNL_TARGET_ANNUAL_VOL,
    PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
    PNL_VOL_TARGET_MAX_SCALE,
    RAM_BUDGET_FRACTION,
    RAM_RESERVE_FLOOR_BYTES,
    RAM_RESERVE_FRACTION,
    REGISTERED_POLICY_THRESHOLDS,
    REVERSAL_HORIZON_CANDIDATES_HOURS,
    SEARCH_TRIALS_ATTEMPTED,
    TREND_SLEEVE_HORIZONS_HOURS,
    WORKER_PEAK_RSS_BYTES,
)

__all__ = [
    "BOOK_BLEND_WEIGHTS",
    "BOOK_SPECS",
    "COMMITTEE_GROWTH_BARS_PER_YEAR",
    "COMMITTEE_GROWTH_HORIZON_YEARS",
    "COMMITTEE_GROWTH_MAX_DRAWDOWN",
    "COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB",
    "COMMITTEE_GROWTH_MAX_RUIN_PROB",
    "COMMITTEE_GROWTH_N_PATHS",
    "COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS",
    "COMMITTEE_GROWTH_RUIN_FRACTION",
    "COMMITTEE_MEMBERS",
    "COMMITTEE_OOS_START",
    "COMMITTEE_PURGE_HOURS",
    "COMMITTEE_REGIME_ADAPTIVE_WINDOW",
    "COMMITTEE_TARGET_GROSS",
    "COMMITTEE_TARGET_VOL",
    "COMMITTEE_TRANCHE_COUNT",
    "CRASH_REGIME_REFERENCE_SYMBOLS",
    "DISCOVERY_START",
    "EXPOSURE_DRAWDOWN_BRAKE_FLOOR",
    "EXPOSURE_DRAWDOWN_BRAKE_K",
    "FEATURE_MIN_COVERAGE",
    "FILL_MARK_MAX_LOG_DIVERGENCE",
    "FILL_MARK_PRICE_PROTECTION_BAND",
    "FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS",
    "FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS",
    "FUNDING_CARRY_SLEEVE_WEIGHT",
    "GROWTH_ENVELOPE_DEFAULT",
    "GROWTH_RISK_ENVELOPES",
    "MEASURED_EXECUTION_COST_TIERS_BPS",
    "MOMENTUM_HORIZON_CANDIDATES_HOURS",
    "PNL_TARGET_ANNUAL_VOL",
    "PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS",
    "PNL_VOL_TARGET_MAX_SCALE",
    "RAM_BUDGET_FRACTION",
    "RAM_RESERVE_FLOOR_BYTES",
    "RAM_RESERVE_FRACTION",
    "REGISTERED_POLICY_THRESHOLDS",
    "REVERSAL_HORIZON_CANDIDATES_HOURS",
    "SEARCH_TRIALS_ATTEMPTED",
    "TREND_SLEEVE_HORIZONS_HOURS",
    "WORKER_PEAK_RSS_BYTES",
    "BookSpec",
    "ExecutionSpec",
    "HorizonBand",
]



@dataclass(frozen=True, slots=True)
class HorizonBand:
    """A measured return band. ``sign`` is a measured band property.

    The fast band reverses (``sign=-1``) and the slow band follows momentum
    (``sign=+1``); both signs were measured on discovery data (spec §2.1) and
    are never inferred at call sites.
    """

    name: str
    horizons_hours: tuple[int, ...]
    sign: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.horizons_hours:
            raise ValueError("horizons_hours must not be empty")
        if any(h <= 0 for h in self.horizons_hours):
            raise ValueError(f"horizons_hours must all be > 0, got {self.horizons_hours}")
        if tuple(self.horizons_hours) != tuple(sorted(set(self.horizons_hours))):
            raise ValueError(
                f"horizons_hours must be strictly ascending, got {self.horizons_hours}"
            )
        if self.sign not in (-1, 1):
            raise ValueError(f"sign must be -1 or +1, got {self.sign}")


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Passive-execution cost and fill contract.

    ``one_way_taker_bps`` reproduces ``CostModel()``'s 8.0 bp one-way assumption
    so the two cost models stay comparable.
    """

    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    taker_slippage_bps: float = 3.0
    passive_timeout_minutes: int = 30
    require_trade_through: bool = True
    ladder_tranches: int = 4

    def __post_init__(self) -> None:
        if min(self.maker_fee_bps, self.taker_fee_bps, self.taker_slippage_bps) < 0:
            raise ValueError("fees and slippage must be non-negative")
        if self.passive_timeout_minutes < 1:
            raise ValueError(f"passive_timeout_minutes must be >= 1, got {self.passive_timeout_minutes}")
        if self.ladder_tranches < 1:
            raise ValueError(f"ladder_tranches must be >= 1, got {self.ladder_tranches}")

    def one_way_taker_bps(self) -> float:
        """One-way all-in taker cost in bps (fee + slippage)."""
        return self.taker_fee_bps + self.taker_slippage_bps


@dataclass(frozen=True, slots=True)
class BookSpec:
    """One frozen Phase 1 book: band, signal horizon, decision step, min symbols.

    ``tranche_count()`` is the number of overlapping phase tranches held
    simultaneously (``horizon_hours // step_hours``), the phase-ensemble
    construction that makes the result independent of an arbitrary
    decision-clock offset (spec §2.4).
    """

    band: HorizonBand
    horizon_hours: int
    step_hours: int
    min_symbols: int = 8

    def __post_init__(self) -> None:
        if self.horizon_hours not in self.band.horizons_hours:
            raise ValueError(
                f"horizon_hours {self.horizon_hours} not in band horizons {self.band.horizons_hours}"
            )
        if self.step_hours < 1:
            raise ValueError(f"step_hours must be >= 1, got {self.step_hours}")
        if self.min_symbols < 2:
            raise ValueError(f"min_symbols must be >= 2, got {self.min_symbols}")
        if self.horizon_hours % self.step_hours != 0:
            raise ValueError("horizon_hours must be divisible by step_hours")

    def tranche_count(self) -> int:
        return self.horizon_hours // self.step_hours


_FAST_BAND = HorizonBand(name="fast_reversal", horizons_hours=REVERSAL_HORIZON_CANDIDATES_HOURS, sign=-1)
_SLOW_BAND = HorizonBand(name="slow_momentum", horizons_hours=MOMENTUM_HORIZON_CANDIDATES_HOURS, sign=1)

BOOK_SPECS: dict[str, BookSpec] = {
    "fast_reversal": BookSpec(
        band=_FAST_BAND, horizon_hours=48, step_hours=6, min_symbols=8,
    ),
    "slow_momentum": BookSpec(
        band=_SLOW_BAND, horizon_hours=168, step_hours=24, min_symbols=8,
    ),
}
