"""Frozen Phase 1 MHS contracts.

All literals here are preregistered measurement outputs or frozen architecture
decisions (``docs/architecture/multi-horizon-market-state.md``). A change to any of
them is a new contract revision, never an inline edit at a call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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


MEASURED_EXECUTION_COST_TIERS_BPS: dict[str, float] = {
    "optimistic": 2.64,
    "base": 4.18,
    "stress": 6.07,
}

# Frozen MHS discovery window start (spec docs/specs/mhs_data_period_and_gap_hardening.md).
# Domain-layer single source: src/mhs/evaluation.py folds and the application
# orchestrator import it instead of re-typing the literal.
MHS_DISCOVERY_START: pd.Timestamp = pd.Timestamp("2021-01-01", tz="UTC")

# Measured admission (docs/results/mhs_horizon_diagnostic.json
# books.fast_reversal.prescreen): fast_reversal's 446-symbol prescreen net
# t-stat stays below the |t| >= 2.0 admission floor across every cost tier,
# from the 0.0bps pre-cost bound (net_t=+0.577) through the 2.64bps optimistic
# tier (net_t=-0.150, sign already unstable) to the 6.07bps stress tier
# (net_t=-1.094). slow_momentum clears |t| >= 2.0 pre-cost (net_t=+1.859) and
# stays above the floor through the 2.64bps tier (net_t=+1.634), with the
# pre-registered momentum sign throughout.
# fast_reversal keeps its signal/prescreen/phase computation for re-measurement
# but carries zero capital in the Research-GO blend. One-time evidence-based
# revision (same governance pattern as MEASURED_EXECUTION_COST_TIERS_BPS), not
# a performance-selected fit.
PHASE_1_BOOK_BLEND_WEIGHTS: dict[str, float] = {
    "fast_reversal": 0.0,
    "slow_momentum": 1.0,
}

# Fixed reference basket for the opt-in crash-regime directional tilt
# (src/mhs/regime.py crash_regime_tilt_weights, docs/specs/mhs_crash_regime_tilt_overlay.md
# §6). BTCUSDT is listed continuously across the full 2021-2025 dev window --
# chosen for listing-date stability (no universe-composition drift, unlike an
# eligible-symbol basket, ADR_20260812_MHS_MOMENTUM_STRATEGY_REDESIGN_REVIEW
# §3.2), not for backtested performance.
MHS_CRASH_REGIME_REFERENCE_SYMBOLS: tuple[str, ...] = ("BTCUSDT",)

# The fast band's allowed set is the full measured reversal candidate grid
# (docs/specs/mhs_universe_horizon_redesign.md §3.1, 7 horizons, 24h-168h step
# 24) so a fold-scoped discovery/qualification selection may re-verify the
# frozen 48h default. ``PHASE_1_BOOK_SPECS["fast_reversal"].horizon_hours``
# stays 48 (the frozen fallback default); widening only the band is
# diagnostic-only and never changes the 0.0 capital allocation.
REVERSAL_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)
_FAST_BAND = HorizonBand(name="fast_reversal", horizons_hours=REVERSAL_HORIZON_CANDIDATES_HOURS, sign=-1)
# The slow band's allowed set is the full measured momentum candidate grid
# (docs/results/mhs-res.md §2, 19 horizons, 72h-504h step 24) so a
# fold-scoped discovery/qualification selection may replace the frozen 168h
# default. ``PHASE_1_BOOK_SPECS["slow_momentum"].horizon_hours`` stays 168
# (the frozen fallback default).
MOMENTUM_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (
    72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336,
    360, 384, 408, 432, 456, 480, 504,
)
_SLOW_BAND = HorizonBand(name="slow_momentum", horizons_hours=MOMENTUM_HORIZON_CANDIDATES_HOURS, sign=1)

# Measured candidate grid for the funding-rate carry return source
# (docs/specs/mhs_return_source_breadth_expansion.md §2). A measured candidate
# grid -- not a frozen BookSpec -- following the exact governance pattern of
# REVERSAL_HORIZON_CANDIDATES_HOURS/MOMENTUM_HORIZON_CANDIDATES_HOURS: the
# fold-scoped discovery/qualification gate selects from it on fold-train-only
# evidence, and it is explicitly NOT wired into PHASE_1_BOOK_SPECS or
# PHASE_1_BOOK_BLEND_WEIGHTS in this contract (P0 diagnostic only; capital
# allocation is P1 gated on the fold-train results).
MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS: tuple[int, ...] = (24, 72, 168, 336, 504)

PHASE_1_BOOK_SPECS: dict[str, BookSpec] = {
    "fast_reversal": BookSpec(
        band=_FAST_BAND, horizon_hours=48, step_hours=6, min_symbols=8,
    ),
    "slow_momentum": BookSpec(
        band=_SLOW_BAND, horizon_hours=168, step_hours=24, min_symbols=8,
    ),
}

# Preregistered count of sequential discovery trials the 2021-2025 dev search
# has actually accumulated (docs/specs/mhs_strategy_foundation_reset.md RC-6).
# The 20 iterations of horizon-grid/flag/overlay search all ran on the SAME dev
# window, so they are one deliberate multi-trial search and the multiple-testing
# deflation must account for them; ``trials_attempted = 1`` was the un-audited
# placeholder. A revision of this constant is a contract revision, never an
# inline edit at a call site.
MHS_SEARCH_TRIALS_ATTEMPTED: int = 20

# P0-D: the two Research-GO policy gates named in ``_mhs_research_go``'s
# contract are not registered in source. A value of None means "unregistered"
# and keeps Research GO conservative (the gate reports UNSPECIFIED_POLICY);
# registering a value is a deliberate policy act following the
# MEASURED_EXECUTION_COST_TIERS_BPS governance pattern, never an inline literal
# at a call site and never a performance-flattering number.
MHS_REGISTERED_POLICY_THRESHOLDS: dict[str, float | None] = {
    "cap_30_roster": None,
    "primary_annual_return": None,
}
