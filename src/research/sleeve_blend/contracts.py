from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from src.research.contracts import CostModel

if TYPE_CHECKING:
    from src.research.baseline.backtest import BacktestResult
    from src.research.evaluation.gate_feasibility import GateFeasibility
    from src.research.evaluation.promotion import PromotionResult
    from src.research.evaluation.reliability import (
        FoldDistributionResult,
        ReliabilityGateResult,
    )

_CORE5_V1_SYMBOLS = ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")
_UNIVERSES: dict[str, tuple[str, ...]] = {"core5_v1": _CORE5_V1_SYMBOLS}


@dataclass(frozen=True, slots=True, init=False)
class BlendUniverseSpec:
    """Immutable, source-controlled production sleeve universe.

    ``universe_id`` names one registered universe in ``_UNIVERSES``; ``symbols``
    must equal that universe's exact declared ordered tuple. An empty,
    duplicate, reordered, substituted, or CLI-supplied tuple raises
    ``ValueError``: a change in membership is a new source-controlled id with a
    separate pre-registration record, never a runtime argument.
    """

    universe_id: str
    symbols: tuple[str, ...]

    def __init__(
        self,
        universe_id: str = "core5_v1",
        symbols: tuple[str, ...] | None = None,
    ) -> None:
        declared = _UNIVERSES.get(universe_id)
        if declared is None:
            raise ValueError(
                f"unknown universe_id '{universe_id}'; registered universes "
                f"are {sorted(_UNIVERSES)}"
            )
        resolved = declared if symbols is None else symbols
        if resolved != declared:
            raise ValueError(
                f"symbols must equal the exact declared {universe_id} order "
                f"{declared}, got {resolved}"
            )
        object.__setattr__(self, "universe_id", universe_id)
        object.__setattr__(self, "symbols", resolved)


@dataclass(frozen=True, slots=True)
class CausalLeverageSpec:
    """Frozen parameters of the causal ex-ante leverage schedule.

    At each declared rebalance bar the schedule uses only completed unit-leverage
    marked returns strictly before that bar, computes the trailing realized
    drawdown over ``lookback_days``, and sizes ``abs(mdd_floor) *
    risk_budget_fraction / abs(trailing_mdd)`` bounded by the source-controlled
    ``max_gross_leverage`` hard cap. Before a complete lookback the exposure is
    zero. This is an ex-ante risk control, never a promise that future MDD stays
    inside the budget.
    """

    lookback_days: int = 365
    risk_budget_fraction: float = 0.85
    max_gross_leverage: float = 3.0

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ValueError(f"lookback_days must be >= 1, got {self.lookback_days}")
        if not 0.0 < self.risk_budget_fraction < 1.0:
            raise ValueError(
                f"risk_budget_fraction must be in (0, 1), got {self.risk_budget_fraction}"
            )
        if self.max_gross_leverage < 1.0:
            raise ValueError(
                f"max_gross_leverage must be >= 1.0, got {self.max_gross_leverage}"
            )


@dataclass(frozen=True, slots=True)
class CausalFractionalKellySpec:
    """Frozen parameters of the causal fractional-Kelly exposure policy.

    At each bar ``t`` the estimate uses only completed unit-leverage simple
    returns strictly before ``t`` over a finished ``lookback_days`` window,
    scaled by the fixed ``fraction`` (quarter-Kelly). Full Kelly (``fraction
    >= 1``) is forbidden and the lookback must equal
    ``CausalLeverageSpec.lookback_days`` so the Kelly and MDD caps cover
    identical history.
    """

    fraction: float = 0.25
    lookback_days: int = 365

    def __post_init__(self) -> None:
        if not math.isfinite(self.fraction) or not 0.0 < self.fraction < 1.0:
            raise ValueError(
                f"fraction must be finite and in (0, 1), got {self.fraction}"
            )
        if self.lookback_days < 1:
            raise ValueError(
                f"lookback_days must be >= 1, got {self.lookback_days}"
            )
        if self.lookback_days != CausalLeverageSpec().lookback_days:
            raise ValueError(
                "lookback_days must equal CausalLeverageSpec.lookback_days "
                f"({CausalLeverageSpec().lookback_days}), got {self.lookback_days}"
            )


@dataclass(frozen=True, slots=True)
class PortfolioBlendTournamentRequest:
    """Immutable request for the pre-registered five-source strategy tournament.

    ``universe`` is the fixed production sleeve universe; ``discovery_end`` is
    the chronological boundary of the discovery window and
    ``qualification_interval`` is a pandas frequency/duration (e.g. ``"365D"`` or
    ``"365D"``) for the untouched qualification window that follows it.
    Discovery alone decides membership, weights, and leverage; qualification is
    evaluated once and can never change the selection.
    """

    universe: BlendUniverseSpec = field(default_factory=BlendUniverseSpec)
    discovery_end: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
    )
    qualification_interval: str = "365D"
    start: str | None = None
    end: str | pd.Timestamp | None = None
    costs: CostModel = field(default_factory=CostModel)
    initial_equity: float = 10_000.0

    def __post_init__(self) -> None:
        from pandas.tseries.frequencies import to_offset

        if self.discovery_end.tzinfo is None:
            raise ValueError("discovery_end must be tz-aware UTC")
        if self.initial_equity <= 0:
            raise ValueError(f"initial_equity must be > 0, got {self.initial_equity}")
        if not self.qualification_interval:
            raise ValueError("qualification_interval must not be empty")
        to_offset(self.qualification_interval)  # raises for an invalid frequency


@dataclass(frozen=True, slots=True)
class TournamentCandidateEvidence:
    """One source's immutable discovery/qualification evidence and verdict.

    ``feasibility_binding`` records the binding constraint from
    ``compute_gate_feasibility``; an infeasible or data-integrity-failed
    candidate is CASH/REJECTED with its reason in ``rejected_reason`` and is
    never silently dropped. The qualification gates are evidence only and can
    never change ``admitted``.
    """

    return_source: str
    feasibility: GateFeasibility | None
    feasibility_binding: str | None
    discovery_observation: ReliabilityGateResult | None
    discovery_fold: FoldDistributionResult | None
    discovery_stress: ReliabilityGateResult | None
    discovery_promotion: PromotionResult | None
    qualification_observation: ReliabilityGateResult | None
    qualification_fold: FoldDistributionResult | None
    qualification_stress: ReliabilityGateResult | None
    qualification_promotion: PromotionResult | None
    admitted: bool
    rejected_reason: str | None


@dataclass(frozen=True, slots=True)
class PortfolioBlendTournamentReport:
    """Outcome of one sealed tournament: selection, schedule, and evidence.

    ``base_result``/``stress_result`` are the executed total-equity ledgers of
    the admitted equal-weight blend under base and stressed costs, both driven
    by the identical ``leverage_schedule`` (whose hash is ``schedule_hash``).
    A tournament with no admitted source is an all-CASH ledger with an empty
    ``selected_return_sources``.
    """

    request: PortfolioBlendTournamentRequest
    universe: BlendUniverseSpec
    candidates: tuple[TournamentCandidateEvidence, ...]
    selected_return_sources: tuple[str, ...]
    blend_weights: tuple[float, ...]
    leverage_schedule: pd.Series
    schedule_hash: str
    base_result: BacktestResult
    stress_result: BacktestResult
    qualification_start: pd.Timestamp
    qualification_end: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class FixedSleevePortfolioSpec:
    """Frozen fixed-weight multi-symbol sleeve blend configuration.

    ``symbols`` is the fixed, non-rotating sleeve set (each already-measured as
    independently positive). ``mdd_budget_fraction`` is the fraction of the
    reliability gate's own ``mdd_floor`` budget the calibrated leverage targets,
    an explicit safety margin in ``(0, 1)`` rather than a fitted-to-pass
    constant.
    """

    symbols: tuple[str, ...]
    mdd_budget_fraction: float = 0.85

    def __post_init__(self) -> None:
        if len(self.symbols) < 2:
            raise ValueError(
                f"symbols must contain at least 2 symbols, got {len(self.symbols)}"
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(f"symbols must not contain duplicates, got {self.symbols}")
        if not 0.0 < self.mdd_budget_fraction < 1.0:
            raise ValueError(
                f"mdd_budget_fraction must be in (0, 1), got {self.mdd_budget_fraction}"
            )


FIXED_DIRECTIONAL_SYMBOLS = _CORE5_V1_SYMBOLS


@dataclass(frozen=True, slots=True)
class DirectionalSleeveSpec:
    """Frozen directional funding-gated sleeve configuration.

    ``symbols`` is the fixed, non-rotating 5-symbol set shared with the baseline
    observation window (adding or re-selecting symbols in this window is
    forbidden). ``history_days`` is the completed marked-return lookback used by
    the inverse-volatility risk budget and ``max_symbol_weight`` caps the sum of
    a symbol's long+short weights before renormalization.
    """

    symbols: tuple[str, ...] = FIXED_DIRECTIONAL_SYMBOLS
    history_days: int = 30
    max_symbol_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.symbols != FIXED_DIRECTIONAL_SYMBOLS:
            raise ValueError(
                f"symbols must equal the fixed directional set "
                f"{FIXED_DIRECTIONAL_SYMBOLS}, got {self.symbols}"
            )
        if self.history_days < 1:
            raise ValueError(f"history_days must be >= 1, got {self.history_days}")
        if not 0.0 < self.max_symbol_weight <= 1.0:
            raise ValueError(
                f"max_symbol_weight must be in (0, 1], got {self.max_symbol_weight}"
            )
