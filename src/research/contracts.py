from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from src.research.baseline.backtest import BacktestResult
    from src.research.evaluation.metrics import Metrics
    from src.research.evaluation.promotion import PromotionResult
    from src.research.evaluation.reliability import (
        FoldDistributionResult,
        ReliabilityGateResult,
    )


@dataclass(frozen=True, slots=True)
class StrategySpec:
    symbol: str = "BTCUSDT"
    timeframe: str = "4h"
    entry_period: int = 20
    exit_period: int = 10
    ema_period: int = 200
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    risk_per_trade: float = 0.005
    max_leverage: float = 2.0
    max_positions: int = 1
    allow_same_bar_reentry: bool = False
    ambiguous_bar_policy: Literal["stop_first"] = "stop_first"
    min_taker_buy_ratio: float | None = None

    def __post_init__(self) -> None:
        import math

        for name, val in [
            ("entry_period", self.entry_period),
            ("exit_period", self.exit_period),
            ("ema_period", self.ema_period),
            ("atr_period", self.atr_period),
        ]:
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        if self.stop_atr_mult <= 0:
            raise ValueError(f"stop_atr_mult must be > 0, got {self.stop_atr_mult}")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(f"risk_per_trade must be in (0, 1], got {self.risk_per_trade}")
        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {self.max_leverage}")
        if (
            self.min_taker_buy_ratio is not None
            and (not math.isfinite(self.min_taker_buy_ratio)
                 or not 0 < self.min_taker_buy_ratio <= 1)
        ):
            raise ValueError(
                f"min_taker_buy_ratio must be finite and in (0, 1] when set, "
                f"got {self.min_taker_buy_ratio}"
            )


@dataclass(frozen=True, slots=True)
class PortfolioSpec:
    """Immutable portfolio-execution configuration, separate from StrategySpec.

    Deliberately carries no signal or performance parameters: it only fixes the
    number of liquidity slots, the maximum concurrent positions, and the trailing
    liquidity lookback. Single-symbol StrategySpec defaults are never modified.
    """

    universe_size: int = 5
    max_positions: int = 5
    liquidity_lookback_days: int = 30

    def __post_init__(self) -> None:
        if not self.universe_size >= self.max_positions >= 1:
            raise ValueError(
                f"universe_size >= max_positions >= 1 required, got "
                f"universe_size={self.universe_size} max_positions={self.max_positions}"
            )
        if self.liquidity_lookback_days < 1:
            raise ValueError(
                f"liquidity_lookback_days must be >= 1, got {self.liquidity_lookback_days}"
            )


@dataclass(frozen=True, slots=True)
class CostModel:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0003

    def __post_init__(self) -> None:
        if self.fee_rate < 0:
            raise ValueError(f"fee_rate must be >= 0, got {self.fee_rate}")
        if self.slippage_rate < 0:
            raise ValueError(f"slippage_rate must be >= 0, got {self.slippage_rate}")

    def round_trip_bps(self) -> float:
        return 2 * self.fee_rate * 10000 + 2 * self.slippage_rate * 10000

    def buy_fill(self, price: float) -> float:
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        return price * (1 + self.slippage_rate)

    def sell_fill(self, price: float) -> float:
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        return price * (1 - self.slippage_rate)


@dataclass(frozen=True, slots=True)
class BaselineEvaluationRequest:
    """Immutable request for a single-symbol Donchian evaluation."""

    symbol: str = "BTCUSDT"
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    min_taker_buy_ratio: float | None = None
    funding_path: str | None = None
    unseal_holdout: bool = False
    log_run: bool = True


@dataclass(frozen=True, slots=True)
class PortfolioEvaluationRequest:
    """Immutable request for a causal liquidity portfolio evaluation."""

    symbols: tuple[str, ...]
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    unseal_holdout: bool = False
    log_run: bool = True


@dataclass(frozen=True, slots=True)
class CashCarryEvaluationRequest:
    """Immutable request for a sealed cash-and-carry research evaluation."""

    symbol: str
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    unseal_holdout: bool = False
    log_run: bool = True


@dataclass(frozen=True, slots=True)
class OIDeleveragingEvaluationRequest:
    """Immutable request for the sealed open-interest deleveraging screen.

    ``symbol`` is the single fixed-universe symbol evaluated with the immutable
    one-day deleveraging hypothesis; ``unseal_holdout`` is the only way to
    extend past the sealed observation end, and ``log_run`` controls the single
    provenance record.
    """

    symbol: str
    start: str | None = None
    end: str | pd.Timestamp | None = None
    unseal_holdout: bool = False
    log_run: bool = True


@dataclass(frozen=True, slots=True)
class SleeveBlendEvaluationRequest:
    """Immutable request for a fixed-sleeve, directional-sleeve, or tournament evaluation.

    ``candidate_kind`` selects the sealed candidate: ``"fixed_long_only_v1"`` is
    the existing MDD-budget equal-weight long-only blend and
    ``"funding_signed_directional_v1"`` is the funding-gated long/short sleeve
    (both retained strictly as read-only controls), while
    ``"core5_causal_tournament_v1`` is the pre-registered five-source strategy
    tournament on the fixed ``universe_id`` with a causal leverage schedule.
    For the tournament kind, ``discovery_end`` is the chronological boundary of
    the discovery window and ``qualification_interval`` the untouched window
    that follows it; ``symbols`` is then ignored in favor of the universe's
    exact declared membership.
    """

    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")
    universe_id: str = "core5_v1"
    mdd_budget_fraction: float = 0.85
    candidate_kind: Literal[
        "fixed_long_only_v1",
        "funding_signed_directional_v1",
        "core5_causal_tournament_v1",
    ] = "fixed_long_only_v1"
    discovery_end: str | pd.Timestamp | None = None
    qualification_interval: str = "365D"
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    unseal_holdout: bool = False
    log_run: bool = True

    def __post_init__(self) -> None:
        if self.candidate_kind not in (
            "fixed_long_only_v1",
            "funding_signed_directional_v1",
            "core5_causal_tournament_v1",
        ):
            raise ValueError(
                f"candidate_kind must be one of "
                f"'fixed_long_only_v1'/'funding_signed_directional_v1'/"
                f"'core5_causal_tournament_v1', got {self.candidate_kind}"
            )
        from src.research.sleeve_blend.contracts import BlendUniverseSpec

        BlendUniverseSpec(universe_id=self.universe_id)
        if self.candidate_kind == "core5_causal_tournament_v1":
            if self.discovery_end is None:
                raise ValueError(
                    "discovery_end is required for candidate_kind "
                    "'core5_causal_tournament_v1'"
                )
            end_ts = pd.Timestamp(self.discovery_end)
            if end_ts.tzinfo is None:
                raise ValueError("discovery_end must be tz-aware UTC")
            if not self.qualification_interval:
                raise ValueError("qualification_interval must not be empty")


@dataclass(frozen=True, slots=True)
class TechnicalExpertEvaluationRequest:
    """Immutable request for one sealed technical-expert candidate screen.

    ``candidate_id`` names exactly one of the eighteen frozen directional
    candidates and ``symbol`` one of the five fixed liquid symbols; the sealed
    window flags and logging option are the only other switches, so no indicator
    or threshold can be tuned on the command line.
    """

    candidate_id: str
    symbol: str
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    unseal_holdout: bool = False
    log_run: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.initial_equity <= 0:
            raise ValueError(f"initial_equity must be > 0, got {self.initial_equity}")


@dataclass(frozen=True)
class EvaluationReport:
    """Composed result of a sealed evaluation executed by an application service.

    Carries the frozen backtest result, metrics, every gate verdict, and the
    composed promotion. ``status`` is ``"PASS"`` for a fully evaluated run and
    ``"PENDING"`` for a fail-closed early return (e.g. missing data under the
    sealed window). ``record`` is the appended JSONL row when ``log_run`` was
    requested.
    """

    status: str
    result: BacktestResult
    metrics: Metrics
    observation: ReliabilityGateResult
    fold_distribution: FoldDistributionResult
    stress: ReliabilityGateResult
    holdout: ReliabilityGateResult | None
    promotion: PromotionResult
    record: dict[str, object] | None = None
