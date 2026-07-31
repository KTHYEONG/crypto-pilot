from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


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
class CashCarrySpec:
    """Immutable cash-and-carry execution configuration.

    Fixes the margin model only: initial margin is reserved before opening the
    short leg and the maintenance buffer triggers a forced close when violated.
    No fitted alpha parameter and no directionally exposed sizing policy.
    """

    symbol: str
    initial_margin_rate: float = 0.10
    maintenance_margin_rate: float = 0.05

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not 0 < self.initial_margin_rate <= 1:
            raise ValueError(
                f"initial_margin_rate must be in (0, 1], got {self.initial_margin_rate}"
            )
        if not 0 < self.maintenance_margin_rate < self.initial_margin_rate:
            raise ValueError(
                "maintenance_margin_rate must be in (0, initial_margin_rate), got "
                f"{self.maintenance_margin_rate}"
            )


@dataclass(frozen=True, slots=True)
class CarryCostModel:
    """Two-leg cost model with venue-specific fees.

    Binance spot maker and taker fees are both modelled at 0.1% by default.
    The perpetual leg retains its own explicit fee schedule.
    """

    spot_fee_rate: float = 0.001
    perp_fee_rate: float = 0.0005
    slippage_rate: float = 0.0003

    def __post_init__(self) -> None:
        if self.spot_fee_rate < 0:
            raise ValueError(f"spot_fee_rate must be >= 0, got {self.spot_fee_rate}")
        if self.perp_fee_rate < 0:
            raise ValueError(f"perp_fee_rate must be >= 0, got {self.perp_fee_rate}")
        if self.slippage_rate < 0:
            raise ValueError(f"slippage_rate must be >= 0, got {self.slippage_rate}")


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
