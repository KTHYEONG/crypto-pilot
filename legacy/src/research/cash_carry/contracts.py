from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class CashCarrySpec:
    """Immutable cash-and-carry execution configuration.

    Fixes the margin model only: initial margin is reserved before opening the
    short leg and the maintenance buffer triggers a forced close when violated.
    No fitted alpha parameter and no directionally exposed sizing policy.
    """

    symbol: str
    initial_margin_rate: float = 0.30
    maintenance_margin_rate: float = 0.15

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
class CarryHysteresisConfig:
    """Cost-derived hysteresis band for the cash-and-carry signal.

    ``lookback_settlements`` is the trailing settlement window over which the
    net-carry rate is averaged and over which the round-trip cost is amortized
    into a breakeven rate; ``min_hold_settlements`` is the minimum number of
    fresh-settlement bars a position must survive before any CLOSE is
    considered; ``confirm_settlements`` is the number of consecutive negative
    readings required to close a matured position. Defaults are structurally
    derived ratios (min_hold == lookback, confirm == ceil(lookback/3)), not
    independently fitted constants.
    """

    lookback_settlements: int = 21
    min_hold_settlements: int = 21
    confirm_settlements: int = 7

    def __post_init__(self) -> None:
        if self.lookback_settlements < 1:
            raise ValueError(
                f"lookback_settlements must be >= 1, got {self.lookback_settlements}"
            )
        if self.min_hold_settlements < 1:
            raise ValueError(
                f"min_hold_settlements must be >= 1, got {self.min_hold_settlements}"
            )
        if self.confirm_settlements < 1:
            raise ValueError(
                f"confirm_settlements must be >= 1, got {self.confirm_settlements}"
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
class CarryMarketData:
    """Aligned cash-and-carry research inputs for one symbol.

    ``spot`` and ``perp`` are identical-grid tz-aware OHLCV frames of the same
    asset; ``funding`` carries the actual settlement timestamps of the perpetual
    short leg (variable sub-eight-hour cadence is allowed, never a fixed
    three-events-per-day assumption); ``borrow`` holds the per-bar finite
    quote-cash financing rate. The grid is the spot index.
    """

    symbol: str
    spot: pd.DataFrame
    perp: pd.DataFrame
    funding: pd.Series
    borrow: pd.Series
