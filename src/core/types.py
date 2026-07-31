from __future__ import annotations

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
