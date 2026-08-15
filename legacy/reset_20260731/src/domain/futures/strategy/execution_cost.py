from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ExecutionCostModel:
    """SSOT execution cost model (replaces flat 24bps constant).

    Default: maker_ratio=0.75, RT≈7.5bps, stress≈11.25bps.
    Verification: one_way = 0.75*2 + 0.25*5 + 1 = 3.75bps → RT=7.5bps.
    """

    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    maker_ratio: float = 0.75
    slippage_bps: float = 1.0
    impact_coeff_bps: float = 0.0
    stress_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.maker_ratio <= 1.0):
            raise ValueError("maker_ratio must be in [0.0, 1.0]")
        if self.maker_fee_bps < 0.0:
            raise ValueError("maker_fee_bps must be non-negative")
        if self.taker_fee_bps < 0.0:
            raise ValueError("taker_fee_bps must be non-negative")
        if self.slippage_bps < 0.0:
            raise ValueError("slippage_bps must be non-negative")
        if self.impact_coeff_bps < 0.0:
            raise ValueError("impact_coeff_bps must be non-negative")
        if self.stress_multiplier < 1.0:
            raise ValueError("stress_multiplier must be >= 1.0")

    def one_way_bps(self) -> float:
        fee = self.maker_ratio * self.maker_fee_bps + (1.0 - self.maker_ratio) * self.taker_fee_bps
        return fee + self.slippage_bps + self.impact_coeff_bps

    def round_trip_bps(self) -> float:
        return 2.0 * self.one_way_bps()

    def taker_round_trip_bps(self) -> float:
        """Return two taker fees plus two-way slippage and impact."""
        taker_one_way = self.taker_fee_bps + self.slippage_bps + self.impact_coeff_bps
        return 2.0 * taker_one_way

    def stress_round_trip_bps(self) -> float:
        return self.stress_multiplier * self.round_trip_bps()
