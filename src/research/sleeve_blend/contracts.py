from __future__ import annotations

from dataclasses import dataclass


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


FIXED_DIRECTIONAL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")


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
