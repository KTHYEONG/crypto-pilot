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
