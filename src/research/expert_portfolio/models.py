"""Core expert-library, router, portfolio request, and LCB models.

Owns the pre-registered definition/spec/request dataclasses and the LCB
quantile constant. This module has no admission dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

_LCB_Z_SCORES: dict[float, float] = {
    0.80: 0.8416212335729143,
    0.85: 1.0364333894937896,
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
    0.99: 2.3263478740408408,
}


def lcb_z_score(confidence: float) -> float:
    """Standard-normal one-sided lower-quantile for a pre-registered confidence level.

    The quantile is a mathematical constant, never a fitted parameter. Only the
    pre-registered levels are supported so the allocator stays deterministic and
    auditable; an unsupported level fails closed.
    """
    try:
        return _LCB_Z_SCORES[float(confidence)]
    except KeyError as exc:  # noqa: PERF203
        raise ValueError(
            f"confidence must be one of {sorted(_LCB_Z_SCORES)}, got {confidence}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ExpertDefinition:
    """Immutable pre-registered definition of one return source.

    ``expert_id`` is a stable identity, ``return_source`` names the economic
    hypothesis, ``family`` groups correlated parameter variants that must share
    an exposure budget, ``symbols`` names the underlying exposures, ``runner``
    is the existing runner that creates the causal return series, and
    ``code_hash`` pins the producing code. Invalid metadata fails closed; a
    definition rejected by anti-pattern evidence is never loadable.
    """

    expert_id: str
    return_source: str
    family: str
    symbols: tuple[str, ...]
    runner: str
    code_hash: str

    def __post_init__(self) -> None:
        if not self.expert_id:
            raise ValueError("expert_id must not be empty")
        if not self.return_source:
            raise ValueError("return_source must not be empty")
        if not self.family:
            raise ValueError("family must not be empty")
        if not self.symbols:
            raise ValueError("symbols must contain at least one underlying symbol")
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError(f"symbols must not contain duplicates, got {self.symbols}")
        if not self.runner:
            raise ValueError("runner must not be empty")
        if not self.code_hash:
            raise ValueError("code_hash must not be empty")


@dataclass(frozen=True, slots=True)
class ContextualRouterSpec:
    """Immutable pre-registered contextual winner router specification.

    ``context_symbol`` names the market whose OHLCV defines the decision
    context, ``trend_lookback_bars`` is the completed trend window,
    ``volatility_lookback_bars`` is the completed rolling-volatility window,
    ``min_context_history_bars`` is the minimum number of completed samples of
    one state before a conditional allocation is permitted, and ``confidence``
    is the lower-confidence-bound level shared with the LCB allocator. Every
    bar count and the confidence level are frozen before the holdout is seen;
    nothing is fitted.
    """

    context_symbol: str
    trend_lookback_bars: int
    volatility_lookback_bars: int
    min_context_history_bars: int
    confidence: float = 0.90

    def __post_init__(self) -> None:
        if not self.context_symbol:
            raise ValueError("context_symbol must not be empty")
        if self.trend_lookback_bars < 1:
            raise ValueError(
                f"trend_lookback_bars must be >= 1, got {self.trend_lookback_bars}"
            )
        if self.volatility_lookback_bars < 1:
            raise ValueError(
                f"volatility_lookback_bars must be >= 1, got {self.volatility_lookback_bars}"
            )
        if self.min_context_history_bars < 1:
            raise ValueError(
                f"min_context_history_bars must be >= 1, got {self.min_context_history_bars}"
            )
        lcb_z_score(self.confidence)


@dataclass(frozen=True, slots=True)
class ExpertPortfolioSpec:
    """Immutable pre-registered expert library and allocator constraints.

    ``experts`` is the eligible library. ``gross_exposure`` caps the total risky
    allocation, ``family_exposure_limit`` caps any single source-family, and
    ``symbol_exposure_limit`` caps any single underlying symbol: correlated
    parameter variants therefore share an exposure budget and cash is always
    feasible. ``min_history_bars`` is the completed-history requirement before
    an expert can receive capital, ``confidence`` is the block-aware lower
    confidence bound level, and ``router`` is the optional pre-registered
    contextual winner router; ``None`` preserves the causal LCB-mix behaviour
    exactly. No constraint is tuned on the sealed result.
    """

    experts: tuple[ExpertDefinition, ...]
    gross_exposure: float = 1.0
    family_exposure_limit: float = 1.0
    symbol_exposure_limit: float = 1.0
    min_history_bars: int = 30
    confidence: float = 0.90
    router: ContextualRouterSpec | None = None

    def __post_init__(self) -> None:
        if not self.experts:
            raise ValueError("experts must contain at least one expert")
        ids = [e.expert_id for e in self.experts]
        if len(ids) != len(set(ids)):
            raise ValueError(f"expert ids must be unique, got {ids}")
        if not 0.0 < self.gross_exposure <= 1.0:
            raise ValueError(
                f"gross_exposure must be in (0, 1], got {self.gross_exposure}"
            )
        if not 0.0 < self.family_exposure_limit <= 1.0:
            raise ValueError(
                f"family_exposure_limit must be in (0, 1], got {self.family_exposure_limit}"
            )
        if not 0.0 < self.symbol_exposure_limit <= 1.0:
            raise ValueError(
                f"symbol_exposure_limit must be in (0, 1], got {self.symbol_exposure_limit}"
            )
        if self.min_history_bars < 1:
            raise ValueError(
                f"min_history_bars must be >= 1, got {self.min_history_bars}"
            )
        lcb_z_score(self.confidence)

    def fingerprint(self) -> dict[str, object]:
        """Deterministic fingerprint over definitions, code hashes, and allocator config.

        A fingerprint changed after registration is a distinct candidate: the
        record binds the evaluation to the exact library that produced it.
        """
        return {
            "experts": [asdict(e) for e in self.experts],
            "gross_exposure": self.gross_exposure,
            "family_exposure_limit": self.family_exposure_limit,
            "symbol_exposure_limit": self.symbol_exposure_limit,
            "min_history_bars": self.min_history_bars,
            "confidence": self.confidence,
            "router": asdict(self.router) if self.router is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ExpertPortfolioEvaluationRequest:
    """Immutable request for one sealed pre-registered expert portfolio evaluation.

    Only a registered ``library_id`` may be supplied; the sealed window flags
    and logging option are the only other switches, so no candidate parameters
    can be tuned on the command line.
    """

    library_id: str
    start: str | None = None
    end: str | pd.Timestamp | None = None
    initial_equity: float = 10_000.0
    unseal_holdout: bool = False
    log_run: bool = True

    def __post_init__(self) -> None:
        if not self.library_id:
            raise ValueError("library_id must not be empty")


__all__ = [
    "ContextualRouterSpec",
    "ExpertDefinition",
    "ExpertPortfolioEvaluationRequest",
    "ExpertPortfolioSpec",
    "lcb_z_score",
]
