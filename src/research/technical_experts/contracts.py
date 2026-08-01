"""Frozen value object for one directional technical-expert candidate.

The candidate is the smallest unit of the frozen screen: one indicator family,
one direction (LONG/SHORT), one fixed indicator configuration, and the minimum
completed 4h history required before a decision. Every identity and parameter
is source-controlled by the catalog; the CLI never supplies indicator inputs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

CandidateSide = Literal["LONG", "SHORT"]

_RETURN_SOURCE_PATTERN = re.compile(r"^technical_[a-z0-9_]+_(long|short)_v1$")


@dataclass(frozen=True, slots=True)
class TechnicalCandidate:
    """Immutable, source-controlled definition of one directional candidate.

    ``return_source`` is the precise ``technical_<family>_<side>_v1`` identity
    under which the candidate is evaluated and (if ever rejected) retired, so a
    rejected identity can never be re-labelled. ``config`` holds only fixed,
    catalog-supplied indicator inputs; no parameter is user-tunable.
    """

    candidate_id: str
    return_source: str
    family: str
    side: CandidateSide
    config: Mapping[str, int | float]
    min_history_bars: int

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.return_source:
            raise ValueError("return_source must not be empty")
        if not self.family:
            raise ValueError("family must not be empty")
        if self.side not in ("LONG", "SHORT"):
            raise ValueError(f"side must be 'LONG' or 'SHORT', got {self.side!r}")
        if self.min_history_bars < 1:
            raise ValueError(
                f"min_history_bars must be >= 1, got {self.min_history_bars}"
            )
        if not self.config:
            raise ValueError("config must not be empty")
        for name, value in self.config.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"config values must be numeric, got {name}={value!r}"
                )
        match = _RETURN_SOURCE_PATTERN.match(self.return_source)
        if match is None or match.group(1).upper() != self.side:
            raise ValueError(
                f"return_source must be 'technical_<family>_<side>_v1' matching side "
                f"{self.side}, got {self.return_source!r}"
            )
        expected = f"technical_{self.family}_{self.side.lower()}_v1"
        if self.return_source != expected:
            raise ValueError(
                f"return_source {self.return_source!r} does not match family/side "
                f"identity {expected!r}"
            )


def _check_contract() -> None:
    """Executable assertions locking the frozen candidate contract surface."""
    from inspect import signature

    candidate = TechnicalCandidate(
        "x", "technical_ema_alignment_long_v1", "ema_alignment", "LONG",
        {"fast": 20, "mid": 50, "slow": 200}, 201,
    )
    assert candidate.side == "LONG"
    assert candidate.return_source == "technical_ema_alignment_long_v1"
    assert list(signature(TechnicalCandidate).parameters) == [
        "candidate_id", "return_source", "family", "side", "config", "min_history_bars",
    ]


_check_contract()
