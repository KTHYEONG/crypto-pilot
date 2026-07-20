from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

CrisisWindowStatus = Literal[
    "pending",
    "stress_data_invalid",
    "stress_tested_fail",
    "stress_tested_pass",
]
CrisisAssessmentStatus = Literal[
    "untested_no_data",
    "stress_data_invalid",
    "stress_tested_fail",
    "stress_tested_pass",
]


@dataclass(slots=True, frozen=True)
class CrisisWindowMetrics:
    """Immutable metrics captured from one crisis replay window. [ADR_20260717_L2_CRISIS_SURVIVAL_POLICY]"""

    label: str
    status: CrisisWindowStatus
    detail: str
    symbol_count: int
    observation_days: int
    bar_count: int
    event_count: int
    trade_count: int
    mdd: float | None
    cagr: float | None
    cvar_95: float | None


@dataclass(slots=True, frozen=True)
class CrisisReliabilityAssessment:
    """Aggregate crisis survival decision across every configured window. [ADR_20260717_L2_CRISIS_SURVIVAL_POLICY]"""

    status: CrisisAssessmentStatus
    verified: bool
    detail: str
    window_results: tuple[CrisisWindowMetrics, ...]
    blockers: tuple[str, ...]
    usable_window_count: int


def evaluate_crisis_survival(
    window_results: tuple[CrisisWindowMetrics, ...],
    *,
    max_mdd_abs: float,
    min_cagr: float,
    max_cvar_95: float,
    min_symbols: int,
    min_observation_days: int,
    min_trades: int,
    min_usable_windows: int,
) -> CrisisReliabilityAssessment:
    """Evaluate immutable metrics against survival limits. [ADR_20260717_L2_CRISIS_SURVIVAL_POLICY]

    Placeholder statuses from creation time are replaced with canonical values:
    ``stress_tested_pass``, ``stress_tested_fail``, or ``stress_data_invalid``.
    """
    if max_mdd_abs <= 0.0 or min_usable_windows < 1 or min_symbols < 1 or min_observation_days < 1 or min_trades < 1:
        raise ValueError(
            f"thresholds must be positive: max_mdd_abs={max_mdd_abs}, min_usable_windows={min_usable_windows}, "
            f"min_symbols={min_symbols}, min_observation_days={min_observation_days}, min_trades={min_trades}"
        )

    blockers: list[str] = []
    all_pass = True
    usable_count = 0

    evaluated: list[CrisisWindowMetrics] = []

    for wm in window_results:
        if wm.mdd is None or wm.cagr is None or wm.cvar_95 is None:
            evaluated.append(_replace_status(wm, "stress_data_invalid"))
            continue

        if wm.symbol_count < min_symbols:
            blockers.append(f"{wm.label}:symbols")
            all_pass = False
            evaluated.append(_replace_status(wm, "stress_data_invalid"))
            continue
        if wm.observation_days < min_observation_days:
            blockers.append(f"{wm.label}:observation_days")
            all_pass = False
            evaluated.append(_replace_status(wm, "stress_data_invalid"))
            continue
        if wm.trade_count < min_trades:
            blockers.append(f"{wm.label}:trades")
            all_pass = False
            evaluated.append(_replace_status(wm, "stress_data_invalid"))
            continue

        if not (math.isfinite(wm.mdd) and math.isfinite(wm.cagr) and math.isfinite(wm.cvar_95)):
            evaluated.append(_replace_status(wm, "stress_data_invalid"))
            all_pass = False
            continue

        usable_count += 1
        window_blockers: list[str] = []

        if wm.mdd > max_mdd_abs:
            window_blockers.append(f"{wm.label}:mdd_abs")
        if wm.cagr < min_cagr:
            window_blockers.append(f"{wm.label}:cagr")
        if wm.cvar_95 > max_cvar_95:
            window_blockers.append(f"{wm.label}:cvar_95")

        if window_blockers:
            all_pass = False
            blockers.extend(window_blockers)
            evaluated.append(_replace_status(wm, "stress_tested_fail"))
        else:
            evaluated.append(_replace_status(wm, "stress_tested_pass"))

    evaluated_tuple = tuple(evaluated)

    if usable_count < min_usable_windows:
        detail = (
            f"usable_windows={usable_count} < min_usable_windows={min_usable_windows}: "
            f"insufficient usable crisis windows"
        )
        return CrisisReliabilityAssessment(
            status="untested_no_data",
            verified=False,
            detail=detail,
            window_results=evaluated_tuple,
            blockers=tuple(blockers) if blockers else ("insufficient_usable_windows",),
            usable_window_count=usable_count,
        )

    if not all_pass:
        detail = "; ".join(blockers) if blockers else "window metric failure"
        return CrisisReliabilityAssessment(
            status="stress_tested_fail",
            verified=False,
            detail=detail,
            window_results=evaluated_tuple,
            blockers=tuple(blockers),
            usable_window_count=usable_count,
        )

    return CrisisReliabilityAssessment(
        status="stress_tested_pass",
        verified=True,
        detail=f"all {len(window_results)} windows pass",
        window_results=evaluated_tuple,
        blockers=(),
        usable_window_count=usable_count,
    )


def _replace_status(wm: CrisisWindowMetrics, new_status: CrisisWindowStatus) -> CrisisWindowMetrics:
    """Return a new ``CrisisWindowMetrics`` with the status replaced."""
    import dataclasses
    return dataclasses.replace(wm, status=new_status)
