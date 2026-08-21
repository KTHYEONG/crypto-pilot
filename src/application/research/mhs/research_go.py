"""MHS Research-GO gate: fail-closed decision from fold and book evidence.

This module owns the GO reason-code constants and the top-level Research-GO
decision logic. It composes the frozen ``src.mhs`` primitives and the
application request/report contracts; no alpha, cost, ranking, liquidity,
funding, or inventory arithmetic is reimplemented here.
"""

from __future__ import annotations

import numpy as np

from src.application.research.mhs.contracts import (
    MhsDiagnosticRequest,
    MhsFoldReport,
    MhsResearchGoResult,
)
from src.mhs.params import COMMITTEE_TARGET_GROSS_UNSET
from src.mhs.types import (
    COMMITTEE_GROWTH_MAX_DRAWDOWN,
    COMMITTEE_TARGET_GROSS,
    REGISTERED_POLICY_THRESHOLDS,
)

GO_REASON_INCOMPLETE_FOLD = "INCOMPLETE_ANCHORED_FOLD"
GO_REASON_INVALID_PRIMARY = "INVALID_PRIMARY_LEDGER"
GO_REASON_NONFINITE_EQUITY = "NONFINITE_EQUITY"
GO_REASON_EXECUTION_GAP = "RELEVANT_EXECUTION_DATA_GAP"
GO_REASON_PRIMARY_SHARPE = "PRIMARY_AUTOCORR_SHARPE_BELOW_0_6"
GO_REASON_STRESS_SHARPE = "STRESS_SHARPE_NOT_POSITIVE"
GO_REASON_PRIMARY_RETURN_BELOW_FLOOR = "PRIMARY_ANNUAL_RETURN_BELOW_FLOOR"
GO_REASON_CAPITAL_BREACH = "CAPITAL_INVARIANT_BREACH"
GO_REASON_UNSPECIFIED_POLICY = "UNSPECIFIED_POLICY"
GO_REASON_RESOURCE_BREACH = "RESOURCE_BUDGET_BREACH"
GO_REASON_PATH_DIVERGENCE = "FOLD_BLEND_PATH_DIVERGENCE"
GO_REASON_FOLD_GROWTH_CONCENTRATION = "FOLD_GROWTH_CONCENTRATION"
# Blocking alpha/risk code for a completed blend whose realized drawdown exceeds
# the registered budget. NOT a data-integrity reason: the data was intact; the
# risk contract was exceeded.
GO_REASON_DRAWDOWN_OVER_BUDGET = "PRIMARY_MAX_DRAWDOWN_OVER_BUDGET"
# Data-integrity reason codes: fail-closed evidence that the canonical input
# data itself was missing or invalid, as opposed to pure alpha-quality failures
# (GO_REASON_PRIMARY_SHARPE / GO_REASON_STRESS_SHARPE) or the policy
# registration state (GO_REASON_UNSPECIFIED_POLICY). Consumers distinguish
# "data was intact but alpha underperformed" from "data itself was deficient"
# by whether MhsResearchGoResult.data_integrity_reason_codes is non-empty.
GO_REASON_DATA_INTEGRITY_CODES = frozenset[str]({
    GO_REASON_INCOMPLETE_FOLD,
    GO_REASON_INVALID_PRIMARY,
    GO_REASON_NONFINITE_EQUITY,
    GO_REASON_EXECUTION_GAP,
    GO_REASON_CAPITAL_BREACH,
    GO_REASON_RESOURCE_BREACH,
    GO_REASON_PATH_DIVERGENCE,
})


def _resolved_committee_target_gross(request: MhsDiagnosticRequest) -> float | None:
    """The effective committee target gross: the registered default when the
    caller never set the field, else the caller's explicit value (including
    an explicit ``None``, which keeps the diluted book)."""
    if request.committee_target_gross is COMMITTEE_TARGET_GROSS_UNSET:
        return COMMITTEE_TARGET_GROSS
    return request.committee_target_gross


def _resolved_committee_members(request: MhsDiagnosticRequest) -> tuple[str, ...]:
    """Single resolution seam for committee member set (invariant I4).

    Returns the member tuple from COMMITTEE_MEMBER_SETS keyed by
    request.committee_member_set. An unregistered key raises ValueError
    naming the registered keys.
    """
    from src.mhs.params import COMMITTEE_MEMBER_SETS

    key = request.committee_member_set
    if key not in COMMITTEE_MEMBER_SETS:
        registered = sorted(COMMITTEE_MEMBER_SETS)
        raise ValueError(
            f"unknown committee_member_set '{key}'; "
            f"registered keys: {registered}"
        )
    return COMMITTEE_MEMBER_SETS[key]


def _drawdown_budget_reasons(
    primary_max_drawdown: float | None,
    max_drawdown: float = COMMITTEE_GROWTH_MAX_DRAWDOWN,
) -> tuple[str, ...]:
    """Pure risk-contract gate: a completed blend breaching the drawdown budget.

    Returns ``(GO_REASON_DRAWDOWN_OVER_BUDGET,)`` iff ``primary_max_drawdown``
    is a finite float strictly below ``-max_drawdown``; ``()`` for ``None`` or
    non-finite (an absent replay is already blocked by its own code). Raises
    ``ValueError`` when ``max_drawdown <= 0``.
    """
    if max_drawdown <= 0:
        raise ValueError(f"max_drawdown must be > 0, got {max_drawdown}")
    if primary_max_drawdown is None or not np.isfinite(primary_max_drawdown):
        return ()
    if primary_max_drawdown < -max_drawdown:
        return (GO_REASON_DRAWDOWN_OVER_BUDGET,)
    return ()


def _mhs_research_go(
    folds: tuple[MhsFoldReport, ...],
    book_reasons: tuple[str, ...] = (),
    extra_reasons: tuple[str, ...] = (),
    blend_primary_max_drawdown: float | None = None,
) -> MhsResearchGoResult:
    """Fail-closed top-level Research-GO decision from fold and book evidence.

    A fold that was not replayed, an invalid primary, non-finite equity, a
    relevant execution gap, a strict-Sharpe failure, or a non-positive stress
    Sharpe each block the decision with a stable reason code. A book-level
    strict replay rejection (capital invariant breach, execution gap, invalid
    primary, or resource-budget breach) is aggregated with the fold reasons.
    ``extra_reasons`` carries observational gate codes (e.g. fold/blend path
    divergence) surfaced by report assembly. ``blend_primary_max_drawdown``
    feeds the registered drawdown-budget gate: a completed blend whose realized
    drawdown breaches ``COMMITTEE_GROWTH_MAX_DRAWDOWN`` blocks the decision
    with ``PRIMARY_MAX_DRAWDOWN_OVER_BUDGET`` (a risk-contract code, never a
    data-integrity code). The cap-30 roster and primary annual-return gate
    thresholds live in ``REGISTERED_POLICY_THRESHOLDS``; while any is
    unregistered (``None``) the decision reports ``UNSPECIFIED_POLICY`` and
    stays conservative (false).
    """
    reasons: list[str] = [
        *book_reasons,
        *extra_reasons,
        *_drawdown_budget_reasons(blend_primary_max_drawdown),
    ]
    passed = 0
    for fold_report in folds:
        if fold_report.strict is None:
            reasons.append(GO_REASON_INCOMPLETE_FOLD)
            continue
        if not fold_report.failures:
            passed += 1
        reasons.extend(fold_report.failures)
    # P0-D: UNSPECIFIED_POLICY only when a registered policy threshold is absent.
    if any(v is None for v in REGISTERED_POLICY_THRESHOLDS.values()):
        reasons.append(GO_REASON_UNSPECIFIED_POLICY)
    reasons = sorted(set(reasons))
    data_integrity_reasons = tuple(
        sorted(r for r in reasons if r in GO_REASON_DATA_INTEGRITY_CODES)
    )
    return MhsResearchGoResult(
        eligible=not reasons,
        reason_codes=tuple(reasons),
        evaluated_folds=len(folds),
        folds_passed=passed,
        data_integrity_reason_codes=data_integrity_reasons,
    )
