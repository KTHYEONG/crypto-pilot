"""MHS Research-GO gate: fail-closed decision from fold and book evidence.

This module owns the GO reason-code constants and the top-level Research-GO
decision logic. It composes the frozen ``src.mhs`` primitives and the
application request/report contracts; no alpha, cost, ranking, liquidity,
funding, or inventory arithmetic is reimplemented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.research.mhs.contracts import (
    MhsDiagnosticRequest,
    MhsFoldReport,
    MhsResearchGoResult,
)
from src.mhs.params import (
    COMMITTEE_TARGET_GROSS_UNSET,
    GO_PRIMARY_SHARPE_FLOOR,
    GROWTH_RISK_ENVELOPES,
    GrowthRiskEnvelope,
)
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
# Risk-contract code for a drawdown budget that can never bind: an envelope
# whose max_drawdown exceeds the registered ceiling cannot gate any realized
# drawdown (-100% is capital extinction), so judging GO under it would
# silently disable the risk gate instead of blocking.
GO_REASON_DRAWDOWN_BUDGET_NON_BINDING = "DRAWDOWN_BUDGET_NON_BINDING"
# Risk-contract codes for the Deflated Sharpe Ratio gate (never data-integrity
# codes): a DSR under the registered threshold blocks the decision, and a
# missing/non-finite DSR fails closed instead of passing silently.
GO_REASON_DEFLATED_SHARPE_BELOW_THRESHOLD = "DEFLATED_SHARPE_BELOW_THRESHOLD"
GO_REASON_DEFLATED_SHARPE_UNAVAILABLE = "DEFLATED_SHARPE_UNAVAILABLE"
# Observational disclosure code: the report window intersects the window the
# CLI defaults were selected on. Surfaced through extra_reasons; it never
# blocks the decision by itself.
GO_REASON_SELECTION_WINDOW_OVERLAP = "SELECTION_WINDOW_OVERLAP"
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


def _resolved_growth_envelope(request: MhsDiagnosticRequest) -> GrowthRiskEnvelope:
    """Single resolution seam for growth risk envelope (I3: single seam).

    Returns the ``GrowthRiskEnvelope`` keyed by ``request.growth_envelope``.
    An unregistered key raises ``ValueError`` naming the sorted registered
    keys, mirroring ``_resolved_committee_members``.
    """
    key = request.growth_envelope
    if key not in GROWTH_RISK_ENVELOPES:
        registered = sorted(GROWTH_RISK_ENVELOPES)
        raise ValueError(
            f"unknown growth_envelope '{key}'; "
            f"registered keys: {registered}"
        )
    return GROWTH_RISK_ENVELOPES[key]


def _drawdown_budget_reasons(
    primary_max_drawdown: float | None,
    max_drawdown: float = COMMITTEE_GROWTH_MAX_DRAWDOWN,
) -> tuple[str, ...]:
    """Pure risk-contract gate: a completed blend breaching the drawdown budget.

    A budget above the registered ``max_drawdown_budget_ceiling`` can never
    bind, so it blocks with ``(GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,)``
    regardless of the observed drawdown -- silence would let an unenforceable
    risk contract pass. Otherwise returns
    ``(GO_REASON_DRAWDOWN_OVER_BUDGET,)`` iff ``primary_max_drawdown`` is a
    finite float strictly below ``-max_drawdown``; ``()`` for ``None`` or
    non-finite (an absent replay is already blocked by its own code). Raises
    ``ValueError`` when ``max_drawdown <= 0``.
    """
    if max_drawdown <= 0:
        raise ValueError(f"max_drawdown must be > 0, got {max_drawdown}")
    ceiling = REGISTERED_POLICY_THRESHOLDS.get("max_drawdown_budget_ceiling")
    if ceiling is not None and max_drawdown > float(ceiling):
        return (GO_REASON_DRAWDOWN_BUDGET_NON_BINDING,)
    if primary_max_drawdown is None or not np.isfinite(primary_max_drawdown):
        return ()
    if primary_max_drawdown < -max_drawdown:
        return (GO_REASON_DRAWDOWN_OVER_BUDGET,)
    return ()


def _pooled_level_gate_reasons(
    pooled_evidence: dict[str, Any],
    sharpe_floor: float = GO_PRIMARY_SHARPE_FLOOR,
    return_floor: float | None = None,
) -> tuple[str, ...]:
    """Level-family gate: pooled lower bounds vs registered absolute floors.

    I-NO-CIRCULAR: the comparison is against absolute economic floors, never a
    null bootstrapped from the strategy's own returns (a zero-edge strategy
    would pass its own null). Fewer than two measurable folds defers entirely
    to ``INCOMPLETE_ANCHORED_FOLD``, which already blocks upstream, so no level
    codes are added here.
    """
    if int(pooled_evidence.get("n_measured_folds") or 0) < 2:
        return ()
    reasons: list[str] = []
    if float(pooled_evidence["pooled_sharpe_lcb"]) <= sharpe_floor:
        reasons.append(GO_REASON_PRIMARY_SHARPE)
    if float(pooled_evidence["pooled_stress_sharpe_lcb"]) <= 0.0:
        reasons.append(GO_REASON_STRESS_SHARPE)
    effective_return_floor = (
        REGISTERED_POLICY_THRESHOLDS["primary_annual_return"]
        if return_floor is None
        else return_floor
    )
    if effective_return_floor is not None and (
        float(pooled_evidence["pooled_annual_log_return"]) < effective_return_floor
    ):
        reasons.append(GO_REASON_PRIMARY_RETURN_BELOW_FLOOR)
    return tuple(reasons)


def _deflated_sharpe_gate_reasons(
    deflated_sharpe_ratio: float | None,
) -> tuple[str, ...]:
    """DSR gate: fail-closed under the registered ``deflated_sharpe_ratio`` cut.

    A missing or non-finite DSR blocks with ``DEFLATED_SHARPE_UNAVAILABLE``
    (silence would let uncorrected selection bias through), and a finite DSR
    strictly below ``REGISTERED_POLICY_THRESHOLDS['deflated_sharpe_ratio']``
    blocks with ``DEFLATED_SHARPE_BELOW_THRESHOLD``. An unregistered threshold
    is treated as unavailable so the gate can never pass by omission.
    """
    threshold = REGISTERED_POLICY_THRESHOLDS.get("deflated_sharpe_ratio")
    if deflated_sharpe_ratio is None or not np.isfinite(deflated_sharpe_ratio) or (
        threshold is None
    ):
        return (GO_REASON_DEFLATED_SHARPE_UNAVAILABLE,)
    if float(deflated_sharpe_ratio) < float(threshold):
        return (GO_REASON_DEFLATED_SHARPE_BELOW_THRESHOLD,)
    return ()


def _mhs_research_go(
    folds: tuple[MhsFoldReport, ...],
    book_reasons: tuple[str, ...] = (),
    extra_reasons: tuple[str, ...] = (),
    blend_primary_max_drawdown: float | None = None,
    max_drawdown: float = COMMITTEE_GROWTH_MAX_DRAWDOWN,
    deflated_sharpe_ratio: float | None = None,
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
    drawdown breaches ``max_drawdown`` (the caller's resolved
    ``GrowthRiskEnvelope.max_drawdown`` -- ``COMMITTEE_GROWTH_MAX_DRAWDOWN`` by
    default, i.e. the ``conservative`` envelope) blocks the decision with
    ``PRIMARY_MAX_DRAWDOWN_OVER_BUDGET`` (a risk-contract code, never a
    data-integrity code). A budget above the registered drawdown-budget
    ceiling blocks with ``DRAWDOWN_BUDGET_NON_BINDING`` regardless of the
    observed drawdown. ``deflated_sharpe_ratio`` feeds the DSR gate against
    ``REGISTERED_POLICY_THRESHOLDS['deflated_sharpe_ratio']``: below-threshold
    evidence blocks with ``DEFLATED_SHARPE_BELOW_THRESHOLD`` and an absent DSR
    fails closed with ``DEFLATED_SHARPE_UNAVAILABLE``; the gate can only turn
    ``eligible`` from True to False, never False to True. The cap-30 roster and
    primary annual-return gate thresholds live in
    ``REGISTERED_POLICY_THRESHOLDS``; while any is unregistered (``None``) the
    decision reports ``UNSPECIFIED_POLICY`` and stays conservative (false).
    """
    reasons: list[str] = [
        *book_reasons,
        *extra_reasons,
        *_drawdown_budget_reasons(blend_primary_max_drawdown, max_drawdown),
        *_deflated_sharpe_gate_reasons(deflated_sharpe_ratio),
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
