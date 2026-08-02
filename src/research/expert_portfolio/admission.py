"""Pure vectorized library admission evaluator.

The evaluator treats the whole requested candidate universe as one sealed
diagnostic: it validates the common-index completed return panel, filters
candidates on integrity and activity evidence, bounds the exact structural
combination count (failing closed when the universe is too large), computes the
pairwise completed log-return correlation and joint-negative-return
compatibility matrices once, and enumerates every feasible expert subset with
deterministic backtracking. Only the supplied contextual-router state coverage
decides proposal eligibility; no return-based ranking ever occurs.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.expert_portfolio.admission_reports import LibraryAdmissionReport
from src.research.expert_portfolio.admission_types import (
    AdmissionProposal,
    CandidateAdmissionResult,
    LibraryAdmissionConfig,
)
from src.research.expert_portfolio.contextual_router import state_labels
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition

_UNAVAILABLE = "unavailable"


def pairwise_log_return_correlation(completed_returns: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of completed log returns (one matrix, no 3-D array)."""
    return np.asarray(np.corrcoef(np.log1p(completed_returns), rowvar=False))


def pairwise_joint_negative_rates(completed_returns: np.ndarray) -> np.ndarray:
    """Pairwise joint-negative-return rates via a negative-incidence matrix product."""
    negative = (completed_returns < 0.0).astype(np.float64)
    return np.asarray(negative.T @ negative / completed_returns.shape[0])


def evaluate_library_admission(
    component_returns: pd.DataFrame,
    component_trade_counts: Mapping[str, int],
    experts: tuple[ExpertDefinition, ...],
    decision_context: pd.Series,
    router: ContextualRouterSpec,
    config: LibraryAdmissionConfig,
) -> LibraryAdmissionReport:
    """Diagnose every feasible expert subset under the sealed admission limits.

    Returns a ``LibraryAdmissionReport`` with ``status == "FAIL_CLOSED"`` and no
    proposals when the exact feasible structural-combination count exceeds
    ``config.max_combinations``; otherwise every pair-compatible structural
    combination in ``min_experts..max_experts`` is enumerated in lexical
    ``expert_id`` order and marked eligible only when the router state coverage
    is sufficient. A malformed panel or misaligned context fails closed with an
    exception; nothing is zero-filled.
    """
    ordered = tuple(sorted(experts, key=lambda e: e.expert_id))
    _validate_panel(component_returns, ordered)
    _validate_decision_context(decision_context, component_returns.index)

    panel_values = component_returns.to_numpy(dtype=np.float64)
    completed = panel_values[1:]

    candidates, admitted_indexes = _candidate_evidence(
        ordered, completed, component_trade_counts, config,
    )
    admitted_experts = tuple(ordered[i] for i in admitted_indexes)

    coverage_counts, covered_states = _context_coverage(decision_context, router)
    coverage_sufficient = covered_states >= config.min_context_covered_states

    structural_total = _exact_structural_count(admitted_experts, config)
    if structural_total > config.max_combinations:
        return LibraryAdmissionReport(
            status="FAIL_CLOSED",
            window_start=str(component_returns.index[0]),
            window_end=str(component_returns.index[-1]),
            experts=ordered,
            candidates=candidates,
            proposals=(),
            context_coverage=coverage_counts,
            covered_states=covered_states,
            coverage_sufficient=coverage_sufficient,
            router=router,
            admission=config,
            structural_combinations=structural_total,
        )

    admitted_values = completed[:, list(admitted_indexes)]
    families = np.array([e.family for e in admitted_experts], dtype=object)
    symbols = np.array([e.symbols[0] for e in admitted_experts], dtype=object)
    correlation = pairwise_log_return_correlation(admitted_values)
    joint_negative = pairwise_joint_negative_rates(admitted_values)
    compat = _pair_compatibility_matrix(
        correlation, joint_negative, families, symbols, config,
    )

    enumerated = _enumerate_proposals(
        admitted_experts, compat, config.min_experts, config.max_experts,
    )
    proposals = tuple(
        AdmissionProposal(
            expert_ids=tuple(admitted_experts[i].expert_id for i in subset),
            eligible=coverage_sufficient,
            **_proposal_pair_diagnostics(correlation, joint_negative, subset),
        )
        for subset in enumerated
    )
    return LibraryAdmissionReport(
        status="COMPLETE",
        window_start=str(component_returns.index[0]),
        window_end=str(component_returns.index[-1]),
        experts=ordered,
        candidates=candidates,
        proposals=proposals,
        context_coverage=coverage_counts,
        covered_states=covered_states,
        coverage_sufficient=coverage_sufficient,
        router=router,
        admission=config,
        structural_combinations=structural_total,
    )


def _validate_panel(panel: pd.DataFrame, experts: tuple[ExpertDefinition, ...]) -> None:
    if not isinstance(panel, pd.DataFrame):
        raise DataIntegrityError(
            f"component_returns must be a pd.DataFrame, got {type(panel).__name__}"
        )
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise DataIntegrityError("component_returns must have a DatetimeIndex")
    if not panel.index.is_monotonic_increasing:
        raise DataIntegrityError("component_returns index must be monotonic increasing")
    if panel.index.has_duplicates:
        raise DataIntegrityError(
            "component_returns index must not contain duplicate timestamps"
        )
    if len(panel) < 2:
        raise DataIntegrityError(
            f"component_returns must have at least 2 rows, got {len(panel)}"
        )
    expected = [e.expert_id for e in experts]
    if list(panel.columns) != expected:
        raise DataIntegrityError(
            "component_returns columns must exactly match the experts in "
            "expert_id lexical order"
        )
    values = panel.to_numpy(dtype=np.float64)
    if not np.isnan(values[0]).all():
        raise DataIntegrityError(
            "component_returns must have exactly one initial all-NaN return row"
        )
    later = values[1:]
    if not np.isfinite(later).all():
        raise DataIntegrityError(
            "completed returns must be finite on every sample row; a missing "
            "return is never zero-filled"
        )
    if (later <= -1.0).any():
        raise DataIntegrityError(
            "completed returns must be strictly greater than -1.0 (total loss)"
        )


def _validate_decision_context(context: pd.Series, index: pd.DatetimeIndex) -> None:
    if not isinstance(context, pd.Series):
        raise DataIntegrityError(
            f"decision_context must be a pd.Series, got {type(context).__name__}"
        )
    if not context.index.equals(index):
        raise DataIntegrityError(
            "decision_context must be exactly aligned to the component_returns index"
        )
    if context.isna().any():
        raise DataIntegrityError("decision_context must not contain missing labels")
    known = set(state_labels()) | {_UNAVAILABLE}
    unknown = sorted({value for value in context.unique() if value not in known})
    if unknown:
        raise DataIntegrityError(f"decision_context contains unknown labels: {unknown}")


def _candidate_evidence(
    experts: tuple[ExpertDefinition, ...],
    completed: np.ndarray,
    trade_counts: Mapping[str, int],
    config: LibraryAdmissionConfig,
) -> tuple[tuple[CandidateAdmissionResult, ...], tuple[int, ...]]:
    results: list[CandidateAdmissionResult] = []
    admitted: list[int] = []
    for i, expert in enumerate(experts):
        if expert.expert_id not in trade_counts:
            raise DataIntegrityError(
                f"closed-trade count missing for expert {expert.expert_id}"
            )
        closed = int(trade_counts[expert.expert_id])
        active = int(np.count_nonzero(completed[:, i] != 0.0))
        reason: str | None = None
        admitted_flag = (
            closed >= config.min_closed_trades
            and active >= config.min_active_return_bars
        )
        if closed < config.min_closed_trades:
            reason = f"closed_trades={closed} < min {config.min_closed_trades}"
        elif active < config.min_active_return_bars:
            reason = f"active_return_bars={active} < min {config.min_active_return_bars}"
        results.append(
            CandidateAdmissionResult(
                expert.expert_id, closed, active, admitted_flag, reason,
            )
        )
        if admitted_flag:
            admitted.append(i)
    return tuple(results), tuple(admitted)


def _exact_structural_count(
    experts: tuple[ExpertDefinition, ...],
    config: LibraryAdmissionConfig,
) -> int:
    """Exact number of family/symbol-unique combinations in the requested sizes."""
    symbols = sorted({e.symbols[0] for e in experts})
    symbol_index = {s: i for i, s in enumerate(symbols)}
    counts: dict[tuple[str, str], int] = {}
    for e in experts:
        key = (e.family, e.symbols[0])
        counts[key] = counts.get(key, 0) + 1

    dp: dict[int, int] = {0: 1}
    for family in sorted({e.family for e in experts}):
        edges = {s: c for (f, s), c in counts.items() if f == family}
        if not edges:
            continue
        new_dp = dict(dp)
        for mask, ways in dp.items():
            for symbol, choice_count in edges.items():
                bit = 1 << symbol_index[symbol]
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_dp[new_mask] = new_dp.get(new_mask, 0) + ways * choice_count
        dp = new_dp
    return sum(
        ways
        for mask, ways in dp.items()
        if config.min_experts <= mask.bit_count() <= config.max_experts
    )


def _pair_compatibility_matrix(
    correlation: np.ndarray,
    joint_negative: np.ndarray,
    families: np.ndarray,
    symbols: np.ndarray,
    config: LibraryAdmissionConfig,
) -> np.ndarray:
    finite = np.isfinite(correlation)
    within_correlation = (
        np.abs(correlation) <= config.max_abs_pairwise_log_return_correlation
    )
    within_tail = joint_negative <= config.max_joint_negative_return_rate
    distinct_family = families[:, None] != families[None, :]
    distinct_symbol = symbols[:, None] != symbols[None, :]
    compat = finite & within_correlation & within_tail & distinct_family & distinct_symbol
    np.fill_diagonal(compat, False)
    return np.asarray(compat, dtype=bool)


def _proposal_pair_diagnostics(
    correlation: np.ndarray,
    joint_negative: np.ndarray,
    indexes: tuple[int, ...],
) -> dict[str, float]:
    """Selection-window pair diagnostics for one enumerated expert subset."""
    if len(indexes) < 2:
        return {
            "max_abs_pair_log_return_correlation": 0.0,
            "max_pair_joint_negative_rate": 0.0,
            "mean_abs_pair_log_return_correlation": 0.0,
            "mean_pair_joint_negative_rate": 0.0,
        }
    pairs = [
        (indexes[a], indexes[b])
        for a in range(len(indexes))
        for b in range(a + 1, len(indexes))
    ]
    abs_correlation = [abs(correlation[i, j]) for i, j in pairs]
    joint_negative_rates = [float(joint_negative[i, j]) for i, j in pairs]
    return {
        "max_abs_pair_log_return_correlation": float(max(abs_correlation)),
        "max_pair_joint_negative_rate": float(max(joint_negative_rates)),
        "mean_abs_pair_log_return_correlation": float(np.mean(abs_correlation)),
        "mean_pair_joint_negative_rate": float(np.mean(joint_negative_rates)),
    }


def _enumerate_proposals(
    experts: tuple[ExpertDefinition, ...],
    compat: np.ndarray,
    min_experts: int,
    max_experts: int,
) -> tuple[tuple[int, ...], ...]:
    n = len(experts)
    families = [e.family for e in experts]
    symbols = [e.symbols[0] for e in experts]
    proposals: list[tuple[int, ...]] = []

    def _backtrack(
        start: int,
        partial: list[int],
        used_families: set[str],
        used_symbols: set[str],
    ) -> None:
        size = len(partial)
        if size >= min_experts:
            proposals.append(tuple(partial))
        if size >= max_experts:
            return
        for j in range(start, n):
            if families[j] in used_families or symbols[j] in used_symbols:
                continue
            if not all(compat[i, j] for i in partial):
                continue
            _backtrack(
                j + 1,
                [*partial, j],
                used_families | {families[j]},
                used_symbols | {symbols[j]},
            )

    _backtrack(0, [], set(), set())
    return tuple(proposals)


_SHORTLIST_PER_SIZE = 6


def shortlist_admission_proposals(
    proposals: tuple[AdmissionProposal, ...],
    max_backtest_proposals: int,
) -> tuple[AdmissionProposal, ...]:
    """Deterministic size-stratified shortlist capped by the backtest budget.

    Proposals are grouped by expert count and ranked within each size by the
    ``AdmissionProposal.rank_key``: maximum then mean absolute pairwise
    correlation, maximum then mean joint-negative rate, and the lexical
    proposal id. Six proposals per available size are selected first, then any
    unused budget is assigned one at a time to the remaining sizes in ascending
    rank order. Only selection-window pair diagnostics participate; portfolio
    metrics, promotion verdicts, and OOS data never enter.
    """
    if max_backtest_proposals < 1:
        raise ValueError(
            f"max_backtest_proposals must be >= 1, got {max_backtest_proposals}"
        )
    if not proposals:
        return ()
    by_size: dict[int, list[AdmissionProposal]] = {}
    for proposal in proposals:
        by_size.setdefault(len(proposal.expert_ids), []).append(proposal)
    sizes = sorted(by_size)
    for size in sizes:
        by_size[size].sort(key=lambda p: p.rank_key())

    selected: dict[int, list[AdmissionProposal]] = {size: [] for size in sizes}
    budget = max_backtest_proposals
    for _round in range(_SHORTLIST_PER_SIZE):
        for size in sizes:
            if budget == 0:
                break
            ranked = by_size[size]
            if len(ranked) > len(selected[size]):
                selected[size].append(ranked[len(selected[size])])
                budget -= 1
        if budget == 0:
            break
    while budget > 0:
        added = False
        for size in sizes:
            if budget == 0:
                break
            ranked = by_size[size]
            if len(ranked) > len(selected[size]):
                selected[size].append(ranked[len(selected[size])])
                budget -= 1
                added = True
        if not added:
            break
    return tuple(proposal for size in sizes for proposal in selected[size])


def _context_coverage(
    decision_context: pd.Series,
    router: ContextualRouterSpec,
) -> tuple[dict[str, int], int]:
    labels = decision_context.to_numpy(dtype=object)
    n = len(labels)
    counts: dict[str, int] = {}
    for state in state_labels():
        counts[state] = int(np.count_nonzero(labels[: n - 1] == state))
    covered = sum(
        1 for count in counts.values() if count >= router.min_context_history_bars
    )
    return counts, covered


def _check_contract() -> None:
    """Executable assertions locking the admission evaluator surface."""
    assert evaluate_library_admission.__name__ == "evaluate_library_admission"
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    panel = pd.DataFrame(
        {
            "b": [np.nan, 0.01, 0.02, -0.02, 0.03, -0.01],
            "a": [np.nan, 0.02, -0.01, 0.03, -0.02, 0.01],
        },
        index=index,
    )
    panel = panel[["a", "b"]]
    experts = (
        ExpertDefinition("a", "technical_macd_histogram_regime_long_v1", "f1", ("S1",), "run_technical_expert", "h"),
        ExpertDefinition("b", "technical_rsi_trend_pullback_long_v1", "f2", ("S2",), "run_technical_expert", "h2"),
    )
    context = pd.Series(["up_low_vol"] * 6, index=index)
    config = LibraryAdmissionConfig(1, 2, 1, 1, 0.9, 0.9, 1, 100)
    router = ContextualRouterSpec("S1", 1, 1, 1)
    report = evaluate_library_admission(
        panel, {"a": 2, "b": 2}, experts, context, router, config,
    )
    assert report.status == "COMPLETE"
    assert {p.expert_ids for p in report.proposals} == {
        ("a",), ("b",), ("a", "b"),
    }


_check_contract()
