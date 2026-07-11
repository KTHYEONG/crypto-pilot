"""Phase 0 measure-first diagnostic for L1 per-symbol atomization dilution.
[ADR_20260711_L1_POOLED_ALPHA_ADMISSION_GENERALIZATION]

Pure re-aggregation over already-computed SymbolStrategyEvidence — no new
statistics, no L0<->L1 cross-layer plumbing. Log-only; does not alter any
gate decision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median

from src.domain.futures.strategy.candidate_contracts import SymbolStrategyEvidence


@dataclass(slots=True, frozen=True)
class AtomizationDiagnosticReport:
    strategy_id: str
    n_cells: int
    n_cells_below_min_effective_obs: int
    pooled_mean_gross_bps: float
    atomized_mean_gross_bps_median: float
    sign_flip_ratio: float
    sign_flip_ratio_weighted: float
    reject_reason_counts: dict[str, int]
    dominant_reject_reason: str | None


def _build_report(
    strategy_id: str,
    cells: list[SymbolStrategyEvidence],
    *,
    min_effective_obs: float,
) -> AtomizationDiagnosticReport:
    n_cells = len(cells)
    n_cells_below = sum(1 for c in cells if c.effective_n < min_effective_obs)

    total_obs = sum(c.n_obs for c in cells)
    pooled_gross = sum(c.n_obs * c.mean_gross_bps for c in cells) / total_obs if total_obs > 0 else 0.0

    gross_values = [c.mean_gross_bps for c in cells]
    atomized_median = float(median(gross_values))

    if n_cells > 0:
        n_flip = sum(
            1 for c in cells if (c.mean_gross_bps > 0) != (pooled_gross > 0)
        )
        sign_flip = n_flip / n_cells
        if total_obs > 0:
            n_flip_weighted = sum(
                c.n_obs
                for c in cells
                if (c.mean_gross_bps > 0) != (pooled_gross > 0)
            )
            sign_flip_w = n_flip_weighted / total_obs
        else:
            sign_flip_w = sign_flip
    else:
        sign_flip = 0.0
        sign_flip_w = 0.0

    reason_counts: Counter[str] = Counter()
    for c in cells:
        for r in c.structural_reasons:
            reason_counts[r] += 1
    dominant = reason_counts.most_common(1)[0][0] if reason_counts else None

    return AtomizationDiagnosticReport(
        strategy_id=strategy_id,
        n_cells=n_cells,
        n_cells_below_min_effective_obs=n_cells_below,
        pooled_mean_gross_bps=pooled_gross,
        atomized_mean_gross_bps_median=atomized_median,
        sign_flip_ratio=sign_flip,
        sign_flip_ratio_weighted=sign_flip_w,
        reject_reason_counts=dict(reason_counts),
        dominant_reject_reason=dominant,
    )


def diagnose_strategy_atomization(
    evidence: tuple[SymbolStrategyEvidence, ...],
    *,
    min_effective_obs: float,
) -> tuple[AtomizationDiagnosticReport, ...]:
    """Re-aggregate per-symbol evidence cells back to strategy_id level.

    Args:
        evidence: Output of compute_symbol_strategy_evidence (one row per
            (symbol, strategy_id, activation_context) cell).
        min_effective_obs: Same threshold used by the live L1 gate
            (cfg.l1_pair_min_effective_obs), for n_cells_below_min_effective_obs.

    Returns:
        One AtomizationDiagnosticReport per distinct strategy_id, sorted by
        strategy_id for deterministic log ordering.

    Note:
        Time: O(N) single pass over evidence. Space: O(S) where S=distinct
        strategy_ids. Degenerate cells (sum of n_obs == 0 for a strategy_id)
        yield pooled_mean_gross_bps=0.0 rather than raising.
    """
    if not evidence:
        return ()

    groups: dict[str, list[SymbolStrategyEvidence]] = {}
    for ev in evidence:
        groups.setdefault(ev.key.strategy_id, []).append(ev)

    reports = [
        _build_report(sid, cells, min_effective_obs=min_effective_obs)
        for sid, cells in sorted(groups.items())
    ]
    return tuple(reports)
