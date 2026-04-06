"""Optional diagnostics: shared-cash rank_score vs entry_signal alignment (tmp.md)."""
from __future__ import annotations

import numpy as np


def rank_entry_alignment_issues(
    entry_signal: np.ndarray,
    rank_score: np.ndarray,
    *,
    min_entry_bars: int = 20,
) -> list[str]:
    """
    Heuristic: on entry bars, mean rank should not be systematically below non-entry bars.
    Returns human-readable issue strings (empty if no obvious conflict).
    """
    issues: list[str] = []
    ent = np.asarray(entry_signal, dtype=np.float64).ravel()
    rk = np.asarray(rank_score, dtype=np.float64).ravel()
    if ent.size != rk.size or ent.size == 0:
        return issues
    m_ent = ent > 0.5
    n_e = int(np.sum(m_ent))
    n_n = int(np.sum(~m_ent))
    if n_e < min_entry_bars or n_n < min_entry_bars:
        return issues
    mean_e = float(np.nanmean(rk[m_ent]))
    mean_n = float(np.nanmean(rk[~m_ent]))
    if mean_e + 1e-6 < mean_n:
        issues.append(
            f"mean_rank_on_entries({mean_e:.6f}) < mean_rank_off_entries({mean_n:.6f})"
        )
    return issues
