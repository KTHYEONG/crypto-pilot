"""Alpha Foundry diversity and effective test count helpers. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.contracts import CheapGateEvidence
from src.domain.futures.signals.contracts import CandidateSignalPanel


def _flatten_panel_scores(panel: CandidateSignalPanel) -> NDArray[np.float64]:
    flat = panel.signed_score_2d.ravel()
    return flat[np.isfinite(flat)]


def compute_panel_correlation_matrix(
    panels: Sequence[CandidateSignalPanel],
) -> NDArray[np.float64]:
    if not panels:
        raise ValueError("panels must not be empty")
    m = len(panels)
    flat_scores = [_flatten_panel_scores(p) for p in panels]
    corr = np.full((m, m), np.nan, dtype=np.float64)
    for i in range(m):
        for j in range(m):
            si = flat_scores[i]
            sj = flat_scores[j]
            min_len = min(len(si), len(sj))
            if min_len < 2:
                corr[i, j] = 0.0
            else:
                c = np.corrcoef(si[:min_len], sj[:min_len])[0, 1]
                corr[i, j] = c if np.isfinite(c) else 0.0
        corr[i, i] = 1.0
    return corr


def cluster_correlated_recipes(
    *,
    evidences: Sequence[CheapGateEvidence],
    corr: NDArray[np.float64],
    max_corr: float,
) -> tuple[tuple[str, ...], ...]:
    n = len(evidences)
    if corr.shape != (n, n):
        raise ValueError(
            f"corr shape {corr.shape} != (n_evidences={n}, {n})"
        )
    recipe_ids = [e.recipe_id for e in evidences]
    assigned: set[int] = set()
    clusters: list[tuple[str, ...]] = []
    for i in range(n):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if corr[i, j] > max_corr:
                cluster.append(j)
                assigned.add(j)
        clusters.append(tuple(recipe_ids[idx] for idx in cluster))
    return tuple(clusters)


def estimate_effective_test_count(
    corr: NDArray[np.float64],
) -> float:
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got shape {corr.shape}")
    n = corr.shape[0]
    if n <= 1:
        return float(n)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = np.sum(eigenvalues)
    if total <= 0.0:
        return 1.0
    probs = eigenvalues / total
    entropy = -np.sum(probs * np.log(probs + 1e-15))
    m_eff = np.exp(entropy)
    return float(max(1.0, min(m_eff, float(n))))
