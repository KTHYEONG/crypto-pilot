"""Cluster-balanced symbol weights for spot portfolio aggregation."""

from __future__ import annotations

from typing import Dict, List


def build_cluster_balanced_symbol_weights(
    symbols: List[str],
    cluster_map: Dict[str, str],
    cluster_weights: Dict[str, float],
    *,
    default_cluster: str = "default",
) -> Dict[str, float]:
    """
    Per-symbol weights: cluster_weight / count(symbols in cluster), renormalized to sum 1.
    """
    if not symbols:
        return {}
    counts: Dict[str, int] = {}
    for sym in symbols:
        cluster = cluster_map.get(sym, default_cluster)
        counts[cluster] = counts.get(cluster, 0) + 1

    raw: Dict[str, float] = {}
    for sym in symbols:
        cluster = cluster_map.get(sym, default_cluster)
        cw = float(cluster_weights.get(cluster, 1.0 / max(len(cluster_weights), 1)))
        denom = max(int(counts.get(cluster, 1)), 1)
        raw[sym] = cw / float(denom)

    total = float(sum(raw.values()))
    if total <= 1e-18:
        u = 1.0 / float(len(symbols))
        return {s: u for s in symbols}
    return {k: v / total for k, v in raw.items()}
