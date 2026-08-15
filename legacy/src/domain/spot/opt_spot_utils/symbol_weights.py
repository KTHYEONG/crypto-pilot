"""Cluster-balanced symbol weights for spot portfolio aggregation."""

from __future__ import annotations


def build_cluster_balanced_symbol_weights(
    symbols: list[str],
    cluster_map: dict[str, str],
    cluster_weights: dict[str, float],
    *,
    default_cluster: str = "default",
) -> dict[str, float]:
    """Per-symbol weights: cluster_weight / count(symbols in cluster), renormalized to sum 1.
    """
    if not symbols:
        return {}
    counts: dict[str, int] = {}
    for sym in symbols:
        cluster = cluster_map.get(sym, default_cluster)
        counts[cluster] = counts.get(cluster, 0) + 1

    raw: dict[str, float] = {}
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
