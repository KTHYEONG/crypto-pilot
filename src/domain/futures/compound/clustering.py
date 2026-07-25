from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans

from src.domain.futures.compound.contracts import (
    ClusteringAlgorithm,
    ClusterPanel,
    MarketFeatureCube,
)

_LOGGER = logging.getLogger(__name__)

_FEATURE_LOOKBACK: int = 60
_EPS_DEFAULT: float = 0.75
_MIN_SAMPLES_DEFAULT: int = 5


def _log_returns(close: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.full_like(close, np.nan)
    result[1:] = np.log(np.maximum(close[1:], 1e-12)) - np.log(np.maximum(close[:-1], 1e-12))
    return result


def _hurst_exponent(price: NDArray[np.float64]) -> float:
    n = price.shape[0]
    if n < 10:
        return 0.5
    lags = range(2, min(n // 2, 32))
    tau = [math.sqrt(float(np.nanstd(np.diff(price[::lag])))) for lag in lags]
    if not tau or not all(t > 0 for t in tau):
        return 0.5
    log_lags = [math.log(lag) for lag in lags]
    log_tau = [math.log(t) for t in tau]
    n_l = len(log_lags)
    mean_x = sum(log_lags) / n_l
    mean_y = sum(log_tau) / n_l
    num = sum((log_lags[i] - mean_x) * (log_tau[i] - mean_y) for i in range(n_l))
    den = sum((log_lags[i] - mean_x) ** 2 for i in range(n_l))
    return float(num / den) if den != 0 else 0.5


def _extract_clustering_features(
    market: MarketFeatureCube,
    lookback: int = _FEATURE_LOOKBACK,
) -> NDArray[np.float64]:
    close_raw = market.fields_2d.get("close")
    volume_raw = market.fields_2d.get("quote_volume")
    if close_raw is None or volume_raw is None:
        raise ValueError("MarketFeatureCube must contain 'close' and 'quote_volume' fields")

    close = np.asarray(close_raw, dtype=np.float64)
    volume = np.asarray(volume_raw, dtype=np.float64)
    n_syms = close.shape[1]
    features = np.zeros((n_syms, 4), dtype=np.float64)

    if close.shape[0] < 2:
        raise ValueError("insufficient valid close data for feature computation")

    btc_idx: int | None = None
    for i, sym in enumerate(market.symbols):
        if "BTC" in sym.upper():
            btc_idx = i
            break
    if btc_idx is None:
        btc_idx = 0

    btc_close = close[:, btc_idx]
    btc_ret = _log_returns(btc_close)

    for i in range(n_syms):
        sym_close = close[:, i]
        sym_vol = volume[:, i]
        valid_mask = np.isfinite(sym_close) & (sym_close > 0)

        if np.any(valid_mask):
            valid_close = sym_close[valid_mask]
            sym_ret = _log_returns(sym_close)
            lookback_actual = min(lookback, sym_ret.shape[0])
            sym_volatility = float(np.nanstd(sym_ret[-lookback_actual:], ddof=1)) if lookback_actual > 1 else 0.0
            features[i, 0] = sym_volatility

            vol_valid = sym_vol[np.isfinite(sym_vol) & (sym_vol > 0)]
            features[i, 1] = float(np.log(np.mean(vol_valid))) if vol_valid.size > 0 else 0.0

            both_valid = np.isfinite(btc_ret) & np.isfinite(sym_ret)
            if np.sum(both_valid) > 5:
                b = btc_ret[both_valid]
                s = sym_ret[both_valid]
                cov = float(np.cov(s, b, ddof=1)[0, 1])
                var_b = float(np.var(b, ddof=1))
                features[i, 2] = cov / max(var_b, 1e-12)
            else:
                features[i, 2] = 0.0

            if valid_close.size >= 10:
                features[i, 3] = _hurst_exponent(np.log(valid_close))
            else:
                features[i, 3] = 0.5
        else:
            features[i, :] = [0.0, 0.0, 0.0, 0.5]

    return features


def _winsorize(features: NDArray[np.float64], pct: float = 0.05) -> NDArray[np.float64]:
    if pct <= 0.0:
        return features
    result = features.copy()
    n = features.shape[1]
    for j in range(n):
        col = features[:, j]
        finite = np.isfinite(col)
        if np.any(finite):
            lower = float(np.percentile(col[finite], pct * 100.0))
            upper = float(np.percentile(col[finite], (1.0 - pct) * 100.0))
            col_clipped = np.clip(col, lower, upper)
            result[:, j] = np.where(finite, col_clipped, col)
    return result


def _standardize(features: NDArray[np.float64]) -> NDArray[np.float64]:
    result = features.copy()
    n = features.shape[1]
    for j in range(n):
        col = features[:, j]
        finite = np.isfinite(col)
        if np.sum(finite) > 1:
            mean = float(np.mean(col[finite]))
            std = float(np.std(col[finite], ddof=1))
            result[:, j] = np.where(finite, (col - mean) / max(std, 1e-12), 0.0)
        else:
            result[:, j] = np.where(finite, col, 0.0)
    return result


def _enforce_min_cluster_size(
    labels: NDArray[np.int32],
    centroids: NDArray[np.float64],
    features: NDArray[np.float64],
    min_size: int,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    unique = np.unique(labels)
    if len(unique) <= 1:
        return labels, centroids

    result_labels = labels.copy()
    large = [c for c in unique if int(np.sum(labels == c)) >= min_size]
    if len(large) == len(unique):
        return result_labels, centroids

    large_centroids = centroids[large] if large else np.mean(features, axis=0, keepdims=True)
    large_labels = large if large else [0]

    for label in unique:
        mask = labels == label
        count = int(np.sum(mask))
        if count < min_size:
            for idx in np.where(mask)[0]:
                dists = np.linalg.norm(features[idx] - large_centroids, axis=1)
                closest = int(np.argmin(dists))
                result_labels[idx] = large_labels[closest]
            _LOGGER.info(
                "[CLUSTER] merged cluster %d (%d syms) into nearest large cluster",
                label, count,
            )

    merged_centroids = centroids[large].copy() if large else large_centroids
    return result_labels, merged_centroids


def compute_market_regime_clusters(
    market: MarketFeatureCube,
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.ROBUST_KMEANS,
    k_clusters: int = 4,
    min_cluster_size: int = 5,
    winsorize_pct: float = 0.05,
) -> ClusterPanel:
    if k_clusters < 2:
        raise ValueError("k_clusters must be >= 2")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")

    features = _extract_clustering_features(market)
    features = _winsorize(features, winsorize_pct)
    features = _standardize(features)

    n_syms = features.shape[0]
    if n_syms < k_clusters:
        k_clusters = max(2, n_syms)

    if algorithm == ClusteringAlgorithm.ROBUST_KMEANS:
        n_init = min(10, n_syms)
        km = KMeans(
            n_clusters=k_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=300,
            random_state=42,
            algorithm="lloyd",
        )
        km.fit(features)
        labels = km.labels_.astype(np.int32)
        centroids = km.cluster_centers_
    elif algorithm == ClusteringAlgorithm.HIERARCHICAL_WARD:
        z_link = linkage(features, method="ward")
        labels = fcluster(z_link, t=k_clusters, criterion="maxclust").astype(np.int32) - 1
        centroids = np.zeros((k_clusters, features.shape[1]), dtype=np.float64)
        for k in range(k_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = np.mean(features[mask], axis=0)
    elif algorithm == ClusteringAlgorithm.DBSCAN:
        db = DBSCAN(eps=_EPS_DEFAULT, min_samples=_MIN_SAMPLES_DEFAULT).fit(features)
        labels = db.labels_.astype(np.int32)
        noise_mask = labels == -1
        unique_non_noise = np.unique(labels[~noise_mask]) if np.any(~noise_mask) else np.array([], dtype=np.int32)
        k_actual = len(unique_non_noise)
        if k_actual == 0:
            _LOGGER.warning("[CLUSTER] DBSCAN found zero clusters, falling back to single cluster")
            labels = np.zeros(n_syms, dtype=np.int32)
            centroids = np.mean(features, axis=0, keepdims=True)
        else:
            centroids = np.zeros((k_actual, features.shape[1]), dtype=np.float64)
            for j, k in enumerate(unique_non_noise):
                mask = labels == k
                centroids[j] = np.mean(features[mask], axis=0)
            centroids = np.vstack([centroids, np.full((max(k_clusters - k_actual, 0), features.shape[1]), np.nan, dtype=np.float64)])
    else:
        raise ValueError(f"unknown clustering algorithm: {algorithm}")

    if algorithm in (ClusteringAlgorithm.ROBUST_KMEANS, ClusteringAlgorithm.HIERARCHICAL_WARD):
        labels, centroids = _enforce_min_cluster_size(labels, centroids, features, min_cluster_size)

    final_k = centroids.shape[0]

    _LOGGER.info(
        "[CLUSTER] algorithm=%s k=%d n_syms=%d clusters=%s",
        algorithm.value, final_k, n_syms, sorted(int(x) for x in np.unique(labels)),
    )

    return ClusterPanel(
        symbols=market.symbols,
        cluster_labels=labels,
        cluster_centroids=centroids,
        k_clusters=final_k,
    )
