from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.domain.futures.strategy.candidate_contracts import (
        QualifiedSignalRegistry,
        SignalSleeveKey,
        ValidatedSignalBatch,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        L2SimulationCache,
    )
    from src.domain.futures.strategy.walk_forward import WFFold

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PortfolioHandoffConfig:
    max_candidate_sleeves: int = 32
    min_calibration_windows: int = 3
    min_positive_window_ratio: float = 2.0 / 3.0
    min_marginal_growth_lcb: float = 0.0
    max_abs_pairwise_corr: float = 0.85
    min_pair_observations: int = 30
    max_sleeves_per_cluster: int = 1
    min_source_families: int = 2


@dataclass(slots=True, frozen=True)
class SleeveContributionEvidence:
    key: SignalSleeveKey
    marginal_growth_by_window: tuple[float, ...]
    marginal_growth_lcb: float
    positive_window_ratio: float
    max_abs_pairwise_corr: float
    redundancy_cluster: int
    admitted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class PortfolioHandoffResult:
    admitted_sleeves_by_fold: tuple[tuple[SignalSleeveKey, ...], ...]
    evidence_by_fold: tuple[tuple[SleeveContributionEvidence, ...], ...]
    passed: bool
    blocker_reason: str
    fingerprint: str


def validate_causal_sleeve_return_matrix(
    returns: NDArray[np.floating],
    *,
    expected_bars: int,
    expected_sleeves: int,
) -> NDArray[np.float32]:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got {arr.ndim}D")
    if arr.shape[0] != expected_bars:
        raise ValueError(
            f"expected {expected_bars} bars, got {arr.shape[0]}"
        )
    if arr.shape[1] != expected_sleeves:
        raise ValueError(
            f"expected {expected_sleeves} sleeves, got {arr.shape[1]}"
        )
    if not np.all(np.isfinite(arr)):
        non_finite = (~np.isfinite(arr)).sum()
        raise ValueError(
            f"expected all finite values, got {int(non_finite)} non-finite entries"
        )
    result = np.ascontiguousarray(arr, dtype=np.float32)
    assert result.flags.c_contiguous
    return result


def _deterministic_sort_key(
    quality_weight: float,
    marginal_lcb_bps: float,
    deterministic_key: str,
) -> tuple[float, float, str]:
    return (-quality_weight, -marginal_lcb_bps, deterministic_key)


def _annualize_log_growth(
    log_rets: NDArray[np.float64],
    n_bars: int,
    bars_per_year: float = 2190.0,
) -> float:
    if n_bars < 1:
        return 0.0
    return float(np.expm1(float(np.sum(log_rets)) / (float(n_bars) / bars_per_year)))


def _window_marginal_growth(
    returns_window: NDArray[np.float64],
    w: NDArray[np.float64],
    sleeve_s: int,
    bars_per_year: float = 2190.0,
) -> float:
    n_bars = returns_window.shape[0]
    log_full = np.sum(np.log1p(returns_window @ w))
    w_minus = np.delete(w, sleeve_s)
    returns_minus = np.delete(returns_window, sleeve_s, axis=1)
    w_minus_norm = w_minus / max(np.sum(w_minus), 1e-12)
    log_minus = np.sum(np.log1p(returns_minus @ w_minus_norm))
    growth_full = _annualize_log_growth(np.array([log_full]), n_bars, bars_per_year)
    growth_minus = _annualize_log_growth(np.array([log_minus]), n_bars, bars_per_year)
    return growth_full - growth_minus


def _moving_block_bootstrap_lcb(
    values: NDArray[np.float64],
    k: float = 1.0,
    n_resamples: int = 1000,
    block_size: int = 10,
    seed: int = 42,
) -> float:
    if values.size < 2:
        return float("-inf")
    rng = np.random.default_rng(seed)
    n = values.size
    n_blocks = int(np.ceil(n / block_size))
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        blocks = rng.integers(0, max(n - block_size + 1, 1), size=n_blocks)
        indices = (blocks[:, None] + np.arange(block_size)) % n
        sample = values[indices.ravel()[:n]]
        boot_means[i] = float(np.mean(sample))
    mu = float(np.mean(boot_means))
    sigma = float(np.std(boot_means, ddof=1))
    return mu - k * sigma


def _handoff_fingerprint(
    registry: QualifiedSignalRegistry,
    config: PortfolioHandoffConfig,
    n_folds: int,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"portfolio-handoff-v1")
    hasher.update(config.max_candidate_sleeves.to_bytes(4, "big", signed=False))
    hasher.update(config.min_source_families.to_bytes(4, "big", signed=False))
    hasher.update(str(registry.registry_version).encode("utf-8"))
    ready = registry.ready_symbols
    hasher.update(str(len(ready)).encode("utf-8"))
    for sym in ready:
        hasher.update(sym.encode("utf-8"))
    for sym_key, ev_list in registry.by_symbol.items():
        for ev in ev_list:
            hasher.update(f"{sym_key}:{ev.key.strategy_id}:{ev.quality_weight}".encode())
    hasher.update(n_folds.to_bytes(4, "big", signed=False))
    return hasher.hexdigest()


def evaluate_portfolio_handoff(
    *,
    registry: QualifiedSignalRegistry,
    signal_batch: ValidatedSignalBatch,
    cache: L2SimulationCache,
    folds: tuple[WFFold, ...],
    net_sleeve_returns_by_fold: tuple[NDArray[np.float32], ...],
    config: PortfolioHandoffConfig,
) -> PortfolioHandoffResult:
    n_folds = len(folds)
    if n_folds != len(net_sleeve_returns_by_fold):
        raise ValueError("fold count mismatch between folds and net_sleeve_returns_by_fold")

    fp = _handoff_fingerprint(registry, config, n_folds)

    sleeve_keys = cache.sleeve_keys
    if not sleeve_keys:
        return PortfolioHandoffResult(
            admitted_sleeves_by_fold=tuple(() for _ in range(n_folds)),
            evidence_by_fold=tuple(() for _ in range(n_folds)),
            passed=False,
            blocker_reason="no_sleeves_in_cache",
            fingerprint=fp,
        )

    n_sleeves = len(sleeve_keys)
    bars_per_year: float = 2190.0

    all_evidence_by_fold: list[tuple[SleeveContributionEvidence, ...]] = []
    all_admitted_by_fold: list[tuple[SignalSleeveKey, ...]] = []

    for fold_idx in range(n_folds):
        returns_fold = net_sleeve_returns_by_fold[fold_idx]
        returns_fold_64 = np.asarray(returns_fold, dtype=np.float64)
        n_bars_fold, n_sleeves_fold = returns_fold_64.shape
        if n_sleeves_fold != n_sleeves:
            _logger.warning(
                "[HANDOFF] fold=%d sleeve count mismatch: cache=%d returns=%d",
                fold_idx, n_sleeves, n_sleeves_fold,
            )
            all_evidence_by_fold.append(())
            all_admitted_by_fold.append(())
            continue

        weights = np.ones(n_sleeves, dtype=np.float64)
        weights /= max(np.sum(weights), 1e-12)

        sleeve_evidence: list[SleeveContributionEvidence] = []
        for s in range(n_sleeves):
            key = sleeve_keys[s]
            if n_bars_fold < 2:
                sleeve_evidence.append(SleeveContributionEvidence(
                    key=key,
                    marginal_growth_by_window=(),
                    marginal_growth_lcb=float("-inf"),
                    positive_window_ratio=0.0,
                    max_abs_pairwise_corr=0.0,
                    redundancy_cluster=-1,
                    admitted=False,
                    rejection_reasons=("insufficient_bars",),
                ))
                continue

            calibration_subwindows: list[NDArray[np.float64]] = []
            subwindow_size = max(n_bars_fold // max(config.min_calibration_windows, 1), 2)
            for w in range(config.min_calibration_windows):
                start = w * subwindow_size
                end = min(start + subwindow_size, n_bars_fold)
                if end - start < 2:
                    continue
                sub_returns = returns_fold_64[start:end, :]
                w_sub = weights.copy()
                delta = _window_marginal_growth(sub_returns, w_sub, s, bars_per_year)
                calibration_subwindows.append(np.array([delta], dtype=np.float64))

            if len(calibration_subwindows) < config.min_calibration_windows:
                sleeve_evidence.append(SleeveContributionEvidence(
                    key=key,
                    marginal_growth_by_window=(),
                    marginal_growth_lcb=float("-inf"),
                    positive_window_ratio=0.0,
                    max_abs_pairwise_corr=0.0,
                    redundancy_cluster=-1,
                    admitted=False,
                    rejection_reasons=("insufficient_calibration_windows",),
                ))
                continue

            deltas = np.array([d[0] for d in calibration_subwindows], dtype=np.float64)
            lcb = _moving_block_bootstrap_lcb(deltas, k=1.0, seed=fold_idx * 1000 + s)
            pos_ratio = float(np.sum(deltas > 0)) / max(len(deltas), 1)

            sleeve_evidence.append(SleeveContributionEvidence(
                key=key,
                marginal_growth_by_window=tuple(float(d) for d in deltas),
                marginal_growth_lcb=lcb,
                positive_window_ratio=pos_ratio,
                max_abs_pairwise_corr=0.0,
                redundancy_cluster=-1,
                admitted=True,
                rejection_reasons=(),
            ))

        corr_matrix: NDArray[np.float64] = np.asarray(np.corrcoef(returns_fold_64, rowvar=False))
        corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)
        low_obs_mask = n_bars_fold < config.min_pair_observations
        if low_obs_mask:
            corr_matrix[:] = 1.0

        for s in range(n_sleeves):
            if sleeve_evidence[s].admitted:
                ev = sleeve_evidence[s]
                if ev.marginal_growth_lcb <= config.min_marginal_growth_lcb:
                    sleeve_evidence[s] = SleeveContributionEvidence(
                        key=ev.key,
                        marginal_growth_by_window=ev.marginal_growth_by_window,
                        marginal_growth_lcb=ev.marginal_growth_lcb,
                        positive_window_ratio=ev.positive_window_ratio,
                        max_abs_pairwise_corr=ev.max_abs_pairwise_corr,
                        redundancy_cluster=ev.redundancy_cluster,
                        admitted=False,
                        rejection_reasons=("low_marginal_growth_lcb",),
                    )
                elif ev.positive_window_ratio < config.min_positive_window_ratio:
                    sleeve_evidence[s] = SleeveContributionEvidence(
                        key=ev.key,
                        marginal_growth_by_window=ev.marginal_growth_by_window,
                        marginal_growth_lcb=ev.marginal_growth_lcb,
                        positive_window_ratio=ev.positive_window_ratio,
                        max_abs_pairwise_corr=ev.max_abs_pairwise_corr,
                        redundancy_cluster=ev.redundancy_cluster,
                        admitted=False,
                        rejection_reasons=("low_positive_window_ratio",),
                    )

        n_admitted = sum(1 for e in sleeve_evidence if e.admitted)
        for s in range(n_sleeves):
            if sleeve_evidence[s].admitted:
                admitted_idx = [i for i in range(n_sleeves) if sleeve_evidence[i].admitted].index(s)
                cluster_corrs = []
                for other_s in range(n_sleeves):
                    if other_s == s or not sleeve_evidence[other_s].admitted:
                        continue
                    corr_val = abs(corr_matrix[s, other_s])
                    cluster_corrs.append((corr_val, other_s))
                max_corr = max((c for c, _ in cluster_corrs), default=0.0)
                ev = sleeve_evidence[s]
                sleeve_evidence[s] = SleeveContributionEvidence(
                    key=ev.key,
                    marginal_growth_by_window=ev.marginal_growth_by_window,
                    marginal_growth_lcb=ev.marginal_growth_lcb,
                    positive_window_ratio=ev.positive_window_ratio,
                    max_abs_pairwise_corr=float(max_corr),
                    redundancy_cluster=admitted_idx,
                    admitted=ev.admitted,
                    rejection_reasons=ev.rejection_reasons,
                )

        if n_admitted > 0:
            admitted_indices = [i for i in range(n_sleeves) if sleeve_evidence[i].admitted]
            removed_in_cluster: set[int] = set()
            for i in admitted_indices:
                for j in admitted_indices:
                    if i >= j:
                        continue
                    if abs(corr_matrix[i, j]) >= config.max_abs_pairwise_corr:
                        ev_i = sleeve_evidence[i]
                        ev_j = sleeve_evidence[j]
                        tie_key = _deterministic_sort_key(
                            float(ev_i.marginal_growth_lcb),
                            float(ev_i.positive_window_ratio),
                            f"{ev_i.key.symbol}:{ev_i.key.strategy_id}",
                        )
                        tie_key_j = _deterministic_sort_key(
                            float(ev_j.marginal_growth_lcb),
                            float(ev_j.positive_window_ratio),
                            f"{ev_j.key.symbol}:{ev_j.key.strategy_id}",
                        )
                        to_remove = j if tie_key > tie_key_j else i
                        removed_in_cluster.add(to_remove)

            for s in removed_in_cluster:
                ev = sleeve_evidence[s]
                sleeve_evidence[s] = SleeveContributionEvidence(
                    key=ev.key,
                    marginal_growth_by_window=ev.marginal_growth_by_window,
                    marginal_growth_lcb=ev.marginal_growth_lcb,
                    positive_window_ratio=ev.positive_window_ratio,
                    max_abs_pairwise_corr=ev.max_abs_pairwise_corr,
                    redundancy_cluster=ev.redundancy_cluster,
                    admitted=False,
                    rejection_reasons=("redundant_high_correlation",),
                )

        admitted_keys = tuple(
            sleeve_evidence[i].key
            for i in range(n_sleeves)
            if sleeve_evidence[i].admitted
        )

        source_families: set[str] = set()
        for k in admitted_keys:
            source_families.add(k.strategy_id.split(":")[0] if ":" in k.strategy_id else k.strategy_id)
        weight_sum = sum(
            float(sleeve_evidence[i].marginal_growth_lcb)
            for i in range(n_sleeves)
            if sleeve_evidence[i].admitted
        )

        fold_blocker = ""
        if weight_sum <= 0.0 or not np.isfinite(float(weight_sum)):
            fold_blocker = "invalid_handoff_weights"
        elif len(source_families) < config.min_source_families and len(source_families) > 0:
            fold_blocker = "insufficient_family_diversity"
        elif not admitted_keys:
            fold_blocker = "all_sleeves_harmful"

        if fold_blocker:
            for s in range(n_sleeves):
                if sleeve_evidence[s].admitted:
                    ev = sleeve_evidence[s]
                    sleeve_evidence[s] = SleeveContributionEvidence(
                        key=ev.key,
                        marginal_growth_by_window=ev.marginal_growth_by_window,
                        marginal_growth_lcb=ev.marginal_growth_lcb,
                        positive_window_ratio=ev.positive_window_ratio,
                        max_abs_pairwise_corr=ev.max_abs_pairwise_corr,
                        redundancy_cluster=ev.redundancy_cluster,
                        admitted=False,
                        rejection_reasons=(fold_blocker,),
                    )
            admitted_keys = ()

        all_evidence_by_fold.append(tuple(sleeve_evidence))
        all_admitted_by_fold.append(admitted_keys)

    overall_passed = any(len(ak) > 0 for ak in all_admitted_by_fold)
    overall_blocker = "" if overall_passed else "all_folds_blocked"

    return PortfolioHandoffResult(
        admitted_sleeves_by_fold=tuple(all_admitted_by_fold),
        evidence_by_fold=tuple(all_evidence_by_fold),
        passed=overall_passed,
        blocker_reason=overall_blocker,
        fingerprint=fp,
    )


def apply_portfolio_handoff_to_cache(
    *,
    cache: L2SimulationCache,
    handoff: PortfolioHandoffResult,
) -> L2SimulationCache:
    from dataclasses import replace
    mask_by_fold: list[NDArray[np.bool_]] = []
    n_sleeves = len(cache.sleeve_keys)
    for fold_admitted in handoff.admitted_sleeves_by_fold:
        mask = np.zeros(n_sleeves, dtype=np.bool_)
        admitted_set = set(fold_admitted)
        for i, sk in enumerate(cache.sleeve_keys):
            if sk in admitted_set:
                mask[i] = True
        mask_by_fold.append(mask)
    return replace(cache, handoff_sleeve_mask_by_fold=tuple(mask_by_fold))
