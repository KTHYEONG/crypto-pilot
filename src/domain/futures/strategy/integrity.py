from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import FeatureIntegrityConfig
from src.domain.futures.strategy.contracts import FeaturePanel
from src.domain.futures.strategy.diagnostics import rolling_ic


@dataclass(slots=True, frozen=True)
class DataIntegrityReport:
    zero_price_ratio: float
    ohlc_violation_ratio: float
    bar_gap_count: int
    source_coverage: dict[str, dict[str, float]]
    nan_decomposition: dict[str, float]
    coverage_within_eligible: float  # |finite(target)∩eligible| / |eligible|, OOS slice
    hard_fail: bool
    fail_reasons: list[str]


@dataclass(slots=True, frozen=True)
class FeatureIntegrityReport:
    per_feature: dict[str, dict[str, float]]
    constant_features: list[str]
    drifted_features: list[str]
    redundant_pairs: list[tuple[str, str, float]]
    leakage_suspects: list[str]


def _slice_ratio(arr: NDArray[np.float64] | None, sl: slice) -> float:
    if arr is None:
        return 0.0
    seg = np.asarray(arr[sl], dtype=np.float64)
    if seg.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(seg)))


def _psi(train_vals: NDArray[np.float64], oos_vals: NDArray[np.float64]) -> float:
    train = train_vals[np.isfinite(train_vals)]
    oos = oos_vals[np.isfinite(oos_vals)]
    if train.size < 16 or oos.size < 16:
        return 0.0
    q = np.quantile(train, np.linspace(0.0, 1.0, 11))
    q = np.unique(q)
    if q.size < 3:
        return 0.0
    eps = 1e-6
    tr_hist, _ = np.histogram(train, bins=q)
    oo_hist, _ = np.histogram(oos, bins=q)
    tr_p = tr_hist / max(np.sum(tr_hist), 1)
    oo_p = oo_hist / max(np.sum(oo_hist), 1)
    tr_p = np.clip(tr_p.astype(np.float64), eps, 1.0)
    oo_p = np.clip(oo_p.astype(np.float64), eps, 1.0)
    return float(np.sum((oo_p - tr_p) * np.log(oo_p / tr_p)))


def verify_data_integrity(
    aligned: AlignedMarketData,
    *,
    oos_start_idx: int,
    forward_gross_ret: NDArray[np.float64],
    eligible_mask: NDArray[np.bool_],
) -> DataIntegrityReport:
    oos_slice = slice(max(0, int(oos_start_idx)), aligned.close_2d.shape[0])
    active = np.asarray(aligned.active_mask, dtype=bool)
    px_invalid = (
        (~np.isfinite(aligned.open_2d))
        | (~np.isfinite(aligned.close_2d))
        | (aligned.open_2d <= 0.0)
        | (aligned.close_2d <= 0.0)
    )
    active_cnt = int(np.count_nonzero(active))
    zero_price_ratio = float(np.count_nonzero(px_invalid & active) / max(active_cnt, 1))

    ohlc_violation = (
        (aligned.high_2d < np.maximum.reduce([aligned.open_2d, aligned.close_2d, aligned.low_2d]))
        | (aligned.low_2d > np.minimum.reduce([aligned.open_2d, aligned.close_2d, aligned.high_2d]))
    ) & active
    ohlc_violation_ratio = float(np.count_nonzero(ohlc_violation) / max(active_cnt, 1))

    dt_ns = aligned.datetimes.astype("datetime64[ns]").astype(np.int64)
    bar_gap_count = 0
    if dt_ns.size >= 3:
        diffs = np.diff(dt_ns)
        median_gap = int(np.median(diffs))
        bar_gap_count = int(np.count_nonzero(diffs != median_gap))

    is_slice = slice(0, oos_slice.start)
    source_coverage = {
        "funding_2d": {
            "is": _slice_ratio(aligned.funding_2d, is_slice),
            "oos": _slice_ratio(aligned.funding_2d, oos_slice),
        },
        "basis_2d": {
            "is": _slice_ratio(aligned.basis_2d, is_slice),
            "oos": _slice_ratio(aligned.basis_2d, oos_slice),
        },
        "oi_2d": {
            "is": _slice_ratio(aligned.oi_2d, is_slice),
            "oos": _slice_ratio(aligned.oi_2d, oos_slice),
        },
        "adv_usdt_2d": {
            "is": _slice_ratio(aligned.adv_usdt_2d, is_slice),
            "oos": _slice_ratio(aligned.adv_usdt_2d, oos_slice),
        },
    }

    target_nan = ~np.isfinite(np.asarray(forward_gross_ret, dtype=np.float64))[oos_slice]
    eligible = np.asarray(eligible_mask[oos_slice], dtype=bool)
    kill = np.asarray(aligned.kill_mask[oos_slice], dtype=bool)
    warm = np.asarray(aligned.warm_mask[oos_slice], dtype=bool)
    valid_px = ~px_invalid[oos_slice]

    denom = int(np.count_nonzero(target_nan))
    if denom == 0:
        nan_decomp = {
            "universe_inactive": 0.0,
            "price_missing": 0.0,
            "warmup": 0.0,
            "kill": 0.0,
        }
    else:
        u_inactive = target_nan & (~eligible)
        p_missing = target_nan & eligible & (~valid_px)
        wup = target_nan & eligible & valid_px & warm
        kll = target_nan & eligible & valid_px & (~warm) & kill
        unknown = target_nan & ~(u_inactive | p_missing | wup | kll)
        kll = kll | unknown
        nan_decomp = {
            "universe_inactive": float(np.count_nonzero(u_inactive) / denom),
            "price_missing": float(np.count_nonzero(p_missing) / denom),
            "warmup": float(np.count_nonzero(wup) / denom),
            "kill": float(np.count_nonzero(kll) / denom),
        }

    fwd_oos = np.asarray(forward_gross_ret, dtype=np.float64)[oos_slice]
    elig_cnt = int(np.count_nonzero(eligible))
    finite_in_elig = int(np.count_nonzero(np.isfinite(fwd_oos) & eligible))
    coverage_within_eligible = float(finite_in_elig / elig_cnt) if elig_cnt > 0 else 0.0

    fail_reasons: list[str] = []
    if zero_price_ratio > 0.0:
        fail_reasons.append("zero_or_nonpositive_price_in_active_mask")
    if ohlc_violation_ratio > 0.0:
        fail_reasons.append("ohlc_consistency_violation")
    hard_fail = len(fail_reasons) > 0

    return DataIntegrityReport(
        zero_price_ratio=zero_price_ratio,
        ohlc_violation_ratio=ohlc_violation_ratio,
        bar_gap_count=bar_gap_count,
        source_coverage=source_coverage,
        nan_decomposition=nan_decomp,
        coverage_within_eligible=coverage_within_eligible,
        hard_fail=hard_fail,
        fail_reasons=fail_reasons,
    )


def verify_feature_integrity(
    features: FeaturePanel,
    *,
    train_slice: slice,
    oos_slice: slice,
    target_2d: NDArray[np.float64],
    breakeven_ic: float,
) -> FeatureIntegrityReport:
    per_feature: dict[str, dict[str, float]] = {}
    constant_features: list[str] = []
    drifted_features: list[str] = []
    leakage_suspects: list[str] = []
    vals = np.asarray(features.values, dtype=np.float64)
    names = tuple(features.feature_names)

    train_feat = vals[train_slice]
    train_target = np.asarray(target_2d[train_slice], dtype=np.float64)
    oos_feat = vals[oos_slice]

    # Cache observed rolling_ic series to avoid double calculations
    obs_ic_cached: list[np.ndarray] = []
    for idx, name in enumerate(names):
        tr = train_feat[:, :, idx]
        oo = oos_feat[:, :, idx]
        nan_tr = float(np.mean(~np.isfinite(tr))) if tr.size else 0.0
        nan_oos = float(np.mean(~np.isfinite(oo))) if oo.size else 0.0
        tr_vals = tr[np.isfinite(tr)]
        std = float(np.std(tr_vals)) if tr_vals.size else 0.0
        psi = _psi(tr.reshape(-1), oo.reshape(-1))
        
        # Calculate observed rolling_ic series once
        ic_series = rolling_ic(tr, train_target, method="spearman")
        obs_ic_cached.append(ic_series)
        
        ic = float(np.nanmean(ic_series)) if np.any(np.isfinite(ic_series)) else 0.0
        gap = float(ic - breakeven_ic)
        per_feature[name] = {
            "nan_tr": nan_tr,
            "nan_oos": nan_oos,
            "std": std,
            "psi": psi,
            "max_corr": 0.0,
            "ic": ic,
            "gap": gap,
        }
        if std < 1e-9:
            constant_features.append(name)
        if psi > 0.25:
            drifted_features.append(name)

    flat = train_feat.reshape(-1, train_feat.shape[2])
    corr = np.corrcoef(np.nan_to_num(flat, nan=0.0), rowvar=False)
    corr_2d = np.atleast_2d(np.asarray(np.nan_to_num(corr, nan=0.0), dtype=np.float64))
    redundant_pairs: list[tuple[str, str, float]] = []
    for i in range(len(names)):
        if len(names) > 1:
            per_feature[names[i]]["max_corr"] = float(
                np.max(np.abs(np.delete(corr_2d[i], i)))
            )
        for j in range(i + 1, len(names)):
            c = float(corr_2d[i, j])
            if abs(c) > 0.95:
                redundant_pairs.append((names[i], names[j], c))

    # Fast permutation test for leakage detection using cached observed IC series
    rng = np.random.default_rng(7)
    train_shape = train_target.shape
    train_target_flat = train_target.copy().reshape(-1)
    
    for idx, name in enumerate(names):
        obs_series = obs_ic_cached[idx]
        obs = float(np.nanmean(np.abs(obs_series))) if np.any(np.isfinite(obs_series)) else 0.0
        
        tr = train_feat[:, :, idx]
        null_vals: list[float] = []
        
        # 8 permutation shuffles
        for _ in range(8):
            rng.shuffle(train_target_flat)
            perm = train_target_flat.reshape(train_shape)
            ic_series = rolling_ic(tr, perm, method="spearman")
            if np.any(np.isfinite(ic_series)):
                null_vals.append(float(np.nanmean(np.abs(ic_series))))
                
        if null_vals:
            thr = float(np.mean(null_vals) + 5.0 * np.std(null_vals))
            if obs > max(thr, 0.10):
                leakage_suspects.append(name)


    return FeatureIntegrityReport(
        per_feature=per_feature,
        constant_features=constant_features,
        drifted_features=drifted_features,
        redundant_pairs=redundant_pairs,
        leakage_suspects=leakage_suspects,
    )


def select_features(
    report: FeatureIntegrityReport,
    feature_names: tuple[str, ...],
    cfg: FeatureIntegrityConfig,
) -> tuple[str, ...]:
    kept: list[str] = []
    dropped: set[str] = set()
    pair_map: dict[str, set[str]] = {}
    for left, right, corr in report.redundant_pairs:
        if abs(corr) <= cfg.tau_corr:
            continue
        pair_map.setdefault(left, set()).add(right)
        pair_map.setdefault(right, set()).add(left)

    for name in feature_names:
        row = report.per_feature.get(name)
        if row is None:
            continue
        if row["nan_tr"] > cfg.tau_nan or row["std"] < cfg.epsilon or row["psi"] > cfg.tau_psi:
            dropped.add(name)
            continue
        if name in report.leakage_suspects:
            dropped.add(name)
            continue
        kept.append(name)

    for a, bs in pair_map.items():
        for b in bs:
            if a in dropped or b in dropped:
                continue
            if a not in kept or b not in kept:
                continue
            ic_a = abs(float(report.per_feature.get(a, {}).get("ic", 0.0)))
            ic_b = abs(float(report.per_feature.get(b, {}).get("ic", 0.0)))
            drop = b if ic_a >= ic_b else a
            dropped.add(drop)
            if drop in kept:
                kept.remove(drop)

    ic_floor = cfg.ic_floor
    if ic_floor is not None:
        kept = [n for n in kept if abs(float(report.per_feature[n]["ic"])) >= ic_floor]

    if len(kept) < cfg.min_keep:
        ranked = sorted(
            (
                (name, abs(float(report.per_feature.get(name, {}).get("ic", 0.0))))
                for name in feature_names
                if name not in kept and name not in report.leakage_suspects
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for name, _ in ranked:
            if name in kept:
                continue
            kept.append(name)
            if len(kept) >= cfg.min_keep:
                break
    return tuple(kept)
