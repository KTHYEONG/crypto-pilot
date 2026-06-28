from __future__ import annotations

import logging
import re as _re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.cs_rank import SymbolSignal
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        L2SimulationCache,
        RegimeRoutingPlan,
    )
    from src.domain.futures.strategy.walk_forward import WFFold

from src.domain.futures.strategy.market_regime import compress_regime_codes
from src.domain.futures.strategy.regime_evaluation import evaluate_regime_lift_proof
from src.domain.futures.strategy.tiered_workflow.bucket_reliability import (
    build_bucket_reliability,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    RegimeBucketReliability,
    RegimeCellDebugStat,
    RegimeCellPolicy,
    RegimeDebugDiagnostics,
    RegimeGranularityDebugStat,
    RegimePolicyApplication,
    RegimePolicyDiagnostics,
    RegimePolicyMode,
    RegimeRoutingDiagnostics,
    RegimeRoutingPlan,
)
from src.domain.futures.strategy.tiered_workflow.metrics import _newey_west_ic_tstat

logger = logging.getLogger(__name__)


def _build_bucket_reliability(
    *,
    regime: int,
    family: str,
    tf: str,
    fit_edge_bps: float,
    cal_edge_bps: float,
    n_fit: int,
    n_cal: int,
    min_fit_n: int,
    min_cal_n: int,
    min_cal_lift_bps: float,
    min_reliability: float,
) -> RegimeBucketReliability:
    return build_bucket_reliability(
        regime=regime,
        family=family,
        tf=tf,
        fit_edge_bps=fit_edge_bps,
        cal_edge_bps=cal_edge_bps,
        n_fit=n_fit,
        n_cal=n_cal,
        min_fit_n=min_fit_n,
        min_cal_n=min_cal_n,
        min_cal_lift_bps=min_cal_lift_bps,
        min_reliability=min_reliability,
    )


@dataclass(frozen=True, slots=True)
class SleeveMetaSamples:
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    event_t: NDArray[np.int64]
    event_sym: NDArray[np.int64]
    sleeve_tf: tuple[str, ...]
    sleeve_family: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetaFeasibilityReport:
    oos_meta_ic: float
    oos_meta_ic_tstat: float
    net_edge_lift_bps: float
    auc_sign: float
    n_oos: int
    bucket_table: dict[str, float]


@dataclass(frozen=True, slots=True)
class _RegimeEdgeStat:
    edge_bps: float
    n_obs: int


def _parse_meta_group_ids(strategy_id: str) -> tuple[str, str]:
    m = _re.search(r"_(\d+h)$", strategy_id)
    if m:
        tf = m.group(1)
        family = strategy_id[: m.start()]
        return family, tf
    return "unknown", "unknown"


def _available_micro_feature(
    aligned: AlignedMarketData,
    name: str,
    t: int,
    sym_col: int,
) -> float | None:
    arr = getattr(aligned, name, None)
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and t < arr.shape[0] and sym_col < arr.shape[1]:
        v = float(arr[t, sym_col])
        return v if np.isfinite(v) else None
    return None


def build_sleeve_meta_dataset(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    regime_code_1d: NDArray[np.int8],
    start: int,
    end: int,
    *,
    cost_bps: float,
) -> SleeveMetaSamples:
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0 or start >= end:
        return SleeveMetaSamples(
            X=np.empty((0, 0), dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            event_t=np.empty(0, dtype=np.int64),
            event_sym=np.empty(0, dtype=np.int64),
            sleeve_tf=(),
            sleeve_family=(),
            feature_names=(),
        )

    t_max = cache.signal_mask_2d.shape[0]
    if end > t_max:
        end = t_max

    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)
    n_sym = close_2d.shape[1]

    forward_bps = np.zeros((t_max, n_sym), dtype=np.float64)
    if t_max > 1:
        c = close_2d
        denom = np.maximum(np.abs(c[:-1]), 1e-12)
        forward_bps[:-1] = ((c[1:] - c[:-1]) / denom) * 10000.0

    _has_basis = aligned.basis_2d is not None
    _has_oi = aligned.oi_2d is not None
    _has_lsr = aligned.lsr_2d is not None
    _has_taker = aligned.taker_buy_2d is not None

    feature_rows: list[list[float]] = []
    y_list: list[float] = []
    event_t_list: list[int] = []
    event_sym_list: list[int] = []
    tf_list: list[str] = []
    family_list: list[str] = []

    sleeve_to_sym_arr = cache.sleeve_to_sym
    sleeve_ids = cache.sleeve_ids
    sleeve_to_tf_arr = cache.sleeve_to_tf

    for t in range(start, end):
        mask = cache.signal_mask_2d[t]
        active_js = np.where(mask)[0]
        if len(active_js) == 0:
            continue

        sym_to_sleeves: dict[int, list[int]] = {}
        for j in active_js:
            sc = int(sleeve_to_sym_arr[int(j)])
            sym_to_sleeves.setdefault(sc, []).append(int(j))

        sym_agreement: dict[int, float] = {}
        for sc, js in sym_to_sleeves.items():
            if len(js) <= 1:
                sym_agreement[sc] = 1.0
            else:
                sides = [float(cache.side_2d[t, j]) for j in js]
                pos = sum(1 for s in sides if s > 0)
                neg = sum(1 for s in sides if s < 0)
                sym_agreement[sc] = max(pos, neg) / len(js)

        for j in active_js:
            sj = int(j)
            sym_col = int(sleeve_to_sym_arr[sj])

            regime_v = float(regime_code_1d[t]) if t < len(regime_code_1d) else 0.0
            vol_v = float(cache.vol_matrix_2d[t, sym_col]) if cache.vol_matrix_2d.shape[0] > t else 0.0
            funding_v = float(aligned.funding_2d[t, sym_col]) if aligned.funding_2d.shape[0] > t else 0.0

            row = [regime_v, vol_v, funding_v]

            if _has_basis:
                v = _available_micro_feature(aligned, "basis_2d", t, sym_col)
                row.append(v if v is not None else 0.0)
            if _has_oi:
                v = _available_micro_feature(aligned, "oi_2d", t, sym_col)
                row.append(v if v is not None else 0.0)
            if _has_lsr:
                v = _available_micro_feature(aligned, "lsr_2d", t, sym_col)
                row.append(v if v is not None else 0.0)
            if _has_taker:
                v = _available_micro_feature(aligned, "taker_buy_2d", t, sym_col)
                row.append(v if v is not None else 0.0)

            row.append(sym_agreement[sym_col])
            row.append(float(cache.expected_net_bps_2d[t, sj]))
            row.append(float(cache.quality_weight_2d[t, sj]))

            h = int(cache.holding_bars_2d[t, sj])
            h = max(h, 1)
            end_bar = min(t + h, t_max - 1)
            fwd_sum = float(np.sum(forward_bps[t:end_bar, sym_col]))

            side = float(cache.side_2d[t, sj])
            label = side * fwd_sum - cost_bps

            feature_rows.append(row)
            y_list.append(label)
            event_t_list.append(t)
            event_sym_list.append(sym_col)

            tf_key = sleeve_to_tf_arr[sj] if sj < len(sleeve_to_tf_arr) else "unk"
            tf_list.append(tf_key)

            strat_id = sleeve_ids[sj][1]
            fam_key, _ = _parse_meta_group_ids(strat_id)
            family_list.append(fam_key)

    n_events = len(feature_rows)
    if n_events == 0:
        return SleeveMetaSamples(
            X=np.empty((0, 0), dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            event_t=np.empty(0, dtype=np.int64),
            event_sym=np.empty(0, dtype=np.int64),
            sleeve_tf=(),
            sleeve_family=(),
            feature_names=(),
        )

    x_mat = np.asarray(feature_rows, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.float64)
    event_t_arr = np.asarray(event_t_list, dtype=np.int64)
    event_sym_arr = np.asarray(event_sym_list, dtype=np.int64)
    tf_tuple = tuple(tf_list)
    family_tuple = tuple(family_list)

    base_names = ["regime_code", "vol_bps", "funding_bps"]
    micro_names: list[str] = []
    if _has_basis:
        micro_names.append("basis_bps")
    if _has_oi:
        micro_names.append("oi_bps")
    if _has_lsr:
        micro_names.append("lsr_bps")
    if _has_taker:
        micro_names.append("taker_buy_bps")
    extra_names = ["agreement", "expected_net_bps", "quality_weight"]
    feature_names = tuple(base_names + micro_names + extra_names)

    return SleeveMetaSamples(
        X=x_mat,
        y=y_arr,
        event_t=event_t_arr,
        event_sym=event_sym_arr,
        sleeve_tf=tf_tuple,
        sleeve_family=family_tuple,
        feature_names=feature_names,
    )


def _purged_train_val_split(
    event_t: NDArray[np.int64],
    val_start: int,
    val_end: int,
    embargo_bars: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = len(event_t)
    all_idx = np.arange(n, dtype=np.int64)
    val_idx = all_idx[val_start:val_end]
    if len(val_idx) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    train_candidates = np.concatenate([all_idx[:val_start], all_idx[val_end:]])
    if len(train_candidates) == 0:
        return np.empty(0, dtype=np.int64), val_idx

    val_t_min = int(np.min(event_t[val_idx]))
    val_t_max = int(np.max(event_t[val_idx]))

    train_t = event_t[train_candidates]
    keep = ~((train_t >= val_t_min - embargo_bars) & (train_t <= val_t_max + embargo_bars))
    train_idx = train_candidates[keep]
    return train_idx, val_idx


def _fold_bucket_means(
    x_mat: NDArray[np.float64],
    y: NDArray[np.float64],
    event_t: NDArray[np.int64],
    tf_labels: tuple[str, ...],
    family_labels: tuple[str, ...],
    regime_col: int,
    train_idx: NDArray[np.int64],
    val_idx: NDArray[np.int64],
) -> dict[str, float]:
    bucket_map: dict[str, list[float]] = {}
    for idx in train_idx:
        ii = int(idx)
        r = int(x_mat[ii, regime_col])
        tf = tf_labels[ii]
        fam = family_labels[ii]
        key = f"regime={r}/family={fam}/TF={tf}"
        bucket_map.setdefault(key, []).append(float(y[ii]))

    train_means: dict[str, float] = {
        k: float(np.mean(vs)) if vs else 0.0 for k, vs in bucket_map.items()
    }

    val_edges: dict[str, list[float]] = {}
    for idx in val_idx:
        ii = int(idx)
        r = int(x_mat[ii, regime_col])
        tf = tf_labels[ii]
        fam = family_labels[ii]
        key = f"regime={r}/family={fam}/TF={tf}"
        if key in train_means:
            val_edges.setdefault(key, []).append(float(y[ii]))

    return {k: float(np.mean(vs)) if vs else 0.0 for k, vs in val_edges.items()}


def evaluate_meta_feasibility(
    samples: SleeveMetaSamples,
    *,
    n_splits: int,
    embargo_bars: int,
    threshold_quantile: float,
) -> MetaFeasibilityReport:
    _t_meta_start = time.perf_counter()
    n = len(samples.y)
    if n == 0 or samples.X.shape[1] == 0:
        return MetaFeasibilityReport(
            oos_meta_ic=float("nan"),
            oos_meta_ic_tstat=0.0,
            net_edge_lift_bps=0.0,
            auc_sign=float("nan"),
            n_oos=0,
            bucket_table={},
        )

    order = np.argsort(samples.event_t, kind="stable")
    x_sorted = samples.X[order]
    y_sorted = samples.y[order]
    t_sorted = samples.event_t[order]
    tf_sorted = tuple(samples.sleeve_tf[int(i)] for i in order)
    fam_sorted = tuple(samples.sleeve_family[int(i)] for i in order)

    fold_edges = np.linspace(0, n, n_splits + 1, dtype=int)

    fold_ics: list[float] = []
    fold_tstats: list[float] = []
    fold_lifts: list[float] = []
    fold_aucs: list[float] = []
    fold_ns: list[int] = []
    all_bucket_tables: list[dict[str, float]] = []

    for fold in range(n_splits):
        val_start = int(fold_edges[fold])
        val_end = int(fold_edges[fold + 1])

        if val_end - val_start < 2:
            continue

        train_idx, val_idx = _purged_train_val_split(
            t_sorted, val_start, val_end, embargo_bars,
        )

        if len(train_idx) < 10 or len(val_idx) < 4:
            continue

        x_train = x_sorted[train_idx]
        y_train = y_sorted[train_idx]
        x_val = x_sorted[val_idx]
        y_val = y_sorted[val_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        y_sign_train = (y_train > 0).astype(np.float64)
        nonzero_cls = int(np.sum(y_sign_train))
        if nonzero_cls == 0 or nonzero_cls == len(y_sign_train):
            continue

        try:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(x_train_scaled, y_sign_train)
            meta_score = clf.predict_proba(x_val_scaled)[:, 1]
        except Exception:
            logger.warning("[L2-META-FEAS] fold=%d LogisticRegression failed, skipping", fold)
            continue

        if not np.all(np.isfinite(meta_score)) or np.nanstd(meta_score) < 1e-12:
            continue

        ic_val, _ = spearmanr(meta_score, y_val)
        if not np.isfinite(ic_val):
            ic_val = 0.0

        tstat_val = _newey_west_ic_tstat(meta_score, y_val)
        if not np.isfinite(tstat_val):
            tstat_val = 0.0

        threshold = float(np.quantile(meta_score, float(threshold_quantile)))
        high_mask = meta_score > threshold
        lift_val = float(np.mean(y_val[high_mask])) - float(np.mean(y_val)) if np.any(high_mask) else 0.0

        y_sign_val = (y_val > 0).astype(int)
        n_pos = int(np.sum(y_sign_val))
        n_neg = len(y_sign_val) - n_pos
        if n_pos > 0 and n_neg > 0:
            try:
                auc_val = float(roc_auc_score(y_sign_val, meta_score))
            except Exception:
                auc_val = float("nan")
        else:
            auc_val = float("nan")

        logger.info(
            "[L2-META-FEAS] fold=%d meta_ic=%.3f tstat=%.2f "
            "net_lift_bps=%.2f auc=%.3f n=%d",
            fold, ic_val, tstat_val, lift_val, auc_val if np.isfinite(auc_val) else 0.0, len(val_idx),
        )

        fold_ics.append(ic_val)
        fold_tstats.append(tstat_val)
        fold_lifts.append(lift_val)
        fold_aucs.append(auc_val)
        fold_ns.append(len(val_idx))

        regime_col = 0
        if x_sorted.shape[1] > 0:
            bt = _fold_bucket_means(
                x_sorted, y_sorted, t_sorted, tf_sorted, fam_sorted,
                regime_col=regime_col,
                train_idx=train_idx,
                val_idx=val_idx,
            )
            if bt:
                all_bucket_tables.append(bt)

    if not fold_ics:
        return MetaFeasibilityReport(
            oos_meta_ic=float("nan"),
            oos_meta_ic_tstat=0.0,
            net_edge_lift_bps=0.0,
            auc_sign=float("nan"),
            n_oos=0,
            bucket_table={},
        )

    oos_ic = float(np.mean(fold_ics))
    oos_tstat = float(np.mean(fold_tstats))
    oos_lift = float(np.mean(fold_lifts))
    oos_auc = float(np.nanmean(fold_aucs)) if any(np.isfinite(a) for a in fold_aucs) else float("nan")
    oos_n = int(np.sum(fold_ns))

    merged_bucket: dict[str, list[float]] = {}
    for bt in all_bucket_tables:
        for k, v in bt.items():
            merged_bucket.setdefault(k, []).append(v)
    bucket_table = {k: float(np.mean(vs)) for k, vs in merged_bucket.items()}

    logger.debug(
        "[L2-META] evaluate_meta_feasibility took=%.4fs n_splits=%d n_samples=%d",
        time.perf_counter() - _t_meta_start,
        n_splits,
        n,
    )

    return MetaFeasibilityReport(
        oos_meta_ic=oos_ic,
        oos_meta_ic_tstat=oos_tstat,
        net_edge_lift_bps=oos_lift,
        auc_sign=oos_auc,
        n_oos=oos_n,
        bucket_table=bucket_table,
    )


def compute_bucket_realized_edge_stats(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    start: int,
    end: int,
    regime_code_1d: NDArray[np.int8],
    *,
    cost_bps: float = 6.0,
    min_n: int = 30,
    shrinkage: float = 0.3,
) -> dict[tuple[int, str, str], _RegimeEdgeStat]:
    """Compute realized edge statistics by `(regime, family, tf)` on a closed window."""
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0 or start >= end:
        return {}

    t_max, _ = cache.signal_mask_2d.shape
    end = min(end, t_max)

    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)

    sleeve_ids = cache.sleeve_ids

    bucket_sum: dict[tuple[int, str, str], float] = {}
    bucket_cnt: dict[tuple[int, str, str], int] = {}

    for t in range(start, end):
        if t + 1 >= t_max:
            break
        active_js = np.where(cache.signal_mask_2d[t])[0]
        if len(active_js) == 0:
            continue
        regime = int(regime_code_1d[t]) if t < len(regime_code_1d) else 0
        for j in active_js:
            sj = int(j)
            family, tf = _parse_meta_group_ids(sleeve_ids[sj][1])
            edge = _compute_sleeve_realized_edge_bps(
                cache=cache,
                close_2d=close_2d,
                t=t,
                sleeve_idx=sj,
                window_end=end,
                cost_bps=cost_bps,
            )
            key = (regime, family, tf)
            bucket_sum[key] = bucket_sum.get(key, 0.0) + edge
            bucket_cnt[key] = bucket_cnt.get(key, 0) + 1

    if not bucket_sum:
        return {}

    family_raw_edges: dict[str, list[float]] = {}
    for (regime, family, tf), total in bucket_sum.items():
        count = bucket_cnt[(regime, family, tf)]
        family_raw_edges.setdefault(family, []).append(total / count)

    family_prior = {
        family: float(np.mean(edges)) for family, edges in family_raw_edges.items()
    }
    result: dict[tuple[int, str, str], _RegimeEdgeStat] = {}
    for key, total in bucket_sum.items():
        count = bucket_cnt[key]
        raw_edge = total / count
        family = key[1]
        if count < min_n and family in family_prior:
            edge_val = (1.0 - shrinkage) * raw_edge + shrinkage * family_prior[family]
        else:
            edge_val = raw_edge
        result[key] = _RegimeEdgeStat(edge_bps=float(edge_val), n_obs=count)

    if logger.isEnabledFor(logging.DEBUG):
        _n_regimes = len({k[0] for k in bucket_sum})
        _n_families = len({k[1] for k in bucket_sum})
        _n_tfs = len({k[2] for k in bucket_sum})
        _n_total_buckets = len(bucket_sum)
        _n_shrunk = sum(1 for count in bucket_cnt.values() if count < min_n)
        logger.debug(
            "[L2-BUCKET-STATS] fit=[%d,%d) n_buckets=%d n_regimes=%d n_fams=%d n_tfs=%d n_shrunk=%d/%d",
            start,
            end,
            _n_total_buckets,
            _n_regimes,
            _n_families,
            _n_tfs,
            _n_shrunk,
            _n_total_buckets,
        )
        for key, total in sorted(bucket_sum.items(), key=lambda item: -item[1]):
            count = bucket_cnt[key]
            raw_edge = total / count
            final_edge = result[key].edge_bps
            logger.debug(
                "[L2-BUCKET-EDGE-FIT] regime=%d family=%s tf=%s cnt=%d raw=%.2f final=%.2f shrunk=%s",
                key[0],
                key[1],
                key[2],
                count,
                raw_edge,
                final_edge,
                count < min_n,
            )

    return result


def compute_bucket_realized_edges(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    fit_start: int,
    fit_end: int,
    regime_code_1d: NDArray[np.int8],
    *,
    cost_bps: float = 6.0,
    min_n: int = 30,
    shrinkage: float = 0.3,
) -> dict[tuple[int, str, str], float]:
    """fit-leg [fit_start, fit_end) 구간에서 버킷별 실현 순엣지 계산.

    버킷 = (regime_code, family, TF) triplet.
    실현엣지 = side_j * fwd_ret(sym_j) * 10000 - cost_bps, bar 단위 평균.
    min_n 미달 버킷은 family prior로 shrinkage 보정.

    Args:
        cache: L2SimulationCache (sleeve 행렬 포함).
        aligned: AlignedMarketData (close_2d 필요).
        fit_start: fit-leg 시작 bar index (inclusive).
        fit_end: fit-leg 종료 bar index (exclusive).
        regime_code_1d: [T] bar별 regime code.
        cost_bps: 거래비용 (bps).
        min_n: 최소 event 수. 미달 시 shrinkage.
        shrinkage: raw_edge -> family prior 축소율.

    Returns:
        {(regime, family, TF): edge_bps}. 미관측 버킷은 포함 안 됨.
    """
    stats = compute_bucket_realized_edge_stats(
        cache=cache,
        aligned=aligned,
        start=fit_start,
        end=fit_end,
        regime_code_1d=regime_code_1d,
        cost_bps=cost_bps,
        min_n=min_n,
        shrinkage=shrinkage,
    )
    return {key: stat.edge_bps for key, stat in stats.items()}


def _compute_sleeve_realized_edge_bps(
    *,
    cache: L2SimulationCache,
    close_2d: NDArray[np.float64],
    t: int,
    sleeve_idx: int,
    window_end: int,
    cost_bps: float,
) -> float:
    """Compute realized sleeve edge with holding-bar-aware exit clamped to the window."""
    t_max = int(close_2d.shape[0])
    if t < 0 or t >= t_max - 1:
        return 0.0

    holding_raw = float(cache.holding_bars_2d[t, sleeve_idx])
    holding_bars = max(round(holding_raw), 1)
    exit_t = min(t + holding_bars, window_end - 1, t_max - 1)
    exit_t = max(exit_t, t)
    sym_col = int(cache.sleeve_to_sym[sleeve_idx])
    entry_price = max(float(abs(close_2d[t, sym_col])), 1e-12)
    realized_bps = (float(close_2d[exit_t, sym_col]) - float(close_2d[t, sym_col])) / entry_price * 10000.0
    return float(cache.side_2d[t, sleeve_idx]) * realized_bps - cost_bps


def compute_pooled_realized_edge_stats(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    start: int,
    end: int,
    *,
    cost_bps: float = 6.0,
    min_n: int = 30,
) -> dict[tuple[str, str], _RegimeEdgeStat]:
    """Compute realized sleeve edge statistics pooled by `(family, tf)`."""
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0 or start >= end:
        return {}

    t_max, _ = cache.signal_mask_2d.shape
    end = min(end, t_max)
    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)

    bucket_sum: dict[tuple[str, str], float] = {}
    bucket_cnt: dict[tuple[str, str], int] = {}
    for t in range(start, end):
        if t + 1 >= t_max:
            break
        active_js = np.where(cache.signal_mask_2d[t])[0]
        if len(active_js) == 0:
            continue
        for j in active_js:
            family, tf = _parse_meta_group_ids(cache.sleeve_ids[int(j)][1])
            edge = _compute_sleeve_realized_edge_bps(
                cache=cache,
                close_2d=close_2d,
                t=t,
                sleeve_idx=int(j),
                window_end=end,
                cost_bps=cost_bps,
            )
            key = (family, tf)
            bucket_sum[key] = bucket_sum.get(key, 0.0) + edge
            bucket_cnt[key] = bucket_cnt.get(key, 0) + 1

    result: dict[tuple[str, str], _RegimeEdgeStat] = {}
    for key, total in bucket_sum.items():
        count = bucket_cnt[key]
        if count >= min_n:
            result[key] = _RegimeEdgeStat(edge_bps=float(total / count), n_obs=count)
    return result


def compute_pooled_realized_edges(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    fit_start: int,
    fit_end: int,
    *,
    cost_bps: float = 6.0,
    min_n: int = 30,
) -> dict[tuple[str, str], float]:
    """Compute realized sleeve edges pooled by (family, tf) over a fit leg."""
    stats = compute_pooled_realized_edge_stats(
        cache=cache,
        aligned=aligned,
        start=fit_start,
        end=fit_end,
        cost_bps=cost_bps,
        min_n=min_n,
    )
    return {key: stat.edge_bps for key, stat in stats.items()}


def replicate_pooled_edges_by_regime(
    pooled_edges: Mapping[tuple[str, str], float],
    *,
    state_count: int,
) -> dict[tuple[int, str, str], float]:
    """Replicate pooled `(family, tf)` edges across every regime state."""
    replicated: dict[tuple[int, str, str], float] = {}
    for (family, tf), edge in pooled_edges.items():
        for state in range(state_count):
            replicated[(state, family, tf)] = float(edge)
    return replicated


def _default_policy_reason(*, mode: RegimePolicyMode, enabled: bool) -> str:
    if not enabled:
        return "policy_disabled"
    return "observe_only" if mode == "observe" else "policy_ready"


def _lift_sign(lift_bps: float, *, allow_threshold: float, block_threshold: float) -> int:
    if lift_bps >= allow_threshold:
        return 1
    if lift_bps <= block_threshold:
        return -1
    return 0


def build_regime_policy_by_fold(
    *,
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    awf_folds: Sequence[WFFold],
    regime_code_1d: NDArray[np.int8],
    state_names: tuple[str, ...],
    mode: RegimePolicyMode = "soft",
    cost_bps: float = 6.0,
    min_n: int = 15,
    shrinkage: float = 0.3,
    cal_min_n: int = 20,
    min_cal_lift_bps: float = 8.0,
    block_lift_bps: float = -12.0,
    downweight_min: float = 0.50,
    downweight_max: float = 1.0,
    min_confidence: float = 0.55,
    hard_block_enabled: bool = False,
    block_min_confidence: float = 0.80,
    require_sign_consistency: bool = True,
    pooled_is_passthrough: bool = False,
    min_fit_n_floor: int = 5,
    require_fit_n_for_downweight: bool = True,
) -> tuple[tuple[dict[tuple[int, str, str], RegimeCellPolicy], ...], RegimePolicyDiagnostics]:
    """Build fold-local regime policy using fit/cal windows only."""
    if mode == "filter":
        return tuple({} for _ in awf_folds), RegimePolicyDiagnostics(
            mode=mode,
            enabled=False,
            global_reliable=False,
            reason="legacy_filter",
            n_cells_total=0,
            n_allow=0,
            n_downweight=0,
            n_block=0,
            n_pooled=0,
            n_unstable=0,
            n_hard_block_eligible=0,
            mean_fit_lift_bps=0.0,
            mean_cal_lift_bps=0.0,
            min_cal_lift_bps=0.0,
            max_cal_lift_bps=0.0,
            mean_confidence=0.0,
            sign_consistency_ratio=0.0,
            hard_block_enabled=hard_block_enabled,
        )

    policy_by_fold: list[dict[tuple[int, str, str], RegimeCellPolicy]] = []
    fit_lifts_all: list[float] = []
    cal_lifts_all: list[float] = []
    confidences_all: list[float] = []
    n_allow = 0
    n_downweight = 0
    n_block = 0
    n_pooled = 0
    n_unstable = 0
    n_hard_block_eligible = 0
    sign_consistent_total = 0
    sign_comparable_total = 0

    for fold in awf_folds:
        fit_stats = compute_bucket_realized_edge_stats(
            cache=cache,
            aligned=aligned,
            start=int(fold.fit_start),
            end=int(fold.fit_end),
            regime_code_1d=regime_code_1d,
            cost_bps=cost_bps,
            min_n=min_n,
            shrinkage=shrinkage,
        )
        cal_stats = compute_bucket_realized_edge_stats(
            cache=cache,
            aligned=aligned,
            start=int(fold.cal_start),
            end=int(fold.cal_end),
            regime_code_1d=regime_code_1d,
            cost_bps=cost_bps,
            min_n=min_n,
            shrinkage=shrinkage,
        )
        pooled_fit_stats = compute_pooled_realized_edge_stats(
            cache=cache,
            aligned=aligned,
            start=int(fold.fit_start),
            end=int(fold.fit_end),
            cost_bps=cost_bps,
            min_n=min_n,
        )
        pooled_cal_stats = compute_pooled_realized_edge_stats(
            cache=cache,
            aligned=aligned,
            start=int(fold.cal_start),
            end=int(fold.cal_end),
            cost_bps=cost_bps,
            min_n=min_n,
        )

        keys = sorted(set(fit_stats) | set(cal_stats))
        fold_policy: dict[tuple[int, str, str], RegimeCellPolicy] = {}
        for state, family, tf in keys:
            fit_stat = fit_stats.get((state, family, tf))
            cal_stat = cal_stats.get((state, family, tf))
            pooled_fit = pooled_fit_stats.get((family, tf), _RegimeEdgeStat(0.0, 0))
            pooled_cal = pooled_cal_stats.get((family, tf), _RegimeEdgeStat(0.0, 0))
            fit_edge_bps = float(fit_stat.edge_bps) if fit_stat is not None else 0.0
            cal_edge_bps = float(cal_stat.edge_bps) if cal_stat is not None else 0.0
            n_fit = fit_stat.n_obs if fit_stat is not None else 0
            n_cal = cal_stat.n_obs if cal_stat is not None else 0
            fit_lift_bps = fit_edge_bps - float(pooled_fit.edge_bps)
            cal_lift_bps = cal_edge_bps - float(pooled_cal.edge_bps)
            confidence = min(1.0, float(n_cal) / max(float(cal_min_n), 1.0))
            reliability = _build_bucket_reliability(
                regime=state,
                family=family,
                tf=tf,
                fit_edge_bps=fit_lift_bps,
                cal_edge_bps=cal_lift_bps,
                n_fit=n_fit,
                n_cal=n_cal,
                min_fit_n=min_n,
                min_cal_n=cal_min_n,
                min_cal_lift_bps=min_cal_lift_bps,
                min_reliability=min_confidence,
            )
            fit_lifts_all.append(fit_lift_bps)
            cal_lifts_all.append(cal_lift_bps)
            confidences_all.append(confidence)
            fit_sign = _lift_sign(
                fit_lift_bps,
                allow_threshold=min_cal_lift_bps,
                block_threshold=block_lift_bps,
            )
            cal_sign = _lift_sign(
                cal_lift_bps,
                allow_threshold=min_cal_lift_bps,
                block_threshold=block_lift_bps,
            )
            sign_consistent = fit_sign == 0 or cal_sign == 0 or fit_sign == cal_sign
            if fit_sign != 0 and cal_sign != 0:
                sign_comparable_total += 1
                if sign_consistent:
                    sign_consistent_total += 1

            action: Literal["pooled", "allow", "downweight", "block"] = "pooled"
            reason: str = "neutral"
            edge_multiplier = 1.0
            hard_block_eligible = False
            if n_fit < min_n:
                # B-2: insufficient_fit but good cal → allow
                if n_fit >= min_fit_n_floor and n_cal >= cal_min_n and cal_lift_bps >= min_cal_lift_bps:
                    action = "allow"
                    edge_multiplier = 1.0
                    reason = "insufficient_fit_but_good_cal"
                else:
                    reason = "insufficient_fit"
            elif n_cal < cal_min_n:
                # B-3: insufficient_cal but good fit → partial downweight
                if n_fit >= min_n and fit_lift_bps >= min_cal_lift_bps and not require_fit_n_for_downweight:
                    action = "downweight"
                    edge_multiplier = downweight_max * 0.8
                    reason = "insufficient_cal_partial"
                else:
                    reason = "insufficient_cal"
            elif require_sign_consistency and not sign_consistent and fit_sign != 0 and cal_sign != 0:
                reason = "cal_sign_unstable"
                n_unstable += 1
            elif cal_lift_bps <= block_lift_bps:
                hard_block_eligible = (
                    mode == "hybrid"
                    and hard_block_enabled
                    and confidence >= block_min_confidence
                    and fit_lift_bps <= block_lift_bps
                    and sign_consistent
                )
                if hard_block_eligible:
                    action = "block"
                    edge_multiplier = 0.0
                else:
                    action = "downweight"
                    severity = min(1.0, abs(cal_lift_bps) / max(abs(block_lift_bps), 1e-12))
                    edge_multiplier = downweight_max - severity * (downweight_max - downweight_min)
                    edge_multiplier = float(np.clip(edge_multiplier, downweight_min, downweight_max))
                reason = "negative_cal_lift"
            elif reliability.action == "allow" and cal_lift_bps >= min_cal_lift_bps:
                action = "allow"
                edge_multiplier = 1.0
                reason = "positive_cal_lift"
            elif reliability.action == "downweight" or cal_lift_bps < min_cal_lift_bps:
                action = "downweight"
                severity = max(
                    1.0 - float(reliability.reliability),
                    min(1.0, abs(min(cal_lift_bps, 0.0)) / max(abs(block_lift_bps), 1e-12)),
                )
                edge_multiplier = downweight_max - severity * (downweight_max - downweight_min)
                edge_multiplier = float(np.clip(edge_multiplier, downweight_min, downweight_max))
                reason = "neutral" if cal_lift_bps >= 0.0 else "negative_cal_lift"
            else:
                reason = "global_unreliable"

            # B-1: pooled passthrough — optional override
            if pooled_is_passthrough and action == "pooled":
                action = "allow"
                edge_multiplier = 1.0
                reason = "pooled_passthrough"

            if mode == "observe":
                reason = "observe_only"
            if hard_block_eligible:
                n_hard_block_eligible += 1

            fold_policy[(state, family, tf)] = RegimeCellPolicy(
                state=state,
                state_name=_state_name_for_index(state_names, state),
                family=family,
                tf=tf,
                action=action,
                reason=cast(
                    Literal[
                        "legacy_filter",
                        "observe_only",
                        "global_unreliable",
                        "insufficient_fit",
                        "insufficient_cal",
                        "cal_sign_unstable",
                        "negative_cal_lift",
                        "positive_cal_lift",
                        "neutral",
                        "pooled_passthrough",
                        "insufficient_fit_but_good_cal",
                        "insufficient_cal_partial",
                    ],
                    reason,
                ),
                edge_multiplier=float(np.clip(edge_multiplier, 0.0, 1.0)),
                confidence=confidence,
                fit_edge_bps=fit_edge_bps,
                pooled_fit_edge_bps=float(pooled_fit.edge_bps),
                cal_edge_bps=cal_edge_bps,
                pooled_cal_edge_bps=float(pooled_cal.edge_bps),
                fit_lift_bps=fit_lift_bps,
                cal_lift_bps=cal_lift_bps,
                sign_consistent=sign_consistent,
                hard_block_eligible=hard_block_eligible,
                n_fit=n_fit,
                n_cal=n_cal,
                reliability=float(reliability.reliability),
            )
            if action == "allow":
                n_allow += 1
            elif action == "downweight":
                n_downweight += 1
            elif action == "block":
                n_block += 1
            else:
                n_pooled += 1

        policy_by_fold.append(fold_policy)

    enabled = True
    sign_consistency_ratio = (
        float(sign_consistent_total) / float(sign_comparable_total) if sign_comparable_total > 0 else 0.0
    )
    global_reliable = (n_allow + n_downweight + n_block) > 0 and sign_consistency_ratio > 0.0
    if not cal_lifts_all:
        reason = _default_policy_reason(mode=mode, enabled=enabled)
        mean_fit_lift = 0.0
        min_lift = 0.0
        max_lift = 0.0
        mean_lift = 0.0
        mean_conf = 0.0
    else:
        mean_fit_lift = float(np.mean(fit_lifts_all))
        mean_lift = float(np.mean(cal_lifts_all))
        min_lift = float(np.min(cal_lifts_all))
        max_lift = float(np.max(cal_lifts_all))
        mean_conf = float(np.mean(confidences_all))
        reason = "global_unreliable" if not global_reliable else _default_policy_reason(mode=mode, enabled=enabled)
    diagnostics = RegimePolicyDiagnostics(
        mode=mode,
        enabled=enabled,
        global_reliable=global_reliable,
        reason=reason,
        n_cells_total=sum(len(policy_map) for policy_map in policy_by_fold),
        n_allow=n_allow,
        n_downweight=n_downweight,
        n_block=n_block,
        n_pooled=n_pooled,
        n_unstable=n_unstable,
        n_hard_block_eligible=n_hard_block_eligible,
        mean_fit_lift_bps=mean_fit_lift,
        mean_cal_lift_bps=mean_lift,
        min_cal_lift_bps=min_lift,
        max_cal_lift_bps=max_lift,
        mean_confidence=mean_conf,
        sign_consistency_ratio=sign_consistency_ratio,
        hard_block_enabled=hard_block_enabled,
    )
    return tuple(policy_by_fold), diagnostics


def apply_regime_cell_policy(
    sleeve_sigs: Mapping[tuple[str, str], SymbolSignal],
    sleeve_edges: Mapping[tuple[str, str], float],
    policy_map: Mapping[tuple[int, str, str], RegimeCellPolicy],
    regime_now: int,
    *,
    mode: RegimePolicyMode,
    scale_signal_mu: bool = True,
    scale_quality_weight: bool = True,
) -> RegimePolicyApplication:
    """Apply regime policy to current sleeve signals and edge scores."""
    if not sleeve_sigs:
        return RegimePolicyApplication(
            sleeve_sigs={},
            sleeve_edges={},
            n_input=0,
            n_allow=0,
            n_downweight=0,
            n_block=0,
            n_pooled=0,
        )

    def _scale_symbol_signal(
        sig: SymbolSignal,
        *,
        multiplier: float,
        scale_quality_weight_inner: bool,
    ) -> SymbolSignal:
        clipped = float(np.clip(multiplier, 0.0, 1.0))
        quality_weight = (
            max(float(sig.quality_weight), 0.0) * clipped
            if scale_quality_weight_inner
            else sig.quality_weight
        )
        return replace(
            sig,
            raw_mu=float(sig.raw_mu) * clipped,
            quality_weight=quality_weight,
        )

    next_sigs: dict[tuple[str, str], SymbolSignal] = {}
    next_edges: dict[tuple[str, str], float] = {}
    n_allow = 0
    n_downweight = 0
    n_block = 0
    n_pooled = 0
    gross_edge_before_bps = float(sum(abs(float(edge)) for edge in sleeve_edges.values()))
    abs_mu_before_bps = float(sum(abs(float(sig.raw_mu)) for sig in sleeve_sigs.values()))
    quality_weight_before = float(sum(max(float(sig.quality_weight), 0.0) for sig in sleeve_sigs.values()))
    for key, sig in sleeve_sigs.items():
        family, tf = _parse_meta_group_ids(key[1])
        policy = policy_map.get((regime_now, family, tf))
        gross_bps = float(sleeve_edges.get(key, 0.0))
        if policy is None:
            next_sigs[key] = sig
            next_edges[key] = gross_bps
            n_pooled += 1
            continue

        intended_action = policy.action
        multiplier = 1.0
        if mode == "observe":
            pass
        elif intended_action == "block":
            if mode == "hybrid" and policy.hard_block_eligible:
                n_block += 1
                continue
            if policy.edge_multiplier > 0.0:
                multiplier = policy.edge_multiplier
        elif intended_action == "downweight" and mode in {"soft", "hybrid"}:
            multiplier = policy.edge_multiplier

        next_edges[key] = gross_bps * multiplier
        next_sigs[key] = (
            _scale_symbol_signal(
                sig,
                multiplier=multiplier,
                scale_quality_weight_inner=scale_quality_weight,
            )
            if scale_signal_mu
            else (
                replace(
                    sig,
                    quality_weight=max(float(sig.quality_weight), 0.0) * float(np.clip(multiplier, 0.0, 1.0)),
                )
                if scale_quality_weight and multiplier != 1.0
                else sig
            )
        )

        if intended_action == "allow":
            n_allow += 1
        elif intended_action == "downweight":
            n_downweight += 1
        elif intended_action == "block":
            if mode == "observe":
                n_block += 1
            elif mode == "hybrid" and policy.hard_block_eligible:
                pass
            else:
                n_downweight += 1
        else:
            n_pooled += 1

    gross_edge_after_bps = float(sum(abs(float(edge)) for edge in next_edges.values()))
    abs_mu_after_bps = float(sum(abs(float(sig.raw_mu)) for sig in next_sigs.values()))
    quality_weight_after = float(sum(max(float(sig.quality_weight), 0.0) for sig in next_sigs.values()))
    return RegimePolicyApplication(
        sleeve_sigs=next_sigs,
        sleeve_edges=next_edges,
        n_input=len(sleeve_sigs),
        n_allow=n_allow,
        n_downweight=n_downweight,
        n_block=n_block,
        n_pooled=n_pooled,
        gross_edge_before_bps=gross_edge_before_bps,
        gross_edge_after_bps=gross_edge_after_bps,
        abs_mu_before_bps=abs_mu_before_bps,
        abs_mu_after_bps=abs_mu_after_bps,
        quality_weight_before=quality_weight_before,
        quality_weight_after=quality_weight_after,
    )


def apply_regime_risk_cap(
    weights: NDArray[np.float64],
    regime_now: int,
    state_names: tuple[str, ...],
    *,
    enabled: bool = True,
    bull_gross_cap: float = 1.0,
    bear_gross_cap: float = 0.75,
    crisis_gross_cap: float = 0.55,
) -> tuple[NDArray[np.float64], float]:
    """Scale portfolio weights so state-specific gross exposure does not exceed the configured cap."""
    if bull_gross_cap <= 0.0 or bull_gross_cap > 1.0:
        raise ValueError("bull_gross_cap must be in range (0.0, 1.0]")
    if bear_gross_cap <= 0.0 or bear_gross_cap > 1.0:
        raise ValueError("bear_gross_cap must be in range (0.0, 1.0]")
    if crisis_gross_cap <= 0.0 or crisis_gross_cap > 1.0:
        raise ValueError("crisis_gross_cap must be in range (0.0, 1.0]")
    if not enabled:
        return np.asarray(weights, dtype=np.float64).copy(), 1.0
    arr = np.asarray(weights, dtype=np.float64)
    gross = float(np.sum(np.abs(arr)))
    if gross <= 0.0:
        return arr.copy(), 1.0
    state_name = state_names[regime_now] if 0 <= regime_now < len(state_names) else "crisis"
    if state_name.startswith("bull"):
        cap = bull_gross_cap
    elif state_name.startswith("bear"):
        cap = bear_gross_cap
    else:
        cap = crisis_gross_cap
    if gross <= cap:
        return arr.copy(), 1.0
    multiplier = float(cap / gross)
    return arr * multiplier, multiplier


def _compute_js_divergence(
    codes: NDArray[np.int8],
    *,
    fit_start: int,
    oos_start: int,
    oos_end: int,
    state_count: int,
) -> float:
    fit_slice = codes[fit_start:oos_start]
    oos_slice = codes[oos_start:oos_end]
    if len(fit_slice) == 0 or len(oos_slice) == 0:
        return 0.0

    fit_freq = np.zeros(state_count, dtype=np.float64)
    oos_freq = np.zeros(state_count, dtype=np.float64)
    fit_uniq, fit_counts = np.unique(fit_slice, return_counts=True)
    oos_uniq, oos_counts = np.unique(oos_slice, return_counts=True)
    for state, count in zip(fit_uniq.tolist(), fit_counts.tolist(), strict=True):
        if 0 <= int(state) < state_count:
            fit_freq[int(state)] = float(count) / float(len(fit_slice))
    for state, count in zip(oos_uniq.tolist(), oos_counts.tolist(), strict=True):
        if 0 <= int(state) < state_count:
            oos_freq[int(state)] = float(count) / float(len(oos_slice))
    mean_freq = (fit_freq + oos_freq) / 2.0
    js = 0.0
    for p, q in zip(fit_freq, mean_freq, strict=True):
        if p > 0.0 and q > 0.0:
            js += p * np.log2(p / q)
    for p, q in zip(oos_freq, mean_freq, strict=True):
        if p > 0.0 and q > 0.0:
            js += p * np.log2(p / q)
    return float(js / 2.0)


@dataclass(slots=True)
class _RegimeCellAccumulator:
    fold_idx: int
    state: int
    state_name: str
    family: str
    tf: str
    n_fit: int = 0
    n_oos: int = 0
    oos_realized_sum_bps: float = 0.0
    sign_hit_count: int = 0
    selected_hit_count: int = 0


def _state_name_for_index(state_names: Sequence[str], state: int) -> str:
    if 0 <= state < len(state_names):
        return state_names[state]
    return f"unknown({state})"


def _collect_regime_cell_accumulators(
    *,
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    awf_folds: Sequence[WFFold],
    regime_code_1d: NDArray[np.int8],
    state_names: tuple[str, ...],
    bucket_edges_by_fold: Sequence[Mapping[tuple[int, str, str], float]],
    pooled_edges_by_fold: Sequence[Mapping[tuple[str, str], float]],
    cost_bps: float,
    edge_floor_bps: float,
) -> dict[tuple[int, int, str, str], _RegimeCellAccumulator]:
    t_max = int(cache.signal_mask_2d.shape[0])
    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)
    accumulators: dict[tuple[int, int, str, str], _RegimeCellAccumulator] = {}

    for fold_idx, fold in enumerate(awf_folds):
        fit_start = int(fold.fit_start)
        fit_end = int(fold.oos_start)
        oos_start = int(fold.oos_start)
        oos_end = min(int(fold.oos_end), t_max)
        fit_edges = bucket_edges_by_fold[fold_idx] if fold_idx < len(bucket_edges_by_fold) else {}

        for t in range(fit_start, fit_end):
            if t + 1 >= t_max:
                break
            active_js = np.where(cache.signal_mask_2d[t])[0]
            if len(active_js) == 0:
                continue
            state = int(regime_code_1d[t]) if t < regime_code_1d.shape[0] else 0
            state_name = _state_name_for_index(state_names, state)
            for j in active_js:
                family, tf = _parse_meta_group_ids(cache.sleeve_ids[int(j)][1])
                key = (fold_idx, state, family, tf)
                acc = accumulators.get(key)
                if acc is None:
                    acc = _RegimeCellAccumulator(
                        fold_idx=fold_idx,
                        state=state,
                        state_name=state_name,
                        family=family,
                        tf=tf,
                    )
                    accumulators[key] = acc
                acc.n_fit += 1

        for t in range(oos_start, oos_end):
            if t + 1 >= t_max:
                break
            active_js = np.where(cache.signal_mask_2d[t])[0]
            if len(active_js) == 0:
                continue
            state = int(regime_code_1d[t]) if t < regime_code_1d.shape[0] else 0
            state_name = _state_name_for_index(state_names, state)
            for j in active_js:
                family, tf = _parse_meta_group_ids(cache.sleeve_ids[int(j)][1])
                key = (fold_idx, state, family, tf)
                acc = accumulators.get(key)
                if acc is None:
                    acc = _RegimeCellAccumulator(
                        fold_idx=fold_idx,
                        state=state,
                        state_name=state_name,
                        family=family,
                        tf=tf,
                    )
                    accumulators[key] = acc
                realized_edge = _compute_sleeve_realized_edge_bps(
                    cache=cache,
                    close_2d=close_2d,
                    t=t,
                    sleeve_idx=int(j),
                        window_end=oos_end,
                        cost_bps=cost_bps,
                    )
                fit_edge = float(fit_edges.get((state, family, tf), 0.0))
                acc.n_oos += 1
                acc.oos_realized_sum_bps += realized_edge
                if fit_edge > edge_floor_bps:
                    acc.selected_hit_count += 1
                if fit_edge != 0.0 and realized_edge != 0.0 and np.sign(fit_edge) == np.sign(realized_edge):
                    acc.sign_hit_count += 1

    return accumulators


def _finalize_cell_stats(
    accumulators: Mapping[tuple[int, int, str, str], _RegimeCellAccumulator],
    *,
    bucket_edges_by_fold: Sequence[Mapping[tuple[int, str, str], float]],
    pooled_edges_by_fold: Sequence[Mapping[tuple[str, str], float]],
) -> tuple[RegimeCellDebugStat, ...]:
    stats: list[RegimeCellDebugStat] = []
    for key in sorted(accumulators):
        acc = accumulators[key]
        fit_edges = bucket_edges_by_fold[acc.fold_idx] if acc.fold_idx < len(bucket_edges_by_fold) else {}
        pooled_edges = pooled_edges_by_fold[acc.fold_idx] if acc.fold_idx < len(pooled_edges_by_fold) else {}
        fit_edge_bps = float(fit_edges.get((acc.state, acc.family, acc.tf), 0.0))
        pooled_fit_edge_bps = float(pooled_edges.get((acc.family, acc.tf), 0.0))
        oos_realized_edge_bps = acc.oos_realized_sum_bps / max(acc.n_oos, 1)
        stats.append(
            RegimeCellDebugStat(
                fold_idx=acc.fold_idx,
                state=acc.state,
                state_name=acc.state_name,
                family=acc.family,
                tf=acc.tf,
                n_fit=acc.n_fit,
                n_oos=acc.n_oos,
                fit_edge_bps=fit_edge_bps,
                pooled_fit_edge_bps=pooled_fit_edge_bps,
                oos_realized_edge_bps=oos_realized_edge_bps,
                edge_gap_bps=oos_realized_edge_bps - fit_edge_bps,
                sign_hit_rate=acc.sign_hit_count / max(acc.n_oos, 1),
                selected_hit_pct=acc.selected_hit_count / max(acc.n_oos, 1),
            )
    )
    return tuple(stats)


def _build_regime_proof_inputs(
    *,
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    awf_folds: Sequence[WFFold],
    regime_code_1d: NDArray[np.int8],
    bucket_edges_by_fold: Sequence[Mapping[tuple[int, str, str], float]],
    pooled_edges_by_fold: Sequence[Mapping[tuple[str, str], float]],
    cost_bps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    t_max = int(cache.signal_mask_2d.shape[0])
    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)
    regime_cond_edges: list[float] = []
    pooled_edges: list[float] = []
    realized_edges: list[float] = []
    fold_ids: list[int] = []

    for fold_idx, fold in enumerate(awf_folds):
        oos_start = int(fold.oos_start)
        oos_end = min(int(fold.oos_end), t_max)
        fit_edges = bucket_edges_by_fold[fold_idx] if fold_idx < len(bucket_edges_by_fold) else {}
        pooled_fit_edges = pooled_edges_by_fold[fold_idx] if fold_idx < len(pooled_edges_by_fold) else {}
        for t in range(oos_start, oos_end):
            if t + 1 >= t_max:
                break
            active_js = np.where(cache.signal_mask_2d[t])[0]
            if len(active_js) == 0:
                continue
            state = int(regime_code_1d[t]) if t < regime_code_1d.shape[0] else 0
            for j in active_js:
                family, tf = _parse_meta_group_ids(cache.sleeve_ids[int(j)][1])
                regime_cond_edges.append(float(fit_edges.get((state, family, tf), 0.0)))
                pooled_edges.append(float(pooled_fit_edges.get((family, tf), 0.0)))
                realized_edges.append(
                    _compute_sleeve_realized_edge_bps(
                        cache=cache,
                        close_2d=close_2d,
                        t=t,
                        sleeve_idx=int(j),
                        window_end=oos_end,
                        cost_bps=cost_bps,
                    )
                )
                fold_ids.append(fold_idx)

    return (
        np.asarray(regime_cond_edges, dtype=np.float64),
        np.asarray(pooled_edges, dtype=np.float64),
        np.asarray(realized_edges, dtype=np.float64),
        np.asarray(fold_ids, dtype=np.int32),
    )


def compute_regime_cell_debug_stats(
    *,
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    awf_folds: Sequence[WFFold],
    regime_code_1d: NDArray[np.int8],
    state_names: tuple[str, ...],
    bucket_edges_by_fold: Sequence[Mapping[tuple[int, str, str], float]],
    pooled_edges_by_fold: Sequence[Mapping[tuple[str, str], float]],
    cost_bps: float = 6.0,
    edge_floor_bps: float = 0.0,
) -> tuple[RegimeCellDebugStat, ...]:
    """Compute fold/state/family/TF regime debug statistics."""
    accumulators = _collect_regime_cell_accumulators(
        cache=cache,
        aligned=aligned,
        awf_folds=awf_folds,
        regime_code_1d=regime_code_1d,
        state_names=state_names,
        bucket_edges_by_fold=bucket_edges_by_fold,
        pooled_edges_by_fold=pooled_edges_by_fold,
        cost_bps=cost_bps,
        edge_floor_bps=edge_floor_bps,
    )
    return _finalize_cell_stats(
        accumulators,
        bucket_edges_by_fold=bucket_edges_by_fold,
        pooled_edges_by_fold=pooled_edges_by_fold,
    )


def _granularity_row(
    *,
    label: Literal["pooled", "effective_3", "raw_6"],
    state_count: int,
    proof_result: object,
    bucket_hit_pct_by_fold: Sequence[float],
    cell_stats: Sequence[RegimeCellDebugStat],
) -> RegimeGranularityDebugStat:
    fit_vals = np.array([stat.fit_edge_bps for stat in cell_stats], dtype=np.float64)
    oos_vals = np.array([stat.oos_realized_edge_bps for stat in cell_stats], dtype=np.float64)
    if fit_vals.size >= 3 and oos_vals.size >= 3:
        from scipy.stats import spearmanr

        ic = float(spearmanr(fit_vals, oos_vals).correlation)
        if not np.isfinite(ic):
            ic = 0.0
    else:
        ic = 0.0
    if fit_vals.size > 0:
        errors = oos_vals - fit_vals
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))
    else:
        rmse = 0.0
        bias = 0.0
    mean_hit = float(np.mean(bucket_hit_pct_by_fold)) if bucket_hit_pct_by_fold else 0.0
    proof_passed = bool(getattr(proof_result, "proof_passed", False))
    conditioning_path = cast(
        Literal["regime_conditioned", "pooled_fallback"],
        getattr(proof_result, "conditioning_path", "pooled_fallback"),
    )
    mean_lift_bps = float(getattr(proof_result, "mean_lift_bps", 0.0))
    nw_tstat = float(getattr(proof_result, "nw_tstat", 0.0))
    fold_pass_ratio = float(getattr(proof_result, "fold_pass_ratio", 0.0))
    n_folds_evaluated = int(getattr(proof_result, "n_folds_evaluated", 0))
    return RegimeGranularityDebugStat(
        label=label,
        state_count=state_count,
        proof_passed=proof_passed,
        conditioning_path=conditioning_path,
        mean_lift_bps=mean_lift_bps,
        nw_tstat=nw_tstat,
        fold_pass_ratio=fold_pass_ratio,
        n_folds_evaluated=n_folds_evaluated,
        bucket_hit_pct_mean=mean_hit,
        oos_cell_ic=ic,
        oos_cell_rmse_bps=rmse,
        oos_cell_bias_bps=bias,
    )


def evaluate_regime_granularity_debug(
    *,
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    awf_folds: Sequence[WFFold],
    raw_regime_code_1d: NDArray[np.int8],
    effective_regime_code_1d: NDArray[np.int8],
    effective_bucket_edges_by_fold: Sequence[Mapping[tuple[int, str, str], float]],
    raw_bucket_edges_by_fold: Sequence[Mapping[tuple[int, str, str], float]],
    pooled_edges_by_fold: Sequence[Mapping[tuple[str, str], float]],
    cost_bps: float = 6.0,
    edge_floor_bps: float = 0.0,
    proof_nw_tstat_threshold: float = 1.5,
    proof_fold_pass_ratio_threshold: float = 0.60,
    max_holding_bars: int = 6,
    top_k: int = 10,
) -> RegimeDebugDiagnostics:
    """Build DEBUG observability payload for pooled / compressed / raw regime layers."""
    pool_regime_code = np.zeros_like(effective_regime_code_1d, dtype=np.int8)
    pooled_bucket_edges_by_fold = tuple(
        replicate_pooled_edges_by_regime(pooled_edges, state_count=1) for pooled_edges in pooled_edges_by_fold
    )
    pooled_stats = compute_regime_cell_debug_stats(
        cache=cache,
        aligned=aligned,
        awf_folds=awf_folds,
        regime_code_1d=pool_regime_code,
        state_names=("pooled",),
        bucket_edges_by_fold=pooled_bucket_edges_by_fold,
        pooled_edges_by_fold=pooled_edges_by_fold,
        cost_bps=cost_bps,
        edge_floor_bps=edge_floor_bps,
    )

    pooled_regime_cond, pooled_pooled_edges, pooled_realized_edges, pooled_fold_ids = _build_regime_proof_inputs(
        cache=cache,
        aligned=aligned,
        awf_folds=awf_folds,
        regime_code_1d=pool_regime_code,
        bucket_edges_by_fold=pooled_bucket_edges_by_fold,
        pooled_edges_by_fold=pooled_edges_by_fold,
        cost_bps=cost_bps,
    )
    pooled_proof = evaluate_regime_lift_proof(
        regime_cond_edges=pooled_regime_cond,
        pooled_edges=pooled_pooled_edges,
        realized_edges=pooled_realized_edges,
        fold_ids=pooled_fold_ids,
        n_regime_cells=max(len(pooled_bucket_edges_by_fold[0]) if pooled_bucket_edges_by_fold else 0, 1),
        nw_tstat_threshold=proof_nw_tstat_threshold,
        fold_pass_ratio_threshold=proof_fold_pass_ratio_threshold,
        max_holding_bars=max_holding_bars,
        proof_enabled=True,
    )
    effective_stats = compute_regime_cell_debug_stats(
        cache=cache,
        aligned=aligned,
        awf_folds=awf_folds,
        regime_code_1d=effective_regime_code_1d,
        state_names=("bull", "bear", "crisis"),
        bucket_edges_by_fold=effective_bucket_edges_by_fold,
        pooled_edges_by_fold=pooled_edges_by_fold,
        cost_bps=cost_bps,
        edge_floor_bps=edge_floor_bps,
    )
    effective_regime_cond, effective_pooled_edges, effective_realized_edges, effective_fold_ids = (
        _build_regime_proof_inputs(
            cache=cache,
            aligned=aligned,
            awf_folds=awf_folds,
            regime_code_1d=effective_regime_code_1d,
            bucket_edges_by_fold=effective_bucket_edges_by_fold,
            pooled_edges_by_fold=pooled_edges_by_fold,
            cost_bps=cost_bps,
        )
    )
    effective_proof = evaluate_regime_lift_proof(
        regime_cond_edges=effective_regime_cond,
        pooled_edges=effective_pooled_edges,
        realized_edges=effective_realized_edges,
        fold_ids=effective_fold_ids,
        n_regime_cells=max(len(effective_bucket_edges_by_fold[0]) if effective_bucket_edges_by_fold else 0, 1),
        nw_tstat_threshold=proof_nw_tstat_threshold,
        fold_pass_ratio_threshold=proof_fold_pass_ratio_threshold,
        max_holding_bars=max_holding_bars,
        proof_enabled=True,
    )
    raw_stats = compute_regime_cell_debug_stats(
        cache=cache,
        aligned=aligned,
        awf_folds=awf_folds,
        regime_code_1d=raw_regime_code_1d,
        state_names=("bull_q", "bull_v", "bear_q", "bear_v", "trans", "crash"),
        bucket_edges_by_fold=raw_bucket_edges_by_fold,
        pooled_edges_by_fold=pooled_edges_by_fold,
        cost_bps=cost_bps,
        edge_floor_bps=edge_floor_bps,
    )
    raw_regime_cond, raw_pooled_edges, raw_realized_edges, raw_fold_ids = _build_regime_proof_inputs(
        cache=cache,
        aligned=aligned,
        awf_folds=awf_folds,
        regime_code_1d=raw_regime_code_1d,
        bucket_edges_by_fold=raw_bucket_edges_by_fold,
        pooled_edges_by_fold=pooled_edges_by_fold,
        cost_bps=cost_bps,
    )
    raw_proof = evaluate_regime_lift_proof(
        regime_cond_edges=raw_regime_cond,
        pooled_edges=raw_pooled_edges,
        realized_edges=raw_realized_edges,
        fold_ids=raw_fold_ids,
        n_regime_cells=max(len(raw_bucket_edges_by_fold[0]) if raw_bucket_edges_by_fold else 0, 1),
        nw_tstat_threshold=proof_nw_tstat_threshold,
        fold_pass_ratio_threshold=proof_fold_pass_ratio_threshold,
        max_holding_bars=max_holding_bars,
        proof_enabled=True,
    )

    granularity_stats = (
        _granularity_row(
            label="pooled",
            state_count=1,
            proof_result=pooled_proof,
            bucket_hit_pct_by_fold=[0.0 for _ in awf_folds],
            cell_stats=pooled_stats,
        ),
        _granularity_row(
            label="effective_3",
            state_count=3,
            proof_result=effective_proof,
            bucket_hit_pct_by_fold=[
                float(x)
                for x in _bucket_hit_pct_by_fold_for(
                    cache=cache,
                    aligned=aligned,
                    awf_folds=awf_folds,
                    regime_code_1d=effective_regime_code_1d,
                    bucket_edges_by_fold=effective_bucket_edges_by_fold,
                    edge_floor_bps=edge_floor_bps,
                )
            ],
            cell_stats=effective_stats,
        ),
        _granularity_row(
            label="raw_6",
            state_count=6,
            proof_result=raw_proof,
            bucket_hit_pct_by_fold=[
                float(x)
                for x in _bucket_hit_pct_by_fold_for(
                    cache=cache,
                    aligned=aligned,
                    awf_folds=awf_folds,
                    regime_code_1d=raw_regime_code_1d,
                    bucket_edges_by_fold=raw_bucket_edges_by_fold,
                    edge_floor_bps=edge_floor_bps,
                )
            ],
            cell_stats=raw_stats,
        ),
    )

    effective_state_count = max(
        (state for fold_map in effective_bucket_edges_by_fold for state, _, _ in fold_map),
        default=-1,
    ) + 1
    if effective_state_count <= 0 and effective_regime_code_1d.size > 0:
        effective_state_count = int(np.max(effective_regime_code_1d)) + 1

    if effective_stats and effective_state_count > 0:
        state_return_sum = np.zeros(effective_state_count, dtype=np.float64)
        state_return_count = np.zeros(effective_state_count, dtype=np.int64)
        for stat in effective_stats:
            if 0 <= stat.state < effective_state_count:
                state_return_sum[stat.state] += stat.oos_realized_edge_bps * stat.n_oos
                state_return_count[stat.state] += stat.n_oos
        selected_regime_return_bps = tuple(
            float(state_return_sum[idx] / max(state_return_count[idx], 1)) for idx in range(effective_state_count)
        )
        selected_regime_bar_count = tuple(int(state_return_count[idx]) for idx in range(effective_state_count))
    else:
        selected_regime_return_bps = tuple(0.0 for _ in range(max(effective_state_count, 0)))
        selected_regime_bar_count = tuple(0 for _ in range(max(effective_state_count, 0)))

    def _mean_lift_for(label: str) -> float:
        for stat in granularity_stats:
            if stat.label == label:
                return float(stat.mean_lift_bps)
        return 0.0

    top_positive_cells = tuple(sorted(effective_stats, key=lambda s: s.oos_realized_edge_bps, reverse=True)[:top_k])
    top_negative_cells = tuple(sorted(effective_stats, key=lambda s: s.oos_realized_edge_bps)[:top_k])
    worst_error_cells = tuple(sorted(effective_stats, key=lambda s: abs(s.edge_gap_bps), reverse=True)[:top_k])

    return RegimeDebugDiagnostics(
        granularity_stats=granularity_stats,
        top_positive_cells=top_positive_cells,
        top_negative_cells=top_negative_cells,
        worst_error_cells=worst_error_cells,
        compression_loss_bps=_mean_lift_for("raw_6") - _mean_lift_for("effective_3"),
        selected_regime_return_bps=selected_regime_return_bps,
        selected_regime_bar_count=selected_regime_bar_count,
    )


def _bucket_hit_pct_by_fold_for(
    *,
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    awf_folds: Sequence[WFFold],
    regime_code_1d: NDArray[np.int8],
    bucket_edges_by_fold: Sequence[Mapping[tuple[int, str, str], float]],
    edge_floor_bps: float,
) -> tuple[float, ...]:
    t_max = int(cache.signal_mask_2d.shape[0])
    hit_pcts: list[float] = []
    for fold_idx, fold in enumerate(awf_folds):
        oos_start = int(fold.oos_start)
        oos_end = min(int(fold.oos_end), t_max)
        if oos_start >= oos_end:
            hit_pcts.append(0.0)
            continue
        active_bars = 0
        bars_with_hit = 0
        for t in range(oos_start, oos_end):
            if t + 1 >= t_max:
                break
            active_js = np.where(cache.signal_mask_2d[t])[0]
            if len(active_js) == 0:
                continue
            active_bars += 1
            state = int(regime_code_1d[t]) if t < regime_code_1d.shape[0] else 0
            has_hit = False
            for j in active_js:
                family, tf = _parse_meta_group_ids(cache.sleeve_ids[int(j)][1])
                if float(bucket_edges_by_fold[fold_idx].get((state, family, tf), 0.0)) > edge_floor_bps:
                    has_hit = True
                    break
            if has_hit:
                bars_with_hit += 1
        hit_pcts.append((float(bars_with_hit) / float(active_bars) * 100.0) if active_bars else 0.0)
    return tuple(hit_pcts)


def build_regime_routing_plan(
    *,
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    awf_folds: Sequence[WFFold],
    raw_regime_code_1d: NDArray[np.int8],
    compression_enabled: bool = True,
    cost_bps: float = 6.0,
    min_n: int = 15,
    shrinkage: float = 0.3,
    proof_enabled: bool = True,
    proof_nw_tstat_threshold: float = 1.5,
    proof_fold_pass_ratio_threshold: float = 0.60,
    max_holding_bars: int = 6,
    fallback_mode: Literal["pooled", "empty"] = "pooled",
    debug_diagnostics_enabled: bool = False,
    edge_floor_bps: float = 0.0,
    debug_top_k: int = 10,
    policy_mode: RegimePolicyMode = "hybrid",
    policy_cal_min_n: int = 20,
    policy_min_cal_lift_bps: float = 8.0,
    policy_block_lift_bps: float = -12.0,
    policy_downweight_min: float = 0.50,
    policy_downweight_max: float = 1.0,
    policy_min_confidence: float = 0.55,
    policy_hard_block_enabled: bool = False,
    policy_block_min_confidence: float = 0.80,
    policy_require_sign_consistency: bool = True,
    policy_pooled_is_passthrough: bool = False,
    policy_min_fit_n_floor: int = 5,
    policy_require_fit_n_for_downweight: bool = True,
) -> RegimeRoutingPlan:
    """Build the fold-local routing plan used by L2 bucket selection."""
    n_bars = int(np.asarray(aligned.close_2d).shape[0])
    regime_code_arr = np.asarray(raw_regime_code_1d, dtype=np.int8)
    if regime_code_arr.shape[0] != n_bars:
        raise ValueError(
            f"raw_regime_code_1d length must match aligned.close_2d rows: {regime_code_arr.shape[0]} != {n_bars}"
        )

    effective_codes = compress_regime_codes(regime_code_arr) if compression_enabled else regime_code_arr.copy()
    state_names = ("bull", "bear", "crisis") if compression_enabled else (
        "bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash"
    )
    state_count = len(state_names)

    effective_bucket_edges_fit_by_fold: list[dict[tuple[int, str, str], float]] = []
    raw_bucket_edges_by_fold: list[dict[tuple[int, str, str], float]] = []
    pooled_edges_by_fold: list[dict[tuple[str, str], float]] = []
    proof_regime_cond: list[float] = []
    proof_pooled: list[float] = []
    proof_realized: list[float] = []
    proof_fold_ids: list[int] = []
    bucket_hit_pct_by_fold: list[float] = []
    js_divergence_by_fold: list[float] = []

    t_max = cache.signal_mask_2d.shape[0]
    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)

    for fold_idx, fold in enumerate(awf_folds):
        fit_start = int(fold.fit_start)
        fit_end = int(fold.oos_start)
        oos_start = int(fold.oos_start)
        oos_end = min(int(fold.oos_end), t_max)

        effective_bucket_edges = (
            compute_bucket_realized_edges(
                cache,
                aligned,
                fit_start,
                fit_end,
                effective_codes,
                cost_bps=cost_bps,
                min_n=min_n,
                shrinkage=shrinkage,
            )
            if fit_start < fit_end
            else {}
        )
        raw_bucket_edges = (
            compute_bucket_realized_edges(
                cache,
                aligned,
                fit_start,
                fit_end,
                regime_code_arr,
                cost_bps=cost_bps,
                min_n=min_n,
                shrinkage=shrinkage,
            )
            if compression_enabled and fit_start < fit_end
            else dict(effective_bucket_edges)
        )
        pooled_edges = (
            compute_pooled_realized_edges(
                cache,
                aligned,
                fit_start,
                fit_end,
                cost_bps=cost_bps,
                min_n=min_n,
            )
            if fit_start < fit_end
            else {}
        )
        effective_bucket_edges_fit_by_fold.append(effective_bucket_edges)
        raw_bucket_edges_by_fold.append(raw_bucket_edges)
        pooled_edges_by_fold.append(pooled_edges)
        js_divergence_by_fold.append(
            _compute_js_divergence(
                effective_codes,
                fit_start=fit_start,
                oos_start=oos_start,
                oos_end=oos_end,
                state_count=state_count,
            )
        )

        active_bars = 0
        bars_with_hit = 0
        for t in range(oos_start, oos_end):
            if t + 1 >= t_max:
                break
            active_js = np.where(cache.signal_mask_2d[t])[0]
            if len(active_js) == 0:
                continue
            active_bars += 1
            regime_now = int(effective_codes[t])
            has_hit = False
            for j in active_js:
                family, tf = _parse_meta_group_ids(cache.sleeve_ids[int(j)][1])
                realized_edge = _compute_sleeve_realized_edge_bps(
                    cache=cache,
                    close_2d=close_2d,
                    t=t,
                    sleeve_idx=int(j),
                    window_end=oos_end,
                    cost_bps=cost_bps,
                )
                regime_edge = float(effective_bucket_edges.get((regime_now, family, tf), 0.0))
                pooled_edge = float(pooled_edges.get((family, tf), 0.0))
                proof_regime_cond.append(regime_edge)
                proof_pooled.append(pooled_edge)
                proof_realized.append(realized_edge)
                proof_fold_ids.append(fold_idx)
                if (regime_now, family, tf) in effective_bucket_edges:
                    has_hit = True
            if has_hit:
                bars_with_hit += 1
        bucket_hit_pct_by_fold.append((float(bars_with_hit) / float(active_bars) * 100.0) if active_bars else 0.0)

    proof_result = evaluate_regime_lift_proof(
        regime_cond_edges=np.asarray(proof_regime_cond, dtype=np.float64),
        pooled_edges=np.asarray(proof_pooled, dtype=np.float64),
        realized_edges=np.asarray(proof_realized, dtype=np.float64),
        fold_ids=np.asarray(proof_fold_ids, dtype=np.int32),
        n_regime_cells=max(state_count * max(len(pooled_edges_by_fold[0]) if pooled_edges_by_fold else 0, 1), 1),
        nw_tstat_threshold=proof_nw_tstat_threshold,
        fold_pass_ratio_threshold=proof_fold_pass_ratio_threshold,
        max_holding_bars=max_holding_bars,
        proof_enabled=proof_enabled,
    )
    dsr = float(proof_result.deflated_sharpe)
    if not np.isfinite(dsr):
        dsr = 0.0
    proof_passed = proof_result.proof_passed or (
        proof_enabled
        and proof_result.mean_lift_bps > 0.0
        and proof_result.nw_tstat >= proof_nw_tstat_threshold
        and proof_result.fold_pass_ratio >= proof_fold_pass_ratio_threshold
    )
    conditioning_path: Literal["regime_conditioned", "pooled_fallback"] = (
        "regime_conditioned" if proof_passed else "pooled_fallback"
    )

    if proof_passed:
        effective_bucket_edges_by_fold = tuple(effective_bucket_edges_fit_by_fold)
    elif fallback_mode == "pooled":
        effective_bucket_edges_by_fold = tuple(
            replicate_pooled_edges_by_regime(pooled_edges, state_count=state_count)
            for pooled_edges in pooled_edges_by_fold
        )
    else:
        effective_bucket_edges_by_fold = tuple({} for _ in awf_folds)

    policy_by_fold, policy_diagnostics = build_regime_policy_by_fold(
        cache=cache,
        aligned=aligned,
        awf_folds=awf_folds,
        regime_code_1d=effective_codes,
        state_names=state_names,
        mode=policy_mode,
        cost_bps=cost_bps,
        min_n=min_n,
        shrinkage=shrinkage,
        cal_min_n=policy_cal_min_n,
        min_cal_lift_bps=policy_min_cal_lift_bps,
        block_lift_bps=policy_block_lift_bps,
        downweight_min=policy_downweight_min,
        downweight_max=policy_downweight_max,
        min_confidence=policy_min_confidence,
        hard_block_enabled=policy_hard_block_enabled,
        block_min_confidence=policy_block_min_confidence,
        require_sign_consistency=policy_require_sign_consistency,
        pooled_is_passthrough=policy_pooled_is_passthrough,
        min_fit_n_floor=policy_min_fit_n_floor,
        require_fit_n_for_downweight=policy_require_fit_n_for_downweight,
    )

    debug_diagnostics: RegimeDebugDiagnostics | None = None
    if debug_diagnostics_enabled:
        debug_diagnostics = evaluate_regime_granularity_debug(
            cache=cache,
            aligned=aligned,
            awf_folds=awf_folds,
            raw_regime_code_1d=regime_code_arr,
            effective_regime_code_1d=effective_codes,
            effective_bucket_edges_by_fold=effective_bucket_edges_by_fold,
            raw_bucket_edges_by_fold=raw_bucket_edges_by_fold,
            pooled_edges_by_fold=pooled_edges_by_fold,
            cost_bps=cost_bps,
            edge_floor_bps=edge_floor_bps,
            proof_nw_tstat_threshold=proof_nw_tstat_threshold,
            proof_fold_pass_ratio_threshold=proof_fold_pass_ratio_threshold,
            max_holding_bars=max_holding_bars,
            top_k=debug_top_k,
        )

    diagnostics = RegimeRoutingDiagnostics(
        active_state_count=state_count,
        active_state_names=state_names,
        compression_enabled=compression_enabled,
        proof_passed=proof_passed,
        conditioning_path=conditioning_path,
        mean_lift_bps=proof_result.mean_lift_bps,
        n_eff=proof_result.n_eff,
        nw_tstat=proof_result.nw_tstat,
        deflated_sharpe=dsr,
        fold_pass_ratio=proof_result.fold_pass_ratio,
        n_folds_evaluated=proof_result.n_folds_evaluated,
        bucket_hit_pct_by_fold=tuple(bucket_hit_pct_by_fold),
        js_divergence_by_fold=tuple(js_divergence_by_fold),
        policy_diagnostics=policy_diagnostics,
        debug_diagnostics=debug_diagnostics,
    )
    return RegimeRoutingPlan(
        effective_bucket_edges_by_fold=effective_bucket_edges_by_fold,
        raw_bucket_edges_by_fold=tuple(raw_bucket_edges_by_fold),
        pooled_edges_by_fold=tuple(pooled_edges_by_fold),
        effective_regime_code_1d=effective_codes,
        diagnostics=diagnostics,
        policy_by_fold=policy_by_fold,
    )


def filter_sleeves_by_bucket(
    sleeve_sigs: Mapping[tuple[str, str], SymbolSignal],
    bucket_edges: Mapping[tuple[int, str, str], float],
    regime_now: int,
    *,
    edge_floor_bps: float = 0.0,
) -> dict[tuple[str, str], SymbolSignal]:
    """현재 regime의 버킷 엣지로 sleeve 필터링.

    (sym, strat_id) -> 해당 버킷 edge > edge_floor_bps 인 sleeve만 통과.
    버킷 미관측(KeyError) sleeve는 edge=0 처리 -> 통상 제거됨.

    Args:
        sleeve_sigs: sleeve -> SymbolSignal dict.
        bucket_edges: {(regime, family, TF): edge_bps}.
        regime_now: 현재 bar의 regime code.
        edge_floor_bps: 필터 임계값 (bps). 이하 버킷은 배치 안 함.

    Returns:
        통과한 sleeve만 담긴 dict (순서 보존).
    """
    if not sleeve_sigs:
        return {}

    result: dict[tuple[str, str], SymbolSignal] = {}
    for key, sig in sleeve_sigs.items():
        _, strat_id = key
        family, tf = _parse_meta_group_ids(strat_id)
        bucket_key = (regime_now, family, tf)
        edge = bucket_edges.get(bucket_key, 0.0)
        if edge > edge_floor_bps:
            result[key] = sig

    return result
