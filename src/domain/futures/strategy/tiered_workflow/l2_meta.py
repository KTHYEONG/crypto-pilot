from __future__ import annotations

import logging
import re as _re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
from src.domain.futures.strategy.tiered_workflow.metrics import _newey_west_ic_tstat

logger = logging.getLogger(__name__)


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
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0 or fit_start >= fit_end:
        return {}

    t_max, _ = cache.signal_mask_2d.shape
    if fit_end > t_max:
        fit_end = t_max

    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)
    n_sym = close_2d.shape[1]

    fwd_bps = np.zeros((t_max, n_sym), dtype=np.float64)
    if t_max > 1:
        c = close_2d
        denom = np.maximum(np.abs(c[:-1]), 1e-12)
        fwd_bps[:-1] = ((c[1:] - c[:-1]) / denom) * 10000.0

    sleeve_ids = cache.sleeve_ids
    sleeve_to_sym_arr = cache.sleeve_to_sym

    bucket_sum: dict[tuple[int, str, str], float] = {}
    bucket_cnt: dict[tuple[int, str, str], int] = {}

    for t in range(fit_start, fit_end):
        if t + 1 >= t_max:
            break
        mask = cache.signal_mask_2d[t]
        active_js = np.where(mask)[0]
        if len(active_js) == 0:
            continue

        regime = int(regime_code_1d[t]) if t < len(regime_code_1d) else 0

        for j in active_js:
            sj = int(j)
            strat_id = sleeve_ids[sj][1]
            family, tf = _parse_meta_group_ids(strat_id)
            sym_col = int(sleeve_to_sym_arr[sj])

            edge = float(cache.side_2d[t, sj]) * float(fwd_bps[t, sym_col]) - cost_bps

            key = (regime, family, tf)
            bucket_sum[key] = bucket_sum.get(key, 0.0) + edge
            bucket_cnt[key] = bucket_cnt.get(key, 0) + 1

    if not bucket_sum:
        return {}

    family_raw_edges: dict[str, list[float]] = {}
    for (regime, family, tf), s in bucket_sum.items():
        cnt = bucket_cnt[(regime, family, tf)]
        raw = s / cnt
        family_raw_edges.setdefault(family, []).append(raw)

    family_prior: dict[str, float] = {}
    for family, edges in family_raw_edges.items():
        family_prior[family] = float(np.mean(edges))

    result: dict[tuple[int, str, str], float] = {}
    for (regime, family, tf), s in bucket_sum.items():
        cnt = bucket_cnt[(regime, family, tf)]
        raw = s / cnt
        if cnt < min_n and family in family_prior:
            edge_val = (1.0 - shrinkage) * raw + shrinkage * family_prior[family]
        else:
            edge_val = raw
        result[(regime, family, tf)] = edge_val

    # Step E: aggregate DEBUG stats
    if logger.isEnabledFor(logging.DEBUG):
        _n_regimes = len({k[0] for k in bucket_sum})
        _n_families = len({k[1] for k in bucket_sum})
        _n_tfs = len({k[2] for k in bucket_sum})
        _n_total_buckets = len(bucket_sum)
        _n_shrunk = sum(1 for k, v in bucket_cnt.items() if v < min_n)
        logger.debug(
            "[L2-BUCKET-STATS] fit=[%d,%d) n_buckets=%d n_regimes=%d n_fams=%d n_tfs=%d n_shrunk=%d/%d",
            fit_start, fit_end, _n_total_buckets, _n_regimes, _n_families, _n_tfs, _n_shrunk, _n_total_buckets,
        )
        for (_br, _bfam, _btf), _s in sorted(bucket_sum.items(), key=lambda x: -x[1]):
            _cnt = bucket_cnt[(_br, _bfam, _btf)]
            _raw = _s / _cnt
            _final = result.get((_br, _bfam, _btf), 0.0)
            _shrunk = _cnt < min_n
            logger.debug(
                "[L2-BUCKET-EDGE-FIT] regime=%d family=%s tf=%s cnt=%d raw=%.2f final=%.2f shrunk=%s",
                _br, _bfam, _btf, _cnt, _raw, _final, _shrunk,
            )

    return result


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
    n_sleeve = cache.signal_mask_2d.shape[1]
    if n_sleeve == 0 or fit_start >= fit_end:
        return {}

    t_max, _ = cache.signal_mask_2d.shape
    fit_end = min(fit_end, t_max)
    close_2d = np.asarray(aligned.close_2d, dtype=np.float64)

    bucket_sum: dict[tuple[str, str], float] = {}
    bucket_cnt: dict[tuple[str, str], int] = {}
    for t in range(fit_start, fit_end):
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
                window_end=fit_end,
                cost_bps=cost_bps,
            )
            key = (family, tf)
            bucket_sum[key] = bucket_sum.get(key, 0.0) + edge
            bucket_cnt[key] = bucket_cnt.get(key, 0) + 1

    result: dict[tuple[str, str], float] = {}
    for key, total in bucket_sum.items():
        count = bucket_cnt[key]
        if count >= min_n:
            result[key] = total / count
    return result


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
) -> RegimeRoutingPlan:
    """Build the fold-local routing plan used by L2 bucket selection."""
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        RegimeRoutingDiagnostics,
        RegimeRoutingPlan,
    )

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

        raw_bucket_edges = (
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
                regime_edge = float(raw_bucket_edges.get((regime_now, family, tf), 0.0))
                pooled_edge = float(pooled_edges.get((family, tf), 0.0))
                proof_regime_cond.append(regime_edge)
                proof_pooled.append(pooled_edge)
                proof_realized.append(realized_edge)
                proof_fold_ids.append(fold_idx)
                if (regime_now, family, tf) in raw_bucket_edges:
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
        effective_bucket_edges_by_fold = tuple(raw_bucket_edges_by_fold)
    elif fallback_mode == "pooled":
        effective_bucket_edges_by_fold = tuple(
            replicate_pooled_edges_by_regime(pooled_edges, state_count=state_count)
            for pooled_edges in pooled_edges_by_fold
        )
    else:
        effective_bucket_edges_by_fold = tuple({} for _ in awf_folds)

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
    )
    return RegimeRoutingPlan(
        effective_bucket_edges_by_fold=effective_bucket_edges_by_fold,
        raw_bucket_edges_by_fold=tuple(raw_bucket_edges_by_fold),
        pooled_edges_by_fold=tuple(pooled_edges_by_fold),
        effective_regime_code_1d=effective_codes,
        diagnostics=diagnostics,
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
