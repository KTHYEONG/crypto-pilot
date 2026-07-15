# src/domain/futures/strategy/tiered_workflow/signal_selection.py

from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray

from src.core.utils.utils import PERF
from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    Layer1FoldReadiness,
    Layer1GateCheck,
    Layer1GateReport,
    Layer1InferenceArtifact,
    MatchedBaselineKey,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.candidate_dataset import (
    build_candidate_dataset,
    fit_candidate_feature_schema,
)
from src.domain.futures.strategy.candidate_ensemble import (
    fit_regime_conditional_ensemble,
    predict_regime_conditional_ensemble,
)
from src.domain.futures.strategy.cs_rank import VOL_FLOOR, SymbolSignal
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _one_sided_p_value,
    _series_tstat,
    moving_block_bootstrap_mean,
)

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.walk_forward import WFFold

logger = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def _t_ppf_cached(q_thousandths: int, df_int: int) -> float:
    """Cached t-distribution PPF to avoid repeated scipy calls."""
    return float(stats.t.ppf(q_thousandths / 1000.0, float(df_int)))


def _holding_bucket(holding_bars: int) -> int:
    if holding_bars <= 4:
        return 4
    if holding_bars <= 8:
        return 8
    if holding_bars <= 12:
        return 12
    if holding_bars <= 24:
        return 24
    return int(max(holding_bars, 1))


def _compute_incremental_bps(
    frame: pd.DataFrame,
    *,
    mode: Literal["peer_exclusive", "absolute"],
) -> pd.Series:
    """Compute per-event incremental bps relative to peer strategies.

    Args:
        frame: Event DataFrame with columns gross_event_bps, symbol, side,
            holding_bucket, strategy_id.
        mode: ``peer_exclusive`` uses leave-self-out peer mean as baseline.
            ``absolute`` sets baseline=0 (incremental == gross).

    Returns:
        pd.Series of incremental bps aligned with frame.index.

    Note:
        Time complexity: O(N) two-pass groupby.
        Space: O(B*S) where B=buckets, S=strategies.
        peer_count==0 (single strategy in bucket) falls back to absolute
        (baseline=0) to avoid ``incremental ≡ 0`` degenerate case.
    """
    # gross: shape [N_events]
    gross = frame["gross_event_bps"]
    if mode == "absolute":
        return gross.copy()

    bucket_key = ["symbol", "side", "holding_bucket"]
    strategy_key = [*bucket_key, "strategy_id"]

    bucket_sum = frame.groupby(bucket_key)["gross_event_bps"].transform("sum")
    bucket_count = frame.groupby(bucket_key)["gross_event_bps"].transform("count")
    strat_sum = frame.groupby(strategy_key)["gross_event_bps"].transform("sum")
    strat_count = frame.groupby(strategy_key)["gross_event_bps"].transform("count")

    peer_count = bucket_count - strat_count
    peer_sum = bucket_sum - strat_sum

    safe = peer_count.clip(lower=1)
    peer_mean = np.where(
        peer_count > 0,
        peer_sum.to_numpy(dtype=np.float64) / safe.to_numpy(dtype=np.float64),
        0.0,
    )

    return pd.Series(
        gross.to_numpy(dtype=np.float64) - peer_mean,
        index=frame.index,
    )


def _expected_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.expected_gross_bps, dtype=np.float64)


def _expected_net_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.expected_net_bps, dtype=np.float64)


def _q10_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q10_gross_bps, dtype=np.float64)


def _q10_net_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q10_net_bps, dtype=np.float64)


def _q90_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q90_gross_bps, dtype=np.float64)


def _q90_net_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q90_net_bps, dtype=np.float64)


def _resolve_activation_context(
    row: pd.Series,
    *,
    qualify_by_regime: bool,
) -> str:
    """Resolve activation_context for evidence grouping/activation key.

    When qualify_by_regime=False, collapses all regime cells to "all",
    restoring statistical power via sample pooling.
    """
    if not qualify_by_regime:
        return "all"
    return (
        str(
            row.get(
                "activation_context",
                row.get("signal_cell", row.get("entry_regime", "all")),
            )
        )
        or "all"
    )


def _signal_source_key_from_row(row: pd.Series, *, qualify_by_regime: bool = True) -> SignalSourceKey:
    strategy_id = str(
        row.get(
            "strategy_id",
            f"{row.get('family', '')}:{row.get('variant', '')}",
        )
    )
    activation_context = _resolve_activation_context(row, qualify_by_regime=qualify_by_regime)
    return SignalSourceKey(
        symbol=str(row.get("symbol", "")),
        strategy_id=strategy_id,
        activation_context=activation_context,
    )


def _batch_to_frame(batch: ValidatedSignalBatch) -> pd.DataFrame:
    if not batch.events:
        return pd.DataFrame(
            columns=[
                "decision_idx",
                "decision_time",
                "symbol",
                "strategy_id",
                "activation_context",
                "side",
                "expected_gross_bps",
                "q10_gross_bps",
                "q90_gross_bps",
                "expected_holding_bars",
                "reliability",
                "registry_version",
                "model_version",
            ]
        )
    return pd.DataFrame(
        [
            {
                "decision_idx": event.decision_idx,
                "decision_time": event.decision_time,
                "symbol": event.symbol,
                "strategy_id": event.strategy_id,
                "activation_context": event.activation_context,
                "side": event.side,
                "expected_gross_bps": event.expected_gross_bps,
                "q10_gross_bps": event.q10_gross_bps,
                "q90_gross_bps": event.q90_gross_bps,
                "expected_holding_bars": event.expected_holding_bars,
                "quality_weight": event.quality_weight,
                "registry_version": event.registry_version,
                "model_version": event.model_version,
            }
            for event in batch.events
        ]
    )


def align_outer_opportunities_with_realized(
    *,
    opportunities: ValidatedSignalBatch,
    realized_event_results: pd.DataFrame,
    activation_match_regime: bool,
) -> tuple[pd.DataFrame, int, int]:
    opp_frame = _batch_to_frame(opportunities)
    if opp_frame.empty:
        return opp_frame, 0, 0
    realized = realized_event_results.copy()
    if "strategy_id" not in realized.columns:
        realized["strategy_id"] = (
            realized.get("family", pd.Series("", index=realized.index)).astype(str)
            + ":"
            + realized.get("variant", pd.Series("", index=realized.index)).astype(str)
        )
    if "activation_context" not in realized.columns:
        realized["activation_context"] = realized.get(
            "signal_cell",
            realized.get("entry_regime", pd.Series("all", index=realized.index)),
        ).astype(str)
    realized["decision_idx"] = (
        pd.to_numeric(
            realized.get("entry_idx", pd.Series(0, index=realized.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        - 1
    )
    if "realized_side_adjusted_gross_bps" not in realized.columns:
        realized["realized_side_adjusted_gross_bps"] = pd.to_numeric(
            realized.get("gross_event_bps", pd.Series(np.nan, index=realized.index)),
            errors="coerce",
        )
    merge_keys_full = ["decision_idx", "symbol", "strategy_id"]
    if activation_match_regime:
        merge_keys_full.append("activation_context")
    duplicate_mask = realized.duplicated(subset=merge_keys_full, keep=False)
    if bool(duplicate_mask.any()):
        raise ValueError("duplicate realized opportunity key")
    merge_cols = [*merge_keys_full, "realized_side_adjusted_gross_bps"]
    if "exit_idx" in realized.columns:
        merge_cols.append("exit_idx")
    merged_full = opp_frame.merge(
        realized[merge_cols],
        on=merge_keys_full,
        how="left",
        indicator=True,
    )
    unmatched_full = merged_full.loc[merged_full["_merge"] != "both"]
    # [LIMIT-01] 3-key(activation_context 제외) 재병합으로 label-drift 여부 판별
    label_drift_unmatched_count = 0
    if activation_match_regime and not unmatched_full.empty:
        merge_keys_3 = ["decision_idx", "symbol", "strategy_id"]
        rematch = unmatched_full[merge_keys_3].merge(
            realized[[*merge_keys_3, "realized_side_adjusted_gross_bps"]],
            on=merge_keys_3,
            how="inner",
        )
        label_drift_unmatched_count = len(rematch)
    true_unmatched_count = int((merged_full["_merge"] != "both").sum() - label_drift_unmatched_count)
    merged = merged_full.loc[merged_full["_merge"] == "both"].drop(columns="_merge").copy()
    return merged, true_unmatched_count, label_drift_unmatched_count


def _by_q_values(
    p_values: NDArray[np.float64],
    m_eff: float | None = None,
) -> NDArray[np.float64]:
    if p_values.size == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(p_values)
    ordered = p_values[order]
    # m_eff < p_values.size → less conservative (fewer effective tests)
    m = float(m_eff) if (m_eff is not None and m_eff > 0.0) else float(p_values.size)
    m_int = max(1, round(m))
    harmonic = float(np.sum(1.0 / np.arange(1, m_int + 1, dtype=np.float64)))
    n = ordered.size
    adjusted = np.empty_like(ordered)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        candidate = min(1.0, ordered[idx] * m * harmonic / rank)
        running = min(running, candidate)
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


_PROBE_TF_PATTERN = re.compile(r"_(\d+[hm])$")  # e.g. _6h, _1h, _30m


def _compute_probe_m_eff(
    groups: list[tuple[str, str, str]],
    diversity_corr: dict[str, float],
) -> float:
    """Effective independent tests correcting for correlated probe TF hypotheses.

    Args:
        groups: List of (symbol, strategy_id, activation_context) tuples.
        diversity_corr: Pairwise Pearson r keyed as ``"{sym}:{family}:{tf_a}~{tf_b}"``.

    Returns:
        Effective number of independent hypotheses (>= 1.0).

    Formula per cluster: ``m_eff_cluster = k / (1 + (k-1) * r̄_cluster)``.
    Non-probe groups each contribute 1 independent test.

    Time: O(k²) per cluster where k = TFs per (sym, family). Space: O(k²).
    """
    # cluster_key = (sym, family); value = list of tf strings
    probe_clusters: dict[tuple[str, str], list[str]] = {}
    non_probe_count = 0

    for sym, strat_id, _ in groups:
        family, _, variant = strat_id.partition(":")
        m = _PROBE_TF_PATTERN.search(variant)
        if m is not None:
            probe_tf = m.group(1)  # e.g. "6h"
            probe_clusters.setdefault((sym, family), []).append(probe_tf)
        else:
            non_probe_count += 1

    m_eff = float(non_probe_count)
    for (sym, family), tfs in probe_clusters.items():
        k = len(tfs)
        if k <= 1:
            m_eff += float(k)
            continue
        # Collect pairwise correlations (try both orderings)
        r_vals: list[float] = []
        for i, tf_a in enumerate(tfs):
            for tf_b in tfs[i + 1 :]:
                r = diversity_corr.get(
                    f"{sym}:{family}:{tf_a}~{tf_b}",
                    diversity_corr.get(f"{sym}:{family}:{tf_b}~{tf_a}", 0.0),
                )
                r_vals.append(abs(r))
        r_bar = float(np.mean(r_vals)) if r_vals else 0.0
        m_eff += k / (1.0 + (k - 1) * r_bar)

    return max(1.0, m_eff)


def _event_results_from_fold_output(
    *,
    fold_id: int,
    fold_out: Any,
) -> pd.DataFrame:
    event_frame = getattr(fold_out.model_output, "events", pd.DataFrame()).copy()
    if event_frame.empty:
        return event_frame
    gross_pred = _expected_gross_bps(fold_out.model_output)
    q10_pred = _q10_gross_bps(fold_out.model_output)
    q90_pred = _q90_gross_bps(fold_out.model_output)
    size = min(len(event_frame), gross_pred.size)
    event_frame = event_frame.iloc[:size].reset_index(drop=True)
    event_frame["expected_gross_bps"] = gross_pred[:size]
    event_frame["q10_gross_bps"] = q10_pred[:size]
    event_frame["q90_gross_bps"] = q90_pred[:size]
    event_frame["fold_id"] = int(fold_id)
    event_frame["decision_idx"] = (
        pd.to_numeric(
            event_frame.get("entry_idx", pd.Series(0, index=event_frame.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        - 1
    )
    if "strategy_id" not in event_frame.columns:
        event_frame["strategy_id"] = (
            event_frame.get("family", pd.Series("", index=event_frame.index)).astype(str)
            + ":"
            + event_frame.get("variant", pd.Series("", index=event_frame.index)).astype(str)
        )
    if "activation_context" not in event_frame.columns:
        event_frame["activation_context"] = event_frame.get(
            "signal_cell",
            event_frame.get("entry_regime", pd.Series("all", index=event_frame.index)),
        ).astype(str)
    if "uniqueness_weight" not in event_frame.columns:
        oos_set = getattr(fold_out, "oos_set", None)
        weights = getattr(oos_set, "edge_weight", None) if oos_set is not None else None
        if weights is not None and len(weights) >= size:
            event_frame["uniqueness_weight"] = np.asarray(weights[:size], dtype=np.float64)
        else:
            event_frame["uniqueness_weight"] = np.ones(size, dtype=np.float64)
    if "gross_event_bps" not in event_frame.columns:
        oos_set = getattr(fold_out, "oos_set", None)
        y_return = getattr(oos_set, "y_return_bps", None) if oos_set is not None else None
        if y_return is not None and len(y_return) >= size:
            event_frame["gross_event_bps"] = np.asarray(y_return[:size], dtype=np.float64)
        else:
            event_frame["gross_event_bps"] = np.zeros(size, dtype=np.float64)
    event_frame["realized_side_adjusted_gross_bps"] = pd.to_numeric(
        event_frame["gross_event_bps"],
        errors="coerce",
    ).fillna(0.0)
    return event_frame


def compute_symbol_strategy_evidence(
    *,
    event_results: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    seed: int,
    registry_as_of_idx: int,
    snapshot_index: int = -1,
    probe_diversity_corr: dict[str, float] | None = None,
    xs_admission: dict[str, XsAdmissionBasis] | None = None,
) -> tuple[SymbolStrategyEvidence, ...]:
    """Compute per-source signal evidence from event-level OOS results."""
    if event_results.empty:
        return ()
    frame = event_results.copy()
    if "strategy_id" not in frame.columns:
        frame["strategy_id"] = (
            frame.get("family", pd.Series("", index=frame.index)).astype(str)
            + ":"
            + frame.get("variant", pd.Series("", index=frame.index)).astype(str)
        )
    if "activation_context" not in frame.columns:
        frame["activation_context"] = frame.get(
            "signal_cell",
            frame.get("entry_regime", pd.Series("all", index=frame.index)),
        ).astype(str)
    qualify_by_regime: bool = bool(getattr(cfg, "l1_qualify_by_regime", True))
    if not qualify_by_regime:
        frame["activation_context"] = "all"
    if "uniqueness_weight" not in frame.columns:
        frame["uniqueness_weight"] = 1.0
    frame["gross_event_bps"] = pd.to_numeric(
        frame.get("gross_event_bps", frame.get("realized_side_adjusted_gross_bps", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    frame["side"] = (
        pd.to_numeric(frame.get("side", pd.Series(1, index=frame.index)), errors="coerce").fillna(1.0).astype(int)
    )
    frame["expected_holding_bars"] = (
        pd.to_numeric(
            frame.get("expected_holding_bars", pd.Series(1, index=frame.index)),
            errors="coerce",
        )
        .fillna(1)
        .clip(lower=1)
        .astype(int)
    )
    frame["fold_id"] = (
        pd.to_numeric(frame.get("fold_id", pd.Series(0, index=frame.index)), errors="coerce").fillna(0).astype(int)
    )
    if "exit_idx" in frame.columns:
        exit_idx = pd.to_numeric(frame["exit_idx"], errors="coerce").fillna(np.inf).astype(float)
        mature = exit_idx < float(registry_as_of_idx)
        lookback_bars = getattr(cfg, "l1_evidence_lookback_bars", None)
        if lookback_bars is not None:
            min_exit_idx = float(registry_as_of_idx - int(lookback_bars))
            mature = mature & (exit_idx >= min_exit_idx)
        frame = frame.loc[mature]
    if frame.empty:
        return ()
    frame["holding_bucket"] = frame["expected_holding_bars"].map(_holding_bucket)
    baseline_mode: Literal["peer_exclusive", "absolute"] = getattr(cfg, "l1_baseline_mode", "peer_exclusive")
    frame["incremental_bps"] = _compute_incremental_bps(frame, mode=baseline_mode)
    grouped = frame.groupby(["symbol", "strategy_id", "activation_context"], sort=False)
    # 사전 일괄 벡터화: per-pair inner groupby 제거 (P2 최적화)
    _pair_fold_means: dict[tuple[str, str, str], list[float]] = {}
    _pfm_raw = (
        frame.groupby(["symbol", "strategy_id", "activation_context", "fold_id"], sort=True)["incremental_bps"]
        .mean()
        .reset_index()
    )
    for row_t in _pfm_raw.itertuples(index=False):
        key3 = (str(row_t.symbol), str(row_t.strategy_id), str(row_t.activation_context))
        _pair_fold_means.setdefault(key3, []).append(float(row_t.incremental_bps))
    evidence_list: list[SymbolStrategyEvidence] = []
    raw_p_values: list[float] = []

    t_core = time.perf_counter()
    t_prep_total = 0.0
    t_stats_total = 0.0
    t_qualify_total = 0.0
    for (symbol, strategy_id, activation_context), group in grouped:
        t_step = time.perf_counter()
        weights = pd.to_numeric(group["uniqueness_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        gross = group["gross_event_bps"].to_numpy(dtype=np.float64, copy=False)
        incremental = group["incremental_bps"].to_numpy(dtype=np.float64, copy=False)
        n_obs = int(group.shape[0])
        weight_sum = float(np.sum(weights))
        effective_n = 0.0
        if weight_sum > 0.0:
            denom = float(np.sum(np.square(weights)))
            if denom > 0.0:
                effective_n = (weight_sum * weight_sum) / denom
        mean_gross = float(np.average(gross, weights=weights)) if weight_sum > 0.0 else 0.0
        mean_incremental = float(np.average(incremental, weights=weights)) if weight_sum > 0.0 else 0.0
        fold_means = _pair_fold_means.get((str(symbol), str(strategy_id), str(activation_context)), [])
        positive_fold_ratio = (
            float(sum(1 for value in fold_means if value > 0.0) / len(fold_means)) if fold_means else 0.0
        )
        t_stat = _series_tstat(incremental)
        p_value = _one_sided_p_value(t_stat)
        boot_means = moving_block_bootstrap_mean(
            incremental.astype(np.float64, copy=False),
            group.get("decision_idx", group.get("entry_idx", pd.Series(0, index=group.index))).to_numpy(
                dtype=np.int64, copy=False
            ),
            block_bars=_resolve_block_bars_eff(cfg),
            n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 200)),
            seed=seed
            + int.from_bytes(
                sha256(f"{symbol}:{strategy_id}:{activation_context}".encode()).digest()[:4],
                byteorder="big",
            )
            % 10_000,
        )
        probability_positive = (
            float(np.mean(boot_means > 0.0)) if boot_means.size > 0 else (1.0 if mean_incremental > 0.0 else 0.0)
        )
        lcb_net = float(np.quantile(boot_means, 0.05)) if boot_means.size > 0 else mean_incremental
        block_tstat = (
            float(mean_incremental / (np.std(boot_means, ddof=1) + 1e-12))
            if boot_means.size >= 2 and float(np.std(boot_means, ddof=1)) > 0.0
            else t_stat
        )
        # XS alpha admission: substitute factor-level inputs for all 3 gates
        if xs_admission is not None:
            _xs = xs_admission.get(str(strategy_id))
            if _xs is not None:
                mean_gross = _xs.mean_bps
                mean_incremental = _xs.mean_bps
                probability_positive = _xs.probability_positive
                lcb_net = _xs.lcb_bps
                t_stat = _xs.sharpe * float(np.sqrt(max(_xs.n_bars, 1)))
                p_value = _one_sided_p_value(t_stat)
                block_tstat = t_stat
        t_prep_total += time.perf_counter() - t_step

        t_qs = time.perf_counter()

        # 1. 자유도 기반 적응적 t-critical value 계산
        df = effective_n - 1.0
        if df >= 2.0:
            alpha = float(getattr(cfg, "l1_pair_alpha", 0.05))
            power = float(getattr(cfg, "l1_pair_power", 0.80))
            t_crit = _t_ppf_cached(round((1.0 - alpha) * 1000), round(df))
            t_power = _t_ppf_cached(round(power * 1000), round(df))
        else:
            t_crit = np.inf
            t_power = 0.0

        # 2. MDES (Minimum Detectable Effect Size) 기반 유의 효과 크기(bps) 역산
        std_incremental = float(np.std(incremental, ddof=1)) if len(incremental) >= 2 else 0.0
        if effective_n > 0.0 and np.isfinite(t_crit):
            mdes_standardized = (t_crit + t_power) / np.sqrt(effective_n)
            mdes_bps = mdes_standardized * std_incremental
        else:
            mdes_bps = 0.0

        # P3: Adaptive evidence gate — relax effective_obs/folds thresholds for early snapshots
        early_snapshots = int(getattr(cfg, "l1_evidence_early_snapshots", 0))
        use_early = snapshot_index >= 0 and snapshot_index < early_snapshots and early_snapshots > 0
        min_eff_obs = float(cfg.l1_pair_min_effective_obs_early) if use_early else float(cfg.l1_pair_min_effective_obs)
        min_folds = int(cfg.l1_pair_min_folds_early) if use_early else int(cfg.l1_pair_min_folds)
        structural_reasons: list[str] = []
        if effective_n < min_eff_obs:
            structural_reasons.append("insufficient_effective_obs")
        if len(fold_means) < min_folds:
            structural_reasons.append("insufficient_folds")
        if mean_gross <= float(cfg.l1_pair_min_mean_gross_bps):
            structural_reasons.append("negative_gross_edge")
        if mean_incremental <= float(cfg.l1_pair_min_incremental_bps):
            structural_reasons.append("no_incremental_edge")

        diagnostic_flags: list[str] = []
        if t_stat < t_crit:
            diagnostic_flags.append("weak_tstat")
        mdes_mult = float(getattr(cfg, "l1_pair_mdes_multiplier", 0.5))
        if mean_incremental <= (mdes_bps * mdes_mult):
            diagnostic_flags.append("insufficient_effect_size")
        if positive_fold_ratio < float(cfg.l1_pair_min_positive_fold_ratio):
            diagnostic_flags.append("unstable_folds")
        hard_eligible = not structural_reasons
        n_target = max(float(cfg.l1_pair_min_effective_obs) * 2.0, 1.0)
        sample_scale = min(1.0, np.sqrt(effective_n / n_target)) if effective_n > 0.0 else 0.0
        quality_weight = 0.0
        if hard_eligible:
            if bool(getattr(cfg, "l1_quality_weight_enabled", True)):
                quality_weight = max(0.0, 2.0 * probability_positive - 1.0)
                quality_weight *= max(positive_fold_ratio, 0.0)
                quality_weight *= sample_scale
                _qw_floor = float(getattr(cfg, "l1_qw_floor", 0.0))
                if _qw_floor > 0.0 and quality_weight < _qw_floor:
                    quality_weight = _qw_floor
            else:
                quality_weight = 1.0
        t_stats_total += time.perf_counter() - t_qs

        t_qf = time.perf_counter()
        adverse_lcb, adverse_n, adverse_defended = compute_adverse_regime_evidence(
            group,
            cfg=cfg,
            fold_id=int(group["fold_id"].iloc[0]) if len(group) else 0,
            seed=seed,
        )
        evidence_list.append(
            SymbolStrategyEvidence(
                key=SignalSourceKey(
                    symbol=str(symbol),
                    strategy_id=str(strategy_id),
                    activation_context=str(activation_context or "all"),
                ),
                mean_gross_bps=mean_gross,
                mean_incremental_bps=mean_incremental,
                block_tstat_incremental=block_tstat,
                probability_positive=probability_positive,
                p_value=p_value,
                q_value=1.0,
                positive_fold_ratio=positive_fold_ratio,
                n_obs=n_obs,
                effective_n=effective_n,
                n_folds=len(fold_means),
                quality_weight=quality_weight,
                hard_eligible=hard_eligible,
                structural_reasons=tuple(structural_reasons),
                diagnostic_flags=tuple(diagnostic_flags),
                lcb_net_bps=lcb_net,
                adverse_regime_lcb_bps=adverse_lcb,
                adverse_regime_n_obs=adverse_n,
                adverse_regime_defended=adverse_defended,
            )
        )
        raw_p_values.append(p_value)
        t_qualify_total += time.perf_counter() - t_qf
    # C2: effective-N correction for correlated probe TF hypotheses
    _m_eff: float | None = None
    if probe_diversity_corr is not None and raw_p_values:
        _m_eff = _compute_probe_m_eff(
            groups=[(e.key.symbol, e.key.strategy_id, str(e.key.activation_context)) for e in evidence_list],
            diversity_corr=probe_diversity_corr,
        )
    q_values = _by_q_values(
        np.asarray(raw_p_values, dtype=np.float64),
        m_eff=_m_eff,
    )
    final_evidence: list[SymbolStrategyEvidence] = []
    for idx, evidence in enumerate(evidence_list):
        q_value = float(q_values[idx])
        diag_flags = list(evidence.diagnostic_flags)
        quality_weight = evidence.quality_weight
        fdr_hard = bool(getattr(cfg, "l1_fdr_hard_reject", False))
        if q_value > float(cfg.l1_pair_fdr_alpha):
            diag_flags.append("fdr_reject")
            if fdr_hard:
                # Demote to hard ineligible: zero weight and mark as structural rejection
                quality_weight = 0.0
            elif evidence.hard_eligible and bool(getattr(cfg, "l1_quality_weight_enabled", True)):
                quality_weight *= max(0.0, 1.0 - q_value)
        elif evidence.hard_eligible and bool(getattr(cfg, "l1_quality_weight_enabled", True)):
            quality_weight *= max(0.0, 1.0 - q_value)
        ev_item = SymbolStrategyEvidence(
            key=evidence.key,
            mean_gross_bps=evidence.mean_gross_bps,
            mean_incremental_bps=evidence.mean_incremental_bps,
            block_tstat_incremental=evidence.block_tstat_incremental,
            probability_positive=evidence.probability_positive,
            p_value=evidence.p_value,
            q_value=q_value,
            positive_fold_ratio=evidence.positive_fold_ratio,
            n_obs=evidence.n_obs,
            effective_n=evidence.effective_n,
            n_folds=evidence.n_folds,
            quality_weight=quality_weight,
            hard_eligible=evidence.hard_eligible,
            structural_reasons=evidence.structural_reasons,
            diagnostic_flags=tuple(diag_flags),
            lcb_net_bps=evidence.lcb_net_bps,
            adverse_regime_lcb_bps=evidence.adverse_regime_lcb_bps,
            adverse_regime_n_obs=evidence.adverse_regime_n_obs,
            adverse_regime_defended=evidence.adverse_regime_defended,
        )
        final_evidence.append(ev_item)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[L1-EVIDENCE-PAIR] %s:%s:%s -> hard_eligible=%s structural_reasons=%s diag_flags=%s "
                "eff_n=%.2f/%s gross=%.2f inc=%.2f t_stat=%.2f folds=%d/%d qw=%.4f lcb=%.4f",
                ev_item.key.symbol,
                ev_item.key.strategy_id,
                ev_item.key.activation_context,
                ev_item.hard_eligible,
                ev_item.structural_reasons,
                ev_item.diagnostic_flags,
                ev_item.effective_n,
                min_eff_obs,
                ev_item.mean_gross_bps,
                ev_item.mean_incremental_bps,
                ev_item.block_tstat_incremental,
                ev_item.n_folds,
                min_folds,
                ev_item.quality_weight,
                ev_item.lcb_net_bps,
            )
    qualified_count = sum(1 for ev in final_evidence if ev.hard_eligible and ev.quality_weight > 0.0)
    t_elapsed = time.perf_counter() - t_core
    logger.log(
        PERF,
        "[PERF] signal_evidence n_pairs=%d n_qualified=%d prep=%.4fs stats=%.4fs qualify=%.4fs took=%.4fs",
        len(evidence_list),
        qualified_count,
        t_prep_total,
        t_stats_total,
        t_qualify_total,
        t_elapsed,
    )
    # 진단: registry 공집합 경고
    if qualified_count == 0 and final_evidence:
        reasons: Counter[str] = Counter(r for ev in final_evidence for r in ev.structural_reasons)
        qw_zero = sum(1 for ev in final_evidence if ev.hard_eligible and ev.quality_weight <= 0.0)
        logger.debug(
            "[L1-EVIDENCE] as_of=%d: %d pairs, 0 qualified. structural_reasons=%s, hard_eligible_but_qw_zero=%d",
            registry_as_of_idx,
            len(final_evidence),
            dict(reasons),
            qw_zero,
        )
    if logger.isEnabledFor(logging.DEBUG):
        _log_family_admission_diag(tuple(final_evidence))
    return tuple(final_evidence)


def _log_family_admission_diag(evidence: tuple[SymbolStrategyEvidence, ...]) -> None:
    """진단: family별 admission 통계를 DEBUG 레벨로 로깅.

    Args:
        evidence: 최종 evidence 튜플.
    """
    _family_stats: dict[str, dict[str, float]] = {}
    _family_reason_counts: dict[str, Counter[str]] = {}
    for ev in evidence:
        _fam = ev.key.strategy_id.split(":")[0] if ":" in ev.key.strategy_id else ev.key.strategy_id
        _stats = _family_stats.setdefault(_fam, {"n_obs_sum": 0.0, "effective_n_sum": 0.0, "n_pairs": 0.0})
        _stats["n_obs_sum"] += ev.n_obs
        _stats["effective_n_sum"] += ev.effective_n
        _stats["n_pairs"] += 1
        _reasons = _family_reason_counts.setdefault(_fam, Counter())
        for r in ev.structural_reasons:
            _reasons[r] += 1
    for _fam, _stats in sorted(_family_stats.items(), key=lambda kv: -kv[1]["n_obs_sum"])[:10]:
        _ratio = _stats["effective_n_sum"] / _stats["n_obs_sum"] if _stats["n_obs_sum"] > 0 else 0.0
        _reasons_str = ",".join(f"{k}={v}" for k, v in _family_reason_counts[_fam].most_common(3))
        logger.debug(
            "[SYS] stage=l1_family_admission_diag family=%s n_pairs=%d n_obs_sum=%.0f"
            " effective_n_sum=%.2f eff_n_over_n_obs=%.4f top_reasons=%s",
            _fam, int(_stats["n_pairs"]), _stats["n_obs_sum"], _stats["effective_n_sum"], _ratio,
            _reasons_str if _reasons_str else "none(all_hard_eligible)",
        )


def build_qualified_signal_registry(
    *,
    evidence: tuple[SymbolStrategyEvidence, ...],
    symbols: tuple[str, ...],
    min_signals_per_symbol: int,
    registry_version: str,
    cfg: CandidateStrategyConfig | None = None,
    probe_prior_map: dict[tuple[str, str, str], float] | None = None,
    advisory_penalty: float = 1.0,
) -> QualifiedSignalRegistry:
    t_reg = time.perf_counter()
    grouped: dict[str, list[SymbolStrategyEvidence]] = defaultdict(list)
    # cfg=None → LCB gate disabled (backward compat for tests / callers without cfg)
    breakeven: float | None = float(getattr(cfg, "l1_breakeven_floor_bps", 0.0)) if cfg is not None else None
    for item in evidence:
        hard_eligible = bool(getattr(item, "hard_eligible", getattr(item, "qualified", False)))
        quality_weight = float(getattr(item, "quality_weight", getattr(item, "reliability", 0.0))) * advisory_penalty
        lcb_net_bps = float(getattr(item, "lcb_net_bps", 0.0))
        lcb_pass = breakeven is None or lcb_net_bps > breakeven
        if probe_prior_map is not None:
            _family = item.key.strategy_id.split(":")[0] if ":" in item.key.strategy_id else item.key.strategy_id
            _variant = item.key.strategy_id.split(":")[1] if ":" in item.key.strategy_id else ""
            _probe_map_key = (_family, _variant, item.key.symbol)
            _probe_floor = probe_prior_map.get(_probe_map_key, 0.0)
            if _probe_floor > 0.0 and quality_weight < _probe_floor:
                quality_weight = _probe_floor
        if hard_eligible and quality_weight > 0.0 and lcb_pass:
            grouped[item.key.symbol].append(item)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                reject_reasons = []
                if not hard_eligible:
                    reject_reasons.append("not_hard_eligible")
                if quality_weight <= 0.0:
                    reject_reasons.append("zero_quality_weight")
                if not lcb_pass:
                    reject_reasons.append(f"lcb_fail({lcb_net_bps:.4f} <= {breakeven})")
                logger.debug(
                    "[L1-REGISTRY-REJECT] %s:%s:%s -> reasons=%s",
                    item.key.symbol,
                    item.key.strategy_id,
                    item.key.activation_context,
                    ", ".join(reject_reasons),
                )
    by_symbol: dict[str, tuple[SymbolStrategyEvidence, ...]] = {}
    ready_symbols: list[str] = []
    for symbol in symbols:
        items = tuple(
            sorted(
                grouped.get(symbol, ()),
                key=lambda candidate: (
                    float(getattr(candidate, "quality_weight", getattr(candidate, "reliability", 0.0))),
                    float(getattr(candidate, "mean_incremental_bps", 0.0)),
                    float(
                        getattr(
                            candidate,
                            "block_tstat_incremental",
                            getattr(candidate, "bootstrap_tstat_incremental", 0.0),
                        )
                    ),
                ),
                reverse=True,
            )
        )
        if len(items) >= min_signals_per_symbol:
            by_symbol[symbol] = items
            ready_symbols.append(symbol)
    logger.log(
        PERF,
        "[PERF] l1_build_registry n_evidence=%d n_ready=%d n_symbols=%d took=%.4fs",
        len(evidence),
        len(ready_symbols),
        len(symbols),
        time.perf_counter() - t_reg,
    )
    return QualifiedSignalRegistry(
        by_symbol=by_symbol,
        ready_symbols=tuple(ready_symbols),
        trade_scope_count=len(symbols),
        registry_version=registry_version,
    )


def _registry_to_symbol_signals(
    registry: QualifiedSignalRegistry,
) -> dict[str, SymbolSignal]:
    """Compatibility adapter until Layer2 consumes ValidatedSignalBatch directly."""
    adapted: dict[str, SymbolSignal] = {}
    for symbol, evidence_items in registry.by_symbol.items():
        if not evidence_items:
            continue
        best = evidence_items[0]
        adapted[symbol] = SymbolSignal(
            raw_mu=float(best.mean_incremental_bps),  # gross → incremental (net) edge
            volatility=VOL_FLOOR,
            n_obs=max(round(best.effective_n), 0),
            t_stat=float(best.block_tstat_incremental),
            valid=best.quality_weight > 0.0,
            beta_btc=None,
            quality_weight=float(best.quality_weight),
        )
    return adapted


def _candidate_output_to_signal_batch(
    *,
    model_output: CandidateModelOutput,
    registry: QualifiedSignalRegistry,
    datetimes: NDArray[np.datetime64],
    symbols: tuple[str, ...],
    model_version: str,
    activation_floor_bps: float,
    cfg: CandidateStrategyConfig | None = None,
) -> ValidatedSignalBatch:
    _t_batch_total = time.perf_counter()
    frame = model_output.events.reset_index(drop=True).copy()
    logger.debug("[L2-SIGNAL] raw_events=%d", len(frame))
    if frame.empty:
        logger.warning("[L2-SIGNAL] model_output.events is empty — no predictions")
        return ValidatedSignalBatch(
            events=(),
            start_idx=0,
            end_idx=0,
            symbols=symbols,
            registry_version=registry.registry_version,
            model_version=model_version,
        )
    _t_pred = time.perf_counter()
    gross = _expected_gross_bps(model_output)
    net = _expected_net_bps(model_output)
    q10 = _q10_gross_bps(model_output)
    q10_net = _q10_net_bps(model_output)
    q90 = _q90_gross_bps(model_output)
    q90_net = _q90_net_bps(model_output)
    _t_pred_took = time.perf_counter() - _t_pred
    activation_match_regime: bool = bool(getattr(cfg, "l1_activation_match_regime", True)) if cfg is not None else True
    _t_keys = time.perf_counter()
    source_keys: set[tuple[str, str, str]] = set()
    source_keys_relaxed: set[tuple[str, str]] = set()
    if activation_match_regime:
        source_keys = {
            (item.key.symbol, item.key.strategy_id, item.key.activation_context)
            for items in registry.by_symbol.values()
            for item in items
        }
    else:
        source_keys_relaxed = {
            (item.key.symbol, item.key.strategy_id) for items in registry.by_symbol.values() for item in items
        }
    _t_keys_took = time.perf_counter() - _t_keys
    events: list[ValidatedSignalEvent] = []
    start_idx = int(frame["entry_idx"].min()) if "entry_idx" in frame.columns and not frame.empty else 0
    end_idx = int(frame["entry_idx"].max()) + 1 if "entry_idx" in frame.columns and not frame.empty else 0
    has_explicit_gross = bool(getattr(model_output, "_has_explicit_expected_gross_bps", True))
    n_raw = len(frame)
    _t_loop = time.perf_counter()
    # ── key column vectors (column-existence cascade — equiv. to _signal_source_key_from_row) ──
    sym_v = frame["symbol"].astype(str).to_numpy() if "symbol" in frame.columns else np.full(n_raw, "", dtype=object)
    if "strategy_id" in frame.columns:
        strat_v = frame["strategy_id"].astype(str).to_numpy()
    else:
        fam_s = frame["family"].astype(str) if "family" in frame.columns else pd.Series([""] * n_raw)
        var_s = frame["variant"].astype(str) if "variant" in frame.columns else pd.Series([""] * n_raw)
        strat_v = (fam_s + ":" + var_s).to_numpy()
    if not activation_match_regime:
        actx_v: np.ndarray = np.full(n_raw, "all", dtype=object)
    elif "activation_context" in frame.columns:
        actx_v = frame["activation_context"].astype(str).to_numpy()
    elif "signal_cell" in frame.columns:
        actx_v = frame["signal_cell"].astype(str).to_numpy()
    elif "entry_regime" in frame.columns:
        actx_v = frame["entry_regime"].astype(str).to_numpy()
    else:
        actx_v = np.full(n_raw, "all", dtype=object)
    actx_v = np.where(actx_v == "", "all", actx_v)  # mirrors `str(...) or "all"`
    # ── registry membership mask (composite isin — C-level) ──────────────────
    if activation_match_regime:
        _composite = pd.Series(sym_v) + "|" + pd.Series(strat_v) + "|" + pd.Series(actx_v)
        _keyset: set[str] = {f"{s}|{st}|{a}" for (s, st, a) in source_keys}
    else:
        _composite = pd.Series(sym_v) + "|" + pd.Series(strat_v)
        _keyset = {f"{s}|{st}" for (s, st) in source_keys_relaxed}
    mask_reg = _composite.isin(_keyset).to_numpy()
    n_registry_pass = int(mask_reg.sum())
    if logger.isEnabledFor(logging.DEBUG):
        _raw_strategy_v = np.array([s.split(":")[0] if ":" in s else s for s in strat_v], dtype=object)
        _raw_families, _raw_counts = np.unique(_raw_strategy_v, return_counts=True)
        _raw_family_freq = dict(zip(_raw_families.tolist(), _raw_counts.tolist(), strict=False))
        _reg_families = {
            (item.key.strategy_id.split(":")[0] if ":" in item.key.strategy_id else item.key.strategy_id)
            for items in registry.by_symbol.values()
            for item in items
        }
        _raw_family_set = set(_raw_family_freq.keys())
        _missing_families = sorted(_raw_family_set - _reg_families, key=lambda f: -_raw_family_freq[f])[:8]
        logger.debug(
            "[SYS] stage=l1_registry_overlap_diag n_raw_symbols=%d n_raw_families=%d n_registry_symbols=%d"
            " n_registry_families=%d families_never_in_registry=%s",
            len(set(sym_v.tolist())),
            len(_raw_family_set),
            len(registry.by_symbol),
            len(_reg_families),
            ",".join(f"{f}({_raw_family_freq[f]})" for f in _missing_families) if _missing_families else "none",
        )

    # ── prediction arrays padded to n_raw with cascaded fallback ─────────────
    def _pad(arr: NDArray[np.float64], fb: NDArray[np.float64]) -> NDArray[np.float64]:
        out = fb.copy()
        sz = min(arr.size, n_raw)
        out[:sz] = arr[:sz]
        return out

    g = np.zeros(n_raw, dtype=np.float64)
    g[: min(gross.size, n_raw)] = gross[: min(gross.size, n_raw)]
    n_net_p = _pad(net, g)
    q10_p = _pad(q10, g)
    q10_net_p = _pad(q10_net, q10_p)
    q90_p = _pad(q90, g)
    q90_net_p = _pad(q90_net, q90_p)
    # ── gate masks ────────────────────────────────────────────────────────────
    mask_gross = mask_reg if has_explicit_gross else np.zeros(n_raw, dtype=bool)
    n_gross_pass = int(mask_gross.sum())
    mask_thr = mask_gross & (g >= activation_floor_bps)
    n_threshold_pass = int(mask_thr.sum())
    if "entry_idx" in frame.columns:
        entry_arr = pd.to_numeric(frame["entry_idx"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        entry_arr = np.zeros(n_raw, dtype=int)
    dec_arr = entry_arr - 1
    mask_dec = mask_thr & (dec_arr >= 0) & (dec_arr < datetimes.shape[0])
    n_decision_pass = int(mask_dec.sum())
    # ── side / holding ────────────────────────────────────────────────────────
    side_raw = (
        pd.to_numeric(frame["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        if "side" in frame.columns
        else np.ones(n_raw, dtype=np.float64)
    )
    side_arr_v = np.where(side_raw >= 0, 1, -1).astype(np.int64)
    hold_raw = (
        pd.to_numeric(frame["expected_holding_bars"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        if "expected_holding_bars" in frame.columns
        else np.ones(n_raw, dtype=np.float64)
    )
    hold_arr_v = np.maximum(hold_raw.astype(np.int64), 1)
    # ── quality_weight lookup (registry flattened once) ───────────────────────
    qw_lookup: dict[tuple[str, str, str], float] = {}
    for _evs in registry.by_symbol.values():
        for _ev in _evs:
            _k3 = (_ev.key.symbol, _ev.key.strategy_id, _ev.key.activation_context)
            if _k3 not in qw_lookup:
                qw_lookup[_k3] = _ev.quality_weight
    # ── small loop over n_out survivors only ──────────────────────────────────
    for _i in np.flatnonzero(mask_dec):
        _s, _st, _a = str(sym_v[_i]), str(strat_v[_i]), str(actx_v[_i])
        _d = int(dec_arr[_i])
        events.append(
            ValidatedSignalEvent(
                decision_idx=_d,
                decision_time=datetimes[_d],
                symbol=_s,
                strategy_id=_st,
                activation_context=_a,
                side=int(side_arr_v[_i]),  # type: ignore[arg-type]
                expected_net_bps=float(n_net_p[_i]),
                expected_gross_bps=float(g[_i]),
                q10_net_bps=float(q10_net_p[_i]),
                q10_gross_bps=float(q10_p[_i]),
                q90_net_bps=float(q90_net_p[_i]),
                q90_gross_bps=float(q90_p[_i]),
                expected_holding_bars=int(hold_arr_v[_i]),
                quality_weight=qw_lookup.get((_s, _st, _a), 0.0),
                registry_version=registry.registry_version,
                model_version=model_version,
            )
        )
    _t_loop_took = time.perf_counter() - _t_loop
    logger.debug(
        "[L2-SIGNAL] gates: raw=%d registry=%d gross=%d threshold=%d decision=%d final=%d activation_floor=%.1f",
        n_raw,
        n_registry_pass,
        n_gross_pass,
        n_threshold_pass,
        n_decision_pass,
        len(events),
        activation_floor_bps,
    )
    _t_sort = time.perf_counter()
    events.sort(key=lambda item: (item.decision_idx, item.symbol, item.strategy_id, item.activation_context))
    _t_sort_took = time.perf_counter() - _t_sort
    logger.log(
        PERF,
        "[PERF] signal_batch_convert n_raw=%d n_out=%d pred=%.4fs keys=%.4fs loop=%.4fs sort=%.4fs total=%.4fs",
        n_raw,
        len(events),
        _t_pred_took,
        _t_keys_took,
        _t_loop_took,
        _t_sort_took,
        time.perf_counter() - _t_batch_total,
    )
    return ValidatedSignalBatch(
        events=tuple(events),
        start_idx=start_idx,
        end_idx=end_idx,
        symbols=symbols,
        registry_version=registry.registry_version,
        model_version=model_version,
    )


def select_outer_symbol_opportunities(
    *,
    predictions: ValidatedSignalBatch,
    registry: QualifiedSignalRegistry,
) -> ValidatedSignalBatch:
    del registry
    best_by_slot: dict[tuple[int, str], ValidatedSignalEvent] = {}
    for event in predictions.events:
        slot = (event.decision_idx, event.symbol)
        candidate = best_by_slot.get(slot)
        if candidate is None:
            best_by_slot[slot] = event
            continue
        current_score = event.expected_gross_bps / max(event.expected_holding_bars, 1) * max(event.quality_weight, 0.0)
        best_score = (
            candidate.expected_gross_bps / max(candidate.expected_holding_bars, 1) * max(candidate.quality_weight, 0.0)
        )
        if current_score > best_score or (
            np.isclose(current_score, best_score)
            and (event.strategy_id, event.activation_context) < (candidate.strategy_id, candidate.activation_context)
        ):
            best_by_slot[slot] = event
    selected = tuple(
        sorted(
            best_by_slot.values(),
            key=lambda item: (item.decision_idx, item.symbol, item.strategy_id, item.activation_context),
        )
    )
    return ValidatedSignalBatch(
        events=selected,
        start_idx=predictions.start_idx,
        end_idx=predictions.end_idx,
        symbols=tuple(dict.fromkeys(event.symbol for event in selected)),
        registry_version=predictions.registry_version,
        model_version=predictions.model_version,
    )


@dataclass(frozen=True)
class ProbeBreadthDiagnostics:
    fold_id: int
    n_events: int
    n_decisions: int
    avg_breadth_per_decision: float
    probe_gross_by_k: dict[int, float]
    probe_net_by_k: dict[int, float]
    rank_ic_all: float
    rank_ic_tstat: float
    realized_mean_all: float
    realized_median_all: float
    realized_pos_fraction_all: float
    rt_cost_bps: float
    # regime -> (n_events, gross_mean_bps, net_mean_bps, pos_fraction, rank_ic)
    regime_breakdown: dict[str, tuple[int, float, float, float, float]] = field(default_factory=dict)
    # --- Residual-alpha decomposition (multi-event bars only) ---
    # beta_edge: per-bar 횡단면 평균의 평균 (바스켓 보유=시계열 추세 premium)
    # selection_alpha: top-k 선택 - per-bar 평균 (횡단면 선택 부가가치)
    # residual_ic: IC(expected, real - per-bar 평균) (신호의 횡단면 판별력)
    beta_edge_bps: float = 0.0
    selection_alpha_bps: float = 0.0
    residual_ic: float = 0.0
    residual_ic_tstat: float = 0.0
    n_residual_events: int = 0
    # regime -> (beta_edge_bps, selection_alpha_bps, residual_ic)
    regime_residual: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # regime -> (long_fraction, long_real_mean_bps, short_real_mean_bps, n_long, n_short)
    regime_side_split: dict[str, tuple[float, float, float, int, int]] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class XsFactorSpreadDiagnostics:
    fold_id: int
    by_factor: dict[str, tuple[int, int, float, float, float, float, float, float, float, float]]


@dataclass(slots=True, frozen=True)
class XsAdmissionBasis:
    """Factor-level admission basis for XS alpha portfolio-level gate substitution. [ADR_20260703_L1_XS]"""

    mean_bps: float
    lcb_bps: float
    sharpe: float
    probability_positive: float
    n_bars: int


@dataclass(slots=True, frozen=True)
class FamilyRegimeDiagnostics:
    """Phase 0 measure-first: family x entry_regime_code gross 엣지 진단.

    docs/specs/l1-regime-diversification-tf-expansion.md 참조. 모든 값은 gross
    (비용 미차감) — Phase 2 go/no-go 판정 시 l1_breakeven_floor_bps와 비교해야 함.
    """

    fold_id: int
    by_family_regime: dict[tuple[str, int], tuple[int, int, float, float, float, float, float]]
    by_family_regime_side: dict[tuple[str, int, str], tuple[int, int, float, float, float, float, float]] | None = None


def _xs_rank_ic(
    g: pd.DataFrame,
) -> tuple[float, float]:
    score_col = "score_z"
    real_col = "realized_side_adjusted_gross_bps"
    if score_col not in g.columns or real_col not in g.columns:
        return 0.0, 0.0
    ic_list: list[float] = []
    n_total = 0
    for _d, bar in g.groupby("decision_idx", sort=True):
        scores = bar[score_col].to_numpy(dtype=np.float64)
        realized = bar[real_col].to_numpy(dtype=np.float64)
        n_bar = scores.size
        if n_bar < 3:
            continue
        with np.errstate(invalid="ignore"):
            rho = float(stats.spearmanr(scores, realized)[0])
        if np.isnan(rho):
            continue
        ic_list.append(rho)
        n_total += n_bar
    if not ic_list:
        return 0.0, 0.0
    ic = float(np.nanmean(ic_list))
    if abs(ic) >= 1.0 or n_total <= 2:
        return ic, 0.0
    ict = ic * np.sqrt((n_total - 2) / (1.0 - ic * ic))
    return ic, float(ict)


def compute_xs_factor_spread_diagnostics(
    *,
    realized_event_results: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    fold_id: int,
    seed: int = 0,
    xs_archetypes: tuple[str, ...] = ("xs_alpha",),
    xs_family_fallback: tuple[str, ...] = (
        "xs_momentum",
        "xs_flow",
        "xs_oi_skew",
    ),
    min_bars: int = 8,
) -> XsFactorSpreadDiagnostics | None:
    """Pooled factor-level spread diagnostics, scoped by archetype.

    [ADR_20260703_L1_XS][ADR_20260711_L1_POOLED_ALPHA_ADMISSION_GENERALIZATION]
    xs_archetypes generalizes the original xs_alpha-only scope to any
    archetype set (e.g. trend/ts_mom) that L0 already validated via
    universe-pooled evidence.
    """
    if realized_event_results.empty:
        return None
    df = realized_event_results
    if "archetype" in df.columns:
        xs = df[df["archetype"].astype(str).isin(xs_archetypes)]
    else:
        xs = df[df["family"].astype(str).isin(xs_family_fallback)]
    if xs.empty:
        return None
    by_factor: dict[str, tuple[int, int, float, float, float, float, float, float, float, float]] = {}
    for sid, g in xs.groupby("strategy_id", sort=True):
        bar_means = g.groupby("decision_idx")["realized_side_adjusted_gross_bps"].mean()
        spread = bar_means.to_numpy(dtype=np.float64)
        n_bars = spread.size
        if n_bars < min_bars:
            continue
        mean = float(spread.mean())
        std = float(spread.std(ddof=1)) if n_bars > 1 else 0.0
        sharpe = mean / max(std, 1e-9)
        boot = moving_block_bootstrap_mean(
            spread,
            np.arange(n_bars, dtype=np.int64),
            block_bars=_resolve_block_bars_eff(cfg),
            n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 200)),
            seed=seed + fold_id,
        )
        lcb = float(np.quantile(boot, 0.05)) if boot.size > 0 else mean
        ic, ict = _xs_rank_ic(g)
        long_frac = float((g["side"].to_numpy() > 0).mean())
        prob_positive = float(np.mean(boot > 0.0)) if boot.size > 0 else (1.0 if mean > 0.0 else 0.0)
        by_factor[sid] = (n_bars, len(g), mean, std, sharpe, lcb, ic, ict, long_frac, prob_positive)
    if not by_factor:
        return None
    return XsFactorSpreadDiagnostics(fold_id=fold_id, by_factor=by_factor)


def resolve_xs_alpha_admission(
    diag: XsFactorSpreadDiagnostics | None,
    cfg: CandidateStrategyConfig,
) -> dict[str, XsAdmissionBasis]:
    """xs_alpha 팩터의 admission 여부를 factor-level spread 진단으로 판정. [ADR_20260703_L1_XS]

    Returns:
        strategy_id -> XsAdmissionBasis. 통과한 항목만 key 존재.
    """
    if not bool(getattr(cfg, "l1_xs_alpha_admission_enabled", False)):
        return {}
    if diag is None or not diag.by_factor:
        return {}
    breakeven = float(getattr(cfg, "l1_breakeven_floor_bps", 0.0))
    min_sharpe = float(getattr(cfg, "l1_xs_admission_min_sharpe", 0.15))
    result: dict[str, XsAdmissionBasis] = {}
    for sid, (n_bars, _n_events, mean, _std, sharpe, lcb, _ic, _ict, _lf, prob_positive) in diag.by_factor.items():
        if lcb > breakeven and sharpe >= min_sharpe:
            result[sid] = XsAdmissionBasis(
                mean_bps=mean,
                lcb_bps=lcb,
                sharpe=sharpe,
                probability_positive=prob_positive,
                n_bars=n_bars,
            )
    return result


def _family_regime_cell_stats(
    g: pd.DataFrame,
    *,
    cfg: CandidateStrategyConfig,
    fold_id: int,
    seed: int,
    min_bars: int,
) -> tuple[int, int, float, float, float, float, float] | None:
    bar_means = g.groupby("decision_idx")["realized_side_adjusted_gross_bps"].mean()
    spread = bar_means.to_numpy(dtype=np.float64)
    n_bars = spread.size
    if n_bars < min_bars:
        return None
    mean = float(spread.mean())
    std = float(spread.std(ddof=1)) if n_bars > 1 else 0.0
    sharpe = mean / max(std, 1e-9)
    boot = moving_block_bootstrap_mean(
        spread,
        np.arange(n_bars, dtype=np.int64),
        block_bars=_resolve_block_bars_eff(cfg),
        n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 200)),
        seed=seed + fold_id,
    )
    lcb = float(np.quantile(boot, 0.05)) if boot.size > 0 else mean
    ic, _ict = _xs_rank_ic(g)
    return (n_bars, len(g), mean, std, sharpe, lcb, ic)


def compute_adverse_regime_evidence(
    g: pd.DataFrame,
    *,
    cfg: CandidateStrategyConfig,
    fold_id: int,
    seed: int,
    adverse_regime_codes: frozenset[int] = frozenset({1, 2}),
    min_bars: int = 8,
    undefended_lcb_floor_bps: float = 0.0,
) -> tuple[float | None, int, bool]:
    """Adverse-regime(bear/crisis) LCB 진단. [ADR_20260705_L1L2_REGIME_CONDITIONAL_WEIGHT]"""
    if "entry_regime_code" not in g.columns:
        return (None, 0, True)
    adverse_g = g[g["entry_regime_code"].isin(adverse_regime_codes)]
    n_adverse = len(adverse_g)
    if n_adverse < min_bars:
        return (None, n_adverse, True)
    stats = _family_regime_cell_stats(
        adverse_g,
        cfg=cfg,
        fold_id=fold_id,
        seed=seed,
        min_bars=min_bars,
    )
    if stats is None:
        return (None, n_adverse, True)
    lcb = stats[5]
    defended = (lcb is None) or (lcb > undefended_lcb_floor_bps)
    return (lcb, n_adverse, defended)


def compute_family_regime_edge_diagnostics(
    *,
    realized_event_results: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    fold_id: int,
    seed: int = 0,
    min_bars: int = 8,
    split_side: bool = False,
) -> FamilyRegimeDiagnostics | None:
    """Family x entry_regime_code 셀별 gross 엣지 진단 (Phase 0, measure-first).

    비추세 패밀리(dual_momentum/residual_reversion 등)가 특정 regime에서 trend 계열과
    차등 엣지를 갖는지 측정. 게이트 무영향, DEBUG 로그 전용. 산출값은 전부 gross —
    Phase 2 진행 여부는 lcb_gross_bps > cfg.l1_breakeven_floor_bps 기준으로 판단할 것
    (gross > 0만으로는 불충분, docs/specs/l1-regime-diversification-tf-expansion.md 참조).

    split_side=True이면 trend family의 bear-regime short 엣지 부재 여부를 판별하기
    위해 side(+1/-1)별로 추가 분해한 by_family_regime_side를 함께 산출한다
    (docs/specs/l1-nontrend-diversification-measure-first.md C3).
    """
    if realized_event_results.empty:
        return None
    df = realized_event_results
    if "family" not in df.columns or "entry_regime_code" not in df.columns:
        return None
    by_family_regime: dict[tuple[str, int], tuple[int, int, float, float, float, float, float]] = {}
    for (family, regime_code), g in df.groupby(["family", "entry_regime_code"], sort=True):
        stats_tuple = _family_regime_cell_stats(
            g,
            cfg=cfg,
            fold_id=fold_id,
            seed=seed,
            min_bars=min_bars,
        )
        if stats_tuple is None:
            continue
        by_family_regime[(str(family), int(regime_code))] = stats_tuple
    if not by_family_regime:
        return None

    by_family_regime_side: dict[tuple[str, int, str], tuple[int, int, float, float, float, float, float]] | None = None
    if split_side and "side" in df.columns:
        by_family_regime_side = {}
        side_numeric = pd.to_numeric(df["side"], errors="coerce")
        df_side = df.assign(_side_numeric=side_numeric)
        df_side = df_side[df_side["_side_numeric"] != 0]
        for (family, regime_code, side_val), g in df_side.groupby(
            ["family", "entry_regime_code", "_side_numeric"], sort=True
        ):
            stats_tuple = _family_regime_cell_stats(
                g,
                cfg=cfg,
                fold_id=fold_id,
                seed=seed,
                min_bars=min_bars,
            )
            if stats_tuple is None:
                continue
            side_label = "long" if side_val > 0 else "short"
            by_family_regime_side[(str(family), int(regime_code), side_label)] = stats_tuple
        if not by_family_regime_side:
            by_family_regime_side = None

    return FamilyRegimeDiagnostics(
        fold_id=fold_id,
        by_family_regime=by_family_regime,
        by_family_regime_side=by_family_regime_side,
    )


def compute_probe_breadth_diagnostics(
    *,
    merged: pd.DataFrame,
    volatility_2d: NDArray[np.float64],
    symbol_to_idx: dict[str, int],
    cfg: CandidateStrategyConfig,
    fold_id: int,
    seed: int = 0,
    regime_code_1d: NDArray[np.int8] | None = None,
    regime_names: tuple[str, ...] = ("bull", "bear", "crisis"),
) -> ProbeBreadthDiagnostics | None:
    if merged.empty:
        return None
    rt_cost = float(getattr(cfg, "expected_cost_bps", 0.0))
    exp = merged["expected_gross_bps"].to_numpy(dtype=np.float64)
    real = merged["realized_side_adjusted_gross_bps"].fillna(0.0).to_numpy(dtype=np.float64)
    n = len(exp)

    di = merged["decision_idx"].to_numpy(dtype=np.int64)
    symbols = merged["symbol"].to_numpy(dtype=str)
    side_raw = merged["side"].to_numpy(dtype=np.int64) if "side" in merged.columns else np.ones(n, dtype=np.int64)
    side_norm = np.where(side_raw >= 0, np.int64(1), np.int64(-1))
    if "quality_weight" in merged.columns:
        qw = merged["quality_weight"].to_numpy(dtype=np.float64)
    else:
        qw = np.ones(n, dtype=np.float64)

    n_decisions = int(merged["decision_idx"].nunique()) if "decision_idx" in merged.columns else 0

    min_obs = 3
    if n >= min_obs:
        with np.errstate(invalid="ignore"):
            rho = float(stats.spearmanr(exp, real)[0])
        if np.isnan(rho):
            rho = 0.0
        fisher_z = rho * np.sqrt((n - 2) / (1.0 - rho * rho)) if abs(rho) < 1.0 and n > 2 else 0.0
    else:
        rho = 0.0
        fisher_z = 0.0

    realized_mean = float(np.mean(real))
    realized_median = float(np.median(real))
    realized_pos_frac = float(np.mean(real > 0.0))

    def _risk_topk_idx(rows: NDArray[np.int64], k: int) -> list[int]:
        """rows(전역 인덱스) 중 risk=|exp|*qw/vol 상위 k의 전역 인덱스."""
        scored: list[tuple[float, int]] = []
        for gi in rows.tolist():
            sidx = symbol_to_idx.get(str(symbols[gi]))
            d_i = int(di[gi])
            if sidx is None or d_i < 0 or d_i >= volatility_2d.shape[0]:
                continue
            denom = max(float(volatility_2d[d_i, int(sidx)]), float(VOL_FLOOR))
            scored.append((abs(float(exp[gi])) * max(float(qw[gi]), 0.0) / denom, gi))
        scored.sort(reverse=True)
        return [gi for _, gi in scored[:k]]

    def _residual_decompose(sel_mask: NDArray[np.bool_], topk: int = 3) -> tuple[float, float, float, int]:
        """multi-event bar만 사용해 beta/selection_alpha/residual_ic 분해."""
        all_idx = np.flatnonzero(sel_mask)
        if all_idx.size == 0:
            return 0.0, 0.0, 0.0, 0
        order = np.argsort(di[all_idx], kind="stable")
        all_idx = all_idx[order]
        bar_vals = di[all_idx]
        beta_list: list[float] = []
        alpha_list: list[float] = []
        res_vals: list[float] = []
        res_exp: list[float] = []
        start = 0
        ntot = all_idx.size
        for end in range(1, ntot + 1):
            if end < ntot and bar_vals[end] == bar_vals[start]:
                continue
            bar_rows = all_idx[start:end]
            start = end
            if bar_rows.size < 2:  # 횡단면 없음 → residual 정의 불가
                continue
            bar_mean = float(np.mean(real[bar_rows]))
            beta_list.append(bar_mean)
            top_idx = _risk_topk_idx(bar_rows, topk)
            if top_idx:
                alpha_list.append(float(np.mean(real[np.asarray(top_idx)])) - bar_mean)
            for gi in bar_rows.tolist():
                res_vals.append(float(real[gi]) - bar_mean)
                res_exp.append(float(exp[gi]))
        beta_edge = float(np.mean(beta_list)) if beta_list else 0.0
        sel_alpha = float(np.mean(alpha_list)) if alpha_list else 0.0
        nres = len(res_vals)
        if nres >= min_obs:
            with np.errstate(invalid="ignore"):
                r_ic = float(stats.spearmanr(np.asarray(res_exp), np.asarray(res_vals))[0])
            if np.isnan(r_ic):
                r_ic = 0.0
        else:
            r_ic = 0.0
        return beta_edge, sel_alpha, r_ic, nres

    beta_edge_bps, selection_alpha_bps, residual_ic, n_residual = _residual_decompose(np.ones(n, dtype=bool))
    residual_ic_tstat = (
        residual_ic * np.sqrt((n_residual - 2) / (1.0 - residual_ic * residual_ic))
        if abs(residual_ic) < 1.0 and n_residual > 2
        else 0.0
    )

    k_values = (3, 10, 20, -1)
    probe_gross: dict[int, float] = {}
    probe_net: dict[int, float] = {}

    for k in k_values:
        selected_real_list: list[float] = []
        for d_idx_val in sorted(merged["decision_idx"].unique()):
            mask = di == d_idx_val
            sub_exp = exp[mask]
            sub_real = real[mask]
            sub_qw = qw[mask]
            sub_symbols = symbols[mask]
            sub_len = len(sub_exp)
            if sub_len == 0:
                continue
            risk_scores: list[tuple[float, int]] = []
            for ri in range(sub_len):
                sidx = symbol_to_idx.get(str(sub_symbols[ri]))
                if sidx is None:
                    continue
                decision_idx_i = int(d_idx_val)
                if decision_idx_i < 0 or decision_idx_i >= volatility_2d.shape[0]:
                    continue
                vol = float(volatility_2d[int(decision_idx_i), int(sidx)])
                denom = max(vol, float(VOL_FLOOR))
                risk_scores.append((abs(float(sub_exp[ri])) * max(float(sub_qw[ri]), 0.0) / denom, ri))
            if not risk_scores:
                continue
            risk_scores.sort(reverse=True)
            n_take = k if k != -1 else sub_len
            take_idx = [ri for _, ri in risk_scores[:n_take]]
            selected = sub_real[np.asarray(take_idx, dtype=np.int64)]
            if selected.size > 0:
                selected_real_list.append(float(np.mean(selected)))
        if selected_real_list:
            probe_gross[k] = float(np.mean(selected_real_list))
            probe_net[k] = probe_gross[k] - rt_cost
        else:
            probe_gross[k] = 0.0
            probe_net[k] = 0.0 - rt_cost

    avg_breadth = float(n / max(n_decisions, 1))

    # --- Regime decomposition ---
    # 우선순위: 시장 regime code_1d(decision_idx 매핑) > entry_regime 컬럼.
    regime_breakdown: dict[str, tuple[int, float, float, float, float]] = {}
    regime_residual: dict[str, tuple[float, float, float]] = {}
    regime_side_split: dict[str, tuple[float, float, float, int, int]] = {}
    regimes: NDArray[Any] | None = None
    if regime_code_1d is not None and len(regime_code_1d) > 0:
        t_max = len(regime_code_1d)
        codes = np.clip(di, 0, t_max - 1)
        name_arr = np.asarray(regime_names, dtype=object)
        mapped = np.asarray(regime_code_1d, dtype=np.int64)[codes]
        regimes = name_arr[np.clip(mapped, 0, len(name_arr) - 1)]
    else:
        regime_col = None
        for cand in ("entry_regime", "activation_context"):
            if cand in merged.columns:
                regime_col = cand
                break
        if regime_col is not None:
            regimes = merged[regime_col].astype(str).to_numpy()
    if regimes is not None:
        for rname in sorted(set(regimes.tolist())):
            rmask = regimes == rname
            r_real = real[rmask]
            r_exp = exp[rmask]
            rn = int(r_real.size)
            if rn == 0:
                continue
            r_gross = float(np.mean(r_real))
            r_pos = float(np.mean(r_real > 0.0))
            if rn >= min_obs:
                with np.errstate(invalid="ignore"):
                    r_ic = float(stats.spearmanr(r_exp, r_real)[0])
                if np.isnan(r_ic):
                    r_ic = 0.0
            else:
                r_ic = 0.0
            regime_breakdown[rname] = (rn, r_gross, r_gross - rt_cost, r_pos, r_ic)
            r_beta, r_alpha, r_res_ic, _ = _residual_decompose(rmask)
            regime_residual[rname] = (r_beta, r_alpha, r_res_ic)

            r_side_s = side_norm[rmask]
            r_real_s = real[rmask]
            long_mask_s = r_side_s > 0
            short_mask_s = ~long_mask_s
            n_long_s = int(long_mask_s.sum())
            n_short_s = int(short_mask_s.sum())
            total_side = n_long_s + n_short_s
            if total_side > 0:
                long_frac = n_long_s / total_side
                long_mean = float(r_real_s[long_mask_s].mean()) if n_long_s > 0 else 0.0
                short_mean = float(r_real_s[short_mask_s].mean()) if n_short_s > 0 else 0.0
                regime_side_split[rname] = (long_frac, long_mean, short_mean, n_long_s, n_short_s)

    return ProbeBreadthDiagnostics(
        fold_id=fold_id,
        n_events=n,
        n_decisions=n_decisions,
        avg_breadth_per_decision=avg_breadth,
        probe_gross_by_k=probe_gross,
        probe_net_by_k=probe_net,
        rank_ic_all=rho,
        rank_ic_tstat=fisher_z,
        realized_mean_all=realized_mean,
        realized_median_all=realized_median,
        realized_pos_fraction_all=realized_pos_frac,
        rt_cost_bps=rt_cost,
        regime_breakdown=regime_breakdown,
        beta_edge_bps=beta_edge_bps,
        selection_alpha_bps=selection_alpha_bps,
        residual_ic=residual_ic,
        residual_ic_tstat=float(residual_ic_tstat),
        n_residual_events=n_residual,
        regime_residual=regime_residual,
        regime_side_split=regime_side_split,
    )


def _l1_probe_diag_enabled() -> bool:
    return os.environ.get("L1_PROBE_DIAG", "") not in ("", "0", "false", "False")


def _l1_xs_spread_diag_enabled() -> bool:
    return os.environ.get("L1_XS_SPREAD_DIAG", "") not in ("", "0", "false", "False")


def _l1_family_regime_diag_enabled() -> bool:
    return os.environ.get("L1_FAMILY_REGIME_DIAG", "") not in ("", "0", "false", "False")


def _format_probe_diag(diag: ProbeBreadthDiagnostics) -> str:
    parts = [
        f"fold={diag.fold_id}",
        f"n_events={diag.n_events}",
        f"n_decisions={diag.n_decisions}",
        f"breadth={diag.avg_breadth_per_decision:.1f}",
        f"gross_k3={diag.probe_gross_by_k.get(3, 0):.2f}",
        f"net_k3={diag.probe_net_by_k.get(3, 0):.2f}",
        f"gross_all={diag.probe_gross_by_k.get(-1, 0):.2f}",
        f"rank_ic={diag.rank_ic_all:.4f}",
        f"ic_t={diag.rank_ic_tstat:.2f}",
        f"real_mean={diag.realized_mean_all:.2f}",
        f"pos_frac={diag.realized_pos_fraction_all:.2%}",
        f"rt_cost={diag.rt_cost_bps:.1f}",
        f"beta_edge={diag.beta_edge_bps:.1f}",
        f"sel_alpha={diag.selection_alpha_bps:+.1f}",
        f"res_ic={diag.residual_ic:+.4f}",
        f"res_ic_t={diag.residual_ic_tstat:+.2f}",
    ]
    for rname, (rn, _rg, rnet, rpos, ric) in diag.regime_breakdown.items():
        rr = diag.regime_residual.get(rname)
        rr_str = f"/beta{rr[0]:.1f}/alpha{rr[1]:+.1f}/resic{rr[2]:+.3f}" if rr is not None else ""
        parts.append(f"REG[{rname}]=n{rn}/net{rnet:.1f}/pos{rpos:.0%}/ic{ric:+.3f}{rr_str}")
        ss = diag.regime_side_split.get(rname)
        if ss is not None:
            lf, lr, sr, nl, ns = ss
            parts.append(f"SIDE[{rname}]=long{lf:.0%}/lr{lr:+.1f}/sr{sr:+.1f}/nl{nl}/ns{ns}")
    return " | ".join(parts)


def _format_xs_spread_diag(diag: XsFactorSpreadDiagnostics) -> str:
    parts: list[str] = []
    for sid, (nbars, _nevents, _mean, _std, sharpe, lcb, ic, ict, lf, pp) in diag.by_factor.items():
        parts.append(f"XS[{sid}]=sh{sharpe:+.2f}/lcb{lcb:+.1f}/ic{ic:+.3f}/t{ict:+.2f}/n{nbars}/lf{lf:.0%}/pp{pp:.2f}")
    return " | ".join(parts) if parts else ""


def _format_family_regime_diag(diag: FamilyRegimeDiagnostics) -> str:
    parts: list[str] = []
    for (family, regime_code), (nbars, _nevents, mean, _std, sharpe, lcb, ic) in diag.by_family_regime.items():
        parts.append(f"{family}@R{regime_code}=gross{mean:+.1f}/sh{sharpe:+.2f}/lcb{lcb:+.1f}/ic{ic:+.3f}/n{nbars}")
    return " | ".join(parts) if parts else ""


def _format_family_regime_side_diag(diag: FamilyRegimeDiagnostics) -> str:
    if diag.by_family_regime_side is None:
        return ""
    parts: list[str] = []
    for (family, regime_code, side_label), (
        nbars,
        _nevents,
        mean,
        _std,
        sharpe,
        lcb,
        ic,
    ) in diag.by_family_regime_side.items():
        parts.append(
            f"{family}@R{regime_code}/{side_label}=gross{mean:+.1f}/sh{sharpe:+.2f}/lcb{lcb:+.1f}/ic{ic:+.3f}/n{nbars}"
        )
    return " | ".join(parts) if parts else ""


def evaluate_outer_signal_opportunities(
    *,
    opportunities: ValidatedSignalBatch,
    realized_event_results: pd.DataFrame,
    volatility_2d: NDArray[np.float64],
    aligned_symbols: tuple[str, ...],
    fold: WFFold,
    fold_id: int,
    cfg: CandidateStrategyConfig,
    seed: int,
    regime_code_1d: NDArray[np.int8] | None = None,
) -> Layer1FoldReadiness:
    opp_frame = _batch_to_frame(opportunities)
    if opp_frame.empty:
        logger.debug(
            "[DATA] stage=l1_nested_opportunity_diag fold=%d locus=registry_empty"
            " registry_ready=0 n_predictions=0 n_realized=0",
            fold_id,
        )
        return Layer1FoldReadiness(
            fold_id=fold_id,
            registry_source_end_idx=fold.fit_end,
            outer_oos_start_idx=fold.oos_start,
            outer_oos_end_idx=fold.oos_end,
            ready_symbols=(),
            matched_event_count=0,
            unmatched_event_count=0,
            realized_match_ratio=0.0,
            unique_decision_count=0,
            prediction_unique_count=0,
            opportunity_ic=None,
            opportunity_ic_tstat=0.0,
            probe_bps=0.0,
            probe_lcb_bps=0.0,
            probe_series_bps=(),
            effective_symbol_count=0.0,
            passed=False,
            blockers=("empty_opportunities:registry_empty",),
        )
    merged, true_unmatched, label_drift = align_outer_opportunities_with_realized(
        opportunities=opportunities,
        realized_event_results=realized_event_results,
        activation_match_regime=bool(getattr(cfg, "l1_activation_match_regime", True)),
    )
    unmatched_count = true_unmatched
    dropped_by_maturity = 0
    if "exit_idx" in merged.columns:
        before_maturity = len(merged)
        merged = merged.loc[
            pd.to_numeric(merged["exit_idx"], errors="coerce").fillna(fold.oos_end - 1).astype(int) < fold.oos_end
        ].copy()
        dropped_by_maturity = before_maturity - len(merged)
    if merged.empty:
        n_predictions = len(opportunities.events) if opportunities.events else 0
        n_realized = len(realized_event_results) if realized_event_results is not None else 0
        logger.debug(
            "[DATA] stage=l1_nested_opportunity_diag fold=%d locus=prediction_unmatched"
            " registry_ready=%d n_predictions=%d n_realized=%d",
            fold_id, len(opportunities.symbols), n_predictions, n_realized,
        )
        return Layer1FoldReadiness(
            fold_id=fold_id,
            registry_source_end_idx=fold.fit_end,
            outer_oos_start_idx=fold.oos_start,
            outer_oos_end_idx=fold.oos_end,
            ready_symbols=(),
            matched_event_count=0,
            unmatched_event_count=unmatched_count,
            realized_match_ratio=0.0,
            unique_decision_count=0,
            prediction_unique_count=0,
            opportunity_ic=None,
            opportunity_ic_tstat=0.0,
            probe_bps=0.0,
            probe_lcb_bps=0.0,
            probe_series_bps=(),
            effective_symbol_count=0.0,
            passed=False,
            blockers=("empty_opportunities:prediction_unmatched",),
            dropped_by_maturity_count=dropped_by_maturity,
            label_drift_unmatched_count=label_drift,
        )
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(aligned_symbols)}
    probe_series: list[float] = []
    probe_mode: str = str(getattr(cfg, "l1_opp_ic_mode", "cross_section"))
    probe_metric: str = str(getattr(cfg, "l1_probe_metric", "breadth"))

    if probe_mode == "time_series":
        for _symbol, sym_group in merged.groupby("symbol", sort=True):
            sym_group = sym_group.drop_duplicates(subset=["decision_idx"], keep="first")
            real_ts = sym_group["realized_side_adjusted_gross_bps"].fillna(0.0).to_numpy(dtype=np.float64)
            risk_scores_ts: list[tuple[float, int]] = []
            for row_i, row in enumerate(sym_group.itertuples(index=False)):
                symbol_idx = symbol_to_idx.get(str(row.symbol))
                d_idx = int(getattr(row, "decision_idx", 0))
                if symbol_idx is None or d_idx < 0 or d_idx >= volatility_2d.shape[0]:
                    continue
                vol = (
                    float(volatility_2d[d_idx, symbol_idx]) if volatility_2d.ndim == 2 else float(volatility_2d[d_idx])
                )
                denom = max(vol, VOL_FLOOR)
                risk_scores_ts.append(
                    (abs(float(row.expected_gross_bps)) * max(float(row.quality_weight), 0.0) / denom, row_i)
                )
            if risk_scores_ts:
                if probe_metric == "breadth":
                    selected_real = real_ts
                else:
                    risk_scores_ts.sort(reverse=True)
                    selected_idx_arr = [ri for _, ri in risk_scores_ts[: int(cfg.l1_probe_top_k)]]
                    selected_real = real_ts[np.asarray(selected_idx_arr, dtype=np.int64)]
                if selected_real.size > 0:
                    probe_series.append(float(np.mean(selected_real)))
    else:
        for decision_idx, group in merged.groupby("decision_idx", sort=True):
            group = group.drop_duplicates(subset=["symbol"], keep="first")
            if group.shape[0] < int(cfg.l1_min_cross_section):
                continue
            real = group["realized_side_adjusted_gross_bps"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            if probe_metric == "breadth":
                if real.size > 0:
                    probe_series.append(float(np.mean(real)))
                continue
            risk_scores: list[tuple[float, int]] = []
            for row_idx, row in enumerate(group.itertuples(index=False)):
                symbol_idx = symbol_to_idx.get(str(row.symbol))
                if symbol_idx is None or decision_idx < 0 or decision_idx >= volatility_2d.shape[0]:
                    continue
                vol = float(volatility_2d[int(decision_idx), symbol_idx])
                denom = max(vol, VOL_FLOOR)
                risk_scores.append(
                    (abs(float(row.expected_gross_bps)) * max(float(row.quality_weight), 0.0) / denom, row_idx)
                )
            if risk_scores:
                risk_scores.sort(reverse=True)
                selected_idx = [row_idx for _, row_idx in risk_scores[: int(cfg.l1_probe_top_k)]]
                selected_real = real[np.asarray(selected_idx, dtype=np.int64)]
                if selected_real.size > 0:
                    probe_series.append(float(np.mean(selected_real)))
    ready_symbols = tuple(sorted(str(symbol) for symbol in merged["symbol"].dropna().unique()))
    probe_gross_edge = float(np.mean(probe_series)) if probe_series else 0.0
    probe_boot = moving_block_bootstrap_mean(
        np.asarray(probe_series, dtype=np.float64),
        np.arange(len(probe_series), dtype=np.int64),
        block_bars=_resolve_block_bars_eff(cfg),
        n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 200)),
        seed=seed + fold_id,
    )
    probe_lcb = float(np.quantile(probe_boot, 0.05)) if probe_boot.size > 0 else probe_gross_edge
    matched_event_count = int(merged.shape[0])
    unique_decision_count = int(merged["decision_idx"].nunique()) if "decision_idx" in merged.columns else 0
    realized_match_ratio = float(matched_event_count / max(1, matched_event_count + unmatched_count))
    effective_symbol_count = float(len(ready_symbols))
    blockers: list[str] = []
    fold_min_ready_symbols = max(1, min(int(cfg.l1_min_sym_count), int(cfg.l1_min_cross_section)))
    if len(ready_symbols) < fold_min_ready_symbols:
        blockers.append("insufficient_ready_symbols")
    if realized_match_ratio < float(getattr(cfg, "l1_min_realized_match_ratio", 1.0)):
        blockers.append("insufficient_realized_match_ratio")
    if matched_event_count < int(getattr(cfg, "l1_min_matched_events_per_fold", 1)):
        blockers.append("insufficient_matched_events")
    l1_min_fold_probe = float(getattr(cfg, "l1_min_fold_probe_bps", 0.0))
    if probe_gross_edge <= l1_min_fold_probe:
        blockers.append("non_positive_gross_edge")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[SYS] stage=l1_per_fold_diag fold=%d blockers=%s matched=%d true_unmatched=%d"
            " label_drift=%d ready_symbols=%d match_ratio=%.3f probe_gross_edge=%.3f",
            fold_id,
            ",".join(blockers) if blockers else "none",
            matched_event_count,
            unmatched_count,
            label_drift,
            len(ready_symbols),
            realized_match_ratio,
            probe_gross_edge,
        )
    rank_ic_all_val = 0.0
    rank_ic_tstat_val = 0.0
    if _l1_probe_diag_enabled() and logger.isEnabledFor(logging.DEBUG):
        diag = compute_probe_breadth_diagnostics(
            merged=merged,
            volatility_2d=volatility_2d,
            symbol_to_idx=symbol_to_idx,
            cfg=cfg,
            fold_id=fold_id,
            seed=seed,
            regime_code_1d=regime_code_1d,
        )
        if diag is not None:
            logger.debug("[L1-PROBE-DIAG] %s", _format_probe_diag(diag))
            rank_ic_all_val = diag.rank_ic_all
            rank_ic_tstat_val = diag.rank_ic_tstat
    if _l1_xs_spread_diag_enabled() and logger.isEnabledFor(logging.DEBUG):
        xs_diag = compute_xs_factor_spread_diagnostics(
            realized_event_results=realized_event_results,
            cfg=cfg,
            fold_id=fold_id,
            seed=seed,
        )
        if xs_diag is not None:
            logger.debug("[L1-XS-SPREAD-DIAG] %s", _format_xs_spread_diag(xs_diag))
    if _l1_family_regime_diag_enabled() and logger.isEnabledFor(logging.DEBUG):
        family_regime_diag = compute_family_regime_edge_diagnostics(
            realized_event_results=realized_event_results,
            cfg=cfg,
            fold_id=fold_id,
            seed=seed,
            split_side=True,
        )
        if family_regime_diag is not None:
            logger.debug("[L1-FAMILY-REGIME-DIAG] %s", _format_family_regime_diag(family_regime_diag))
            side_diag_str = _format_family_regime_side_diag(family_regime_diag)
            if side_diag_str:
                logger.debug("[L1-FAMILY-SIDE-DIAG] %s", side_diag_str)
    return Layer1FoldReadiness(
        fold_id=fold_id,
        registry_source_end_idx=fold.fit_end,
        outer_oos_start_idx=fold.oos_start,
        outer_oos_end_idx=fold.oos_end,
        ready_symbols=ready_symbols,
        matched_event_count=matched_event_count,
        unmatched_event_count=unmatched_count,
        realized_match_ratio=realized_match_ratio,
        unique_decision_count=unique_decision_count,
        prediction_unique_count=0,
        opportunity_ic=None,
        opportunity_ic_tstat=0.0,
        probe_bps=probe_gross_edge,
        probe_lcb_bps=probe_lcb,
        probe_series_bps=tuple(probe_series),
        effective_symbol_count=effective_symbol_count,
        passed=not blockers,
        blockers=tuple(blockers),
        dropped_by_maturity_count=dropped_by_maturity,
        rank_ic_all=rank_ic_all_val,
        rank_ic_tstat=rank_ic_tstat_val,
        label_drift_unmatched_count=label_drift,
    )


def _compute_effective_sym_n(
    fold_reports: tuple[Layer1FoldReadiness, ...],
) -> float:
    """
    HHI 기반 실질 분산 심볼 수 계산.
    각 fold의 ready_symbols를 통합 후 HHI -> 1/HHI.

    Returns:
        float: Effective N (완전집중=1.0, 완전분산=심볼수)
    """
    symbol_counts: Counter[str] = Counter()
    for report in fold_reports:
        symbol_counts.update(report.ready_symbols)
    if not symbol_counts:
        return 0.0
    total = sum(symbol_counts.values())
    hhi = sum((count / total) ** 2 for count in symbol_counts.values())
    return 1.0 / hhi if hhi > 0 else 0.0


def _compute_pooled_probe_lcb(
    fold_reports: tuple[Layer1FoldReadiness, ...],
    cfg: CandidateStrategyConfig,
    seed: int,
) -> float:
    """
    passed fold의 probe_series_bps를 pool하여 단일 bootstrap LCB 계산.

    Returns:
        float: 5th percentile bootstrap mean (bps)
    """
    pooled: list[float] = []
    for r in fold_reports:
        if r.passed:
            pooled.extend(r.probe_series_bps)
    if not pooled:
        return -float("inf")
    series = np.asarray(pooled, dtype=np.float64)
    boot = moving_block_bootstrap_mean(
        series,
        np.arange(len(series), dtype=np.int64),
        block_bars=_resolve_block_bars_eff(cfg),
        n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 200)),
        seed=seed,
    )
    return float(np.quantile(boot, 0.05)) if boot.size > 0 else float(np.mean(series))


def _resolve_block_bars_eff(cfg: CandidateStrategyConfig) -> int:
    base = int(getattr(cfg, "l1_bootstrap_block_bars", 6))
    holding_bars = int(getattr(cfg, "max_holding_bars", 1))
    return max(base, 2 * holding_bars)


def _wilson_lower_bound(successes: int, n: int, confidence: float = 0.90) -> float:
    if n <= 0:
        return 0.0
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p_hat = successes / n
    denom = 1.0 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    margin = z * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
    return max(0.0, (center - margin) / denom)  # type: ignore[no-any-return]


def evaluate_layer1_readiness(
    *,
    fold_reports: tuple[Layer1FoldReadiness, ...],
    fold_cov: float,
    trade_scope_count: int,
    cfg: CandidateStrategyConfig,
    seed: int = 0,
) -> Layer1GateReport:
    t_gate = time.perf_counter()
    effective_symbol_count = 0.0
    probe_series: list[float] = []
    probe_lcbs: list[float] = []
    ready_fold_count = 0
    total_matched = 0
    total_true_unmatched = 0
    for report in fold_reports:
        if report.passed:
            ready_fold_count += 1
        effective_symbol_count = max(effective_symbol_count, report.effective_symbol_count)
        total_matched += report.matched_event_count
        total_true_unmatched += report.unmatched_event_count
        probe_series.extend([value for value in report.probe_series_bps if np.isfinite(value)])
        probe_lcbs.append(report.probe_lcb_bps)
    fold_ratio = float(ready_fold_count / len(fold_reports)) if fold_reports else 0.0
    probe_bps = float(np.mean(probe_series)) if probe_series else 0.0

    # [LIMIT-02] Pooled Wilson LCB for match_ratio
    match_ratio = _wilson_lower_bound(total_matched, total_matched + total_true_unmatched, confidence=0.90)

    if bool(getattr(cfg, "l1_probe_lcb_pooled", True)):
        probe_lcb = _compute_pooled_probe_lcb(fold_reports, cfg, seed=seed)
    else:
        probe_lcb = float(np.mean(probe_lcbs)) if probe_lcbs else probe_bps

    sym_count_mode = str(getattr(cfg, "l1_sym_count_mode", "effective_n"))
    if sym_count_mode == "effective_n":
        effective_sym_metric = _compute_effective_sym_n(fold_reports)
        sym_threshold = float(getattr(cfg, "l1_min_effective_sym_n", 3.0))
    else:
        effective_sym_metric = effective_symbol_count
        sym_threshold = max(float(cfg.l1_min_sym_count), float(cfg.l1_min_sym_ratio) * max(1, trade_scope_count))

    # IC pooled diagnostics — DEBUG 모니터링 전용 (하드 게이트 보류, spec §Track A)
    if logger.isEnabledFor(logging.DEBUG):
        ic_vals: list[float] = []
        ic_tstats: list[float] = []
        for report in fold_reports:
            ic_vals.append(report.rank_ic_all)
            ic_tstats.append(report.rank_ic_tstat)
        valid_tstats = [t for t in ic_tstats if np.isfinite(t) and t != 0.0]
        pooled_ic_tstat = float(np.sum(valid_tstats) / np.sqrt(max(len(valid_tstats), 1))) if valid_tstats else 0.0
        valid_ics = [v for v in ic_vals if np.isfinite(v)]
        ic_sign_ratio = float(np.mean([1.0 if v > 0.0 else 0.0 for v in valid_ics])) if valid_ics else 0.0
        logger.debug(
            "[L1-IC-DIAG] pooled_ic_tstat=%.3f ic_sign_ratio=%.3f n_folds=%d",
            pooled_ic_tstat,
            ic_sign_ratio,
            len(fold_reports),
        )

    # [LIMIT-02, LIMIT-03] Structural checks (blocking) vs advisory checks (non-blocking)
    structural_specs = (
        ("fold_cov", fold_cov, float(getattr(cfg, "l1_min_fold_cov", 0.8)), "ge"),
        ("sym_count", effective_sym_metric, sym_threshold, "ge"),
        ("probe_lcb_bps", probe_lcb, max(float(cfg.l1_min_probe_bps), float(cfg.l1_breakeven_floor_bps)), "gt"),
    )
    advisory_specs = (
        ("match_ratio", match_ratio, float(getattr(cfg, "l1_min_realized_match_ratio", 0.90)), "ge"),
        ("fold_ratio", fold_ratio, float(cfg.l1_min_fold_ratio), "ge"),
    )

    def _build_check(key: str, value: float, threshold: float, comparator: str, *, blocking: bool) -> Layer1GateCheck:
        finite_value = np.isfinite(value)
        passed = finite_value and (value >= threshold if comparator == "ge" else value > threshold)
        blocker_: str | None = None if passed else f"{value:.3f}"
        comparator_lit = cast(Literal["ge", "gt"], comparator)
        return Layer1GateCheck(
            key=key, value=float(value), threshold=float(threshold),
            comparator=comparator_lit, passed=passed, blocker=blocker_,
            blocking=blocking,
        )

    checks: list[Layer1GateCheck] = []
    blockers: list[str] = []
    advisory_checks: list[Layer1GateCheck] = []

    for key, value, threshold, comparator in structural_specs:
        ck = _build_check(key, value, threshold, comparator, blocking=True)
        checks.append(ck)
        if ck.blocker is not None:
            blockers.append(f"{key}:{ck.blocker}")

    for key, value, threshold, comparator in advisory_specs:
        ck = _build_check(key, value, threshold, comparator, blocking=False)
        advisory_checks.append(ck)
        if ck.blocker is not None:
            blockers.append(f"{key}:{ck.blocker}")

    structural_passed = all(ck.passed for ck in checks)
    advisory_passed = all(ck.passed for ck in advisory_checks)

    logger.log(
        PERF,
        "[PERF] l1_gate_eval n_folds=%d n_passed=%d structural=%s advisory=%s took=%.4fs",
        len(fold_reports),
        ready_fold_count,
        structural_passed,
        advisory_passed,
        time.perf_counter() - t_gate,
    )
    return Layer1GateReport(
        checks=tuple(checks),
        passed=structural_passed and advisory_passed,
        blockers=tuple(blockers),
        structural_passed=structural_passed,
        advisory_checks=tuple(advisory_checks),
    )


class _Layer1ModelCore:
    """Lightweight container for pre-fit model output (registry excluded)."""

    __slots__ = ("baseline_by_key", "config_hash", "feature_schema", "l1_fit_end_idx", "model", "model_version")

    def __init__(
        self,
        feature_schema: Any,
        model: Any,
        baseline_by_key: dict[MatchedBaselineKey, float],
        l1_fit_end_idx: int,
        model_version: str,
        config_hash: str,
    ) -> None:
        self.feature_schema = feature_schema
        self.model = model
        self.baseline_by_key = baseline_by_key
        self.l1_fit_end_idx = l1_fit_end_idx
        self.model_version = model_version
        self.config_hash = config_hash


def prefit_layer1_model(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    fit_start_idx: int,
    fit_end_idx: int,
    cfg: CandidateStrategyConfig,
) -> _Layer1ModelCore:
    """Fit the L1 model core (schema + ensemble + baseline) — registry-independent.

    This function contains the expensive work (~20s) and is safe to run
    in a background process while evidence IPC is still in flight.
    """
    schema = fit_candidate_feature_schema(
        labeled_events=labeled_events,
        cfg=cfg,
        split_start=fit_start_idx,
        split_end=fit_end_idx,
    )
    fit_set = build_candidate_dataset(
        labeled_events=labeled_events,
        aligned=aligned,
        cfg=cfg,
        schema=schema,
        split_start=fit_start_idx,
        split_end=fit_end_idx,
        is_fit_split=True,
    )
    train_events = fit_set.event_index.copy()
    gross_targets = getattr(fit_set, "y_gross_return_bps", None)
    if gross_targets is None:
        gross_targets = fit_set.y_return_bps
    train_events["gross_return_bps"] = np.asarray(gross_targets, dtype=np.float64)
    l1_cfg = replace(cfg, ensemble_conditioning="archetype_only", ensemble_score_calibration_enabled=False)
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=l1_cfg, tag="ENS-FINAL")
    baseline_by_key: dict[MatchedBaselineKey, float] = {}
    baseline_frame = fit_set.event_index.copy()
    if "gross_event_bps" not in baseline_frame.columns and gross_targets is not None:
        baseline_frame["gross_event_bps"] = np.asarray(gross_targets, dtype=np.float64)
    if "side" in baseline_frame.columns and "expected_holding_bars" in baseline_frame.columns:
        baseline_frame["holding_bucket"] = (
            pd.to_numeric(
                baseline_frame["expected_holding_bars"],
                errors="coerce",
            )
            .fillna(1)
            .astype(int)
            .map(_holding_bucket)
        )
        grouped = baseline_frame.groupby(["symbol", "side", "holding_bucket"], sort=False)
        for (symbol, side, holding_bucket), group in grouped:
            side_literal: Literal[-1, 1] = 1 if int(side) >= 0 else -1
            baseline_by_key[MatchedBaselineKey(str(symbol), side_literal, int(holding_bucket))] = float(
                pd.to_numeric(group["gross_event_bps"], errors="coerce").fillna(0.0).mean()
            )
    config_hash = sha256(str(cfg).encode("utf-8")).hexdigest()[:12]
    return _Layer1ModelCore(
        feature_schema=schema,
        model=model,
        baseline_by_key=baseline_by_key,
        l1_fit_end_idx=fit_end_idx,
        model_version=schema.version,
        config_hash=config_hash,
    )


def assemble_layer1_artifact(
    core: _Layer1ModelCore,
    deployment_registry: QualifiedSignalRegistry,
    fit_end_idx: int,
) -> Layer1InferenceArtifact:
    """Lightweight assembly of pre-fit core + deployment registry into a full artifact."""
    return Layer1InferenceArtifact(
        feature_schema=core.feature_schema,
        model=core.model,
        deployment_registry=deployment_registry,
        baseline_by_key=core.baseline_by_key,
        l1_fit_end_idx=fit_end_idx,
        model_version=core.model_version,
        config_hash=core.config_hash,
    )


def fit_layer1_inference_artifact(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    deployment_registry: QualifiedSignalRegistry,
    fit_start_idx: int,
    fit_end_idx: int,
    cfg: CandidateStrategyConfig,
    seed: int,
) -> Layer1InferenceArtifact:
    """Full L1 artifact fit (pre-fit core + registry assembly)."""
    del seed
    core = prefit_layer1_model(
        labeled_events=labeled_events,
        aligned=aligned,
        fit_start_idx=fit_start_idx,
        fit_end_idx=fit_end_idx,
        cfg=cfg,
    )
    return assemble_layer1_artifact(
        core=core,
        deployment_registry=deployment_registry,
        fit_end_idx=fit_end_idx,
    )


def predict_layer1_signals(
    *,
    artifact: Layer1InferenceArtifact,
    candidate_events: pd.DataFrame,
    aligned: AlignedMarketData,
    start_idx: int,
    end_idx: int,
    cfg: CandidateStrategyConfig,
) -> ValidatedSignalBatch:
    inference_set = build_candidate_dataset(
        labeled_events=candidate_events,
        aligned=aligned,
        cfg=cfg,
        schema=artifact.feature_schema,
        split_start=start_idx,
        split_end=end_idx,
        require_label_within_split=False,
    )
    prediction = predict_regime_conditional_ensemble(
        model=artifact.model,
        oos_events=inference_set.event_index,
        cfg=cfg,
    )
    logger.debug(
        "[L2-SIGNAL-PRE] predict_layer1_signals: registry_symbols=%d activation_floor=%.1f start_idx=%d end_idx=%d",
        len(artifact.deployment_registry.by_symbol),
        float(cfg.l1_signal_activation_floor_bps),
        start_idx,
        end_idx,
    )
    return _candidate_output_to_signal_batch(
        model_output=prediction,
        registry=artifact.deployment_registry,
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        model_version=artifact.model_version,
        activation_floor_bps=float(cfg.l1_signal_activation_floor_bps),
        cfg=cfg,
    )


def predict_layer1_signals_multi_tf(
    *,
    artifacts_by_tf: dict[str, Layer1InferenceArtifact],
    candidate_events: pd.DataFrame,
    aligned: AlignedMarketData,
    start_idx: int,
    end_idx: int,
    cfg: CandidateStrategyConfig,
) -> ValidatedSignalBatch:
    """전 TF artifact에서 ValidatedSignalBatch를 통합 생성한다.

    TF별 ``predict_layer1_signals``를 호출하여 events를 병합한다.
    ``strategy_id``가 TF를 내장(예: ``donchian_72_8h``)하므로
    ``(symbol, strategy_id)`` 쌍은 TF 간 자연히 구분된다 — 별도 TF 컬럼 불필요.

    Invariant: 동일 ``(symbol, strategy_id, decision_idx)`` 중복 이벤트 0.
    위반 시 fail-closed 경고 로깅 후 첫 발생만 보존.

    Args:
        artifacts_by_tf: TF → Layer1InferenceArtifact 매핑.
        candidate_events: 전 TF 레이블 이벤트 DataFrame (``native_tf`` 컬럼 포함 시 필터 적용).
        aligned: 공통 base grid AlignedMarketData.
        start_idx: OOS 시작 bar index (look-ahead 방어: TF 공통 동일 window).
        end_idx: OOS 종료 bar index.
        cfg: CandidateStrategyConfig.

    Returns:
        ValidatedSignalBatch: 전 TF 이벤트 합본.
        artifacts_by_tf 빈 경우 빈 batch 반환.

    Time Complexity: O(T x predict_cost) where T = n_timeframes.
    Space Complexity: O(Σ events_per_tf).
    """
    if not artifacts_by_tf:
        logger.warning(
            "[MULTI-TF] artifacts_by_tf 비어있음 → 빈 ValidatedSignalBatch 반환 start_idx=%d end_idx=%d",
            start_idx,
            end_idx,
        )
        return ValidatedSignalBatch(
            events=(),
            start_idx=start_idx,
            end_idx=end_idx,
            symbols=aligned.symbols,
            registry_version="empty",
            model_version="empty",
        )

    batches: list[ValidatedSignalBatch] = []
    has_native_tf = "native_tf" in candidate_events.columns

    for tf in sorted(artifacts_by_tf):
        art = artifacts_by_tf[tf]
        # native_tf 필터: look-ahead 없음 — 이미 L1 단계에서 base grid 투영된 컬럼
        ev_tf: pd.DataFrame = (
            candidate_events[candidate_events["native_tf"] == tf] if has_native_tf else candidate_events
        )
        if ev_tf.empty:
            logger.debug("[MULTI-TF] tf=%s candidate_events 비어있음 — skip", tf)
            continue

        try:
            batch = predict_layer1_signals(
                artifact=art,
                candidate_events=ev_tf,
                aligned=aligned,
                start_idx=start_idx,
                end_idx=end_idx,
                cfg=cfg,
            )
        except Exception:
            logger.exception("[MULTI-TF] tf=%s predict_layer1_signals 실패 — skip", tf)
            continue

        batches.append(batch)
        logger.debug(
            "[MULTI-TF] tf=%s events=%d registry_symbols=%d",
            tf,
            len(batch.events),
            len(art.deployment_registry.by_symbol),
        )

    if not batches:
        logger.warning("[MULTI-TF] 모든 TF 빈 결과 → 빈 batch 반환")
        return ValidatedSignalBatch(
            events=(),
            start_idx=start_idx,
            end_idx=end_idx,
            symbols=aligned.symbols,
            registry_version="empty",
            model_version="empty",
        )

    # 이벤트 병합 — (symbol, strategy_id, decision_idx) 중복 체크
    merged_events: list[ValidatedSignalEvent] = []
    seen_keys: set[tuple[str, str, int]] = set()
    dup_count = 0
    for batch in batches:
        for ev in batch.events:
            key = (ev.symbol, ev.strategy_id, int(ev.decision_idx))
            if key in seen_keys:
                dup_count += 1
                continue
            seen_keys.add(key)
            merged_events.append(ev)

    if dup_count > 0:
        logger.warning(
            "[MULTI-TF] 중복 (symbol, strategy_id, decision_idx) %d건 — fail-closed 제거",
            dup_count,
        )

    # registry_version: TF별 registry_version SHA 합성
    _composite_rv = sha256("|".join(b.registry_version for b in batches).encode()).hexdigest()[:16]
    _composite_mv = sha256("|".join(b.model_version for b in batches).encode()).hexdigest()[:16]

    merged_start = min(b.start_idx for b in batches)
    merged_end = max(b.end_idx for b in batches)

    logger.debug(
        "[MULTI-TF] 병합 완료: n_tf=%d total_events=%d dup_removed=%d",
        len(batches),
        len(merged_events),
        dup_count,
    )

    return ValidatedSignalBatch(
        events=tuple(merged_events),
        start_idx=merged_start,
        end_idx=merged_end,
        symbols=aligned.symbols,
        registry_version=_composite_rv,
        model_version=_composite_mv,
    )
