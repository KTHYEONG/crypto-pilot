# src/domain/futures/strategy/tiered_workflow/signal_selection.py

from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import replace
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
    return str(
        row.get(
            "activation_context",
            row.get("signal_cell", row.get("entry_regime", "all")),
        )
    ) or "all"


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
) -> tuple[pd.DataFrame, int]:
    opp_frame = _batch_to_frame(opportunities)
    if opp_frame.empty:
        return opp_frame, 0
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
    merge_keys = ["decision_idx", "symbol", "strategy_id"]
    if activation_match_regime:
        merge_keys.append("activation_context")
    duplicate_mask = realized.duplicated(subset=merge_keys, keep=False)
    if bool(duplicate_mask.any()):
        raise ValueError("duplicate realized opportunity key")
    merge_cols = [*merge_keys, "realized_side_adjusted_gross_bps"]
    if "exit_idx" in realized.columns:
        merge_cols.append("exit_idx")
    merged = opp_frame.merge(
        realized[merge_cols],
        on=merge_keys,
        how="left",
        indicator=True,
    )
    unmatched_count = int((merged["_merge"] != "both").sum())
    merged = merged.loc[merged["_merge"] == "both"].drop(columns="_merge").copy()
    return merged, unmatched_count


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
        pd.to_numeric(frame.get("side", pd.Series(1, index=frame.index)), errors="coerce")
        .fillna(1.0)
        .astype(int)
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
        pd.to_numeric(frame.get("fold_id", pd.Series(0, index=frame.index)), errors="coerce")
        .fillna(0)
        .astype(int)
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
        frame.groupby(["symbol", "strategy_id", "activation_context", "fold_id"], sort=True)[
            "incremental_bps"
        ]
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
        fold_means = _pair_fold_means.get(
            (str(symbol), str(strategy_id), str(activation_context)), []
        )
        positive_fold_ratio = (
            float(sum(1 for value in fold_means if value > 0.0) / len(fold_means))
            if fold_means
            else 0.0
        )
        t_stat = _series_tstat(incremental)
        p_value = _one_sided_p_value(t_stat)
        boot_means = moving_block_bootstrap_mean(
            incremental.astype(np.float64, copy=False),
            group.get("decision_idx", group.get("entry_idx", pd.Series(0, index=group.index)))
            .to_numpy(dtype=np.int64, copy=False),
            block_bars=int(getattr(cfg, "l1_bootstrap_block_bars", 6)),
            n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 200)),
            seed=seed + int.from_bytes(
                sha256(f"{symbol}:{strategy_id}:{activation_context}".encode()).digest()[:4],
                byteorder="big",
            ) % 10_000,
        )
        probability_positive = (
            float(np.mean(boot_means > 0.0))
            if boot_means.size > 0
            else (1.0 if mean_incremental > 0.0 else 0.0)
        )
        lcb_net = (
            float(np.quantile(boot_means, 0.05))
            if boot_means.size > 0
            else mean_incremental
        )
        block_tstat = (
            float(mean_incremental / (np.std(boot_means, ddof=1) + 1e-12))
            if boot_means.size >= 2 and float(np.std(boot_means, ddof=1)) > 0.0
            else t_stat
        )
        t_prep_total += time.perf_counter() - t_step

        t_qs = time.perf_counter()

        # 1. 자유도 기반 적응적 t-critical value 계산
        df = effective_n - 1.0
        if df >= 2.0:
            alpha = float(getattr(cfg, "l1_pair_alpha", 0.05))
            power = float(getattr(cfg, "l1_pair_power", 0.80))
            t_crit = float(np.asarray(stats.t.ppf(1.0 - alpha, float(df)), dtype=np.float64))
            t_power = float(np.asarray(stats.t.ppf(power, float(df)), dtype=np.float64))
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
            )
        )
        raw_p_values.append(p_value)
        t_qualify_total += time.perf_counter() - t_qf
    # C2: effective-N correction for correlated probe TF hypotheses
    _m_eff: float | None = None
    if probe_diversity_corr is not None and raw_p_values:
        _m_eff = _compute_probe_m_eff(
            groups=[
                (e.key.symbol, e.key.strategy_id, str(e.key.activation_context))
                for e in evidence_list
            ],
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
        final_evidence.append(
            SymbolStrategyEvidence(
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
            )
        )
    qualified_count = sum(
        1 for ev in final_evidence
        if ev.hard_eligible and ev.quality_weight > 0.0
    )
    t_elapsed = time.perf_counter() - t_core
    logger.log(
        PERF,
        "[PERF] signal_evidence n_pairs=%d n_qualified=%d "
        "prep=%.4fs stats=%.4fs qualify=%.4fs took=%.4fs",
        len(evidence_list), qualified_count,
        t_prep_total, t_stats_total, t_qualify_total, t_elapsed,
    )
    # 진단: registry 공집합 경고
    if qualified_count == 0 and final_evidence:
        reasons: Counter[str] = Counter(
            r for ev in final_evidence for r in ev.structural_reasons
        )
        qw_zero = sum(
            1 for ev in final_evidence
            if ev.hard_eligible and ev.quality_weight <= 0.0
        )
        logger.warning(
            "[L1-EVIDENCE] as_of=%d: %d pairs, 0 qualified. "
            "structural_reasons=%s, hard_eligible_but_qw_zero=%d",
            registry_as_of_idx,
            len(final_evidence),
            dict(reasons),
            qw_zero,
        )
    return tuple(final_evidence)


def build_qualified_signal_registry(
    *,
    evidence: tuple[SymbolStrategyEvidence, ...],
    symbols: tuple[str, ...],
    min_signals_per_symbol: int,
    registry_version: str,
    cfg: CandidateStrategyConfig | None = None,
    probe_prior_map: dict[tuple[str, str, str], float] | None = None,
) -> QualifiedSignalRegistry:
    grouped: dict[str, list[SymbolStrategyEvidence]] = defaultdict(list)
    # cfg=None → LCB gate disabled (backward compat for tests / callers without cfg)
    breakeven: float | None = (
        float(getattr(cfg, "l1_breakeven_floor_bps", 0.0)) if cfg is not None else None
    )
    for item in evidence:
        hard_eligible = bool(getattr(item, "hard_eligible", getattr(item, "qualified", False)))
        quality_weight = float(getattr(item, "quality_weight", getattr(item, "reliability", 0.0)))
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
            raw_mu=float(best.mean_incremental_bps),   # gross → incremental (net) edge
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
            (item.key.symbol, item.key.strategy_id)
            for items in registry.by_symbol.values()
            for item in items
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
        if "side" in frame.columns else np.ones(n_raw, dtype=np.float64)
    )
    side_arr_v = np.where(side_raw >= 0, 1, -1).astype(np.int64)
    hold_raw = (
        pd.to_numeric(frame["expected_holding_bars"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        if "expected_holding_bars" in frame.columns else np.ones(n_raw, dtype=np.float64)
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
        n_raw, n_registry_pass, n_gross_pass, n_threshold_pass, n_decision_pass, len(events), activation_floor_bps,
    )
    _t_sort = time.perf_counter()
    events.sort(key=lambda item: (item.decision_idx, item.symbol, item.strategy_id, item.activation_context))
    _t_sort_took = time.perf_counter() - _t_sort
    logger.log(
        PERF,
        "[PERF] signal_batch_convert n_raw=%d n_out=%d pred=%.4fs keys=%.4fs loop=%.4fs sort=%.4fs total=%.4fs",
        n_raw, len(events), _t_pred_took, _t_keys_took, _t_loop_took, _t_sort_took,
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
        current_score = (
            event.expected_gross_bps
            / max(event.expected_holding_bars, 1)
            * max(event.quality_weight, 0.0)
        )
        best_score = (
            candidate.expected_gross_bps
            / max(candidate.expected_holding_bars, 1)
            * max(candidate.quality_weight, 0.0)
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
) -> Layer1FoldReadiness:
    opp_frame = _batch_to_frame(opportunities)
    if opp_frame.empty:
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
            blockers=("empty_opportunities",),
        )
    merged, unmatched_count = align_outer_opportunities_with_realized(
        opportunities=opportunities,
        realized_event_results=realized_event_results,
        activation_match_regime=bool(getattr(cfg, "l1_activation_match_regime", True)),
    )
    dropped_by_maturity = 0
    if "exit_idx" in merged.columns:
        before_maturity = len(merged)
        merged = merged.loc[
            pd.to_numeric(merged["exit_idx"], errors="coerce").fillna(fold.oos_end - 1).astype(int) < fold.oos_end
        ].copy()
        dropped_by_maturity = before_maturity - len(merged)
    if merged.empty:
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
            blockers=("empty_realized_merge",),
            dropped_by_maturity_count=dropped_by_maturity,
        )
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(aligned_symbols)}
    probe_series: list[float] = []
    probe_mode: str = str(getattr(cfg, "l1_opp_ic_mode", "cross_section"))

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
                    float(volatility_2d[d_idx, symbol_idx])
                    if volatility_2d.ndim == 2
                    else float(volatility_2d[d_idx])
                )
                denom = max(vol, VOL_FLOOR)
                risk_scores_ts.append(
                    (abs(float(row.expected_gross_bps)) * max(float(row.quality_weight), 0.0) / denom, row_i)
                )
            if risk_scores_ts:
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
        block_bars=int(getattr(cfg, "l1_bootstrap_block_bars", 6)),
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
        block_bars=int(getattr(cfg, "l1_bootstrap_block_bars", 6)),
        n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 200)),
        seed=seed,
    )
    return float(np.quantile(boot, 0.05)) if boot.size > 0 else float(np.mean(series))


def evaluate_layer1_readiness(
    *,
    fold_reports: tuple[Layer1FoldReadiness, ...],
    fold_cov: float,
    trade_scope_count: int,
    cfg: CandidateStrategyConfig,
    seed: int = 0,
) -> Layer1GateReport:
    effective_symbol_count = 0.0
    probe_series: list[float] = []
    match_ratios: list[float] = []
    probe_lcbs: list[float] = []
    ready_fold_count = 0
    for report in fold_reports:
        if report.passed:
            ready_fold_count += 1
        effective_symbol_count = max(effective_symbol_count, report.effective_symbol_count)
        match_ratios.append(report.realized_match_ratio)
        probe_series.extend([value for value in report.probe_series_bps if np.isfinite(value)])
        probe_lcbs.append(report.probe_lcb_bps)
    fold_ratio = float(ready_fold_count / len(fold_reports)) if fold_reports else 0.0
    match_ratio = float(np.mean(match_ratios)) if match_ratios else 0.0
    probe_bps = float(np.mean(probe_series)) if probe_series else 0.0
    
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

    check_specs = (
        (
            "fold_cov",
            fold_cov,
            float(getattr(cfg, "l1_min_fold_cov", 0.8)),
            "ge",
        ),
        ("match_ratio", match_ratio, float(getattr(cfg, "l1_min_realized_match_ratio", 0.90)), "ge"),
        ("sym_count", effective_sym_metric, sym_threshold, "ge"),
        ("fold_ratio", fold_ratio, float(cfg.l1_min_fold_ratio), "ge"),
        ("probe_lcb_bps", probe_lcb, float(cfg.l1_min_probe_bps), "gt"),
    )
    checks: list[Layer1GateCheck] = []
    blockers: list[str] = []
    for key, value, threshold, comparator in check_specs:
        finite_value = np.isfinite(value)
        passed = finite_value and (value >= threshold if comparator == "ge" else value > threshold)
        blocker = None if passed else f"{value:.3f}"
        comparator_literal = cast(Literal["ge", "gt"], comparator)
        if blocker is not None:
            blockers.append(f"{key}:{blocker}")
        checks.append(
            Layer1GateCheck(
                key=key,
                value=float(value),
                threshold=float(threshold),
                comparator=comparator_literal,
                passed=passed,
                blocker=blocker,
            )
        )
    return Layer1GateReport(
        checks=tuple(checks),
        passed=all(check.passed for check in checks),
        blockers=tuple(blockers),
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
    del seed
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
    # D2: Layer1은 regime μ 조건화 배제 — archetype_only 고정 (regime → Layer2 risk overlay 전용)
    l1_cfg = replace(cfg, ensemble_conditioning="archetype_only", ensemble_score_calibration_enabled=False)
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=l1_cfg, tag="ENS-FINAL")
    baseline_by_key: dict[MatchedBaselineKey, float] = {}
    baseline_frame = fit_set.event_index.copy()
    if "gross_event_bps" not in baseline_frame.columns and gross_targets is not None:
        baseline_frame["gross_event_bps"] = np.asarray(gross_targets, dtype=np.float64)
    if "side" in baseline_frame.columns and "expected_holding_bars" in baseline_frame.columns:
        baseline_frame["holding_bucket"] = pd.to_numeric(
            baseline_frame["expected_holding_bars"],
            errors="coerce",
        ).fillna(1).astype(int).map(_holding_bucket)
        grouped = baseline_frame.groupby(["symbol", "side", "holding_bucket"], sort=False)
        for (symbol, side, holding_bucket), group in grouped:
            side_literal: Literal[-1, 1] = 1 if int(side) >= 0 else -1
            baseline_by_key[MatchedBaselineKey(str(symbol), side_literal, int(holding_bucket))] = float(
                pd.to_numeric(group["gross_event_bps"], errors="coerce").fillna(0.0).mean()
            )
    config_hash = sha256(str(cfg).encode("utf-8")).hexdigest()[:12]
    return Layer1InferenceArtifact(
        feature_schema=schema,
        model=model,
        deployment_registry=deployment_registry,
        baseline_by_key=baseline_by_key,
        l1_fit_end_idx=fit_end_idx,
        model_version=schema.version,
        config_hash=config_hash,
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
