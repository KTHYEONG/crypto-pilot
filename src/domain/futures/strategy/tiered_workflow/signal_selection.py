# src/domain/futures/strategy/tiered_workflow/signal_selection.py

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray
from scipy.stats import spearmanr

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

    # Aggregate per (symbol, side, holding_bucket) bucket
    bucket_stats = frame.groupby(bucket_key, sort=False)["gross_event_bps"].agg(
        _bucket_sum="sum", _bucket_count="count"
    )
    # Aggregate per (symbol, side, holding_bucket, strategy_id)
    strat_stats = frame.groupby(strategy_key, sort=False)["gross_event_bps"].agg(
        _strat_sum="sum", _strat_count="count"
    )

    merged = frame[strategy_key].copy()
    merged = merged.join(bucket_stats, on=bucket_key)
    merged = merged.join(strat_stats, on=strategy_key)

    peer_count = merged["_bucket_count"] - merged["_strat_count"]
    peer_sum = merged["_bucket_sum"] - merged["_strat_sum"]

    # peer_count == 0 → single strategy in bucket → absolute fallback (baseline=0)
    safe_peer_count = peer_count.clip(lower=1)
    peer_mean = np.where(
        peer_count > 0,
        peer_sum.to_numpy(dtype=np.float64) / safe_peer_count.to_numpy(dtype=np.float64),
        0.0,
    )

    return pd.Series(
        gross.to_numpy(dtype=np.float64) - peer_mean,
        index=frame.index,
    )


def _expected_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.expected_gross_bps, dtype=np.float64)


def _q10_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q10_gross_bps, dtype=np.float64)


def _q90_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q90_gross_bps, dtype=np.float64)


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


def _by_q_values(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    if p_values.size == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(p_values)
    ordered = p_values[order]
    m = float(p_values.size)
    harmonic = float(np.sum(1.0 / np.arange(1, p_values.size + 1, dtype=np.float64)))
    adjusted = np.empty_like(ordered)
    running = 1.0
    for idx in range(ordered.size - 1, -1, -1):
        rank = float(idx + 1)
        candidate = min(1.0, ordered[idx] * m * harmonic / rank)
        running = min(running, candidate)
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


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
        frame = frame.loc[
            exit_idx < float(registry_as_of_idx)
        ].copy()
        lookback_bars = getattr(cfg, "l1_evidence_lookback_bars", None)
        if lookback_bars is not None:
            min_exit_idx = float(registry_as_of_idx - int(lookback_bars))
            frame = frame.loc[exit_idx.loc[frame.index] >= min_exit_idx].copy()
    if frame.empty:
        return ()
    frame["holding_bucket"] = frame["expected_holding_bars"].map(_holding_bucket)
    baseline_mode: Literal["peer_exclusive", "absolute"] = getattr(cfg, "l1_baseline_mode", "peer_exclusive")
    frame["incremental_bps"] = _compute_incremental_bps(frame, mode=baseline_mode)
    grouped = frame.groupby(["symbol", "strategy_id", "activation_context"], sort=True)
    evidence_list: list[SymbolStrategyEvidence] = []
    raw_p_values: list[float] = []
    for (symbol, strategy_id, activation_context), group in grouped:
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
        fold_means = [
            float(group_fold["incremental_bps"].mean())
            for _, group_fold in group.groupby("fold_id", sort=True)
            if not group_fold.empty
        ]
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
            block_bars=int(getattr(cfg, "l1_bootstrap_block_bars", 1)),
            n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 1)),
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
        block_tstat = (
            float(mean_incremental / (np.std(boot_means, ddof=1) + 1e-12))
            if boot_means.size >= 2 and float(np.std(boot_means, ddof=1)) > 0.0
            else t_stat
        )

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

        structural_reasons: list[str] = []
        if effective_n < float(cfg.l1_pair_min_effective_obs):
            structural_reasons.append("insufficient_effective_obs")
        if len(fold_means) < int(cfg.l1_pair_min_folds):
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
            else:
                quality_weight = 1.0
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
            )
        )
        raw_p_values.append(p_value)
    q_values = _by_q_values(np.asarray(raw_p_values, dtype=np.float64))
    final_evidence: list[SymbolStrategyEvidence] = []
    for idx, evidence in enumerate(evidence_list):
        q_value = float(q_values[idx])
        diag_flags = list(evidence.diagnostic_flags)
        quality_weight = evidence.quality_weight
        if q_value > float(cfg.l1_pair_fdr_alpha):
            diag_flags.append("fdr_reject")
        if evidence.hard_eligible and bool(getattr(cfg, "l1_quality_weight_enabled", True)):
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
            )
        )
    # 진단: registry 공집합 경고
    qualified_count = sum(
        1 for ev in final_evidence
        if ev.hard_eligible and ev.quality_weight > 0.0
    )
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
) -> QualifiedSignalRegistry:
    grouped: dict[str, list[SymbolStrategyEvidence]] = defaultdict(list)
    for item in evidence:
        hard_eligible = bool(getattr(item, "hard_eligible", getattr(item, "qualified", False)))
        quality_weight = float(getattr(item, "quality_weight", getattr(item, "reliability", 0.0)))
        if hard_eligible and quality_weight > 0.0:
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
            raw_mu=float(best.mean_gross_bps),
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
    frame = model_output.events.reset_index(drop=True).copy()
    if frame.empty:
        return ValidatedSignalBatch(
            events=(),
            start_idx=0,
            end_idx=0,
            symbols=symbols,
            registry_version=registry.registry_version,
            model_version=model_version,
        )
    gross = _expected_gross_bps(model_output)
    q10 = _q10_gross_bps(model_output)
    q90 = _q90_gross_bps(model_output)
    activation_match_regime: bool = bool(getattr(cfg, "l1_activation_match_regime", True)) if cfg is not None else True
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
    events: list[ValidatedSignalEvent] = []
    start_idx = int(frame["entry_idx"].min()) if "entry_idx" in frame.columns and not frame.empty else 0
    end_idx = int(frame["entry_idx"].max()) + 1 if "entry_idx" in frame.columns and not frame.empty else 0
    has_explicit_gross = bool(getattr(model_output, "_has_explicit_expected_gross_bps", True))
    for idx, row in frame.iterrows():
        key = _signal_source_key_from_row(row, qualify_by_regime=activation_match_regime)
        if activation_match_regime:
            if (key.symbol, key.strategy_id, key.activation_context) not in source_keys:
                continue
        else:
            if (key.symbol, key.strategy_id) not in source_keys_relaxed:
                continue
        if not has_explicit_gross:
            continue
        pred = float(gross[idx]) if idx < gross.size else 0.0
        if pred < activation_floor_bps:
            continue
        entry_idx = int(pd.to_numeric(row.get("entry_idx", 0), errors="coerce"))
        decision_idx = entry_idx - 1
        if decision_idx < 0 or decision_idx >= datetimes.shape[0]:
            continue
        side_val = int(pd.to_numeric(row.get("side", 1), errors="coerce"))
        side: int = 1 if side_val >= 0 else -1
        holding = max(int(pd.to_numeric(row.get("expected_holding_bars", 1), errors="coerce")), 1)
        quality_weight = 0.0
        for evidence in registry.by_symbol.get(key.symbol, ()):
            if evidence.key == key:
                quality_weight = evidence.quality_weight
                break
        events.append(
            ValidatedSignalEvent(
                decision_idx=decision_idx,
                decision_time=datetimes[decision_idx],
                symbol=key.symbol,
                strategy_id=key.strategy_id,
                activation_context=key.activation_context,
                side=1 if side >= 0 else -1,
                expected_gross_bps=pred,
                q10_gross_bps=float(q10[idx]) if idx < q10.size else pred,
                q90_gross_bps=float(q90[idx]) if idx < q90.size else pred,
                expected_holding_bars=holding,
                quality_weight=quality_weight,
                registry_version=registry.registry_version,
                model_version=model_version,
            )
        )
    events.sort(key=lambda item: (item.decision_idx, item.symbol, item.strategy_id, item.activation_context))
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
    if "exit_idx" in merged.columns:
        merged = merged.loc[
            pd.to_numeric(merged["exit_idx"], errors="coerce").fillna(fold.oos_end - 1).astype(int) < fold.oos_end
        ].copy()
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
        )
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(aligned_symbols)}
    ic_series: list[float] = []
    probe_series: list[float] = []
    ic_mode: str = str(getattr(cfg, "l1_opp_ic_mode", "cross_section"))
    prediction_unique_count = 0

    if ic_mode == "time_series":
        prediction_unique_threshold = int(getattr(cfg, "l1_min_prediction_unique_values", 3))
        for _symbol, sym_group in merged.groupby("symbol", sort=True):
            sym_group = sym_group.drop_duplicates(subset=["decision_idx"], keep="first")
            pred_ts = sym_group["expected_gross_bps"].to_numpy(dtype=np.float64)
            real_ts = sym_group["realized_side_adjusted_gross_bps"].fillna(0.0).to_numpy(dtype=np.float64)
            pred_unique_count = int(np.unique(np.round(pred_ts, decimals=12)).size)
            prediction_unique_count = max(
                prediction_unique_count,
                pred_unique_count,
            )
            if (
                sym_group.shape[0] >= max(3, prediction_unique_threshold)
                and pred_unique_count >= prediction_unique_threshold
            ):
                ic_val, _ = spearmanr(pred_ts, real_ts)
            else:
                ic_val = np.nan
            if np.isfinite(ic_val):
                ic_series.append(float(ic_val))
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
        prediction_unique_threshold = int(getattr(cfg, "l1_min_prediction_unique_values", 3))
        for decision_idx, group in merged.groupby("decision_idx", sort=True):
            group = group.drop_duplicates(subset=["symbol"], keep="first")
            if group.shape[0] < int(cfg.l1_min_cross_section):
                continue
            pred = group["expected_gross_bps"].to_numpy(dtype=np.float64, copy=False)
            real = group["realized_side_adjusted_gross_bps"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            pred_unique_count = int(np.unique(np.round(pred, decimals=12)).size)
            prediction_unique_count = max(
                prediction_unique_count,
                pred_unique_count,
            )
            if pred_unique_count >= prediction_unique_threshold:
                ic_val, _ = spearmanr(pred, real)
            else:
                ic_val = np.nan
            if np.isfinite(ic_val):
                ic_series.append(float(ic_val))
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
    opportunity_ic = float(np.mean(ic_series)) if ic_series else None
    opportunity_ic_tstat = _series_tstat(np.asarray(ic_series, dtype=np.float64))
    probe_gross_edge = float(np.mean(probe_series)) if probe_series else 0.0
    probe_boot = moving_block_bootstrap_mean(
        np.asarray(probe_series, dtype=np.float64),
        np.arange(len(probe_series), dtype=np.int64),
        block_bars=int(getattr(cfg, "l1_bootstrap_block_bars", 1)),
        n_bootstrap=int(getattr(cfg, "l1_bootstrap_samples", 1)),
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
        prediction_unique_count=prediction_unique_count,
        opportunity_ic=opportunity_ic,
        opportunity_ic_tstat=opportunity_ic_tstat,
        probe_bps=probe_gross_edge,
        probe_lcb_bps=probe_lcb,
        probe_series_bps=tuple(probe_series),
        effective_symbol_count=effective_symbol_count,
        passed=not blockers,
        blockers=tuple(blockers),
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
    return _candidate_output_to_signal_batch(
        model_output=prediction,
        registry=artifact.deployment_registry,
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        model_version=artifact.model_version,
        activation_floor_bps=float(cfg.l1_signal_activation_floor_bps),
        cfg=cfg,
    )
