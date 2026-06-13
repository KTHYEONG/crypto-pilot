# src/domain/futures/strategy/tiered_workflow/signal_selection.py

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
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
)

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.walk_forward import WFFold


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


def _expected_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.expected_gross_bps, dtype=np.float64)


def _q10_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q10_gross_bps, dtype=np.float64)


def _q90_gross_bps(model_output: CandidateModelOutput) -> NDArray[np.float64]:
    return np.asarray(model_output.q90_gross_bps, dtype=np.float64)


def _signal_source_key_from_row(row: pd.Series) -> SignalSourceKey:
    strategy_id = str(
        row.get(
            "strategy_id",
            f"{row.get('family', '')}:{row.get('variant', '')}",
        )
    )
    activation_context = str(
        row.get(
            "activation_context",
            row.get("signal_cell", row.get("entry_regime", "all")),
        )
    )
    return SignalSourceKey(
        symbol=str(row.get("symbol", "")),
        strategy_id=strategy_id,
        activation_context=activation_context or "all",
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
                "reliability": event.reliability,
                "registry_version": event.registry_version,
                "model_version": event.model_version,
            }
            for event in batch.events
        ]
    )


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
) -> tuple[SymbolStrategyEvidence, ...]:
    """Compute per-source signal evidence from event-level OOS results."""
    del seed
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
    frame["holding_bucket"] = frame["expected_holding_bars"].map(_holding_bucket)
    if "baseline_gross_bps" not in frame.columns:
        baseline_map = (
            frame.groupby(["symbol", "side", "holding_bucket"], sort=False)["gross_event_bps"]
            .mean()
            .to_dict()
        )
        frame["baseline_gross_bps"] = [
            baseline_map.get((str(symbol), int(side), int(bucket)), 0.0)
            for symbol, side, bucket in zip(
                frame["symbol"],
                frame["side"],
                frame["holding_bucket"],
                strict=True,
            )
        ]
    frame["incremental_bps"] = frame["gross_event_bps"] - pd.to_numeric(
        frame["baseline_gross_bps"],
        errors="coerce",
    ).fillna(0.0)
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
        reliability = float(
            np.clip(
                max(mean_incremental, 0.0)
                * max(t_stat, 0.0)
                / max(float(getattr(cfg, "l1_pair_min_incremental_tstat", 1.0)), 1.0)
                / max(abs(float(getattr(cfg, "l1_pair_min_mean_gross_bps", 1.0))) + 1.0, 1.0),
                0.0,
                1.0,
            )
        )
        rejection_reasons: list[str] = []
        if effective_n < float(cfg.l1_pair_min_effective_obs):
            rejection_reasons.append("insufficient_effective_obs")
        if len(fold_means) < int(cfg.l1_pair_min_folds):
            rejection_reasons.append("insufficient_folds")
        if mean_gross <= float(cfg.l1_pair_min_mean_gross_bps):
            rejection_reasons.append("negative_gross_edge")
        if mean_incremental <= float(cfg.l1_pair_min_incremental_bps):
            rejection_reasons.append("no_incremental_edge")
        if t_stat < float(cfg.l1_pair_min_incremental_tstat):
            rejection_reasons.append("weak_tstat")
        if positive_fold_ratio < float(cfg.l1_pair_min_positive_fold_ratio):
            rejection_reasons.append("unstable_folds")
        evidence_list.append(
            SymbolStrategyEvidence(
                key=SignalSourceKey(
                    symbol=str(symbol),
                    strategy_id=str(strategy_id),
                    activation_context=str(activation_context or "all"),
                ),
                mean_gross_bps=mean_gross,
                mean_incremental_bps=mean_incremental,
                bootstrap_tstat_incremental=t_stat,
                p_value=p_value,
                q_value=1.0,
                positive_fold_ratio=positive_fold_ratio,
                n_obs=n_obs,
                effective_n=effective_n,
                n_folds=len(fold_means),
                reliability=reliability,
                qualified=False,
                rejection_reasons=tuple(rejection_reasons),
            )
        )
        raw_p_values.append(p_value)
    q_values = _by_q_values(np.asarray(raw_p_values, dtype=np.float64))
    final_evidence: list[SymbolStrategyEvidence] = []
    for idx, evidence in enumerate(evidence_list):
        reasons = list(evidence.rejection_reasons)
        q_value = float(q_values[idx])
        if q_value > float(cfg.l1_pair_fdr_alpha):
            reasons.append("fdr_reject")
        final_evidence.append(
            SymbolStrategyEvidence(
                key=evidence.key,
                mean_gross_bps=evidence.mean_gross_bps,
                mean_incremental_bps=evidence.mean_incremental_bps,
                bootstrap_tstat_incremental=evidence.bootstrap_tstat_incremental,
                p_value=evidence.p_value,
                q_value=q_value,
                positive_fold_ratio=evidence.positive_fold_ratio,
                n_obs=evidence.n_obs,
                effective_n=evidence.effective_n,
                n_folds=evidence.n_folds,
                reliability=evidence.reliability,
                qualified=not reasons,
                rejection_reasons=tuple(reasons),
            )
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
        if item.qualified:
            grouped[item.key.symbol].append(item)
    by_symbol: dict[str, tuple[SymbolStrategyEvidence, ...]] = {}
    ready_symbols: list[str] = []
    for symbol in symbols:
        items = tuple(
            sorted(
                grouped.get(symbol, ()),
                key=lambda candidate: (
                    candidate.reliability,
                    candidate.mean_incremental_bps,
                    candidate.bootstrap_tstat_incremental,
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
            t_stat=float(best.bootstrap_tstat_incremental),
            valid=True,
            beta_btc=None,
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
    source_keys = {
        (item.key.symbol, item.key.strategy_id, item.key.activation_context)
        for items in registry.by_symbol.values()
        for item in items
    }
    events: list[ValidatedSignalEvent] = []
    start_idx = int(frame["entry_idx"].min()) if "entry_idx" in frame.columns and not frame.empty else 0
    end_idx = int(frame["entry_idx"].max()) + 1 if "entry_idx" in frame.columns and not frame.empty else 0
    for idx, row in frame.iterrows():
        key = _signal_source_key_from_row(row)
        if (key.symbol, key.strategy_id, key.activation_context) not in source_keys:
            continue
        pred = float(gross[idx]) if idx < gross.size else 0.0
        if pred <= activation_floor_bps:
            continue
        entry_idx = int(pd.to_numeric(row.get("entry_idx", 0), errors="coerce"))
        decision_idx = entry_idx - 1
        if decision_idx < 0 or decision_idx >= datetimes.shape[0]:
            continue
        side_val = int(pd.to_numeric(row.get("side", 1), errors="coerce"))
        side: int = 1 if side_val >= 0 else -1
        holding = max(int(pd.to_numeric(row.get("expected_holding_bars", 1), errors="coerce")), 1)
        reliability = 0.0
        for evidence in registry.by_symbol.get(key.symbol, ()):
            if evidence.key == key:
                reliability = evidence.reliability
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
                reliability=reliability,
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
            * max(event.reliability, 0.0)
        )
        best_score = (
            candidate.expected_gross_bps
            / max(candidate.expected_holding_bars, 1)
            * max(candidate.reliability, 0.0)
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
    fold: WFFold,
    cfg: CandidateStrategyConfig,
    seed: int,
) -> Layer1FoldReadiness:
    del seed
    opp_frame = _batch_to_frame(opportunities)
    if opp_frame.empty:
        return Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=fold.fit_end,
            outer_oos_start_idx=fold.oos_start,
            outer_oos_end_idx=fold.oos_end,
            ready_symbols=(),
            valid_opportunity_timestamp_count=0,
            opportunity_ic=0.0,
            opportunity_ic_series=(),
            probe_bps=0.0,
            probe_gross_edge_series_bps=(),
            passed=False,
            blockers=("empty_opportunities",),
        )
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
        pd.to_numeric(realized.get("entry_idx", pd.Series(0, index=realized.index)), errors="coerce")
        .fillna(0)
        .astype(int)
        - 1
    )
    if "realized_side_adjusted_gross_bps" not in realized.columns:
        realized["realized_side_adjusted_gross_bps"] = pd.to_numeric(
            realized.get("gross_event_bps", pd.Series(0.0, index=realized.index)),
            errors="coerce",
        ).fillna(0.0)
    merge_cols = [
        "decision_idx",
        "symbol",
        "strategy_id",
        "activation_context",
        "realized_side_adjusted_gross_bps",
    ]
    if "exit_idx" in realized.columns:
        merge_cols.append("exit_idx")
    merged = opp_frame.merge(
        realized[merge_cols],
        on=["decision_idx", "symbol", "strategy_id", "activation_context"],
        how="left",
    )
    if "exit_idx" in merged.columns:
        merged = merged.loc[
            pd.to_numeric(merged["exit_idx"], errors="coerce").fillna(fold.oos_end - 1).astype(int) < fold.oos_end
        ].copy()
    if merged.empty:
        return Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=fold.fit_end,
            outer_oos_start_idx=fold.oos_start,
            outer_oos_end_idx=fold.oos_end,
            ready_symbols=(),
            valid_opportunity_timestamp_count=0,
            opportunity_ic=0.0,
            opportunity_ic_series=(),
            probe_bps=0.0,
            probe_gross_edge_series_bps=(),
            passed=False,
            blockers=("empty_realized_merge",),
        )
    symbol_to_idx = {
        symbol: idx for idx, symbol in enumerate(opportunities.symbols)
    }
    ic_series: list[float] = []
    probe_series: list[float] = []
    for decision_idx, group in merged.groupby("decision_idx", sort=True):
        group = group.drop_duplicates(subset=["symbol"], keep="first")
        if group.shape[0] < int(cfg.l1_min_cross_section):
            continue
        pred = (
            group["side"].to_numpy(dtype=np.float64, copy=False)
            * group["expected_gross_bps"].to_numpy(dtype=np.float64, copy=False)
            / np.maximum(group["expected_holding_bars"].to_numpy(dtype=np.float64, copy=False), 1.0)
        )
        real = (
            group["side"].to_numpy(dtype=np.float64, copy=False)
            * group["realized_side_adjusted_gross_bps"].fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            / np.maximum(group["expected_holding_bars"].to_numpy(dtype=np.float64, copy=False), 1.0)
        )
        ic_val, _ = spearmanr(pred, real)
        if np.isfinite(ic_val):
            ic_series.append(float(ic_val))
        risk_scores: list[tuple[float, int]] = []
        for row_idx, row in enumerate(group.itertuples(index=False)):
            symbol_idx = symbol_to_idx.get(str(row.symbol))
            if symbol_idx is None or decision_idx < 0 or decision_idx >= volatility_2d.shape[0]:
                continue
            vol = float(volatility_2d[int(decision_idx), symbol_idx])
            denom = max(vol, VOL_FLOOR)
            risk_scores.append((abs(float(row.expected_gross_bps)) / denom, row_idx))
        if risk_scores:
            risk_scores.sort(reverse=True)
            selected_idx = [row_idx for _, row_idx in risk_scores[: int(cfg.l1_probe_top_k)]]
            selected_real = real[np.asarray(selected_idx, dtype=np.int64)]
            if selected_real.size > 0:
                probe_series.append(float(np.mean(selected_real)))
    ready_symbols = tuple(sorted(str(symbol) for symbol in merged["symbol"].dropna().unique()))
    opportunity_ic = float(np.mean(ic_series)) if ic_series else 0.0
    probe_gross_edge = float(np.mean(probe_series)) if probe_series else 0.0
    blockers: list[str] = []
    fold_min_ready_symbols = max(1, min(int(cfg.l1_min_sym_count), int(cfg.l1_min_cross_section)))
    if len(ready_symbols) < fold_min_ready_symbols:
        blockers.append("insufficient_ready_symbols")
    if len(ic_series) < int(cfg.l1_min_opportunity_timestamps):
        blockers.append("insufficient_opportunity_timestamps")
    if not np.isfinite(opportunity_ic):
        blockers.append("non_finite_ic")
    if probe_gross_edge <= 0.0:
        blockers.append("non_positive_probe")
    return Layer1FoldReadiness(
        fold_id=fold.oos_start,
        registry_source_end_idx=fold.fit_end,
        outer_oos_start_idx=fold.oos_start,
        outer_oos_end_idx=fold.oos_end,
        ready_symbols=ready_symbols,
        valid_opportunity_timestamp_count=len(ic_series),
        opportunity_ic=opportunity_ic,
        opportunity_ic_series=tuple(ic_series),
        probe_bps=probe_gross_edge,
        probe_gross_edge_series_bps=tuple(probe_series),
        passed=not blockers,
        blockers=tuple(blockers),
    )


def evaluate_layer1_readiness(
    *,
    fold_reports: tuple[Layer1FoldReadiness, ...],
    fold_cov: float,
    trade_scope_count: int,
    cfg: CandidateStrategyConfig,
) -> Layer1GateReport:
    symbol_counter: Counter[str] = Counter()
    ic_series: list[float] = []
    probe_series: list[float] = []
    ready_fold_count = 0
    for report in fold_reports:
        if report.passed:
            ready_fold_count += 1
        symbol_counter.update(report.ready_symbols)
        ic_series.extend([value for value in report.opportunity_ic_series if np.isfinite(value)])
        probe_series.extend([value for value in report.probe_gross_edge_series_bps if np.isfinite(value)])
    stable_ready_symbols = [
        symbol
        for symbol, count in symbol_counter.items()
        if count >= int(cfg.l1_min_ready_outer_folds)
    ]
    fold_ratio = float(ready_fold_count / len(fold_reports)) if fold_reports else 0.0
    opp_ic = float(np.mean(ic_series)) if ic_series else 0.0
    opp_tstat = _series_tstat(np.asarray(ic_series, dtype=np.float64))
    probe_bps = float(np.mean(probe_series)) if probe_series else 0.0
    probe_tstat = _series_tstat(np.asarray(probe_series, dtype=np.float64))
    sym_ratio = float(len(stable_ready_symbols) / max(1, trade_scope_count))
    check_specs = (
        (
            "fold_cov",
            fold_cov,
            float(getattr(cfg, "l1_min_fold_cov", 0.8)),
            "ge",
        ),
        ("sym_count", float(len(stable_ready_symbols)), float(cfg.l1_min_sym_count), "ge"),
        ("sym_ratio", sym_ratio, float(cfg.l1_min_sym_ratio), "ge"),
        ("fold_ratio", fold_ratio, float(cfg.l1_min_fold_ratio), "ge"),
        ("opp_ic", opp_ic, float(cfg.l1_min_opp_ic), "ge"),
        ("opp_tstat", opp_tstat, float(cfg.l1_min_opp_tstat), "ge"),
        ("probe_bps", probe_bps, float(cfg.l1_min_probe_bps), "gt"),
        ("probe_tstat", probe_tstat, float(cfg.l1_min_probe_tstat), "ge"),
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
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
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
    )
