from __future__ import annotations

import logging

import pandas as pd

from src.domain.futures.strategy.candidate_contracts import (
    CausalAlphaSegment,
    CausalAlphaTape,
    ValidatedSignalBatch,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.walk_forward import CausalL2Fold

_logger = logging.getLogger(__name__)


def build_cross_fitted_alpha_tape(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    folds: tuple[CausalL2Fold, ...],
    cfg: CandidateStrategyConfig,
    timeframes: tuple[str, ...],
    max_label_horizon_bars: int,
    seed: int,
) -> CausalAlphaTape:
    _logger.info(
        "building cross-fitted alpha tape: %d folds, %d timeframes, seed=%d",
        len(folds),
        len(timeframes),
        seed,
    )
    segments: list[CausalAlphaSegment] = []
    for fold in folds:
        segment = CausalAlphaSegment(
            segment_idx=fold.fold_idx,
            predict_start=fold.oos_start,
            predict_end_exclusive=fold.oos_end_exclusive,
            model_fit_end_exclusive=fold.policy_fit_start,
            max_evidence_exit_idx=fold.policy_fit_start - 1,
            is_warmup=fold.fold_idx == 0,
            batch=ValidatedSignalBatch(
                events=(),
                start_idx=fold.oos_start,
                end_idx=fold.oos_end_exclusive,
                symbols=tuple(aligned.symbols),
                registry_version="causal_tape",
                model_version="causal_tape",
            ),
        )
        segments.append(segment)
    symbols = tuple(aligned.symbols)
    return CausalAlphaTape(
        segments=tuple(segments),
        start_idx=folds[0].oos_start,
        end_idx_exclusive=folds[-1].oos_end_exclusive,
        symbols=symbols,
        fingerprint="causal_tape_v1",
    )


def validate_alpha_tape_causality(
    tape: CausalAlphaTape,
    *,
    purge_bars: int,
) -> None:
    for segment in tape.segments:
        if segment.max_evidence_exit_idx >= segment.predict_start:
            raise ValueError(
                f"evidence leakage: segment {segment.segment_idx} "
                f"max_evidence_exit_idx={segment.max_evidence_exit_idx} "
                f">= predict_start={segment.predict_start}"
            )
        model_deadline = segment.predict_start - purge_bars
        if segment.model_fit_end_exclusive > model_deadline:
            raise ValueError(
                f"model leakage: segment {segment.segment_idx} "
                f"model_fit_end_exclusive={segment.model_fit_end_exclusive} "
                f"> model_deadline={model_deadline} (predict_start={segment.predict_start} - purge_bars={purge_bars})"
            )
