from __future__ import annotations

import numpy as np
import pandas as pd


def detect_internal_gaps(
    valid: pd.DataFrame,
    min_useful_bars: int = 720,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]]:
    """Report per-symbol internal gaps (False runs) between first and last valid bar.

    Read-only diagnostic: ``valid`` is never mutated and no rows are ever
    deleted -- the pipeline's ``load_base_panel`` / ``liquid_half_eligibility`` /
    ``horizon_log_return`` already tolerate NaN gaps, so no destructive trim
    utility should ever be reintroduced. A gap is a run of False strictly
    between a symbol's own first and last True; leading/trailing False are edge
    padding and are not reported. Symbols with no internal gap are absent from
    the result, never present with an empty list.

    ``min_useful_bars`` is accepted for API compatibility with the historical
    ``MIN_USEFUL_BARS=720`` threshold semantics but does NOT filter any gap:
    every internal gap is reported so a human reviewer sees full information.
    Filtering / severity judgment is a caller concern.

    Each reported tuple is ``(gap_start, gap_end, gap_length_bars)`` using the
    DataFrame's own DatetimeIndex values (inclusive of both endpoints), where
    ``gap_length_bars`` is the count of False bars in that run.
    """
    if not isinstance(valid, pd.DataFrame):
        raise ValueError("valid must be a DataFrame of boolean dtype")
    if not (valid.dtypes == np.dtype("bool")).all():
        raise ValueError("valid must be a DataFrame of boolean dtype")

    index = valid.index
    gaps: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
    for symbol in valid.columns:
        column = valid[symbol].to_numpy(dtype=bool)
        true_positions = np.flatnonzero(column)
        if true_positions.size == 0:
            continue
        first_true = int(true_positions[0])
        last_true = int(true_positions[-1])
        interior = column[first_true : last_true + 1]
        # Vectorized run-length encoding of contiguous False runs between the
        # symbol's own first/last True (same diff-based approach as
        # src/mhs/evaluation.py): a zero-padded diff marks each run start
        # (+1) and end (-1).
        padded = np.concatenate(([0], (~interior).astype(np.int8), [0]))
        deltas = np.diff(padded)
        starts = np.flatnonzero(deltas == 1)
        ends = np.flatnonzero(deltas == -1)
        if starts.size == 0:
            continue
        spans = [
            (
                index[first_true + int(start)],
                index[first_true + int(end) - 1],
                int(end - start),
            )
            for start, end in zip(starts, ends, strict=True)
        ]
        gaps[symbol] = spans
    return gaps
