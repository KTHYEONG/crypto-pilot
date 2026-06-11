from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(slots=True, frozen=True)
class WFFold:
    """Single walk-forward fold with purged fit / calibration / OOS windows."""

    fit_start: int
    fit_end: int
    cal_start: int
    cal_end: int
    oos_start: int
    oos_end: int


@dataclass(slots=True, frozen=True)
class CPCVFold:
    """Combinatorial Purged CV fold.

    Attributes:
        fit_groups: Indices of groups used for fitting.
        test_groups: Indices of groups used for out-of-sample testing.
        fit_spans: Purge/embargo-adjusted fit bar spans as (start, end) tuples.
        test_spans: OOS bar spans as (start, end) tuples.
    """

    fit_groups: tuple[int, ...]
    test_groups: tuple[int, ...]
    fit_spans: tuple[tuple[int, int], ...]
    test_spans: tuple[tuple[int, int], ...]


def build_cpcv_folds(
    *,
    n_bars: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_bars: int,
    purge_bars: int,
) -> tuple[CPCVFold, ...]:
    """Build Combinatorial Purged Cross-Validation folds.

    Partitions ``n_bars`` into ``n_groups`` equal-sized groups and enumerates
    all C(n_groups, n_test_groups) combinations as test sets.  For each
    combination the adjacent fit groups are trimmed by ``purge_bars`` (on the
    leading edge of a test group) and ``embargo_bars`` (on the trailing edge
    of a test group) to prevent information leakage.

    Args:
        n_bars: Total number of bars in the dataset.
        n_groups: Number of equally-spaced groups to partition bars into.
        n_test_groups: Number of groups held out as OOS per fold.
        embargo_bars: Bars to remove from the *start* of a fit group that
            immediately follows a test group (post-test embargo).
        purge_bars: Bars to remove from the *end* of a fit group that
            immediately precedes a test group (pre-test purge).

    Returns:
        Tuple of :class:`CPCVFold` instances.  Falls back to a single
        degenerate fold when no valid folds can be constructed.

    Time Complexity:  O(C(n_groups, n_test_groups) * n_groups)
    Space Complexity: O(C(n_groups, n_test_groups))
    """
    # --- degenerate guard -----------------------------------------------
    def _single_fallback() -> tuple[CPCVFold, ...]:
        return (CPCVFold(
            fit_groups=(0,),
            test_groups=(),
            fit_spans=((0, n_bars),),
            test_spans=(),
        ),)

    if n_test_groups >= n_groups or n_groups < 1 or n_bars < 1:
        return _single_fallback()

    # --- group boundary computation [T, 2] ------------------------------
    bounds: np.ndarray = np.linspace(0, n_bars, n_groups + 1).astype(int)
    # group_spans[i] = (start_bar, end_bar)  shape: (n_groups, 2)
    group_spans: list[tuple[int, int]] = [
        (int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)
    ]

    folds: list[CPCVFold] = []
    test_group_set: set[int]

    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_groups: tuple[int, ...] = combo  # already sorted
        test_group_set = set(test_groups)
        fit_groups: tuple[int, ...] = tuple(
            i for i in range(n_groups) if i not in test_group_set
        )
        test_spans: tuple[tuple[int, int], ...] = tuple(
            group_spans[i] for i in test_groups
        )

        # --- purge/embargo per fit group --------------------------------
        fit_span_list: list[tuple[int, int]] = []
        for i in fit_groups:
            gs, ge = group_spans[i]

            # preceding group (i-1) is a test group → apply embargo to start
            if (i - 1) in test_group_set:
                gs = group_spans[i - 1][1] + embargo_bars

            # following group (i+1) is a test group → apply purge to end
            if (i + 1) in test_group_set:
                ge = group_spans[i + 1][0] - purge_bars

            if gs < ge:  # skip degenerate (fully consumed) spans
                fit_span_list.append((gs, ge))

        if not fit_span_list:
            continue  # no usable fit data → skip this combination

        folds.append(CPCVFold(
            fit_groups=fit_groups,
            test_groups=test_groups,
            fit_spans=tuple(fit_span_list),
            test_spans=test_spans,
        ))

    return tuple(folds) if folds else _single_fallback()


def build_walk_forward_folds(
    *,
    n_bars: int,
    cfg: CandidateStrategyConfig,
    max_holding_bars: int | None = None,
) -> tuple[WFFold, ...]:
    """Build purged + embargoed anchored/rolling WF folds.

    anchored: fit_start=0 fixed; OOS window advances n_folds slices.
    rolling : fit window length fixed, moves forward.
    single  : wraps _candidate_ml_split_indices for backward-compat.

    Purge gap is inserted between fit_end and cal_start.
    Embargo gap is inserted between cal_end and oos_start.
    """
    from src.domain.futures.strategy.config import resolve_purge_and_embargo_bars, with_max_holding_bars
    from src.domain.futures.strategy_runtime.bridge import _candidate_ml_split_indices

    resolved_cfg = with_max_holding_bars(cfg, max_holding_bars=max_holding_bars)
    purge, embargo = resolve_purge_and_embargo_bars(resolved_cfg)
    n_folds = cfg.wf_n_folds
    scheme = cfg.wf_scheme

    def _single_fold() -> tuple[WFFold, ...]:
        try:
            fit_s, fit_e, cal_s, cal_e, oos_s, oos_e = _candidate_ml_split_indices(
                n_bars=n_bars,
                fit_fraction=cfg.ml_fit_fraction,
                calibration_fraction=cfg.ml_calibration_fraction,
                purge_bars=purge,
                embargo_bars=embargo,
            )
        except ValueError:
            # Not enough bars — return a degenerate fold covering what we have
            fit_e = max(1, int(n_bars * 0.6))
            cal_e = max(fit_e + 1, int(n_bars * 0.8))
            return (WFFold(
                fit_start=0, fit_end=fit_e,
                cal_start=fit_e, cal_end=cal_e,
                oos_start=cal_e, oos_end=n_bars,
            ),)
        return (WFFold(fit_start=fit_s, fit_end=fit_e, cal_start=cal_s, cal_end=cal_e, oos_start=oos_s, oos_end=oos_e),)

    if scheme == "single":
        return _single_fold()

    # Determine global OOS region (mirrors single-split OOS start)
    total_fraction = cfg.ml_fit_fraction + cfg.ml_calibration_fraction
    global_oos_start = int(n_bars * total_fraction) + embargo
    global_oos_end = n_bars

    if global_oos_start >= global_oos_end or n_folds < 1:
        return _single_fold()

    oos_len = global_oos_end - global_oos_start
    base_fold_len = oos_len // n_folds

    if base_fold_len < 1:
        return _single_fold()

    # Calibration fraction relative to fit+cal combined
    cal_frac = cfg.ml_calibration_fraction / max(total_fraction, 1e-9)

    folds: list[WFFold] = []
    for k in range(n_folds):
        oos_s = global_oos_start + k * base_fold_len
        oos_e = global_oos_start + (k + 1) * base_fold_len if k < n_folds - 1 else global_oos_end

        if oos_s >= oos_e:
            continue

        # Calibration window ends just before embargo before OOS
        cal_e = oos_s - embargo
        if cal_e <= 0:
            continue

        # Calibration length = fraction of available pre-OOS region
        available = cal_e
        cal_len = max(1, int(available * cal_frac))
        cal_s = cal_e - cal_len

        if scheme == "anchored":
            fit_s = 0
        else:  # rolling
            fit_len = max(1, int(available * (1.0 - cal_frac)))
            fit_s = max(0, cal_s - purge - fit_len)

        fit_e = cal_s - purge

        if fit_s < 0 or fit_e <= fit_s or cal_s >= cal_e:
            continue

        folds.append(WFFold(
            fit_start=fit_s,
            fit_end=fit_e,
            cal_start=cal_s,
            cal_end=cal_e,
            oos_start=oos_s,
            oos_end=oos_e,
        ))

    return tuple(folds) if folds else _single_fold()
