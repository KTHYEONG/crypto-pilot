from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

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

    .. deprecated::
        build_l1_swf_folds로 교체됨. CPCV는 disjoint OOS collapse 버그로 인해 L1 검증에 부적합.
        기존 테스트 호환성을 위해 함수 본체는 유지됨.

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
        return (
            CPCVFold(
                fit_groups=(0,),
                test_groups=(),
                fit_spans=((0, n_bars),),
                test_spans=(),
            ),
        )

    if n_test_groups >= n_groups or n_groups < 1 or n_bars < 1:
        return _single_fallback()

    # --- group boundary computation [T, 2] ------------------------------
    bounds: np.ndarray = np.linspace(0, n_bars, n_groups + 1).astype(int)
    # group_spans[i] = (start_bar, end_bar)  shape: (n_groups, 2)
    group_spans: list[tuple[int, int]] = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_groups)]

    folds: list[CPCVFold] = []
    test_group_set: set[int]

    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_groups: tuple[int, ...] = combo  # already sorted
        test_group_set = set(test_groups)
        fit_groups: tuple[int, ...] = tuple(i for i in range(n_groups) if i not in test_group_set)
        test_spans: tuple[tuple[int, int], ...] = tuple(group_spans[i] for i in test_groups)

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

        folds.append(
            CPCVFold(
                fit_groups=fit_groups,
                test_groups=test_groups,
                fit_spans=tuple(fit_span_list),
                test_spans=test_spans,
            )
        )

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
            return (
                WFFold(
                    fit_start=0,
                    fit_end=fit_e,
                    cal_start=fit_e,
                    cal_end=cal_e,
                    oos_start=cal_e,
                    oos_end=n_bars,
                ),
            )
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

        folds.append(
            WFFold(
                fit_start=fit_s,
                fit_end=fit_e,
                cal_start=cal_s,
                cal_end=cal_e,
                oos_start=oos_s,
                oos_end=oos_e,
            )
        )

    return tuple(folds) if folds else _single_fold()


def build_l1_swf_folds(
    *,
    n_bars: int,
    n_folds: int = 5,
    l1_start_bars: int,
    l1_end_bars: int,
    purge_bars: int,
    embargo_bars: int,
    cal_fraction: float = 0.15,
    boundary_mode: Literal["exact_label_interval", "fixed_gap"] = "exact_label_interval",
    allocation_backend: Literal["ensemble_b0", "ml_edge"] = "ensemble_b0",
) -> tuple[WFFold, ...]:
    """L1 신호 검증용 Purged Sequential Walk-Forward folds.

    [l1_start_bars, l1_end_bars)를 initial-train 1개 + OOS n_folds개 block으로 분할한다.
    fit은 l1_start_bars 기점 expanding이며 pre-L1 bar는 feature warm-up 전용으로만 사용한다.

    Args:
        n_bars: aligned.datetimes 전체 길이.
        n_folds: OOS 창 수 (기본 5).
        l1_start_bars: L1 학습 시작 bar index.
        l1_end_bars: production OOS start bar index
            (searchsorted(aligned.datetimes, oos_start_ts)).
        purge_bars: OOS 직전 fit에서 제거할 bar 수 (leakage 방지).
        embargo_bars: 미사용 (시그니처 일관성 유지용).
        cal_fraction: fit 구간 후반 calibration 비율.

    Returns:
        WFFold 튜플.

    Time Complexity: O(n_folds)
    Space Complexity: O(n_folds)
    """
    _ = embargo_bars  # exact_label_interval에서는 미사용, fixed_gap에서만 purge 사용
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if l1_start_bars < 0 or l1_end_bars > n_bars or l1_start_bars >= l1_end_bars:
        raise ValueError(
            f"invalid L1 bar range: l1_start_bars={l1_start_bars}, l1_end_bars={l1_end_bars}, n_bars={n_bars}"
        )

    available: int = l1_end_bars - l1_start_bars
    block_len: int = available // (n_folds + 1)
    if block_len < 1:
        raise ValueError(f"insufficient bars for L1 SWF blocks: available={available}, n_folds={n_folds}")

    folds: list[WFFold] = []
    for k in range(n_folds):
        oos_s = l1_start_bars + (k + 1) * block_len
        oos_e = l1_start_bars + (k + 2) * block_len if k < n_folds - 1 else l1_end_bars
        fit_e = oos_s if boundary_mode == "exact_label_interval" else oos_s - purge_bars
        if fit_e <= l1_start_bars:
            raise ValueError(
                f"insufficient fit span for L1 SWF fold: fold={k}, fit_end={fit_e}, l1_start_bars={l1_start_bars}"
            )
        if oos_e <= oos_s:
            raise ValueError(f"invalid OOS span for L1 SWF fold: fold={k}, oos_start={oos_s}, oos_end={oos_e}")
        if boundary_mode == "exact_label_interval" and allocation_backend == "ensemble_b0":
            fit_e = oos_s
            cal_s = oos_s
            cal_end = oos_s
        else:
            fit_len = fit_e - l1_start_bars
            cal_len = max(1, int(fit_len * cal_fraction))
            cal_s = fit_e - cal_len
            cal_end = oos_s if boundary_mode == "exact_label_interval" else fit_e
        folds.append(
            WFFold(
                fit_start=l1_start_bars,
                fit_end=fit_e,
                cal_start=cal_s,
                cal_end=cal_end,
                oos_start=oos_s,
                oos_end=oos_e,
            )
        )

    return tuple(folds)


def build_l1_nested_swf_folds(
    *,
    n_bars: int,
    l1_start_idx: int,
    l1_end_idx: int,
    max_label_horizon_bars: int,
    cfg: CandidateStrategyConfig,
) -> tuple[WFFold, ...]:
    """Build outer anchored folds for nested Layer1 SWF validation.

    The first block is reserved as initial train history and the remaining
    blocks become sequential non-overlapping outer OOS windows.
    """
    if l1_start_idx < 0 or l1_end_idx > n_bars or l1_start_idx >= l1_end_idx:
        raise ValueError(
            f"invalid L1 nested bar range: l1_start_idx={l1_start_idx}, l1_end_idx={l1_end_idx}, n_bars={n_bars}"
        )
    n_folds = max(1, int(getattr(cfg, "wf_n_folds", 1)))
    warmup = max(1, int(getattr(cfg, "l1_outer_warmup_blocks", 2)))
    available = l1_end_idx - l1_start_idx
    block_len = available // (n_folds + warmup)
    if block_len < 1:
        raise ValueError(
            f"insufficient bars for nested L1 SWF blocks: available={available}, n_folds={n_folds}, warmup={warmup}"
        )
    purge_cfg = int(getattr(cfg, "purge_bars", 0) or 0)
    embargo_cfg = int(getattr(cfg, "embargo_bars", 0) or 0)
    boundary_mode = cast(
        Literal["exact_label_interval", "fixed_gap"],
        getattr(cfg, "l1_boundary_mode", "exact_label_interval"),
    )
    allocation_backend = cast(
        Literal["ensemble_b0", "ml_edge"],
        getattr(cfg, "allocation_backend", "ensemble_b0"),
    )
    purge_bars = max(int(max_label_horizon_bars), purge_cfg)
    embargo_bars = max(0, embargo_cfg)
    cal_fraction = float(getattr(cfg, "ml_calibration_fraction", 0.2))

    folds: list[WFFold] = []
    for fold_idx in range(n_folds):
        oos_start = l1_start_idx + (fold_idx + warmup) * block_len
        oos_end = l1_start_idx + (fold_idx + warmup + 1) * block_len if fold_idx < n_folds - 1 else l1_end_idx
        outer_fit_end = oos_start if boundary_mode == "exact_label_interval" else oos_start - purge_bars
        if outer_fit_end <= l1_start_idx:
            raise ValueError(
                "insufficient fit span for nested L1 fold: "
                f"fold={fold_idx}, fit_end={outer_fit_end}, l1_start_idx={l1_start_idx}"
            )
        if boundary_mode == "exact_label_interval":
            if allocation_backend == "ensemble_b0":
                fit_train_end = outer_fit_end
                cal_start = outer_fit_end
                cal_end = outer_fit_end
            else:
                fit_len = outer_fit_end - l1_start_idx
                cal_len = max(1, int(fit_len * cal_fraction))
                cal_start = max(l1_start_idx + 1, outer_fit_end - cal_len)
                cal_end = outer_fit_end
                fit_train_end = cal_start
        else:
            cal_end = outer_fit_end
            fit_len = outer_fit_end - l1_start_idx
            cal_len = max(1, int(fit_len * cal_fraction))
            cal_start = max(l1_start_idx + 1, cal_end - cal_len)
            fit_train_end = max(l1_start_idx + 1, cal_start - embargo_bars)
            if fit_train_end <= l1_start_idx:
                fit_train_end = max(l1_start_idx + 1, cal_start)
        if oos_end <= oos_start:
            raise ValueError(f"invalid nested L1 OOS span: fold={fold_idx}, oos_start={oos_start}, oos_end={oos_end}")
        folds.append(
            WFFold(
                fit_start=l1_start_idx,
                fit_end=fit_train_end,
                cal_start=cal_start,
                cal_end=cal_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
    return tuple(folds)


def build_l2_simulation_folds(
    *,
    n_bars: int,
    l2_start_idx: int,
    holdout_start_idx: int,
    cfg: CandidateStrategyConfig,
) -> tuple[WFFold, ...]:
    """Build L2 AWF folds with config-driven purge/embargo gaps."""
    if l2_start_idx < 0 or holdout_start_idx > n_bars or l2_start_idx >= holdout_start_idx:
        raise ValueError(
            "invalid L2 simulation bar range: "
            f"l2_start_idx={l2_start_idx}, holdout_start_idx={holdout_start_idx}, n_bars={n_bars}"
        )
    sim_cfg = cfg
    folds = build_walk_forward_folds(n_bars=holdout_start_idx, cfg=sim_cfg)
    filtered = tuple(fold for fold in folds if fold.oos_start >= l2_start_idx and fold.oos_end <= holdout_start_idx)
    if filtered:
        return filtered
    cal_end = max(l2_start_idx - 1, 1)
    return (
        WFFold(
            fit_start=0,
            fit_end=cal_end,
            cal_start=max(0, cal_end - max(1, cal_end // 5)),
            cal_end=cal_end,
            oos_start=l2_start_idx,
            oos_end=holdout_start_idx,
        ),
    )


def resolve_l2_fold_cfg(
    cfg: CandidateStrategyConfig,
    l2_wf_n_folds: int | None,
) -> CandidateStrategyConfig:
    """[ADR_20260718_L2_FOLD_GRANULARITY_ROBUSTNESS] L2 전용 walk-forward fold 개수 override.

    L1(build_l1_swf_folds, cfg 미의존)·live 실행·ablation 등 cfg.wf_n_folds를
    공유하는 다른 소비처는 이 함수를 거치지 않으므로 영향받지 않는다.
    l2_wf_n_folds가 None이거나 cfg.wf_n_folds와 동일하면 cfg를 그대로 반환한다
    (불필요한 객체 생성 방지, 완전한 하위호환).
    """
    if l2_wf_n_folds is None or int(l2_wf_n_folds) == int(cfg.wf_n_folds):
        return cfg
    if int(l2_wf_n_folds) < 2:
        raise ValueError(f"l2_wf_n_folds must be >= 2, got {l2_wf_n_folds}")
    return replace(cfg, wf_n_folds=int(l2_wf_n_folds))



@dataclass(slots=True, frozen=True)
class CausalL2Fold:
    fold_idx: int
    policy_fit_start: int
    policy_fit_end_exclusive: int
    oos_start: int
    oos_end_exclusive: int


def build_causal_l2_folds(
    *,
    n_bars: int,
    l2_start_idx: int,
    holdout_start_idx: int,
    n_folds: int,
    min_warmup_bars: int,
) -> tuple[CausalL2Fold, ...]:
    l2_span = holdout_start_idx - l2_start_idx
    warmup_bars = max(min_warmup_bars, l2_span // 4)
    remaining = l2_span - warmup_bars
    if remaining < n_folds or l2_span < min_warmup_bars + n_folds:
        raise ValueError(
            f"insufficient L2 span: l2_span={l2_span}, warmup_bars={warmup_bars}, "
            f"remaining={remaining}, n_folds={n_folds}"
        )
    fold_size = remaining // n_folds
    folds: list[CausalL2Fold] = []
    for i in range(n_folds):
        oos_start = l2_start_idx + warmup_bars + i * fold_size
        oos_end = oos_start + (fold_size if i < n_folds - 1 else remaining - (n_folds - 1) * fold_size)
        policy_fit_start = l2_start_idx
        policy_fit_end = oos_start
        folds.append(
            CausalL2Fold(
                fold_idx=i,
                policy_fit_start=policy_fit_start,
                policy_fit_end_exclusive=policy_fit_end,
                oos_start=oos_start,
                oos_end_exclusive=oos_end,
            )
        )
    return tuple(folds)
