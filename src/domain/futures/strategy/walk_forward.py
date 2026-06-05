from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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


def build_walk_forward_folds(
    *,
    n_bars: int,
    cfg: CandidateStrategyConfig,
) -> tuple[WFFold, ...]:
    """Build purged + embargoed anchored/rolling WF folds.

    anchored: fit_start=0 fixed; OOS window advances n_folds slices.
    rolling : fit window length fixed, moves forward.
    single  : wraps _candidate_ml_split_indices for backward-compat.

    Purge gap is inserted between fit_end and cal_start.
    Embargo gap is inserted between cal_end and oos_start.
    """
    from src.domain.futures.strategy_runtime.bridge import _candidate_ml_split_indices

    purge = cfg.purge_bars
    embargo = cfg.embargo_bars
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
