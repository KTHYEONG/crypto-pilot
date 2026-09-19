"""Explicit label maturity and refit-clock evidence for process fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.backtest.contracts import ProcessMarketData
from src.mhs.backtest.market_data import apply_process_execution_availability
from src.mhs.params import PROCESS_SMOOTHING_HALFLIFE_DAYS
from src.mhs.process import RefitPoint, ema_smoothing_rate, smoothed_book_path

_logger = logging.getLogger(__name__)

RefitCutoffs = tuple[RefitPoint, ...]


@dataclass(frozen=True, slots=True)
class ProcessClockSpec:
    """Separate feature publication, forward return duration and fit latency.
    A backward feature lookback is a warmup requirement, not a forward label embargo.
    """

    decision_period: pd.Timedelta
    bar_completion_lag: pd.Timedelta
    fit_latency: pd.Timedelta


@dataclass(frozen=True, slots=True)
class MaturedMemberReturns:
    """Carry member return labels with their complete economic interval and availability.
    Knowledge and evidence source travel with returns so fitting cannot consume an
    unfinished label or promote a screening proxy into execution evidence.
    """

    returns: pd.DataFrame
    known: pd.DataFrame
    label_start: pd.DatetimeIndex
    label_end: pd.DatetimeIndex
    available_at: pd.DatetimeIndex
    source: Literal["daily_step_proxy", "inventory_3m"]
    procedure_digest: str
    input_manifest_digest: str | None


def _require_clock(clock: ProcessClockSpec) -> None:
    if (
        clock.decision_period <= pd.Timedelta(0)
        or clock.bar_completion_lag <= pd.Timedelta(0)
        or clock.fit_latency < pd.Timedelta(0)
    ):
        raise DataIntegrityError(
            "clock requires a positive decision period and completion lag"
            " with nonnegative fit latency"
        )


def _require_matured_labels(labels: MaturedMemberReturns) -> None:
    returns, known = labels.returns, labels.known
    if (
        returns.shape != known.shape
        or list(returns.columns) != list(known.columns)
        or len(set(returns.columns)) != len(returns.columns)
    ):
        raise DataIntegrityError("returns and knowledge must share identical unique member columns")
    if not (
        len(labels.label_start)
        == len(labels.label_end)
        == len(labels.available_at)
        == len(returns)
    ):
        raise DataIntegrityError("interval arrays must align exactly with return rows")
    if bool(((labels.label_start >= labels.label_end) | (labels.label_end > labels.available_at)).any()):
        raise DataIntegrityError("labels require label_start < label_end <= available_at")
    values = returns.to_numpy(dtype="float64")
    flags = known.to_numpy(dtype=bool)
    if bool((flags & ~(np.isfinite(values) & (values > -1.0))).any()):
        raise DataIntegrityError("known returns must be finite and strictly greater than minus one")
    if bool(((~flags) & ~np.isnan(values)).any()):
        raise DataIntegrityError("unknown returns must remain missing")


def build_proxy_member_returns(
    data: ProcessMarketData,
    *,
    clock: ProcessClockSpec,
    one_way_bps: float,
    procedure_digest: str,
    input_manifest_digest: str | None,
) -> MaturedMemberReturns:
    """Construct provisional daily member labels from the same causally executable
    holdings used by process targets. Smoothing may retain stale weights, so each
    decision's execution availability is re-applied before turnover, funding,
    price and maturity evidence are evaluated.

    Args:
        data: Canonical published features, candidate books and observed funding knowledge.
        clock: Registered decision period, completion assumption and fit latency.
        one_way_bps: Registered finite nonnegative one-way screening friction.
        procedure_digest: Identity binding books, smoothing, clock and screening costs.
        input_manifest_digest: Validated source identity or absent provenance.
    Returns:
        Matured labels marked daily-step proxy; unknown intervals remain unknown.
    Raises:
        DataIntegrityError: Inputs, intervals, funding knowledge or identities are inconsistent.
    """
    _require_clock(clock)
    try:
        friction = float(one_way_bps)
    except (TypeError, ValueError):
        raise DataIntegrityError(f"one_way_bps must be finite and >= 0, got {one_way_bps}") from None
    if not np.isfinite(friction) or friction < 0.0:
        raise DataIntegrityError(f"one_way_bps must be finite and >= 0, got {one_way_bps}")
    if not isinstance(procedure_digest, str) or not procedure_digest:
        raise DataIntegrityError("procedure_digest must be a nonempty identity")
    if input_manifest_digest is not None and (
        not isinstance(input_manifest_digest, str) or not input_manifest_digest
    ):
        raise DataIntegrityError("input_manifest_digest must be None or a nonempty identity")
    books = data.member_books
    if not books:
        raise DataIntegrityError("member evidence requires at least one candidate book")
    decisions = data.log_close_step.index
    if not data.funding_step.index.equals(decisions):
        raise DataIntegrityError("price and funding steps must share decision labels")
    symbols = list(data.log_close_step.columns)
    for name, book in books.items():
        if not book.index.equals(decisions) or list(book.columns) != symbols:
            raise DataIntegrityError(
                f"candidate book '{name}' must share decision labels and symbol order"
            )
    knowledge = data.funding_known_1h
    if knowledge is None or list(knowledge.columns) != symbols:
        raise DataIntegrityError("member evidence requires aligned observed funding knowledge")
    mask = data.execution_mask
    if not mask.index.equals(decisions) or list(mask.columns) != symbols:
        raise DataIntegrityError("execution_mask must share decision labels and symbol order")
    if bool((mask.dtypes.apply(lambda dt: dt.kind != "b")).any()):
        raise DataIntegrityError("execution_mask must be boolean")
    if bool(mask.isna().to_numpy().any()):
        raise DataIntegrityError("execution_mask must not be missing")
    rate = ema_smoothing_rate(PROCESS_SMOOTHING_HALFLIFE_DAYS)
    logv = data.log_close_step.to_numpy(dtype="float64")
    fundv = data.funding_step.to_numpy(dtype="float64")
    knownv = knowledge.to_numpy(dtype=bool)
    hourly = knowledge.index.to_numpy()
    stamps = decisions.to_numpy()
    label_start = decisions[:-1] + clock.bar_completion_lag
    label_end = decisions[1:] + clock.bar_completion_lag
    forward = logv[1:] - logv[:-1]
    price_move = np.where(np.isfinite(forward), np.exp(forward) - 1.0, np.nan)
    safe_price = np.where(np.isfinite(price_move), price_move, 0.0)
    safe_fund = np.where(np.isfinite(fundv[:-1]), fundv[:-1], 0.0)
    cost_rate = friction * 1e-4
    members: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    for name, book in books.items():
        smoothed = smoothed_book_path(book, pd.Series(rate, index=book.index))
        executable = apply_process_execution_availability(smoothed, mask)
        entry = executable.to_numpy(dtype="float64")[:-1]
        held = entry != 0.0
        usable = ((~held) | np.isfinite(price_move)).all(axis=1) & ((~held) | np.isfinite(fundv[:-1])).all(axis=1)
        prev = np.vstack([np.zeros((1, entry.shape[1])), entry[:-1]])
        turnover = np.abs(entry - prev).sum(axis=1)
        raw = (entry * safe_price).sum(axis=1) - (entry * safe_fund).sum(axis=1) - cost_rate * turnover
        row = np.where(usable, raw, np.nan)
        mature = np.ones(len(row), dtype=bool)
        for pos in range(len(row)):
            left = int(np.searchsorted(hourly, stamps[pos], side="right"))
            right = int(np.searchsorted(hourly, stamps[pos + 1], side="right"))
            held_cols = np.flatnonzero(held[pos])
            if held_cols.size:
                mature[pos] = bool(knownv[left:right, held_cols].all())
        mature = mature & np.isfinite(row) & (row > -1.0)
        members[name] = np.where(mature, row, np.nan)
        flags[name] = mature
    returns = pd.DataFrame(members, index=decisions[:-1])
    known = pd.DataFrame(flags, index=decisions[:-1])
    return MaturedMemberReturns(
        returns=returns,
        known=known,
        label_start=label_start,
        label_end=label_end,
        available_at=label_end,
        source="daily_step_proxy",
        procedure_digest=procedure_digest,
        input_manifest_digest=input_manifest_digest,
    )


def select_matured_training_returns(
    labels: MaturedMemberReturns,
    *,
    fit_cutoff: pd.Timestamp,
    train_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Select complete common member evidence that was available before fitting.

    Args:
        labels: Registered member labels with explicit intervals and knowledge.
        fit_cutoff: UTC fitting information cutoff, before the resulting application.
        train_start: Optional inclusive lower bound on the label's exposure start.
    Returns:
        Rows whose full labels are known, mature and within the declared train window.
    Raises:
        DataIntegrityError: Labels or window bounds violate causality or member alignment.
    """
    _require_matured_labels(labels)
    if not isinstance(fit_cutoff, pd.Timestamp) or fit_cutoff.tz is None or str(fit_cutoff.tz) != "UTC":
        raise DataIntegrityError("fit_cutoff must be a timezone-aware UTC timestamp")
    if train_start is not None and (
        not isinstance(train_start, pd.Timestamp) or train_start.tz is None or str(train_start.tz) != "UTC"
    ):
        raise DataIntegrityError("train_start must be a timezone-aware UTC timestamp")
    if train_start is not None and train_start > fit_cutoff:
        raise DataIntegrityError("train_start must not be later than fit_cutoff")
    common = labels.known.to_numpy(dtype=bool).all(axis=1)
    timely = np.asarray(labels.available_at <= fit_cutoff, dtype=bool) & np.asarray(
        labels.label_end <= fit_cutoff, dtype=bool
    )
    mask = common & timely
    if train_start is not None:
        mask = mask & np.asarray(labels.label_start >= train_start, dtype=bool)
    picked = labels.returns.loc[mask]
    _logger.info(
        "[DATA] stage=matured_training_selected rows=%d excluded=%d cutoff=%s",
        int(mask.sum()),
        int(len(mask) - mask.sum()),
        fit_cutoff,
    )
    return picked
