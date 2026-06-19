"""Execution eligibility engine for PIT universe rewrite.

Provides three public functions:
  - resolve_execution_rules: asof-join rule history to a decision timestamp.
  - evaluate_execution_eligibility: hard-gate evaluation for all instruments.
  - build_universe_state_cube: dense TxN eligibility cube over a calendar.

Time Complexity: O(T*N) for cube construction.
Space Complexity: O(T*N) for dense bool/float arrays.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.domain.futures.universe.contracts import (
    DataConfidence,
    EligibilityCode,
    EligibilityReason,
    EligibilitySnapshot,
    ExecutionEligibility,
    ExecutionRules,
    InstrumentRecord,  # noqa: F401 - re-exported for callers
    UniverseStateCube,
)

__all__ = [
    "ExecutionEligibilityConfig",
    "RuleFallbackPolicy",
    "build_universe_state_cube",
    "evaluate_execution_eligibility",
    "resolve_execution_rules",
]

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DataConfidence ordering (OBSERVED > RECONSTRUCTED > UNKNOWN)
# ---------------------------------------------------------------------------
_CONFIDENCE_LEVEL: dict[DataConfidence, int] = {
    DataConfidence.OBSERVED: 2,
    DataConfidence.RECONSTRUCTED: 1,
    DataConfidence.UNKNOWN: 0,
}


def _confidence_ge(a: DataConfidence, b: DataConfidence) -> bool:
    """Return True if confidence ``a`` is at least as good as ``b``."""
    return _CONFIDENCE_LEVEL[a] >= _CONFIDENCE_LEVEL[b]


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleFallbackPolicy:
    """Conservative fallback values when historical execution rules are missing.

    Attributes:
        allow_reconstructed: If True, return conservative values instead of
            raising when no historical rule row is found.
        conservative_tick_size: Fallback tick size in quote units.
        conservative_step_size: Fallback quantity step size.
        conservative_min_qty: Fallback minimum order quantity.
        conservative_min_notional: Fallback minimum notional in USDT.
        conservative_taker_fee_bps: Fallback taker fee in basis points.
    """

    allow_reconstructed: bool = True
    conservative_tick_size: float = 0.01
    conservative_step_size: float = 0.001
    conservative_min_qty: float = 0.001
    conservative_min_notional: float = 10.0
    conservative_taker_fee_bps: float = 5.0


@dataclass(frozen=True, slots=True)
class ExecutionEligibilityConfig:
    """Configuration for the execution eligibility evaluation.

    Attributes:
        max_staleness_bars: Maximum number of bars a metric value may be stale.
        min_metric_observations: Minimum observation count before a metric is
            considered valid (guards against INSUFFICIENT_OBSERVATIONS).
        max_round_trip_cost_bps: Hard ceiling on estimated round-trip cost.
        max_participation_rate: Maximum fraction of ADV30 for capacity calc.
        min_data_confidence: Minimum acceptable DataConfidence for instruments.
        default_intended_notional_usdt: Bootstrap notional for eligibility
            evaluation; overridden by actual L2 target at execution time.
    """

    max_staleness_bars: int = 1
    min_metric_observations: int = 20
    max_round_trip_cost_bps: float = 50.0
    max_participation_rate: float = 0.01
    min_data_confidence: DataConfidence = DataConfidence.RECONSTRUCTED
    default_intended_notional_usdt: float = 10_000.0


# ---------------------------------------------------------------------------
# Function 1: resolve_execution_rules
# ---------------------------------------------------------------------------


def resolve_execution_rules(
    instrument_id: str,
    *,
    decision_at: datetime,
    rule_history: pd.DataFrame,
    fallback_policy: RuleFallbackPolicy,
) -> ExecutionRules:
    """Return the most recent execution rules known at ``decision_at``.

    Performs a point-in-time asof join: filters rows where
    ``instrument_id`` matches and ``available_at <= decision_at``, then
    returns the row with the latest ``available_at``.

    Args:
        instrument_id: Unique contract identifier.
        decision_at: The decision timestamp (UTC-aware or naive consistent
            with rule_history).
        rule_history: DataFrame with columns
            [instrument_id, available_at, tick_size, step_size, min_qty,
             min_notional, taker_fee_bps, confidence].
        fallback_policy: Policy controlling conservative fallback behaviour
            when no historical rows are found.

    Returns:
        ExecutionRules instance populated from the latest valid row, or from
        conservative fallback values when permitted.

    Raises:
        RuntimeError: If no rows exist and ``fallback_policy.allow_reconstructed``
            is False.

    Time Complexity: O(R) where R = rows for instrument_id.
    Space Complexity: O(R) filtered subset.
    """
    mask = (rule_history["instrument_id"] == instrument_id) & (
        rule_history["available_at"] <= decision_at
    )
    subset = rule_history.loc[mask]

    if subset.empty:
        if not fallback_policy.allow_reconstructed:
            raise RuntimeError(
                f"MISSING_RULES: no execution rules for {instrument_id} at {decision_at}"
            )
        _LOG.warning(
            "resolve_execution_rules: no rows for %s at %s; using conservative fallback",
            instrument_id,
            decision_at,
        )
        return ExecutionRules(
            instrument_id=instrument_id,
            decision_at=decision_at,
            tick_size=fallback_policy.conservative_tick_size,
            step_size=fallback_policy.conservative_step_size,
            min_qty=fallback_policy.conservative_min_qty,
            min_notional=fallback_policy.conservative_min_notional,
            taker_fee_bps=fallback_policy.conservative_taker_fee_bps,
            tick_size_confidence=DataConfidence.RECONSTRUCTED,
            step_size_confidence=DataConfidence.RECONSTRUCTED,
            min_qty_confidence=DataConfidence.RECONSTRUCTED,
            min_notional_confidence=DataConfidence.RECONSTRUCTED,
            taker_fee_confidence=DataConfidence.RECONSTRUCTED,
        )

    row = subset.sort_values("available_at").iloc[-1]
    confidence_val = DataConfidence(row["confidence"]) if "confidence" in row.index else DataConfidence.OBSERVED
    return ExecutionRules(
        instrument_id=instrument_id,
        decision_at=decision_at,
        tick_size=float(row["tick_size"]),
        step_size=float(row["step_size"]),
        min_qty=float(row["min_qty"]),
        min_notional=float(row["min_notional"]),
        taker_fee_bps=float(row["taker_fee_bps"]),
        tick_size_confidence=confidence_val,
        step_size_confidence=confidence_val,
        min_qty_confidence=confidence_val,
        min_notional_confidence=confidence_val,
        taker_fee_confidence=confidence_val,
    )


# ---------------------------------------------------------------------------
# Internal helpers for evaluate_execution_eligibility
# ---------------------------------------------------------------------------


def _extract_metric(
    obs_pivot: pd.DataFrame,
    instrument_id: str,
    metric: str,
) -> float:
    """Return metric value or NaN if missing."""
    if instrument_id not in obs_pivot.index:
        return float("nan")
    if metric not in obs_pivot.columns:
        return float("nan")
    val = obs_pivot.at[instrument_id, metric]
    return float(val) if pd.notna(val) else float("nan")


def _compute_impact_bps(intended_notional: float, adv30: float) -> float:
    """Estimate market-impact in basis points using square-root model.

    Formula: ``impact_bps = 18.0 * sqrt(intended_notional / max(ADV30, eps))``

    Args:
        intended_notional: Intended order size in USDT.
        adv30: 30-day average daily volume in USDT.

    Returns:
        Estimated impact in basis points (float64).
    """
    return 18.0 * math.sqrt(intended_notional / max(adv30, 1e-12))


def _compute_round_trip_cost(
    intended_notional: float,
    adv30: float,
    taker_fee_bps: float,
) -> float:
    """Estimate total round-trip cost in basis points.

    Formula:
        ``cost = 2*(taker_fee_bps + 1.0) + 2*impact_bps``

    Args:
        intended_notional: Order size in USDT.
        adv30: 30-day ADV in USDT.
        taker_fee_bps: Taker fee in basis points.

    Returns:
        Round-trip cost estimate in basis points.
    """
    impact = _compute_impact_bps(intended_notional, adv30)
    return 2.0 * (taker_fee_bps + 1.0) + 2.0 * impact


def _capacity_from_cost(
    adv30: float,
    taker_fee_bps: float,
    max_cost_bps: float,
) -> float:
    """Solve cost equation for the maximum notional satisfying the cost cap.

    From: ``max_cost_bps = 2*(fee+1) + 2*18*sqrt(q/adv)``
    =>   ``sqrt(q/adv) = (max_cost - 2*(fee+1)) / 36``
    =>   ``q = adv * ((max_cost - 2*(fee+1)) / 36)**2``

    Returns 0.0 when the fixed fee already exceeds the cap.
    """
    fixed_cost = 2.0 * (taker_fee_bps + 1.0)
    slack = max_cost_bps - fixed_cost
    if slack <= 0.0:
        return 0.0
    return adv30 * (slack / 36.0) ** 2


# ---------------------------------------------------------------------------
# Function 2: evaluate_execution_eligibility
# ---------------------------------------------------------------------------


def evaluate_execution_eligibility(
    *,
    decision_at: datetime,
    instruments: pd.DataFrame,
    observations: pd.DataFrame,
    rules: Mapping[str, ExecutionRules],
    intended_notional_usdt: Mapping[str, float],
    config: ExecutionEligibilityConfig,
) -> EligibilitySnapshot:
    """Evaluate execution eligibility for all tracked instruments at ``decision_at``.

    Applies hard gates in strict order; the first failing gate short-circuits
    further evaluation for that instrument.

    Args:
        decision_at: The decision timestamp. All ``available_at`` values in
            ``observations`` must be <= this value (PIT guard).
        instruments: DataFrame with columns
            [instrument_id, status, available_at, confidence].
        observations: DataFrame with columns
            [instrument_id, metric, available_at, value, source, confidence].
            Every row must satisfy ``available_at <= decision_at``.
        rules: Mapping from instrument_id to ExecutionRules resolved for this
            decision timestamp.
        intended_notional_usdt: Mapping from instrument_id to the intended
            order size in USDT used for cost and rounding evaluation.
        config: Evaluation configuration.

    Returns:
        EligibilitySnapshot for ``decision_at`` containing one
        ExecutionEligibility per tracked instrument.

    Raises:
        RuntimeError: If any observation has ``available_at > decision_at``
            (PIT boundary violation).

    Time Complexity: O(I*G) where I = instruments, G = number of gates (const).
    Space Complexity: O(I) for eligibility results.
    """
    # ------------------------------------------------------------------
    # PIT guard: reject any future observations
    # ------------------------------------------------------------------
    if not observations.empty:
        future_mask = observations["available_at"] > decision_at
        if future_mask.any():
            _LOG.error(
                "PIT violation: %d observations have available_at > %s",
                int(future_mask.sum()),
                decision_at,
            )
            raise RuntimeError("PIT observation boundary violated")

    # ------------------------------------------------------------------
    # Build pivot: instrument_id x metric -> latest value
    # ------------------------------------------------------------------
    obs_pivot: pd.DataFrame
    if observations.empty:
        obs_pivot = pd.DataFrame()
    else:
        # Take the latest available value per (instrument_id, metric)
        latest_obs = (
            observations.sort_values("available_at")
            .groupby(["instrument_id", "metric"], sort=False)
            .last()
            .reset_index()
        )
        obs_pivot = latest_obs.pivot(
            index="instrument_id", columns="metric", values="value"
        )

    eligibilities: list[ExecutionEligibility] = []

    for _, inst_row in instruments.iterrows():
        iid: str = str(inst_row["instrument_id"])
        notional: float = float(
            intended_notional_usdt.get(iid, config.default_intended_notional_usdt)
        )

        # ------------------------------------------------------------------
        # Gate 1: NOT_ONBOARDED - instrument available_at > decision_at or NaT
        # ------------------------------------------------------------------
        avail_at = inst_row.get("available_at")
        if pd.isna(avail_at) or avail_at > decision_at:
            eligibilities.append(
                ExecutionEligibility(
                    instrument_id=iid,
                    decision_at=decision_at,
                    eligible=False,
                    code=EligibilityCode.NOT_ONBOARDED,
                    reasons=(
                        EligibilityReason(
                            code=EligibilityCode.NOT_ONBOARDED,
                            hard=True,
                            observed_value=None,
                            threshold=None,
                            source="instrument_registry",
                            confidence=DataConfidence.OBSERVED,
                        ),
                    ),
                    intended_notional_usdt=notional,
                )
            )
            continue

        # ------------------------------------------------------------------
        # Gate 2: STATUS_NOT_TRADING
        # ------------------------------------------------------------------
        status: str = str(inst_row.get("status", ""))
        if status != "TRADING":
            eligibilities.append(
                ExecutionEligibility(
                    instrument_id=iid,
                    decision_at=decision_at,
                    eligible=False,
                    code=EligibilityCode.STATUS_NOT_TRADING,
                    reasons=(
                        EligibilityReason(
                            code=EligibilityCode.STATUS_NOT_TRADING,
                            hard=True,
                            observed_value=None,
                            threshold=None,
                            source="instrument_registry",
                            confidence=DataConfidence.OBSERVED,
                        ),
                    ),
                    intended_notional_usdt=notional,
                )
            )
            continue

        # ------------------------------------------------------------------
        # Gate 3: DATA_CONFIDENCE_LOW
        # ------------------------------------------------------------------
        raw_conf = inst_row.get("confidence", DataConfidence.UNKNOWN)
        inst_confidence = (
            DataConfidence(raw_conf) if isinstance(raw_conf, str) else raw_conf
        )
        if not _confidence_ge(inst_confidence, config.min_data_confidence):
            eligibilities.append(
                ExecutionEligibility(
                    instrument_id=iid,
                    decision_at=decision_at,
                    eligible=False,
                    code=EligibilityCode.DATA_CONFIDENCE_LOW,
                    reasons=(
                        EligibilityReason(
                            code=EligibilityCode.DATA_CONFIDENCE_LOW,
                            hard=True,
                            observed_value=_CONFIDENCE_LEVEL[inst_confidence],
                            threshold=_CONFIDENCE_LEVEL[config.min_data_confidence],
                            source="instrument_registry",
                            confidence=inst_confidence,
                        ),
                    ),
                    intended_notional_usdt=notional,
                )
            )
            continue

        # ------------------------------------------------------------------
        # Gate 4: MISSING_RULES
        # ------------------------------------------------------------------
        exec_rules = rules.get(iid)
        if exec_rules is None:
            eligibilities.append(
                ExecutionEligibility(
                    instrument_id=iid,
                    decision_at=decision_at,
                    eligible=False,
                    code=EligibilityCode.MISSING_RULES,
                    reasons=(
                        EligibilityReason(
                            code=EligibilityCode.MISSING_RULES,
                            hard=True,
                            observed_value=None,
                            threshold=None,
                            source="rule_history",
                            confidence=DataConfidence.UNKNOWN,
                        ),
                    ),
                    intended_notional_usdt=notional,
                )
            )
            continue

        # ------------------------------------------------------------------
        # Gate 5: STALE_MARKET_DATA - adv30_usdt metric must be present
        # ------------------------------------------------------------------
        adv30 = _extract_metric(obs_pivot, iid, "adv30_usdt")
        if math.isnan(adv30):
            eligibilities.append(
                ExecutionEligibility(
                    instrument_id=iid,
                    decision_at=decision_at,
                    eligible=False,
                    code=EligibilityCode.STALE_MARKET_DATA,
                    reasons=(
                        EligibilityReason(
                            code=EligibilityCode.STALE_MARKET_DATA,
                            hard=True,
                            observed_value=None,
                            threshold=None,
                            source="market_observations",
                            confidence=DataConfidence.UNKNOWN,
                        ),
                    ),
                    intended_notional_usdt=notional,
                )
            )
            continue

        # ------------------------------------------------------------------
        # Gate 6: ORDER_TOO_SMALL
        # ------------------------------------------------------------------
        last_price = _extract_metric(obs_pivot, iid, "last_price")
        rounded_qty = float("nan")
        rounded_notional = float("nan")

        if not math.isnan(last_price) and last_price > 0.0:
            qty_raw = notional / last_price
            step = exec_rules.step_size
            rounded_qty = math.floor(qty_raw / step) * step
            rounded_notional = rounded_qty * last_price

            if rounded_qty < exec_rules.min_qty or rounded_notional < exec_rules.min_notional:
                eligibilities.append(
                    ExecutionEligibility(
                        instrument_id=iid,
                        decision_at=decision_at,
                        eligible=False,
                        code=EligibilityCode.ORDER_TOO_SMALL,
                        reasons=(
                            EligibilityReason(
                                code=EligibilityCode.ORDER_TOO_SMALL,
                                hard=True,
                                observed_value=rounded_notional,
                                threshold=exec_rules.min_notional,
                                source="execution_rules",
                                confidence=exec_rules.min_notional_confidence,
                            ),
                        ),
                        intended_notional_usdt=notional,
                        rounded_notional_usdt=rounded_notional if not math.isnan(rounded_notional) else 0.0,
                    )
                )
                continue

        # ------------------------------------------------------------------
        # Gate 7: COST_TOO_HIGH
        # ------------------------------------------------------------------
        round_trip_cost = _compute_round_trip_cost(
            notional, adv30, exec_rules.taker_fee_bps
        )
        if round_trip_cost > config.max_round_trip_cost_bps:
            eligibilities.append(
                ExecutionEligibility(
                    instrument_id=iid,
                    decision_at=decision_at,
                    eligible=False,
                    code=EligibilityCode.COST_TOO_HIGH,
                    reasons=(
                        EligibilityReason(
                            code=EligibilityCode.COST_TOO_HIGH,
                            hard=True,
                            observed_value=round_trip_cost,
                            threshold=config.max_round_trip_cost_bps,
                            source="cost_model",
                            confidence=exec_rules.taker_fee_confidence,
                        ),
                    ),
                    intended_notional_usdt=notional,
                    rounded_notional_usdt=rounded_notional if not math.isnan(rounded_notional) else 0.0,
                    cost_bps=round_trip_cost,
                )
            )
            continue

        # ------------------------------------------------------------------
        # ELIGIBLE - compute capacity and risk metadata
        # ------------------------------------------------------------------
        cap_cost = _capacity_from_cost(
            adv30, exec_rules.taker_fee_bps, config.max_round_trip_cost_bps
        )
        cap_participation = adv30 * config.max_participation_rate
        capacity = min(cap_cost, cap_participation)

        vol30 = _extract_metric(obs_pivot, iid, "vol30")
        risk_scale = vol30 if (not math.isnan(vol30) and vol30 > 0.0) else 1.0

        eligibilities.append(
            ExecutionEligibility(
                instrument_id=iid,
                decision_at=decision_at,
                eligible=True,
                code=EligibilityCode.ELIGIBLE,
                reasons=(),
                intended_notional_usdt=notional,
                rounded_notional_usdt=rounded_notional if not math.isnan(rounded_notional) else 0.0,
                capacity_usdt=capacity,
                risk_scale=risk_scale,
                cost_bps=round_trip_cost,
                confidence=inst_confidence,
            )
        )

    eligible_count = sum(1 for e in eligibilities if e.eligible)
    instrument_ids: tuple[str, ...] = tuple(
        str(r["instrument_id"]) for _, r in instruments.iterrows()
    )

    return EligibilitySnapshot(
        decision_at=decision_at,
        eligibilities=tuple(eligibilities),
        instrument_ids=instrument_ids,
        metadata={"n_eligible": str(eligible_count)},
    )


# ---------------------------------------------------------------------------
# Function 3: build_universe_state_cube
# ---------------------------------------------------------------------------


def build_universe_state_cube(
    *,
    calendar: pd.DatetimeIndex,
    instruments: tuple[str, ...],
    snapshots: list[EligibilitySnapshot],
) -> UniverseStateCube:
    """Build a dense TxN eligibility cube from a list of eligibility snapshots.

    For each calendar bar the function looks up the most recent snapshot
    with ``decision_at <= bar`` (forward-fill). Bars with no preceding
    snapshot default to ineligible / entry_block=True (fail-closed).

    Eligible→ineligible transitions set ``exit_required[t, n] = True``
    and ``entry_block[t+1:, n] = True`` for all subsequent bars.

    Args:
        calendar: Monotonic UTC DatetimeIndex of decision bars (length T).
        instruments: Ordered tuple of instrument_id strings (length N).
        snapshots: List of EligibilitySnapshot instances. Order is arbitrary;
            the function indexes by ``decision_at``.

    Returns:
        UniverseStateCube with dense bool/float64 arrays of shape (T, N).

    Raises:
        ValueError: If any snapshot contains an instrument_id not present in
            the ``instruments`` tuple.

    Time Complexity: O(T*N) array fill + O(S) snapshot indexing.
    Space Complexity: O(T*N) for six dense arrays.
    """
    n_bars = len(calendar)
    n_inst = len(instruments)
    inst_index: dict[str, int] = {iid: idx for idx, iid in enumerate(instruments)}

    # Validate snapshot instrument ids
    for snap in snapshots:
        for elig in snap.eligibilities:
            if elig.instrument_id not in inst_index:
                raise ValueError(
                    f"universe symbol axis mismatch: "
                    f"'{elig.instrument_id}' not in instruments tuple"
                )

    # Index snapshots by decision_at (pd.Timestamp UTC)
    snap_index: dict[pd.Timestamp, EligibilitySnapshot] = {}
    for snap in snapshots:
        if snap.decision_at.tzinfo is None:
            ts = pd.Timestamp(snap.decision_at, tz="UTC")
        else:
            ts = pd.Timestamp(snap.decision_at)
        snap_index[ts] = snap

    sorted_snap_times = sorted(snap_index.keys())

    # Dense arrays - shape (n_bars, n_inst)
    eligible = np.zeros((n_bars, n_inst), dtype=np.bool_)
    entry_block = np.ones((n_bars, n_inst), dtype=np.bool_)  # fail-closed default
    exit_required = np.zeros((n_bars, n_inst), dtype=np.bool_)
    capacity_usdt = np.zeros((n_bars, n_inst), dtype=np.float64)
    risk_scale = np.ones((n_bars, n_inst), dtype=np.float64)
    cost_bps = np.zeros((n_bars, n_inst), dtype=np.float64)

    # Forward-fill: for each bar find the most recent snapshot <= bar
    prev_snap: EligibilitySnapshot | None = None
    snap_cursor = 0

    for t, bar_ts in enumerate(calendar):
        bar_pd = pd.Timestamp(bar_ts)

        # Advance cursor to include all snapshots with decision_at <= bar_ts
        while (
            snap_cursor < len(sorted_snap_times)
            and sorted_snap_times[snap_cursor] <= bar_pd
        ):
            prev_snap = snap_index[sorted_snap_times[snap_cursor]]
            snap_cursor += 1

        if prev_snap is None:
            # No snapshot yet: fail-closed (entry_block=True, eligible=False)
            continue

        # Fill arrays from the current (forward-filled) snapshot
        for elig in prev_snap.eligibilities:
            n = inst_index[elig.instrument_id]
            eligible[t, n] = elig.eligible
            entry_block[t, n] = not elig.eligible
            capacity_usdt[t, n] = elig.capacity_usdt
            risk_scale[t, n] = elig.risk_scale
            cost_bps[t, n] = elig.cost_bps

    # ------------------------------------------------------------------
    # Detect eligible->ineligible transitions and set exit_required /
    # propagate entry_block forward.
    # ------------------------------------------------------------------
    for t in range(1, n_bars):
        # Transition mask: was eligible at t-1 but not at t
        transition = eligible[t - 1] & ~eligible[t]
        exit_required[t] = transition
        if transition.any():
            # Block new entries for all bars after this transition
            for n in np.where(transition)[0]:
                if t + 1 < n_bars:
                    entry_block[t + 1 :, n] = True

    return UniverseStateCube(
        calendar=calendar,
        instrument_ids=instruments,
        eligible=eligible,
        entry_block=entry_block,
        exit_required=exit_required,
        capacity_usdt=capacity_usdt,
        risk_scale=risk_scale,
        cost_bps=cost_bps,
    )
