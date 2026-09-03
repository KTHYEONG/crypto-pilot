"""Diagnostic-only leverage-frontier scanner over registered growth envelopes.

Read-only by contract: loads an already-persisted daily ledger artifact,
slices the leak-free pre-OOS reference window, and reports the bootstrap
mdd-breach/ruin frontier per candidate leverage multiple. It never writes to
docs/results/, never mutates ``GROWTH_RISK_ENVELOPES`` or
``COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS``, and never constructs a diagnostic
request -- adopting a new ceiling still requires a human-registered envelope
rung plus a real replay re-verification.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.params import (
    COMMITTEE_GROWTH_BARS_PER_YEAR,
    COMMITTEE_GROWTH_N_PATHS,
    COMMITTEE_OOS_START,
    GROWTH_RISK_ENVELOPES,
    PNL_VOL_TARGET_BURN_IN_DAYS,
)
from src.mhs.report.persist import mhs_horizon_diagnostic_report_path
from src.quant.risk.growth_sizing import (
    FrontierScanPoint,
    GrowthSizingConfig,
    scan_leverage_frontier,
)

_logger = logging.getLogger("MhsLeverageFrontierScan")

_REFERENCE_REPLAY_ID = "blend_pre_vol_target_reference"


def _default_artifact_path() -> pathlib.Path:
    """Mirror persist.py's compact artifact_root derivation exactly."""
    report = pathlib.Path(mhs_horizon_diagnostic_report_path())
    return report.parent / f"{report.stem}_artifacts" / "daily_ledger.parquet"


def _load_pre_oos_reference_returns(
    artifact_path: pathlib.Path,
    oos_start: pd.Timestamp = COMMITTEE_OOS_START,
) -> pd.Series:
    """Load the pre-OOS ``blend_pre_vol_target_reference`` daily returns.

    Fail-closed with no silent fallback (standalone diagnostic tool): a missing
    artifact file, a missing reference replay_id, or a pre-OOS slice shorter
    than ``PNL_VOL_TARGET_BURN_IN_DAYS`` finite rows each raise
    ``DataIntegrityError`` naming the specific cause.
    """
    if not artifact_path.exists():
        raise DataIntegrityError(
            f"daily ledger artifact not found: {artifact_path} -- "
            "run the regular mhs-horizon-diagnostic first"
        )
    frame = pd.read_parquet(artifact_path)
    if _REFERENCE_REPLAY_ID not in set(frame["replay_id"]):
        raise DataIntegrityError(
            f"daily ledger artifact {artifact_path} has no "
            f"'{_REFERENCE_REPLAY_ID}' replay_id; found: {sorted(set(frame['replay_id']))}"
        )
    reference = frame.loc[frame["replay_id"] == _REFERENCE_REPLAY_ID].set_index("date")
    train = reference["daily_return"].dropna()
    train = train.loc[train.index < oos_start]
    if len(train) < PNL_VOL_TARGET_BURN_IN_DAYS:
        raise DataIntegrityError(
            f"pre-OOS reference unresolved: {len(train)} finite rows before "
            f"{oos_start} in {artifact_path}, require >= {PNL_VOL_TARGET_BURN_IN_DAYS}"
        )
    return train


def run_leverage_frontier_scan(
    envelope_name: str,
    candidate_multiples: tuple[float, ...],
    artifact_path: str | None = None,
) -> tuple[FrontierScanPoint, ...]:
    """Scan candidate leverage multiples against a registered envelope's frontier.

    Read-only diagnostic: resolves the registered envelope and the persisted
    pre-OOS reference ledger, computes ``reference_risk = std(ddof=1)`` on that
    series (the same anchor the exposure-cap verification uses), runs the
    shared-bootstrap frontier scan, logs one ``[EVAL]`` line per point, and
    returns the points unchanged. Nothing is selected, persisted, or promoted.
    """
    if envelope_name not in GROWTH_RISK_ENVELOPES:
        registered = sorted(GROWTH_RISK_ENVELOPES)
        raise ValueError(
            f"unknown growth_envelope '{envelope_name}'; "
            f"registered keys: {registered}"
        )
    envelope = GROWTH_RISK_ENVELOPES[envelope_name]

    path = (
        pathlib.Path(artifact_path)
        if artifact_path is not None
        else _default_artifact_path()
    )
    train = _load_pre_oos_reference_returns(path)

    reference_risk = float(train.std(ddof=1))
    if not np.isfinite(reference_risk) or reference_risk <= 0:
        raise ValueError(
            f"envelope '{envelope.name}' has leverage_ceiling={envelope.leverage_ceiling} "
            f"but reference_daily_returns has too little history/variance "
            f"({len(train)} finite rows) to verify the bootstrap ruin frontier"
        )

    config = GrowthSizingConfig(
        risk_grid=(reference_risk,),
        reference_risk=reference_risk,
        max_drawdown=envelope.max_drawdown,
        max_drawdown_prob=envelope.max_drawdown_prob,
        ruin_fraction=envelope.ruin_fraction,
        max_ruin_prob=envelope.max_ruin_prob,
        horizon_years=envelope.horizon_years,
        n_paths=COMMITTEE_GROWTH_N_PATHS,
        bars_per_year=COMMITTEE_GROWTH_BARS_PER_YEAR,
    )
    points = scan_leverage_frontier(
        train.to_numpy(), config, candidate_multiples,
    )
    for point in points:
        _logger.info(
            "[EVAL] leverage_frontier_scan envelope=%s multiple=%.2f "
            "mdd_breach_prob=%.3f ruin_prob=%.3f feasible=%s",
            envelope_name, point.multiple, point.mdd_breach_prob,
            point.ruin_prob, point.feasible,
        )
    return points
