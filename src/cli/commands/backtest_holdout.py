"""Holdout one-look gate shared by the strategy and account backtest commands."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_logger = logging.getLogger("MhsBacktestCli")


def enforce_holdout_gate(
    strategy_id: str, family: str, strategy: Any, start: pd.Timestamp, end: pd.Timestamp, evaluate_holdout: bool,
) -> None:
    """Refuse windows overlapping a consumed holdout unless the one look is recorded first."""
    from src.common.errors import DataIntegrityError
    from src.evaluation.holdout import consume_holdout_look, holdout_overlaps
    from src.strategy.release import load_release, releases_dir, strategy_signal_digest

    path = releases_dir() / f"{family}.holdout.jsonl"
    release = load_release("flow_mom_top20" if family == "flow_mom" else strategy_id)
    signal_digest = strategy_signal_digest(strategy)
    if end > release.design_data_cutoff and not evaluate_holdout:
        raise SystemExit("post-design windows require --evaluate-holdout before observing returns")
    overlaps = holdout_overlaps(strategy_id, (start, end), path=path)
    if overlaps and not evaluate_holdout:
        raise SystemExit(f"window overlaps a consumed holdout for {strategy_id}; pass --evaluate-holdout for the one look")
    if not (overlaps or evaluate_holdout):
        return
    try:
        if not overlaps and start < release.design_data_cutoff:
            raise DataIntegrityError("holdout window must start at or after the design cutoff")
        rematerialized = consume_holdout_look(
            strategy_id, release.spec_digest, (start, end), path=path, signal_digest=signal_digest,
        )
    except DataIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    if overlaps and rematerialized is False:
        _logger.info("[EVAL] holdout re-materialized window=%s..%s", start.isoformat(), end.isoformat())


def require_journaled_holdout(end: pd.Timestamp) -> None:
    """Refuse account replays past the design cutoff unless the signal's holdout look is journaled."""
    from src.evaluation.holdout import holdout_covered
    from src.strategy.release import strategy_signal_digest
    from src.strategy.targets import FLOW_MOM_TOP20

    if end <= FLOW_MOM_TOP20.design_data_cutoff:
        return
    covered = holdout_covered(
        "flow_mom_top20", strategy_signal_digest(FLOW_MOM_TOP20), (FLOW_MOM_TOP20.design_data_cutoff, end),
    )
    if not covered:
        raise SystemExit("account replay past the design cutoff requires a journaled holdout look for this signal")
