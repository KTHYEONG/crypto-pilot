"""Holdout look journaling shared by the strategy, account and calibration backtest commands."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_logger = logging.getLogger("MhsBacktestCli")


def record_post_design_look(
    strategy_id: str, family: str, strategy: Any, start: pd.Timestamp, end: pd.Timestamp, *, purpose: str,
) -> None:
    """Journal the post-design part of [start, end) as observed by this signal; design-only windows pass."""
    from src.common.errors import DataIntegrityError
    from src.evaluation.holdout import record_holdout_look
    from src.strategy.release import load_release, releases_dir, strategy_signal_digest

    if end <= strategy.design_data_cutoff:
        return
    release = load_release("flow_mom_top20" if family == "flow_mom" else strategy_id)
    cutoff = release.design_data_cutoff
    window = (max(start, cutoff), end)
    try:
        added = record_holdout_look(
            strategy_id, release.spec_digest, window,
            path=releases_dir() / f"{family}.holdout.jsonl",
            signal_digest=strategy_signal_digest(strategy), purpose=purpose,
        )
    except DataIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    if added:
        _logger.info(
            "[ALGO] stage=holdout_look recorded strategy=%s purpose=%s window=%s..%s",
            strategy_id, purpose, window[0].isoformat(), window[1].isoformat(),
        )
