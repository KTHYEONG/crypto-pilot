from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.domain.futures.universe.contracts import UniverseStateCube

_logger = logging.getLogger(__name__)


def sync_universe_ledger(
    config: CompoundRunConfig,
) -> tuple[datetime, bool]:
    ref_dt: datetime
    if config.reference_date is not None:
        ref_dt = datetime.strptime(config.reference_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        ref_dt = datetime.now(UTC)
    if config.sync == "skip":
        _logger.info("universe ledger sync=skip, reference_date=%s", config.reference_date)
        return ref_dt, False
    _logger.info("universe ledger sync triggered")
    return ref_dt, True


def resolve_universe_symbols(
    config: CompoundRunConfig,
    reference_dt: datetime,
    *,
    max_symbols: int = 120,
) -> tuple[str, ...]:
    return ()


def build_pit_universe_state(
    symbols: tuple[str, ...],
    reference_dt: datetime,
    *,
    bars: int = 2048,
) -> UniverseStateCube:
    calendar = pd.date_range(end=reference_dt, periods=bars, freq="h", tz="UTC")
    n_bars = len(calendar)
    n_syms = len(symbols)

    return UniverseStateCube(
        calendar=calendar,
        instrument_ids=symbols,
        eligible=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt=np.full((n_bars, n_syms), 1e8, dtype=np.float64),
        risk_scale=np.ones((n_bars, n_syms), dtype=np.float64),
        cost_bps=np.full((n_bars, n_syms), 12.0, dtype=np.float64),
    )
