"""Shared completing-fold synthetic market for slow-tier fold tests."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src.mhs.evidence import AnchoredPurgedFold

COMPLETING_FOLD_MARKET_HOURS: Final[int] = 4400
"""Hourly bars in the shared completing-fold market (2021-01-01 00:00 → 2021-07-03 07:00 UTC)."""

COMPLETING_FOLD: Final[AnchoredPurgedFold] = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-05-20", tz="UTC"),
    pd.Timestamp("2021-05-28", tz="UTC"),
    pd.Timestamp("2021-06-27", tz="UTC"),
    168,
    168,
)
"""Smallest anchored purged fold that completes end to end on the synthetic market.

The train reference ``[train_start + FOLD_PANEL_WARMUP_HOURS, train_end)`` yields 100 daily
returns (burn-in requires ``PNL_VOL_TARGET_BURN_IN_DAYS``). Validation starts after the 168h
purge and ends inside the market span, so strict/stress replays, train-only discovery and
resource telemetry all execute instead of aborting into an incomplete fold report.
"""


def write_completing_fold_market(root: Path) -> pd.Timestamp:
    """Write the shared completing-fold synthetic lake under ``root``.

    Produces ``1h`` (with ``volume`` for the zombie-mask data policy), ``1m``, native-aggregated
    ``3m`` execution bars and ``funding`` parquet for the 10 dev-partition ``MHS*USDT``
    symbols, deterministically seeded and byte-identical across calls.

    Args:
        root: Empty or non-existent directory (created as needed).
    Returns:
        The last hourly bar timestamp (``2021-07-03 07:00 UTC``), used as the request ``end``.
    """
    from tests.unit.mhs.test_evaluation_appresearch import _write_3m_cache, _write_mhs_market

    end = _write_mhs_market(root, n_hours=COMPLETING_FOLD_MARKET_HOURS)
    _write_3m_cache(root)
    return end
