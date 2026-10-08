"""Geometry guard for the shared completing-fold fixture (default suite, no I/O)."""

from __future__ import annotations

import pandas as pd

from src.core.params import FOLD_PANEL_WARMUP_HOURS, PNL_VOL_TARGET_BURN_IN_DAYS
from tests.fixtures.mhs_fold_market import COMPLETING_FOLD, COMPLETING_FOLD_MARKET_HOURS


def _expected_reference_rows() -> int:
    ref_start = COMPLETING_FOLD.train_start + pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS)
    return int((COMPLETING_FOLD.train_end - ref_start) // pd.Timedelta(days=1)) - 1


def test_train_reference_meets_burn_in() -> None:
    count = _expected_reference_rows()
    assert count >= PNL_VOL_TARGET_BURN_IN_DAYS, (
        f"FOLD_PANEL_WARMUP_HOURS={FOLD_PANEL_WARMUP_HOURS} leaves {count} reference rows, "
        f"require >= PNL_VOL_TARGET_BURN_IN_DAYS={PNL_VOL_TARGET_BURN_IN_DAYS}"
    )


def test_purge_embargo_honoured() -> None:
    gap_hours = (COMPLETING_FOLD.validation_start - COMPLETING_FOLD.train_end) // pd.Timedelta(hours=1)
    assert gap_hours >= COMPLETING_FOLD.purge_hours


def test_validation_inside_market_span() -> None:
    last_bar = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(hours=COMPLETING_FOLD_MARKET_HOURS - 1)
    assert COMPLETING_FOLD.validation_end <= last_bar
    assert COMPLETING_FOLD.validation_start - pd.Timedelta(hours=FOLD_PANEL_WARMUP_HOURS) >= COMPLETING_FOLD.train_start
