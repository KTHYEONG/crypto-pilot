"""Unit invariants for exit-deferral primitives: cause, retry search, episode clock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution.exit_deferral import (
    ExitBlockCause,
    ExitEpisodeClock,
    classify_exit_block_cause,
    first_viable_retry_bar,
)
from src.mhs.params import EXIT_DEFERRAL_MAX_AGE


@pytest.mark.parametrize(
    ("halted", "quote_volume", "mark_valid", "funding_known", "expected"),
    [
        (True, 0.0, True, True, ExitBlockCause.VENUE_HALT),
        (True, 123.0, True, True, ExitBlockCause.VENUE_HALT),
        (True, 0.0, False, True, None),
        (True, 0.0, True, False, None),
        (True, float("nan"), True, True, None),
        (True, -5.0, True, True, None),
        (True, float("inf"), True, True, None),
        (False, 0.0, True, True, ExitBlockCause.SYMBOL_NO_TRADE),
        (False, 0.0, False, True, None),
        (False, 0.0, True, False, None),
        (False, float("nan"), True, True, None),
        (False, -1.0, True, True, None),
        (False, float("inf"), True, True, None),
        (False, 10.0, True, True, None),
    ],
)
def test_classification_matrix(halted, quote_volume, mark_valid, funding_known, expected) -> None:
    assert classify_exit_block_cause(
        halted=halted, quote_volume=quote_volume,
        mark_valid=mark_valid, funding_known=funding_known,
    ) == expected


def _search_arrays(n: int = 10, last_trade_idx: int = 9) -> dict[str, np.ndarray]:
    grid_ns = np.arange(n, dtype="int64") * 180_000_000_000
    return {
        "halted": np.zeros(n, dtype=bool),
        "quote_volume": np.full(n, 1000.0),
        "funding_known": np.ones(n, dtype=bool),
        "close": np.full(n, 100.0),
        "grid_ns": grid_ns,
        "last_trade_ns": int(grid_ns[last_trade_idx]),
    }


def test_retry_bar_search_skips_inviable_bars() -> None:
    arr = _search_arrays()
    arr["halted"][[2, 3]] = True
    arr["quote_volume"][4] = 0.0
    arr["funding_known"][5] = False
    arr["close"][6] = float("nan")
    assert first_viable_retry_bar(after_pos=1, deadline_pos=9, **arr) == 7  # type: ignore[arg-type]


def test_retry_bar_search_returns_minus_one_without_viable_bar() -> None:
    arr = _search_arrays()
    arr["quote_volume"][2:] = 0.0
    assert first_viable_retry_bar(after_pos=1, deadline_pos=9, **arr) == -1  # type: ignore[arg-type]


def test_retry_bar_search_ignores_bars_past_deadline() -> None:
    arr = _search_arrays()
    arr["quote_volume"][2:6] = 0.0
    assert first_viable_retry_bar(after_pos=1, deadline_pos=5, **arr) == -1  # type: ignore[arg-type]
    arr["quote_volume"][6:] = 0.0
    frozen = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in arr.items()}
    arr["quote_volume"][6] = 1000.0
    assert first_viable_retry_bar(after_pos=1, deadline_pos=5, **arr) == first_viable_retry_bar(  # type: ignore[arg-type]
        after_pos=1, deadline_pos=5, **frozen,  # type: ignore[arg-type]
    ) == -1


def test_retry_bar_search_respects_last_trade_cutoff() -> None:
    arr = _search_arrays(last_trade_idx=7)
    assert first_viable_retry_bar(after_pos=6, deadline_pos=9, **arr) == -1  # type: ignore[arg-type]


def test_retry_bar_search_clamps_negative_start() -> None:
    arr = _search_arrays()
    assert first_viable_retry_bar(after_pos=-2, deadline_pos=9, **arr) == 0  # type: ignore[arg-type]


def test_clock_rejects_negative_max_age() -> None:
    with pytest.raises(ValueError, match="max_age_ns"):
        ExitEpisodeClock(-1)


def test_clock_episode_identity_and_age() -> None:
    clock = ExitEpisodeClock(100)
    assert clock.admit(0, 10.0, 1000) is True
    assert clock.admit(0, 10.0, 1050) is True
    assert clock.admit(0, 10.0, 1100) is True
    assert clock.admit(0, 10.0, 1101) is False
    assert clock.admit(0, 5.0, 1101) is True
    assert clock.admit(1, 10.0, 1101) is True
    with pytest.raises(DataIntegrityError):
        clock.admit(0, 5.0, 1099)


def test_clock_bound_matches_registered_max_age() -> None:
    assert pd.Timedelta(hours=24) == EXIT_DEFERRAL_MAX_AGE
    clock = ExitEpisodeClock(int(EXIT_DEFERRAL_MAX_AGE.value))
    day_ns = 86_400_000_000_000
    assert clock.admit(3, 7.0, 0) is True
    assert clock.admit(3, 7.0, day_ns) is True
    assert clock.admit(3, 7.0, day_ns + 1) is False
