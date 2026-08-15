from __future__ import annotations

import pytest

from src.domain.futures.strategy.timeframe_contracts import select_crisis_load_tf


@pytest.mark.parametrize(("target_tf", "expected"), [("4h", "4h"), ("1h", "1h")])
def test_select_crisis_load_tf_source_backed_tf_returns_unchanged(
    target_tf: str, expected: str
) -> None:
    result = select_crisis_load_tf(target_tf)
    assert result == expected


@pytest.mark.parametrize("target_tf", ["8h", "12h", "1d"])
def test_select_crisis_load_tf_coarse_master_selects_coarsest_compatible_source(
    target_tf: str,
) -> None:
    result = select_crisis_load_tf(target_tf)
    assert result == "4h"


def test_select_crisis_load_tf_finer_than_probe_returns_target_unchanged() -> None:
    result = select_crisis_load_tf("15m")
    assert result == "15m"


def test_select_crisis_load_tf_non_multiple_tf_falls_back_to_finest_compatible() -> None:
    result = select_crisis_load_tf("6h")
    assert result == "1h"
