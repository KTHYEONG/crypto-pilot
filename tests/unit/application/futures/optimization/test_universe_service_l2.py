"""Tests for discover_universe_timeline new parameters: l2_start, min_history_bars."""

from __future__ import annotations

import inspect


def test_discover_universe_timeline_accepts_l2_start_param() -> None:
    """l2_start 파라미터가 시그니처에 존재하는지 확인."""
    from src.application.futures.optimization.universe_service import (
        discover_universe_timeline,
    )

    sig = inspect.signature(discover_universe_timeline)
    assert "l2_start" in sig.parameters
    assert sig.parameters["l2_start"].default is None


def test_discover_universe_timeline_accepts_min_history_bars_param() -> None:
    """min_history_bars 파라미터가 시그니처에 존재하는지 확인."""
    from src.application.futures.optimization.universe_service import (
        discover_universe_timeline,
    )

    sig = inspect.signature(discover_universe_timeline)
    assert "min_history_bars" in sig.parameters
    assert sig.parameters["min_history_bars"].default == 0


def test_discover_universe_timeline_backward_compat_kwargs() -> None:
    """기존 파라미터 (is_start, oos_start, end_date, tf) 여전히 존재."""
    from src.application.futures.optimization.universe_service import (
        discover_universe_timeline,
    )

    sig = inspect.signature(discover_universe_timeline)
    for param in ("tf", "is_start", "oos_start", "end_date"):
        assert param in sig.parameters, f"Missing param: {param}"
