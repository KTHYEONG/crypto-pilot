from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.optimization.optimizer import compute_multi_alignment_info
from src.domain.futures.strategy.builder import build_strategy_alpha
from src.domain.futures.strategy.config import MomentumConfig, StrategyConfig


def _make_maps(n: int = 240, symbols: int = 6) -> tuple[dict[str, dict[str, Any]], list[str]]:
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(hours=4 * i) for i in range(n)]
    out: dict[str, dict[str, Any]] = {}
    syms: list[str] = []
    for i in range(symbols):
        sym = f"S{i}USDT"
        syms.append(sym)
        close = np.linspace(100.0 + i, 200.0 + i * 2.0, n)
        out[sym] = {
            "4h": pd.DataFrame({"datetime": pd.to_datetime(dates), "close": close}),
            "is_start_idx_4h": 0,
        }
    return out, syms


def test_panel_index() -> None:
    maps, syms = _make_maps()
    cfg = StrategyConfig(momentum=MomentumConfig(lookback_bars=6, min_symbols_for_xs=5))
    panel = build_strategy_alpha(maps, syms, "4h", cfg)
    assert isinstance(panel.index, pd.MultiIndex)
    assert panel.index.names == ["datetime", "symbol"]
    assert panel.index.is_monotonic_increasing


def test_panel_columns() -> None:
    maps, syms = _make_maps()
    panel = build_strategy_alpha(maps, syms, "4h", StrategyConfig())
    assert list(panel.columns) == ["alpha_long", "alpha_short"]


def test_alignment_consistency() -> None:
    maps, syms = _make_maps()
    info = compute_multi_alignment_info(maps, syms, "4h", embargo=0)
    assert info is not None
    panel = build_strategy_alpha(maps, syms, "4h", StrategyConfig())
    expected_first_dt = info["common_is_start_dt"]
    assert panel.index.get_level_values("datetime").min() == expected_first_dt


def test_insufficient_symbols_raises() -> None:
    maps, syms = _make_maps(symbols=4)
    cfg = StrategyConfig(momentum=MomentumConfig(min_symbols_for_xs=5))
    with pytest.raises(ValueError, match="strategy needs >="):
        build_strategy_alpha(maps, syms, "4h", cfg)
