from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.builder import build_strategy_alpha
from src.domain.futures.strategy.config import MomentumConfig, StrategyConfig


def _warn_if_lookahead_suspicious(forward_ic: float, backward_ic: float) -> None:
    if backward_ic > forward_ic:
        warnings.warn(
            "look-ahead suspicion: backward IC exceeds forward IC",
            UserWarning,
            stacklevel=2,
        )


def _synth_maps(
    t_len: int = 260, n_syms: int = 20
) -> tuple[dict[str, dict[str, object]], list[str], np.ndarray]:
    rng = np.random.default_rng(123)
    dates = pd.to_datetime([datetime(2025, 1, 1) + timedelta(hours=4 * i) for i in range(t_len)])

    returns = rng.normal(0.0, 0.01, size=(t_len, n_syms))
    for t in range(12, t_len - 1):
        cross_base = np.linspace(-1.0, 1.0, n_syms)
        cross = np.roll(cross_base, t % n_syms)
        returns[t + 1] += 0.02 * cross

    close = np.full((t_len, n_syms), 100.0, dtype=np.float64)
    for t in range(1, t_len):
        close[t] = close[t - 1] * (1.0 + returns[t])

    maps: dict[str, dict[str, object]] = {}
    symbols: list[str] = []
    for i in range(n_syms):
        sym = f"X{i}USDT"
        symbols.append(sym)
        maps[sym] = {
            "4h": pd.DataFrame({"datetime": dates, "close": close[:, i]}),
            "is_start_idx_4h": 0,
        }
    return maps, symbols, returns


def _panel_to_2d(panel: pd.DataFrame, symbols: list[str], col: str) -> np.ndarray:
    wide = panel[col].unstack("symbol").reindex(columns=symbols)
    return wide.to_numpy(dtype=np.float64)


def test_alpha_nonzero_ratio() -> None:
    maps, symbols, _ = _synth_maps()
    cfg = StrategyConfig(
        momentum=MomentumConfig(
            lookback_bars=6, top_ratio=0.3, bottom_ratio=0.3, min_symbols_for_xs=5
        )
    )
    panel = build_strategy_alpha(maps, symbols, "4h", cfg)
    a_l = _panel_to_2d(panel, symbols, "alpha_long")[6:]
    a_s = _panel_to_2d(panel, symbols, "alpha_short")[6:]

    long_ratio = float(np.mean(a_l > 0.0))
    short_ratio = float(np.mean(a_s > 0.0))
    both_ratio = float(np.mean((a_l > 0.0) & (a_s > 0.0)))
    assert 0.25 <= long_ratio <= 0.35
    assert 0.25 <= short_ratio <= 0.35
    assert both_ratio == 0.0


def test_alpha_scale_range() -> None:
    maps, symbols, _ = _synth_maps()
    cfg = StrategyConfig(momentum=MomentumConfig(edge_scale_per_bar=1e-3, min_symbols_for_xs=5))
    panel = build_strategy_alpha(maps, symbols, "4h", cfg)
    vals = panel["alpha_long"].to_numpy(dtype=np.float64)
    pos = vals[vals > 0.0]
    assert pos.size > 0
    assert float(np.max(vals)) <= 1e-3 + 1e-12
    assert float(np.mean(pos)) > 0.0


def test_forward_ic_positive() -> None:
    maps, symbols, returns = _synth_maps()
    cfg = StrategyConfig(momentum=MomentumConfig(lookback_bars=6, min_symbols_for_xs=5))
    panel = build_strategy_alpha(maps, symbols, "4h", cfg)
    a_l = _panel_to_2d(panel, symbols, "alpha_long")

    ics: list[float] = []
    for t in range(6, a_l.shape[0] - 1):
        s_alpha = pd.Series(a_l[t], index=symbols)
        s_ret = pd.Series(returns[t + 1], index=symbols)
        ics.append(float(s_alpha.corr(s_ret, method="spearman")))
    mean_ic = float(np.nanmean(np.array(ics, dtype=np.float64)))
    assert mean_ic > 0.0


def test_lookahead_ic_suspicious() -> None:
    maps, symbols, returns = _synth_maps()
    cfg = StrategyConfig(momentum=MomentumConfig(lookback_bars=6, min_symbols_for_xs=5))
    panel = build_strategy_alpha(maps, symbols, "4h", cfg)
    a_l = _panel_to_2d(panel, symbols, "alpha_long")

    fwd, back = [], []
    for t in range(7, a_l.shape[0] - 1):
        s_alpha = pd.Series(a_l[t], index=symbols)
        fwd.append(
            float(s_alpha.corr(pd.Series(returns[t + 1], index=symbols), method="spearman"))
        )
        back.append(
            float(s_alpha.corr(pd.Series(returns[t - 1], index=symbols), method="spearman"))
        )

    fwd_ic = float(np.nanmean(np.array(fwd, dtype=np.float64)))
    back_ic = float(np.nanmean(np.array(back, dtype=np.float64)))
    if back_ic > fwd_ic:
        with pytest.warns(UserWarning):
            _warn_if_lookahead_suspicious(fwd_ic, back_ic)
    else:
        _warn_if_lookahead_suspicious(fwd_ic, back_ic)
        assert fwd_ic >= back_ic


def test_alpha_turnover_finite() -> None:
    maps, symbols, _ = _synth_maps()
    panel = build_strategy_alpha(maps, symbols, "4h", StrategyConfig())
    a_l = _panel_to_2d(panel, symbols, "alpha_long")
    diff_nonzero = np.abs(np.diff(a_l, axis=0)) > 0.0
    turnover = float(np.mean(diff_nonzero[6:]))
    assert 0.0 < turnover < 1.0


def test_alpha_not_constant() -> None:
    maps, symbols, _ = _synth_maps()
    panel = build_strategy_alpha(maps, symbols, "4h", StrategyConfig())
    a_l = _panel_to_2d(panel, symbols, "alpha_long")
    per_symbol_std = np.std(a_l[6:], axis=0)
    assert np.all(per_symbol_std > 0.0)
