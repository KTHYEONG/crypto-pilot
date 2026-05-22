from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.optimization.optimizer import compute_multi_alignment_info
from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy.momentum import compute_xs_momentum_alpha


def _assert_no_legacy_imports() -> None:
    if any(name.startswith("src.domain.futures.legacy") for name in sys.modules):
        raise RuntimeError("legacy import forbidden in strategy module")


def build_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Build long-format alpha panel from aligned close panel."""
    _assert_no_legacy_imports()
    info = compute_multi_alignment_info(data_maps, symbols, tf, embargo=0)
    if info is None:
        return pd.DataFrame(columns=["alpha_long", "alpha_short"])

    eff_len = int(info["eff_ref_len"])
    offsets: dict[str, int] = info["alignment_offsets"]
    valid_symbols = [
        sym
        for sym in symbols
        if sym in offsets and sym in data_maps and tf in data_maps[sym]
    ]

    if len(valid_symbols) < cfg.momentum.min_symbols_for_xs:
        raise ValueError(
            f"strategy needs >= {cfg.momentum.min_symbols_for_xs} symbols, got {len(valid_symbols)}"
        )

    close_2d = np.zeros((eff_len, len(valid_symbols)), dtype=np.float64)
    datetimes: np.ndarray | None = None

    for col_idx, sym in enumerate(valid_symbols):
        df = data_maps[sym][tf]
        start_idx = offsets[sym]
        end_idx = start_idx + eff_len
        close_2d[:, col_idx] = df["close"].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
        if datetimes is None:
            datetimes = df["datetime"].iloc[start_idx:end_idx].to_numpy()

    if datetimes is None:
        return pd.DataFrame(columns=["alpha_long", "alpha_short"])

    alpha_long, alpha_short = compute_xs_momentum_alpha(close_2d, cfg.momentum)

    idx = pd.MultiIndex.from_product([datetimes, valid_symbols], names=["datetime", "symbol"])
    panel = pd.DataFrame(
        {
            "alpha_long": alpha_long.reshape(-1),
            "alpha_short": alpha_short.reshape(-1),
        },
        index=idx,
    ).sort_index()

    vals = panel[["alpha_long", "alpha_short"]].to_numpy(dtype=np.float64)
    bad_mask = ~np.isfinite(vals)
    if bad_mask.any():
        bad_row = int(np.argwhere(bad_mask)[0][0])
        raise RuntimeError(f"alpha_panel contains NaN/Inf at idx={bad_row}")
    if not panel.index.is_monotonic_increasing:
        raise RuntimeError("alpha_panel must be sorted by (datetime, symbol)")

    panel.attrs["strategy_name"] = cfg.name
    panel.attrs["lookback_bars"] = cfg.momentum.lookback_bars
    return panel
