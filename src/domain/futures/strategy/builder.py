from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.domain.futures.optimization.optimizer import compute_multi_alignment_info
from src.domain.futures.strategy.config import StrategyConfig

_logger = logging.getLogger(__name__)


def build_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Build long-format alpha panel from aligned expected return signals.

    Args:
        data_maps: Dictionary containing historical data per symbol.
        symbols: List of target symbols.
        tf: Timeframe string.
        cfg: Top-level StrategyConfig.

    Returns:
        pd.DataFrame sorted by (datetime, symbol) with columns ["alpha_long", "alpha_short"].

    """
    if cfg.name in {"candidate_ml", "rule_baseline"}:
        from src.domain.futures.strategy_runtime.bridge import run_candidate_strategy_for_universe
        res = run_candidate_strategy_for_universe(
            symbols=symbols,
            tf=tf,
            strategy_cfg=cfg,
            preloaded_data_maps=data_maps,
        )
        return res.alpha_panel

    # 1. Align price panels (using compute_multi_alignment_info base)
    info = compute_multi_alignment_info(data_maps, symbols, tf, embargo=0)
    if info is None:
        return pd.DataFrame(columns=["alpha_long", "alpha_short"])

    offsets: dict[str, int] = info["alignment_offsets"]
    valid_symbols = [
        sym for sym in symbols if sym in offsets and sym in data_maps and tf in data_maps[sym]
    ]

    min_syms = cfg.blend.min_symbols
    if len(valid_symbols) < min_syms:
        raise ValueError(f"strategy needs >= {min_syms} symbols, got {len(valid_symbols)}")

    # Legacy sleeve logic removed. Only candidate_ml path is fully supported.
    _logger.warning(f"Strategy {cfg.name} not fully implemented in non-legacy mode.")
    return pd.DataFrame(columns=["alpha_long", "alpha_short"])
