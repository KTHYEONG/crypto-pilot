from __future__ import annotations

from typing import Any

import pandas as pd

from src.domain.futures.strategy.config import StrategyConfig


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

    raise ValueError(
        f"unsupported active strategy name: {cfg.name}; allowed: candidate_ml, rule_baseline"
    )
