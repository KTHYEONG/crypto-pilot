from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.application.futures.optimization.config import FuturesRunConfig
from src.domain.futures.strategy import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import (
    MLPipelineOutput,
    run_ml_pipeline_for_universe,
)


def pick_strategy_data_maps(
    oos_data_maps: dict[str, dict[str, Any]],
    is_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> dict[str, dict[str, Any]]:
    """Return IS maps for ML training to prevent look-ahead leakage from OOS period.

    Strips ``is_start_idx_{tf}`` so that alignment falls back to fetch_start (row 0),
    giving the full warmup + IS window (~36 months) for walk-forward fold generation.
    OOS data after ``oos_start`` is never included because IS maps only contain
    rows up to ``oos_start``.
    """
    del oos_data_maps, valid_symbols
    is_start_key = f"is_start_idx_{tf}"
    result: dict[str, dict[str, Any]] = {}
    for sym, sym_dict in is_data_maps.items():
        if is_start_key in sym_dict:
            cleaned: dict[str, Any] = {k: v for k, v in sym_dict.items() if k != is_start_key}
            result[sym] = cleaned
        else:
            result[sym] = sym_dict
    return result


def assert_strategy_alpha_ready(
    *,
    ml_out: MLPipelineOutput,
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    """Validate strategy alpha contract after merge."""
    alpha_panel = getattr(ml_out, "alpha_panel", None)
    if alpha_panel is None or alpha_panel.empty:
        raise RuntimeError("strategy mode requires non-empty alpha_panel")
    for required_col in ("alpha_long", "alpha_short"):
        if required_col not in alpha_panel.columns:
            raise RuntimeError(f"strategy alpha_panel missing required column: {required_col}")

    long_non_zero = 0
    short_non_zero = 0
    merged_count = 0
    for sym in valid_symbols:
        df = oos_data_maps.get(sym, {}).get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "alpha_long" not in df.columns or "alpha_short" not in df.columns:
            raise RuntimeError(f"strategy merge missing alpha columns for symbol={sym}")
        merged_count += 1
        long_non_zero += int(np.count_nonzero(df["alpha_long"].to_numpy(dtype=np.float64)))
        short_non_zero += int(np.count_nonzero(df["alpha_short"].to_numpy(dtype=np.float64)))

    if merged_count == 0:
        raise RuntimeError("strategy mode has no merged symbol frames for selected timeframe")
    if long_non_zero <= 0 or short_non_zero <= 0:
        raise RuntimeError(
            "strategy merge produced zero-only alpha columns "
            f"(nonzero long={long_non_zero}, short={short_non_zero})"
        )


def run_active_strategy_output_bridge(
    *,
    run_config: FuturesRunConfig,
    symbols: list[str],
    tf: str,
    fetch_start: str | None,
    end_date: str | None,
    opt_config: dict[str, Any],
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
) -> MLPipelineOutput:
    """Run active strategy bridge for quick-backtest/strategy modes."""
    if run_config.mode == "quick-backtest":
        return MLPipelineOutput()
    if run_config.mode not in {"strategy", "strategy-smoke"}:
        raise ValueError(f"unsupported mode for active strategy bridge: {run_config.mode}")
    if run_config.strategy is None:
        raise ValueError("strategy mode requires strategy")
    if preloaded_data_maps is None:
        raise ValueError("strategy mode requires preloaded_data_maps")

    strategy_cfg = StrategyConfig(name=run_config.strategy)
    return run_ml_pipeline_for_universe(
        symbols=symbols,
        tf=tf,
        fetch_start=fetch_start,
        end_date=end_date,
        opt_config=opt_config,
        strategy_cfg=strategy_cfg,
        preloaded_data_maps=preloaded_data_maps,
    )
