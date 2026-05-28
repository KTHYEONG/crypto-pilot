from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.application.futures.optimization.config import FuturesRunConfig
from src.domain.futures.strategy import StrategyConfig
from src.domain.futures.strategy.inference import validate_alpha_forecast_metadata
from src.domain.futures.strategy_runtime.bridge import (
    MLPipelineOutput,
    run_ml_pipeline_for_universe,
)

_logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class StrategyAlphaReadinessReport:
    """Alpha merge readiness diagnostics for strategy preflight."""

    merged_symbols: int
    panel_long_non_zero: int
    panel_short_non_zero: int
    merged_panel_long_non_zero: int
    merged_panel_short_non_zero: int
    target_oos_long_non_zero: int
    target_oos_short_non_zero: int
    target_oos_rows: int
    panel_start: str
    panel_end: str
    warnings: tuple[str, ...] = ()


def summarize_strategy_alpha_readiness(
    *,
    ml_out: MLPipelineOutput,
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> StrategyAlphaReadinessReport:
    """Summarize alpha-panel merge coverage without assuming target OOS coverage."""
    alpha_panel = getattr(ml_out, "alpha_panel", None)
    if alpha_panel is None or alpha_panel.empty:
        raise RuntimeError("strategy mode requires non-empty alpha_panel")
    for required_col in ("alpha_long", "alpha_short"):
        if required_col not in alpha_panel.columns:
            raise RuntimeError(f"strategy alpha_panel missing required column: {required_col}")
    validate_alpha_forecast_metadata(alpha_panel)

    panel_reset = alpha_panel.reset_index()
    if "datetime" not in panel_reset.columns or "symbol" not in panel_reset.columns:
        raise RuntimeError("strategy alpha_panel missing datetime/symbol index columns")
    panel_reset["_merge_datetime"] = (
        pd.to_datetime(panel_reset["datetime"], utc=True).dt.tz_localize(None)
    )
    panel_long_non_zero = int(
        np.count_nonzero(panel_reset["alpha_long"].to_numpy(dtype=np.float64))
    )
    panel_short_non_zero = int(
        np.count_nonzero(panel_reset["alpha_short"].to_numpy(dtype=np.float64))
    )
    panel_start = str(panel_reset["_merge_datetime"].min())
    panel_end = str(panel_reset["_merge_datetime"].max())
    panel_dt_by_symbol = {
        str(sym): set(sym_rows["_merge_datetime"].to_numpy())
        for sym, sym_rows in panel_reset.groupby("symbol", sort=False)
    }

    merged_panel_long_non_zero = 0
    merged_panel_short_non_zero = 0
    target_oos_long_non_zero = 0
    target_oos_short_non_zero = 0
    target_oos_rows = 0
    merged_symbols = 0
    for sym in valid_symbols:
        df = oos_data_maps.get(sym, {}).get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "alpha_long" not in df.columns or "alpha_short" not in df.columns:
            raise RuntimeError(f"strategy merge missing alpha columns for symbol={sym}")
        merged_symbols += 1
        frame_dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        panel_window_mask = frame_dt.isin(panel_dt_by_symbol.get(sym, set())).to_numpy()
        alpha_long = df["alpha_long"].to_numpy(dtype=np.float64)
        alpha_short = df["alpha_short"].to_numpy(dtype=np.float64)
        merged_panel_long_non_zero += int(np.count_nonzero(alpha_long[panel_window_mask]))
        merged_panel_short_non_zero += int(np.count_nonzero(alpha_short[panel_window_mask]))
        oos_start_idx = int(oos_data_maps.get(sym, {}).get(f"oos_start_idx_{tf}", 0))
        target_oos_mask = np.arange(len(df), dtype=np.int64) >= oos_start_idx
        target_oos_rows += int(np.count_nonzero(target_oos_mask))
        target_oos_long_non_zero += int(np.count_nonzero(alpha_long[target_oos_mask]))
        target_oos_short_non_zero += int(np.count_nonzero(alpha_short[target_oos_mask]))

    warnings: list[str] = []
    if (
        target_oos_rows > 0
        and target_oos_long_non_zero <= 0
        and target_oos_short_non_zero <= 0
        and merged_panel_long_non_zero > 0
        and merged_panel_short_non_zero > 0
    ):
        warnings.append("target_oos_alpha_absent_preflight")

    return StrategyAlphaReadinessReport(
        merged_symbols=merged_symbols,
        panel_long_non_zero=panel_long_non_zero,
        panel_short_non_zero=panel_short_non_zero,
        merged_panel_long_non_zero=merged_panel_long_non_zero,
        merged_panel_short_non_zero=merged_panel_short_non_zero,
        target_oos_long_non_zero=target_oos_long_non_zero,
        target_oos_short_non_zero=target_oos_short_non_zero,
        target_oos_rows=target_oos_rows,
        panel_start=panel_start,
        panel_end=panel_end,
        warnings=tuple(warnings),
    )


def assert_strategy_alpha_ready(
    *,
    ml_out: MLPipelineOutput,
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
    require_target_oos_alpha: bool = False,
) -> StrategyAlphaReadinessReport:
    """Validate strategy alpha contract after merge."""
    report = summarize_strategy_alpha_readiness(
        ml_out=ml_out,
        oos_data_maps=oos_data_maps,
        valid_symbols=valid_symbols,
        tf=tf,
    )
    if report.merged_symbols == 0:
        raise RuntimeError("strategy mode has no merged symbol frames for selected timeframe")
    if report.panel_long_non_zero <= 0 or report.panel_short_non_zero <= 0:
        raise RuntimeError(
            "strategy alpha_panel is zero-only "
            f"(nonzero long={report.panel_long_non_zero}, short={report.panel_short_non_zero})"
        )
    if report.merged_panel_long_non_zero <= 0 or report.merged_panel_short_non_zero <= 0:
        raise RuntimeError(
            "strategy merge produced zero-only alpha columns in panel window "
            f"(nonzero long={report.merged_panel_long_non_zero}, "
            f"short={report.merged_panel_short_non_zero})"
        )
    if require_target_oos_alpha and (
        report.target_oos_long_non_zero <= 0 or report.target_oos_short_non_zero <= 0
    ):
        raise RuntimeError(
            "strategy target OOS alpha is zero-only "
            f"(nonzero long={report.target_oos_long_non_zero}, "
            f"short={report.target_oos_short_non_zero})"
        )
    if report.warnings:
        _logger.warning(
            "[STRATEGY-ALPHA-READINESS] warnings=%s panel=[%s..%s] merged_symbols=%d",
            list(report.warnings),
            report.panel_start,
            report.panel_end,
            report.merged_symbols,
        )
    return report


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
    if run_config.mode not in {"strategy", "strategy-smoke", "alpha"}:
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
