from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from src.application.futures.optimization.config import FuturesRunConfig
from src.domain.futures.strategy import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import (
    CandidatePipelineOutput,
    run_candidate_strategy_for_universe,
)

_logger = logging.getLogger(__name__)


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _parse_str_tuple(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item))
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item))
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return tuple(item for item in items if item)
    return (str(value),)


def build_candidate_strategy_config(
    *,
    strategy_cfg: StrategyConfig,
    opt_config: dict[str, Any],
    timeframe: str,
    signal_only: bool = False,
) -> StrategyConfig:
    """Build a runtime candidate strategy config from opt_config overrides."""
    candidate = strategy_cfg.candidate
    candidate = replace(
        candidate,
        timeframe=timeframe,
        candidate_families=_parse_str_tuple(
            opt_config.get("FUTURES_CANDIDATE_FAMILIES"),
            default=candidate.candidate_families,
        ),
        enabled_candidate_variants=_parse_str_tuple(
            opt_config.get("FUTURES_CANDIDATE_ENABLED_VARIANTS"),
            default=candidate.enabled_candidate_variants,
        ),
        side_flip_candidate_variants=_parse_str_tuple(
            opt_config.get("FUTURES_CANDIDATE_SIDE_FLIP_VARIANTS"),
            default=candidate.side_flip_candidate_variants,
        ),
        diagnostic_top_k=int(opt_config.get("FUTURES_CANDIDATE_DIAGNOSTIC_TOP_K", candidate.diagnostic_top_k)),
        min_variant_oos_obs=int(opt_config.get("FUTURES_CANDIDATE_MIN_VARIANT_OOS_OBS", candidate.min_variant_oos_obs)),
        min_variant_oos_edge_bps=float(
            opt_config.get("FUTURES_CANDIDATE_MIN_VARIANT_OOS_EDGE_BPS", candidate.min_variant_oos_edge_bps)
        ),
        min_variant_oos_profit_bps=float(
            opt_config.get("FUTURES_CANDIDATE_MIN_VARIANT_OOS_PROFIT_BPS", candidate.min_variant_oos_profit_bps)
        ),
        ensemble_variant_prior_enabled=_parse_bool(
            opt_config.get("FUTURES_CANDIDATE_ENSEMBLE_VARIANT_PRIOR_ENABLED"),
            default=candidate.ensemble_variant_prior_enabled,
        ),
        ensemble_variant_shrinkage_k=float(
            opt_config.get("FUTURES_CANDIDATE_ENSEMBLE_VARIANT_SHRINKAGE_K", candidate.ensemble_variant_shrinkage_k)
        ),
        ensemble_variant_min_obs=int(
            opt_config.get("FUTURES_CANDIDATE_ENSEMBLE_VARIANT_MIN_OBS", candidate.ensemble_variant_min_obs)
        ),
        min_variant_oos_hit_rate=float(
            opt_config.get("FUTURES_CANDIDATE_MIN_VARIANT_OOS_HIT_RATE", candidate.min_variant_oos_hit_rate)
        ),
        regime_cell_admission_enabled=_parse_bool(
            opt_config.get("FUTURES_CANDIDATE_REGIME_CELL_ADMISSION_ENABLED"),
            default=candidate.regime_cell_admission_enabled,
        ),
        min_regime_cell_oos_obs=int(
            opt_config.get("FUTURES_CANDIDATE_MIN_REGIME_CELL_OOS_OBS", candidate.min_regime_cell_oos_obs)
        ),
        min_regime_cell_edge_bps=float(
            opt_config.get("FUTURES_CANDIDATE_MIN_REGIME_CELL_EDGE_BPS", candidate.min_regime_cell_edge_bps)
        ),
        max_admitted_cells_per_variant=int(
            opt_config.get("FUTURES_CANDIDATE_MAX_ADMITTED_CELLS_PER_VARIANT", candidate.max_admitted_cells_per_variant)
        ),
        min_admission_posterior_prob=float(
            opt_config.get("FUTURES_CANDIDATE_MIN_ADMISSION_POSTERIOR_PROB", candidate.min_admission_posterior_prob)
        ),
        admission_use_newey_west=_parse_bool(
            opt_config.get("FUTURES_CANDIDATE_ADMISSION_USE_NEWEY_WEST"),
            default=candidate.admission_use_newey_west,
        ),
        admission_tau_prior_bps=float(
            opt_config.get("FUTURES_CANDIDATE_ADMISSION_TAU_PRIOR_BPS", candidate.admission_tau_prior_bps)
        ),
        ensemble_score_calibration_enabled=_parse_bool(
            opt_config.get("FUTURES_CANDIDATE_ENSEMBLE_SCORE_CALIBRATION_ENABLED"),
            default=candidate.ensemble_score_calibration_enabled,
        ),
        ensemble_score_z_clip=float(
            opt_config.get("FUTURES_CANDIDATE_ENSEMBLE_SCORE_Z_CLIP", candidate.ensemble_score_z_clip)
        ),
        ensemble_score_calibration_min_obs=int(
            opt_config.get(
                "FUTURES_CANDIDATE_ENSEMBLE_SCORE_CALIBRATION_MIN_OBS",
                candidate.ensemble_score_calibration_min_obs,
            )
        ),
        ensemble_score_slope_k=float(
            opt_config.get("FUTURES_CANDIDATE_ENSEMBLE_SCORE_SLOPE_K", candidate.ensemble_score_slope_k)
        ),
        l1_boundary_mode=cast(
            Literal["exact_label_interval", "fixed_gap"],
            str(opt_config.get("FUTURES_L1_BOUNDARY_MODE", candidate.l1_boundary_mode)),
        ),
        l1_boundary_buffer_bars=int(
            opt_config.get("FUTURES_L1_BOUNDARY_BUFFER_BARS", candidate.l1_boundary_buffer_bars)
        ),
        signal_only=signal_only,
    )
    return replace(strategy_cfg, candidate=candidate)


def pick_strategy_data_maps(
    oos_data_maps: dict[str, dict[str, Any]],
    is_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> dict[str, dict[str, Any]]:
    """Merge IS and OOS frames per symbol/timeframe into one contiguous map.

    Time Complexity: O(sum(len(is_df) + len(oos_df))) for concat+sort+dedup per symbol.
    Space Complexity: O(sum(len(is_df) + len(oos_df))) for the merged result.

    Args:
        oos_data_maps: Out-of-sample frames keyed by symbol -> timeframe -> DataFrame.
        is_data_maps: In-sample frames keyed by symbol -> timeframe -> DataFrame.
        valid_symbols: Symbols eligible for the merged result; others are dropped.
        tf: Timeframe key to merge (e.g. "4h").

    Returns:
        Mapping of symbol -> dict (timeframe -> merged DataFrame, plus passthrough
        metadata keys excluding `is_start_idx_{tf}`).
    """
    is_start_key = f"is_start_idx_{tf}"
    valid_set = set(valid_symbols)
    result: dict[str, dict[str, Any]] = {}
    for sym, sym_dict in is_data_maps.items():
        if sym not in valid_set:
            continue
        merged_sym: dict[str, Any] = {k: v for k, v in sym_dict.items() if k != is_start_key}
        is_df = sym_dict.get(tf)
        oos_df = oos_data_maps.get(sym, {}).get(tf)
        if isinstance(is_df, pd.DataFrame) and isinstance(oos_df, pd.DataFrame) and not oos_df.empty:
            if is_df.empty:
                merged_sym[tf] = oos_df
            else:
                merged_sym[tf] = (
                    pd.concat([is_df, oos_df], ignore_index=True)
                    .sort_values("datetime")
                    .drop_duplicates(subset="datetime", keep="first")
                    .reset_index(drop=True)
                )
        result[sym] = merged_sym
    return result


@dataclass(frozen=True, slots=True)
class CandidateOutputReadinessReport:
    merged_symbols: int
    panel_target_weight_non_zero: int
    merged_panel_target_weight_non_zero: int
    target_oos_target_weight_non_zero: int
    target_oos_rows: int
    panel_start: str
    panel_end: str
    warnings: tuple[str, ...] = ()


def _candidate_panel_has_non_finite_metadata(alpha_panel: pd.DataFrame) -> bool:
    meta = alpha_panel.attrs.get("alpha_forecast_metadata")
    if not isinstance(meta, dict) or not meta:
        return False

    def _scan(value: Any) -> bool:
        if isinstance(value, dict):
            return any(_scan(item) for item in value.values())
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if arr.size == 0:
            return False
        return not bool(np.all(np.isfinite(arr)))

    return any(_scan(value) for value in meta.values())


def summarize_candidate_output_readiness(
    *,
    candidate_out: CandidatePipelineOutput,
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> CandidateOutputReadinessReport:
    alpha_panel = getattr(candidate_out, "alpha_panel", None)
    if alpha_panel is None or alpha_panel.empty:
        raise RuntimeError("strategy mode requires non-empty candidate alpha_panel")
    if "target_weight" not in alpha_panel.columns:
        raise RuntimeError("candidate alpha_panel missing required column: target_weight")

    panel_reset = alpha_panel.reset_index()
    if "datetime" not in panel_reset.columns or "symbol" not in panel_reset.columns:
        raise RuntimeError("candidate alpha_panel missing datetime/symbol index columns")
    panel_reset["_merge_datetime"] = pd.to_datetime(panel_reset["datetime"], utc=True).dt.tz_localize(None)
    if _candidate_panel_has_non_finite_metadata(alpha_panel):
        raise RuntimeError("metadata contains non-finite values")

    panel_target_weight_non_zero = int(
        np.count_nonzero(panel_reset["target_weight"].to_numpy(dtype=np.float64))
    )
    panel_start = str(panel_reset["_merge_datetime"].min())
    panel_end = str(panel_reset["_merge_datetime"].max())
    panel_dt_by_symbol = {
        str(sym): set(sym_rows["_merge_datetime"].to_numpy())
        for sym, sym_rows in panel_reset.groupby("symbol", sort=False)
    }

    merged_panel_target_weight_non_zero = 0
    target_oos_target_weight_non_zero = 0
    target_oos_rows = 0
    merged_symbols = 0
    for sym in valid_symbols:
        df = oos_data_maps.get(sym, {}).get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "target_weight" not in df.columns:
            raise RuntimeError(f"strategy merge missing target_weight for symbol={sym}")
        merged_symbols += 1
        frame_dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        panel_window_mask = frame_dt.isin(panel_dt_by_symbol.get(sym, set())).to_numpy()
        target_weight = df["target_weight"].to_numpy(dtype=np.float64)
        merged_panel_target_weight_non_zero += int(np.count_nonzero(target_weight[panel_window_mask]))
        oos_start_idx = int(oos_data_maps.get(sym, {}).get(f"oos_start_idx_{tf}", 0))
        target_oos_mask = np.arange(len(df), dtype=np.int64) >= oos_start_idx
        target_oos_rows += int(np.count_nonzero(target_oos_mask))
        target_oos_target_weight_non_zero += int(np.count_nonzero(target_weight[target_oos_mask]))

    warnings: list[str] = []
    if (
        target_oos_rows > 0
        and target_oos_target_weight_non_zero <= 0
        and merged_panel_target_weight_non_zero > 0
    ):
        warnings.append("target_oos_candidate_output_absent_preflight")

    return CandidateOutputReadinessReport(
        merged_symbols=merged_symbols,
        panel_target_weight_non_zero=panel_target_weight_non_zero,
        merged_panel_target_weight_non_zero=merged_panel_target_weight_non_zero,
        target_oos_target_weight_non_zero=target_oos_target_weight_non_zero,
        target_oos_rows=target_oos_rows,
        panel_start=panel_start,
        panel_end=panel_end,
        warnings=tuple(warnings),
    )


def assert_candidate_output_ready(
    *,
    candidate_out: CandidatePipelineOutput,
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
    require_target_oos_target_weight: bool = False,
) -> CandidateOutputReadinessReport:
    report = summarize_candidate_output_readiness(
        candidate_out=candidate_out,
        oos_data_maps=oos_data_maps,
        valid_symbols=valid_symbols,
        tf=tf,
    )
    if report.merged_symbols == 0:
        raise RuntimeError("strategy mode has no merged symbol frames for selected timeframe")
    if report.panel_target_weight_non_zero <= 0:
        raise RuntimeError(
            "candidate alpha_panel is zero-only "
            f"(nonzero target_weight={report.panel_target_weight_non_zero})"
        )
    if report.merged_panel_target_weight_non_zero <= 0:
        raise RuntimeError(
            "strategy merge produced zero-only target_weight in panel window "
            f"(nonzero target_weight={report.merged_panel_target_weight_non_zero})"
        )
    if require_target_oos_target_weight and report.target_oos_target_weight_non_zero <= 0:
        raise RuntimeError(
            "strategy target OOS target_weight is zero-only "
            f"(nonzero target_weight={report.target_oos_target_weight_non_zero})"
        )
    if report.warnings:
        _logger.warning(
            "[CANDIDATE-OUTPUT-READINESS] warnings=%s panel=[%s..%s] merged_symbols=%d",
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
    trading_symbols: tuple[str, ...] | None = None,
    silent: bool = False,
    extra_probe_cells: tuple[Any, ...] | None = None,
) -> CandidatePipelineOutput:
    del (
        fetch_start,
        end_date,
    )
    if run_config.phase not in {"l1", "l2", "l3"}:
        raise ValueError(f"unsupported phase for active strategy bridge: {run_config.phase}")
    if preloaded_data_maps is None:
        raise ValueError("active strategy bridge requires preloaded_data_maps")

    strategy_name = str(opt_config.get("FUTURES_STRATEGY_NAME", "candidate_ml"))
    use_tiered = bool(opt_config.get("USE_CS_RANK_ENGINE", False))
    strategy_cfg = build_candidate_strategy_config(
        strategy_cfg=StrategyConfig(name=strategy_name),
        opt_config=opt_config,
        timeframe=run_config.timeframe,
        signal_only=(run_config.phase == "l1") or use_tiered,
    )
    candidate_scope = list(trading_symbols or tuple(symbols))
    effective_symbols = [sym for sym in dict.fromkeys(candidate_scope) if sym in preloaded_data_maps]
    if not effective_symbols:
        raise ValueError("candidate ML scope is empty")

    return run_candidate_strategy_for_universe(
        symbols=effective_symbols,
        tf=tf,
        strategy_cfg=strategy_cfg,
        preloaded_data_maps=preloaded_data_maps,
        silent=silent,
        extra_probe_cells=extra_probe_cells,
    )
