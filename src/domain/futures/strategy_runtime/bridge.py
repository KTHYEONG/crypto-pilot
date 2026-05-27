from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import StrategyConfig

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MLPipelineOutput:
    """Neutral bridge output for runtime compatibility."""

    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_probs: pd.DataFrame = field(default_factory=pd.DataFrame)
    integrity_report: dict[str, Any] = field(default_factory=dict)
    meta_feature_frame_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)


def _enrich_with_gp_features(df: pd.DataFrame, tf: str = "1h") -> pd.DataFrame:
    del tf
    return df


def run_ml_pipeline_for_universe(
    symbols: list[str],
    tf: str,
    fetch_start: str | None,
    end_date: str | None,
    opt_config: dict[str, Any],
    *,
    strategy_cfg: StrategyConfig | None = None,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> MLPipelineOutput:
    t0 = time.perf_counter()
    # Extract anchoring params BEFORE discarding kwargs
    _anchor_end_idx: int | None = kwargs.get("anchor_end_idx")
    _target_start: int | None = kwargs.get("target_start_idx")
    _target_end: int | None = kwargs.get("target_end_idx")
    _precomputed_panels = kwargs.get("precomputed_panels")
    del fetch_start, end_date, opt_config, kwargs
    if strategy_cfg is None or preloaded_data_maps is None:
        return MLPipelineOutput()
    from src.domain.futures.strategy.builder import build_strategy_alpha

    if _anchor_end_idx is not None and _target_start is not None and _target_end is not None:
        from src.domain.futures.strategy.ml_builder import build_ml_strategy_alpha_anchored

        alpha_panel = build_ml_strategy_alpha_anchored(
            data_maps=preloaded_data_maps,
            symbols=symbols,
            tf=tf,
            cfg=strategy_cfg,
            anchor_end_idx=int(_anchor_end_idx),
            target_start=int(_target_start),
            target_end=int(_target_end),
            precomputed_panels=_precomputed_panels,
        )
    else:
        alpha_panel = build_strategy_alpha(
            data_maps=preloaded_data_maps,
            symbols=symbols,
            tf=tf,
            cfg=strategy_cfg,
        )

    market_probs = pd.DataFrame()
    if strategy_cfg.regime.enabled:
        from src.domain.futures.strategy.regime.provider import compute_regime_posterior

        from src.domain.futures.optimization.optimizer import compute_multi_alignment_info

        info = compute_multi_alignment_info(preloaded_data_maps, symbols, tf, embargo=0)
        if info is not None:
            eff_len = int(info["eff_ref_len"])
            offsets = info["alignment_offsets"]
            valid_symbols = [
                sym
                for sym in symbols
                if (
                    sym in offsets and sym in preloaded_data_maps and tf in preloaded_data_maps[sym]
                )
            ]
            if len(valid_symbols) >= strategy_cfg.blend.min_symbols:
                close_2d: np.ndarray = np.zeros(
                    (eff_len, len(valid_symbols)),
                    dtype=np.float64,
                )
                datetimes = None
                for col_idx, sym in enumerate(valid_symbols):
                    df = preloaded_data_maps[sym][tf]
                    start_idx = offsets[sym]
                    end_idx = start_idx + eff_len
                    close_2d[:, col_idx] = (
                        df["close"].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
                    )
                    if datetimes is None:
                        datetimes = df["datetime"].iloc[start_idx:end_idx].to_numpy()

                if datetimes is not None:
                    probs_dict = compute_regime_posterior(close_2d, strategy_cfg.regime)
                    probs_dict["datetime"] = datetimes
                    market_probs = pd.DataFrame(probs_dict).set_index("datetime")

    out = MLPipelineOutput(alpha_panel=alpha_panel, market_probs=market_probs)
    anchored = bool(
        _anchor_end_idx is not None and _target_start is not None and _target_end is not None
    )
    alpha_rows = len(alpha_panel)
    _logger.info(
        "[ML-PIPE-PROF] anchored=%s symbols=%d tf=%s elapsed=%.2fs alpha_rows=%d",
        anchored,
        len(symbols),
        tf,
        time.perf_counter() - t0,
        alpha_rows,
    )
    return out


def merge_ml_output_into_is_and_oos(
    ml_out: MLPipelineOutput,
    is_maps: dict[str, dict[str, Any]],
    oos_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    merge_ml_output_into_data_maps(ml_out, is_maps, valid_symbols, tf, log_tag="is")
    merge_ml_output_into_data_maps(ml_out, oos_maps, valid_symbols, tf, log_tag="oos")


def copy_data_maps_tf_clone(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        out[sym] = dict(data_maps.get(sym, {}))
        frame = out[sym].get(tf)
        if isinstance(frame, pd.DataFrame):
            out[sym][tf] = frame.copy()
    return out


def merge_ml_output_into_data_maps(
    ml_out: MLPipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    log_tag: str = "",
) -> None:
    panel = getattr(ml_out, "alpha_panel", None)
    if panel is None or panel.empty:
        return
    required = {"alpha_long", "alpha_short"}
    if not required.issubset(panel.columns):
        _logger.warning("[%s] alpha_panel missing required columns; skip merge", log_tag)
        return

    by_sym = panel.reset_index().groupby("symbol", sort=False)
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        try:
            sym_rows = by_sym.get_group(sym)
        except KeyError:
            continue
        df = data_maps[sym][tf]
        if "alpha_long" in df.columns or "alpha_short" in df.columns:
            _logger.warning("[%s] overwrite alpha columns for symbol=%s", log_tag, sym)
        left = df[["datetime"]].copy()
        right = sym_rows[["datetime", "alpha_long", "alpha_short"]].copy()
        left["_merge_datetime"] = pd.to_datetime(left["datetime"], utc=True).dt.tz_localize(None)
        right["_merge_datetime"] = pd.to_datetime(right["datetime"], utc=True).dt.tz_localize(None)
        merged = left.merge(
            right[["_merge_datetime", "alpha_long", "alpha_short"]],
            on="_merge_datetime",
            how="left",
        )

        # === [DIAG-MERGE] ===
        raw_long_nz = int(np.count_nonzero(right["alpha_long"].fillna(0.0).to_numpy()))
        merged_long_nz = int(np.count_nonzero(merged["alpha_long"].fillna(0.0).to_numpy()))
        if log_tag == "oos":
            l_sample = [str(x)[:19] for x in left["_merge_datetime"].iloc[:2]]
            r_sample = [str(x)[:19] for x in right["_merge_datetime"].iloc[:2]]
            _logger.info(
                f" [DIAG-SHORT] sym={sym} raw_L_nz={raw_long_nz} merged_L_nz={merged_long_nz} "
                f"L_rows={len(left)} R_rows={len(right)} L_dt={l_sample} R_dt={r_sample}"
            )

        df["alpha_long"] = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        df["alpha_short"] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)

    # [ALPHA-MERGE] summary log
    merged_symbols = sum(
        1
        for sym in symbols
        if sym in data_maps
        and tf in data_maps[sym]
        and "alpha_long" in data_maps[sym][tf].columns
    )
    _long_nz_ratios: list[float] = []
    _short_nz_ratios: list[float] = []
    _target_oos_long_nz_ratios: list[float] = []
    _target_oos_short_nz_ratios: list[float] = []
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        _df = data_maps[sym][tf]
        if "alpha_long" in _df.columns:
            _long_nz_ratios.append(float((_df["alpha_long"] != 0).mean()))
        if "alpha_short" in _df.columns:
            _short_nz_ratios.append(float((_df["alpha_short"] != 0).mean()))
        _oos_key = f"oos_start_idx_{tf}"
        if _oos_key in data_maps[sym]:
            _start = int(data_maps[sym][_oos_key])
            _mask = np.arange(len(_df), dtype=np.int64) >= _start
            if "alpha_long" in _df.columns:
                _target_oos_long_nz_ratios.append(
                    float(np.mean(_df["alpha_long"].to_numpy(dtype=np.float64)[_mask] != 0.0))
                )
            if "alpha_short" in _df.columns:
                _target_oos_short_nz_ratios.append(
                    float(np.mean(_df["alpha_short"].to_numpy(dtype=np.float64)[_mask] != 0.0))
                )
    _alpha_long_nz = float(np.mean(_long_nz_ratios)) if _long_nz_ratios else 0.0
    _alpha_short_nz = float(np.mean(_short_nz_ratios)) if _short_nz_ratios else 0.0
    _panel_start = "na"
    _panel_end = "na"
    try:
        _panel_dt = pd.to_datetime(panel.index.get_level_values("datetime"), utc=True).tz_localize(
            None
        )
        _panel_start = str(_panel_dt.min())
        _panel_end = str(_panel_dt.max())
    except Exception as exc:
        _logger.debug("[ALPHA-MERGE] panel datetime extraction failed: %s", exc)
    _target_oos_long_nz = (
        float(np.mean(_target_oos_long_nz_ratios)) if _target_oos_long_nz_ratios else 0.0
    )
    _target_oos_short_nz = (
        float(np.mean(_target_oos_short_nz_ratios)) if _target_oos_short_nz_ratios else 0.0
    )
    _logger.info(
        "[ALPHA-MERGE] merged_syms=%d panel_start=%s panel_end=%s "
        "alpha_long_nz=%.3f alpha_short_nz=%.3f "
        "target_oos_long_nz=%.3f target_oos_short_nz=%.3f",
        merged_symbols,
        _panel_start,
        _panel_end,
        _alpha_long_nz,
        _alpha_short_nz,
        _target_oos_long_nz,
        _target_oos_short_nz,
    )


class FuturesMLStrategy:
    """Neutral strategy stub for bridge contract."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        """Store strategy identity and params."""
        self.name = name
        self.params = params

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return pass-through copy of input frame."""
        return df.copy()
