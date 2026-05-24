from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import StrategyConfig

HMM_SEMANTIC_PROB_COLUMNS: list[str] = [
    "hmm_prob_bull_calm",
    "hmm_prob_bull_vol_up",
    "hmm_prob_bear_trend",
    "hmm_prob_chop",
    "hmm_prob_crisis",
]
_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MLPipelineOutput:
    """Neutral bridge output for runtime compatibility."""

    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_probs: pd.DataFrame = field(default_factory=pd.DataFrame)
    hmm_report: dict[str, Any] = field(default_factory=dict)
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
    # Extract anchoring params BEFORE discarding kwargs
    _anchor_end_idx: int | None = kwargs.get("anchor_end_idx")
    _target_start: int | None = kwargs.get("target_start_idx")
    _target_end: int | None = kwargs.get("target_end_idx")
    del fetch_start, end_date, opt_config, kwargs
    if strategy_cfg is None or preloaded_data_maps is None:
        return MLPipelineOutput()
    from src.domain.futures.strategy.builder import build_strategy_alpha

    if (
        _anchor_end_idx is not None
        and _target_start is not None
        and _target_end is not None
    ):
        from src.domain.futures.strategy.ml_builder import build_ml_strategy_alpha_anchored
        alpha_panel = build_ml_strategy_alpha_anchored(
            data_maps=preloaded_data_maps,
            symbols=symbols,
            tf=tf,
            cfg=strategy_cfg,
            anchor_end_idx=int(_anchor_end_idx),
            target_start=int(_target_start),
            target_end=int(_target_end),
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
                    sym in offsets
                    and sym in preloaded_data_maps
                    and tf in preloaded_data_maps[sym]
                )
            ]
            if len(valid_symbols) >= strategy_cfg.blend.min_symbols:
                close_2d = np.zeros((eff_len, len(valid_symbols)), dtype=np.float64)
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

    return MLPipelineOutput(alpha_panel=alpha_panel, market_probs=market_probs)


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
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        _df = data_maps[sym][tf]
        if "alpha_long" in _df.columns:
            _long_nz_ratios.append(float((_df["alpha_long"] != 0).mean()))
        if "alpha_short" in _df.columns:
            _short_nz_ratios.append(float((_df["alpha_short"] != 0).mean()))
    _alpha_long_nz = float(np.mean(_long_nz_ratios)) if _long_nz_ratios else 0.0
    _alpha_short_nz = float(np.mean(_short_nz_ratios)) if _short_nz_ratios else 0.0
    _regime_broadcast = (
        getattr(ml_out, "market_probs", None) is not None
        and not getattr(ml_out, "market_probs", pd.DataFrame()).empty
    )
    _logger.info(
        "[ALPHA-MERGE] merged_syms=%d alpha_long_nz=%.3f alpha_short_nz=%.3f regime_broadcast=%s",
        merged_symbols,
        _alpha_long_nz,
        _alpha_short_nz,
        _regime_broadcast,
    )

    # Broadcast market_probs (Regime) to all symbols
    mp = getattr(ml_out, "market_probs", None)
    if mp is not None and not mp.empty:
        prob_cols = [
            "hmm_prob_bull_calm",
            "hmm_prob_bull_vol_up",
            "hmm_prob_bear_trend",
            "hmm_prob_chop",
            "hmm_prob_crisis",
        ]
        for sym in symbols:
            if sym not in data_maps or tf not in data_maps[sym]:
                continue
            df = data_maps[sym][tf]
            # Merge probabilities
            merged_probs = df[["datetime"]].merge(mp.reset_index(), on="datetime", how="left")
            for c in prob_cols:
                # Fallback: bull_calm is 1.0, other states are 0.0 if not present
                fill_val = 1.0 if c == "hmm_prob_bull_calm" else 0.0
                df[c] = merged_probs[c].fillna(fill_val).to_numpy(dtype=np.float64)



class FuturesMLStrategy:
    """Neutral strategy stub for bridge contract."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        """Store strategy identity and params."""
        self.name = name
        self.params = params

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return pass-through copy of input frame."""
        return df.copy()
