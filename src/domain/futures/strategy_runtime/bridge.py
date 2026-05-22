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
    del fetch_start, end_date, opt_config, kwargs
    if strategy_cfg is None or preloaded_data_maps is None:
        return MLPipelineOutput()
    from src.domain.futures.strategy.builder import build_strategy_alpha

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
        if df["datetime"].dtype != sym_rows["datetime"].dtype:
            raise RuntimeError(
                f"datetime dtype mismatch: {df['datetime'].dtype} != {sym_rows['datetime'].dtype}"
            )
        merged = df[["datetime"]].merge(
            sym_rows[["datetime", "alpha_long", "alpha_short"]],
            on="datetime",
            how="left",
        )
        df["alpha_long"] = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        df["alpha_short"] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)

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
