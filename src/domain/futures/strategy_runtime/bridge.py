from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

HMM_SEMANTIC_PROB_COLUMNS: list[str] = []


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
    **kwargs: Any,
) -> MLPipelineOutput:
    del symbols, tf, fetch_start, end_date, opt_config, kwargs
    return MLPipelineOutput()


def merge_ml_output_into_is_and_oos(
    ml_out: MLPipelineOutput,
    is_maps: dict[str, dict[str, Any]],
    oos_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    del ml_out, is_maps, oos_maps, valid_symbols, tf


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
    del ml_out, data_maps, symbols, tf, log_tag


class FuturesMLStrategy:
    """Neutral strategy stub for bridge contract."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self.name = name
        self.params = params

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()
