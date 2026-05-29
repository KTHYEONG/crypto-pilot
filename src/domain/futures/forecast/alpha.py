"""Alpha forecast adapter: wraps ml_builder panel output into typed AlphaForecast."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.forecast.contracts import AlphaArtifactHash, AlphaForecast


def _hash_payload(obj: Any) -> str:
    """Return 16-char hex SHA-256 of a JSON-serialisable object.

    Args:
        obj: Any JSON-serialisable value.

    Returns:
        16-char hex string.

    """
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def to_alpha_forecast(
    panel: pd.DataFrame,
    *,
    cfg_ml_dict: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    label_cfg_dict: dict[str, Any] | None = None,
    fold_specs_list: list[dict[str, Any]] | None = None,
) -> AlphaForecast:
    """Wrap an ml_builder alpha panel (long-format DataFrame) into a typed AlphaForecast.

    Values are NOT transformed — this is a pure typed adapter.

    Args:
        panel: Long-format DataFrame with MultiIndex (datetime, symbol) and
            columns alpha_long, alpha_short. attrs must contain config_hash,
            selected_horizon, model_family.
        cfg_ml_dict: Serialized ML config dict (for alpha_config_hash).
        feature_names: Feature name list (for feature_config_hash).
        label_cfg_dict: Label config dict (for label_config_hash).
        fold_specs_list: List of FoldSpec dicts (for fold_spec_hash/train_window_hash).

    Returns:
        Typed AlphaForecast contract.

    """
    attrs = panel.attrs

    alpha_config_hash: str = str(attrs.get("config_hash", ""))
    if not alpha_config_hash and cfg_ml_dict:
        alpha_config_hash = _hash_payload(cfg_ml_dict)

    if feature_names:
        feature_config_hash = _hash_payload({"feature_names": sorted(feature_names)})
    else:
        feature_config_hash = _hash_payload({"feature_names": [], "label": None})

    label_config_hash = _hash_payload(label_cfg_dict or {})
    fold_spec_hash = _hash_payload(fold_specs_list or [])

    train_window_hash = "unknown"
    if fold_specs_list:
        windows = [
            {"train_start": fs.get("train_start"), "train_end": fs.get("train_end")}
            for fs in fold_specs_list
        ]
        train_window_hash = _hash_payload(windows)

    artifact_hash = AlphaArtifactHash(
        alpha_config_hash=alpha_config_hash,
        feature_config_hash=feature_config_hash,
        label_config_hash=label_config_hash,
        train_window_hash=train_window_hash,
        fold_spec_hash=fold_spec_hash,
        model_family=str(attrs.get("model_family", "unknown")),
        selected_horizon=int(attrs.get("selected_horizon", -1)),
    )

    reset = panel.reset_index()
    datetimes_arr: np.ndarray = reset["datetime"].unique()
    symbols_tup: tuple[str, ...] = tuple(reset["symbol"].unique().tolist())
    t_len = len(datetimes_arr)
    n_len = len(symbols_tup)

    sym_to_col: dict[str, int] = {s: i for i, s in enumerate(symbols_tup)}
    dt_to_row: dict[Any, int] = {d: i for i, d in enumerate(datetimes_arr)}

    alpha_long_2d = np.zeros((t_len, n_len), dtype=np.float32)
    alpha_short_2d = np.zeros((t_len, n_len), dtype=np.float32)
    eligible_mask = np.zeros((t_len, n_len), dtype=bool)

    for row in reset.itertuples(index=False):
        t = dt_to_row.get(row.datetime)
        s = sym_to_col.get(row.symbol)
        if t is None or s is None:
            continue
        al = float(getattr(row, "alpha_long", 0.0))
        as_ = float(getattr(row, "alpha_short", 0.0))
        alpha_long_2d[t, s] = max(al, 0.0)
        alpha_short_2d[t, s] = max(as_, 0.0)
        eligible_mask[t, s] = np.isfinite(al) and np.isfinite(as_)

    meta: dict[str, Any] = attrs.get(
        "alpha_forecast_metadata",
        attrs.get(
            "forecast_metadata",
            attrs.get(
                "alpha_forecast_v3",
                attrs.get("forecast_metadata_v3", {}),
            ),
        ),
    )

    def _reshape(key: str) -> np.ndarray | None:
        arr = meta.get(key)
        if arr is None:
            return None
        flat = np.asarray(arr, dtype=np.float32)
        if flat.size == t_len * n_len:
            return flat.reshape(t_len, n_len)
        return None

    return AlphaForecast(
        datetimes=datetimes_arr,
        symbols=symbols_tup,
        alpha_long_2d=alpha_long_2d,
        alpha_short_2d=alpha_short_2d,
        q10_long_2d=_reshape("q10_long"),
        q50_long_2d=_reshape("q50_long"),
        q90_long_2d=_reshape("q90_long"),
        q10_short_2d=_reshape("q10_short"),
        q50_short_2d=_reshape("q50_short"),
        q90_short_2d=_reshape("q90_short"),
        confidence_long_2d=_reshape("confidence_long"),
        confidence_short_2d=_reshape("confidence_short"),
        eligible_mask=eligible_mask,
        source=str(attrs.get("strategy_name", "ml_builder")),
        artifact_hash=artifact_hash,
        rank_score_long_2d=_reshape("rank_score_long"),
        rank_score_short_2d=_reshape("rank_score_short"),
    )
