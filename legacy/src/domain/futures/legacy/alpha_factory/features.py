"""Safe feature extraction/normalization helpers for AlphaFactoryV1."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.domain.futures.alpha_factory.config import FeatureNormConfig
from src.domain.futures.alpha_factory.contracts import safe_div

_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "ret_24": ("ret_24",),
    "ret_6": ("ret_6",),
    "ret_12": ("ret_12",),
    "funding_z_72": ("funding_z_72",),
    "funding_rate": ("funding_rate",),
    "funding_mom_24": ("funding_mom_24",),
    "oi_momentum_24h": ("oi_momentum_24h",),
    "oi_price_divergence_24h": ("oi_price_divergence_24h",),
    "taker_imbalance_z_24": ("taker_imbalance_z_24",),
    "cvd_divergence_24h": ("cvd_divergence_24h",),
    "vpin_proxy_12": ("vpin_proxy_12",),
    "tail_risk_24": ("tail_risk_24",),
    "vol_surface_24_168": ("vol_surface_24_168", "macro_vol_regime_shift"),
    "idiosyncratic_return_24h": ("idiosyncratic_return_24h",),
    "btc_beta": ("btc_beta",),
    "range_pos_24": ("range_pos_24",),
}


def _as_series(source: pd.DataFrame | pd.Series | Mapping[str, float]) -> pd.Series:
    if isinstance(source, pd.DataFrame):
        if source.empty:
            return pd.Series(dtype=np.float64)
        return source.iloc[-1].astype(np.float64)
    if isinstance(source, pd.Series):
        return source.astype(np.float64)
    return pd.Series(source, dtype=np.float64)


def _bounded_z(x: float, clip_abs: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, -clip_abs, clip_abs) / max(clip_abs, 1e-12))


def extract_alpha_features(
    source: pd.DataFrame | pd.Series | Mapping[str, float],
    norm: FeatureNormConfig,
) -> dict[str, float]:
    """Extract required feature set from engineered feature row safely."""
    row = _as_series(source)
    out: dict[str, float] = {}
    for key, aliases in _FEATURE_ALIASES.items():
        v = 0.0
        for alias in aliases:
            if alias in row.index:
                raw = float(pd.to_numeric(row[alias], errors="coerce"))
                v = 0.0 if not np.isfinite(raw) else raw
                break
        out[key] = v

    # Stable transforms for bounded sleeves.
    out["ret_momentum"] = _bounded_z(out["ret_24"] + 0.5 * out["ret_6"], norm.clip_abs)
    out["ret_reversal"] = _bounded_z(-(out["ret_6"] - out["ret_24"]), norm.clip_abs)
    out["carry_pressure"] = _bounded_z(
        -(out["funding_z_72"] + 0.5 * out["funding_mom_24"]),
        norm.clip_abs,
    )
    out["flow_pressure"] = _bounded_z(
        0.55 * out["taker_imbalance_z_24"]
        + 0.30 * out["cvd_divergence_24h"]
        - 0.15 * out["vpin_proxy_12"],
        norm.clip_abs,
    )
    out["idio_edge"] = _bounded_z(
        out["idiosyncratic_return_24h"] - 0.35 * out["btc_beta"],
        norm.clip_abs,
    )

    vol_guard = 1.0 - np.clip(out["tail_risk_24"], 0.0, 1.5)
    out["vol_guard"] = float(np.clip(vol_guard, -1.0, 1.0))
    quality_raw = 0.5 + 0.5 * safe_div(out["range_pos_24"] - 0.5, 0.5, eps=norm.eps)
    out["quality"] = float(np.clip(quality_raw, 0.0, 1.0))
    return out
