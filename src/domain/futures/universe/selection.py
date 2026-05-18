"""Stage 6: final selection and ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage6Config

ANCHORS = ("BTCUSDT", "ETHUSDT")
DEFAULT_MAX_SYMBOLS = 24
DEFAULT_K_IN = 18
DEFAULT_K_OUT = 30
DEFAULT_W_LIQ = 0.40
DEFAULT_W_COST_INV = 0.30
DEFAULT_W_QUALITY = 0.20
DEFAULT_W_STABILITY = 0.10
RETURN_VECTOR_COLUMNS = (
    "return_vector",
    "recent_returns",
    "returns",
    "return_series",
    "ret_series",
)
CLUSTER_ID_SOURCE_COLUMNS = ("cluster_id", "corr_cluster_id", "correlation_cluster_id")
CORR_CLUSTER_THRESHOLD = 0.70


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _symbol_key(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _normalize_unit(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    low = float(valid.min())
    high = float(valid.max())
    if np.isclose(high, low):
        return pd.Series(0.5, index=series.index)
    scaled = (numeric - low) / (high - low)
    return scaled.fillna(0.0).clip(lower=0.0, upper=1.0)


def _to_return_vector(value: object) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        vector = np.asarray(value, dtype=float)
    elif isinstance(value, (list, tuple)):
        vector = np.asarray(value, dtype=float)
    else:
        return None
    if vector.ndim != 1 or vector.size < 2:
        return None
    vector = vector[np.isfinite(vector)]
    if vector.size < 2:
        return None
    return np.asarray(vector, dtype=float)


def _first_available_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _compute_beta_vs_market(
    out: pd.DataFrame,
    *,
    basket_ref: tuple[str, ...],
    basket_weights: tuple[float, ...],
) -> pd.Series:
    base = pd.to_numeric(
        out.get("beta_vs_market", pd.Series(0.0, index=out.index)),
        errors="coerce",
    )
    fallback = base.fillna(0.0)
    vector_column = _first_available_column(out, RETURN_VECTOR_COLUMNS)
    if vector_column is None or not basket_ref:
        return fallback
    vectors = out[vector_column].map(_to_return_vector)
    ref_weight_pairs = list(zip(basket_ref, basket_weights, strict=False))
    if not ref_weight_pairs:
        return fallback
    ref_returns: list[np.ndarray] = []
    ref_weights: list[float] = []
    for ref_symbol, weight in ref_weight_pairs:
        ref_rows = out.loc[out["_symbol_key"] == _symbol_key(ref_symbol)]
        if ref_rows.empty:
            continue
        ref_vector = vectors.loc[ref_rows.index[0]]
        if ref_vector is None:
            continue
        weight_float = float(weight)
        if not np.isfinite(weight_float) or weight_float <= 0.0:
            continue
        ref_returns.append(ref_vector)
        ref_weights.append(weight_float)
    if not ref_returns:
        return fallback
    min_len = min(vector.size for vector in ref_returns)
    if min_len < 2:
        return fallback
    ref_matrix = np.vstack([vector[-min_len:] for vector in ref_returns])
    weights = np.asarray(ref_weights, dtype=float)
    weights = weights / weights.sum()
    market = weights @ ref_matrix
    market_var = float(np.var(market))
    if market_var <= 0.0 or not np.isfinite(market_var):
        return fallback
    available_idx = [
        idx
        for idx, vector in vectors.items()
        if vector is not None and vector.size >= min_len
    ]
    if not available_idx:
        return fallback
    symbol_matrix = np.vstack([vectors.loc[idx][-min_len:] for idx in available_idx])
    symbol_centered = symbol_matrix - symbol_matrix.mean(axis=1, keepdims=True)
    market_centered = market - market.mean()
    covariances = (symbol_centered @ market_centered) / float(min_len)
    beta_values = covariances / market_var
    beta_series = fallback.copy()
    beta_series.loc[available_idx] = beta_values
    return beta_series


def _compute_cluster_ids(
    out: pd.DataFrame,
    *,
    corr_threshold: float = CORR_CLUSTER_THRESHOLD,
) -> pd.Series:
    fallback = pd.Series(-1, index=out.index, dtype=int)
    source_col = _first_available_column(out, CLUSTER_ID_SOURCE_COLUMNS)
    if source_col is not None:
        return pd.to_numeric(out[source_col], errors="coerce").fillna(-1).astype(int)
    vector_column = _first_available_column(out, RETURN_VECTOR_COLUMNS)
    if vector_column is None:
        return fallback
    vectors = out[vector_column].map(_to_return_vector)
    valid_idx = [idx for idx, vector in vectors.items() if vector is not None]
    if len(valid_idx) < 2:
        return fallback
    min_len = min(vectors.loc[idx].size for idx in valid_idx)
    if min_len < 2:
        return fallback
    matrix = np.vstack([vectors.loc[idx][-min_len:] for idx in valid_idx])
    corr = np.corrcoef(matrix)
    if corr.ndim != 2:
        return fallback
    adjacency = np.isfinite(corr) & (corr >= corr_threshold)
    np.fill_diagonal(adjacency, True)
    assigned = np.full(len(valid_idx), -1, dtype=int)
    cluster_id = 0
    for row_idx in range(len(valid_idx)):
        if assigned[row_idx] != -1:
            continue
        frontier = [row_idx]
        assigned[row_idx] = cluster_id
        while frontier:
            node = frontier.pop()
            neighbors = np.flatnonzero(adjacency[node] & (assigned == -1))
            if neighbors.size == 0:
                continue
            assigned[neighbors] = cluster_id
            frontier.extend(neighbors.tolist())
        cluster_id += 1
    cluster_series = fallback.copy()
    cluster_series.loc[valid_idx] = assigned
    return cluster_series


def apply_selection_stage(
    frame: pd.DataFrame,
    *,
    config: Stage6Config | None = None,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    previous_selection: tuple[str, ...] | None = None,
    k_in: int = DEFAULT_K_IN,
    k_out: int = DEFAULT_K_OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank survivors, enforce anchors, and return selected symbols."""
    if frame.empty:
        empty_report = pd.DataFrame(columns=["symbol", "stage", "passed", "reason", "rank"])
        return frame.copy(), empty_report
    cfg = config or Stage6Config()
    if config is not None:
        max_symbols = int(config.k_in)
        k_in = int(config.k_in)
        k_out = int(config.k_out)

    anchors = tuple(cfg.anchor_symbols) if cfg.anchor_symbols else ANCHORS
    anchor_keys = {_symbol_key(anchor): anchor for anchor in anchors}
    min_dwell_days = int(cfg.min_dwell_days)
    out = frame.copy()
    out["symbol"] = out.get("symbol", pd.Series("", index=out.index)).astype("string")
    out["_symbol_key"] = out["symbol"].astype(str).map(_symbol_key)
    out["cluster_id"] = _compute_cluster_ids(out, corr_threshold=float(cfg.corr_cluster_threshold))
    out["beta_vs_market"] = _compute_beta_vs_market(
        out,
        basket_ref=tuple(cfg.basket_ref),
        basket_weights=tuple(cfg.basket_weights),
    )
    liq_norm = _normalize_unit(out.get("adv_usdt_median", pd.Series(0.0, index=out.index)))
    execution_cost = pd.to_numeric(
        out.get("execution_cost_bps", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    positive_cost = execution_cost.where(execution_cost > 0.0, np.nan)
    cost_inv_norm = _normalize_unit(1.0 / positive_cost)
    quality_norm = _normalize_unit(out.get("last_60d_coverage", pd.Series(0.0, index=out.index)))
    stability_norm = _normalize_unit(out.get("listing_age_days", pd.Series(0.0, index=out.index)))
    out["tradeable_score"] = (
        (DEFAULT_W_LIQ * liq_norm)
        + (DEFAULT_W_COST_INV * cost_inv_norm)
        + (DEFAULT_W_QUALITY * quality_norm)
        + (DEFAULT_W_STABILITY * stability_norm)
    )

    out = out.sort_values("tradeable_score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["hysteresis_state"] = "candidate"
    selected = out.head(max_symbols).copy()
    selected["hysteresis_state"] = "selected_k_in"

    if previous_selection:
        prev_keys = {_symbol_key(symbol) for symbol in previous_selection}
        prev_mask = out["_symbol_key"].isin(prev_keys)
        kout_threshold = int(max(k_out, max_symbols))
        within_kout = out["rank"] <= kout_threshold
        dwell_col = next(
            (col for col in ("membership_days", "dwell_days") if col in out.columns),
            None,
        )
        if dwell_col is None:
            retained_mask = prev_mask & within_kout
            retained = out.loc[retained_mask].copy()
            retained["hysteresis_state"] = "retained_k_out"
        else:
            dwell_days = pd.to_numeric(out[dwell_col], errors="coerce")
            under_min_dwell = dwell_days < float(min_dwell_days)
            retained_mask = prev_mask & (within_kout | under_min_dwell.fillna(False))
            retained = out.loc[retained_mask].copy()
            retained["hysteresis_state"] = np.where(
                retained["rank"] <= kout_threshold,
                "retained_k_out",
                "retained_min_dwell",
            )
        selected = selected[selected["rank"] <= int(min(k_in, max_symbols))]
        selected = (
            pd.concat([selected, retained], ignore_index=True)
            .drop_duplicates("_symbol_key", keep="first")
            .copy()
        )

    selected_keys = set(selected["_symbol_key"].tolist())
    next_rank = int(out["rank"].max()) + 1
    for anchor_key, anchor_symbol in anchor_keys.items():
        if anchor_key in selected_keys:
            continue
        anchor_rows = out[out["_symbol_key"] == anchor_key].copy()
        if anchor_rows.empty:
            synth_payload: dict[str, object] = {col: np.nan for col in out.columns}
            synth_payload["symbol"] = anchor_symbol
            synth_payload["_symbol_key"] = anchor_key
            synth_payload["tradeable_score"] = float("-inf")
            synth_payload["rank"] = next_rank
            synth_payload["hysteresis_state"] = "anchor_forced"
            synth_payload["cluster_id"] = -1
            synth_payload["beta_vs_market"] = 0.0
            anchor_rows = pd.DataFrame([synth_payload], columns=out.columns)
            next_rank += 1
        else:
            anchor_rows = anchor_rows.sort_values("tradeable_score", ascending=False).head(1)
            anchor_rows["hysteresis_state"] = "anchor_forced"
        selected = (
            pd.concat([selected, anchor_rows], ignore_index=True)
            .drop_duplicates("_symbol_key", keep="first")
            .copy()
        )
        selected_keys.add(anchor_key)

    selected["_is_anchor"] = selected["_symbol_key"].isin(set(anchor_keys))
    anchor_priority = {key: idx for idx, key in enumerate(anchor_keys)}
    selected["_anchor_priority"] = selected["_symbol_key"].map(anchor_priority).fillna(10_000)
    anchor_rows = selected[selected["_is_anchor"]].sort_values(
        "_anchor_priority", ascending=True
    )
    regular_rows = selected[~selected["_is_anchor"]].sort_values(
        "tradeable_score", ascending=False
    )
    if len(anchor_rows) >= max_symbols:
        selected = anchor_rows.head(max_symbols).copy()
    else:
        selected = pd.concat(
            [anchor_rows, regular_rows.head(max_symbols - len(anchor_rows))],
            ignore_index=True,
        )
    selected["role"] = np.where(
        selected["_is_anchor"],
        "anchor",
        "regular",
    )
    selected_keys = set(selected["_symbol_key"].tolist())
    anchor_key_set = set(anchor_keys)
    reason = np.where(
        out["_symbol_key"].isin(selected_keys),
        np.where(
            out["_symbol_key"].isin(anchor_key_set),
            "anchor_selected",
            "selected",
        ),
        "not_selected",
    )
    report = pd.DataFrame(
        {
            "symbol": out["symbol"].astype("string"),
            "stage": "stage6_selection",
            "passed": out["_symbol_key"].isin(selected_keys),
            "reason": pd.Series(reason, index=out.index, dtype="string"),
            "rank": out["rank"].astype(int),
        }
    )
    missing_anchor_keys = selected_keys.difference(set(out["_symbol_key"].tolist()))
    if missing_anchor_keys:
        missing_rows = selected[selected["_symbol_key"].isin(missing_anchor_keys)].copy()
        if not missing_rows.empty:
            missing_report = pd.DataFrame(
                {
                    "symbol": missing_rows["symbol"].astype("string"),
                    "stage": "stage6_selection",
                    "passed": True,
                    "reason": "anchor_selected",
                    "rank": missing_rows["rank"].astype(int),
                }
            )
            report = pd.concat([report, missing_report], ignore_index=True)
    selected = selected.drop(
        columns=["_symbol_key", "_is_anchor", "_anchor_priority"], errors="ignore"
    )
    return selected, report
