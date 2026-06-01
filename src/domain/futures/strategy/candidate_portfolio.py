from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps
from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig


def select_candidate_events_for_portfolio(
    *,
    model_output: CandidateModelOutput,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Select at most one active candidate per symbol per timestamp.

    Filters candidates by gate and edge criteria, resolving long/short conflicts.
    """
    events = model_output.events
    if events is None or events.empty:
        return pd.DataFrame(columns=[
            "datetime", "symbol", "family", "variant", "side", "raw_score", "score_z",
            "expected_holding_bars", "min_holding_bars", "stop_atr_mult", "take_profit_atr_mult",
            "turnover_proxy", "cost_floor_bps", "entry_idx", "p_pass", "mu_net_decision_bps",
            "q10_net_bps", "utility_score"
        ])

    df = events.copy()
    df["p_pass"] = np.asarray(model_output.p_pass, dtype=np.float64)
    df["mu_net_decision_bps"] = np.asarray(model_output.mu_net_decision_bps, dtype=np.float64)
    df["q10_net_bps"] = np.asarray(model_output.q10_net_bps, dtype=np.float64)
    df["utility_score"] = np.asarray(model_output.utility_score, dtype=np.float64)

    # Apply gate and edge threshold filters
    mask = (
        (df["p_pass"] >= cfg.min_gate_probability) &
        (df["mu_net_decision_bps"] >= cfg.min_expected_net_bps) &
        (df["q10_net_bps"] >= -cfg.max_expected_shortfall_bps)
    )
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return filtered

    # Ensure datetime format is uniform for grouping
    filtered["_merge_dt"] = pd.to_datetime(filtered["datetime"], utc=True).dt.tz_localize(None)

    # Sort to resolve conflicts: pick highest utility score first
    filtered = filtered.sort_values(["_merge_dt", "symbol", "utility_score"], ascending=[True, True, False])
    
    # Per (datetime, symbol), pick the variant with the highest utility
    selected = filtered.groupby(["_merge_dt", "symbol"], as_index=False).first()
    
    return selected.drop(columns=["_merge_dt"]).reset_index(drop=True)


def build_candidate_target_weights(
    *,
    selected_events: pd.DataFrame,
    close_2d: NDArray[np.float64],
    symbols: tuple[str, ...],
    beta_2d: NDArray[np.float64] | None,
    sigma_3d: NDArray[np.float64] | None,
    cfg: CandidateStrategyConfig,
) -> NDArray[np.float64]:
    """Build target_weights_2d for the backtest engine using Fractional Kelly & Caps."""
    n_times, n_symbols = close_2d.shape
    raw_weights = np.zeros((n_times, n_symbols), dtype=np.float64)

    if selected_events.empty:
        return raw_weights

    # Map symbols to index
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}

    # Group selected events by entry_idx so metadata lands on the execution bar.
    for row in selected_events.itertuples(index=False):
        sym = str(row.symbol)
        if sym not in sym_to_idx:
            continue
        s_idx = sym_to_idx[sym]
        t = int(row.entry_idx)
        if t < 0 or t >= n_times:
            continue

        side = float(row.side)
        # Expected edge return converted to return fraction per bar scale
        mu_i = float(row.mu_net_decision_bps) * 1e-4

        # Trailing variance retrieval
        variance_i = 1e-4  # Default fallback
        if sigma_3d is not None:
            # sigma_3d shape is usually [T, N, N] covariance matrix
            variance_i = float(sigma_3d[t, s_idx, s_idx])
        else:
            # Fallback trailing close returns std
            st = max(0, t - 20)
            if t > st:
                ret = np.diff(close_2d[st : t + 1, s_idx]) / np.maximum(close_2d[st:t, s_idx], 1e-12)
                v = float(np.var(ret))
                if np.isfinite(v) and v > 1e-12:
                    variance_i = v

        variance_i = max(variance_i, 1e-12)
        # Fractional Kelly: raw_weight = kelly_fraction * mu_i / variance_i
        raw_w = cfg.kelly_fraction * mu_i / variance_i
        raw_weights[t, s_idx] = raw_w * np.sign(side)

    # Apply 5-cap multi-cap projection per timestamp
    caps = PortfolioCaps(
        gross=cfg.gross_cap,
        per_symbol=cfg.max_symbol_weight,
        net=cfg.net_cap,
        beta=cfg.beta_cap,
        target_ann_vol=cfg.target_ann_vol,
    )

    target_weights = np.zeros_like(raw_weights)
    bars_per_year = 2190.0  # Default 4h bars per year (365 * 6)
    if cfg.timeframe == "1h":
        bars_per_year = 8760.0
    elif cfg.timeframe == "1d":
        bars_per_year = 365.0

    for t in range(n_times):
        w_pre = raw_weights[t]
        beta_t = beta_2d[t] if beta_2d is not None else np.zeros(n_symbols)
        
        # Portfolio standard deviation calculation
        sigma_port_t = 1e-3
        if sigma_3d is not None:
            cov = sigma_3d[t]
            var_port = float(np.dot(w_pre, np.dot(cov, w_pre)))
            if np.isfinite(var_port) and var_port > 0.0:
                sigma_port_t = math.sqrt(var_port)
        else:
            # Simple vol target fallback standard deviation
            sigma_port_t = float(np.nanstd(w_pre)) if np.any(w_pre) else 1e-3

        target_weights[t] = project_all_caps(
            w=w_pre,
            btc_beta=beta_t,
            sigma_port=sigma_port_t,
            bars_per_year=bars_per_year,
            caps=caps,
        )

    return target_weights


def build_candidate_alpha_panel(
    *,
    selected_events: pd.DataFrame,
    target_weights_2d: NDArray[np.float64],
    datetimes: NDArray[np.datetime64],
    symbols: tuple[str, ...],
) -> pd.DataFrame:
    """Build long-format panel for merge into data maps."""
    n_times, n_symbols = target_weights_2d.shape
    rows: list[pd.DataFrame] = []

    # Map symbols to index
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}

    # Group by execution index so metadata aligns with the target weight row.
    df_selected = selected_events.copy()
    df_selected["_entry_idx"] = df_selected["entry_idx"].astype(int)
    grouped = df_selected.groupby("_entry_idx")

    for t in range(n_times):
        # Default empty attributes
        alpha_long = np.zeros(n_symbols, dtype=np.float64)
        alpha_short = np.zeros(n_symbols, dtype=np.float64)
        target_w = target_weights_2d[t]

        # Extract direction components
        alpha_long[target_w > 0.0] = target_w[target_w > 0.0]
        alpha_short[target_w < 0.0] = -target_w[target_w < 0.0]

        families = [""] * n_symbols
        variants = [""] * n_symbols
        p_pass = np.zeros(n_symbols, dtype=np.float64)
        mu_bps = np.zeros(n_symbols, dtype=np.float64)
        q10_bps = np.zeros(n_symbols, dtype=np.float64)
        utility = np.zeros(n_symbols, dtype=np.float64)

        if t in grouped.groups:
            dt_group = grouped.get_group(t)
            for row in dt_group.itertuples(index=False):
                sym = str(row.symbol)
                if sym in sym_to_idx:
                    s_idx = sym_to_idx[sym]
                    families[s_idx] = str(row.family)
                    variants[s_idx] = str(row.variant)
                    p_pass[s_idx] = float(row.p_pass)
                    mu_bps[s_idx] = float(row.mu_net_decision_bps)
                    q10_bps[s_idx] = float(row.q10_net_bps)
                    utility[s_idx] = float(row.utility_score)

        df_t = pd.DataFrame({
            "datetime": datetimes[t],
            "symbol": list(symbols),
            "alpha_long": alpha_long,
            "alpha_short": alpha_short,
            "target_weight": target_w,
            "candidate_family": families,
            "candidate_variant": variants,
            "p_pass": p_pass,
            "mu_net_decision_bps": mu_bps,
            "q10_net_bps": q10_bps,
            "utility_score": utility,
        })
        rows.append(df_t)

    if not rows:
        empty_df = pd.DataFrame(columns=[
            "alpha_long", "alpha_short", "target_weight", "candidate_family",
            "candidate_variant", "p_pass", "mu_net_decision_bps", "q10_net_bps", "utility_score"
        ])
        empty_df.index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="datetime64[ns]"), pd.Index([], dtype="object")],
            names=["datetime", "symbol"]
        )
        return empty_df

    panel = (
        pd.concat(rows, axis=0, ignore_index=True)
        .set_index(["datetime", "symbol"])
        .sort_index()
    )
    return panel


