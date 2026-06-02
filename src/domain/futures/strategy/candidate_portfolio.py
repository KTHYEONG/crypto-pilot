from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, project_all_caps
from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _candidate_variant_key(frame: pd.DataFrame) -> str:
    return f"{frame['family'].iloc[0]!s}:{frame['variant'].iloc[0]!s}"


def _q10_mask_for_mode(df: pd.DataFrame, cfg: CandidateStrategyConfig) -> pd.Series:
    if cfg.selection_shortfall_mode == "penalty_only":
        return pd.Series(True, index=df.index, dtype=bool)
    if cfg.selection_shortfall_mode == "catastrophic":
        return df["q10_net_bps"] >= -cfg.catastrophic_shortfall_bps
    return df["q10_net_bps"] >= -cfg.max_expected_shortfall_bps


def _catastrophic_q10_mask(df: pd.DataFrame, cfg: CandidateStrategyConfig) -> pd.Series:
    return df["q10_net_bps"] >= -cfg.catastrophic_shortfall_bps


def _utility_threshold(
    *,
    df: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    model_output: CandidateModelOutput,
) -> float:
    threshold = model_output.selection_thresholds.get("utility_min")
    if threshold is not None and np.isfinite(threshold):
        return float(threshold)
    finite = pd.to_numeric(df["utility_score"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("-inf")
    quantile = max(0.0, min(1.0, 1.0 - float(cfg.selection_top_quantile)))
    return float(np.quantile(finite, quantile))


def compute_selection_sensitivity(
    *,
    events: pd.DataFrame,
    gate_grid: tuple[float, ...],
    edge_grid_bps: tuple[float, ...],
    q10_grid_bps: tuple[float, ...],
) -> pd.DataFrame:
    """Return pass counts across gate, edge, and q10 threshold grids."""
    if events.empty:
        return pd.DataFrame(
            columns=[
                "gate_threshold",
                "edge_threshold_bps",
                "q10_shortfall_bps",
                "total",
                "gate_pass",
                "edge_pass",
                "q10_pass",
                "all_pass",
                "all_pass_rate",
                "top_variant",
                "top_variant_pass",
            ]
        )

    records: list[dict[str, float | int | str]] = []
    total = int(events.shape[0])
    variant_keys = (
        events["family"].astype(str).str.cat(events["variant"].astype(str), sep=":")
        if {"family", "variant"}.issubset(events.columns)
        else pd.Series([""] * total, index=events.index, dtype="object")
    )
    for gate_threshold in gate_grid:
        gate_mask = events["p_pass"] >= gate_threshold
        for edge_threshold in edge_grid_bps:
            edge_mask = events["mu_net_decision_bps"] >= edge_threshold
            for q10_threshold in q10_grid_bps:
                q10_mask = events["q10_net_bps"] >= -q10_threshold
                all_mask = gate_mask & edge_mask & q10_mask
                passed = variant_keys.loc[all_mask]
                if passed.empty:
                    top_variant = ""
                    top_variant_pass = 0
                else:
                    counts = passed.value_counts(sort=True)
                    top_variant = str(counts.index[0])
                    top_variant_pass = int(counts.iloc[0])
                records.append(
                    {
                        "gate_threshold": float(gate_threshold),
                        "edge_threshold_bps": float(edge_threshold),
                        "q10_shortfall_bps": float(q10_threshold),
                        "total": total,
                        "gate_pass": int(gate_mask.sum()),
                        "edge_pass": int(edge_mask.sum()),
                        "q10_pass": int(q10_mask.sum()),
                        "all_pass": int(all_mask.sum()),
                        "all_pass_rate": float(all_mask.mean()),
                        "top_variant": top_variant,
                        "top_variant_pass": top_variant_pass,
                    }
                )
    return pd.DataFrame.from_records(records)


def _log_selection_sensitivity(df: pd.DataFrame, *, cfg: CandidateStrategyConfig) -> None:
    if df.empty:
        return
    logger = logging.getLogger(__name__)
    top = df.sort_values(
        ["all_pass", "all_pass_rate", "gate_threshold", "edge_threshold_bps", "q10_shortfall_bps"],
        ascending=[False, False, True, True, True],
    ).head(max(1, int(cfg.diagnostic_top_k)))
    for row in top.itertuples(index=False):
        logger.info(
            "[DIAG][SELECT_SENS] gate>=%.2f edge>=%.1f q10>=-%.1f passed=%d pass_rate=%.4f top_variant=%s top_pass=%d",
            float(row.gate_threshold),
            float(row.edge_threshold_bps),
            float(row.q10_shortfall_bps),
            int(row.all_pass),
            float(row.all_pass_rate),
            str(row.top_variant),
            int(row.top_variant_pass),
        )


def _log_selection_by_variant(
    *,
    df: pd.DataFrame,
    gate_mask: pd.Series,
    edge_mask: pd.Series,
    q10_mask: pd.Series,
    pass_mask: pd.Series,
    cfg: CandidateStrategyConfig,
) -> None:
    """Log grouped candidate selection failure reasons."""
    if df.empty:
        return

    logger = logging.getLogger(__name__)
    grouped = df.groupby(["family", "variant"], sort=False, dropna=False)
    rows: list[tuple[str, int, int, int, int, int, float, float, float]] = []
    for (family, variant), group in grouped:
        idx = group.index
        rows.append(
            (
                f"{family}:{variant}",
                int(group.shape[0]),
                int((~gate_mask.loc[idx]).sum()),
                int((~edge_mask.loc[idx]).sum()),
                int((~q10_mask.loc[idx]).sum()),
                int(pass_mask.loc[idx].sum()),
                float(pd.to_numeric(group["p_pass"], errors="coerce").mean()),
                float(pd.to_numeric(group["mu_net_decision_bps"], errors="coerce").max()),
                float(pd.to_numeric(group["q10_net_bps"], errors="coerce").max()),
            )
        )

    for key, total, gate_fail, edge_fail, q10_fail, passed, mean_p, max_mu, max_q10 in sorted(
        rows,
        key=lambda item: item[1],
        reverse=True,
    )[: max(1, int(getattr(cfg, "diagnostic_top_k", 10)))]:
        logger.info(
            (
                "[DIAG][SELECT_VARIANT] key=%s total=%d gate_fail=%d edge_fail=%d "
                "q10_fail=%d passed=%d mean_p=%.3f max_mu=%.1f max_q10=%.1f"
            ),
            key,
            total,
            gate_fail,
            edge_fail,
            q10_fail,
            passed,
            mean_p,
            max_mu,
            max_q10,
        )


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
        return pd.DataFrame(
            columns=[
                "datetime",
                "symbol",
                "family",
                "variant",
                "side",
                "raw_score",
                "score_z",
                "expected_holding_bars",
                "min_holding_bars",
                "stop_atr_mult",
                "take_profit_atr_mult",
                "turnover_proxy",
                "cost_floor_bps",
                "entry_idx",
                "side_flipped",
                "p_pass",
                "mu_net_decision_bps",
                "q10_net_bps",
                "utility_score",
            ]
        )

    df = events.copy()
    df["p_pass"] = np.asarray(model_output.p_pass, dtype=np.float64)
    df["mu_net_decision_bps"] = np.asarray(model_output.mu_net_decision_bps, dtype=np.float64)
    df["q10_net_bps"] = np.asarray(model_output.q10_net_bps, dtype=np.float64)
    df["utility_score"] = np.asarray(model_output.utility_score, dtype=np.float64)

    if cfg.selection_sensitivity_enabled:
        sensitivity = compute_selection_sensitivity(
            events=df,
            gate_grid=cfg.selection_gate_grid,
            edge_grid_bps=cfg.selection_edge_grid_bps,
            q10_grid_bps=cfg.selection_q10_grid_bps,
        )
        _log_selection_sensitivity(sensitivity, cfg=cfg)

    gate_mask = df["p_pass"] >= cfg.min_gate_probability
    edge_mask = df["mu_net_decision_bps"] >= cfg.min_expected_net_bps
    q10_mask = _q10_mask_for_mode(df, cfg)
    catastrophic_mask = _catastrophic_q10_mask(df, cfg)
    utility_threshold = _utility_threshold(df=df, cfg=cfg, model_output=model_output)
    utility_mask = df["utility_score"] >= utility_threshold

    if cfg.selection_policy == "hard":
        mask = gate_mask & edge_mask & q10_mask
    elif cfg.selection_policy == "validation_quantile":
        mask = catastrophic_mask & (df["mu_net_decision_bps"] >= 0.0) & utility_mask
    else:
        # utility_topk: safety filter then rank by utility, keep top fraction
        eligible = catastrophic_mask & (df["mu_net_decision_bps"] >= 0.0)
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            mask = eligible
        else:
            n_keep = max(1, math.ceil(n_eligible * cfg.selection_top_quantile))
            utility_vals = pd.to_numeric(df.loc[eligible, "utility_score"], errors="coerce")
            top_idx = utility_vals.nlargest(n_keep).index
            mask = pd.Series(False, index=df.index, dtype=bool)
            mask.loc[top_idx] = True

    _log_selection_by_variant(
        df=df,
        gate_mask=gate_mask,
        edge_mask=edge_mask,
        q10_mask=catastrophic_mask if cfg.selection_policy != "hard" else q10_mask,
        pass_mask=mask,
        cfg=cfg,
    )

    _sel_logger = logging.getLogger(__name__)
    _sel_logger.info(
        "[DIAG][SELECT] total=%d gate_fail=%d edge_fail=%d q10_fail=%d "
        "all_fail=%d passed=%d | policy=%s thresholds(gate>=%.2f edge_net>=%.1f q10>=-%.1f utility>=%.3f)",
        len(df),
        int((~gate_mask).sum()),
        int((~edge_mask).sum()),
        int((~(catastrophic_mask if cfg.selection_policy != "hard" else q10_mask)).sum()),
        int((~mask).sum()),
        int(mask.sum()),
        cfg.selection_policy,
        cfg.min_gate_probability,
        cfg.min_expected_net_bps,
        cfg.catastrophic_shortfall_bps if cfg.selection_policy != "hard" else cfg.max_expected_shortfall_bps,
        utility_threshold,
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

    # ---------- Pass 1: entry_idx bar에 raw_weight 기록 ----------
    # 각 이벤트의 (entry_idx, symbol) → raw_weight 를 먼저 매핑한다.
    # 타임스텝 정렬된 리스트도 함께 수집해 Pass 2에서 재사용한다.
    # event_records: list of (entry_idx, s_idx, raw_signed_weight, holding_bars)
    event_records: list[tuple[int, int, float, int]] = []

    for row in selected_events.itertuples(index=False):
        sym = str(row.symbol)
        if sym not in sym_to_idx:
            continue
        s_idx = sym_to_idx[sym]
        t = int(row.entry_idx)
        if t < 0 or t >= n_times:
            continue

        side = float(row.side)
        # Normalise expected edge to per-bar scale before Kelly calculation.
        # mu_net_decision_bps is a per-HORIZON figure; dividing by holding_bars
        # converts it to per-bar, matching the per-bar variance denominator.
        holding_bars = max(int(getattr(row, "expected_holding_bars", 1)), 1)
        mu_i_per_bar = float(row.mu_net_decision_bps) * 1e-4 / holding_bars

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
        # Fractional Kelly: raw_weight = kelly_fraction * mu_i_per_bar / variance_i
        raw_w = cfg.kelly_fraction * mu_i_per_bar / variance_i
        signed_w = raw_w * np.sign(side)
        raw_weights[t, s_idx] = signed_w
        event_records.append((t, s_idx, signed_w, holding_bars))

    # ---------- Pass 2: entry_idx → entry_idx + holding_bars - 1 구간 forward-fill ----------
    # 타임스텝 오름차순 정렬 후 순회: 이미 비영값(다른 이벤트로 채워진 구간)은 덮어쓰지 않는다.
    event_records.sort(key=lambda r: r[0])
    for entry_t, s_idx, signed_w, holding_bars in event_records:
        fill_end = min(entry_t + holding_bars, n_times)
        for fill_t in range(entry_t + 1, fill_end):
            if raw_weights[fill_t, s_idx] == 0.0:
                raw_weights[fill_t, s_idx] = signed_w

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
        stop_atr_mult = np.zeros(n_symbols, dtype=np.float64)
        take_profit_atr_mult = np.zeros(n_symbols, dtype=np.float64)

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
                    stop_atr_mult[s_idx] = float(getattr(row, "stop_atr_mult", 0.0))
                    take_profit_atr_mult[s_idx] = float(getattr(row, "take_profit_atr_mult", 0.0))

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
            "candidate_stop_atr_mult": stop_atr_mult,
            "candidate_take_profit_atr_mult": take_profit_atr_mult,
        })
        rows.append(df_t)

    if not rows:
        empty_df = pd.DataFrame(columns=[
            "alpha_long", "alpha_short", "target_weight", "candidate_family",
            "candidate_variant", "p_pass", "mu_net_decision_bps", "q10_net_bps", "utility_score",
            "candidate_stop_atr_mult", "candidate_take_profit_atr_mult",
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
