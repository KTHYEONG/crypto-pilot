from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.backtest.engine import FuturesBacktestEngine
from src.domain.futures.strategy.candidate_dataset import build_candidate_dataset
from src.domain.futures.strategy.candidate_edge import fit_candidate_edge_models, predict_candidate_edges
from src.domain.futures.strategy.candidate_evaluation import evaluate_compound_backtest
from src.domain.futures.strategy.candidate_gate import fit_candidate_gate, predict_candidate_gate
from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.candidate_portfolio import (
    build_candidate_target_weights,
    select_candidate_events_for_portfolio,
)
from src.domain.futures.strategy.common.alignment import align_data_maps
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.rule_diagnostics import compute_rule_diagnostics
from src.domain.futures.strategy.rule_signals import build_rule_signal_panels, candidate_panels_to_events

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AblationRow:
    """Represents a row in the ablation comparison study."""

    variant: str
    mean_log_growth: float
    cagr: float
    max_drawdown: float
    mar: float
    turnover: float
    final_equity: float
    pass_compound_gate: bool


def _build_rule_equal_size_weights(
    *,
    raw_events: pd.DataFrame,
    close_2d: np.ndarray,
    symbols: tuple[str, ...],
    max_symbol_weight: float,
) -> np.ndarray:
    raw_w = np.zeros_like(close_2d)
    n_bars = close_2d.shape[0]
    for row in raw_events.itertuples(index=False):
        t = int(row.entry_idx)
        for s_idx, sym in enumerate(symbols):
            if sym == row.symbol and 0 <= t < n_bars:
                raw_w[t, s_idx] = float(row.side) * max_symbol_weight
    return raw_w


def _build_uncapped_kelly_edge_weights(
    *,
    selected_events: pd.DataFrame,
    close_2d: np.ndarray,
    symbols: tuple[str, ...],
    kelly_fraction: float,
) -> np.ndarray:
    n_times, n_symbols = close_2d.shape
    raw_kelly_edge_w = np.zeros((n_times, n_symbols), dtype=np.float64)
    sym_to_idx = {sym: idx for idx, sym in enumerate(symbols)}
    for row in selected_events.itertuples(index=False):
        sym = str(row.symbol)
        if sym not in sym_to_idx:
            continue
        s_idx = sym_to_idx[sym]
        t = int(row.entry_idx)
        if not (0 <= t < n_times):
            continue

        side = float(row.side)
        holding_bars = max(int(getattr(row, "expected_holding_bars", 1)), 1)
        mu_i_per_bar = float(row.mu_net_decision_bps) * 1e-4 / holding_bars

        st = max(0, t - 20)
        variance_i = 1e-4
        if t > st:
            ret = np.diff(close_2d[st : t + 1, s_idx]) / np.maximum(close_2d[st:t, s_idx], 1e-12)
            v = float(np.var(ret))
            if np.isfinite(v) and v > 1e-12:
                variance_i = v
        raw_w_val = kelly_fraction * mu_i_per_bar / max(variance_i, 1e-12)
        raw_kelly_edge_w[t, s_idx] = raw_w_val * np.sign(side)
    return raw_kelly_edge_w


def run_candidate_ablation(
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: tuple[str, ...],
    tf: str,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Run ablation variants to prove each complexity layer adds compounding value."""
    # 1. Align market data
    aligned = align_data_maps(data_maps, list(symbols), tf)
    
    # 2. Build hypothesis rule candidates
    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg)
    raw_events = candidate_panels_to_events(
        panels,
        min_abs_score=cfg.min_rule_net_bps * 1e-4,
        side_flip_variants=cfg.side_flip_candidate_variants,
    )

    if raw_events.empty:
        return pd.DataFrame(columns=[
            "variant", "mean_log_growth", "cagr", "max_drawdown", "mar",
            "turnover", "final_equity", "pass_compound_gate"
        ])

    # 3. Label events and split dataset
    labeled = label_candidate_events(events=raw_events, aligned=aligned, cfg=cfg)
    diag = compute_rule_diagnostics(
        labeled_events=labeled,
        aligned=aligned,
        cfg=cfg,
        min_obs=max(cfg.min_candidate_obs, 100),
    )
    _logger.info(
        "[DIAG][RULE_RECOMMEND_ABLATION] keep=%s flip=%s",
        ",".join(diag.recommended_keep_variants) if diag.recommended_keep_variants else "",
        ",".join(diag.recommended_flip_variants) if diag.recommended_flip_variants else "",
    )
    n_bars = aligned.close_2d.shape[0]
    
    # Create fold splits (80% train, 20% validation)
    split_val = int(n_bars * 0.8)
    train_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=cfg, split_start=0, split_end=split_val
    )
    valid_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=cfg, split_start=split_val, split_end=n_bars
    )
    full_set = build_candidate_dataset(
        labeled_events=labeled, aligned=aligned, cfg=cfg, split_start=0, split_end=n_bars
    )

    # 4. Train ML Models
    gate_model = fit_candidate_gate(train=train_set, valid=valid_set, cfg=cfg)
    edge_models = fit_candidate_edge_models(train=train_set, valid=valid_set, cfg=cfg)

    # 5. Predict outcomes for full sample
    p_pass = predict_candidate_gate(model=gate_model, dataset=full_set)
    ml_out = predict_candidate_edges(models=edge_models, dataset=full_set, p_pass=p_pass, cfg=cfg)
    ml_out = replace(ml_out, events=full_set.event_index)

    rows: list[AblationRow] = []

    # Variant 1: rule_only_equal_size (Simple benchmark)
    # equal weight assigned to any rule trigger
    raw_w = _build_rule_equal_size_weights(
        raw_events=raw_events,
        close_2d=aligned.close_2d,
        symbols=symbols,
        max_symbol_weight=cfg.max_symbol_weight,
    )

    rows.append(_run_backtest_and_evaluate(raw_w, data_maps, symbols, tf, "rule_only_equal_size", cfg))

    # Variant 2: rule_only_fractional_kelly (Kelly Sizing but no ML)
    # create artificial mock edge output using constant score
    mock_events = raw_events.copy()
    mock_events["p_pass"] = 1.0
    mock_events["mu_net_decision_bps"] = 50.0  # Constant expectation
    mock_events["q10_net_bps"] = -10.0
    mock_events["utility_score"] = 1.0
    raw_kelly_w = build_candidate_target_weights(
        selected_events=mock_events,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(_run_backtest_and_evaluate(raw_kelly_w, data_maps, symbols, tf, "rule_only_fractional_kelly", cfg))

    # Variant 3: rule_plus_ml_gate (Gate filtering only)
    # ML gate filters events, but sizes them using constant ex-ante edge dummy values
    gate_events_only = select_candidate_events_for_portfolio(model_output=ml_out, cfg=cfg)
    # Override mu to ex-ante constant proxy for Variant 3
    gate_events_only_mock = gate_events_only.copy()
    gate_events_only_mock["mu_net_decision_bps"] = 50.0
    gate_only_w = build_candidate_target_weights(
        selected_events=gate_events_only_mock,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(_run_backtest_and_evaluate(
        gate_only_w, data_maps, symbols, tf, "rule_plus_ml_gate", cfg
    ))

    # Variant 4: rule_plus_ml_gate_plus_edge (Gate + Edge, but uncapped/uncapped Kelly)
    # Sized dynamically using predicted expected edge mu, but bypasses the cap projection loop (raw fractional Kelly)
    # Calculate raw Kelly weights manually to bypass project_all_caps
    raw_kelly_edge_w = _build_uncapped_kelly_edge_weights(
        selected_events=gate_events_only,
        close_2d=aligned.close_2d,
        symbols=symbols,
        kelly_fraction=cfg.kelly_fraction,
    )

    rows.append(_run_backtest_and_evaluate(
        raw_kelly_edge_w, data_maps, symbols, tf, "rule_plus_ml_gate_plus_edge", cfg
    ))

    # Variant 5: rule_plus_ml_gate_plus_edge_plus_portfolio_caps (Full sizing caps applied)
    # Full constraint projection on Kelly weights
    gate_plus_edge_plus_caps_w = build_candidate_target_weights(
        selected_events=gate_events_only,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(_run_backtest_and_evaluate(
        gate_plus_edge_plus_caps_w, data_maps, symbols, tf, "rule_plus_ml_gate_plus_edge_plus_portfolio_caps", cfg
    ))

    # Variant 6: candidate_ml_full (OOS-only signal — bridges production forward-fill path)
    # Filter gate-passed events to validation split (IS 구간 제외 → look-ahead-free OOS signal)
    n_bars_total = aligned.close_2d.shape[0]
    split_val_oos = int(n_bars_total * 0.8)
    gate_events_oos: pd.DataFrame
    if gate_events_only.empty:
        gate_events_oos = gate_events_only
    else:
        gate_events_oos = gate_events_only[
            gate_events_only["entry_idx"] >= split_val_oos
        ].copy()

    full_ml_w = build_candidate_target_weights(
        selected_events=gate_events_oos,
        close_2d=aligned.close_2d,
        symbols=symbols,
        beta_2d=None,
        sigma_3d=None,
        cfg=cfg,
    )
    rows.append(_run_backtest_and_evaluate(
        full_ml_w, data_maps, symbols, tf, "candidate_ml_full", cfg
    ))

    # Convert results to DataFrame
    df_results = pd.DataFrame([
        {
            "variant": r.variant,
            "mean_log_growth": r.mean_log_growth,
            "cagr": r.cagr,
            "max_drawdown": r.max_drawdown,
            "mar": r.mar,
            "turnover": r.turnover,
            "final_equity": r.final_equity,
            "pass_compound_gate": r.pass_compound_gate,
        }
        for r in rows
    ])

    return df_results


def _run_backtest_and_evaluate(
    target_weights: np.ndarray,
    data_maps: dict[str, dict[str, Any]],
    symbols: tuple[str, ...],
    tf: str,
    variant_name: str,
    cfg: CandidateStrategyConfig,
) -> AblationRow:
    """Helper to inject target_weights into data_maps and run backtest simulation."""
    # Shallow copy data map structure for safe isolated execution
    run_maps = {sym: dict(data_maps[sym]) for sym in symbols if sym in data_maps}
    for col_idx, sym in enumerate(symbols):
        if sym in run_maps and tf in run_maps[sym]:
            df = run_maps[sym][tf].copy()
            df["target_weight"] = target_weights[:, col_idx]
            run_maps[sym][tf] = df

    # Prepare AlignedMarketData from run_maps containing target_weight
    run_aligned = align_data_maps(run_maps, list(symbols), tf)

    from src.domain.futures.strategy.rule_signals import _atr_2d
    atr_2d = _atr_2d(run_aligned.high_2d, run_aligned.low_2d, run_aligned.close_2d, period=14)

    aligned_data = {
        "close": run_aligned.close_2d,
        "high": run_aligned.high_2d,
        "low": run_aligned.low_2d,
        "open": run_aligned.open_2d,
        "volume": run_aligned.volume_2d,
        "atr": atr_2d,
        "target_weights": target_weights,
    }

    # Execute backtest engine
    trades, equity_curve, _, _ = FuturesBacktestEngine.run_multi(
        aligned_data=aligned_data,
        symbol_names=list(symbols),
        strategy_params={},
    )

    # Evaluate compounding growth
    report = evaluate_compound_backtest(
        trades=trades,
        equity_curve=equity_curve,
        cfg=cfg,
    )

    return AblationRow(
        variant=variant_name,
        mean_log_growth=report.mean_log_growth,
        cagr=report.cagr,
        max_drawdown=report.max_drawdown,
        mar=report.mar,
        turnover=report.turnover,
        final_equity=report.final_equity,
        pass_compound_gate=report.pass_compound_gate,
    )
