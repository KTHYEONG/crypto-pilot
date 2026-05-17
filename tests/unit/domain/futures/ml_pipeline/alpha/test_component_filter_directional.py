from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

project_root = str(Path(__file__).resolve().parents[6])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.domain.futures.ml_pipeline.alpha.component_filter as component_filter
from src.domain.futures.ml_pipeline.alpha.component_filter import filter_alpha_components


def _build_panel_and_alpha(
    *,
    n_times: int = 50,
    is_times: int = 30,
    long_strength: float = 1.0,
    short_strength: float = 1.0,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2026-01-01", periods=n_times, freq="h", tz="UTC")
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    idx = pd.MultiIndex.from_product([dt, syms], names=["datetime", "symbol"])
    panel_df = pd.DataFrame(index=idx)

    base_rank = pd.Series({"BTCUSDT": 0.9, "ETHUSDT": 0.7, "SOLUSDT": 0.3, "XRPUSDT": 0.1})
    time_noise = rng.normal(0.0, 0.03, size=(n_times, len(syms)))
    target_mat = np.zeros((n_times, len(syms)), dtype=np.float64)
    for t in range(n_times):
        target_mat[t, :] = np.clip(base_rank.to_numpy() + time_noise[t, :], 0.0, 1.0)
    panel_df["target"] = target_mat.reshape(-1)
    panel_df["regime_pre_hmm"] = 0

    long_pred = target_mat + rng.normal(0.0, 0.04 / max(long_strength, 1e-6), size=target_mat.shape)
    short_target = 1.0 - target_mat
    short_pred = short_target + rng.normal(
        0.0, 0.04 / max(short_strength, 1e-6), size=short_target.shape
    )

    alpha_wide = pd.DataFrame(index=idx)
    alpha_wide["alpha_long_00"] = np.clip(long_pred.reshape(-1), 0.0, 1.0)
    alpha_wide["alpha_short_00"] = np.clip(short_pred.reshape(-1), 0.0, 1.0)

    is_end = dt[is_times].isoformat()
    return panel_df, alpha_wide, is_end


def _run_filter(panel_df: pd.DataFrame, alpha_wide: pd.DataFrame, is_end: str) -> dict:
    _, meta = filter_alpha_components(
        alpha_wide=alpha_wide,
        panel_df=panel_df,
        is_end_date=is_end,
        n_trials=2,
        alpha_cols=["alpha_long_00", "alpha_short_00"],
        fdr_q=0.20,
        require_regime_gate=False,
    )
    return meta


def _patch_non_directional_gates(monkeypatch) -> None:
    monkeypatch.setattr(
        component_filter,
        "_benjamini_hochberg_reject",
        lambda p_values, q: np.ones(len(p_values), dtype=bool),
    )
    monkeypatch.setattr(component_filter, "_deflated_sharpe_threshold", lambda *args, **kwargs: True)
    monkeypatch.setattr(component_filter, "_ic_half_life_bars", lambda *_args, **_kwargs: 10.0)


def test_component_filter_long_only_survives_directional_split(monkeypatch) -> None:
    _patch_non_directional_gates(monkeypatch)
    panel_df, alpha_wide, is_end = _build_panel_and_alpha(long_strength=1.0, short_strength=1.0)
    alpha_wide["alpha_short_00"] = np.nan
    meta = _run_filter(panel_df, alpha_wide, is_end)
    assert "alpha_long_00" in meta.get("survived_long_cols", [])
    assert "alpha_short_00" not in meta.get("survived_short_cols", [])


def test_component_filter_short_only_survives_directional_split(monkeypatch) -> None:
    _patch_non_directional_gates(monkeypatch)
    panel_df, alpha_wide, is_end = _build_panel_and_alpha(long_strength=1.0, short_strength=1.0)
    alpha_wide["alpha_long_00"] = np.nan
    meta = _run_filter(panel_df, alpha_wide, is_end)
    assert "alpha_short_00" in meta.get("survived_short_cols", [])
    assert "alpha_long_00" not in meta.get("survived_long_cols", [])


def test_component_filter_mixed_survival_and_backward_alias(monkeypatch) -> None:
    _patch_non_directional_gates(monkeypatch)
    panel_df, alpha_wide, is_end = _build_panel_and_alpha(long_strength=1.0, short_strength=1.0)
    meta = _run_filter(panel_df, alpha_wide, is_end)
    survived_long = set(meta.get("survived_long_cols", []))
    survived_short = set(meta.get("survived_short_cols", []))
    survived_all = set(meta.get("survived_cols", []))
    assert "alpha_long_00" in survived_long
    assert "alpha_short_00" in survived_short
    assert survived_long | survived_short <= survived_all


def test_component_filter_records_gate_reasons_and_half_life_diag(monkeypatch) -> None:
    panel_df, alpha_wide, is_end = _build_panel_and_alpha(
        n_times=72,
        is_times=60,
        long_strength=1.0,
        short_strength=1.0,
    )

    monkeypatch.setattr(
        component_filter,
        "_benjamini_hochberg_reject",
        lambda p_values, q: np.ones(len(p_values), dtype=bool),
    )
    monkeypatch.setattr(component_filter, "_deflated_sharpe_threshold", lambda *args, **kwargs: True)
    monkeypatch.setattr(component_filter, "_ic_half_life_bars_with_diag", lambda *_args, **_kwargs: (0.0, "zero_variance"))

    _, meta = filter_alpha_components(
        alpha_wide=alpha_wide,
        panel_df=panel_df,
        is_end_date=is_end,
        n_trials=2,
        alpha_cols=["alpha_long_00", "alpha_short_00"],
        fdr_q=0.20,
        require_regime_gate=False,
    )

    fail_reasons = meta.get("gate_fail_reasons_by_col", {})
    gate_status = meta.get("gate_status_by_col", {})
    hl_diag = meta.get("half_life_diag_code_by_col", {})
    assert "alpha_long_00" in fail_reasons
    assert "half_life_fail" in fail_reasons["alpha_long_00"]
    assert hl_diag.get("alpha_long_00") == "zero_variance"
    assert gate_status.get("alpha_long_00", {}).get("half_life_diag_code") == "zero_variance"
