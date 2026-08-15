from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.domain.futures.legacy.alpha_factory.factory as factory_module
from src.domain.futures.legacy.alpha_factory import AlphaFactoryV1
from src.domain.futures.legacy.alpha_factory.config import AlphaFactoryConfig
from src.domain.futures.legacy.alpha_factory.contracts import RegimeDecision, SleeveScores, SleeveWeights
from src.domain.futures.legacy.alpha_factory.features import extract_alpha_features


def _build_panel_df() -> pd.DataFrame:
    dt = pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT"]
    idx = pd.MultiIndex.from_product([dt, symbols], names=["datetime", "symbol"])
    n = len(idx)
    out = pd.DataFrame(index=idx)
    out["ret_6"] = np.array([0.05, -0.02, 0.04, -0.01, 0.03, -0.03], dtype=np.float64)
    out["ret_24"] = np.array([0.10, -0.03, 0.08, -0.02, 0.06, -0.04], dtype=np.float64)
    out["funding_z_72"] = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    out["taker_imbalance_z_24"] = np.linspace(-0.5, 0.5, n, dtype=np.float64)
    out["cvd_divergence_24h"] = np.linspace(-0.2, 0.2, n, dtype=np.float64)
    out["range_pos_24"] = np.array([0.7, 0.4, 0.6, 0.3, 0.55, 0.45], dtype=np.float64)
    out["hmm_prob_bull"] = 0.40
    out["hmm_prob_bear"] = 0.30
    out["hmm_prob_chop"] = 0.20
    out["hmm_prob_crisis"] = 0.10
    return out


def test_factory_v1_mine_alphas_cs_4h_guard() -> None:
    factory = AlphaFactoryV1(timeframe="1h")
    with pytest.raises(ValueError, match="supports only 4h timeframe"):
        factory.mine_alphas_cs(_build_panel_df())


def test_factory_v1_mine_alphas_cs_contract() -> None:
    factory = AlphaFactoryV1(timeframe="4h")
    panel_df = _build_panel_df()
    out = factory.mine_alphas_cs(panel_df)

    required_output_cols = {
        "alpha_long_00",
        "alpha_short_00",
        "alpha_long_signal",
        "alpha_short_signal",
        "alpha_long",
        "alpha_short",
        "alpha_net",
        "alpha_confidence",
    }
    assert required_output_cols.issubset(out.columns)
    assert out.index.equals(panel_df.index)

    assert out["alpha_long"].between(0.0, 1.0).all()
    assert out["alpha_short"].between(0.0, 1.0).all()
    assert out["alpha_confidence"].between(0.0, 1.0).all()
    assert out["alpha_net"].between(-1.0, 1.0).all()
    np.testing.assert_allclose(
        out["alpha_long"].to_numpy(dtype=np.float64, copy=False)
        + out["alpha_short"].to_numpy(dtype=np.float64, copy=False),
        np.ones(len(out), dtype=np.float64),
        atol=1e-12,
    )

    alpha_filter = out.attrs.get("alpha_component_filter", {})
    required_attr_keys = {
        "n_components",
        "n_surviving",
        "n_surviving_long",
        "n_surviving_short",
        "post_agg_selected_long_count",
        "post_agg_selected_short_count",
        "survived_long_cols",
        "survived_short_cols",
        "post_agg_selected_long_cols",
        "post_agg_selected_short_cols",
        "elite_zero_after_survival",
    }
    assert required_attr_keys.issubset(alpha_filter)
    assert float(alpha_filter["n_components"]) == 1.0
    assert 0.0 <= float(alpha_filter["n_surviving"]) <= float(alpha_filter["n_components"])
    assert 0.0 <= float(alpha_filter["n_surviving_long"]) <= float(alpha_filter["n_components"])
    assert 0.0 <= float(alpha_filter["n_surviving_short"]) <= float(alpha_filter["n_components"])
    assert alpha_filter["survived_long_cols"] == ["alpha_long_signal"]
    assert alpha_filter["survived_short_cols"] == ["alpha_short_signal"]
    assert alpha_filter["post_agg_selected_long_cols"] == ["alpha_long_signal"]
    assert alpha_filter["post_agg_selected_short_cols"] == ["alpha_short_signal"]


def test_step1_ridge_sleeves_improves_oos_csic() -> None:
    rng = np.random.default_rng(7)
    dt = pd.date_range("2025-01-01", periods=200, freq="4h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
    idx = pd.MultiIndex.from_product([dt, symbols], names=["datetime", "symbol"])
    n = len(idx)

    panel = pd.DataFrame(index=idx)
    panel["ret_24"] = rng.normal(0.0, 1.0, n)
    panel["ret_6"] = rng.normal(0.0, 1.0, n)
    panel["ret_12"] = rng.normal(0.0, 1.0, n)
    panel["funding_z_72"] = rng.normal(0.0, 1.0, n)
    panel["funding_rate"] = rng.normal(0.0, 1.0, n)
    panel["funding_mom_24"] = rng.normal(0.0, 1.0, n)
    panel["oi_momentum_24h"] = rng.normal(0.0, 1.0, n)
    panel["oi_price_divergence_24h"] = rng.normal(0.0, 1.0, n)
    panel["taker_imbalance_z_24"] = rng.normal(0.0, 1.0, n)
    panel["cvd_divergence_24h"] = rng.normal(0.0, 1.0, n)
    panel["vpin_proxy_12"] = rng.normal(0.0, 1.0, n)
    panel["tail_risk_24"] = np.clip(rng.normal(0.5, 0.3, n), 0.0, 1.5)
    panel["vol_surface_24_168"] = rng.normal(0.0, 1.0, n)
    panel["macro_vol_regime_shift"] = rng.normal(0.0, 1.0, n)
    panel["idiosyncratic_return_24h"] = rng.normal(0.0, 1.0, n)
    panel["btc_beta"] = rng.normal(0.0, 1.0, n)
    panel["range_pos_24"] = np.clip(rng.normal(0.5, 0.25, n), 0.0, 1.0)

    panel["hmm_prob_bull"] = 0.35
    panel["hmm_prob_bear"] = 0.30
    panel["hmm_prob_chop"] = 0.20
    panel["hmm_prob_crisis"] = 0.15

    cfg = AlphaFactoryConfig()
    dt_vals = panel.index.get_level_values("datetime")
    cs_panel = panel.copy()
    for col in cs_panel.select_dtypes(include=[np.number]).columns:
        if str(col).startswith("hmm_prob_") or str(col).startswith("regime_prob_"):
            continue
        ranked = pd.to_numeric(cs_panel[col], errors="coerce").groupby(dt_vals).rank(
            method="average", pct=True
        )
        cs_panel[col] = (ranked - 0.5) * 2.0

    feat_cols = [
        "ret_24",
        "ret_6",
        "ret_12",
        "funding_z_72",
        "funding_rate",
        "funding_mom_24",
        "oi_momentum_24h",
        "oi_price_divergence_24h",
        "taker_imbalance_z_24",
        "cvd_divergence_24h",
        "vpin_proxy_12",
        "tail_risk_24",
        "vol_surface_24_168",
        "macro_vol_regime_shift",
        "idiosyncratic_return_24h",
        "btc_beta",
        "range_pos_24",
    ]
    feature_rows = cs_panel.reindex(columns=feat_cols).to_dict(orient="records")
    extracted = [extract_alpha_features(row, cfg.norm) for row in feature_rows]

    # Deliberately set opposite signs vs heuristic sleeve formula to create a failure case.
    target = np.array(
        [
            -1.3 * f["ret_momentum"]
            - 0.9 * f["flow_pressure"]
            - 1.1 * f["carry_pressure"]
            - 0.8 * f["idio_edge"]
            + 0.25 * f["ret_reversal"]
            + 0.05 * rng.normal()
            for f in extracted
        ],
        dtype=np.float64,
    )
    panel["target"] = target

    split = dt[120].strftime("%Y-%m-%d")
    factory = AlphaFactoryV1(timeframe="4h")
    base = factory.mine_alphas_cs(
        panel,
        is_end_date=split,
        filter_options={"step1_use_ml_sleeves": False},
    )
    ml = factory.mine_alphas_cs(
        panel,
        is_end_date=split,
        filter_options={
            "step1_use_ml_sleeves": True,
            "step1_ridge_alpha": 1.0,
            "step1_min_is_rows": 128,
        },
    )

    base_diag = base.attrs["alpha_component_filter"]["root_cause_diag"]
    ml_diag = ml.attrs["alpha_component_filter"]["root_cause_diag"]
    base_oos = float(base_diag["adjusted_alpha_oos_csic_mean"])
    ml_oos = float(ml_diag["adjusted_alpha_oos_csic_mean"])
    assert ml_oos > base_oos + 0.03


def test_step2_ic_shrinkage_blend_improves_oos_vs_static() -> None:
    rng = np.random.default_rng(23)
    dt = pd.date_range("2025-02-01", periods=220, freq="4h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
    idx = pd.MultiIndex.from_product([dt, symbols], names=["datetime", "symbol"])
    n = len(idx)

    panel = pd.DataFrame(index=idx)
    panel["ret_24"] = rng.normal(0.0, 1.0, n)
    panel["ret_6"] = rng.normal(0.0, 1.0, n)
    panel["ret_12"] = rng.normal(0.0, 1.0, n)
    panel["funding_z_72"] = rng.normal(0.0, 1.0, n)
    panel["funding_rate"] = rng.normal(0.0, 1.0, n)
    panel["funding_mom_24"] = rng.normal(0.0, 1.0, n)
    panel["oi_momentum_24h"] = rng.normal(0.0, 1.0, n)
    panel["oi_price_divergence_24h"] = rng.normal(0.0, 1.0, n)
    panel["taker_imbalance_z_24"] = rng.normal(0.0, 1.0, n)
    panel["cvd_divergence_24h"] = rng.normal(0.0, 1.0, n)
    panel["vpin_proxy_12"] = rng.normal(0.0, 1.0, n)
    panel["tail_risk_24"] = np.clip(rng.normal(0.5, 0.3, n), 0.0, 1.5)
    panel["vol_surface_24_168"] = rng.normal(0.0, 1.0, n)
    panel["macro_vol_regime_shift"] = rng.normal(0.0, 1.0, n)
    panel["idiosyncratic_return_24h"] = rng.normal(0.0, 1.0, n)
    panel["btc_beta"] = rng.normal(0.0, 1.0, n)
    panel["range_pos_24"] = np.clip(rng.normal(0.5, 0.25, n), 0.0, 1.0)
    panel["hmm_prob_bull"] = 0.35
    panel["hmm_prob_bear"] = 0.30
    panel["hmm_prob_chop"] = 0.20
    panel["hmm_prob_crisis"] = 0.15

    cfg = AlphaFactoryConfig()
    dt_vals = panel.index.get_level_values("datetime")
    cs_panel = panel.copy()
    for col in cs_panel.select_dtypes(include=[np.number]).columns:
        if str(col).startswith("hmm_prob_") or str(col).startswith("regime_prob_"):
            continue
        ranked = pd.to_numeric(cs_panel[col], errors="coerce").groupby(dt_vals).rank(
            method="average", pct=True
        )
        cs_panel[col] = (ranked - 0.5) * 2.0
    feat_cols = [
        "ret_24",
        "ret_6",
        "ret_12",
        "funding_z_72",
        "funding_rate",
        "funding_mom_24",
        "oi_momentum_24h",
        "oi_price_divergence_24h",
        "taker_imbalance_z_24",
        "cvd_divergence_24h",
        "vpin_proxy_12",
        "tail_risk_24",
        "vol_surface_24_168",
        "macro_vol_regime_shift",
        "idiosyncratic_return_24h",
        "btc_beta",
        "range_pos_24",
    ]
    feature_rows = cs_panel.reindex(columns=feat_cols).to_dict(orient="records")
    extracted = [extract_alpha_features(row, cfg.norm) for row in feature_rows]
    panel["target"] = np.array(
        [
            1.2 * f["ret_momentum"]
            + 1.0 * f["flow_pressure"]
            + 0.5 * f["ret_reversal"]
            - 1.1 * f["carry_pressure"]
            + 0.6 * f["idio_edge"]
            + 0.05 * rng.normal()
            for f in extracted
        ],
        dtype=np.float64,
    )

    split = dt[130].strftime("%Y-%m-%d")
    factory = AlphaFactoryV1(timeframe="4h")

    step1_static = factory.mine_alphas_cs(
        panel,
        is_end_date=split,
        filter_options={
            "step1_use_ml_sleeves": True,
            "step1_ridge_alpha": 1.0,
            "step1_min_is_rows": 128,
            "step2_use_ic_shrinkage_blend": False,
        },
    )
    step2 = factory.mine_alphas_cs(
        panel,
        is_end_date=split,
        filter_options={
            "step1_use_ml_sleeves": True,
            "step1_ridge_alpha": 1.0,
            "step1_min_is_rows": 128,
            "step2_use_ic_shrinkage_blend": True,
            "step2_prior_strength": 3.0,
            "step2_min_folds": 3,
        },
    )

    base_diag = step1_static.attrs["alpha_component_filter"]["root_cause_diag"]
    step2_diag = step2.attrs["alpha_component_filter"]["root_cause_diag"]
    assert step2_diag["step2_enabled"] is True
    assert step2_diag["step2_blend_mode"] in {"static", "ic_shrinkage"}
    assert set(step2_diag["step2_weights"].keys()) == {
        "trend",
        "reversal",
        "carry",
        "flow",
        "idio",
    }
    assert set(step2_diag["step2_ic_stats"].keys()) == {
        "trend",
        "reversal",
        "carry",
        "flow",
        "idio",
    }

    base_oos = float(base_diag["adjusted_alpha_oos_csic_mean"])
    step2_oos = float(step2_diag["adjusted_alpha_oos_csic_mean"])
    assert step2_oos > base_oos + 1e-4


def test_step2_ic_shrinkage_blend_deterministic_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    dt = pd.date_range("2025-03-01", periods=40, freq="4h", tz="UTC")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    idx = pd.MultiIndex.from_product([dt, symbols], names=["datetime", "symbol"])
    panel = pd.DataFrame(index=idx)

    n_syms = len(symbols)
    rank_template = np.linspace(-1.0, 1.0, n_syms, dtype=np.float64)
    ret24 = np.zeros(len(panel), dtype=np.float64)
    funding = np.zeros(len(panel), dtype=np.float64)
    ret6 = np.zeros(len(panel), dtype=np.float64)
    quality = np.zeros(len(panel), dtype=np.float64)
    target = np.zeros(len(panel), dtype=np.float64)

    row = 0
    for t_idx, _ in enumerate(dt):
        shift = t_idx % n_syms
        trend_cs = np.roll(rank_template, shift)
        carry_cs = -trend_cs
        flow_cs = np.roll(rank_template, shift // 2)
        idio_cs = 0.2 * np.roll(rank_template, (shift + 1) % n_syms)
        reversal_cs = 0.1 * np.roll(rank_template, (shift + 2) % n_syms)
        for s_idx in range(n_syms):
            ret24[row] = trend_cs[s_idx]
            funding[row] = carry_cs[s_idx]
            ret6[row] = flow_cs[s_idx]
            quality[row] = idio_cs[s_idx]
            target[row] = trend_cs[s_idx] + 0.3 * flow_cs[s_idx] - 0.2 * carry_cs[s_idx]
            target[row] += 0.05 * reversal_cs[s_idx]
            row += 1

    panel["ret_24"] = ret24
    panel["ret_6"] = ret6
    panel["ret_12"] = ret6
    panel["funding_z_72"] = funding
    panel["funding_rate"] = funding
    panel["funding_mom_24"] = 0.0
    panel["oi_momentum_24h"] = 0.0
    panel["oi_price_divergence_24h"] = 0.0
    panel["taker_imbalance_z_24"] = ret6
    panel["cvd_divergence_24h"] = ret6
    panel["vpin_proxy_12"] = 0.0
    panel["tail_risk_24"] = 0.1
    panel["vol_surface_24_168"] = 0.0
    panel["macro_vol_regime_shift"] = 0.0
    panel["idiosyncratic_return_24h"] = quality
    panel["btc_beta"] = 0.0
    panel["range_pos_24"] = 0.5 + 0.25 * quality
    panel["hmm_prob_bull"] = 0.25
    panel["hmm_prob_bear"] = 0.25
    panel["hmm_prob_chop"] = 0.25
    panel["hmm_prob_crisis"] = 0.25
    panel["target"] = target

    def _fake_extract(source: pd.Series, _norm: object) -> dict[str, float]:
        return {
            "trend_sig": float(source.get("ret_24", 0.0)),
            "flow_sig": float(source.get("ret_6", 0.0)),
            "carry_sig": float(source.get("funding_z_72", 0.0)),
            "idio_sig": float(source.get("idiosyncratic_return_24h", 0.0)),
            "rev_sig": float(source.get("ret_12", 0.0)),
        }

    def _fake_sleeves(
        features: dict[str, float], _cfg: object, _models: object
    ) -> SleeveScores:
        return SleeveScores(
            trend=features["trend_sig"],
            reversal=0.15 * features["rev_sig"],
            carry=features["carry_sig"],
            flow=0.6 * features["flow_sig"],
            idio=0.3 * features["idio_sig"],
        )

    def _fake_route(
        _posterior: object, _sleeve_cfg: object, _regime_cfg: object
    ) -> RegimeDecision:
        return RegimeDecision(
            weights=SleeveWeights(1.0, 1.0, 1.0, 1.0, 1.0),
            gross_exposure=1.0,
            confidence=1.0,
        )

    def _identity_adjust(
        raw_alpha: float,
        confidence: float,
        gross_exposure: float,
        turnover: float,
        cfg: object,
    ) -> tuple[float, float, float]:
        del confidence, gross_exposure, turnover, cfg
        return raw_alpha, 0.0, 0.0

    monkeypatch.setattr(factory_module, "extract_alpha_features", _fake_extract)
    monkeypatch.setattr(factory_module, "compute_sleeve_scores_with_models", _fake_sleeves)
    monkeypatch.setattr(factory_module, "route_by_regime", _fake_route)
    monkeypatch.setattr(factory_module, "adjust_alpha_for_cost_and_confidence", _identity_adjust)

    split = dt[24].strftime("%Y-%m-%d")
    factory = AlphaFactoryV1(timeframe="4h")
    static_out = factory.mine_alphas_cs(
        panel,
        is_end_date=split,
        filter_options={"step2_use_ic_shrinkage_blend": False},
    )
    step2_out = factory.mine_alphas_cs(
        panel,
        is_end_date=split,
        filter_options={
            "step2_use_ic_shrinkage_blend": True,
            "step2_prior_strength": 3.0,
            "step2_min_folds": 3,
        },
    )

    static_diag = static_out.attrs["alpha_component_filter"]["root_cause_diag"]
    step2_diag = step2_out.attrs["alpha_component_filter"]["root_cause_diag"]
    static_oos = float(static_diag["adjusted_alpha_oos_csic_mean"])
    step2_oos = float(step2_diag["adjusted_alpha_oos_csic_mean"])

    assert step2_diag["step2_blend_mode"] == "ic_shrinkage"
    assert float(step2_diag["step2_weights"]["trend"]) > float(step2_diag["step2_weights"]["carry"])
    assert step2_oos > static_oos + 0.05
