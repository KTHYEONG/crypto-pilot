import pandas as pd

from src.domain.futures.ml_pipeline import pipeline_runner


class _FakeCollector:
    def collect_and_save(self, *args, **kwargs):
        return None


class _FakeUtils:
    def build_panel_df(self, data_maps, tf="1h"):
        frames = []
        for sym, mp in data_maps.items():
            df = mp[tf].copy()
            df = df.set_index("datetime")
            df["symbol"] = sym
            frames.append(df.reset_index().set_index(["datetime", "symbol"]))
        out = pd.concat(frames).sort_index()
        return out

    def add_cross_sectional_features(self, panel_df):
        return panel_df

    def add_systemic_features(self, panel_df):
        out = panel_df.copy()
        out["macro_trend_24h"] = 0.0
        out["macro_vol_24h"] = 0.0
        out["cs_dispersion"] = 0.0
        return out

    def create_multi_horizon_rank_targets(self, panel_df, horizons, weights):
        return pd.Series(0.0, index=panel_df.index)


class _FakeMiner:
    captured_panel_df = None

    def __init__(self, *args, **kwargs):
        pass

    def mine_alphas_cs(self, panel_df, **kwargs):
        _FakeMiner.captured_panel_df = panel_df.copy()
        idx = panel_df.index
        out = pd.DataFrame(index=idx)
        out["alpha_long_00"] = 0.5
        out["alpha_short_00"] = 0.5
        out["alpha_long"] = 0.5
        out["alpha_short"] = 0.5
        return out


class _DummyInferrer:
    pass


def test_alpha_only_skips_fusion_and_injects_hmm_columns(monkeypatch):
    sym = "BTCUSDT"
    dt = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    df_1h = pd.DataFrame(
        {
            "datetime": dt,
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
            "ret_1": [0.0, 0.0, 0.0, 0.0],
        }
    )

    market_probs = pd.DataFrame(
        {
            "datetime": dt,
            "hmm_prob_bull_calm": [0.2, 0.2, 0.2, 0.2],
            "hmm_prob_bull_vol_up": [0.2, 0.2, 0.2, 0.2],
            "hmm_prob_bear_trend": [0.2, 0.2, 0.2, 0.2],
            "hmm_prob_chop": [0.2, 0.2, 0.2, 0.2],
            "hmm_prob_crisis": [0.2, 0.2, 0.2, 0.2],
            "hmm_tail_risk_8bar": [0.1, 0.1, 0.1, 0.1],
        }
    )

    market_hmm_feats = pd.DataFrame(index=dt)
    market_hmm_feats["macro_trend_168h"] = 0.0

    monkeypatch.setattr(pipeline_runner, "DataCollector", _FakeCollector)
    monkeypatch.setattr(pipeline_runner, "CrossSectionalPipelineUtils", _FakeUtils)
    monkeypatch.setattr(pipeline_runner, "MLAlphaMiner", _FakeMiner)
    monkeypatch.setattr(
        pipeline_runner,
        "build_hmm_inferrer_from_config",
        lambda *args, **kwargs: _DummyInferrer(),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_run_systemic_hmm_with_causal_split",
        lambda *args, **kwargs: market_probs.copy(),
    )
    monkeypatch.setattr(
        pipeline_runner,
        "_attach_tail_overlay_if_enabled",
        lambda **kwargs: (kwargs["market_probs"], {}),
    )

    def _fake_modulator(*args, **kwargs):
        return {
            sym: pd.DataFrame(
                {
                    "hmm_modulator_long": [1.0, 1.0, 1.0, 1.0],
                    "hmm_modulator_short": [1.0, 1.0, 1.0, 1.0],
                }
            )
        }

    monkeypatch.setattr(pipeline_runner, "_hmm_modulator_kelly_per_symbol", _fake_modulator)
    monkeypatch.setattr(pipeline_runner, "_print_hmm_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline_runner, "_compute_per_symbol_metrics", lambda *args, **kwargs: {})

    def _forbid_step4_fusion(*args, **kwargs):
        raise AssertionError("Fusion path should not execute when gp_only=True")

    monkeypatch.setattr(pipeline_runner, "_step4_fusion_one_symbol", _forbid_step4_fusion)

    # local import inside function must resolve to patched callable
    import src.domain.futures.ml_pipeline.features.engineering as engineering

    monkeypatch.setattr(engineering, "build_systemic_hmm_features", lambda *args, **kwargs: market_hmm_feats)

    cfg = {
        "FUTURES_HMM_K_STATES": 5,
        "FUTURES_HMM_N_ITER": 1,
        "FUTURES_ML_ALPHA_HORIZONS": (3, 6),
        "FUTURES_ML_ALPHA_SLOTS_PER_THEME": 5,
    }

    out = pipeline_runner._run_ml_pipeline_implementation(
        symbols=[sym],
        tf="1h",
        fetch_start_date="2026-01-01",
        end="2026-01-02",
        cfg=cfg,
        gp_only=True,
        hmm_only=False,
        preloaded_data_maps={sym: {"1h": df_1h.copy()}},
        preloaded_1h_maps={sym: df_1h.copy()},
    )

    injected = _FakeMiner.captured_panel_df
    assert injected is not None
    hmm_cols = [c for c in injected.columns if str(c).startswith("hmm_")]
    assert "hmm_prob_crisis" in hmm_cols
    assert "hmm_tail_risk_8bar" in hmm_cols

    assert isinstance(out.hmm_report, dict)
