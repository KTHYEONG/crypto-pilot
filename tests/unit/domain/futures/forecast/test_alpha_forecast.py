"""Tests for forecast/alpha.py — to_alpha_forecast and AlphaArtifactHash."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.futures.forecast.alpha import _hash_payload, to_alpha_forecast
from src.domain.futures.forecast.contracts import AlphaArtifactHash

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
_T = 10


def _make_panel(al_val: float = 0.005, as_val: float = 0.003) -> pd.DataFrame:
    datetimes = pd.date_range("2024-01-01", periods=_T, freq="4h")
    rows = []
    for dt in datetimes:
        for sym in _SYMBOLS:
            rows.append({"datetime": dt, "symbol": sym, "alpha_long": al_val, "alpha_short": as_val})
    df = pd.DataFrame(rows).set_index(["datetime", "symbol"]).sort_index()
    df.attrs["config_hash"] = "abc123"
    df.attrs["model_family"] = "lightgbm_dual_side_quantile"
    df.attrs["selected_horizon"] = 6
    df.attrs["strategy_name"] = "ml_lambdamart_v1"
    return df


class TestToAlphaForecastLossless:
    def test_alpha_long_values_preserved(self) -> None:
        panel = _make_panel(al_val=0.007)
        af = to_alpha_forecast(panel)
        # 모든 셀이 0.007 이어야 한다
        np.testing.assert_allclose(af.alpha_long_2d, 0.007, rtol=1e-5)

    def test_alpha_short_values_preserved(self) -> None:
        panel = _make_panel(as_val=0.004)
        af = to_alpha_forecast(panel)
        np.testing.assert_allclose(af.alpha_short_2d, 0.004, rtol=1e-5)

    def test_output_shapes(self) -> None:
        panel = _make_panel()
        af = to_alpha_forecast(panel)
        assert af.alpha_long_2d.shape == (_T, len(_SYMBOLS))
        assert af.alpha_short_2d.shape == (_T, len(_SYMBOLS))
        assert af.eligible_mask.shape == (_T, len(_SYMBOLS))

    def test_symbols_count_matches(self) -> None:
        panel = _make_panel()
        af = to_alpha_forecast(panel)
        assert len(af.symbols) == len(_SYMBOLS)

    def test_eligible_mask_true_for_finite(self) -> None:
        panel = _make_panel(al_val=0.005, as_val=0.003)
        af = to_alpha_forecast(panel)
        assert np.all(af.eligible_mask)

    def test_negative_alpha_clamped_to_zero(self) -> None:
        panel = _make_panel(al_val=-0.01, as_val=-0.005)
        af = to_alpha_forecast(panel)
        assert np.all(af.alpha_long_2d >= 0.0)
        assert np.all(af.alpha_short_2d >= 0.0)

    def test_source_from_strategy_name_attr(self) -> None:
        panel = _make_panel()
        af = to_alpha_forecast(panel)
        assert af.source == "ml_lambdamart_v1"

    def test_quantile_arrays_from_attrs(self) -> None:
        panel = _make_panel()
        n_cells = _T * len(_SYMBOLS)
        panel.attrs["forecast_metadata_v3"] = {
            "q10_long": np.full(n_cells, 0.001),
            "q50_long": np.full(n_cells, 0.005),
            "q90_long": np.full(n_cells, 0.009),
            "q10_short": np.full(n_cells, 0.001),
            "q50_short": np.full(n_cells, 0.003),
            "q90_short": np.full(n_cells, 0.007),
            "confidence_long": np.full(n_cells, 0.8),
            "confidence_short": np.full(n_cells, 0.7),
        }
        af = to_alpha_forecast(panel)
        assert af.q10_long_2d is not None
        assert af.q90_long_2d is not None
        np.testing.assert_allclose(af.q10_long_2d, 0.001, rtol=1e-5)
        np.testing.assert_allclose(af.confidence_long_2d, 0.8, rtol=1e-5)


class TestAlphaArtifactHashDeterminism:
    def test_combined_hash_deterministic(self) -> None:
        h = AlphaArtifactHash(
            alpha_config_hash="abc", feature_config_hash="def",
            label_config_hash="ghi", train_window_hash="jkl",
            fold_spec_hash="mno", model_family="lgbm", selected_horizon=6,
        )
        assert h.combined() == h.combined()

    def test_different_inputs_different_hash(self) -> None:
        h1 = AlphaArtifactHash("abc", "def", "ghi", "jkl", "mno", "lgbm", 6)
        h2 = AlphaArtifactHash("abc", "def", "ghi", "jkl", "mno", "lgbm", 12)
        assert h1.combined() != h2.combined()

    def test_same_inputs_same_hash(self) -> None:
        h1 = AlphaArtifactHash("a", "b", "c", "d", "e", "lgbm", 6)
        h2 = AlphaArtifactHash("a", "b", "c", "d", "e", "lgbm", 6)
        assert h1.combined() == h2.combined()

    def test_hash_length_16_chars(self) -> None:
        h = AlphaArtifactHash("a", "b", "c", "d", "e", "f", 1)
        assert len(h.combined()) == 16

    def test_artifact_hash_from_panel_attrs(self) -> None:
        panel = _make_panel()
        af = to_alpha_forecast(panel)
        assert af.artifact_hash.model_family == "lightgbm_dual_side_quantile"
        assert af.artifact_hash.selected_horizon == 6
        assert af.artifact_hash.alpha_config_hash == "abc123"

    def test_hash_payload_stable(self) -> None:
        payload = {"a": 1, "b": [2, 3], "c": "str"}
        h1 = _hash_payload(payload)
        h2 = _hash_payload(payload)
        assert h1 == h2
        assert len(h1) == 16
