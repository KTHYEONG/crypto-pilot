from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.domain.futures.strategy.cache import (
    load_lightgbm_booster,
    make_cache_key_payload,
    read_feature_or_label_npz,
    read_manifest,
    save_lightgbm_model,
    strategy_ml_cache_paths,
    write_artifact_manifest,
    write_feature_or_label_npz,
)


def test_cache_paths_and_manifest_deterministic(tmp_path: Path) -> None:
    paths = strategy_ml_cache_paths(tmp_path / "data" / "cache_futures", run_id="run-1")
    assert paths["features"].as_posix().endswith("strategy_ml/features")
    assert paths["labels"].as_posix().endswith("strategy_ml/labels")
    assert paths["manifest"].as_posix().endswith("strategy_ml/manifest")

    payload = make_cache_key_payload(
        symbols=("BTCUSDT", "ETHUSDT"),
        timeframe="4h",
        start="2024-01-01",
        end="2024-03-01",
        feature_config={"max_features": 64},
        source_column_availability={"funding_rate": True, "basis": False},
        label_horizon=1,
        cost_config={"fee_bps": 4.0, "slippage_bps": 1.0},
        execution_alignment_config={"entry": "open_t1", "exit": "close_t1"},
    )

    m1 = write_artifact_manifest(paths["manifest"], key_payload=payload, features_file="f.npz")
    m2 = write_artifact_manifest(paths["manifest"], key_payload=payload, features_file="f.npz")
    assert m1 == m2

    manifest = read_manifest(m1)
    assert "manifest_hash" in manifest
    assert manifest["key_payload"]["timeframe"] == "4h"


def test_feature_or_label_npz_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "feature.npz"
    arr_a = np.arange(12, dtype=np.float32).reshape(3, 4)
    arr_b = np.array([1, 0, 1], dtype=bool)
    write_feature_or_label_npz(path, values=arr_a, mask=arr_b)

    loaded = read_feature_or_label_npz(path)
    assert np.array_equal(loaded["values"], arr_a)
    assert np.array_equal(loaded["mask"], arr_b)


def test_lightgbm_artifact_save_and_load_roundtrip(tmp_path: Path) -> None:
    x = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.4], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]],
        dtype=np.float32,
    )
    y = np.array([0.0, 1.0, 0.4, 0.2, 0.8, 0.3], dtype=np.float32)
    model = lgb.LGBMRegressor(n_estimators=20, learning_rate=0.1, num_leaves=7, min_data_in_leaf=1)
    model.fit(x, y)

    artifact_path = tmp_path / "models" / "q50.txt"
    save_lightgbm_model(artifact_path, model)
    booster = load_lightgbm_booster(artifact_path)

    pred_model = model.predict(x)
    pred_booster = booster.predict(x)
    assert np.allclose(pred_model, pred_booster)
