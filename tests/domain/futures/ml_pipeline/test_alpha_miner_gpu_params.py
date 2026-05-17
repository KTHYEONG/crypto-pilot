import pytest

from src.domain.futures.ml_pipeline.alpha.miner import MLAlphaMiner


def test_alpha_miner_has_get_lgbm_params_and_gpu_safety_flags():
    miner = MLAlphaMiner()

    assert hasattr(miner, "_get_lgbm_params"), "MLAlphaMiner must define _get_lgbm_params"

    getter = getattr(miner, "_get_lgbm_params")
    assert callable(getter)

    params = getter(seed_offset=0)
    assert isinstance(params, dict)
    assert params.get("task_type") == "GPU"
    assert params.get("devices") == "0"
    assert params.get("allow_writing_files") is False
