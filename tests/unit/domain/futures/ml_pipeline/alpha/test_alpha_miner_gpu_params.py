from __future__ import annotations

import sys
from pathlib import Path
import pytest

project_root = str(Path(__file__).resolve().parents[6])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.alpha.miner import MLAlphaMiner, _GPU_AVAILABLE


def test_alpha_miner_has_get_lgbm_params_and_gpu_safety_flags():
    miner = MLAlphaMiner()

    assert hasattr(miner, "_get_lgbm_params"), "MLAlphaMiner must define _get_lgbm_params"

    getter = getattr(miner, "_get_lgbm_params")
    assert callable(getter)

    params = getter(seed_offset=0)
    assert isinstance(params, dict)
    assert params.get("allow_writing_files") is False

    if _GPU_AVAILABLE:
        # GPU 환경에서만 GPU 파라미터 검증
        assert params.get("task_type") == "GPU"
        assert params.get("devices") == "0"
    else:
        # CPU 환경에서는 CPU 파라미터 검증
        assert params.get("task_type") == "CPU"
