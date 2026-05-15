from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.pipeline_runner import _resolve_hmm_backend_name


class _DummyJaxModel:
    pass


class StudentTMultivariateHMM:
    pass


class _InferrerWithJax:
    _jax_model = _DummyJaxModel()


class _InferrerWithStudentT:
    _jax_model = StudentTMultivariateHMM()


def test_backend_name_uses_config_override() -> None:
    cfg = {"FUTURES_HMM_BACKEND": "student_t"}
    assert _resolve_hmm_backend_name(_InferrerWithJax(), cfg) == "student_t"


def test_backend_name_falls_back_to_inferrer_model_type() -> None:
    assert _resolve_hmm_backend_name(_InferrerWithStudentT(), None) == "student_t"
    assert _resolve_hmm_backend_name(_InferrerWithJax(), None) == "jax"

