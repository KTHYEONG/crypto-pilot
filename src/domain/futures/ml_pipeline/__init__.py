"""ML alpha pipeline (Phase C) for futures: GP, HMM, TBM, meta-labeler."""

from __future__ import annotations

from src.domain.futures.ml_pipeline.feature_engineering import (
    build_gp_input_features,
    build_hmm_input_features,
)
from src.domain.futures.ml_pipeline.gp_alpha_miner import GPAlphaMiner
from src.domain.futures.ml_pipeline.hmm_state_inferrer import HMMStateInferrer
from src.domain.futures.ml_pipeline.meta_labeler import MetaLabeler
from src.domain.futures.ml_pipeline.ml_pipeline_runner import MLPipelineOutput, run_ml_pipeline

__all__ = [
    "GPAlphaMiner",
    "HMMStateInferrer",
    "MLPipelineOutput",
    "MetaLabeler",
    "build_gp_input_features",
    "build_hmm_input_features",
    "run_ml_pipeline",
]
