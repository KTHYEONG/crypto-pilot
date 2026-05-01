"""ML alpha pipeline (Phase C) for futures: GP, HMM, TBM, meta-labeler."""

from __future__ import annotations

from src.domain.futures.ml_pipeline.feature_engineering import (
    build_gp_input_features,
    build_hmm_input_features,
)
from src.domain.futures.ml_pipeline.ml_alpha_miner import MLAlphaMiner
from src.domain.futures.ml_pipeline.hmm_state_inferrer import HMMStateInferrer
from src.domain.futures.ml_pipeline.ml_pipeline_runner import (
    MLPipelineOutput,
    copy_data_maps_tf_clone,
    merge_ml_output_into_data_maps,
    merge_ml_output_into_is_and_oos,
    run_hmm_fusion_for_is_end,
    run_ml_pipeline_for_universe,
)

__all__ = [
    "MLAlphaMiner",
    "HMMStateInferrer",
    "MLPipelineOutput",
    "build_gp_input_features",
    "build_hmm_input_features",
    "copy_data_maps_tf_clone",
    "merge_ml_output_into_data_maps",
    "merge_ml_output_into_is_and_oos",
    "run_hmm_fusion_for_is_end",
    "run_ml_pipeline_for_universe",
]
