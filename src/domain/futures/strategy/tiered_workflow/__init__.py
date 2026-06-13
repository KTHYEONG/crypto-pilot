# src/domain/futures/strategy/tiered_workflow/__init__.py

import logging

from src.domain.futures.portfolio.signal_composer import compose_symbol_signals
from src.domain.futures.strategy.candidate_workflow import (
    _fit_and_predict_single_fold,
)
from src.domain.futures.strategy.tiered_logging import (
    format_layer1_deployment_registry_table,
    format_layer1_gate_table,
    format_layer1_outer_fold_table,
    format_layer1_table,
    format_layer2_table,
    format_layer3_table,
    format_system_status,
)
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _run_awf_simulation,
    _stack_oos_signals,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    FoldDiagnostic,
    Layer1Result,
    Layer2Result,
    Layer3Result,
    PredictionDecompositionDiag,
    StrategySignal,
    SymbolRealizedStat,
)
from src.domain.futures.strategy.tiered_workflow.diagnostics import (
    _compute_fold_ts_ic,
    _fold_eligible_symbol_mask,
    compute_per_strategy_oos_validation,
    compute_per_symbol_ic,
    compute_per_symbol_realized_stats,
    compute_prediction_decomposition_diag,
)
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _cagr,
    _mdd,
    _newey_west_ic_tstat,
    _nw_tstat_realized,
    _sharpe,
    compute_breadth_weighted_ic,
    compute_panel_diversity,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    _TRAINED_FOLD_COVERAGE_THRESHOLD,
    _VALID_COVERAGE_FLAG_THRESHOLD,
    run_l1_nested_swf,
    run_l1_swf,
    run_l2_awf,
    run_l3_holdout,
    run_tiered_pipeline,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _candidate_output_to_signal_batch,
    _event_results_from_fold_output,
    _registry_to_symbol_signals,
    build_qualified_signal_registry,
    compute_symbol_strategy_evidence,
    evaluate_layer1_readiness,
    evaluate_outer_signal_opportunities,
    fit_layer1_inference_artifact,
    select_outer_symbol_opportunities,
)
from src.domain.futures.strategy.walk_forward import (
    build_l1_nested_swf_folds,
    build_l1_swf_folds,
    build_walk_forward_folds,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_TRAINED_FOLD_COVERAGE_THRESHOLD",
    # Public & Private Pipeline / Constants
    "_VALID_COVERAGE_FLAG_THRESHOLD",
    "FoldDiagnostic",
    # Public & Private Dataclasses
    "Layer1Result",
    "Layer2Result",
    "Layer3Result",
    "PredictionDecompositionDiag",
    "StrategySignal",
    "SymbolRealizedStat",
    # Public & Private Metrics
    "_cagr",
    # Public & Private Signal Selection
    "_candidate_output_to_signal_batch",
    # Public & Private Diagnostics
    "_compute_fold_ts_ic",
    "_event_results_from_fold_output",
    # Mocking target functions in tests
    "_fit_and_predict_single_fold",
    "_fold_eligible_symbol_mask",
    "_mdd",
    "_newey_west_ic_tstat",
    "_nw_tstat_realized",
    "_registry_to_symbol_signals",
    # Public & Private AWF Sim
    "_run_awf_simulation",
    "_sharpe",
    "_stack_oos_signals",
    "build_l1_nested_swf_folds",
    "build_l1_swf_folds",
    "build_qualified_signal_registry",
    "build_walk_forward_folds",
    "compose_symbol_signals",
    "compute_breadth_weighted_ic",
    "compute_panel_diversity",
    "compute_per_strategy_oos_validation",
    "compute_per_symbol_ic",
    "compute_per_symbol_realized_stats",
    "compute_prediction_decomposition_diag",
    "compute_symbol_strategy_evidence",
    "evaluate_layer1_readiness",
    "evaluate_outer_signal_opportunities",
    "fit_layer1_inference_artifact",
    "format_layer1_deployment_registry_table",
    "format_layer1_gate_table",
    "format_layer1_outer_fold_table",
    "format_layer1_table",
    "format_layer2_table",
    "format_layer3_table",
    "format_system_status",
    "logger",
    "run_l1_nested_swf",
    "run_l1_swf",
    "run_l2_awf",
    "run_l3_holdout",
    "run_tiered_pipeline",
    "select_outer_symbol_opportunities",
]
