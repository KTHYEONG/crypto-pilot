from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_DATACLASSES = "src.domain.futures.strategy.tiered_workflow.dataclasses"
_L2_META = "src.domain.futures.strategy.tiered_workflow.l2_meta"
_PIPELINE = "src.domain.futures.strategy.tiered_workflow.pipeline"
_METRICS = "src.domain.futures.strategy.tiered_workflow.metrics"
_SIGNAL_SELECTION = "src.domain.futures.strategy.tiered_workflow.signal_selection"
_DIAGNOSTICS = "src.domain.futures.strategy.tiered_workflow.diagnostics"
_AWF_SIM = "src.domain.futures.strategy.tiered_workflow.awf_sim"
_SELECTION = "src.domain.futures.strategy.tiered_workflow.selection"
_L2_GATE = "src.domain.futures.strategy.tiered_workflow.l2_gate"
_L1_COMPOSER = "src.domain.futures.portfolio.signal_composer"
_WALK_FORWARD = "src.domain.futures.strategy.walk_forward"
_TIERED_LOGGING = "src.domain.futures.strategy.tiered_logging"
_CANDIDATE_WORKFLOW = "src.domain.futures.strategy.candidate_workflow"
_RISK_DEPLOYMENT = "src.domain.futures.strategy.tiered_workflow.risk_deployment"

_MODULE_ATTRS: dict[str, tuple[str, str]] = {
    "FoldDiagnostic": (_DATACLASSES, "FoldDiagnostic"),
    "Layer1Result": (_DATACLASSES, "Layer1Result"),
    "Layer2AllocationConfig": (_DATACLASSES, "Layer2AllocationConfig"),
    "Layer2BlockMetric": (_DATACLASSES, "Layer2BlockMetric"),
    "Layer2FoldDiagnostics": (_DATACLASSES, "Layer2FoldDiagnostics"),
    "Layer2GateEvaluation": (_DATACLASSES, "Layer2GateEvaluation"),
    "Layer2Result": (_DATACLASSES, "Layer2Result"),
    "Layer2SignalSchedule": (_DATACLASSES, "Layer2SignalSchedule"),
    "Layer2SimulationDiagnostics": (_DATACLASSES, "Layer2SimulationDiagnostics"),
    "Layer2StudyResult": (_DATACLASSES, "Layer2StudyResult"),
    "Layer2TrialEvaluation": (_DATACLASSES, "Layer2TrialEvaluation"),
    "Layer3ExecutionError": (_PIPELINE, "Layer3ExecutionError"),
    "Layer3Result": (_DATACLASSES, "Layer3Result"),
    "Layer3WindowError": (_PIPELINE, "Layer3WindowError"),
    "LayerUniverseAudit": (_DATACLASSES, "LayerUniverseAudit"),
    "MetaFeasibilityReport": (_L2_META, "MetaFeasibilityReport"),
    "PredictionDecompositionDiag": (_DATACLASSES, "PredictionDecompositionDiag"),
    "SleeveMetaSamples": (_L2_META, "SleeveMetaSamples"),
    "StrategySignal": (_DATACLASSES, "StrategySignal"),
    "SymbolLifecycleRecord": (_DATACLASSES, "SymbolLifecycleRecord"),
    "SymbolRealizedStat": (_DATACLASSES, "SymbolRealizedStat"),
    "TieredPipelineError": (_PIPELINE, "TieredPipelineError"),
    "_TRAINED_FOLD_COVERAGE_THRESHOLD": (_PIPELINE, "_TRAINED_FOLD_COVERAGE_THRESHOLD"),
    "_VALID_COVERAGE_FLAG_THRESHOLD": (_PIPELINE, "_VALID_COVERAGE_FLAG_THRESHOLD"),
    "_cagr": (_METRICS, "_cagr"),
    "_candidate_output_to_signal_batch": (_SIGNAL_SELECTION, "_candidate_output_to_signal_batch"),
    "_compute_fold_ts_ic": (_DIAGNOSTICS, "_compute_fold_ts_ic"),
    "_event_results_from_fold_output": (_SIGNAL_SELECTION, "_event_results_from_fold_output"),
    "_fit_and_predict_single_fold": (_CANDIDATE_WORKFLOW, "_fit_and_predict_single_fold"),
    "_fold_eligible_symbol_mask": (_DIAGNOSTICS, "_fold_eligible_symbol_mask"),
    "_layer2_experiment_key": (_SELECTION, "_layer2_experiment_key"),
    "_mdd": (_METRICS, "_mdd"),
    "_newey_west_ic_tstat": (_METRICS, "_newey_west_ic_tstat"),
    "_nw_tstat_realized": (_METRICS, "_nw_tstat_realized"),
    "_registry_to_symbol_signals": (_SIGNAL_SELECTION, "_registry_to_symbol_signals"),
    "_run_awf_simulation": (_AWF_SIM, "_run_awf_simulation"),
    "_sharpe": (_METRICS, "_sharpe"),
    "_stack_oos_signals": (_AWF_SIM, "_stack_oos_signals"),
    "build_l1_nested_swf_folds": (_WALK_FORWARD, "build_l1_nested_swf_folds"),
    "build_l1_swf_folds": (_WALK_FORWARD, "build_l1_swf_folds"),
    "build_l2_simulation_folds": (_WALK_FORWARD, "build_l2_simulation_folds"),
    "build_l2_simulation_cache": (_AWF_SIM, "build_l2_simulation_cache"),
    "build_l2_signal_schedule": (_AWF_SIM, "build_layer2_signal_schedule"),
    "build_layer2_signal_schedule": (_AWF_SIM, "build_layer2_signal_schedule"),
    "build_layer_universe_audit": (_DIAGNOSTICS, "build_layer_universe_audit"),
    "build_qualified_signal_registry": (_SIGNAL_SELECTION, "build_qualified_signal_registry"),
    "build_sleeve_meta_dataset": (_L2_META, "build_sleeve_meta_dataset"),
    "build_walk_forward_folds": (_WALK_FORWARD, "build_walk_forward_folds"),
    "compose_symbol_signals": (_L1_COMPOSER, "compose_symbol_signals"),
    "compute_breadth_weighted_ic": (_METRICS, "compute_breadth_weighted_ic"),
    "compute_futures_bar_return": (_AWF_SIM, "compute_futures_bar_return"),
    "compute_layer2_fold_diagnostics": (_RISK_DEPLOYMENT, "compute_layer2_fold_diagnostics"),
    "compute_panel_diversity": (_METRICS, "compute_panel_diversity"),
    "compute_per_strategy_oos_validation": (_DIAGNOSTICS, "compute_per_strategy_oos_validation"),
    "compute_per_symbol_ic": (_DIAGNOSTICS, "compute_per_symbol_ic"),
    "compute_per_symbol_realized_stats": (_DIAGNOSTICS, "compute_per_symbol_realized_stats"),
    "compute_prediction_decomposition_diag": (_DIAGNOSTICS, "compute_prediction_decomposition_diag"),
    "compute_rebalance_cost": (_AWF_SIM, "compute_rebalance_cost"),
    "compute_symbol_strategy_evidence": (_SIGNAL_SELECTION, "compute_symbol_strategy_evidence"),
    "evaluate_layer1_readiness": (_SIGNAL_SELECTION, "evaluate_layer1_readiness"),
    "evaluate_layer2_gate": (_L2_GATE, "evaluate_layer2_gate"),
    "evaluate_meta_feasibility": (_L2_META, "evaluate_meta_feasibility"),
    "evaluate_outer_signal_opportunities": (_SIGNAL_SELECTION, "evaluate_outer_signal_opportunities"),
    "fit_layer1_inference_artifact": (_SIGNAL_SELECTION, "fit_layer1_inference_artifact"),
    "format_layer1_deployment_registry_table": (_TIERED_LOGGING, "format_layer1_deployment_registry_table"),
    "format_layer1_gate_table": (_TIERED_LOGGING, "format_layer1_gate_table"),
    "format_layer1_outer_fold_table": (_TIERED_LOGGING, "format_layer1_outer_fold_table"),
    "format_layer1_table": (_TIERED_LOGGING, "format_layer1_table"),
    "format_layer2_table": (_TIERED_LOGGING, "format_layer2_table"),
    "format_layer3_table": (_TIERED_LOGGING, "format_layer3_table"),
    "format_layer_universe_audit_table": (_TIERED_LOGGING, "format_layer_universe_audit_table"),
    "format_system_status": (_TIERED_LOGGING, "format_system_status"),
    "predict_layer1_signals": (_SIGNAL_SELECTION, "predict_layer1_signals"),
    "resolve_active_symbol_signals": (_AWF_SIM, "resolve_active_symbol_signals"),
    "run_l1_nested_swf": (_PIPELINE, "run_l1_nested_swf"),
    "run_l1_swf": (_PIPELINE, "run_l1_swf"),
    "run_l2_awf": (_PIPELINE, "run_l2_awf"),
    "run_l3_holdout": (_PIPELINE, "run_l3_holdout"),
    "run_tiered_pipeline": (_PIPELINE, "run_tiered_pipeline"),
    "select_layer2_champion": (_SELECTION, "select_layer2_champion"),
    "select_outer_symbol_opportunities": (_SIGNAL_SELECTION, "select_outer_symbol_opportunities"),
}

__all__ = tuple(sorted(_MODULE_ATTRS))


def _load_attr(module_name: str, attr_name: str) -> object:
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[attr_name] = value
    return value


def __getattr__(name: str) -> object:
    module_info = _MODULE_ATTRS.get(name)
    if module_info is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _load_attr(*module_info)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
