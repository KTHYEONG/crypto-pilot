# src/domain/futures/strategy/tiered_workflow/selection.py
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna

from src.domain.futures.optimization.evaluator import calc_n_trials_eff_entropy
from src.domain.futures.optimization.workflow import evaluate_l2_trial, layer2_constraints_from_trial
from src.domain.futures.strategy.tiered_workflow import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2GateEvaluation,
    Layer2StudyResult,
)
from src.domain.futures.strategy.tiered_workflow.l2_gate import (
    evaluate_layer2_gate,
)
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _bars_per_year_for_tf,
    _deflated_sharpe_probability,
)
from src.domain.futures.strategy.tiered_workflow.risk_deployment import (
    calibrate_deployment_leverage,
)

_logger = logging.getLogger(__name__)


def _layer2_experiment_key(
    *,
    tf: str,
    window: Any,
    signal_batch: Any,
    search_space_version: str,
) -> str:
    """같은 experiment key의 study를 매 실행마다 보존하기 위한 고유 키 생성."""
    events = getattr(signal_batch, "events", ())
    event_count = len(events)
    # timeframe, l2_start, holdout_start, events_count 및 search_space_version을 고유 요소로 바인딩
    hash_input = (
        f"{tf}_{window.l2_start.isoformat()}_{window.holdout_start.isoformat()}_"
        f"{event_count}_{search_space_version}"
    )
    h = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:12]
    return f"l2_study_{tf}_{h}"


def _build_layer2_replay_frontier(
    feasible_trials: Sequence[optuna.trial.FrozenTrial],
    *,
    fallback_limit: int,
) -> list[optuna.trial.FrozenTrial]:
    if not feasible_trials:
        return []

    def _value_key(trial: optuna.trial.FrozenTrial) -> tuple[float, int]:
        value = float(trial.value) if trial.value is not None else -1e6
        return (value, -int(trial.number))

    def _attr_key(name: str) -> Any:
        def _key(trial: optuna.trial.FrozenTrial) -> tuple[float, int]:
            return (
                float(trial.user_attrs.get(name, -1e6)),
                -int(trial.number),
            )

        return _key

    frontier: dict[int, optuna.trial.FrozenTrial] = {}

    def _take_top(
        trials: Sequence[optuna.trial.FrozenTrial],
        limit: int,
        key: Any,
    ) -> None:
        for trial in sorted(trials, key=key, reverse=True)[: max(limit, 0)]:
            frontier[int(trial.number)] = trial

    _take_top(feasible_trials, fallback_limit, _value_key)
    half_limit = max(fallback_limit // 2, 1)
    _take_top(feasible_trials, half_limit, _attr_key("growth_lcb_hybrid"))
    _take_top(feasible_trials, half_limit, _attr_key("sharpe_hac_hybrid"))
    _take_top(feasible_trials, half_limit, _attr_key("cagr_hybrid"))

    return sorted(
        frontier.values(),
        key=lambda trial: (
            float(trial.value) if trial.value is not None else -1e6,
            float(trial.user_attrs.get("sharpe_hac_hybrid", -1e6)),
            float(trial.user_attrs.get("cagr_hybrid", -1e6)),
            -int(trial.number),
        ),
        reverse=True,
    )


def _trial_metric(evaluation: Any, name: str, fallback_name: str, default: float = 0.0) -> float:
    value = getattr(evaluation, name, None)
    if isinstance(value, (int, float)):
        return float(value)
    fallback = getattr(evaluation, fallback_name, None)
    if isinstance(fallback, (int, float)):
        return float(fallback)
    return float(default)


def _apply_deployment_to_params(
    params: dict[str, Any],
    evaluation: Any,
    tf: str,
) -> dict[str, Any]:
    """Fix-A: champion params에 결정론적 레버리지 L*를 적용.

    kelly_fraction *= L*. max_ann_vol도 존재하면 동일 배율 적용.
    Sharpe/DSR은 스케일 불변이므로 그대로 유지.
    """
    config = Layer2AllocationConfig.from_mapping(params)
    if not config.l2_deploy_enabled:
        return params

    returns_hybrid = getattr(evaluation, "returns_hybrid", None)
    if not returns_hybrid:
        return params

    rets = np.array(returns_hybrid, dtype=np.float64)
    if rets.size < 2:
        return params

    lev, binding = calibrate_deployment_leverage(
        fit_rets=rets,  # OOS 대리: AWF fit-leg 미노출, binding=hard_cap으로 실질 look-ahead 없음
        mdd_cap=float(config.l2_max_mdd_abs),
        cvar_cap=float(config.l2_max_cvar_95),
        mdd_margin=float(config.l2_deploy_mdd_margin),
        l_hard_cap=float(config.l2_deploy_l_hard_cap),
    )

    if lev <= 1.0 + 1e-6:
        return params

    deployed: dict[str, Any] = dict(params)
    new_kelly = float(config.kelly_fraction) * lev
    deployed["kelly_fraction"] = new_kelly

    raw_vol = params.get("max_ann_vol", params.get("vol_target"))
    if isinstance(raw_vol, (int, float)):
        deployed["max_ann_vol"] = float(raw_vol) * lev

    _logger.info(
        "[L2-DEPLOY-FIX-A] L*=%.3f (binding=%s) | kelly %.3f→%.3f | tf=%s",
        lev, binding, config.kelly_fraction, new_kelly, tf,
    )
    return deployed


def select_layer2_champion(
    *,
    study: optuna.Study,
    tf: str,
    signal_batch: Any,
    aligned: Any,
    awf_folds: tuple[Any, ...],
    caps: Any,
    min_dsr: float = 0.60,
) -> Layer2StudyResult:
    """feasible completed trials 중 growth_lcb(objective) 최상위 챔피언 선정 및 검증.

    DSR은 2026-06-16부로 diagnostic으로 강등되었으나, 2026-06-17 무결성 강화
    결정에 따라 pipeline 하드 게이트와 동기화하여 선정 단계에서도 필터링을 수행한다.
    """
    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value > -1e6
        and "l2_block_log_growth_signature" in t.user_attrs
        and "sharpe_hac_hybrid" in t.user_attrs
    ]

    if not complete_trials:
        _logger.warning("[L2-SELECTION] 완료된 trials가 없어 study_error 반환")
        return Layer2StudyResult(
            best_params={},
            best_trial_number=None,
            best_evaluation=None,
            dsr=0.0,
            effective_trial_count=0.0,
            completed_trials=0,
            feasible_trials=0,
            blocker_reason="no_complete_trials",
        )

    # 1. feasible completed trials 분류 (Optuna constraints) — Fix C: 먼저 분류하여 n_trials_eff에 사용
    feasible_trials = [
        t for t in complete_trials
        if all(c <= 0.0 for c in layer2_constraints_from_trial(t))
    ]

    # 2. effective trial count 계산 — Fix C: feasible trials 서명만 사용 (DSR 벤치마크 정직화)
    feasible_signatures = [
        t.user_attrs["l2_block_log_growth_signature"] for t in feasible_trials
    ]
    n_feasible = len(feasible_trials)

    if n_feasible >= 2:
        signatures_arr = np.array(feasible_signatures, dtype=np.float64)
        weights_arr: np.ndarray = np.ones(n_feasible, dtype=np.float64)
        try:
            n_trials_eff = float(calc_n_trials_eff_entropy(signatures_arr, weights_arr))
            n_trials_eff = float(np.clip(n_trials_eff, 1.0, float(n_feasible)))
        except Exception as e:
            _logger.warning("[L2-SELECTION] n_trials_eff 계산 실패: %s", e)
            n_trials_eff = float(n_feasible)
    else:
        n_trials_eff = float(max(n_feasible, 1))

    # 3. objective 값 기준으로 내림차순 정렬
    feasible_sorted = sorted(
        feasible_trials,
        key=lambda t: t.value if t.value is not None else -1e6,
        reverse=True
    )

    if not feasible_sorted:
        _logger.warning("[L2-SELECTION] feasible한 trials가 없음 -> fallback")
        best_overall = max(complete_trials, key=lambda t: t.value if t.value is not None else -1e6)
        return Layer2StudyResult(
            best_params=dict(best_overall.params),
            best_trial_number=int(best_overall.number),
            best_evaluation=None,
            dsr=0.0,
            effective_trial_count=n_trials_eff,
            completed_trials=len(complete_trials),
            feasible_trials=0,
            blocker_reason="no_feasible_trials",
        )

    bars_per_year = _bars_per_year_for_tf(tf)
    # Fix C: DSR pool을 feasible trials로 제한 → 다중검정 벤치마크 정직화
    completed_trial_sharpes = np.array(
        [t.user_attrs["sharpe_hac_hybrid"] for t in feasible_trials],
        dtype=np.float64
    )

    # 4. objective(growth_lcb) 최상위 feasible trial부터 bounded replay fallback 수행
    # + DSR 하드 게이트 필터링 (2026-06-17)
    fallback_limit = max(
        Layer2AllocationConfig.from_mapping(dict(feasible_sorted[0].params)).l2_replay_max_fallbacks,
        24,
    )
    replay_candidates = _build_layer2_replay_frontier(
        feasible_sorted,
        fallback_limit=fallback_limit,
    )
    first_trial = replay_candidates[0]
    first_evaluation = None
    first_dsr = 0.0
    # Fix B: gate-pass 후보를 수집하여 argmax(dsr, cagr)로 champion 선택
    passed_candidates: list[
        tuple[float, float, optuna.trial.FrozenTrial, Any, float]
    ] = []  # (dsr, cagr_hybrid, trial, evaluation, candidate_dsr)
    champion_trial = None
    champion_evaluation = None
    champion_dsr = 0.0
    best_diagnostic_trial = None
    best_diagnostic_evaluation = None
    best_diagnostic_gate: Layer2GateEvaluation | None = None
    best_diagnostic_dsr = float("-inf")

    for candidate in replay_candidates:
        candidate_config = Layer2AllocationConfig.from_mapping(dict(candidate.params))
        candidate_evaluation = evaluate_l2_trial(
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=candidate_config,
            caps=caps,
            tf=tf,
        )
        
        # Calculate DSR for this candidate
        dsr = _deflated_sharpe_probability(
            selected_rets=candidate_evaluation.returns_hybrid,
            completed_trial_sharpes=completed_trial_sharpes,
            effective_trial_count=n_trials_eff,
            bars_per_year=bars_per_year,
        )

        if first_evaluation is None:
            first_evaluation = candidate_evaluation
            first_dsr = dsr

        stored_cagr = float(candidate.user_attrs.get("cagr_hybrid", 0.0))
        stored_growth_lcb = float(candidate.user_attrs.get("growth_lcb_hybrid", 0.0))
        stored_mdd = float(candidate.user_attrs.get("mdd_hybrid", 0.0))
        eps = 1e-5
        replay_mismatch = (
            abs(candidate_evaluation.cagr_hybrid - stored_cagr) > eps
            or abs(candidate_evaluation.growth_lcb_hybrid - stored_growth_lcb) > eps
            or abs(candidate_evaluation.mdd_hybrid - stored_mdd) > eps
        )
        
        pre_dsr_gate = getattr(candidate_evaluation, "gate", None)
        if pre_dsr_gate is None:
            raw_sharpe = _trial_metric(
                candidate_evaluation,
                "sharpe_hybrid",
                "sharpe_hac_hybrid",
            )
            sharpe_hac_baseline_ew = _trial_metric(
                candidate_evaluation,
                "sharpe_hac_baseline_ew",
                "sharpe_hac_baseline",
            )
            pre_dsr_gate = evaluate_layer2_gate(
                deployment_failed=bool(candidate_evaluation.constraint_values[0] > 0.0),
                support_leak_count=0,
                cagr_hybrid=float(candidate_evaluation.cagr_hybrid),
                sharpe_hybrid=raw_sharpe,
                sharpe_hac_hybrid=float(candidate_evaluation.sharpe_hac_hybrid),
                sharpe_hac_baseline=sharpe_hac_baseline_ew,
                sortino_hybrid=float(candidate_evaluation.sortino_hybrid),
                mar_hybrid=float(
                    candidate_evaluation.cagr_hybrid / (candidate_evaluation.mdd_hybrid + 1e-9)
                ),
                mdd_hybrid=float(candidate_evaluation.mdd_hybrid),
                cvar_95_hybrid=float(candidate_evaluation.cvar_95_hybrid),
                fold_pass_ratio=float(candidate_evaluation.fold_pass_ratio),
                active_block_count=len(candidate_evaluation.block_metrics),
                friction_pass_pct=float(candidate_evaluation.break_even_pass_pct),
                trade_count=int(candidate_evaluation.trade_count),
                growth_lcb_hybrid=float(candidate_evaluation.growth_lcb_hybrid),
                growth_lcb_baseline=float(candidate_evaluation.growth_lcb_baseline),
                dsr_hybrid=None,
                config=candidate_config,
            )
        constraints_ok = all(value <= 0.0 for value in pre_dsr_gate.optuna_constraint_values)
        raw_sharpe = _trial_metric(
            candidate_evaluation,
            "sharpe_hybrid",
            "sharpe_hac_hybrid",
        )
        sharpe_hac_baseline_ew = _trial_metric(
            candidate_evaluation,
            "sharpe_hac_baseline_ew",
            "sharpe_hac_baseline",
        )
        final_gate = evaluate_layer2_gate(
            deployment_failed=bool(pre_dsr_gate.optuna_constraint_values[0] > 0.0),
            support_leak_count=0,
            cagr_hybrid=float(candidate_evaluation.cagr_hybrid),
            sharpe_hybrid=raw_sharpe,
            sharpe_hac_hybrid=float(candidate_evaluation.sharpe_hac_hybrid),
            sharpe_hac_baseline=sharpe_hac_baseline_ew,
            sortino_hybrid=float(candidate_evaluation.sortino_hybrid),
            mar_hybrid=float(
                candidate_evaluation.cagr_hybrid / (candidate_evaluation.mdd_hybrid + 1e-9)
            ),
            mdd_hybrid=float(candidate_evaluation.mdd_hybrid),
            cvar_95_hybrid=float(candidate_evaluation.cvar_95_hybrid),
            fold_pass_ratio=float(candidate_evaluation.fold_pass_ratio),
            active_block_count=len(candidate_evaluation.block_metrics),
            friction_pass_pct=float(candidate_evaluation.break_even_pass_pct),
            trade_count=int(candidate_evaluation.trade_count),
            growth_lcb_hybrid=float(candidate_evaluation.growth_lcb_hybrid),
            growth_lcb_baseline=float(candidate_evaluation.growth_lcb_baseline),
            dsr_hybrid=float(dsr),
            config=candidate_config,
        )

        if replay_mismatch:
            _logger.debug(
                "[L2-SELECTION] Replay mismatch on trial #%d: stored_cagr=%.6f replayed_cagr=%.6f",
                candidate.number,
                stored_cagr,
                candidate_evaluation.cagr_hybrid,
            )

        if (
            best_diagnostic_trial is None
            or (
                int(final_gate.promotion_passed),
                float(dsr),
                float(candidate_evaluation.objective_value),
                float(candidate_evaluation.growth_lcb_hybrid),
                float(candidate_evaluation.cagr_hybrid),
                -int(candidate.number),
            )
            > (
                int(best_diagnostic_gate.promotion_passed) if best_diagnostic_gate is not None else 0,
                float(best_diagnostic_dsr),
                float(best_diagnostic_evaluation.objective_value) if best_diagnostic_evaluation is not None else -1e6,
                float(best_diagnostic_evaluation.growth_lcb_hybrid) if best_diagnostic_evaluation is not None else -1e6,
                float(best_diagnostic_evaluation.cagr_hybrid) if best_diagnostic_evaluation is not None else -1e6,
                -int(best_diagnostic_trial.number),
            )
        ):
            best_diagnostic_trial = candidate
            best_diagnostic_evaluation = candidate_evaluation
            best_diagnostic_gate = final_gate
            best_diagnostic_dsr = float(dsr)

        if constraints_ok and final_gate.promotion_passed:
            # Fix B: break 제거 — 전수 수집 후 argmax(dsr, cagr) 선택
            passed_candidates.append((
                float(dsr),
                float(candidate_evaluation.cagr_hybrid),
                candidate,
                candidate_evaluation,
                float(dsr),
            ))

    # Fix B: gate-pass 후보 중 argmax(dsr, cagr_hybrid) 선택
    if passed_candidates:
        best_entry = max(passed_candidates, key=lambda x: (x[0], x[1]))
        champion_trial = best_entry[2]
        champion_evaluation = best_entry[3]
        champion_dsr = best_entry[4]
        _logger.info(
            "[L2-SELECTION] %d gate-pass 후보 수집 → champion Trial #%d DSR=%.4f CAGR=%.4f",
            len(passed_candidates),
            champion_trial.number,
            champion_dsr,
            best_entry[1],
        )

    if champion_trial is None or champion_evaluation is None:
        diagnostic_trial = best_diagnostic_trial or first_trial
        diagnostic_evaluation = best_diagnostic_evaluation or first_evaluation
        diagnostic_gate = best_diagnostic_gate
        reason = (
            diagnostic_gate.promotion_blocker
            if diagnostic_gate is not None and diagnostic_gate.promotion_blocker
            else "non_deterministic_replay"
        )
        _logger.error("[L2-SELECTION] No feasible candidate found within fallback window (reason=%s)", reason)
        # Fix-A: 블로킹 경로에서도 배치 적용 → 파이프라인 최종 재실행에서 CAGR 변화 관측
        deployed_params = _apply_deployment_to_params(
            dict(diagnostic_trial.params),
            diagnostic_evaluation,
            tf,
        )
        return Layer2StudyResult(
            best_params=deployed_params,
            best_trial_number=int(diagnostic_trial.number),
            best_evaluation=diagnostic_evaluation,
            dsr=float(best_diagnostic_dsr if best_diagnostic_dsr > float("-inf") else first_dsr),
            effective_trial_count=n_trials_eff,
            completed_trials=len(complete_trials),
            feasible_trials=len(feasible_trials),
            blocker_reason=reason,
        )

    _logger.info(
        "[L2-SELECTION] Champion selected. Trial #%d, Objective=%.4f, DSR=%.4f (n_eff=%.2f)",
        champion_trial.number,
        champion_trial.value,
        champion_dsr,
        n_trials_eff
    )
    # Fix-A: champion 경로에서도 배치 적용
    deployed_champion_params = _apply_deployment_to_params(
        dict(champion_trial.params),
        champion_evaluation,
        tf,
    )
    return Layer2StudyResult(
        best_params=deployed_champion_params,
        best_trial_number=int(champion_trial.number),
        best_evaluation=champion_evaluation,
        dsr=champion_dsr,
        effective_trial_count=n_trials_eff,
        completed_trials=len(complete_trials),
        feasible_trials=len(feasible_trials),
        blocker_reason="",
    )
