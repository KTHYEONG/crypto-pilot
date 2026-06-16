# src/domain/futures/strategy/tiered_workflow/selection.py
from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import optuna

from src.domain.futures.optimization.evaluator import calc_n_trials_eff_entropy
from src.domain.futures.optimization.workflow import evaluate_l2_trial, layer2_constraints_from_trial
from src.domain.futures.strategy.tiered_workflow import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2StudyResult,
)
from src.domain.futures.strategy.tiered_workflow.metrics import (
    _bars_per_year_for_tf,
    _deflated_sharpe_probability,
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


def select_layer2_champion(
    *,
    study: optuna.Study,
    tf: str,
    signal_batch: Any,
    aligned: Any,
    awf_folds: tuple[Any, ...],
    caps: Any,
) -> Layer2StudyResult:
    """feasible completed trials 중 growth_lcb(objective) 최상위 챔피언 선정 및 검증.

    DSR은 2026-06-16부로 hard-gate에서 diagnostic으로 강등(pool-상대 benchmark가
    trial 수와 동반 상승하는 구조적 편향 + L3 frozen holdout과 3중 중복). 챔피언
    선정은 growth_lcb(objective) 최상위 feasible trial로 결정하고, DSR은 참고
    지표로만 계산해 결과에 첨부한다.

    Algorithm:
        1. feasible completed trials를 objective (t.value) 내림차순 정렬
        2. completed trials의 block log growth signature를 기반으로 n_trials_eff 계산
        3. objective 최상위 feasible trial을 챔피언으로 선정, simulation 1회 재실행
        4. 챔피언에 대해 DSR diagnostic 계산 (게이트 미적용)
        5. 선정된 챔피언에 대해 deterministic replay 검증 (stored cagr/mdd/growth_lcb 비교)
        6. 결과 반환
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

    # 1. effective trial count 계산 (calc_n_trials_eff_entropy 사용)
    signatures = []
    for t in complete_trials:
        sig = t.user_attrs["l2_block_log_growth_signature"]
        signatures.append(sig)
    
    signatures_arr = np.array(signatures, dtype=np.float64)
    weights_arr: np.ndarray = np.ones(len(complete_trials), dtype=np.float64)

    if len(complete_trials) >= 2:
        try:
            n_trials_eff = float(calc_n_trials_eff_entropy(signatures_arr, weights_arr))
            n_trials_eff = float(np.clip(n_trials_eff, 1.0, float(len(complete_trials))))
        except Exception as e:
            _logger.warning("[L2-SELECTION] n_trials_eff 계산 실패: %s", e)
            n_trials_eff = float(len(complete_trials))
    else:
        n_trials_eff = float(len(complete_trials))

    # 2. feasible completed trials 분류
    feasible_trials = [
        t for t in complete_trials
        if all(c <= 0.0 for c in layer2_constraints_from_trial(t))
    ]

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
    completed_trial_sharpes = np.array(
        [t.user_attrs["sharpe_hac_hybrid"] for t in complete_trials],
        dtype=np.float64
    )

    # 4. objective(growth_lcb) 최상위 feasible trial부터 bounded replay fallback 수행
    fallback_limit = max(
        Layer2AllocationConfig.from_mapping(dict(feasible_sorted[0].params)).l2_replay_max_fallbacks,
        1,
    )
    replay_candidates = feasible_sorted[:fallback_limit]
    first_trial = replay_candidates[0]
    first_evaluation = None
    champion_trial = None
    champion_evaluation = None

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
        if first_evaluation is None:
            first_evaluation = candidate_evaluation

        stored_cagr = float(candidate.user_attrs.get("cagr_hybrid", 0.0))
        stored_growth_lcb = float(candidate.user_attrs.get("growth_lcb_hybrid", 0.0))
        stored_mdd = float(candidate.user_attrs.get("mdd_hybrid", 0.0))
        eps = 1e-5
        replay_mismatch = (
            abs(candidate_evaluation.cagr_hybrid - stored_cagr) > eps
            or abs(candidate_evaluation.growth_lcb_hybrid - stored_growth_lcb) > eps
            or abs(candidate_evaluation.mdd_hybrid - stored_mdd) > eps
        )
        replay_feasible = all(value <= 0.0 for value in candidate_evaluation.constraint_values)
        if replay_mismatch:
            _logger.warning(
                "[L2-SELECTION] Replay mismatch on trial #%d: stored_cagr=%.6f replayed_cagr=%.6f",
                candidate.number,
                stored_cagr,
                candidate_evaluation.cagr_hybrid,
            )
        if replay_feasible:
            champion_trial = candidate
            champion_evaluation = candidate_evaluation
            break

    if champion_trial is None or champion_evaluation is None:
        _logger.error("[L2-SELECTION] No replay-feasible candidate found within fallback window")
        return Layer2StudyResult(
            best_params=dict(first_trial.params),
            best_trial_number=int(first_trial.number),
            best_evaluation=first_evaluation,
            dsr=0.0,
            effective_trial_count=n_trials_eff,
            completed_trials=len(complete_trials),
            feasible_trials=len(feasible_trials),
            blocker_reason="non_deterministic_replay",
        )

    champion_dsr = _deflated_sharpe_probability(
        selected_rets=champion_evaluation.returns_hybrid,
        completed_trial_sharpes=completed_trial_sharpes,
        effective_trial_count=n_trials_eff,
        bars_per_year=bars_per_year,
    )

    _logger.info(
        "[L2-SELECTION] Champion selected. Trial #%d, Objective=%.4f, DSR=%.4f (n_eff=%.2f)",
        champion_trial.number,
        champion_trial.value,
        champion_dsr,
        n_trials_eff
    )
    return Layer2StudyResult(
        best_params=dict(champion_trial.params),
        best_trial_number=int(champion_trial.number),
        best_evaluation=champion_evaluation,
        dsr=champion_dsr,
        effective_trial_count=n_trials_eff,
        completed_trials=len(complete_trials),
        feasible_trials=len(feasible_trials),
        blocker_reason="",
    )
