# src/domain/futures/strategy/tiered_workflow/selection.py
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import optuna

from src.domain.futures.allocation.contracts import (
    Layer2AllocationConfig,
    Layer2GateEvaluation,
    Layer2StudyResult,
)
from src.domain.futures.allocation.gates import (
    evaluate_layer2_gate,
)
from src.domain.futures.allocation.metrics import (
    _bars_per_year_for_tf,
    _deflated_sharpe_probability,
)
from src.domain.futures.allocation.parity import (
    assert_selection_replay_parity,
)
from src.domain.futures.allocation.scoring import (
    score_layer2_deployable_fallback as _score_layer2_deployable_fallback,
)
from src.domain.futures.optimization.evaluator import calc_n_trials_eff_entropy
from src.domain.futures.optimization.workflow import (
    evaluate_l2_trial_cached,
    layer2_constraints_from_trial,
)
from src.domain.futures.signals.contracts import ValidatedSignalBatch

_logger = logging.getLogger(__name__)


def _assert_selection_replay_parity(
    *,
    replay_evaluation: Any,
    final_evaluation: Any,
    tolerance: float = 1e-8,
) -> bool:
    return assert_selection_replay_parity(
        replay_evaluation=replay_evaluation,
        final_evaluation=final_evaluation,
        tolerance=tolerance,
    )


def _update_hashed_value(
    hasher: Any,
    *,
    name: str,
    value: bytes,
) -> None:
    """Append a length-prefixed named value to the hash stream."""
    name_bytes = name.encode("utf-8")
    hasher.update(len(name_bytes).to_bytes(8, "big", signed=False))
    hasher.update(name_bytes)
    hasher.update(len(value).to_bytes(8, "big", signed=False))
    hasher.update(value)


def _signal_batch_fingerprint(signal_batch: ValidatedSignalBatch) -> str:
    """Return a deterministic SHA-256 digest for the complete L2 signal input."""
    hasher = hashlib.sha256()
    _update_hashed_value(hasher, name="schema", value=b"l2-signal-batch-v1")
    _update_hashed_value(hasher, name="start_idx", value=str(int(signal_batch.start_idx)).encode("utf-8"))
    _update_hashed_value(hasher, name="end_idx", value=str(int(signal_batch.end_idx)).encode("utf-8"))
    _update_hashed_value(hasher, name="event_count", value=str(len(signal_batch.events)).encode("utf-8"))
    _update_hashed_value(hasher, name="registry_version", value=signal_batch.registry_version.encode("utf-8"))
    _update_hashed_value(hasher, name="model_version", value=signal_batch.model_version.encode("utf-8"))

    symbols = signal_batch.symbols
    _update_hashed_value(hasher, name="symbol_count", value=str(len(symbols)).encode("utf-8"))
    for symbol in symbols:
        _update_hashed_value(hasher, name="symbol", value=symbol.encode("utf-8"))

    for event in signal_batch.events:
        decision_time_ns = int(np.asarray(event.decision_time, dtype="datetime64[ns]").astype(np.int64))
        _update_hashed_value(hasher, name="decision_idx", value=str(int(event.decision_idx)).encode("utf-8"))
        _update_hashed_value(
            hasher,
            name="decision_time",
            value=str(decision_time_ns).encode("utf-8"),
        )
        _update_hashed_value(hasher, name="symbol", value=event.symbol.encode("utf-8"))
        _update_hashed_value(hasher, name="strategy_id", value=event.strategy_id.encode("utf-8"))
        _update_hashed_value(
            hasher,
            name="activation_context",
            value=event.activation_context.encode("utf-8"),
        )
        _update_hashed_value(hasher, name="side", value=str(int(event.side)).encode("utf-8"))
        _update_hashed_value(hasher, name="expected_net_bps", value=float(event.expected_net_bps).hex().encode("utf-8"))
        _update_hashed_value(
            hasher,
            name="expected_gross_bps",
            value=float(event.expected_gross_bps).hex().encode("utf-8"),
        )
        _update_hashed_value(hasher, name="q10_net_bps", value=float(event.q10_net_bps).hex().encode("utf-8"))
        _update_hashed_value(hasher, name="q10_gross_bps", value=float(event.q10_gross_bps).hex().encode("utf-8"))
        _update_hashed_value(hasher, name="q90_net_bps", value=float(event.q90_net_bps).hex().encode("utf-8"))
        _update_hashed_value(hasher, name="q90_gross_bps", value=float(event.q90_gross_bps).hex().encode("utf-8"))
        _update_hashed_value(
            hasher,
            name="expected_holding_bars",
            value=str(int(event.expected_holding_bars)).encode("utf-8"),
        )
        _update_hashed_value(
            hasher,
            name="quality_weight",
            value=float(event.quality_weight).hex().encode("utf-8"),
        )
        _update_hashed_value(hasher, name="registry_version", value=event.registry_version.encode("utf-8"))
        _update_hashed_value(hasher, name="model_version", value=event.model_version.encode("utf-8"))

    return hasher.hexdigest()


def _layer2_experiment_key(
    *,
    tf: str,
    window: Any,
    signal_batch: ValidatedSignalBatch,
    search_space_version: str,
) -> str:
    """같은 experiment key의 study를 매 실행마다 보존하기 위한 고유 키 생성."""
    signal_batch_fingerprint = _signal_batch_fingerprint(signal_batch)
    hash_input = (
        f"{tf}_{window.l2_start.isoformat()}_{window.holdout_start.isoformat()}_"
        f"{signal_batch_fingerprint}_{search_space_version}"
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
    """champion params에 evaluation.deploy_leverage(L*)를 추적용으로만 기록.

    구조적 no-op 판정에 따라 max_ann_vol / gross_exposure_cap 스케일링 제거:
    - vol-targeting이 하향 전용(min(scale,1.0))이라 max_ann_vol *= L*는 무효(결함 #2).
    - gross_exposure_cap *= L* 는 per_symbol이 먼저 binding → 영구 미도달(결함 #2).
    실제 배치는 run_l2_awf(deploy_leverage=L*)에서 수익률 직접 스케일로 실현됨(결함 #3 해소).

    보존:
        deployed["l2_deploy_leverage"] = l_star  (추적 및 run_l2_awf 전달용)
        kelly_fraction 불변.
    """
    config = Layer2AllocationConfig.from_mapping(params)
    if not config.l2_deploy_enabled:
        return params

    # evaluation.deploy_leverage 재사용 (재계산 금지 — 단일 SSOT)
    l_star = float(getattr(evaluation, "deploy_leverage", 1.0))
    _l_binding = str(getattr(evaluation, "deploy_binding", ""))

    if l_star <= 1.0 + 1e-6:
        return params

    deployed: dict[str, Any] = dict(params)

    # 추적 및 run_l2_awf deploy_leverage 전달용 — 천장 스케일링 없음 (realized_mode=return_scaling)
    deployed["l2_deploy_leverage"] = l_star

    _logger.info(
        "[L2-DEPLOY-C4] L*=%.3f (binding=%s) | realized_mode=return_scaling | kelly=%.3f(불변) | tf=%s",
        l_star,
        _l_binding,
        float(config.kelly_fraction),
        tf,
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
    prebuilt_cache: Any | None = None,
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
            awf_folds=awf_folds,
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
            awf_folds=awf_folds,
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

    if prebuilt_cache is not None:
        cache = prebuilt_cache
    else:
        from src.domain.futures.allocation.simulation import build_l2_simulation_cache
        cache = build_l2_simulation_cache(aligned, signal_batch, tf)

    _eval_memo: dict[tuple[Any, ...], Any] = {}

    def _eval_candidate(
        trial: optuna.trial.FrozenTrial,
    ) -> tuple[optuna.trial.FrozenTrial, Any, Layer2AllocationConfig]:
        cfg_mapping = Layer2AllocationConfig.from_mapping(dict(trial.params))
        eval_val = evaluate_l2_trial_cached(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=cfg_mapping,
            caps=caps,
            tf=tf,
            eval_tag="selection",
            _memo=_eval_memo,
        )
        return trial, eval_val, cfg_mapping

    # 1차 필터링: 이미 사전 게이트 통과(l2_promotion_passed) 이력이 있는 trial들 선별
    gate_passed_candidates = [
        t for t in replay_candidates
        if t.user_attrs.get("l2_promotion_passed", False)
        or t.user_attrs.get("promotion_passed", False)
    ]

    if gate_passed_candidates:
        eval_candidates = gate_passed_candidates[:8]
        _logger.info(
            "[L2-SELECTION] Found %d gate-passed trials in frontier. Reducing replay size to %d.",
            len(gate_passed_candidates),
            len(eval_candidates),
        )
    else:
        eval_candidates = replay_candidates[:fallback_limit]
        _logger.info(
            "[L2-SELECTION] No gate-passed trials found. Reducing diagnostic replay size to %d.",
            len(eval_candidates),
        )

    # ThreadPool 평가: numba GIL 해제 활용 → ProcessPool 대비 fork/serialize 오버헤드 제거
    if len(eval_candidates) <= 1:
        evaluated_triples = [_eval_candidate(trial) for trial in eval_candidates]
    else:
        max_workers = min(len(eval_candidates), 4)
        evaluated_triples = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_eval_candidate, t): t for t in eval_candidates}
            for future in as_completed(future_map):
                evaluated_triples.append(future.result())

    for candidate, candidate_evaluation, candidate_config in evaluated_triples:
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

        # 단일 gate 평가: pre-gate + final-gate 중복 제거
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
        gate = evaluate_layer2_gate(
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
            dsr_hybrid=float(dsr),
            psr_hybrid=float(candidate_evaluation.psr_hybrid),
            recent_fold_passed=getattr(candidate_evaluation, "recent_fold_passed", None),
            recent_fold_sharpe=getattr(candidate_evaluation, "recent_fold_sharpe", None),
            worst_fold_cagr=getattr(candidate_evaluation, "worst_fold_cagr", None),
            positive_block_delta_ratio=getattr(
                candidate_evaluation,
                "positive_block_delta_ratio",
                None,
            ),
            config=candidate_config,
        )
        deployable_score = _score_layer2_deployable_fallback(
            candidate_evaluation,
            config=candidate_config,
        )

        if replay_mismatch:
            _logger.debug(
                "[L2-REPLAY] Trial #%d | stored_CAGR=%.4f replay_CAGR=%.4f | "
                "stored_MDD=%.4f replay_MDD=%.4f | stored_LCB=%.4f replay_LCB=%.4f",
                candidate.number,
                stored_cagr,
                candidate_evaluation.cagr_hybrid,
                stored_mdd,
                candidate_evaluation.mdd_hybrid,
                stored_growth_lcb,
                candidate_evaluation.growth_lcb_hybrid,
            )

        # gate 진단 로그
        _logger.debug(
            "[L2-REPLAY-GATE] Trial #%d | gate=%s blocker=%s | "
            "cagr=%.4f sortino=%.4f sharpe=%.4f calmar=%.4f | "
            "mdd=%.4f folds=%.2f trades=%d dsr=%.4f",
            candidate.number,
            gate.promotion_passed,
            gate.promotion_blocker,
            float(candidate_evaluation.cagr_hybrid),
            float(candidate_evaluation.sortino_hybrid),
            float(candidate_evaluation.sharpe_hybrid),
            float(candidate_evaluation.cagr_hybrid / (candidate_evaluation.mdd_hybrid + 1e-9)),
            float(candidate_evaluation.mdd_hybrid),
            float(candidate_evaluation.fold_pass_ratio),
            int(candidate_evaluation.trade_count),
            float(dsr),
        )

        if (
            best_diagnostic_trial is None
            or (
                int(gate.promotion_passed),
                float(deployable_score.score),
                float(dsr),
                float(candidate_evaluation.objective_value),
                float(candidate_evaluation.growth_lcb_hybrid),
                float(candidate_evaluation.cagr_hybrid),
                -int(candidate.number),
            )
            > (
                int(best_diagnostic_gate.promotion_passed) if best_diagnostic_gate is not None else 0,
                float(
                    _score_layer2_deployable_fallback(
                        best_diagnostic_evaluation,
                        config=Layer2AllocationConfig.from_mapping(dict(best_diagnostic_trial.params)),
                    ).score
                )
                if best_diagnostic_trial is not None and best_diagnostic_evaluation is not None
                else -1e6,
                float(best_diagnostic_dsr),
                float(best_diagnostic_evaluation.objective_value) if best_diagnostic_evaluation is not None else -1e6,
                float(best_diagnostic_evaluation.growth_lcb_hybrid) if best_diagnostic_evaluation is not None else -1e6,
                float(best_diagnostic_evaluation.cagr_hybrid) if best_diagnostic_evaluation is not None else -1e6,
                -int(best_diagnostic_trial.number),
            )
        ):
            best_diagnostic_trial = candidate
            best_diagnostic_evaluation = candidate_evaluation
            best_diagnostic_gate = gate
            best_diagnostic_dsr = float(dsr)

        constraints_ok = all(v <= 0.0 for v in gate.optuna_constraint_values)
        if constraints_ok and gate.promotion_passed:
            # D4: argmax(dsr, cagr) → argmax(sortino, cagr) 교체.
            # DSR은 자기참조(동일 신호셋 파라미터 섭동) → 독립성 불성립.
            # Sortino가 하방위험 조정 shape를 직접 반영 (shape 최적화 D1과 정합).
            passed_candidates.append((
                float(candidate_evaluation.sortino_hybrid),
                float(candidate_evaluation.cagr_hybrid),
                candidate,
                candidate_evaluation,
                float(dsr),
            ))

    # D4: gate-pass 후보 중 argmax(sortino_hybrid, cagr_hybrid) 선택
    if passed_candidates:
        best_entry = max(passed_candidates, key=lambda x: (x[0], x[1], -x[2].number))
        champion_trial = best_entry[2]
        champion_evaluation = best_entry[3]
        champion_dsr = best_entry[4]
        _logger.info(
            "[L2-SELECTION] %d gate-pass 후보 수집 → champion Trial #%d Sortino=%.4f CAGR=%.4f",
            len(passed_candidates),
            champion_trial.number,
            best_entry[0],
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
            sim_cache=cache,
            awf_folds=awf_folds,
            eval_memo=_eval_memo,
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
    # Phase A 진단: champion 저장 시점 지문 (stored metric vs deploy param L* 결합 추적).
    if _logger.isEnabledFor(logging.DEBUG):
        from src.domain.futures.allocation.simulation import _content_hash_dataclass

        _champ_rets = getattr(champion_evaluation, "returns_hybrid", ())
        _logger.debug(
            "[L2-CHAMPION-FP] trial=%d cfg_ch=%s stored_cagr=%.6f stored_mdd=%.6f "
            "stored_L*=%.6f deployed_param_L*=%.6f n_rets=%d",
            int(champion_trial.number),
            _content_hash_dataclass(Layer2AllocationConfig.from_mapping(dict(champion_trial.params))),
            float(getattr(champion_evaluation, "cagr_hybrid", 0.0)),
            float(getattr(champion_evaluation, "mdd_hybrid", 0.0)),
            float(getattr(champion_evaluation, "deploy_leverage", 1.0)),
            float(deployed_champion_params.get("l2_deploy_leverage", 1.0)),
            len(_champ_rets),
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
        sim_cache=cache,
        awf_folds=awf_folds,
        eval_memo=_eval_memo,
    )
