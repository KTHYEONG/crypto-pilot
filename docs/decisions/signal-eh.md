---
title: Signal Eval Gate — Decision Records
domain: futures/strategy
type: adr
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/execution/opt_main_futures.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/strategy/rule_diagnostics.py
last_verified: 2026-06-09
---

## [2026-06-09] Regime-Cell Conditional Admission (P1)
- **Delta:** `_regime_cell_admission()` 신규 + `_recommendation_threshold_checks` OR-path 주입. 변이가 특정 regime cell에서 `n_g≥60 ∧ μ_g≥8bps ∧ t_g≥1.0` 충족 시 글로벌 평균 게이트 우회. 5개 config 파라미터(`regime_cell_admission_enabled` 등) + env override(`FUTURES_CANDIDATE_REGIME_CELL_*`) 추가. 활성화 결과: RECOMMENDED 3→10, BLOCKED 30→19, Fold1 −16.1→+20.4bps, Status `blocked`→`wf_eligible`(sel=662).
- **Rationale:** 글로벌 풀링 게이트가 carry/reversion 분산 신호를 평균 희석으로 학살 → B0 ensemble μ(a,g)가 회전할 직교 자산 부재. 적재적소 복리증식의 전제인 archetype 다양성 확보.
- **Edge Cases/Trade-offs:** 안전게이트(`min_obs`/`q10_fail`/`event_density`)는 OR-path에서도 필수 유지. σ_g=0 시 ε 가드. **t_g는 NW 미적용 IID SE → 중첩보유 하 낙관편향**; per-cell OOS 선택은 multiple-comparison 위험 증폭 → purged/embargoed nested validation이 라이브 전 후속 과제. default off(회귀 안전).

## [2026-06-09] Signal Eval 로그/코드 정합성 수정 (P1/P2)
- **Delta:** (1) BLOCKED `mapping`에 `breakeven_hard_gate` 라벨 추가 → raw key 노출 제거. (2) `[:40]` 절단 폐기 후 `_wrap_segments`로 전체 게이트 분포를 41-char 컬럼에 줄바꿈(82-width 테이블 보존). (3) `[GATE FAILURES: PER-VARIANT]` 신규 — variant별 실패 게이트(상위 20). (4) WF fold 테이블 `PriorP90→RlzdMean`(실제 pass 게이트=`realized_mean≥min_fold_realized_edge_bps`), `Rank IC→IC(diag)`(참고값) relabel + `(★gate)` 서브헤더. bridge.py `wf_fold_details`에 `realized_mean_bps`/`selected_total` 추가.
- **Rationale:** 13개 게이트 중 5개만 로그 노출 + Pass 결정인자(realized_mean/lift)가 미표시되고 무관한 Rank IC가 옆에 놓여 "음수 IC가 PASS" 오독 유발. pass/fail을 표시 지표로 역검증 불가했음.
- **Edge Cases/Trade-offs:** per-variant 게이트 리스트는 26-char 초과 시 `…` 절단(전체 분포는 BLOCKED 집계가 SSOT). WF Mode 컬럼 `ensemble_b0`(11>10) 미정렬은 pre-existing. 게이트 임계값/공식은 불변 → architecture 동기화 불필요.

## [2026-06-09] 게이트 진단 결과 — 미구현 후속(deferred)
- **Delta:** 분석만 수행, 코드 미변경. (P3) `mean_edge=10` vs `breakeven(≥max(8,RT·0.6)+tstat≥1.0)` 부분 중복 — breakeven이 우세하므로 dedup 후보. median(-100)/p10(-600)/regime_edge(promotion_level=variant→강제 True)는 분리력 0인 dead 게이트. (P4) 소표본 강엣지(bcr_48: 66bps/77obs<100) EB shrinkage 재평가, 이산 score(donchian류)는 `oos_rank_ic`/`ic_tstat` 구조적 불이익 → 연속/이산 분기.
- **Rationale:** false-negative(억울한 탈락) 위험 정량화. dead 게이트는 평가 기준 투명성 저하.
- **Edge Cases/Trade-offs:** P3/P4는 게이트 임계 변경 → 백테스트 재검증 필수. 미착수.

## [2026-06-09] Regime 모듈의 signal 영향 범위 확정
- **Delta:** regime는 signal pass/fail을 **직접 게이팅하지 않음**. 4기능: (A) 진입 마스킹 — `regime_signal_gating_enabled=False`(전역 off), `mean_reversion_regime_entry_gating_enabled=True`로 mean_reversion archetype만 발효. (B) `regime_code` 라벨 부착 → allocation 조건화 입력. (C) archetype+regime별 exit 정책. (D) 진단 `regime_pass`는 promotion_level=variant→강제 True(비게이팅).
- **Rationale:** RECOMMENDED 추세/모멘텀 3종은 regime 마스킹 무영향 확인. `[[project_regime_alpha_conditioning_disproved]]`와 정합.
- **Edge Cases/Trade-offs:** REGIME_SCORECARD C4=1.0/10(OOS 불안정)은 (B) allocation 조건화 입력의 신뢰도 경고.

## [2026-06-10] FDR 및 SPA 집합검정 Multiplicity Gate 도입
- **Delta:** `CandidateStrategyConfig`에 FDR/SPA 설정 변수 추가. `_summarize_recommendation_variants()`에서 Newey-West 단측 p-value 산출. `compute_rule_diagnostics()`에서 OOS edge circular-block bootstrap(SPA) 연동하여 최종 promote AND-결합 필터링 구현.
- **Rationale:** G1~G10 신규 패밀리 확장에 따른 alpha 모집단 팽창 시 overfit 생존 후보의 급증을 다중 비교 multiplicity control인 FDR과 SPA 집합검정으로 제어하여 라이브 환경의 오버핏 리스크 최소화.
- **Edge Cases/Trade-offs:** default off 설정으로 하방 호환성 및 기존 regression 안정성 확보. SPA의 경우 이벤트 부재 시 fail-closed(차단)로 강하게 통제함.
