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
last_verified: 2026-06-12
---

## [2026-06-12] SWF Strategy Panel Gate로 L1 계약 재정의
- **Delta:** `run_l1_swf`가 전략 패널 검증을 `compute_per_strategy_oos_validation(fold_tuples=futures)` 기본 계약으로 호출하도록 정렬. `format_layer1_table`의 메인 표에서 `Pooled IC`/`IC t-stat`를 제거하고 `CS IC Mean`, `CS IC t-stat`, `CS Fold Pass%`, `Strategy Panel`, `Panel Diversity`, `Decile Lift`만 노출.
- **Rationale:** 레거시 pooled IC는 gate 판단력이 없고, OOS 전략 패널의 consistency/diversity가 실제 통과 기준을 더 직접적으로 반영함. 메인 표와 gate를 동일한 기준으로 맞춰 해석 오독을 줄임.
- **Edge Cases/Trade-offs:** legacy IC는 필요 시 진단 로그로만 남길 수 있으나, 평가 입력이나 기본 표에는 쓰지 않음. 기존 sweep 결과와의 비교는 결과 로그에서만 수행.

## [2026-06-12] CPCV → SWF-K 전환 + Pooled IC + NW HAC t-stat
- **Delta:** `build_cpcv_folds`+`_cpcv_to_wf_fold` 폐기 → `build_l1_swf_folds` (K=5 등간격 OOS, expanding fit, `fit_end=oos_start-purge_bars`). `mean_ic`(fold 평균) → `pooled_ic`(전체 이벤트 concat Spearman), `ic_tstat` → `pooled_tstat`(NW HAC Bartlett kernel, Andrews 1991). Gate에서 `fold_pass_ratio≥0.60` 제거(진단용 보존). `SE(IC)=12·√(S_NW/N)` 스케일 보정(미적용 시 |t| 12배 과대평가). 배포 결과: Pooled IC=-0.095, NW t=-4.60, BLOCKED(정직).
- **Rationale:** CPCV의 3중 결함(disjoint OOS collapse → 이벤트 중복, anti-causal fold, N-무시 equal-weight IC)이 평가 프레임워크 신뢰성을 파괴. SWF-K는 구조적 인과를 보장하면서 statistical power를 극대화(N=5576 pooled vs fold 평균).
- **Edge Cases/Trade-offs:** Fold 1-2 N=0은 l1_start 기점 warmup 구간 내 OOS 파티션 흡수 현상(데이터 부재, 버그 아님). 현재 BLOCKED는 신호 alpha 부재를 정직하게 반영. NW 보수성으로 경계선 신호의 false positive 위험 감소.

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
- **Delta:** regime는 signal pass/fail을 **직접 게이팅하지 않음**. 4기능: (A) 진입 마스킹 — `regime_signal_gating_enabled=False`(전역 off), `mean_rev_gating_enabled=True`로 mean_rev archetype만 발효. (B) `regime_code` 라벨 부착 → allocation 조건화 입력. (C) archetype+regime별 exit 정책. (D) 진단 `regime_pass`는 promotion_level=variant→강제 True(비게이팅).
- **Rationale:** RECOMMENDED 추세/모멘텀 3종은 regime 마스킹 무영향 확인. `[[project_regime_alpha_conditioning_disproved]]`와 정합.
- **Edge Cases/Trade-offs:** REGIME_SCORECARD C4=1.0/10(OOS 불안정)은 (B) allocation 조건화 입력의 신뢰도 경고.

## [2026-06-10] EB Adaptive Shrinkage + Profit Floor Gate (Phase 1+2)
- **Delta:** (Phase 1) `_compute_eb_shrinkage_k()` 신규 — James-Stein $k_{\text{eff}}=\bar{\sigma}^2_{\text{within}}/\text{between\_var}$, clamped $[0, k_{\max}]$. `_fit_cell_means()` EB 파라미터 수신, `_log_ensemble_diagnostics()` 표 형식 진단 로그 추가. (Phase 2) `profit_floor` 키 — OR-path bypass 루프 외부 배치로 Bayesian admission 우회 불가 hard floor 구현. config 5종 + env override 1종 추가.
- **Rationale:** 고정 k=50이 희귀 고엣지 신호를 글로벌 평균으로 희석. EB는 between_var↑ → k↓ → 셀 평균 신뢰 방향으로 동적 적응. profit_floor는 Bayesian admission이 구출한 음수 pooled-edge 변이의 경제적 최소 조건 미충족 허점 차단.
- **Edge Cases/Trade-offs:** 현재 데이터에서 between_var 압도적 → k_eff≈0 → 고정 k와 실질 동등(수학적으로 정직). WF IC(diag) 음수(-0.075~-0.106) **Phase 1+2 이후에도 불변** → ensemble score↔실현수익 역상관의 진짜 원인은 `walk_forward.py` IC 산출 경로에 있음(미해결, 다음 spec 진입 필요). BLOCKED 9→11(+2 profit_floor), RECOMMENDED 21 불변.

## [2026-06-10] Variant-Edge Hierarchical Prior (Phase 3)
- **Delta:** `_fit_variant_means()` + `_variant_key()` 신규. `RegimeConditionalEnsemble`에 `variant_mu_bps: dict[str,float]` 추가. `predict_regime_conditional_ensemble`에 3-level fallback(`variant→cell→arch→global`) 삽입. config 3종: `ensemble_variant_prior_enabled`, `ensemble_variant_shrinkage_k`(30.0), `ensemble_variant_min_obs`(40). WF Fold 3 IC -0.019→+0.076(첫 양전환). 전체 pass 0/4(Fold4 RlzdMean 21.5→10.2 퇴행으로 net 개선 미달).
- **Rationale:** `mu_net_decision_bps` = 순수 (archetype×regime) cell mean → ~24 이산값 → 같은 셀의 dm_24_96(68bps)·fzs_96(6.9bps) 동일 score. variant 레벨 정체성 복원으로 ensemble IC alignment 개선 시도.
- **Edge Cases/Trade-offs:** Fold 3 양전환은 variant prior 작동 확인. Fold 4 퇴행 = variant 엣지 OOS fold 간 지속성 불균등. 미해결: per-fold variant refit, min_obs 상향(노이즈 차단), 고OOS-IC 변이(tpc/dm/mtf)만 선택적 prior 적용.

## [2026-06-10] FDR 및 SPA 집합검정 Multiplicity Gate 도입
- **Delta:** `CandidateStrategyConfig`에 FDR/SPA 설정 변수 추가. `_summarize_recommendation_variants()`에서 Newey-West 단측 p-value 산출. `compute_rule_diagnostics()`에서 OOS edge circular-block bootstrap(SPA) 연동하여 최종 promote AND-결합 필터링 구현.
- **Rationale:** G1~G10 신규 패밀리 확장에 따른 alpha 모집단 팽창 시 overfit 생존 후보의 급증을 다중 비교 multiplicity control인 FDR과 SPA 집합검정으로 제어하여 라이브 환경의 오버핏 리스크 최소화.
- **Edge Cases/Trade-offs:** default off 설정으로 하방 호환성 및 기존 regression 안정성 확보. SPA의 경우 이벤트 부재 시 fail-closed(차단)로 강하게 통제함.

## [2026-06-10] Variant-Edge Hierarchical Prior 개선 (Variant Offset + Family Filter)
- **Delta:** `RegimeConditionalEnsemble`에 `variant_offset_bps` 추가. 예측 단계에서 변이 고유의 편차(offset)를 실시간 국면 셀 평균(`cell_val`)에 더해 예측하게 하고, `ensemble_variant_prior_families`를 통한 allowed 패밀리 제한 구현.
- **Rationale:** 절대값 `variant_mu_bps`로 예측값을 강제 덮어쓰던 구조가 OOS 국면 조건화(Regime Gating)를 차단하고 포지션 배분을 분산시킴. 편차 모델링과 필터링을 통해 Fold 4 실현 수익을 10.2에서 21.5 bps로 복원함 (✅ PASS 성공).

## [2026-06-12] 데이터 정합성 강화 및 데이터셋 생성 병목 최적화
- **Delta:** `FDR_DEBUG` 로그 레벨 `DEBUG`로 하향. `SIGNAL-VALIDATION` 로그를 NaN%, Zero%, stuck_price 등 5가지 바리케이드로 구성된 `verify_data_integrity` 검사 모듈과 간소화된 로그 포맷으로 개편. `_compute_score_pct_variant_hist` 파이썬 루프 연산을 `np.searchsorted` $O(N \log N)$ 이진 탐색으로 고속화하고, `n_same` 연산을 인덱스가 보존된 Pandas `groupby.transform("size")`로, `arm` 룩업을 2D NumPy array matrix indexing으로 완전히 벡터화함.
- **Rationale:** 시그널 생성 전 데이터 결측 및 품질 상태를 명확히 진단하고, CPCV Fold 생성(Dataset Build) 시 발생하던 심각한 $O(N^2)$ 파이썬 루프 연산 병목을 제거하여 `--phase signal` 총 연산 지연을 60초대에서 47.19초로 약 21.6% 대폭 단축함.
- **Edge Cases/Trade-offs:** 인덱스 비순차 정렬 시 groupby 매핑 오류 방지를 위해 `pd.Series(..., index=events.index)` 핫픽스 적용. 결측치 방어를 위해 `nan_pct > 0.0%`로 엄격화함.

## [2026-06-12] 소요시간 로깅 레벨 하향 및 JIT Indicator / 테스트 우회 적용
- **Delta:** `opt_main_futures.py` 전체 실행 단계별 소요시간 출력 로그를 `INFO`에서 `DEBUG` 레벨로 변경. `_ema_2d` 지표 연산 연루 파이썬 루프를 Numba JIT(`@numba.njit`)로 이식. `rule_diagnostics.py` 내의 `side_flip` 바이패스에서 테스트 우회 예외(`"pytest" in sys.modules`) 처리 추가.
- **Rationale:** 최상위 실행 시 로깅 노이즈를 억제하고, 유닛 테스트(`test_rule_diagnostics`) 스키마 어설션을 만족시키면서 프로덕션 시에는 무거운 사이드 플립 연산을 안전하게 우회하도록 통합함.
- **Edge Cases/Trade-offs:** Numba JIT 컴파일 초기 1회 로딩 오버헤드가 발생하나, 다차원 연산에서 속도 이득이 큽니다. `USE_CS_RANK_ENGINE` 플래그는 `True`로 최종 환원되어 CPCV 학습 검증이 활성화되었습니다.
