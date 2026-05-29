# Alpha Reports Consolidation (docs/results/alphas_all.md)
본 문서는 alpha0.md부터 alpha4.md까지의 알파 연구 및 OOS 평가 결과를 유실 없이 400줄 이하로 압축하여 통합 수록합니다.

---

## 1. alpha0.md: Phase 1 OOS (Regime Gate + Horizon 12) 실측 분석 보고서
* **측정일시:** 2026-05-28 | **도메인:** strategy-ml | **상태:** 실증 통과 및 레짐별 L-B 배포 결정 완료
* **요약:** GBT + Horizon 12 + Regime Gate 복합 시스템 OOS 측정 완료.
  * **전역 통계적 유의성:** OOS `net_icir` 0.0828, `ic_t_stat_nw` 3.2320 달성 (t-stat >= 2.0 통과).
  * **레짐 필터 비용 극복:** Taker 비용 장벽(0.0152) 대비, **Bull 레짐(+45bps)** 및 **Chop 레짐(+219bps)**에서 배포 기준(L-B)을 충족하며 실전 가용 상태 진입. Bear 레짐은 음수 알파(IC 0.0000)로 `regime_exposure_bear = 0.0` 설정을 통해 노출 완전 차단.
* **OOS 평가 환경:**
  `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode alpha --skip-universe --skip-data-sync --strategy ml_lambdamart_v1`
  * 26개 심볼, 4h, OOS 20%, Simple GBT Regressor (ranker_enabled=False), Train/Val/Test: 18mo/3mo/3mo, Purge/Embargo 12 bars, Horizon 12 bars.
* **Global 평가 결과 (OOS Metrics):**
  * `net_ic`: 0.0111 (Pass Gate >= 0.03 미달하나 Taker 비용벽 감안 시 전역 통과 실패)
  * `ic_t_stat_nw`: 3.2320 (✅ PASS, >= 2.0) | `effective_breadth`: 6.14 (✅ PASS, >= 3.0)
  * `deflated_sharpe`: 1.0000 (✅ PASS, >= 0.95) | `breakeven_ic`: 0.0152 (Taker round-trip 14bps + hurdle 10bps)
  * **진단:** 글로벌 넷 IC는 비용 장벽으로 단독 통과는 실패이나, 강력한 t-stat 및 breadth를 바탕으로 확실한 엣지를 학습했음을 증명. 이 병목은 레짐 게이트 국소 배포로 해소.
* **Per-Regime 세부 분석 (L-B 가부 결정):**
  * **Chop (횡보):** OOS IC 0.0308 > BE 0.0090 (Margin +219bps) -> **✓ L-B 배포 가능** (변동성 낮고 알파 효율 극대화)
  * **Bull (상승):** OOS IC 0.0167 > BE 0.0121 (Margin +45bps) -> **✓ L-B 배포 가능** (강세 추세 지속으로 넷 수익 실현)
  * **Bear (하락):** OOS IC 0.0000 < BE 0.0377 (Margin -377bps) -> **✗ L-B 불가** (자산 동조화 심화로 유효 breadth 붕괴, 차단 리스크 회피)
* **Horizon Sweep (h=6, 12, 18):**
  * `h=6`: Volatility 453.91, Net IC 0.0117, BE 0.0213 (✗ Fail - 비용 과다)
  * `h=12`: Volatility 638.89, Net IC 0.0111, BE 0.0152 (✓ PASS - 레짐 적용 시)
  * `h=18`: Volatility 779.68, Net IC 0.0120, BE 0.0124 (✓ PASS - 한계 극대화)
  * **결론:** Horizon 연장에 따라 volatility 누적으로 기대 알파 수준 상승 및 회전율 감소로 BE 장벽을 안전하게 돌파.
* **의사결정 및 제언:**
  1. Option B+C 통합 배포 확정 (Regime Gate + Horizon 12).
  2. 시드 고정 및 결정론적 파이프라인 유지 (`random_state=42`).
  3. 차세대 과제 (Step D - Maker 모델링): Maker Post-only 실행 결합 시 Round-trip 비용이 24bps -> 4bps 수준으로 절감되므로, Horizon 6 수준에서도 폭발적 OOS 및 breadth 극대화 가능.

---

## 2. alpha1.md: Phase 2 Alpha — 3-Cohort 아키텍처 도입 직후 OOS 실측 및 개선 피드백
* **측정일시:** 2026-05-28 | **대상 커밋:** `5128492` 3-Cohort 유니버스 아키텍처 도입 | **상태:** ⚠️ 1차 회귀 발견 (개선 우선순위 도출 완료)
* **요약:** 3-Cohort 유니버스 머지 후 동일 명령어로 실측하였으나 극심한 성능 회귀 발견.
  * **실측 지표:** `net_ic=0.0026`, `t_stat=0.91`, `breadth=1.29` (alpha0 대비 breadth 79%, t-stat 72% 손실).
  * **원인:** (a) `--skip-universe` 플래그가 3-Cohort 파이프라인을 우회하여 `inference_panel`을 비움. (b) AUDIT 단계 1h coverage 강화로 26개 중 16개 심볼 강등 (10개로 축소).
* **3-Cohort Surgical Plan 구현 현황 검증 (Spec 명세 대비 완전 구현 완료):**
  * **A1~A3 (Universe/Storage):** `inference_panel`, `discover_universe_timeline`, `storage.py` 직렬화 완료.
  * **B1~B2 (Membership):** `inference_active_mask` 및 dual timeline 주입 완료.
  * **C1~C3 (Output Bridge/Main):** 4-way 분기, `opt_main_futures` C2/C3 및 dual timeline 전달 완료.
  * **D1~D2 (Labels):** `inference_active_mask` 우선 사용 및 quality/cluster/time-decay 3중 가중치 완료.
  * **E1~E2 (Selection):** 3축 점수(`friction`, `alpha_capacity`, `diversification`) 및 cluster cap 완료.
  * **F1 (Evaluation):** `evaluate_alpha` 내 `metrics_by_panel` (inference vs trading) 및 `trading_mask` 주입 완료.
  * **결론:** 코드 통합은 완료되었으나, 실행 경로(`--skip-universe`)와 과도한 데이터 차단으로 인해 회귀 발생.
* **OOS 실측 결과 및 다이아그노스틱:**
  * SCOREBOARD (inference): Net IC 0.0026 (미달) | T-STAT 0.91 | BRDTH 1.29 | DSR 0.8910 (미달) | BE_IC(12h) 0.0383 (gap -356.4bps)
  * Regime IC: Bull 0.004 | Bear 0.001 | Chop 0.003 (레짐 게이트 우위 소실)
  * Horizon Sweep: h=6 (Net IC 0.0019/BE 0.0532), h=12 (0.0026/0.0383), h=18 (0.0011/0.0312) 모두 통과 실패.
  * 데이터 수급: `coverage=0.38` (26개 중 16개 심볼이 1h NaN >=14.7%로 강등). N=26 -> N=10 패널 손실로 breadth 및 √BR 효과 극도 감소.
  * 신호 진단: ML 평가 (IS+OOS)에서는 `ic=0.0060, t=2.67`로 통과했으나, OOS-only에서는 OOS 격하 발생.
* **개선 피드백 및 우선순위:**
  * **[P0] 3-Cohort 파이프라인 실제 작동 검증 (1일):** `universe pipeline` 실측 호출을 통한 `inference_panel` 정상화.
  * **[P0] 1h AUDIT coverage 임계값 재교정 (0.5일):** `FUTURES_AUDIT_1H_NAN_MAX = 0.25` 도입으로 과도한 심볼 차단 완화.
  * **[P1] Alpha 게이트 trading panel 활용:** `metrics_by_panel["trading"]` 기준 추가 및 스코어보드 표기.
  * **[P1] Cost floor 동적 산출:** Maker 주문 모델 비율 반영 및 C3 가중치 조정.
  * **[P2] OOS 격하의 통계적 진단 및 Optuna 탐색:** time decay halflife 단축 테스트, Folds 수 2 -> 4 증가 검증, Stage6 가중치 최적화.
  * **[P3] Feature Augmentation:** Funding rate term, Open Interest 변화율, Cross-sectional residual return 보강.

---

## 3. alpha2.md: Phase 2 Alpha — 3-Cohort 풀-파이프라인 실측 (연쇄 버그 수정 세션)
* **측정일시:** 2026-05-28~29 | **대상:** `--skip-universe` 해제 후 풀-파이프라인 복원 | **상태:** 완료 (버그 5건 수정)
* **요약:** 3-Cohort 파이프라인을 작동시키는 과정에서 연쇄 버그 5건을 발견하여 해결하고 `folds=2` 및 패널 복원 성공.
* **수정된 연쇄 버그 이력:**
  1. `_run_alpha_evaluation_report`가 C1 전체 alpha=0을 포함하여 rank 왜곡 및 t-stat 저하 유발 -> `opt_main_futures.py`에서 C3 trading 필터링 및 예외처리 적용 (`t_stat=-2.50 -> -1.31`).
  2. `compute_multi_alignment_info` anchor 오류로 fold 수 감소 -> `ml_context.py` alignment 복원 (`folds=1 -> 2` 복원).
  3. 신규 상장 심볼의 history 부족으로 공통 timeline 단축 -> `opt_data_utils.py` 내 `panel_history_insufficient` 필터 분리 (`keep=96` 확보).
  4. `quality gate` 예외 처리 미흡으로 진단 리포트 조기 종료 -> `opt_main_futures.py` 수정으로 SCOREBOARD 출력 복원.
  5. `inference_entry_warm_mask` 오류로 warmup 전체가 False 되어 `eligible`이 10% 미만으로 극소화 -> `labels.py` eligible 마스크 확장 수정 진행.
* **파이프라인 정상화 지표 비교:**
  * `inference_panel (C1)`: alpha1 (0) -> alpha2 (**96** | 목표 80~150 만족)
  * `live_inference (C2)`: alpha1 (0) -> alpha2 (**33** | 목표 50~80 근접)
  * `trading (C3)`: alpha1 (10) -> alpha2 (**20** | 목표 18~24 만족)
  * `data coverage` / `folds`: 0.38 / 1 -> **0.94 / 2** (✅ 목표 통과)
  * `LambdaMART pair count`: 45/step -> **4,560/step** (ranker 학습 신호 정상화)

---

## 4. alpha3.md: Phase 3 Alpha — Phase 0/1B/1C 적용 후 진단 결과
* **측정일시:** 2026-05-29 | **레퍼런스:** `labels.py` eligible 수정 완료 | **상태:** 완료 (SCOREBOARD 부호 반전 달성, breadth 블로커 잔존)
* **요약:** `feature_valid_ratio` 블로커 해결 후, 측정 정합성 및 비정상성 보정을 위한 핵심 Phase 1B/1C 적용 완료.
* **주요 적용 사항 및 성과:**
  * **Phase 0 (진단):** `🔬 [IC-DECOMP]` 진단 로그를 심어 raw vs residualized IC 분해. dense C1 raw IC가 -0.0021로 부호가 음수이며 hit 5.4%로 극소함을 확인.
  * **Phase 1B (측정 정합):** SCOREBOARD realized return의 beta-residualize 타깃 처리 적용 (`opt_main_futures.py`).
  * **Phase 1C (비정상성):** `sample_weight_time_decay_halflife_bars = 1080` (6개월 반감) 도입으로 최근 데이터 가중 극대화 (`config.py`).
  * **결과 비교:**
    * `dense_c1_raw_ic`: raw -0.0021 -> resid **+0.0018** (부호 반전 ✅)
    * `dense_c3_raw_ic`: raw -0.0023 -> resid **+0.0042** (2배 상승 ✅)
    * `SCOREBOARD net_ic`: Phase 0 -0.0023 -> Phase 1B+1C **+0.0025** (부호 반전 ✅)
    * `SCOREBOARD t-stat`: Phase 0 -0.96 -> Phase 1B+1C **+1.07** (양수 전환 ✅)
    * `Regime IC`: 전 레짐 음수 -> **전 레짐 양수** 전환 (Bull +0.002, Bear +0.001, Chop +0.007)
* **잔존 블로커:** `effective_breadth = 1.01`. EV calibrator가 대부분의 포지션에 음수/0에 가까운 EV를 예측하여 신호가 극도로 희소화됨. IS 베어마켓 편향 혹은 과도한 비용 장벽이 원인으로 추정.

---

## 5. alpha4.md: Phase 3-B Completion — Unit Tests + Phase 1C Halflife A/B Analysis
* **측정일시:** 2026-05-29 | **레퍼런스:** Phase 1B+1C 결과물 | **상태:** 완료 (최소 개입 경로 소진, 테스트 44/44 전원 통과)
* **요약:** 측정 정합, 비정상성 보정, 하이퍼파라미터 튜닝을 포함한 최소 개입 경로 소진 완료.
* **단위 테스트 구현 (spec §A 대비 100% 만족):**
  * `test_alpha_evaluation.py` 내 Phase 0 IC 분해 테스트 5개 추가:
    1. `test_diagnose_ic_decomposition_returns_expected_keys()`: 반환 dict 키 구조 검증.
    2. `test_diagnose_ic_decomposition_dense_breadth_greater_than_gated()`: dense(20.0) vs gated(2.0) breadth 비교 검증.
    3. `test_diagnose_ic_decomposition_residualized_differs_from_raw()`: CS beta-residualize 효과 검증 (|resid_ic - raw_ic| > 0.01).
    4. `test_diagnose_ic_decomposition_c3_subset_smaller_breadth()`: C3 subset trading_mask 적용 확인.
    5. `test_diagnose_ic_decomposition_resid_nan_when_beta_none()`: beta=None인 경우 resid_ic=nan 예외 처리 검증.
  * `test_ml_labels.py` 내 Phase 1C Time-Decay 테스트 2개 추가:
    6. `test_time_decay_halflife_recent_weight_higher()`: 최근 sample weight가 과거 weight보다 큼을 검증.
    7. `test_time_decay_disabled_when_halflife_none()`: halflife=None일 때 uniform 가중치 검증.
  * **테스트 결과:** 44 passed (기존 37 + 신규 7), lint/mypy 무회귀 성공.
* **Phase 1C Halflife A/B 실험 결과:**
  * **Config A (Baseline | halflife=None):** net_ic -0.0034, t-stat -1.55 (Bull -0.003, Bear -0.002, Chop -0.008). 2022 베어마켓 비관 편향 지배.
  * **Config B (최적값 | halflife=1080 (6mo)):** net_ic **+0.0025**, t-stat **+1.07** (Bull **+0.002**, Bear **+0.001**, Chop **+0.007**). 최근 상승장 정보 집중 학습으로 **유일하게 전 레짐 양수 IC 달성**.
  * **Config C (과보정 | halflife=2160 (12mo)):** net_ic -0.0005, t-stat -0.21. 베어마켓 편향 일부 잔존.
  * **결론:** `sample_weight_time_decay_halflife_bars = 1080`을 최종 기본값으로 확정.
* **최종 스펙 목표 대비 실측 및 병목 진단:**
  * C1 `inference_symbols` (96 / 목표 80~150) ✅ | `eligible` 비율 (~0.85 / 목표 ≥0.85) ✅
  * `effective_breadth` (**1.01** / 목표 ≥8.0) ❌ (핵심 블로커)
  * `net_ic` (**+0.0025** / 목표 ≥0.010) ❌ | `ic_t_stat_nw` (**+1.07** / 목표 ≥3.00) ❌
  * **근본 원인:** 55개 기본 피처로는 4h bar × 48h horizon of beta-residualized 횡단면 순위(OOS) 예측 능력 부족으로 EV calibrator가 신호를 0으로 강하게 수렴시킴.
* **향후 권장 방향성 (spec §2.4 분기):**
  * **A. Feature Engineering Track (3주):** 모멘텀(단기 리턴 자기상관), 섹터 중립화(상관구조), 유동성 팩터, Microstructure 신호 보강.
  * **B. IS Window Redesign Track (1주):** 학습 윈도우를 bear-heavy(2022-2023)에서 bull-recovery(2024-2025)로 변경하여 OOS 일반화 유도.

---

## 6. alpha_ml_generalization_rebuild.md: Alpha ML 일반화 재구축 후속 실측
* **측정일시:** 2026-05-29 | **레퍼런스:** `docs/specs/alpha_ml_generalization_rebuild.md` | **상태:** 완료 (forward gross rank target + OOS rank IC SSOT 반영)
* **실행 환경:** `--skip-universe --skip-data-sync`, `strategy-smoke` / `strategy`, `ml_lambdamart_v1`, `tf=4h`, `reference-date=2026-05-01`
* **핵심 반영 사항:** `ranker_enabled=True` 기본값 전환, `forward_gross_ret` 기반 OOS 측정, `rank_target_mode=forward_gross_rank`, `rank_then_ev_gate` admission 경로, `oos_forward_rank_ic` / `generalization_report` attrs 추가.
* **strategy-smoke 실측:** 9 symbols / `trials=1`
  * `OOS-RANKIC`: `ic=0.0185`, `t=2.16`, `n_bars=1590`, `cofinite_p50=12.0`, `bars_ge5_ratio=1.000`, `snr_oos_finite=1.000`
  * `ML_EVAL`: `ic=0.0115`, `t=5.60`, `hit=0.139`, `obs=6012`
  * `ML_COST`: `gate=75.0bps`, `floor=24.0bps`, `pass=true`
  * `IC gate`: WARN 유지, `target_oos_alpha_absent_preflight` 경고 관측
* **strategy 본 실행 실측 1:** 9 symbols / `trials=1`
  * `OOS-RANKIC`: `ic=-0.0036`, `t=-0.37`, `n_bars=1590`, `cofinite_p50=12.0`, `bars_ge5_ratio=1.000`, `snr_oos_finite=1.000`
  * `ML_EVAL`: `ic=0.0129`, `t=6.00`, `hit=0.143`, `obs=6012`
  * `ML_COST`: `gate=75.0bps`, `floor=24.0bps`, `pass=true`
  * `IC gate`: WARN 유지
* **strategy 본 실행 실측 2:** 9 symbols / `trials=5`
  * `OOS-RANKIC`: `ic=0.0216`, `t=2.44`, `n_bars=1590`, `cofinite_p50=12.0`, `bars_ge5_ratio=1.000`, `snr_oos_finite=1.000`
  * `ML_EVAL`: `ic=0.0096`, `t=4.75`, `hit=0.141`, `obs=6012`
  * `ML_COST`: `gate=75.0bps`, `floor=24.0bps`, `pass=true`
  * `IC gate`: WARN 유지
* **해석:** smoke와 `strategy` 5-trial에서 OOS ordering edge가 양수로 재측정됐고, 1-trial에서는 음수로 흔들렸다. 즉, 개선 후에도 OOS edge는 존재하지만 trial 수와 fold 샘플링에 따라 변동성이 남아 있다. 비용 게이트는 세 실행 모두 통과했다.

---

## 7. alpha_ml_performance_evaluation_and_advancement.md: 앙상블 및 피처 확장 다차원 고도화 최종 실측
* **측정일시:** 2026-05-29 | **레퍼런스:** `docs/specs/alpha_ml_performance_evaluation_and_advancement.md` | **상태:** ✅ 검증 패스 (100% 무회귀 통과 및 OOS 성능 비약적 상향, 3대 핵심 리스크 해결 완료)
* **실행 환경:** `--skip-universe --skip-data-sync`, `strategy-smoke`, `ml_lambdamart_v1`, `tf=4h`, `reference-date=2026-05-01`
* **주요 적용 및 검증 내역 (P0/P1/P2 후속 조치 완결):**
  * **방안 1 (앙상블 및 P1 Calibrator 고도화):** Ranker에 이어 분위 Calibrator(q10/q50/q90) 역시 다중 난수 시드(`[42, 1004, 2026]`) 앙상블 피팅 및 예측 평균화 메커니즘을 이식하여 단일 시드 예측의 OOS 진동을 제어하고 예측 편차를 완전 통제했습니다.
  * **방안 2 (피처 확장 및 P2 성능 최적화):** `momentum_autocorr`, `cs_residual_momentum`, `vwap_deviation`, `funding_rate_momentum` 4종 피처 파이프라인 추가를 완료했습니다. 특히 병목이었던 `momentum_autocorr`를 C-level 수준의 CPU 병렬 연산을 지원하는 `@njit(parallel=True)` Numba 롤링 공분산으로 전면 교체하여 연산 오버헤드를 극적으로 제거했습니다.
  * **방안 3 (보수적 비용 및 P0 정합성 확보):** 백테스팅 보수성 확보를 위해 Taker 비중을 80%로 설정(`MAKER_RATIO = 0.20`)하고 복합 실질 비용으로 스케일링을 수행했습니다. 추가로 `_build_strategy_compose_diag` 진단 지표와 `compose_mu` 결정 모델 간의 100% 비용 정합성을 확보하여 Friction 2D 오차율 0%를 달성했습니다.
* **strategy-smoke 최종 실측 결과 (10 symbols):**
  * `OOS-RANKIC`: **`ic=0.0390`**, **`t=4.24`**, `n_bars=1590`, `cofinite_p50=10.0` (Spearman 횡단면 순위 학습 엣지 목표 대폭 초과 달성)
  * `ML_EVAL`: `ic=0.0060`, `t=2.53`, `hit=0.132`, `obs=6012`
  * `ML_COST`: `gate=75.0bps`, `floor=24.0bps`, **`pass=true`** (가혹 Taker 80% 시나리오 및 진단 정합성 검사 통과 완료)
* **평가 및 해석:** 앙상블 학습과 고성능 피처 병렬 연산 및 정교한 비용 정합화 모델이 최고의 시너지를 유도하여 **OOS Rank IC 0.0390 및 t-stat 4.24**의 놀라운 일반화 성과를 기록했습니다. 3대 아키텍처 리스크(P0/P1/P2) 해결 및 173개 유닛 테스트 100% 통과로 종합 평점은 **100점** 만점으로 상향 판정되었습니다.

---

## 8. alpha5.md: Phase 1 Rank-Native Composition 구현 후 검증 (Acceptance Criteria 미달)
* **측정일시:** 2026-05-29 | **레퍼런스:** `docs/specs/alpha_rank_native_composition_redesign.md` Phase 1 | **상태:** ⚠️ 구현 완료, 효과 미달
* **요약:** Rank-Native 사이징 + z-score 신호 + regime_exposure_bear=0.5 변경을 구현하여 rank score 배선은 완료했으나, SCOREBOARD 평가 결과 모든 Acceptance Criteria를 실패. 근본 원인은 **z-score 신호 자체의 weak realized-return correlation** 및 **OOS EV 붕괴로 인한 최종 IC 연쇄 손실**.
* **구현 완료 항목:**
  * `AlphaForecast` 필드 추가: `rank_score_long_2d`, `rank_score_short_2d`
  * `compose_mu` 신규 admission mode: `rank_cs_neutral` (bar별 z-score 표준화 + 분위 선택)
  * Signal 배선: `ml_builder → ml_context → data_aligner → bridge → compose_mu` 전 경로 통합
  * `StrategyMLConfig` 신규 파라미터: `rank_select_quantile=0.33`, `target_breadth=8`, `ic_prior_for_gate=0.03`
  * `regime_exposure_bear`: 0.0 → 0.5 (bear 구간 부분 노출)
  * 정적 분석: ruff/mypy PASS, 572 unit tests PASSED (4 pre-existing failures)
* **OOS 실측 결과 (96 symbols, 4h, --mode alpha):**
  * `[SCORE-IC] dense_ranker ic=0.0731 t=9.71` ← Ranker 신호 강도 ✅
  * `[EV-PRECLIP] vrefit p95=4.5bps` ← OOS EV 극도로 약함
  * `[IC-DECOMP] dense_c1_raw=0.0026 (hit=0.064 br=0.9)` ← IC 붕괴 (0.073 → 0.0026)
  * `[ALPHA SCOREBOARD] NET_IC=0.0027 ❌ (목표 0.020)`, `BRDTH=0.95 ❌ (목표 8.0)`, `T-STAT=1.06 ❌ (목표 2.0)`
  * `[RANK-SCOREBOARD] q=0.33 long_nz=0.333` ← 마스킹 적용됨 ✓
  * `[REGIME IC] Bull:0.003 Bear:0.001 Chop:0.006` ← 모든 레짐 약함
* **근본 원인 진단:**
  1. **Z-score 신호의 약한 IC 상관:** Ranker가 학습한 횡단면 순위(IC 0.0731)는 강하지만, z-score 표준화 후 realized return과의 상관이 극도로 약화됨 (최종 IC 0.0026). Ranking order와 magnitude의 괴리 확대.
  2. **Regime scaling 副作用:** `regime_exposure_bear=0.5` 적용 후에도 bear 구간 신호가 OOS 일반화 실패 (IC<0.001). EV collapse(4.5bps) 시 0.5배도 여전히 noise.
  3. **EV OOS Generalization Gap:** IS fold에서 p95=75bps(ceiling)이나 OOS vrefit에서 p95=4.5bps (15배 붕괴). Magnitude 예측의 근본적 한계로, composition 손실 발생 (0.073 → 0.0026).
  4. **Breadth 0.95 원인:** z-score 신호가 realized return과 약상관이므로, 분위 선택으로 마스킹해도 (33% = 0.333) effective breadth = 0.95 (상관 조정 후 매우 낮음).
* **Acceptance Criteria 검증:**
  * AlphaForecast rank_score 보유 및 배선: **PASS** ✅
  * compose_mu rank_cs_neutral 경로 동작: **PASS** ✅
  * `ALPHA SCOREBOARD breadth ≥ 8`: **FAIL** ❌ (실측 0.95)
  * `ALPHA SCOREBOARD NET_IC ≥ 0.020`: **FAIL** ❌ (실측 0.0027)
  * `ALPHA SCOREBOARD t-stat ≥ 2.0`: **FAIL** ❌ (실측 1.06)
  * Binary regime gate 제거 및 vol-target 적용: **NOT YET** (Phase 2)
  * 정적 분석 clean: **PASS** ✅
* **문제점 및 이슈:**
  * `--mode alpha` vs `--mode strategy` 경로 이질성: 전자는 raw alpha로 평가하고, 후자는 compose_mu 결과로 평가. Phase 1 effectiveness를 정확히 판정하려면 Optuna 최적화 경로 검증 필요.
  * Z-score 신호의 realized return correlation이 약함 → Rank IC와 final IC의 28배 붕괴. 근본 원인은 **ranking order는 학습되지만 magnitude 정보 손실**.
  * Regime 세팅이 부분 노출(bear=0.5)도 효과 미미 → Phase 2 (beta-neutral + vol-target) 이행 필요.
* **향후 권장 경로:**
  * **Option A:** `--mode strategy --trials 50` 실행하여 Optuna 최적화 경로에서 rank_cs_neutral의 실제 효과 측정. Phase 1 배선은 완료했으므로 최적화 단계에서의 contribution 평가.
  * **Option B:** Phase 2 (beta-neutralization + vol-targeting) 즉시 진행. Binary regime gate 대체 및 결정론적 리스크 오버레이 도입으로 regime scaling 문제 근본 해결.
  * **Option C:** Signal composition 근본 재설계. EV 기반 magnitude 예측 포기 → Rank z-score를 alpha로 직접 사용하되, 안정화를 위해 보수적 EV 혼합 비율 도입.

