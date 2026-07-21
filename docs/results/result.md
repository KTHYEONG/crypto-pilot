# 결과 기록 (docs/results/result.md)

이 문서는 세션별 실측 결과를 최신순으로 기록한다. 각 항목은 "무엇을 했는지 → 실측 수치 → 판정 → 다음 우선순위" 순서를 따른다.

---

# L2 RC-3 Regime-Scoped Fold Override + Phase B 배선버그 완전 수정 — 2026-07-21 (최신)

`docs/specs/l2-regime-scoped-fold-override.md` 구현(`/check` PASS, Cov 44%) — 바로 아래 섹션("Phase B 배선 완성")에서 배선을 고쳤는데도 `recent_fold` 위반이 악화됐던 문제의 진짜 원인을 찾아 고쳤다.

## 무엇을 했나 (쉽게 설명)

1. **더 깊은 배선 버그 추가 발견·수정**: 아래 섹션에서 "배선을 고쳤다"고 했던 것도 사실 절반만 고친 것이었다. 실제 배분 로직을 미리 계산하는 코드(`active_pipeline.py`)가 flag 값을 아예 필드가 없는 엉뚱한 설정 객체에서 읽고 있어서, **항상 무조건 꺼진 상태로 계산**되고 있었다 — 그런데 실제 적용 코드는 다른 경로로 켜진 값을 읽어서 "4개짜리 키"로 조회하려 하니, 미리 계산된 "3개짜리 키" 테이블에서 항상 못 찾고 → **정책 전체가 통째로 무력화(무조건 통과)**되는 이중 버그였다. 이번에 두 지점이 같은 값을 보도록 하나로 통일했다.
2. **진짜 원인 특정**: 배선을 완전히 고치고 나서 재측정해도 `recent_fold`(최근 구간 수익성 검사) 위반이 여전히 늘어나는 걸 확인 — 정밀 로그를 붙여 추적한 결과, `RC-3`이라는 안전장치가 "폴드(구간) 전체 평균이 나쁘면 그 폴드 안의 **모든** 항목을 묻지도 따지지도 않고 무효화"하는 방식이었다. 하락장 숏처럼 실제로 확인된 좋은 신호가, 같은 구간에 섞인 약한 신호들 때문에 통째로 같이 버려지고 있었던 것.
3. **수정**: 이 판정을 "폴드 전체"가 아니라 "장세(regime)별로 따로" 하도록 바꿨다. 나쁜 장세의 신호만 걸러지고, 좋은 장세의 신호는 살아남도록. 기본값은 꺼짐(기존과 100% 동일 동작 보장), 켜면 새 방식 적용.

`/check` 전체 통과(lint+mypy+테스트, Cov 44%).

## 실측 결과 (`--phase l3 --seed 42`, 내부 42/43/44, 두 flag 모두 ON)

| Seed | recent_fold 위반 (기준값 → 버그판 → 이번 수정판) |
|---|---|
| 42 | 11 → 22 → **12** (거의 기준값 복귀) |
| 43 | 21 → 33 → **21** (기준값과 정확히 일치) |
| 44 | 10 → 25 → **14** (기준값에 근접) |

진단 로그로도 확인: 폴드4에서 "묻지도 따지지도 않고 통과" 처리된 비율이 36/36(100%) → 24/36(67%)로 감소, 실제 데이터 기반 결정이 다시 살아남.

## 판정

- **여전히 champion(배포 가능한 후보)은 안 나온다** — `crisis_mdd`/`crisis_cagr`(위기장 위험 예산)가 여전히 최다 차단 사유. 이건 의도적으로 안 건드린 영역(위험 예산은 고정 유지).
- 폴드 1·2번은 이번 수정 후에도 여전히 전부 무효화됐는데, 이건 버그가 아니라 그 구간엔 정말로 장세별로 봐도 쓸 만한 신호가 없었다는 뜻 — 오히려 더 정확해진 판정.
- **재발 방지 교훈(두 번째)**: config flag 하나를 추가할 때 그 값을 읽는 지점이 여러 곳이면, 반드시 "같은 값을 보고 있는지" 확인하는 테스트가 필요하다 — 이번에도 "flag가 죽어있다"는 걸로 끝나지 않고, "flag가 서로 다른 값을 보고 있어서 조용히 망가진다"는 한 단계 더 미묘한 버그였다.

---

# Phase B 배선 완성 + A/B 재실측 — 2026-07-21

## 무엇을 했나 (쉽게 설명)

바로 아래 섹션에서 "Phase B(side-split 버킷) flag를 켜도 아무 효과가 없다"는 배선 결함을 발견했었다. 원인을 더 파보니, 문제가 있던 곳은 처음에 지목했던 지점(`filter_sleeves_by_bucket` 호출부) 하나가 아니라, **실제 운영에 쓰이는 기본 경로 자체(`policy_mode="soft"`)에는 side 배선이 아예 존재하지 않았다** — "side를 반영하는 코드 조각"만 준비되어 있었을 뿐, 실제로 롱/숏을 구분해서 봐야 하는 지점 어디에도 연결이 안 돼 있었다.

이번에 다음을 실제로 연결했다:
1. regime별 실현 손익을 계산하는 함수가 이제 "롱으로 들어간 트레이드"와 "숏으로 들어간 트레이드"를 따로 집계한다(기존엔 섞어서 하나의 숫자로 뭉갰다).
2. 실제 배분 결정을 내리는 핵심 함수(`apply_regime_cell_policy`)가 각 심볼의 실제 방향(롱/숏)을 그때그때 판별해서, 그 방향에 맞는 규칙을 찾아 쓰도록 고쳤다 — **이게 진짜 빠져 있던 연결고리였다.**
3. **재발 방지**: "flag를 켜고 끄면 실제로 결과가 달라지는지" 확인하는 자동 테스트 3개를 새로 추가했다. 이런 테스트가 없어서 지난번 미배선을 놓쳤던 것이므로, 앞으로 같은 실수가 반복되지 않도록 못을 박았다.
4. 부수적으로 발견한 버그(Phase A CAGR 게이트 계산에서 타입 캐스팅 누락) 1건도 같이 고쳤다.

`/check` 전체 통과(lint+mypy+테스트, Cov 43%).

## A/B 재실측 — 이번엔 실제로 다른 결과가 나옴

| Seed | flag OFF(꺼짐, 기존과 동일) | flag ON(side-split 적용) |
|---|---|---|
| 42 | crisis_mdd=110, cagr=108, crisis_cagr=107, recent_fold=11 | crisis_mdd=113, cagr=108, crisis_cagr=106, **recent_fold=22** |
| 43 | crisis_mdd=115, crisis_cagr=111, cagr=108, recent_fold=21 | crisis_cagr=117, crisis_mdd=116, cagr=75, **recent_fold=33** |
| 44 | crisis_mdd=115, crisis_cagr=108, cagr=96, recent_fold=10 | crisis_mdd=114, cagr=100, crisis_cagr=93, **recent_fold=25** |

(숫자 = 120개 후보 중 각 조건에 걸려 탈락한 개수. `admitted=False`, `joint_feasible=0/120`은 on/off 모두 3개 seed 전부 동일 — 아직 배포 가능한 champion은 못 찾음)

## 해석

- **좋은 소식**: 배선이 이제 진짜로 작동한다. flag를 켜고 끄는 것만으로 실제 결과가 달라지는 걸 처음으로 확인했다(지난 실측에서는 완전히 똑같았다).
- **아직 champion은 안 나온다**: 오히려 `recent_fold`(최근 구간에서 수익이 났는지 보는 검사)에 걸리는 개수가 거의 2배로 늘었다. 롱/숏을 나눠서 보면 각 그룹의 표본 수가 줄어들기 때문에, 통계적으로 더 불안정해지는 부작용이 있었던 것으로 보인다(예상 가능했던 트레이드오프).
- **Phase 0에서 확인한 "하락장 숏 엣지/급락 후 롱 엣지가 실재한다"는 사실 자체는 여전히 유효**하다. 다만 그 엣지를 실제로 champion 판정까지 끌고 가려면, 표본 부족 문제를 추가로 다뤄야 할 수도 있다는 새로운 과제가 생겼다.

## 다음 우선순위

- `recent_fold` 위반 증가의 정확한 원인(표본 분할로 인한 통계 불안정 vs 다른 요인)을 더 들여다볼지, 아니면 현재 상태(안전하게 차단됨)를 그대로 두고 L1 알파 자체를 늘리는 방향으로 갈지 사용자 판단 필요.

---

# Phase B(Regime×Side 버킷) A/B 실측 — side-split 배선 결함 발견 (해결됨, 기록용) — 2026-07-21

## 배경

직전 세션 Phase 0 실측(`alpha-funnel-regime-coverage.md`)에서 **bear regime short 셀(LCB +120~254bps)**과 **crash regime long-리버설 셀(LCB +36~78bps)**의 실재 엣지를 확인했다. Phase B는 이 엣지를 실제로 배분에 반영하기 위해 L2 버킷 키를 (regime, family, TF) 3-key에서 **(regime, family, TF, side) 4-key**로 확장하는 설계였고, `l2_regime_bucket_side_split_enabled` flag(기본 `False`)로 게이팅했다.

## A/B 실측 방법

동일 조건(`--phase l3 --trials 120 --timeframe 4h --seed 42`, 내부 42/43/44) 2회 실행:
- **A (flag off, 기본값)**: Phase A(CAGR 상대 게이트) + 확장된 bear/crisis gross cap 탐색공간만 적용.
- **B (flag on)**: `l2_regime_bucket_side_split_enabled=True`로 임시 override 후 동일 실행, 측정 직후 코드 원복.

## 실측 결과 — **A와 B가 완전히 동일**

| Seed | A (flag off) failures | B (flag on) failures |
|---|---|---|
| 42 | `{'crisis_mdd': 110, 'cagr': 108, 'crisis_cagr': 107, 'sharpe_uplift': 60, 'recency_holdout': 49, 'fold': 15, 'recent_fold': 11, 'mdd': 2}` | **바이트 단위 동일** |
| 43 | `{'crisis_mdd': 115, 'crisis_cagr': 111, 'cagr': 108, 'sharpe_uplift': 81, 'recency_holdout': 55, 'fold': 45, 'recent_fold': 21, 'mdd': 1}` | **바이트 단위 동일** |
| 44 | `{'crisis_mdd': 115, 'crisis_cagr': 108, 'cagr': 96, 'sharpe_uplift': 72, 'recency_holdout': 53, 'fold': 20, 'recent_fold': 10, 'mdd': 2}` | **바이트 단위 동일** |

Optuna trial별 `Best CAGR`/`Current` 진행 로그(8.82%→9.79%→10.32%→4.31%→-18.49%→27.71%→...)도 두 실행 전 구간 완전 일치 — `admitted=False`, `exit_code=1` 동일.

## 근본원인 (코드 추적으로 확정)

`l2_regime_bucket_side_split_enabled`는 **선언만 되어 있고 어디서도 읽히지 않는다**:
1. `l2_meta.py::filter_sleeves_by_bucket()` / `apply_bucket_conditional_weight()`는 `side: int = 0` 파라미터를 받아 `side!=0`일 때만 4-key 조회를 하도록 구현은 돼 있으나(순수함수 자체는 정상),
2. 실제 호출부 `awf_sim.py:3283`/`awf_sim.py:3290`가 **`side` 인자를 아예 전달하지 않는다** — 항상 기본값 `side=0`(legacy 3-key) 경로만 실행됨.
3. 더 상류에서 **4-key(side 포함) bucket edge 자체를 계산하는 함수가 없다** — `l2_meta.py`에서 `side_split`을 언급하는 곳은 docstring 한 줄뿐, `RegimeRoutingPlan`/edge 계산 경로 어디에도 side 분해 로직이 없다.

**결론: Phase B는 "배선 준비"만 됐을 뿐 실제로 배분 로직에 연결되지 않은 미완성 구현이다.** `/check` PASS(35%)는 통과했지만, 이는 순수함수 단위 테스트(side 인자를 직접 주입한 케이스)만 검증했을 뿐 awf_sim.py 호출부의 실제 배선(integration)을 검증하지 못했기 때문 — spec의 TDD 매트릭스에 "call-site 배선 검증" 시나리오가 누락된 것이 원인.

## 판정

1. **Phase 0 실측(bear-short/crash-long 엣지 실재)은 여전히 유효하다** — 이번 결함은 그 엣지를 "활용하는 코드"가 없다는 것이지, 엣지 자체가 없다는 뜻이 아니다.
2. **Phase B는 "구현 완료"가 아니라 "설계+헬퍼함수만 존재"** 상태로 재분류한다. 지금 상태로는 flag를 켜도 아무 효과가 없으므로 배포/운영 판단에 아무 영향이 없다(안전).
3. **재발 방지 교훈**: config flag를 추가할 때 "flag가 실제로 동작 분기를 바꾸는지"를 확인하는 A/B 회귀 테스트(예: flag on/off 시 실제 시뮬레이션 산출물이 달라지는지)를 `/check` 필수 시나리오에 반드시 포함해야 한다 — 순수함수 단위 테스트만으로는 배선 누락을 잡지 못한다.

## 다음 우선순위

1. **Phase B 배선 완성** (신규 후속 작업, 별도 spec 불필요 — 기존 spec 범위 내 마무리):
   - `l2_meta.py`에 4-key(side 포함) bucket edge 계산 함수 신설(Phase 0에서 실측한 `entry_regime_code × family × side` 셀 통계 재사용).
   - `awf_sim.py:3283`/`3290` 호출부에 `config.l2_regime_bucket_side_split_enabled`를 읽어 `side=sign(현재 sleeve 방향)`를 실제로 전달.
   - **필수**: flag on/off 시 시뮬레이션 산출물(`rets_hybrid` 등)이 달라지는지 확인하는 integration 회귀 테스트 추가.
2. 배선 완성 후 본 문서의 A/B 절차를 동일하게 재실행해 실제 효과(crisis_mdd/crisis_cagr blocker 감소 여부)를 재측정.

---

# L2/L3 게이트 강화 종합 실측 (Phase A 적용) — 2026-07-21

## 적용 사항

`docs/specs/alpha-funnel-regime-coverage.md` Phase A 구현(`/check` PASS, Cov 35%) — `l2_min_cagr=0.30` 절대 하드플로어를 **EW-baseline 상대 게이트**(`max(0, cagr_baseline_ew + 0.05)`)로 교체. 동시에 Phase B 탐색공간 확장(`l2_regime_bear_gross_cap` [0.35,0.85], `l2_regime_crisis_gross_cap` [0.25,0.85])도 함께 반영(단, 위 섹션에서 확인했듯 side-split 자체는 미배선이라 "노출 상한만 넓어진" 효과).

## 실측 결과 (`--phase l3 --trials 120 --seed 42`, 내부 42/43/44)

| Seed | joint_feasible | 최다 blocker 순위 | exit_code |
|---|---|---|---|
| 42 | 0/120 | crisis_mdd(110) > cagr(108) > crisis_cagr(107) > sharpe_uplift(60) | 1 |
| 43 | 0/120 | crisis_mdd(115) > crisis_cagr(111) > cagr(108) > sharpe_uplift(81) | 1 |
| 44 | 0/120 | crisis_mdd(115) > crisis_cagr(108) > cagr(96) | 1 |

`[MULTI-SEED] pass_count=0/3 required=2 admitted=False window_covered=False`, **exit_code=1**(L3 실패 은폐 없이 정상 반영 확인 — 과거 프로세스 무결성 결함의 fix가 실측으로도 검증됨).

## Before/After 비교 (직전 세션, Phase A 적용 전)

| 지표 | Before (recency gate만 적용) | After (Phase A 적용) | 해석 |
|---|---|---|---|
| seed=42 cagr blocker | 107/120 | 108/120 | 거의 불변 — 아래 참조 |
| seed=42 crisis_mdd blocker | 89/120 | **110/120** | **1위로 부상** |
| Best CAGR(seed=42, study 중 최고) | 미기록 | **31.21%** | 상대 게이트로 도달 가능 확인 |

## 분석

1. **CAGR 게이트는 더 이상 최상위 후보의 발목을 잡지 않는다.** study 진행 중 Best CAGR가 31.21%까지 도달 — 절대 30% 하드플로어 체제에서 흔했던 "CAGR 자체가 낮아 즉시 탈락"이 아니라, 이제는 **좋은 CAGR을 낸 trial이 crisis_mdd/crisis_cagr에서 걸러지는** 구조로 병목이 이동했다.
2. **cagr blocker 카운트가 거의 안 바뀐 이유**: 상대 게이트 자체는 완화됐지만, 동시에 노출 탐색범위가 넓어져 나쁜 조합에서 CAGR이 더 크게 마이너스로 튀는 trial 수도 늘어 상쇄됐다(Phase A·B 효과가 한 실행에 섞여 순수 분리 귀속은 어려움 — 캐비엇).
3. **crisis_mdd가 신규 1위 blocker로 부상한 것은 설계 실패가 아니라 안전장치의 정상 작동**이다. 노출 탐색범위를 넓히자 Optuna가 역경-regime에서 더 공격적인 노출을 시도하는 trial이 늘었고, 그중 다수가 **고정 불변으로 유지한** crisis MDD 21% / crisis CAGR -5% 예산을 실제로 초과해 정확히 차단됐다.
4. **아직 champion 없음(0/3 seed)** — 하지만 병목의 성격이 "알파가 없어서"에서 "실제 위험예산과 노출 사이의 트레이드오프"로 전환된 것은 유의미한 진전이다. Phase B(side-split)가 정상 배선되면 반대 방향 신호의 상쇄 없이 순수한 엣지만 태울 수 있어, 동일 노출로도 crisis_mdd 여유가 생길 가능성이 있다 — 위 섹션의 배선 완성이 선행 조건.

---

# Phase 0: Alpha Funnel Regime×Side 실측 — 2026-07-21

## 배경

기존 "L1 알파 예측력 부재"라는 반복된 결론에 대해, "L0/L1을 통과했다는 것은 유의미한 신호가 있었다는 뜻 아닌가"라는 재검토 요청에 따라 계측을 선행했다. `docs/specs/alpha-funnel-regime-coverage.md` 구현(`/check` PASS) — 기존 고아 함수 `signal_selection.py::compute_family_regime_edge_diagnostics(split_side=True)`(과거 spec 산출물, 호출부 0개였음)를 `log_family_regime_funnel_diagnostics()`로 배선, `pipeline.py` outer fold 루프에 `L2_FUNNEL_ATTR=1` env-gated 호출 추가.

## 실측 (`L2_FUNNEL_ATTR=1 --phase l1 --seed 42`, cold L1, 1,801 계측 라인)

### 판정 1 — "crisis에서 발화 자체가 없다" 가설 반증
L1 이벤트 regime 분포(371k 이벤트 합산): bull_q 27.6% / bull_v 10.1% / bear_q 7.1% / bear_v 4.4% / transition 35.2% / **crash 15.7%**. crash·bear에서도 이벤트는 충분히 발생 — kill site는 이벤트 생성이 아니라 L1 pooled admission → L2 sizing 단계.

### 판정 2 — "crash에서 트렌드 숏" 가설 반증, 두 개의 더 강한 엣지 발견

**(A) bear_q/bear_v에서 거의 전 family의 SHORT가 강한 양수 엣지** (이벤트가중 mean bps / 셀 LCB 중앙값, 비용 기준선 ≈7.5bps):

| 셀 | mean_bps | lcb_med | n_events |
|---|---|---|---|
| bear_v `dual_momentum` short | +635 | +254 | 484 |
| bear_q `trend_donchian` short | +261 | +214 | 213 |
| bear_q `mtf_fusion` short | +212 | +215 | 321 |
| bear_q `dual_momentum` short | +190 | +172 | 967 |
| bear_q `btc_regime_pullback` short | +175 | +180 | 1,978 |
| bear_v `btc_regime_pullback` short | +141 | +120 | 2,552 |

**(B) crash에서는 숏이 아니라 LONG-리버설이 엣지** (Page-CUSUM crash 마킹이 급락 후행이라 crash-바≈바닥권):

| 셀 | mean_bps | lcb_med | n_events |
|---|---|---|---|
| crash `btc_regime_pullback` long | +330 | +36 | 5,942 |
| crash `mtf_fusion` long | +300 | +1.3 | 2,951 |
| crash `taker_imbalance_momentum` long | +130 | +78 | 1,858 |
| (대조) crash `trend_donchian` short | **-149** | -259 | 591 |

### 판정 3 — 종합
**알파는 실재하며 regime×side 조건부다.** 병목은 L1 pooled admission과, side 차원이 없는 L2 버킷 라우팅이 이 조건부 구조를 평균으로 뭉개는 것 — 2026-06-25 Stage A 실측(조건부 버킷 causal corr +0.14~+0.33, 8/8 양수)과 정확히 합치.

## 결론

기존 오진 정정: "crash에서 트렌드 신호가 whitelist로 소거된다"는 가설은 `regime_signal_gating_enabled=False`(기본값, 미override) 확인으로 반증 — 실제 kill site는 L1/L2 admission·라우팅 로직이었다. → Phase A/B(위 섹션)로 이어짐.

---

# L2 Recency-Generalization 게이트 적용 결과 — 2026-07-21

`docs/specs/l2-recency-generalization-gate.md` 구현(`/check` PASS, Cov 57%) — L2→L3 반복 붕괴 패턴에 대한 구조적 방어 2건 추가.

## 무엇을 고쳤나

1. **Recency Holdout 하드게이트**: 기존 `recent_fold` 게이트는 objective 계산에 이미 포함된 fold를 재검증하는 순환 구조였다. objective에 전혀 안 들어가는 study 구간 맨 끝 30일만 떼어 CAGR 기준(-5%) 미달 시 탈락시키는 14번째 Optuna 제약 추가.
2. **"위기장 미검증" 경고 투명화**: `NO-CRISIS-WINDOW` print-only 경고를 `window_bottleneck_covered` 정식 필드로 승격, `[MULTI-SEED]` 로그에 상시 노출.
3. **실측 중 발견한 크래시 버그**: `zip() argument 2 is longer than argument 1` — 제약 이름 목록(13개)과 확장된 제약값 목록(14개) 불일치. 즉시 수정.

## 실측 (`--phase l3 --seed 42`, 내부 42/43/44)

| Seed | joint_feasible | recency_holdout 차단 수 | 최종 판정 |
|---|---|---|---|
| 42 | 0/120 | 53 | ❌ no_feasible_trials |
| 43 | 0/120 | 80 | ❌ no_feasible_trials |
| 44 | 0/120 | 49 | ❌ no_feasible_trials |

새 게이트는 실제로 작동하지만(seed당 49~80개 trial 차단), 최종 결론은 불변 — 당시엔 `cagr`/`crisis_cagr`/`crisis_mdd`가 이미 더 많은 trial을 차단 중이었다. `window_covered=False`가 로그에 상시 노출되는 것이 핵심 개선.

---

# L2/L3 Multi-Seed 강건성 합의 게이트 적용 결과 — 2026-07-21

`docs/specs/l2-l3-multi-seed-robustness-consensus.md` 구현(`/check` PASS, Cov 32%) — L2 champion 승격을 단일 seed에서 **K=3 독립 seed 과반수(2/3) 합의**로 전환, 미달 시 hard block(exit_code=1).

## 실측

```
[MULTI-SEED] seed=42 L2 study blocked: no_feasible_trials
[MULTI-SEED] seed=43 L2 study blocked: no_feasible_trials
[MULTI-SEED] seed=44 L2 study blocked: no_feasible_trials
[MULTI-SEED] pass_count=0/3 required=2 admitted=False
```

소요시간 `real 7m52.7s`(단일-seed 대비 ~2.6배, 의도된 증가). 3개 seed 전부 `no_feasible_trials` — 게이트가 설계대로 정확히 작동(실패가 아니라 성공): 이 코드/설정으로는 L2 탐색 프로세스 자체가 강건한 champion을 찾지 못하는 상태를 정확히 차단.

⚠️ **하지 말아야 할 것**: seed offset을 바꿔가며 "과반수 통과 조합 찾기"는 금지 — p-hacking. `admitted=False` 반복 시 다음 조치는 게이트/탐색공간 설계 변경이어야 한다.

---

# Crisis Replay 매칭 버그 수정 + L1 Multi-TF Registry Merge — 2026-07-21 (배경)

- `docs/specs/crisis-replay-strategy-match-fix.md`: `_build_rule_based_stress_batch()`의 `panel.variant` substring 매칭을 `panel.family:variant` 정확 일치로 교체. crisis reliability 계산 정확도 회복(`stress_tested_pass` 복구), 단 이 수정으로 과거 seed=42의 "L3 DEPLOY-READY(+5.6%)"는 재현 안 됨 — substring 버그 상태에서 채택된 champion의 우연한 산물이었을 가능성 확인.
- `docs/specs/l1-deployment-registry-multi-tf-merge.md`: `_aggregate_per_tf_l1`이 대표 TF 1개의 deployment_registry만 반영하고 나머지 TF의 qualified 신호를 폐기하던 버그 수정 — 전 TF 병합으로 교체. ETHUSDT 신호 활성화 등 side-benefit 확인.
- 두 수정 모두 각자 목적에서는 검증되었으나, L3 forward holdout 일반화 문제(non-stationarity)는 미해결 상태로 이번 세션(Phase 0~B)까지 이어짐.

---

# 프로세스 무결성 결함 및 근본원인 최초 진단 — 2026-07-21 (역사적 기록)

최초 L3 종단 실행에서 발견된 구조적 문제 2가지, 이후 세션에서 모두 해결됨:

1. **종료 코드가 L3 실패를 은폐**: `active_pipeline.py`가 `l2_final.gate_passed`만 검사하고 L3 결과를 검사하지 않아 L3 BLOCKED에도 `exit_code=0` 반환 — Multi-Seed 합의 게이트 도입(위 섹션)으로 해결, 현재 `exit_code=1` 정상 반영 실측 확인됨.
2. **최초 근본원인 진단**: BTC 단독 캐리(`avg_mult=0.895`) vs ETH/BNB 완전 비활성(`avg_mult=0.000`), Exposure 0.2x가 이론적 캡(0.55~0.75x)보다 낮아 "캡이 아니라 신호 부재"로 진단 — 이후 Phase 0 실측(위 섹션)으로 **"신호 자체는 있으나 regime×side 조건부이며 pooled admission이 이를 희석"**으로 정교화됨.

---

# 현재 상태 요약 (2026-07-21 세션 종료 시점, 최신)

- **알파 존재 여부**: 반증됨(부재 아님) — bear-short, crash-long 셀에서 실측 LCB 양수 엣지 확인(비용 대비 10배 이상).
- **정상장 CAGR 게이트**: Phase A로 상대 기준 전환, 병목 아님(Best CAGR 31%+ 도달 확인).
- **Crisis 리스크 예산**: 고정 불변 유지, 현재 최다 blocker(exit_code=1 정상 반환) — 안전장치 정상 작동.
- **Phase B(side-split 버킷) + RC-3 regime-scoped override**: **이중 배선버그(precompute 항상 False 고정 + 4-key/3-key shape mismatch로 정책 전체 무력화) 완전 수정, 부작용(recent_fold 악화)의 근본원인(RC-3 폴드 블랑켓 판정)도 특정·수정 완료**. 실측으로 recent_fold 위반이 baseline 수준으로 복귀 확인. 두 flag 모두 기본값 `False` 유지(운영 영향 없음).
- **아직 배포 가능한 champion 없음** — 병목이 "알파 부재" → "미배선" → "표본 분할 통계 불안정" → 현재는 **"crisis_mdd/crisis_cagr 위험예산과 노출의 근본적 트레이드오프"**로 계속 정교화되고 있음(이번 세션에서 의도적으로 불변 유지한 유일한 축). 다음 단계는 이 트레이드오프 자체를 다룰지, L1 알파 확충으로 갈지 결정 필요.
